import time

from scipy.interpolate import interp1d

from planning.src.data_prep import get_constant_velocity_prediction
from utils.const import OBJECTIVE
from utils.utils import CONFIG, Benchmarker, ExperimentManager, LOG_WARN
from planning.src.types import *


class Planner:
  def __init__(self, solver, model_type, benchmarkers=None):
    self.solver = solver

    if benchmarkers is None:
      self.benchmarkers = []
    else:
      self.benchmarkers = benchmarkers

    # set defaults
    self.was_reset = False
    self.output = PlannerOutput()
    self.warmstart = None

    self.model_type = model_type
    self.state = State(self.model_type)

    self.experiment_manager = ExperimentManager()

    # start launch timer
    self.experiment_manager.set_timer(1)
    self.experiment_manager.start_timer()

  def initialize(self, data):
    self.solver.initialize(data)

  def solve_mpc(self, data: Data):

    is_data_ready = all(module.is_data_ready(data) for module in self.solver.get_module_manager().get_modules())

    if not is_data_ready:
      if self.experiment_manager.timer.has_finished():
        LOG_WARN("Data is not ready")
      self.output.process_solver_result(False, None)
      return

    if self.was_reset:
      self.experiment_manager.set_start_experiment()
      self.was_reset = False

    # planning_benchmarker = Benchmarker("planning")
    # self.benchmarkers.append(planning_benchmarker)
    # if planning_benchmarker.is_running():
    #   planning_benchmarker.cancel()
    # planning_benchmarker.start()
    #
    # optimization_benchmarker = Benchmarker("optimization")
    # self.benchmarkers.append(optimization_benchmarker)
    # if optimization_benchmarker.is_running():
    #   optimization_benchmarker.cancel()
    # optimization_benchmarker.start()

    self.solver.initialize_rollout(self.state)

    for module in self.solver.module_manager.get_modules():
      module.update(self.state, data)

    for k in range(self.solver.horizon):
      for module in self.solver.module_manager.get_modules():
        module.set_parameters(self.solver.parameter_manager, data, k)

    self.propagate_obstacles(data)

    used_time = time.time() - data.planning_start_time

    self.solver.parameter_manager.solver_timeout = 1.0 / CONFIG["control_frequency"] - used_time - 0.006

    for module in self.solver.module_manager.get_modules():
      if hasattr(module, "optimize"):
        exit_flag = module.optimize(self.state, data)
        if exit_flag != -1:
          LOG_WARN("Exit flag: {}".format(exit_flag))
          break

    exit_flag = self.solver.solve()

    if exit_flag != 1:
      self.output.success = False
      LOG_WARN(f"MPC failed: {self.solver.explain_exit_flag(exit_flag)}")
      return self.output

    self.output.success = True
    reference_trajectory = self.solver.get_reference_trajectory()
    LOG_WARN("Reference trajectory: {}".format(reference_trajectory))
    self.output.trajectory_history.append(reference_trajectory)

    if self.output.success and CONFIG["debug_limits"]:
      self.solver.print_if_bound_limited()

    LOG_DEBUG("Planner.solve_mpc done")
    return self.output

  def reset(self, state, data, success):
    if CONFIG["recording"]["enable"]:
      self.experiment_manager.on_task_complete(success)

    self.solver.reset()

    self.state.reset()
    data.reset()
    self.was_reset = True
    self.experiment_manager.start_timer()

  def is_objective_reached(self, data):
    return all(module.is_objective_reached(self.state, data) for module in self.solver.module_manager.modules if module.module_type == OBJECTIVE)

  def on_data_received(self, data, data_name):
    self.solver.on_data_received(data, data_name)

  def visualize(self, data):
    LOG_DEBUG("Planner::visualize")
    for module in self.solver.module_manager.modules:
      module.visualize(data)

    # Visualization methods need to be implemented
    # visualize_trajectory(self.output.trajectory, "planned_trajectory", True, 0.2)
    # Additional visualization calls...
    LOG_DEBUG("Planner::visualize Done")

  def get_data_saver(self):
    return self.experiment_manager.get_data_saver()

  def save_data(self, data):
    if not self.solver.module_manager.is_data_ready(data):
      return

    data_saver = self.experiment_manager.get_data_saver()
    planning_time = self.get_benchmarker("planning").get_last()
    data_saver.add_data("runtime_control_loop", planning_time)
    if planning_time > 1.0 / CONFIG["control_frequency"]:
      LOG_WARN(f"Planning took too long: {planning_time} ms")
    data_saver.add_data("runtime_optimization", self.get_benchmarker("optimization").get_last())

    data_saver.add_data("status", 2. if self.output.success else 3.)
    for module in self.solver.module_manager.modules:
      module.save_data(data_saver)
    self.experiment_manager.update(self.state, self.solver, data)

  def get_benchmarker(self, name):
    for benchmarker in self.benchmarkers:
      if benchmarker.name == name:
        return benchmarker

  def get_state(self):
    return self.state

  def set_state(self, state):
    self.state = state
    LOG_DEBUG("Planner::set_state")

  def propagate_obstacles(self, data, dt=0.1, horizon=10, speed=10, sigma_pos=0.2):
      if not data.dynamic_obstacles:
          return

      for obstacle in data.dynamic_obstacles:
          pred = obstacle.prediction
          path = getattr(pred, "path", None)

          # Fallback: constant velocity if no path
          if path is None:
              velocity = np.array([np.cos(obstacle.angle), np.sin(obstacle.angle)]) * speed
              obstacle.prediction = get_constant_velocity_prediction(obstacle.position, velocity, dt, horizon)
              continue

          total_length = path.s[-1]

          # Initialize progress in arc length
          if not hasattr(obstacle, "s"):
              obstacle.s = 0.0
          obstacle.s += speed * dt

          # If reached end of current path → generate new path
          if obstacle.s >= total_length:
              start = [path.x[-1], path.y[-1], path.z[-1] if hasattr(path, "z") else 0.0]

              # Choose new goal
              if hasattr(obstacle, "road_to_follow"):
                  road = obstacle.road_to_follow
                  s_offset = np.random.uniform(0, road.length)
                  x_center = road.x_spline(s_offset)
                  y_center = road.y_spline(s_offset)

                  # Tangent & lateral offset to simulate lane variation
                  dx = road.x_spline(min(s_offset + 0.1, road.length)) - x_center
                  dy = road.y_spline(min(s_offset + 0.1, road.length)) - y_center
                  tangent = np.array([dx, dy]) / np.linalg.norm([dx, dy])
                  normal = np.array([-tangent[1], tangent[0]])
                  lateral_offset = np.random.uniform(-3.0, 3.0)
                  goal = [x_center + lateral_offset * normal[0], y_center + lateral_offset * normal[1], 0.0]
              else:
                  goal = [
                      np.random.uniform(start[0] - 20, start[0] + 20),
                      np.random.uniform(start[1] - 20, start[1] + 20),
                      0.0
                  ]

              # Generate new reference path
              new_path = generate_reference_path(
                  start, goal, path_type=np.random.choice(["straight", "curved", "s-turn", "circle"]),
                  num_points=10
              )

              obstacle.prediction.path = new_path
              obstacle.s = 0.0
              path = new_path
              total_length = path.s[-1]

          # ✅ Compute position using arc-length splines
          s_now = min(obstacle.s, total_length)
          x = path.x_spline(s_now)
          y = path.y_spline(s_now)
          z = path.z_spline(s_now) if path.z_spline else 0.0
          obstacle.position = np.array([x, y, z])

          # ✅ Compute heading
          ds = 0.1
          s_next = min(s_now + ds, total_length)
          dx = path.x_spline(s_next) - x
          dy = path.y_spline(s_next) - y
          obstacle.angle = np.arctan2(dy, dx)

          # ✅ Build prediction horizon
          pred_steps = []
          s_future = s_now
          for _ in range(horizon):
              s_future = min(s_future + speed * dt, total_length)
              px = path.x_spline(s_future)
              py = path.y_spline(s_future)
              pz = path.z_spline(s_future) if path.z_spline else 0.0

              s_next = min(s_future + ds, total_length)
              dx_f = path.x_spline(s_next) - px
              dy_f = path.y_spline(s_next) - py
              angle = np.arctan2(dy_f, dx_f)

              pos = np.array([px, py, pz])

              # Add noise & uncertainty
              if pred.type == PredictionType.GAUSSIAN:
                  pos += np.random.normal(0, sigma_pos, size=pos.shape)
                  major_r, minor_r = sigma_pos * 2, sigma_pos
              elif pred.type == PredictionType.NONGAUSSIAN:
                  pos += np.random.standard_t(df=3, size=pos.shape) * sigma_pos
                  major_r, minor_r = sigma_pos * 3, sigma_pos * 1.5
              else:
                  major_r, minor_r = 0.1, 0.1

              pred_steps.append(PredictionStep(pos, angle, major_r, minor_r))

          pred.modes = [pred_steps]
          pred.probabilities = [1.0]


class PlannerOutput:
  def __init__(self):
    self.success = False
    self.control_history = [] # The control history is a list of the actual control inputs used during the execution
    self.trajectory_history = [] # This is a list of horizon length trajectories which is added to each time the solver finds a solution
    self.realized_trajectory = Trajectory() # this is the trajectory executed by the physical robot when integrating based on the control history

  def process_solver_result(self, param, param1):
    pass
