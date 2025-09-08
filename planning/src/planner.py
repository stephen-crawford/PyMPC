import time

from planning.src.data_prep import propagate_obstacles
from planning.src.types import *
from utils.const import OBJECTIVE
from utils.utils import CONFIG, ExperimentManager


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

    propagate_obstacles(data, self.solver.timestep, self.solver.horizon)

    for module in self.solver.module_manager.get_modules():
      module.update(self.state, data)

    for k in range(self.solver.horizon):
      for module in self.solver.module_manager.get_modules():
        module.set_parameters(self.solver.parameter_manager, data, k)

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

  def on_data_received(self, data):
    self.solver.on_data_received(data)

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


class PlannerOutput:
  def __init__(self):
    self.success = False
    self.control_history = [] # The control history is a list of the actual control inputs used during the execution
    self.trajectory_history = [] # This is a list of horizon length trajectories which is added to each time the solver finds a solution
    self.realized_trajectory = Trajectory() # this is the trajectory executed by the physical robot when integrating based on the control history

  def process_solver_result(self, param, param1):
    pass
