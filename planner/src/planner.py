import time
from utils.utils import CONFIG, Benchmarker, ExperimentManager, LOG_WARN
from planner.src.types import *


class Planner:
  def __init__(self, solver):
    self.solver = solver
    self.benchmarkers = []
    self.experiment_manager = ExperimentManager()

    self.experiment_manager.set_timer(1)
    self.experiment_manager.start_timer()

    self.was_reset = False
    self.output = None
    self.warmstart = None

  def solve_mpc(self, state, data):
    was_feasible = self.output.success if self.output else False
    self.output = PlannerOutput(self.solver.timestep, self.solver.horizon)
    module_data = {}

    is_data_ready = all(module.is_data_ready(data) for module in self.solver.get_module_manager().get_modules())
    if not is_data_ready:
      if self.experiment_manager.timer.has_finished():
        LOG_WARN("Data is not ready")
      self.output.success = False
      return self.output

    if self.was_reset:
      self.experiment_manager.set_start_experiment()
      self.was_reset = False

    planning_benchmarker = Benchmarker("planning")
    self.benchmarkers.append(planning_benchmarker)
    if planning_benchmarker.is_running():
      planning_benchmarker.cancel()
    planning_benchmarker.start()

    optimization_benchmarker = Benchmarker("optimization")
    self.benchmarkers.append(optimization_benchmarker)
    if optimization_benchmarker.is_running():
      optimization_benchmarker.cancel()
    optimization_benchmarker.start()

    shift_forward = CONFIG["shift_previous_solution_forward"] and CONFIG["enable_output"]
    if was_feasible:
      self.solver.initialize_warmstart(state, shift_forward)
    else:
      self.solver.initialize_with_braking(state)

    self.solver.set_xinit(state)

    for module in self.solver.module_manager.get_modules():
      module.update(state, data, module_data)

    for k in range(self.solver.horizon):
      for module in self.solver.module_manager.get_modules():
        module.set_parameters(self.solver.params, data, module_data, k)

    self.warmstart = Trajectory()
    for k in range(self.solver.horizon):
      self.warmstart.add(self.solver.get_ego_prediction(k, "x"),
                         self.solver.get_ego_prediction(k, "y"))
    self.solver.load_warmstart()

    used_time = time.time() - data.planning_start_time
    self.solver.params.solver_timeout = 1.0 / CONFIG["control_frequency"] - used_time - 0.006

    exit_flag = -1
    for module in self.solver.module_manager.get_modules():
      exit_flag = module.optimize(state, data, module_data)
      if exit_flag != -1:
        print("Exit flag: {}".format(exit_flag))
        print("breaking")
        break

    if exit_flag == -1:
      exit_flag = self.solver.solve()

    planning_benchmarker.stop()

    if exit_flag != 1:
      self.output.success = False
      logger.warning(f"MPC failed: {self.solver.explain_exit_flag(exit_flag)}")
      return self.output

    self.output.success = True
    for k in range(1, self.solver.horizon):
      self.output.trajectory.add(self.solver.get_output(k, "x"),
                                 self.solver.get_output(k, "y"))

    if self.output.success and CONFIG["debug_limits"]:
      self.solver.print_if_bound_limited()

    logger.info("Planner::solveMPC done")
    return self.output

  def get_solution(self, mpc_step, var_name):
    return self.solver.get_output(mpc_step, var_name)

  def get_data_saver(self):
    return self.experiment_manager.get_data_saver()

  def on_data_received(self, data, data_name):
    for module in self.solver.module_manager.modules:
      module.on_data_received(data, data_name)

  def visualize(self, state, data):
    logger.info("Planner::visualize")
    for module in self.solver.module_manager.modules:
      module.visualize(data)

    # Visualization methods need to be implemented
    # visualize_trajectory(self.output.trajectory, "planned_trajectory", True, 0.2)
    # Additional visualization calls...
    logger.info("Planner::visualize Done")

  def save_data(self, state, data):
    if not self.solver.module_manager.is_data_ready(data):
      return

    data_saver = self.experiment_manager.get_data_saver()
    planning_time = self.get_benchmarker("planning").get_last()
    data_saver.add_data("runtime_control_loop", planning_time)
    if planning_time > 1.0 / CONFIG["control_frequency"]:
      logger.warning(f"Planning took too long: {planning_time} ms")
    data_saver.add_data("runtime_optimization", self.get_benchmarker("optimization").get_last())

    data_saver.add_data("status", 2. if self.output.success else 3.)
    for module in self.solver.module_manager.modules:
      module.save_data(data_saver)
    self.experiment_manager.update(state, self.solver, data)

  def reset(self, state, data, success):
    if CONFIG["recording"]["enable"]:
      self.experiment_manager.on_task_complete(success)

    self.solver.reset()
    for module in self.solver.module_manager.modules:
      module.reset()

    state.reset()
    data.reset()
    self.was_reset = True
    self.experiment_manager.start_timer()

  def is_objective_reached(self, state, data):
    return all(module.is_objective_reached(self, state, data) for module in self.solver.module_manager.modules)

  def get_benchmarker(self, name):
    for benchmarker in self.benchmarkers:
      if benchmarker.name == name:
        return benchmarker

class PlannerOutput:
  def __init__(self, dt: float = None, N: int = None):
    self.trajectory = None
    if dt is not None and N is not None:
      self.trajectory = Trajectory(dt, N)
    else:
      self.trajectory = None
    self.success = False