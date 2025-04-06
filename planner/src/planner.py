import time
import logging
from utils.utils import CONFIG, Timer, Benchmarker
from planner.src.types import *

logger = logging.getLogger(__name__)

class Planner:
  def __init__(self, solver, modules):
    self._solver = solver  # Now accepts any solver instance
    self._solver.reset()
    self._modules = modules
    self._init_modules()
    self._startup_timer = Timer(1.0)
    self._was_reset = False
    self._experiment_util = None
    self.output = None
    self.warmstart = None
    self.benchmarkers = []

  def _init_modules(self):
    for module in self._modules:
      module.initialize(self._solver)

  def solve_mpc(self, state, data):
    logger.info("planner.solve_mpc")
    was_feasible = self.output.success if self.output else False
    self.output = PlannerOutput(self._solver.dt, self._solver.N)
    module_data = {}

    is_data_ready = all(module.is_data_ready(data) for module in self._modules)
    if not is_data_ready:
      if self._startup_timer.has_finished():
        logger.warning("Data is not ready")
      self.output.success = False
      return self.output

    if self._was_reset:
      self._experiment_util.set_start_experiment()
      self._was_reset = False

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
      self._solver.initialize_warmstart(state, shift_forward)
    else:
      self._solver.initialize_with_braking(state)

    self._solver.set_xinit(state)

    for module in self._modules:
      module.update(state, data, module_data)

    for k in range(self._solver.N):
      for module in self._modules:
        module.set_parameters(data, module_data, k)

    self.warmstart = Trajectory()
    for k in range(self._solver.N):
      self.warmstart.add(self._solver.get_ego_prediction(k, "x"),
                         self._solver.get_ego_prediction(k, "y"))
    self._solver.load_warmstart()

    used_time = time.time() - data.planning_start_time
    self._solver.params.solver_timeout = 1.0 / CONFIG["control_frequency"] - used_time - 0.006

    exit_flag = -1
    for module in self._modules:
      exit_flag = module.optimize(state, data, module_data)
      if exit_flag != -1:
        break

    if exit_flag == -1:
      exit_flag = self._solver.solve()

    planning_benchmarker.stop()

    if exit_flag != 1:
      self.output.success = False
      logger.warning(f"MPC failed: {self._solver.explain_exit_flag(exit_flag)}")
      return self.output

    self.output.success = True
    for k in range(1, self._solver.N):
      self.output.trajectory.add(self._solver.get_output(k, "x"),
                                 self._solver.get_output(k, "y"))

    if self.output.success and CONFIG["debug_limits"]:
      self._solver.print_if_bound_limited()

    logger.info("Planner::solveMPC done")
    return self.output

  def get_solution(self, k, var_name):
    return self.solver.get_output(k, var_name)

  def get_data_saver(self):
    return self.experiment_util.get_data_saver()

  def on_data_received(self, data, data_name):
    for module in self.modules:
      module.on_data_received(data, data_name)

  def visualize(self, state, data):
    logger.info("Planner::visualize")
    for module in self.modules:
      module.visualize(data)

    # Visualization methods need to be implemented
    # visualize_trajectory(self.output.trajectory, "planned_trajectory", True, 0.2)
    # Additional visualization calls...
    logger.info("Planner::visualize Done")

  def save_data(self, state, data):
    if not self._is_data_ready:
      return

    data_saver = self.experiment_util.get_data_saver()
    planning_time = self.get_benchmarker("planning").get_last()
    data_saver.add_data("runtime_control_loop", planning_time)
    if planning_time > 1.0 / CONFIG["control_frequency"]:
      logger.warning(f"Planning took too long: {planning_time} ms")
    data_saver.add_data("runtime_optimization", self.get_benchmarker("optimization").get_last())

    data_saver.add_data("status", 2. if self.output.success else 3.)
    for module in self.modules:
      module.save_data(data_saver)
    self.experiment_util.update(self, state, self.solver, data)

  def reset(self, state, data, success):
    if CONFIG["recording"]["enable"]:
      self.experiment_util.on_task_complete(success)

    self.solver.reset()
    for module in self.modules:
      module.reset()

    state.reset()
    data.reset()
    self.was_reset = True
    self.startup_timer.start()

  def is_objective_reached(self, state, data):
    return all(module.is_objective_reached(self, state, data) for module in self.modules)

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