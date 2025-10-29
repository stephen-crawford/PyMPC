import time

from planning.src.data_prep import propagate_obstacles
from planning.src.types import *
from utils.const import OBJECTIVE
from utils.utils import CONFIG, ExperimentManager, LOG_INFO


class Planner:
  def __init__(self, problem, config=None):

    if config is None:
      self.config = CONFIG

    else:
      self.config = config

    self.problem = problem
    self.module_manager = ModuleManager()
    self.parameter_manager = ParameterManager()
    self.output = PlannerOutput()
    self.problem_setup()


  def problem_setup(self):
    self.model_type = self.problem.get_model_type()
    self.modules = self.problem.get_modules()
    self.obstacles = self.problem.get_obstacles()
    self.data = self.problem.get_data()
    self.x0 = self.problem.get_x0()
    
    self.module_manager.add_modules(self.modules)
    if self.module_manager.get_objective_modules() is None:
      raise ValueError("No objective modules found")
    if self.module_manager.check_dependencies() is False:
      raise ValueError("Dependencies are not satisfied")
    
    if self.config["solver"] == "casadi":
      self.solver = CasADiSolver(self.config)
    else:
      raise ValueError("Solver not supported")
    self.solver.initialize(self.data)
  
  def solve_mpc(self):

    is_data_ready = all(module.is_data_ready(self.data) for module in self.solver.get_module_manager().get_modules())

    if not is_data_ready:
      return self.output

    self.solver.initialize_rollout(self.state)

    propagate_obstacles(data, self.solver.timestep, self.solver.horizon)

    for module in self.solver.module_manager.get_modules():
      module.update(self.state, data)

    LOG_DEBUG("Planner going to try to set parameters for all modules")

    # Set parameters for all stages including final stage (horizon + 1 stages: 0 to horizon)
    for k in range(self.solver.horizon + 1):
      for module in self.solver.module_manager.get_modules():
        module.set_parameters(self.solver.parameter_manager, data, k)

    used_time = time.time() - data.planning_start_time

    self.solver.parameter_manager.solver_timeout = 1.0 / CONFIG["control_frequency"] - used_time - 0.006
    exit_flag = -1  # Default to failure
    optimization_handled_by_module = False

    # Check if any module has its own custom optimize method
    for module in self.solver.module_manager.get_modules():
      if hasattr(module, "optimize"):
        LOG_INFO(f"Module '{module.get_name()}' is handling the optimization.")
        exit_flag = module.optimize(self.state, data)
        optimization_handled_by_module = True
        break  # Assume only one module will handle optimization

    # If no module handled optimization, run the standard solver
    if not optimization_handled_by_module:
      LOG_INFO("No module optimizer found. Running standard solver.")
      exit_flag = self.solver.solve()

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

  def reset(self):

    self.solver.reset()
    self.state.reset()
    self.data.reset()

  def is_objective_reached(self, data):
    return all(module.is_objective_reached(self.state, data) for module in self.solver.module_manager.modules if module.module_type == OBJECTIVE)

  def on_data_received(self, data):
    self.solver.on_data_received(data)

  def visualize(self):
    LOG_DEBUG("Planner::visualize")
    for module in self.solver.module_manager.modules:
      module.get_visualizer().visualize(self.state, self.data)

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
