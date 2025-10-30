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

  
  def solve(self):
    solver_iterations = 0

    while solver_iterations < self.config["solver_iterations"]:
      mpc_output = self.solve_mpc()
      self.output.control_history.append(mpc_output.control)
      self.state = self.problem.get_state().propagate(mpc_output.control, self.solver.timestep)
      self.data.update(self.state)
      self.output.realized_trajectory.add_state(self.state)
      solver_iterations += 1
      if self.state.is_objective_reached(self.data):
        break
    return self.output
  
  def solve_mpc(self):

    is_data_ready = all(module.is_data_ready(self.data) for module in self.solver.get_module_manager().get_modules())

    if not is_data_ready:
      return self.output

    self.solver.initialize_rollout(self.state, self.data)

    propagate_obstacles(self.state, self.data, self.solver.timestep, self.solver.horizon)

    for module in self.solver.module_manager.get_modules():
      module.update(self.state, self.data)

    # Set data for all stages including final stage (horizon + 1 stages: 0 to horizon)
    for k in range(self.solver.horizon + 1):
      for module in self.solver.module_manager.get_modules():
        self.parameter_manager.set_parameters(module, self.data, k)
    for k in range(self.solver.horizon + 1):
      self.data.set_parameters(self.parameter_manager.get_all(k), k)
      self.data.set_constraints(self.module_manager.get_constraints(self.state, self.data, k), k)
      self.data.set_objectives(self.module_manager.get_objectives(self.state, self.data, k), k)
      self.data.set_lower_bounds(self.module_manager.get_lower_bounds(self.state, self.data, k), k)
      self.data.set_upper_bounds(self.module_manager.get_upper_bounds(self.state, self.data, k), k)


    exit_flag = self.solver.solve(self.state, self.data)

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
