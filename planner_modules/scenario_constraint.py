from logging import Logger
from unittest.result import failfast

from planner.data_prep import logger
from utils.const import CONSTRAINT, DETERMINISTIC
from utils.utils import read_config_file

CONFIG = read_config_file()

class ScenarioConstraints:

  def __init__(self, solver):
    self.module_type = CONSTRAINT
    self.solver = solver
    self.name = "scenario_constraints"
    self.controller = (self.module_type, solver, self.name)
    Logger.log(10, "Initializing Scenario Constraints")

    self._planning_time = 1. / CONFIG["control_frequency"]

    _SCENARIO_CONFIG.Init()
    for i in range(CONFIG["scenario_constraints"]["parallel_solvers"]):

      _scenario_solvers.emplace_back(make_unique(ScenarioSolver(i))) # May need an integer input

    Logger.log(10, "Scenario Constraints successfully initialized")

  def update(self, state, data, module_data):

    for solver in _scenario_solvers:
      solver.solver = *_solver # Copy the main solver, including its initial guess
      solver.scenario_module.update(data, module_data)


  def set_parameters(self, data, module_data, k):
    return

  def optimize(self, state, data, module_data):
    omp_set_nested(1)
    omp_set_max_active_levels(2)
    omp_set_dynamic(0)

    for solver in _scenario_solvers:
      # Set the planning timeout
      used_time = system_clock.now() - data.planning_start_time
      solver.solver._params.solver_timeout = _planning_time - used_time.count() - 0.008

      # Copy solver parameters and initial guess
      solver.solver = *_solver # Copy the main solver

      # Set the scenario constraint parameters for each solver
      for k in range(_solver.N):

        solver.scenario_module.set_parameters(data, k)

      solver.solver.load_warmstart() # Load the previous solution

      solver.exit_code = solver.scenario_module.optimize(data) # Safe Horizon MPC
    omp_set_dynamic(1)

    # Retrieve the lowest cost solution
    lowest_cost = 1e9
    _best_solver = None
    for solver in _scenario_solvers:

      if (solver.exit_code == 1 and solver.solver._info.pobj < lowest_cost):

        lowest_cost = solver.solver._info.pobj
        _best_solver = solver.get()

    if (_best_solver == None) # No feasible solution
      return _scenario_solvers.front().exit_code

    _solver._output = _best_solver.solver._output # Load the solution into the main lmpcc solver
    _solver._info = _best_solver.solver._info
    self._solver._params = _best_solver.solver._params

    return _best_solver.exit_code

  def on_data_received(self, data, data_name):
    logger.log(10, "ScenarioConstraints.on_data_received()")

    if data_name == "dynamic obstacles":

      # Check if uncertainty was provided
      for obs in data.dynamic_obstacles:

        ROSTOOLS_ASSERT(obs.prediction.type != DETERMINISTIC, "When using Scenario Constraints, the predictions should have a non-zero uncertainty. If you are using pedestrian_simulator, set `process_noise` in config/configuration.yaml to a non-zero value to add uncertainty.")

      if (_SCENARIO_CONFIG.enable_safe_horizon_):

#pragma omp parallel for num_threads(4)
        for solver in _scenario_solvers: # Draw different samples for all solvers

          solver.scenario_module.get_sampler().integrate_and_translate_to_mean_and_variance(data.dynamic_obstacles, _solver.dt)


  def is_data_ready(self, data, missing_data):

    if data.dynamic_obstacles.size() != CONFIG["max_obstacles"]:

      missing_data += "Obstacles "
      return False


    for i in range(data.dynamic_obstacles.size()):
      if (data.dynamic_obstacles[i].prediction.empty()):

        missing_data += "Obstacle Prediction "
        return False


      if (data.dynamic_obstacles[i].prediction.type == DETERMINISTIC):

        missing_data += "Uncertain Predictions (scenario-based control cannot use deterministic predictions) "
        return False



    if not _scenario_solvers.front().scenario_module.is_data_ready(data, missing_data)):
      return False

    return True


  def visualize(self, data, module_data):

    bool visualize_all = False

    logger.log(10, "ScenarioConstraints.visualize")

    if (visualize_all):
      for solver in _scenario_solvers:
        solver.scenario_module.visualize(data)
    elif (_best_solver != None):
      _best_solver.scenario_module.visualize(data)

    # Visualize optimized trajectories
    for solver in _scenario_solvers:
      if (solver.exit_code == 1)

        for k in range(_solver.N):
          trajectory.add(solver.solver.get_output(k, "x"), solver.solver.get_output(k, "y"))

        visualize_trajectory(trajectory, _name + "/optimized_trajectories", False, 0.2, solver.solver._solver_id, 2 * _scenario_solvers.size())

    VISUALS.missing_data(_name + "/optimized_trajectories").publish()
