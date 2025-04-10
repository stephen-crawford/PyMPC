import numpy as np
from utils.const import CONSTRAINT, DETERMINISTIC
from utils.utils import read_config_file, LOG_DEBUG
# from utils.visualizer import *
import time
from datetime import datetime

CONFIG = read_config_file()


class ScenarioConstraints:
	def __init__(self, solver):
		self.solver = solver
		self.module_type = CONSTRAINT
		self.name = "scenario_constraints"

		LOG_DEBUG("Initializing Scenario Constraints")

		self._planning_time = 1.0 / CONFIG["control_frequency"]
		self._scenario_solvers = []
		self._best_solver = None

		# Initialize scenario config
		self._SCENARIO_CONFIG = ScenarioConfig()
		self._SCENARIO_CONFIG.Init()

		# Create parallel solvers
		for i in range(CONFIG["scenario_constraints"]["parallel_solvers"]):
			self._scenario_solvers.append(ScenarioSolver(i))

		LOG_DEBUG("Scenario Constraints successfully initialized")

	def update(self, state, data, module_data):
		for solver in self._scenario_solvers:
			# Copy the main solver, including its initial guess
			solver.solver = self.solver
			solver.scenario_module.update(data, module_data)

	def set_parameters(self, data, module_data, k):
		# Not implemented in original code
		return

	def optimize(self, state, data, module_data):
		# Set OpenMP parameters for parallelization
		self.set_openmp_params(nested=1, max_active_levels=2, dynamic=0)

		for solver in self._scenario_solvers:
			# Set the planning timeout
			used_time = (datetime.now() - data.planning_start_time).total_seconds()
			solver.solver.params.solver_timeout = self._planning_time - used_time - 0.008

			# Copy solver parameters and initial guess
			solver.solver = self.solver

			# Set the scenario constraint parameters for each solver
			for k in range(solver.N):
				solver.scenario_module.set_parameters(data, k)

			# Load the previous solution
			solver.solver.load_warmstart()

			# Run optimization (Safe Horizon MPC)
			solver.exit_code = solver.scenario_module.optimize(data)

		# Restore OpenMP dynamic scheduling
		self.set_openmp_params(dynamic=1)

		# Retrieve the lowest cost solution
		lowest_cost = 1e9
		self._best_solver = None

		for solver in self._scenario_solvers:
			if solver.exit_code == 1 and solver.solver._info.pobj < lowest_cost:
				lowest_cost = solver.solver._info.pobj
				self._best_solver = solver

		if self._best_solver is None:  # No feasible solution
			return self._scenario_solvers[0].exit_code

		# Load the solution into the main lmpcc solver
		self.solver.output = self._best_solver.solver.output
		self.solver._info = self._best_solver.solver._info
		self.solver.params = self._best_solver.solver.params

		return self._best_solver.exit_code

	def on_data_received(self, data, data_name):
		LOG_DEBUG("ScenarioConstraints.on_data_received()")

		if data_name == "dynamic obstacles":
			# Check if uncertainty was provided
			for obs in data.dynamic_obstacles:
				assert obs.prediction.type != DETERMINISTIC, (
					"When using Scenario Constraints, the predictions should have a non-zero "
					"uncertainty. If you are using pedestrian_simulator, set `process_noise` in "
					"config/configuration.yml to a non-zero value to add uncertainty."
				)

			if self._SCENARIO_CONFIG.enable_safe_horizon_:
				# Draw different samples for all solvers in parallel
				for solver in self._scenario_solvers:
					solver.scenario_module.get_sampler().integrate_and_translate_to_mean_and_variance(
						data.dynamic_obstacles, solver.dt
					)

	def is_data_ready(self, data, missing_data):
		if data.dynamic_obstacles.size() != CONFIG["max_obstacles"]:
			missing_data += "Obstacles "
			return False

		for i in range(data.dynamic_obstacles.size()):
			if data.dynamic_obstacles[i].prediction.empty():
				missing_data += "Obstacle Prediction "
				return False

			if data.dynamic_obstacles[i].prediction.type == DETERMINISTIC:
				missing_data += "Uncertain Predictions (scenario-based control cannot use deterministic predictions) "
				return False

		if not self._scenario_solvers[0].scenario_module.is_data_ready(data, missing_data):
			return False

		return True

	def visualize(self, data, module_data):
		visualize_all = False

		LOG_DEBUG("ScenarioConstraints.visualize")

		if visualize_all:
			for solver in self._scenario_solvers:
				solver.scenario_module.visualize(data)
		elif self._best_solver is not None:
			self._best_solver.scenario_module.visualize(data)

		# Visualize optimized trajectories
		for solver in self._scenario_solvers:
			if solver.exit_code == 1:
				trajectory = []
				for k in range(solver.N):
					trajectory.append([
						solver.solver.get_output(k, "x"),
						solver.solver.get_output(k, "y")
					])

				visualize_trajectory(
					trajectory,
					f"{self.name}/optimized_trajectories",
					False,
					0.2,
					solver.solver._solver_id,
					2 * len(self._scenario_solvers)
				)

		# Publish visualization
		from utils.visualizer import VISUALS
		VISUALS.missing_data(f"{self.name}/optimized_trajectories").publish()

	def set_openmp_params(self, nested=None, max_active_levels=None, dynamic=None):
		"""Helper function to set OpenMP parameters (mock implementation)"""
		# In a real implementation, this would use the appropriate OpenMP bindings
		if nested is not None:
			pass  # omp_set_nested(nested)
		if max_active_levels is not None:
			pass  # omp_set_max_active_levels(max_active_levels)
		if dynamic is not None:
			pass  # omp_set_dynamic(dynamic)


class ScenarioConfig:
	def __init__(self):
		self.enable_safe_horizon_ = CONFIG.get("scenario_constraints", {}).get("enable_safe_horizon", False)

	def Init(self):
		# Initialize scenario config parameters
		pass


class ScenarioSolver:
	def __init__(self, solver_id):
		self.solver = None
		self.scenario_module = ScenarioModule()
		self.exit_code = 0
		self.N = CONFIG["N"]
		self.dt = CONFIG["dt"]
		self._solver_id = solver_id

	def get(self):
		return self


class ScenarioModule:
	def __init__(self):
		pass

	def update(self, data, module_data):
		pass

	def set_parameters(self, data, k):
		pass

	def optimize(self, data):
		# Returns an exit code (1 for success)
		return 1

	def get_sampler(self):
		return ScenarioSampler()

	def is_data_ready(self, data, missing_data):
		return True

	def visualize(self, data):
		pass


class ScenarioSampler:
	def __init__(self):
		pass

	def integrate_and_translate_to_mean_and_variance(self, dynamic_obstacles, dt):
		pass