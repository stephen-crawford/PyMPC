import numpy as np
from utils.const import DETERMINISTIC
from utils.utils import LOG_DEBUG, read_config_file
from datetime import datetime
from planner_modules.base_constraint import BaseConstraint


class ScenarioConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)
		self.name = "scenario_constraints"  # Override name from BaseConstraint

		LOG_DEBUG("Initializing Scenario Constraints")

		self._planning_time = 1.0 / self.get_config_value("control_frequency")
		self._scenariosolvers = []
		self._best_solver = None

		# Initialize scenario config
		self._SCENARIO_CONFIG = ScenarioConfig()
		self._SCENARIO_CONFIG.Init()

		# Create parallel solvers
		parallel_solvers = self.get_config_value("scenario_constraints.parallelsolvers")
		for i in range(parallel_solvers):
			self._scenariosolvers.append(ScenarioSolver(i))

		LOG_DEBUG("Scenario Constraints successfully initialized")

	def update(self, state, data, module_data):
		LOG_DEBUG("ScenarioConstraints.update")
		for solver in self._scenariosolvers:
			# Copy the main solver, including its initial guess
			solver.solver = self.solver
			solver.scenario_module.update(data, module_data)

	def set_parameters(self, data, module_data, k):
		# Not implemented in original code
		return

	def optimize(self, state, data, module_data):
		# Set OpenMP parameters for parallelization
		self.set_openmp_params(nested=1, max_active_levels=2, dynamic=0)

		# Initialize best solver search
		lowest_cost = 1e9
		self._best_solver = None
		for solver in self._scenariosolvers:

			print(f"Checking solver: {solver}")
			print(f"exit_code: {solver.exit_code}")
			print(f"pobj: {solver.solver._info.pobj}")

			if solver.exit_code == 1 and solver.solver._info.pobj < lowest_cost:
				lowest_cost = solver.solver._info.pobj
				self._best_solver = solver

			# Set the planning timeout
			used_time = (datetime.now() - data.planning_start_time).total_seconds()
			solver.solver.params.solver_timeout = self._planning_time - used_time - 0.008

			# Create a copy of the solver instead of just assigning the reference
			# solver.solver = self.solver  # This line is causing the issue

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

		for solver in self._scenariosolvers:
			if solver.exit_code == 1 and solver.solver._info.pobj < lowest_cost:
				lowest_cost = solver.solver._info.pobj
				self._best_solver = solver

		if self._best_solver is None:  # No feasible solution
			return self._scenariosolvers[0].exit_code

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
				for solver in self._scenariosolvers:
					solver.scenario_module.get_sampler().integrate_and_translate_to_mean_and_variance(
						data.dynamic_obstacles, solver.dt
					)

	def is_data_ready(self, data, missing_data):
		required_fields = ["dynamic_obstacles"]
		missing_fields = self.check_data_availability(data, required_fields)

		if not self.report_missing_data(missing_fields, missing_data):
			return False

		max_obstacles = self.get_config_value("max_obstacles")
		if data.dynamic_obstacles.size() != max_obstacles:
			missing_data += "Obstacles "
			return False

		for i in range(data.dynamic_obstacles.size()):
			if data.dynamic_obstacles[i].prediction.empty():
				missing_data += "Obstacle Prediction "
				return False

			if data.dynamic_obstacles[i].prediction.type == DETERMINISTIC:
				missing_data += "Uncertain Predictions (scenario-based control cannot use deterministic predictions) "
				return False

		if not self._scenariosolvers[0].scenario_module.is_data_ready(data, missing_data):
			return False

		return True

	def visualize(self, data, module_data):
		super().visualize(data, module_data)

		visualize_all = self.get_config_value("scenario_constraints.visualize_all", False)

		LOG_DEBUG("ScenarioConstraints.visualize")

		if visualize_all:
			for solver in self._scenariosolvers:
				solver.scenario_module.visualize(data)
		elif self._best_solver is not None:
			self._best_solver.scenario_module.visualize(data)

		# Visualize optimized trajectories
		for solver in self._scenariosolvers:
			if solver.exit_code == 1:
				trajectory = []
				for k in range(solver.N):
					trajectory.append([
						solver.solver.get_output(k, "x"),
						solver.solver.get_output(k, "y")
					])

				self.visualize_trajectory(
					trajectory,
					"optimized_trajectories",
					scale=0.2,
					color_int=solver.solver_id
				)

	def reset(self):
		super().reset()
		# Reset constraint-specific state
		self._best_solver = None

		# Reset all scenario solvers
		for solver in self._scenariosolvers:
			solver.exit_code = 0

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
		config = read_config_file()
		self.enable_safe_horizon_ = config.get("scenario_constraints", {}).get("enable_safe_horizon", False)

	def Init(self):
		# Initialize scenario config parameters
		pass


class ScenarioSolver:
	def __init__(self, solver_id):
		config = read_config_file()
		self.solver = None
		self.scenario_module = ScenarioModule()
		self.exit_code = 0
		self.N = config["N"]
		self.dt = config["dt"]
		self.solver_id = solver_id

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