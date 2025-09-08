import copy
import time
from concurrent.futures import ThreadPoolExecutor

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planner_modules.src.constraints.scenario_utils.scenario_module import ScenarioSolver
from planning.src.types import PredictionType
from utils.utils import read_config_file, LOG_INFO, LOG_DEBUG

class ScenarioConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)
		self.constraints = []
		self.name = "scenario_constraints"  # Override name from BaseConstraint

		LOG_DEBUG("Initializing Scenario Constraints")

		# Load configuration
		self.config = read_config_file()
		self.planning_time = 1.0 / self.get_config_value("control_frequency")
		self.scenario_solvers = []
		self.best_solver = None

		# Initialize scenario config
		self.enable_safe_horizon = self.get_config_value("scenario_constraints.enable_safe_horizon")
		self.num_discs = self.get_config_value("num_discs")
		self.num_constraints = self.get_config_value("max_constraints") * self.num_discs
		self.use_slack = self.get_config_value("scenario_constraints.use_slack")

		# Create parallel solvers
		parallel_solvers = self.get_config_value("scenario_constraints.parallel_solvers")
		for i in range(parallel_solvers):
			self.scenario_solvers.append(ScenarioSolver(i, self.config))

		# Diagnostic information
		self.optimization_time = 0
		self.feasible_solutions = 0

		LOG_DEBUG("Scenario Constraints successfully initialized")

	def update(self, state, data):

		def worker(solver_wrapper):
			# Deep copy the main solver
			solver_wrapper.solver = copy.deepcopy(self.solver)

			# Update the scenario module
			solver_wrapper.scenario_module.update(data)

		# Run in parallel with 4 threads
		with ThreadPoolExecutor(max_workers=4) as executor:
			executor.map(worker, self.scenario_solvers)

	"""
	This is passthrough
	"""
	def set_parameters(self, parameters):
		LOG_INFO("Set parameters called for Scenario Constraints")
		return

	""" 
	Helper function for parallelized optimization of solvers 
	"""
	def run_optimize(self, scenario_solver, main_solver, planning_time, start_time, data):
		# Set timeout
		used_time = time.time() - start_time
		scenario_solver.solver_timeout = planning_time - used_time - 0.008

		# Copy solver state
		scenario_solver.solver = copy.deepcopy(main_solver)

		# Set scenario constraint parameters
		for step in range(main_solver.horizon):
			scenario_solver.scenario_module.set_parameters(data, step)

		exit_code = scenario_solver.scenario_module.optimize(data)
		return exit_code, scenario_solver.solver.info.pobj, scenario_solver

	def optimize(self, state, data):
		LOG_DEBUG("ScenarioConstraints.optimize")
		start_time = time.time()

		with ThreadPoolExecutor(max_workers=4) as executor:
			futures = [
				executor.submit(self.run_optimize, scenario_solver, self.solver, self.planning_time, start_time, data)
				for scenario_solver in self.scenario_solvers
			]

		results = [f.result() for f in futures]

		# Select best solver
		best_solver = None
		lowest_cost = float('inf')
		for exit_code, cost, solver_wrapper in results:
			solver_wrapper.exit_code = exit_code
			if exit_code == 1 and cost < lowest_cost:
				lowest_cost = cost
				best_solver = solver_wrapper

		if best_solver is None:
			return self.scenario_solvers[0].exit_code

		self.solver.output = best_solver.solver.output
		self.solver.info = best_solver.solver.info
		self.solver.params = best_solver.solver.params
		return best_solver.exit_code

	def on_data_received(self, data):
		if data.has("dynamic_obstacles") and data.dynamic_obstacles is not None:
			# Check if uncertainty was provided
			for obs in data.dynamic_obstacles:
				if obs.prediction.type == PredictionType.DETERMINISTIC:
					LOG_DEBUG("WARNING: Using deterministic prediction with Scenario Constraints")
					LOG_DEBUG("Set `process_noise` to a non-zero value to add uncertainty.")
					return

			# Process the new obstacle data
			if self.enable_safe_horizon:
				def worker(solver_wrapper):
					solver_wrapper.scenario_module.get_sampler().integrate_and_translate_to_mean_and_variance(
						data.dynamic_obstacles, self.solver.timestep
					)

				# Parallelize with 4 workers (threads)
				with ThreadPoolExecutor(max_workers=4) as executor:
					executor.map(worker, self.scenario_solvers)

	def is_data_ready(self, data):
		missing_data = ""

		if not data.has("dynamic_obstacles") or data.dynamic_obstacles is None:
			missing_data += "Dynamic Obstacles "
			LOG_DEBUG("Missing dynamic_obstacles: {}".format(missing_data))
		else:
			for i in range(len(data.dynamic_obstacles)):
				LOG_DEBUG("Obstacle prediction type is {}".format(data.dynamic_obstacles[i].prediction.type))
				if (not hasattr(data.dynamic_obstacles[i], 'prediction') or
						data.dynamic_obstacles[i].prediction is None):
					missing_data += "Obstacle Prediction "


				if (hasattr(data.dynamic_obstacles[i], 'prediction') and
						data.dynamic_obstacles[i].prediction is not None and
						hasattr(data.dynamic_obstacles[i].prediction, 'type') and
						not data.dynamic_obstacles[i].prediction.type is PredictionType.GAUSSIAN):
					missing_data += f"Obstacle Prediction (type must be gaussian) incorrect for dynamic_obstacles {i}"

				if not self.scenario_solvers[0].scenario_module.is_data_ready(data):
					missing_data += "Missing data required for Scenario Solvers"

		LOG_DEBUG("Missing data in Scenario Constraints: {}".format(missing_data))
		return len(missing_data) < 1

	def reset(self):
		super().reset()
		# Reset constraint-specific state
		self.best_solver = None
		self.optimization_time = 0
		self.feasible_solutions = 0

		# Reset all scenario solvers
		for solver in self.scenario_solvers:
			solver.exit_code = 0
			solver.scenario_module.reset() if hasattr(solver.scenario_module, 'reset') else None