import copy

from utils.const import DETERMINISTIC
from utils.utils import LOG_DEBUG, read_config_file
from datetime import datetime
from planner_modules.src.constraints.base_constraint import BaseConstraint


class ScenarioConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)
		self.name = "scenario_constraints"  # Override name from BaseConstraint

		LOG_DEBUG("Initializing Scenario Constraints")

		self.planning_time = 1.0 / self.get_config_value("control_frequency")
		self.scenario_solvers = []
		self.best_solver = None

		# Initialize scenario config
		self.enable_safe_horizon = self.get_config_value("scenario.enable_safe_horizon")
		self.num_discs = self.get_config_value("num_discs")
		self.num_constraints = self.get_config_value("max_constraints") * self.num_discs
		self.use_slack = self.get_config_value("scenario.use_slack")
		self.nh = self.num_constraints

		# Create parallel solvers
		parallel_solvers = self.get_config_value("scenario_constraints.parallel_solvers")
		for i in range(parallel_solvers):
			self.scenario_solvers.append(ScenarioSolver(i))

		LOG_DEBUG("Scenario Constraints successfully initialized")

	def update(self, state, data, module_data):
		LOG_DEBUG("ScenarioConstraints.update")
		for solver in self.scenario_solvers:
			# Copy the main solver, including its initial guess
			solver.solver = self.solver
			solver.scenario_module.update(data, module_data)

	def optimize(self, state, data, module_data):
		# Initialize best solver search
		lowest_cost = 1e9
		self.best_solver = None
		for solver in self.scenario_solvers:

			if solver.exit_code == 1 and solver.info.pobj < lowest_cost:
				lowest_cost = solver.info.pobj
				self.best_solver = solver

			# Set the planning timeout
			used_time = (datetime.now() - data.planning_start_time).total_seconds()
			solver.params.solver_timeout = self.planning_time - used_time - 0.008

			# Create a copy of the solver instead of just assigning the reference
			if hasattr(self.solver, 'copy'):
				# Use custom copy method if available
				solver.solver = self.solver.copy()
			else:
				# Otherwise use a deep copy
				solver.solver = copy.deepcopy(self.solver)

			# Set the scenario constraint parameters for each solver
			for k in range(solver.horizon):
				solver.scenario_module.set_parameters(data, k)

			# Load the previous solution
			solver.solver.load_warmstart()

			# Run optimization (Safe Horizon MPC)
			solver.exit_code = solver.scenario_module.optimize(data)

		# Retrieve the lowest cost solution
		lowest_cost = 1e9
		self.best_solver = None

		for solver in self.scenario_solvers:
			if solver.exit_code == 1 and solver.solver.info.pobj < lowest_cost:
				lowest_cost = solver.solver.info.pobj
				self.best_solver = solver

		if self.best_solver is None:  # No feasible solution
			return self.scenario_solvers[0].exit_code

		# Load the solution into the main lmpcc solver
		self.solver.output = self.best_solver.solver.output
		self.solver.info = self.best_solver.solver.info
		self.solver.params = self.best_solver.solver.params

		return self.best_solver.exit_code

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

			if self.enable_safe_horizon:
				# Draw different samples for all solvers in parallel
				for solver in self.scenario_solvers:
					solver.scenario_module.get_sampler().integrate_and_translate_to_mean_and_variance(
						data.dynamic_obstacles, solver.timestep
					)

	def is_data_ready(self, data):
		missing_data = ""
		max_obstacles = self.get_config_value("max_obstacles")
		if data.dynamic_obstacles.size() != max_obstacles:
			missing_data += "Obstacles "


		for i in range(data.dynamic_obstacles.size()):
			if data.dynamic_obstacles[i].prediction.empty():
				missing_data += "Obstacle Prediction "


			if data.dynamic_obstacles[i].prediction.type == DETERMINISTIC:
				missing_data += "Uncertain Predictions (scenario-based control cannot use deterministic predictions) "


		if not self.scenario_solvers[0].scenario_module.is_data_ready(data, missing_data):
			return False

		return len(missing_data) < 1

	def reset(self):
		super().reset()
		# Reset constraint-specific state
		self.best_solver = None

		# Reset all scenario solvers
		for solver in self.scenario_solvers:
			solver.exit_code = 0

class ScenarioSolver:
	def __init__(self, solver_id):
		self.info = None
		config = read_config_file()
		self.solver = None
		self.scenario_module = ScenarioModule()
		self.exit_code = 0
		self.horizon = config["horizon"]
		self.timestep = config["timestep"]
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