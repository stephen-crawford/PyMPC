import copy
import math

from planner_modules.src.constraints.linearized_constraints import LinearizedConstraints
from utils.const import DETERMINISTIC
from utils.utils import LOG_DEBUG, read_config_file
from datetime import datetime
from planner_modules.src.constraints.base_constraint import BaseConstraint

import copy
import time
from datetime import datetime

from planner_modules.src.constraints.linearized_constraints import LinearizedConstraints
from utils.const import DETERMINISTIC
from utils.utils import LOG_DEBUG, read_config_file
from planner_modules.src.constraints.base_constraint import BaseConstraint


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
		self.nh = self.num_constraints
		self.constraints.append(LinearizedConstraints(solver))

		# Create parallel solvers
		parallel_solvers = self.get_config_value("scenario_constraints.parallel_solvers")
		for i in range(parallel_solvers):
			self.scenario_solvers.append(ScenarioSolver(i, self.config))

		# Diagnostic information
		self.optimization_time = 0
		self.feasible_solutions = 0

		LOG_DEBUG("Scenario Constraints successfully initialized")

	def update(self, state, data, module_data):
		LOG_DEBUG("ScenarioConstraints.update")
		start_time = time.time()

		# First update the dynamic obstacles with uncertainty
		if self.enable_safe_horizon:
			for solver in self.scenario_solvers:
				solver.scenario_module.get_sampler().integrate_and_translate_to_mean_and_variance(
					data.dynamic_obstacles, solver.timestep
				)

		# Now update each solver
		for solver in self.scenario_solvers:
			# Use the main solver as a starting point

			# Update the scenario module with new data
			solver.scenario_module.update(data, module_data)

		update_time = time.time() - start_time
		LOG_DEBUG(f"ScenarioConstraints.update completed in {update_time:.3f}s")

	def optimize(self, state, data, module_data):
		LOG_DEBUG("ScenarioConstraints.optimize")
		start_time = time.time()

		# Reset feasible solutions counter
		self.feasible_solutions = 0

		# For each solver, set parameters and optimize
		for solver in self.scenario_solvers:
			# Set the planning timeout - allocate time proportionally
			used_time = (datetime.now() - data.planning_start_time).total_seconds()
			remaining_time = max(0.01, self.planning_time - used_time - 0.01)  # Safety margin
			solver.tmp_config.solver_timeout = remaining_time / len(self.scenario_solvers)

			print("Solver.solver.info pobj is: " + str(solver.solver.info.pobj))
			# Copy the solver
			# Set the scenario constraint parameters for each time step in the horizon
			for k in range(solver.horizon):
				solver.scenario_module.set_parameters(data, k)

			# Load the previous solution as warmstart if available
			if hasattr(solver.solver, 'load_warmstart'):
				solver.solver.load_warmstart()

			# Run optimization
			LOG_DEBUG(f"Running optimization for solver {solver.solver_id}")
			solver.exit_code = solver.scenario_module.optimize(data)

			# Count feasible solutions
			if solver.exit_code == 1:
				self.feasible_solutions += 1
				LOG_DEBUG(f"Solver {solver.solver_id} found feasible solution with cost {solver.solver.info.pobj}")
			else:
				LOG_DEBUG(f"Solver {solver.solver_id} failed to find a feasible solution")

		# Now find the best solution (lowest cost among feasible ones)
		lowest_cost = float('inf')
		self.best_solver = None

		for solver in self.scenario_solvers:
			print("Solver.solver.info pobj is: " + str(solver.solver.info.pobj))
			if solver.exit_code == 1:  # Feasible solution
				current_cost = solver.solver.info.pobj
				LOG_DEBUG(f"Solver {solver.solver_id} cost: {current_cost}")

				if current_cost < lowest_cost:
					lowest_cost = current_cost
					self.best_solver = solver

		# If we found a feasible solution, load it into the main solver
		if self.best_solver is not None:
			LOG_DEBUG(f"Selected best solver {self.best_solver.solver_id} with cost {lowest_cost}")
			self.solver.output = self.best_solver.solver.output
			self.solver.info = self.best_solver.solver.info
			self.solver.tmp_config = self.best_solver.solver.tmp_config

			# Calculate optimization time
			self.optimization_time = time.time() - start_time
			LOG_DEBUG(f"Optimization completed in {self.optimization_time:.3f}s")

			return 1  # Success
		else:
			LOG_DEBUG("No feasible solution found by any solver")
			self.optimization_time = time.time() - start_time

			# If no feasible solution, try to return the closest to feasible for graceful degradation
			min_constraint_violation = float('inf')
			closest_to_feasible = None

			for solver in self.scenario_solvers:
				if hasattr(solver.solver, 'info') and hasattr(solver.solver.info, 'constraint_violation'):
					violation = solver.solver.info.constraint_violation
					if violation < min_constraint_violation:
						min_constraint_violation = violation
						closest_to_feasible = solver

			if closest_to_feasible is not None:
				LOG_DEBUG(f"Selected closest to feasible solver {closest_to_feasible.solver_id}")
				self.solver.output = closest_to_feasible.solver.output
				self.solver.info = closest_to_feasible.solver.info
				self.solver.tmp_config = closest_to_feasible.solver.tmp_config

			return 0  # Failure

	def on_data_received(self, data, data_name):
		LOG_DEBUG(f"ScenarioConstraints.on_data_received({data_name})")

		if data_name == "dynamic obstacles":
			# Check if uncertainty was provided
			for obs in data.dynamic_obstacles:
				if obs.prediction.type == DETERMINISTIC:
					LOG_DEBUG("WARNING: Using deterministic prediction with Scenario Constraints")
					LOG_DEBUG("Set `process_noise` to a non-zero value to add uncertainty.")
					return

			# Process the new obstacle data
			if self.enable_safe_horizon:
				for solver in self.scenario_solvers:
					solver.scenario_module.get_sampler().integrate_and_translate_to_mean_and_variance(
						data.dynamic_obstacles, solver.timestep
					)

	def is_data_ready(self, data):
		missing_data = ""
		max_obstacles = self.get_config_value("max_obstacles")

		# Check if we have the expected number of obstacles
		if data.dynamic_obstacles.size() > max_obstacles:
			missing_data += f"Too many obstacles (max: {max_obstacles}) "

		# Check if all obstacles have predictions
		for i in range(data.dynamic_obstacles.size()):
			if data.dynamic_obstacles[i].prediction.empty():
				missing_data += f"Obstacle {i} missing prediction "

			if data.dynamic_obstacles[i].prediction.type == DETERMINISTIC:
				missing_data += f"Obstacle {i} has deterministic prediction "

		# Check if any scenario solver is ready
		for solver in self.scenario_solvers:
			if not solver.scenario_module.is_data_ready(data):
				return False

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


class ScenarioSolver:
	def __init__(self, solver_id, config):
		self.config = config
		self.solver = None
		self.tmp_config = None
		self.scenario_module = ScenarioModule()
		self.exit_code = 0
		self.horizon = config["horizon"]
		self.timestep = config["timestep"]
		self.solver_id = solver_id

		LOG_DEBUG(f"Initialized ScenarioSolver {solver_id}")

	def get(self):
		return self


class ScenarioModule:
	def __init__(self):
		# Load config values
		self.config = read_config_file()
		self.horizon = self.config["horizon"]
		self.num_discs = self.config["num_discs"]
		self.max_obstacles = self.config["max_obstacles"]
		self.num_scenarios = self.config["scenario_constraints.num_scenarios"]

		# Initialize scenario-related variables
		self.sampler = ScenarioSampler(self.num_scenarios, self.horizon)
		self.samples = {}  # Stores scenarios for each obstacle
		self.selected_scenarios = []
		self.scenario_probabilities = []

		# Constraints and optimization parameters
		self.constraint_matrices = {}
		self.slack_variables = {}
		self.slack_penalty = self.config["scenario_constraints.slack_penalty"]

		LOG_DEBUG("ScenarioModule initialized")

	def update(self, data, module_data):
		"""Update the module with new data"""
		# Reset current scenarios
		self.samples = {}
		self.selected_scenarios = []

		# Generate new scenario samples for each obstacle
		for i in range(data.dynamic_obstacles.size()):
			obstacle = data.dynamic_obstacles[i]
			if not obstacle.prediction.empty() and obstacle.prediction.type != DETERMINISTIC:
				# Generate scenarios for this obstacle
				self.samples[i] = self.sampler.generate_samples(obstacle)

		# Update module data for visualization
		if "scenario_samples" not in module_data:
			module_data["scenario_samples"] = self.samples

		LOG_DEBUG(f"Updated with {len(self.samples)} obstacles with scenarios")

	def set_parameters(self, data, k):
		"""Set scenario constraints at time step k"""
		if not self.samples:
			LOG_DEBUG("No scenario samples available")
			return

		# For each obstacle, construct constraints based on scenarios
		for obs_id, scenarios in self.samples.items():
			obstacle = data.dynamic_obstacles[obs_id]

			# For each scenario, create a linearized constraint
			for scenario_idx, scenario in enumerate(scenarios[k]):
				# Extract predicted position and uncertainty
				pos_x, pos_y = scenario.position
				uncertainty = scenario.uncertainty

				# Create linearized constraint for this scenario
				# A * x + B * y + C <= 0
				A, B, C = self._linearize_constraint(pos_x, pos_y, uncertainty)

				# Add constraint to the solver
				constraint_id = f"scenario_{obs_id}_{scenario_idx}_t{k}"
				self.constraint_matrices[constraint_id] = (A, B, C)

				# If using slack variables, add them
				if self.config["scenario_constraints.use_slack"]:
					slack_var = f"slack_{obs_id}_{scenario_idx}_t{k}"
					self.slack_variables[slack_var] = self.slack_penalty

		LOG_DEBUG(f"Set parameters for time step {k}")

	def optimize(self, data):
		"""Run the optimization with current scenario constraints"""
		try:
			# Configure the solver with scenario constraints
			for constraint_id, (A, B, C) in self.constraint_matrices.items():
				# Add constraint to the solver
				# Note: The exact syntax depends on your solver interface
				self.solver.add_constraint(A, B, C, constraint_id)

			# Add slack variables if enabled
			if self.config["scenario_constraints.use_slack"]:
				for slack_var, penalty in self.slack_variables.items():
					self.solver.add_slack_variable(slack_var, penalty)

			# Run the solver
			result = self.solver.solve()

			# Check for feasibility
			if result.is_feasible():
				LOG_DEBUG("Scenario optimization successful")
				return 1  # Success
			else:
				LOG_DEBUG("Scenario optimization infeasible")
				return 0  # Failure

		except Exception as e:
			LOG_DEBUG(f"Error in optimization: {str(e)}")
			return -1  # Error

	def get_sampler(self):
		"""Return the scenario sampler"""
		return self.sampler

	def is_data_ready(self, data):
		"""Check if all required data is available"""
		missing_data = ""
		if not data.has("dynamic_obstacles"):
			missing_data += "No dynamic obstacles "


		# Check if we have predictions for each obstacle
		for i in range(data.dynamic_obstacles.size()):
			obstacle = data.dynamic_obstacles[i]
			if obstacle.prediction.empty():
				missing_data += f"Missing prediction for obstacle {i} "


			# Check if predictions have uncertainties
			if obstacle.prediction.type == DETERMINISTIC:
				missing_data += f"Obstacle {i} has deterministic prediction "

			# Check if predictions cover the whole horizon
			if obstacle.prediction.duration < self.horizon:
				missing_data += f"Prediction horizon too short for obstacle {i} "


		return len(missing_data) < 1

	def _linearize_constraint(self, x, y, uncertainty):
		"""Helper method to linearize constraints based on position and uncertainty"""
		# This is a placeholder - actual implementation depends on your specific constraint formulation
		# For example, for collision avoidance with a circular obstacle:
		# The constraint might be ||(x,y) - (obs_x, obs_y)|| >= safe_distance
		# Linearized as: A*x + B*y + C <= 0

		# Simplistic implementation for demonstration
		safe_distance = uncertainty + self.config["vehicle_radius"]

		# Direction from obstacle to vehicle (unit vector)
		norm = max(1e-6, math.sqrt(x * x + y * y))
		dx, dy = x / norm, y / norm

		# Linearized constraint coefficients
		A = -dx
		B = -dy
		C = safe_distance - (A * x + B * y)

		return A, B, C


import numpy as np
from scipy.stats import multivariate_normal
from dataclasses import dataclass


@dataclass
class ScenarioSample:
	"""Class to store a scenario sample"""
	position: tuple  # (x, y) position
	velocity: tuple  # (vx, vy) velocity
	uncertainty: float  # Uncertainty radius
	probability: float  # Probability of this scenario


class ScenarioSampler:
	def __init__(self, num_scenarios=5, horizon=10):
		"""Initialize the scenario sampler

		Args:
			num_scenarios: Number of scenarios to generate per obstacle
			horizon: Planning horizon (number of time steps)
		"""
		self.num_scenarios = num_scenarios
		self.horizon = horizon
		self.config = read_config_file()
		self.dt = self.config["timestep"]

		# Parameters for uncertainty modeling
		self.position_uncertainty_growth = self.config.get("scenario_constraints.position_uncertainty_growth", 0.1)
		self.velocity_uncertainty_growth = self.config.get("scenario_constraints.velocity_uncertainty_growth", 0.05)

		# Random seed for reproducibility (if needed)
		self.rng = np.random.RandomState(42)

		LOG_DEBUG(f"ScenarioSampler initialized with {num_scenarios} scenarios")

	def generate_samples(self, obstacle):
		"""Generate scenario samples for an obstacle over the planning horizon

		Args:
			obstacle: Dynamic obstacle with prediction

		Returns:
			List of scenarios for each time step in the horizon
		"""
		# Initialize samples container for each time step
		horizon_samples = [[] for _ in range(self.horizon)]

		# Get the initial mean and covariance from the obstacle prediction
		mean_pos = np.array([obstacle.prediction.positions[0].x, obstacle.prediction.positions[0].y])
		mean_vel = np.array([obstacle.prediction.velocities[0].x, obstacle.prediction.velocities[0].y])

		# Initial covariance (assuming it's provided in the prediction)
		# If not available, we'll use a default
		if hasattr(obstacle.prediction, 'covariances') and len(obstacle.prediction.covariances) > 0:
			cov_pos = np.array([
				[obstacle.prediction.covariances[0].xx, obstacle.prediction.covariances[0].xy],
				[obstacle.prediction.covariances[0].xy, obstacle.prediction.covariances[0].yy]
			])
		else:
			# Default initial covariance
			initial_pos_uncertainty = self.config.get("scenario_constraints.initial_pos_uncertainty", 0.1)
			cov_pos = np.eye(2) * initial_pos_uncertainty ** 2

		# Initial velocity covariance
		if hasattr(obstacle.prediction, 'velocity_covariances') and len(obstacle.prediction.velocity_covariances) > 0:
			cov_vel = np.array([
				[obstacle.prediction.velocity_covariances[0].xx, obstacle.prediction.velocity_covariances[0].xy],
				[obstacle.prediction.velocity_covariances[0].xy, obstacle.prediction.velocity_covariances[0].yy]
			])
		else:
			# Default initial velocity covariance
			initial_vel_uncertainty = self.config.get("scenario_constraints.initial_vel_uncertainty", 0.05)
			cov_vel = np.eye(2) * initial_vel_uncertainty ** 2

		# Generate samples for each time step
		for t in range(self.horizon):
			# Update mean based on deterministic prediction if available
			if t < len(obstacle.prediction.positions):
				mean_pos = np.array([obstacle.prediction.positions[t].x, obstacle.prediction.positions[t].y])
				mean_vel = np.array([obstacle.prediction.velocities[t].x, obstacle.prediction.velocities[t].y])
			else:
				# Propagate using simple constant velocity model
				mean_pos = mean_pos + mean_vel * self.dt

			# Increase uncertainty over time
			time_factor = t * self.dt
			current_pos_cov = cov_pos + np.eye(2) * (self.position_uncertainty_growth * time_factor) ** 2
			current_vel_cov = cov_vel + np.eye(2) * (self.velocity_uncertainty_growth * time_factor) ** 2

			# Generate samples around the mean
			pos_distribution = multivariate_normal(mean=mean_pos, cov=current_pos_cov)
			vel_distribution = multivariate_normal(mean=mean_vel, cov=current_vel_cov)

			# Generate samples with equal probability
			scenario_probability = 1.0 / self.num_scenarios

			for i in range(self.num_scenarios):
				# Sample position and velocity
				sample_pos = pos_distribution.rvs(random_state=self.rng)
				sample_vel = vel_distribution.rvs(random_state=self.rng)

				# Calculate uncertainty radius (could be based on covariance eigenvalues)
				pos_uncertainty = np.sqrt(np.max(np.linalg.eigvals(current_pos_cov)))

				# Create and store the sample
				sample = ScenarioSample(
					position=(sample_pos[0], sample_pos[1]),
					velocity=(sample_vel[0], sample_vel[1]),
					uncertainty=pos_uncertainty,
					probability=scenario_probability
				)
				horizon_samples[t].append(sample)

		return horizon_samples

	def integrate_and_translate_to_mean_and_variance(self, dynamic_obstacles, dt):
		"""Integrate obstacles over time and compute their mean and variance

		Args:
			dynamic_obstacles: List of dynamic obstacles
			dt: Time step
		"""
		for i in range(dynamic_obstacles.size()):
			obstacle = dynamic_obstacles[i]
			if obstacle.prediction.empty() or obstacle.prediction.type == DETERMINISTIC:
				continue

			# This would integrate the obstacle's state forward in time
			# and update its prediction with new mean and covariance

			# For each time step in the prediction
			for t in range(min(self.horizon, len(obstacle.prediction.positions))):
				# Get current position and velocity
				pos = np.array([obstacle.prediction.positions[t].x, obstacle.prediction.positions[t].y])
				vel = np.array([obstacle.prediction.velocities[t].x, obstacle.prediction.velocities[t].y])

				# Update position using velocity
				if t + 1 < len(obstacle.prediction.positions):
					new_pos = pos + vel * dt
					obstacle.prediction.positions[t + 1].x = new_pos[0]
					obstacle.prediction.positions[t + 1].y = new_pos[1]

					# Update covariance
					time_factor = dt
					pos_uncertainty_growth = self.position_uncertainty_growth * time_factor

					# Simplified covariance propagation
					if hasattr(obstacle.prediction, 'covariances') and t + 1 < len(obstacle.prediction.covariances):
						obstacle.prediction.covariances[t + 1].xx += pos_uncertainty_growth ** 2
						obstacle.prediction.covariances[t + 1].yy += pos_uncertainty_growth ** 2


import time
import numpy as np
from datetime import datetime
from utils.utils import LOG_DEBUG


class ScenarioDiagnostics:
	"""Class for diagnostics and performance monitoring of scenario-based MPC"""

	def __init__(self):
		# Performance metrics
		self.optimization_times = []
		self.success_rates = []
		self.constraint_violations = []
		self.costs = []

		# Scenario statistics
		self.num_scenarios_evaluated = []
		self.num_feasible_scenarios = []

		# Timing information
		self.last_update_time = None
		self.last_optimize_time = None
		self.cumulative_update_time = 0
		self.cumulative_optimize_time = 0
		self.iteration_count = 0

		LOG_DEBUG("ScenarioDiagnostics initialized")

	def start_update(self):
		"""Mark the start of an update operation"""
		self.last_update_time = time.time()

	def end_update(self, num_scenarios=0):
		"""Mark the end of an update operation and record statistics

		Args:
			num_scenarios: Number of scenarios evaluated in this update
		"""
		if self.last_update_time is not None:
			update_time = time.time() - self.last_update_time
			self.cumulative_update_time += update_time
			LOG_DEBUG(f"Update completed in {update_time:.3f}s")

			self.num_scenarios_evaluated.append(num_scenarios)

	def start_optimize(self):
		"""Mark the start of an optimization operation"""
		self.last_optimize_time = time.time()

	def end_optimize(self, num_feasible=0, best_cost=None, constraint_violation=0):
		"""Mark the end of an optimization operation and record statistics

		Args:
			num_feasible: Number of feasible solutions found
			best_cost: Cost of the best solution
			constraint_violation: Constraint violation of the best solution
		"""
		if self.last_optimize_time is not None:
			optimize_time = time.time() - self.last_optimize_time
			self.cumulative_optimize_time += optimize_time
			self.optimization_times.append(optimize_time)

			# Record success rate (0 if no feasible solution)
			if num_feasible > 0:
				success_rate = 1.0
			else:
				success_rate = 0.0