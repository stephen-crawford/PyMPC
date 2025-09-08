import copy
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import casadi as cd

from planner_modules.src.constraints.scenario_utils.math_utils import SafeHorizon, SafetyCertifier
from planner_modules.src.constraints.scenario_utils.sampler import ScenarioSampler
from planner_modules.src.constraints.base_constraint import BaseConstraint
from planning.src.types import (
	ScenarioStatus, ScenarioSolveStatus, ScenarioBase,
	SupportSubsample, PredictionType
)
from solver.src.modules_manager import Module
from utils import utils
from utils.utils import LOG_DEBUG, LOG_WARN, LOG_INFO, read_config_file


class ScenarioModule(Module):
	def __init__(self, solver):
		super().__init__()
		self.solver = solver
		self.status = ScenarioStatus.RESET

		# Load configuration
		self.config = read_config_file()
		self.horizon = self.get_config_value("horizon")
		self.num_discs = self.get_config_value("num_discs", 1)
		self.max_obstacles = self.get_config_value("max_obstacles")
		self.num_scenarios = self.get_config_value("scenario_constraints.num_scenarios")
		self.enable_safe_horizon = self.get_config_value("scenario_constraints.enable_safe_horizon", True)
		self.max_iterations = self.get_config_value("scenario_constraints.max_iterations", 10)

		# Initialize components
		self.safety_certifier = SafetyCertifier()
		self.disc_manager = []

		# Initialize disc managers based on safe horizon setting
		if self.enable_safe_horizon:
			self.sampler = ScenarioSampler()
			for disc_id in range(self.num_discs):
				self.disc_manager.append(SafeHorizon(disc_id, self.solver, self.sampler))
		else:
			self.sampler = None
			for _ in range(self.num_discs):
				self.disc_manager.append(ScenarioBase())

		# Initialize scenario-related variables
		self.samples = {}
		self.selected_scenarios = []
		self.scenario_probabilities = []
		self.constraint_matrices = {}
		self.slack_variables = {}
		self.slack_penalty = self.get_config_value("scenario_constraints.slack_penalty", 1000.0)

		# Solve status tracking
		self.solve_status = ScenarioSolveStatus.INFEASIBLE

		LOG_DEBUG("ScenarioModule initialized")

	def get_config_value(self, key, default=None):
		"""Get configuration value with proper fallback logic"""
		try:
			# Try direct key access
			if key in self.config:
				return self.config[key]

			# Try nested access for scenario_constraints
			if '.' in key:
				parts = key.split('.')
				value = self.config
				for part in parts:
					if isinstance(value, dict) and part in value:
						value = value[part]
					else:
						return default
				return value

			return default
		except Exception:
			return default

	def update(self, data):
		"""Update the module with new data"""
		try:
			# Update all disc managers
			for disc in self.disc_manager:
				if hasattr(disc, 'update'):
					disc.update(data)

			# Check if all discs are in SUCCESS status
			all_success = all(
				getattr(disc, 'status', ScenarioStatus.SUCCESS) == ScenarioStatus.SUCCESS
				for disc in self.disc_manager
			)

			if all_success:
				self.status = ScenarioStatus.SUCCESS
			else:
				self.status = ScenarioStatus.RESET
				LOG_WARN("Not all discs are feasible")

		except Exception as e:
			LOG_WARN(f"Error in ScenarioModule.update: {e}")
			self.status = ScenarioStatus.RESET

	def set_parameters(self, data, step):
		"""Set parameters for all disc managers"""
		try:
			for disc in self.disc_manager:
				if hasattr(disc, 'set_parameters'):
					disc.set_parameters(data, step)
		except Exception as e:
			LOG_WARN(f"Error setting parameters: {e}")

	def reset(self):
		"""Reset the module state"""
		self.status = ScenarioStatus.RESET
		self.solve_status = ScenarioSolveStatus.INFEASIBLE
		self.selected_scenarios.clear()
		self.scenario_probabilities.clear()

		for disc in self.disc_manager:
			if hasattr(disc, 'reset'):
				disc.reset()

	def optimize(self, data):
		"""Run the optimization with current scenario constraints"""
		timeout_timer = utils.Timer(getattr(self.solver, 'timeout', 5.0))
		timeout_timer.start()

		iteration_time = 0
		exit_code = -1
		is_feasible = True
		previous_support_estimate = 0
		new_support_estimate = 0

		# Initialize support subsample
		max_support = self.safety_certifier.get_max_support()
		support = SupportSubsample(max_support)

		iteration_timer = utils.Timer(getattr(self.solver, 'timeout', 5.0))

		for iteration in range(self.max_iterations):
			iteration_timer.start()

			# Debug output
			if (self.get_config_value("scenario.debug_output", False) and
					(iteration == 0 or new_support_estimate > 0)):
				if hasattr(support, 'print_update'):
					support.print_update(
						getattr(self.solver, 'id', 0),
						self.safety_certifier.get_safe_support_bound(),
						iteration
					)

			# Initialize solver for first iteration
			if iteration == 0:
				if hasattr(self.solver, 'initialize_single_iteration'):
					self.solver.initialize_single_iteration()

			# Solve iteration
			if hasattr(self.solver, 'solve_single_iteration'):
				exit_code = self.solver.solve_single_iteration()
			else:
				exit_code = 1  # Default success

			if hasattr(self.solver, 'complete_single_iteration'):
				exit_code = self.solver.complete_single_iteration()

			# Check feasibility
			feasible = True
			infeasible_scenarios = SupportSubsample()

			for disc in self.disc_manager:
				if hasattr(disc, 'compute_active_constraints'):
					disc_feasible = disc.compute_active_constraints(support, infeasible_scenarios)
					feasible = feasible and disc_feasible

			# Handle infeasible case
			if exit_code != 1:
				LOG_WARN(f"[Solver {getattr(self.solver, 'id', 0)}] SQP iteration {iteration} "
						 f"became infeasible with exit code {exit_code}")
				self.solve_status = ScenarioSolveStatus.INFEASIBLE
				if hasattr(self.solver, 'complete_single_iteration'):
					self.solver.complete_single_iteration()
				return exit_code

			# Merge support from all discs
			for disc in self.disc_manager:
				if hasattr(disc, 'support_subsample') and hasattr(support, 'merge_with'):
					disc.support_subsample.merge_with(support)

			# Check support bound
			if hasattr(support, 'get_size') and support.get_size() > max_support:
				break

			# Update support estimate
			new_support_estimate = (support.get_size() if hasattr(support, 'get_size')
									else 0) - previous_support_estimate
			previous_support_estimate = support.get_size() if hasattr(support, 'get_size') else 0

			# Update timing
			iteration_time += iteration_timer.current_runtime()
			iteration_timer.stop()
			iteration_timer.reset()

			# Check timeout
			average_iteration_time = iteration_time / (iteration + 1)
			if (timeout_timer.current_runtime() + average_iteration_time >
					getattr(self.solver, 'timeout', 5.0)):
				LOG_WARN(f"Stopping after {iteration + 1} iterations because planning time exceeded")
				break

		# Get slack value if available
		slack_val = 0.0
		if hasattr(self.solver, 'get_output'):
			try:
				slack_val = self.solver.get_output(1, "slack")
			except:
				slack_val = 0.0

		# Complete solver iteration
		if hasattr(self.solver, 'complete_single_iteration'):
			self.solver.complete_single_iteration()

		# Determine final status
		support_size = support.get_size() if hasattr(support, 'get_size') else 0

		if not is_feasible:
			self.solve_status = ScenarioSolveStatus.INFEASIBLE
			LOG_WARN("Failed to find a provable safe trajectory.")
		elif support_size > max_support:
			self.solve_status = ScenarioSolveStatus.SUPPORT_EXCEEDED
			LOG_WARN(f"Optimized trajectory was not provably safe: support bound exceeded "
					 f"({support_size} > {max_support}).")
		elif slack_val > 1e-3:
			self.solve_status = ScenarioSolveStatus.NONZERO_SLACK
			LOG_WARN(f"Optimized trajectory was not provably safe: slack value was not zero "
					 f"(value = {slack_val}).")
		else:
			self.solve_status = ScenarioSolveStatus.SUCCESS

		# Log support information
		if hasattr(self.safety_certifier, 'log_support'):
			self.safety_certifier.log_support(support_size)

		return exit_code

	def compute_active_constraints(self, active_constraint_aggregate, infeasible_scenarios):
		"""Compute active constraints for all discs"""
		feasible = True
		try:
			for disc in self.disc_manager:
				if hasattr(disc, 'compute_active_constraints'):
					disc_feasible = disc.compute_active_constraints(infeasible_scenarios)
					feasible = feasible and disc_feasible
		except Exception as e:
			LOG_WARN(f"Error computing active constraints: {e}")
			feasible = False

		return feasible

	def merge_support(self, support_estimate):
		"""Merge support estimates from all discs"""
		try:
			for disc in self.disc_manager:
				if hasattr(disc, 'support_subsample') and hasattr(disc.support_subsample, 'merge_with'):
					disc.support_subsample.merge_with(support_estimate)
		except Exception as e:
			LOG_WARN(f"Error merging support: {e}")

	def get_sampler(self):
		"""Return the scenario sampler"""
		return self.sampler

	def is_data_ready(self, data):
		"""Check if all required data is available"""
		try:
			missing_data = ""

			# Check sampler readiness if safe horizon is enabled
			if self.enable_safe_horizon and self.sampler:
				if not hasattr(self.sampler, 'samples_ready') or not self.sampler.samples_ready():
					missing_data += " Sampler "

			# Check disc manager readiness
			for i, disc in enumerate(self.disc_manager):
				if hasattr(disc, 'is_data_ready') and not disc.is_data_ready(data):
					missing_data += f" Disc{i} "

			return len(missing_data) < 1

		except Exception as e:
			LOG_WARN(f"Error checking data readiness: {e}")
			return False

	def is_scenario_status(self, expected_status, status_output):
		"""Check if all discs have the expected status"""
		try:
			for disc in self.disc_manager:
				current_status = getattr(disc, 'status', expected_status)
				if current_status != expected_status:
					if hasattr(status_output, '__setitem__'):
						status_output[0] = current_status
					return False

			if hasattr(status_output, '__setitem__'):
				status_output[0] = expected_status
			return True

		except Exception as e:
			LOG_WARN(f"Error checking scenario status: {e}")
			return False


class ScenarioSolver:
	def __init__(self, solver_id, base_solver):
		self.solver = copy.deepcopy(base_solver)
		self.scenario_module = ScenarioModule(self.solver)
		self.solver_timeout = 1000.0
		self.running_time = 0
		self.solver_id = solver_id
		self.exit_code = 1

		LOG_DEBUG(f"Initialized ScenarioSolver {self.solver_id}")

	def get(self):
		return self


class ScenarioConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)
		self.constraints = []
		self.name = "scenario_constraints"

		LOG_DEBUG("Initializing Scenario Constraints")

		# Load configuration
		self.config = read_config_file()
		self.planning_time = 1.0 / self.get_config_value("control_frequency", 10.0)

		# Initialize scenario solvers
		self.scenario_solvers = []
		self.best_solver = None

		# Load scenario configuration
		self.enable_safe_horizon = self.get_config_value("scenario_constraints.enable_safe_horizon", True)
		self.num_discs = self.get_config_value("num_discs", 1)
		self.num_constraints = (self.get_config_value("max_constraints", 10) * self.num_discs)
		self.use_slack = self.get_config_value("scenario_constraints.use_slack", True)

		# Create parallel solvers
		parallel_solvers = self.get_config_value("scenario_constraints.parallel_solvers", 4)
		for i in range(parallel_solvers):
			self.scenario_solvers.append(ScenarioSolver(i, solver))

		# Diagnostic information
		self.optimization_time = 0
		self.feasible_solutions = 0

		LOG_DEBUG("Scenario Constraints successfully initialized")

	def get_config_value(self, key, default=None):
		"""Get configuration value with proper fallback logic"""
		try:
			if '.' in key:
				parts = key.split('.')
				value = self.config
				for part in parts:
					if isinstance(value, dict) and part in value:
						value = value[part]
					else:
						return default
				return value
			else:
				return self.config.get(key, default)
		except Exception:
			return default

	def update(self, state, data):
		"""Update all scenario solvers with new data"""

		def worker(solver_wrapper):
			try:
				# Deep copy the main solver
				solver_wrapper.solver = copy.deepcopy(self.solver)

				# Update the scenario module
				if hasattr(solver_wrapper, 'scenario_module'):
					solver_wrapper.scenario_module.update(data)
			except Exception as e:
				LOG_WARN(f"Error updating solver {solver_wrapper.solver_id}: {e}")

		# Run in parallel
		max_workers = min(4, len(self.scenario_solvers))
		with ThreadPoolExecutor(max_workers=max_workers) as executor:
			executor.map(worker, self.scenario_solvers)

	def set_parameters(self, parameters):
		"""Set parameters for scenario constraints"""
		LOG_INFO("Set parameters called for Scenario Constraints")
		# This is a passthrough method as mentioned in the original code
		return

	def run_optimize(self, scenario_solver, main_solver, planning_time, start_time, data):
		"""Helper function for parallelized optimization of solvers"""
		try:
			# Set timeout
			used_time = time.time() - start_time
			scenario_solver.solver_timeout = max(0.1, planning_time - used_time - 0.008)

			# Copy solver state
			scenario_solver.solver = copy.deepcopy(main_solver)

			# Set scenario constraint parameters
			horizon = getattr(main_solver, 'horizon',
							  self.get_config_value('horizon', 10))

			for step in range(horizon):
				if hasattr(scenario_solver.scenario_module, 'set_parameters'):
					scenario_solver.scenario_module.set_parameters(data, step)

			# Run optimization
			exit_code = scenario_solver.scenario_module.optimize(data)

			# Get objective value
			objective_value = getattr(scenario_solver.solver.info, 'pobj', float('inf'))

			return exit_code, objective_value, scenario_solver

		except Exception as e:
			LOG_WARN(f"Error in run_optimize for solver {scenario_solver.solver_id}: {e}")
			return -1, float('inf'), scenario_solver

	def optimize(self, state, data):
		"""Run optimization with scenario constraints"""
		LOG_DEBUG("ScenarioConstraints.optimize")
		start_time = time.time()

		# Run parallel optimization
		max_workers = min(4, len(self.scenario_solvers))
		with ThreadPoolExecutor(max_workers=max_workers) as executor:
			futures = [
				executor.submit(
					self.run_optimize,
					scenario_solver,
					self.solver,
					self.planning_time,
					start_time,
					data
				)
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

		# Handle case where no solver succeeded
		if best_solver is None:
			LOG_WARN("No scenario solver found a feasible solution")
			return self.scenario_solvers[0].exit_code

		# Copy results from best solver
		try:
			self.solver.output = getattr(best_solver.solver, 'output', None)
			self.solver.info = getattr(best_solver.solver, 'info', None)
			self.solver.params = getattr(best_solver.solver, 'params', None)
		except Exception as e:
			LOG_WARN(f"Error copying solver results: {e}")

		return best_solver.exit_code

	def on_data_received(self, data):
		"""Process incoming data for scenario constraints"""
		try:
			# Check for dynamic obstacles
			if not (hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles is not None):
				return

			# Validate prediction types
			for obs in data.dynamic_obstacles:
				if not hasattr(obs, 'prediction') or obs.prediction is None:
					continue

				if (hasattr(obs.prediction, 'type') and
						obs.prediction.type == PredictionType.DETERMINISTIC):
					LOG_DEBUG("WARNING: Using deterministic prediction with Scenario Constraints")
					LOG_DEBUG("Set `process_noise` to a non-zero value to add uncertainty.")
					return

			# Process obstacle data if safe horizon is enabled
			if self.enable_safe_horizon:
				def worker(solver_wrapper):
					try:
						sampler = solver_wrapper.scenario_module.get_sampler()
						if sampler and hasattr(sampler, 'integrate_and_translate_to_mean_and_variance'):
							timestep = getattr(self.solver, 'timestep',
											   self.get_config_value('timestep', 0.1))
							sampler.integrate_and_translate_to_mean_and_variance(
								data.dynamic_obstacles, timestep
							)
					except Exception as e:
						LOG_WARN(f"Error processing obstacle data for solver "
								 f"{solver_wrapper.solver_id}: {e}")

				# Parallelize data processing
				max_workers = min(4, len(self.scenario_solvers))
				with ThreadPoolExecutor(max_workers=max_workers) as executor:
					executor.map(worker, self.scenario_solvers)

		except Exception as e:
			LOG_WARN(f"Error in on_data_received: {e}")

	def is_data_ready(self, data):
		"""Check if all required data is available"""
		try:
			missing_data = ""

			# Check for dynamic obstacles
			if not (hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles is not None):
				missing_data += "Dynamic Obstacles "
				LOG_DEBUG(f"Missing dynamic_obstacles: {missing_data}")
				return False

			# Validate obstacle predictions
			for i, obs in enumerate(data.dynamic_obstacles):
				if not hasattr(obs, 'prediction') or obs.prediction is None:
					missing_data += "Obstacle Prediction "
					continue

				if (hasattr(obs.prediction, 'type') and
						obs.prediction.type != PredictionType.GAUSSIAN):
					missing_data += f"Obstacle Prediction (type must be gaussian) for obstacle {i}"

			# Check scenario solver readiness
			if self.scenario_solvers:
				for solver in self.scenario_solvers:
					if not solver.scenario_module.is_data_ready(data):
						missing_data += "Missing data required for Scenario Solvers"
						break

			is_ready = len(missing_data) < 1
			if not is_ready:
				LOG_DEBUG(f"Missing data in Scenario Constraints: {missing_data}")

			return is_ready

		except Exception as e:
			LOG_WARN(f"Error checking data readiness: {e}")
			return False

	def reset(self):
		"""Reset constraint state"""
		try:
			super().reset()

			# Reset constraint-specific state
			self.best_solver = None
			self.optimization_time = 0
			self.feasible_solutions = 0

			# Reset all scenario solvers
			for solver in self.scenario_solvers:
				solver.exit_code = 0
				if hasattr(solver.scenario_module, 'reset'):
					solver.scenario_module.reset()

		except Exception as e:
			LOG_WARN(f"Error in reset: {e}")