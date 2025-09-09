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
from utils.utils import LOG_DEBUG, LOG_WARN, LOG_INFO, read_config_file, get_config_dotted


class ScenarioModule(Module):
    def __init__(self, solver):
       super().__init__()
       self.solver = solver
       self.status = ScenarioStatus.RESET

       # Load configuration
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
       res = self.config.get(key, self.config.get(f"{self.name}.{key}", default))
       if res is None:
          res = get_config_dotted(self.config, key)
       return res

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
             if not hasattr(self.sampler, 'samples_ready'):
                missing_data += " Sampler no samples_ready"
             if  not self.sampler.samples_ready():
                missing_data += " Sampler samples not ready"

          # Check disc manager readiness
          for i, disc in enumerate(self.disc_manager):
             if hasattr(disc, 'is_data_ready') and not disc.is_data_ready(data):
                missing_data += f" Disc{i} "
          if len(missing_data) > 0:
             LOG_WARN(f"Scenario module missing data: " + missing_data)
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