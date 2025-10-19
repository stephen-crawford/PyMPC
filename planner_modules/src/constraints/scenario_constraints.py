import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planner_modules.src.constraints.scenario_utils.scenario_module import ScenarioSolver
from planning.src.types import PredictionType
from utils.utils import LOG_DEBUG, LOG_WARN


class ScenarioConstraints(BaseConstraint):
   def __init__(self, solver):
      super().__init__(solver)
      self.constraints = []
      self.name = "scenario_constraints"

      LOG_DEBUG("Initializing Scenario Constraints")

      # Load configuration
      self.planning_time = 1.0 / self.get_config_value("control_frequency", 10.0)
      self.num_discs = self.get_config_value("num_discs", 1)
      self.max_constraints_per_disc = self.get_config_value("scenario_constraints.max_constraints", 24)
      self.use_slack = self.get_config_value("scenario_constraints.use_slack", True)
      self.slack_penalty_weight = self.get_config_value("scenario_constraints.slack_penalty_weight", 1000.0)
      self.enable_safe_horizon = self.get_config_value("scenario_constraints.enable_safe_horizon", True)
      self.slack = self.get_config_value("scenario_constraints.slack", 0.0)
      # Initialize scenario solvers for parallel computation
      self.scenario_solvers = []
      self.best_solver = None
      parallel_solvers = self.get_config_value("scenario_constraints.parallel_solvers", 4)
      for i in range(parallel_solvers):
         self.scenario_solvers.append(ScenarioSolver(i, solver))

      # Storage for the computed numeric constraint coefficients
      self._a1 = np.zeros((self.num_discs, self.solver.horizon, self.max_constraints_per_disc))
      self._a2 = np.zeros((self.num_discs, self.solver.horizon, self.max_constraints_per_disc))
      self._b = np.full((self.num_discs, self.solver.horizon, self.max_constraints_per_disc),
                        100.0)  # Default "far away" constraint

      self._dummy_a1 = 0.0
      self._dummy_a2 = 0.0
      self._dummy_b = 100.0

      LOG_DEBUG("Scenario Constraints successfully initialized")

   def update(self, state, data):
      """
	  **MODIFIED**: This method now triggers the parallel computation to find and store
	  the optimal scenario constraints for the current planning step.
	  """
      LOG_DEBUG(f"{self.name}::update -> Computing optimal scenario constraints")
      start_time = time.time()

      # Run parallel workers to find the best set of constraints
      with ThreadPoolExecutor(max_workers=len(self.scenario_solvers)) as executor:
         futures = [
            executor.submit(self.run_optimize_worker, s, self.solver, data, start_time)
            for s in self.scenario_solvers
         ]
         results = [f.result() for f in futures]

      # Select the best result
      best_solver = None
      lowest_cost = float('inf')
      for exit_code, cost, solver_wrapper in results:
         if exit_code == 1 and cost < lowest_cost:
            lowest_cost = cost
            best_solver = solver_wrapper

      if best_solver is None:
         LOG_WARN("No scenario solver found a feasible solution for constraints.")
         return

      # Store the best solver for visualization and debugging
      self.best_solver = best_solver

      # Extract and store the computed constraint coefficients from the best solver's SafeHorizon module
      for disc_id, disc_manager in enumerate(self.best_solver.scenario_module.disc_manager):
         for step in range(self.solver.horizon):
            if hasattr(disc_manager, 'polytopes') and step < len(disc_manager.polytopes):
               polytope = disc_manager.polytopes[step]
               num_found_constraints = min(len(polytope.polygon_out), self.max_constraints_per_disc)

               for i in range(num_found_constraints):
                  constraint = polytope.polygon_out[i]
                  self._a1[disc_id][step][i] = constraint.a1
                  self._a2[disc_id][step][i] = constraint.a2
                  self._b[disc_id][step][i] = constraint.b

   def on_data_received(self, data):
      """
	  **MODIFIED**: This method is now simplified. The core sampling logic
	  has been moved to the run_optimize_worker to ensure it runs within
	  the correct context of each parallel solver instance.
	  """
      # We can keep this for future pre-processing, but the main sampling is moved.
      pass

   def run_optimize_worker(self, scenario_solver, main_solver, data, start_time):
      """
	  **FINAL FIX**: Added initialize_rollout() to ensure each worker's solver
	  starts its optimization from the vehicle's current state.
	  """
      try:
         used_time = time.time() - start_time
         scenario_solver.solver_timeout = max(0.1, self.planning_time - used_time - 0.008)
         scenario_solver.solver = main_solver.copy()

         # **THIS IS THE CRUCIAL ADDITION**
         # Initialize the copied solver with the current vehicle state.
         # This gives the inner optimization a good starting point.
         # We get the state from the main_solver, which is up-to-date.
         scenario_solver.solver.initialize_rollout(main_solver.initial_state)

         # The sampling logic we added before
         if self.enable_safe_horizon:
            sampler = scenario_solver.scenario_module.get_sampler()
            if sampler and hasattr(sampler, 'integrate_and_translate_to_mean_and_variance'):
               timestep = getattr(self.solver, 'timestep', 0.1)
               sampler.integrate_and_translate_to_mean_and_variance(
                  data.dynamic_obstacles, timestep
               )

         # Now, the update call will have both a good initial trajectory AND the scenarios
         scenario_solver.scenario_module.update(data)
         exit_code = scenario_solver.scenario_module.optimize(data)

         objective_value = float('inf')
         if hasattr(scenario_solver.solver, 'solution') and scenario_solver.solver.solution:
            objective_value = scenario_solver.solver.solution.optval

         return exit_code, objective_value, scenario_solver
      except Exception as e:
         LOG_WARN(f"Error in scenario worker for solver {scenario_solver.solver_id}: {e}")
         return -1, float('inf'), scenario_solver

   def define_parameters(self, params):
      """Define symbolic parameters for the constraints in the main CasADi solver."""
      for disc_id in range(self.num_discs):
         for step in range(self.solver.horizon + 1):
            for i in range(self.max_constraints_per_disc):
               base_name = f"disc_{disc_id}_scen_constraint_{i}_step_{step}"
               LOG_DEBUG(f"Defining parameters for {base_name}")
               params.add(f"{base_name}_a1")
               params.add(f"{base_name}_a2")
               params.add(f"{base_name}_b")

   def get_constraints(self, symbolic_state, params, stage_idx):
      """
      **FIXED**: This method no longer skips dummy constraints. It creates an expression
      for every constraint slot, ensuring its output list has a consistent length
      that matches the bound lists.
      """
      if stage_idx == 0:
         return []

      constraints = []
      pos_x = symbolic_state.get("x")
      pos_y = symbolic_state.get("y")

      for disc_id in range(self.num_discs):
         for i in range(self.max_constraints_per_disc):
            LOG_DEBUG(f"Trying to get constraints for disc #{disc_id} constraint number {i} at step {stage_idx}.")
            base_name = f"disc_{disc_id}_scen_constraint_{i}_step_{stage_idx}"
            a1 = params.get(f"{base_name}_a1")
            a2 = params.get(f"{base_name}_a2")
            b = params.get(f"{base_name}_b")

            # **THE FIX IS HERE**: The check for dummy constraints is removed.
            # We create an expression for every slot. The optimizer can handle
            # constant expressions like 0*x + 0*y - 100 <= 0 perfectly fine.
            constraint_expr = a1 * pos_x + a2 * pos_y - b - self.slack
            constraints.append(constraint_expr)

      return constraints

   def get_lower_bound(self):
      return [-np.inf] * (self.num_discs * self.max_constraints_per_disc)

   def get_upper_bound(self):
      # All constraints are of the form Expression <= 0
      return [0.0] * (self.num_discs * self.max_constraints_per_disc)

   def get_penalty(self, symbolic_state, params, stage_idx):
      """Adds a penalty to the objective function for using the slack variable."""

      try:
         slack = symbolic_state.get("slack") if self.use_slack else 0.0
      except:
         slack = 0.0

      return self.slack_penalty_weight * slack ** 2

   # **MODIFICATION 4: Flesh out set_parameters**
   def set_parameters(self, parameter_manager, data, step):
      """Populate the symbolic parameters with the computed numeric values."""
      if step == 0:
         for disc_id in range(self.num_discs):
            for i in range(self.max_constraints_per_disc):
               base_name = f"disc_{disc_id}_scen_constraint_{i}_step_{step}"
               LOG_DEBUG(f"Setting parameters for {base_name}")

               parameter_manager.set_parameter(f"{base_name}_a1", self._dummy_a1)
               parameter_manager.set_parameter(f"{base_name}_a2", self._dummy_a2)
               parameter_manager.set_parameter(f"{base_name}_b",  self._dummy_b)
         return

      for disc_id in range(self.num_discs):
         for i in range(self.max_constraints_per_disc):
            base_name = f"disc_{disc_id}_scen_constraint_{i}_step_{step}"
            LOG_DEBUG(f"Setting parameters for {base_name}")
            
            # **FIX**: Add bounds checking and fallback to dummy values
            try:
               # Check if the arrays exist and have the right dimensions
               if (hasattr(self, '_a1') and disc_id < len(self._a1) and 
                   step - 1 < len(self._a1[disc_id]) and i < len(self._a1[disc_id][step - 1])):
                  a1_val = self._a1[disc_id][step - 1][i]
                  a2_val = self._a2[disc_id][step - 1][i]
                  b_val = self._b[disc_id][step - 1][i]
               else:
                  # Fallback to dummy values if arrays are not properly initialized
                  a1_val = self._dummy_a1
                  a2_val = self._dummy_a2
                  b_val = self._dummy_b
            except (IndexError, AttributeError) as e:
               LOG_DEBUG(f"Using dummy values for {base_name} due to: {e}")
               a1_val = self._dummy_a1
               a2_val = self._dummy_a2
               b_val = self._dummy_b

            parameter_manager.set_parameter(f"{base_name}_a1", a1_val)
            parameter_manager.set_parameter(f"{base_name}_a2", a2_val)
            parameter_manager.set_parameter(f"{base_name}_b", b_val)

   def is_data_ready(self, data):
      """
	  **FIXED**: Corrected the logic for checking the obstacle prediction type.
	  It now correctly checks if the type is 'in' the list of allowed types.
	  """
      try:
         missing_data = ""

         if not (hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles is not None):
            missing_data += "Dynamic Obstacles "
            LOG_DEBUG(f"Missing dynamic_obstacles: {missing_data}")
            return False

         for i, obs in enumerate(data.dynamic_obstacles):
            if not hasattr(obs, 'prediction') or obs.prediction is None:
               missing_data += f"Obstacle {i} has no prediction "
               continue

            prediction_type = getattr(obs.prediction, 'type', None)

            # **THE FIX IS HERE**: Check if the type is 'not in' the list of valid types.
            if prediction_type not in [PredictionType.GAUSSIAN, PredictionType.MULTIMODAL]:
               missing_data += f"Obstacle {i} has wrong prediction type ({prediction_type}) "

         is_ready = len(missing_data) < 1
         if not is_ready:
            LOG_DEBUG(f"Missing data in Scenario Constraints: {missing_data}")

         return is_ready

      except Exception as e:
         # Adding the exception to the log message for better debugging
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