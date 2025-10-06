import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import casadi as cd
from planner_modules.src.constraints.base_constraint import BaseConstraint
from planner_modules.src.constraints.scenario_utils.scenario_module import ScenarioSolver
from planning.src.types import PredictionType
from utils.utils import read_config_file, LOG_INFO, LOG_DEBUG, get_config_dotted, LOG_WARN


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

   def run_optimize_worker(self, scenario_solver, main_solver, data, start_time):
      """Helper for parallel execution. Finds a feasible trajectory subject to all scenarios."""
      try:
         used_time = time.time() - start_time
         scenario_solver.solver_timeout = max(0.1, self.planning_time - used_time - 0.008)
         scenario_solver.solver = main_solver.copy()

         # This is the core computation step within the worker
         scenario_solver.scenario_module.update(data)  # Pre-computes distances, etc.
         exit_code = scenario_solver.scenario_module.optimize(data)  # Finds the separating hyperplanes

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
	  **FIXED**: This now correctly builds the symbolic constraint expressions
	  using the populated parameters, resolving the TypeError.
	  """
      # No constraints at the initial state (stage 0)
      if stage_idx == 0:
         return []

      constraints = []
      LOG_DEBUG(f"Going to try to get constraints for symbolic state {symbolic_state} and stage {stage_idx}")
      pos_x = symbolic_state.get("x")
      pos_y = symbolic_state.get("y")

      try:
         slack = symbolic_state.get("slack") if self.use_slack else 0.0
      except:
         slack = 0.0

      for disc_id in range(self.num_discs):
         LOG_DEBUG(f"Getting constraints for disc {disc_id}")
         for i in range(self.max_constraints_per_disc):
            LOG_DEBUG(f"Getting constraint {i}")
            # Use stage_idx directly as it corresponds to the solver's current step
            base_name = f"disc_{disc_id}_scen_constraint_{i}_step_{stage_idx}"
            LOG_DEBUG(f"Base name {base_name}")
            a1 = params.get(f"{base_name}_a1")
            a2 = params.get(f"{base_name}_a2")
            b = params.get(f"{base_name}_b")

            # Skip dummy/invalid constraints
            if abs(a1) < 1e-6 and abs(a2) < 1e-6:
               LOG_DEBUG(f"Skipping dummy constraint for disc {disc_id}")
               continue

            # Constraint form: a1*x + a2*y - b - slack <= 0
            constraint_expr = a1 * pos_x + a2 * pos_y - b - slack
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
            # Get values from storage (computed in the optimize method)
            a1_val = self._a1[disc_id][step - 1][i]
            a2_val = self._a2[disc_id][step - 1][i]
            b_val = self._b[disc_id][step - 1][i]

            parameter_manager.set_parameter(f"{base_name}_a1", a1_val)
            parameter_manager.set_parameter(f"{base_name}_a2", a2_val)
            parameter_manager.set_parameter(f"{base_name}_b", b_val)

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

             # **FIXED**: Allow GAUSSIAN prediction type
             prediction_type = getattr(obs.prediction, 'type', None)
             if prediction_type != PredictionType.GAUSSIAN and prediction_type != PredictionType.MULTIMODAL:
                 missing_data += f"Obstacle Prediction (type must be GAUSSIAN or MULTIMODAL) for obstacle {i}"


          # Check scenario solver readiness
          if self.scenario_solvers:
             # It's sufficient to check one, as they all share the same data readiness logic
             if not self.scenario_solvers[0].scenario_module.is_data_ready(data):
                 missing_data += "Missing data required for Scenario Solvers"

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