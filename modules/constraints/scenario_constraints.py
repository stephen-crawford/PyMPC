import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
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

      # Load scenario configuration
      self.enable_safe_horizon = self.get_config_value("scenario_constraints.enable_safe_horizon", True)
      self.num_discs = self.get_config_value("num_discs", 1)
      self.max_constraints_per_disc = self.get_config_value("scenario_constraints.max_constraints", 24)
      self.use_slack = self.get_config_value("scenario_constraints.use_slack", True)

      # Initialize scenario solvers
      self.scenario_solvers = []
      self.best_solver = None
      parallel_solvers = self.get_config_value("scenario_constraints.parallel_solvers", 4)
      for i in range(parallel_solvers):
         self.scenario_solvers.append(ScenarioSolver(i, solver))

      # **MODIFICATION 1: Add storage for constraint coefficients**
      self._a1 = np.zeros((self.num_discs, self.solver.horizon, self.max_constraints_per_disc))
      self._a2 = np.zeros((self.num_discs, self.solver.horizon, self.max_constraints_per_disc))
      self._b = np.full((self.num_discs, self.solver.horizon, self.max_constraints_per_disc), 100.0)

      LOG_DEBUG("Scenario Constraints successfully initialized")

   def get_config_value(self, key, default=None):
      res = self.config.get(key, self.config.get(f"{self.name}.{key}", default))
      if res is None:
         res = get_config_dotted(self.config, key)
      return res

   def optimize(self, state, data):
      """
	  **MODIFICATION 2: This method now COMPUTES and STORES the constraints.**
	  It no longer replaces the main solver.
	  """
      LOG_DEBUG("ScenarioConstraints.optimize: Computing optimal constraints")
      start_time = time.time()

      # Run parallel optimization to find the best set of scenario constraints
      with ThreadPoolExecutor(max_workers=len(self.scenario_solvers)) as executor:
         futures = [
            executor.submit(self.run_optimize_worker, s, self.solver, data, start_time)
            for s in self.scenario_solvers
         ]
         results = [f.result() for f in futures]

      # Select best solver based on cost
      best_solver = None
      lowest_cost = float('inf')
      for exit_code, cost, solver_wrapper in results:
         if exit_code == 1 and cost < lowest_cost:
            lowest_cost = cost
            best_solver = solver_wrapper

      if best_solver is None:
         LOG_WARN("No scenario solver found a feasible solution for constraints.")
         return -1  # Signal failure

      # Store the best solver for visualization/debugging
      self.best_solver = best_solver

      # Extract and store the computed constraint coefficients from the best solver
      for disc_id, disc_manager in enumerate(self.best_solver.scenario_module.disc_manager):
         for step in range(self.solver.horizon):
            # The SafeHorizon module stores constraints in a `polytopes` list
            polytope = disc_manager.polytopes[step]
            num_found_constraints = min(len(polytope.polygon_out), self.max_constraints_per_disc)

            for i in range(num_found_constraints):
               constraint = polytope.polygon_out[i]
               self._a1[disc_id][step][i] = constraint.a1
               self._a2[disc_id][step][i] = constraint.a2
               self._b[disc_id][step][i] = constraint.b

      return 1  # Signal success

   def run_optimize_worker(self, scenario_solver, main_solver, data, start_time):
      """Helper for parallel execution. Renamed from run_optimize."""
      try:
         used_time = time.time() - start_time
         scenario_solver.solver_timeout = max(0.1, self.planning_time - used_time - 0.008)

         # Each worker gets a fresh copy of the solver to work with
         scenario_solver.solver = main_solver.copy()

         # Set parameters and run the internal optimization
         scenario_solver.scenario_module.update(data)
         exit_code = scenario_solver.scenario_module.optimize(data)

         objective_value = float('inf')
         if hasattr(scenario_solver.solver, 'solution') and scenario_solver.solver.solution:
            objective_value = scenario_solver.solver.solution.optval

         return exit_code, objective_value, scenario_solver
      except Exception as e:
         LOG_WARN(f"Error in scenario worker for solver {scenario_solver.solver_id}: {e}")
         return -1, float('inf'), scenario_solver

   # **MODIFICATION 3: Implement define_parameters, get_constraints, and bounds**
   def define_parameters(self, params):
      """Define symbolic parameters for the constraints in the main solver."""
      # Define parameters for horizon + 1 stages (0 to horizon inclusive)
      for disc_id in range(self.num_discs):
         for step in range(self.solver.horizon + 1):
            for i in range(self.max_constraints_per_disc):
               base_name = f"disc_{disc_id}_scen_constraint_{i}_step_{step}"
               params.add(f"{base_name}_a1")
               params.add(f"{base_name}_a2")
               params.add(f"{base_name}_b")

   def get_constraints(self, symbolic_state, params, stage_idx):
      """Build the symbolic constraint expressions for the main solver."""
      if stage_idx == 0:
         return []

      constraints = []
      pos_x = symbolic_state.get("x")
      pos_y = symbolic_state.get("y")

      slack = symbolic_state.get("slack") if self.use_slack else 0.0

      for disc_id in range(self.num_discs):
         for i in range(self.max_constraints_per_disc):
            base_name = f"disc_{disc_id}_scen_constraint_{i}_step_{stage_idx}"
            a1 = params.get(f"{base_name}_a1")
            a2 = params.get(f"{base_name}_a2")
            b = params.get(f"{base_name}_b")

            # Constraint form: a1*x + a2*y <= b + slack
            constraint_expr = a1 * pos_x + a2 * pos_y - b - slack
            constraints.append(constraint_expr)

      return constraints

   def get_lower_bound(self):
      # All constraints are <= 0
      return [-np.inf] * self.num_discs * self.max_constraints_per_disc

   def get_upper_bound(self):
      # All constraints are <= 0
      return [0.0] * self.num_discs * self.max_constraints_per_disc

   # **MODIFICATION 4: Flesh out set_parameters**
   def set_parameters(self, parameter_manager, data, step):
      """Populate the symbolic parameters with the computed numeric values."""
      # Set parameters for all stages including final stage (horizon)
      # Use step-1 for constraint coefficients (constraints computed for next step)
      # For step 0, use dummy values since no constraints computed yet
      # For step >= horizon, use last computed constraints
      for disc_id in range(self.num_discs):
         for i in range(self.max_constraints_per_disc):
            base_name = f"disc_{disc_id}_scen_constraint_{i}_step_{step}"
            
            if step == 0:
               # Use dummy values for initial step
               a1_val = 0.0
               a2_val = 0.0
               b_val = 100.0
            elif step <= self.solver.horizon:
               # Use computed constraints (step-1 because constraints are for next step)
               coeff_step = min(step - 1, self.solver.horizon - 1)
               a1_val = self._a1[disc_id][coeff_step][i]
               a2_val = self._a2[disc_id][coeff_step][i]
               b_val = self._b[disc_id][coeff_step][i]
            else:
               # Use last computed constraints for steps beyond horizon
               a1_val = self._a1[disc_id][self.solver.horizon - 1][i]
               a2_val = self._a2[disc_id][self.solver.horizon - 1][i]
               b_val = self._b[disc_id][self.solver.horizon - 1][i]

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