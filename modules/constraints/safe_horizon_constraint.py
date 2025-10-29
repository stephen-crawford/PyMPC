"""
Safe Horizon Constraint module for scenario-based MPC with support tracking.
"""
import time
import numpy as np
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planner_modules.src.constraints.scenario_utils.scenario_module import SafeHorizonModule, ScenarioSolver
from planning.src.types import Data, PredictionType
from utils.utils import LOG_DEBUG, LOG_WARN


class SafeHorizonConstraint(BaseConstraint):
    """
    Safe Horizon Constraint implementing scenario-based safe horizon constraints with support tracking.
    
    This class provides:
    - Scenario sampling and optimization
    - Collision constraint formulation
    - Free-space polytope construction
    - Active constraint and support tracking
    - MPC integration hooks
    """
    
    def __init__(self, solver, settings=None):
        super().__init__(solver, settings)
        self.name = "safe_horizon_constraint"
        
        LOG_DEBUG("Initializing Safe Horizon Constraint")
        
        # Initialize parameters from config
        self._initialize_parameters()
        
        # Initialize scenario modules
        self._initialize_scenario_modules()
        
        # Storage for constraint coefficients (pre-allocate for horizon + 1 to handle final stage)
        self._constraint_coefficients = self._initialize_constraint_storage()
        
        # State tracking
        self.best_solver = None
        self.optimization_time = 0.0
        self.feasible_solutions = 0
        
        LOG_DEBUG("Safe Horizon Constraint successfully initialized")
    
    def _initialize_parameters(self):
        """Initialize constraint parameters from configuration."""
        # Core scenario optimization parameters
        self.epsilon_p = self.get_config_value("safe_horizon.epsilon_p", 0.1)
        self.beta = self.get_config_value("safe_horizon.beta", 0.01)
        self.n_bar = self.get_config_value("safe_horizon.n_bar", 10)
        self.num_removal = self.get_config_value("safe_horizon.num_removal", 0)
        
        # Robot and environment parameters
        self.robot_radius = self.get_config_value("robot.radius", 0.5)
        self.horizon_length = self.solver.horizon
        
        # Constraint parameters
        self.num_discs = self.get_config_value("num_discs", 1)
        self.max_constraints_per_disc = self.get_config_value("safe_horizon.max_constraints_per_disc", 24)
        self.use_slack = self.get_config_value("safe_horizon.use_slack", True)
        
        # Sampling parameters
        self.num_scenarios = self.get_config_value("safe_horizon.num_scenarios", 100)
        self.enable_outlier_removal = self.get_config_value("safe_horizon.enable_outlier_removal", True)
        
        # Parallel processing
        self.parallel_solvers = self.get_config_value("safe_horizon.parallel_solvers", 4)
        
        # Timing
        self.planning_time = 1.0 / self.get_config_value("control_frequency", 10.0)
        
        LOG_DEBUG(f"Parameters: epsilon_p={self.epsilon_p}, beta={self.beta}, n_bar={self.n_bar}")
        LOG_DEBUG(f"Robot radius={self.robot_radius}, horizon={self.horizon_length}")
    
    def _initialize_scenario_modules(self):
        """Initialize scenario solver modules."""
        self.scenario_solvers = []
        
        # Create configuration for scenario modules
        scenario_config = {
            "epsilon_p": self.epsilon_p,
            "beta": self.beta,
            "n_bar": self.n_bar,
            "num_removal": self.num_removal,
            "robot_radius": self.robot_radius,
            "horizon_length": self.horizon_length,
            "max_constraints_per_disc": self.max_constraints_per_disc,
            "num_discs": self.num_discs,
            "num_scenarios": self.num_scenarios,
            "enable_outlier_removal": self.enable_outlier_removal
        }
        
        # Initialize parallel scenario solvers
        for i in range(self.parallel_solvers):
            solver_wrapper = ScenarioSolver(i, self.solver)
            solver_wrapper.scenario_module = SafeHorizonModule(self.solver, scenario_config)
            self.scenario_solvers.append(solver_wrapper)
        
        LOG_DEBUG(f"Initialized {len(self.scenario_solvers)} scenario solvers")
    
    def _initialize_constraint_storage(self):
        """Initialize storage for constraint coefficients."""
        return {
            'a1': np.zeros((self.num_discs, self.horizon_length, self.max_constraints_per_disc)),
            'a2': np.zeros((self.num_discs, self.horizon_length, self.max_constraints_per_disc)),
            'b': np.full((self.num_discs, self.horizon_length, self.max_constraints_per_disc), 100.0)
        }
    
    def compute_sample_size(self) -> int:
        """
        Compute required sample size using scenario optimization theory.
        
        Returns:
            Required sample size for given confidence and violation probability
        """
        from planner_modules.src.constraints.scenario_utils.math_utils import compute_sample_size
        return compute_sample_size(self.epsilon_p, self.beta, self.n_bar)
    
    def sample_scenarios(self, obstacles: List, horizon_length: int, timestep: float) -> List:
        """
        Sample scenarios from obstacle predictions.
        
        Args:
            obstacles: List of dynamic obstacles
            horizon_length: MPC horizon length
            timestep: Time step size
            
        Returns:
            List of sampled scenarios
        """
        if not self.scenario_solvers:
            return []
        
        # Use the first solver's sampler
        sampler = self.scenario_solvers[0].scenario_module.sampler
        return sampler.sample_scenarios(obstacles, horizon_length, timestep)
    
    def formulate_collision_constraint(self, robot_pos: np.ndarray, obstacle_pos: np.ndarray,
                                     robot_radius: float, obstacle_radius: float):
        """
        Formulate collision constraint between robot and obstacle.
        
        Args:
            robot_pos: Robot position [x, y]
            obstacle_pos: Obstacle position [x, y]
            robot_radius: Robot radius
            obstacle_radius: Obstacle radius
            
        Returns:
            Linearized collision constraint
        """
        from planner_modules.src.constraints.scenario_utils.math_utils import linearize_collision_constraint
        return linearize_collision_constraint(robot_pos, obstacle_pos, robot_radius, obstacle_radius)
    
    def construct_free_space_polytope(self, scenarios: List, robot_radius: float):
        """
        Construct free-space polytope from scenario constraints.
        
        Args:
            scenarios: List of scenario constraints
            robot_radius: Robot radius for safety margin
            
        Returns:
            Polytope representing free space
        """
        from planner_modules.src.constraints.scenario_utils.math_utils import construct_free_space_polytope
        return construct_free_space_polytope(scenarios, robot_radius)
    
    def track_active_constraints(self, constraints: List, disc_id: int, step: int):
        """
        Track active constraints and support.
        
        Args:
            constraints: List of active constraints
            disc_id: Disc identifier
            step: Time step
        """
        if not self.scenario_solvers:
            return
        
        # Track constraints in the best solver
        if self.best_solver:
            self.best_solver.scenario_module._track_active_constraints(constraints, disc_id, step)
    
    def check_support_limits(self) -> bool:
        """
        Check if support limits are exceeded.
        
        Returns:
            True if support is within limits
        """
        if not self.best_solver:
            return True
        
        return self.best_solver.scenario_module.check_support_limits()
    
    def prepare_iteration(self, _x_init: np.ndarray, perception_data: Data):
        """
        Prepare for MPC iteration with initial state and perception data.
        
        Args:
            x_init: Initial state vector
            perception_data: Perception data containing obstacles
        """
        LOG_DEBUG("Preparing Safe Horizon iteration")
        
        # Update scenario modules with new data
        for solver_wrapper in self.scenario_solvers:
            solver_wrapper.scenario_module.update(perception_data)
        
        # Validate parameters
        if not self.validate_parameters():
            LOG_WARN("Parameter validation failed")
            return
        
        LOG_DEBUG("Safe Horizon iteration prepared")
    
    def post_solve(self, x_solution: np.ndarray, _u_solution: np.ndarray):
        """
        Post-processing after MPC solve.
        
        Args:
            x_solution: State solution trajectory
            u_solution: Control solution trajectory
        """
        LOG_DEBUG("Post-processing Safe Horizon solution")
        
        # Validate collision probability using Monte Carlo
        if self.best_solver and hasattr(self.best_solver.scenario_module, 'validator'):
            validator = self.best_solver.scenario_module.validator
            
            # Extract robot trajectory
            robot_trajectory = []
            for i in range(min(len(x_solution), self.horizon_length)):
                if len(x_solution[i]) >= 2:
                    robot_trajectory.append(x_solution[i][:2])  # x, y positions
            
            # Get obstacles from the solver's data
            obstacles = getattr(self.best_solver.scenario_module, 'scenarios', [])
            
            if robot_trajectory and obstacles:
                is_safe, probability = validator.validate_collision_probability(
                    robot_trajectory, obstacles, self.robot_radius, self.epsilon_p
                )
                
                if not is_safe:
                    LOG_WARN(f"Solution violates safety constraint: P(collision)={probability:.4f}")
        
        LOG_DEBUG("Safe Horizon post-processing completed")
    
    def optimize(self, _state, data):
        """
        Optimize scenario constraints using parallel solvers.
        
        Args:
            state: Current state
            data: Perception data
            
        Returns:
            1 if successful, -1 if failed
        """
        LOG_DEBUG("Starting Safe Horizon optimization")
        start_time = time.time()
        
        # Run parallel optimization
        with ThreadPoolExecutor(max_workers=len(self.scenario_solvers)) as executor:
            futures = [
                executor.submit(self._run_optimize_worker, solver_wrapper, data, start_time)
                for solver_wrapper in self.scenario_solvers
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
            LOG_WARN("No scenario solver found a feasible solution")
            return -1
        
        # Store best solver and extract constraints
        self.best_solver = best_solver
        self._extract_constraint_coefficients()
        
        self.optimization_time = time.time() - start_time
        self.feasible_solutions += 1
        
        LOG_DEBUG(f"Safe Horizon optimization completed in {self.optimization_time:.3f}s")
        return 1
    
    def _run_optimize_worker(self, solver_wrapper: ScenarioSolver, data: Data, start_time: float):
        """Helper for parallel execution."""
        try:
            used_time = time.time() - start_time
            solver_wrapper.solver_timeout = max(0.1, self.planning_time - used_time - 0.008)
            
            # Copy solver for this worker
            solver_wrapper.solver = self.solver.copy()
            
            # Run optimization
            exit_code = solver_wrapper.scenario_module.optimize(data)
            
            # Get objective value
            objective_value = float('inf')
            if hasattr(solver_wrapper.solver, 'solution') and solver_wrapper.solver.solution:
                objective_value = solver_wrapper.solver.solution.optval
            
            return exit_code, objective_value, solver_wrapper
            
        except Exception as e:
            LOG_WARN(f"Error in scenario worker {solver_wrapper.solver_id}: {e}")
            return -1, float('inf'), solver_wrapper
    
    def _extract_constraint_coefficients(self):
        """Extract constraint coefficients from best solver."""
        if not self.best_solver:
            return
        
        for disc_id, disc_manager in enumerate(self.best_solver.scenario_module.disc_manager):
            for step in range(self.horizon_length):
                if step < len(disc_manager.polytopes):
                    polytope = disc_manager.polytopes[step]
                    num_constraints = min(len(polytope.halfspaces), self.max_constraints_per_disc)
                    
                    for i in range(num_constraints):
                        if i < len(polytope.halfspaces):
                            halfspace = polytope.halfspaces[i]
                            A = halfspace.A
                            b = halfspace.b
                            
                            if A.shape[0] > 0 and A.shape[1] >= 2:
                                self._constraint_coefficients['a1'][disc_id][step][i] = A[0, 0]
                                self._constraint_coefficients['a2'][disc_id][step][i] = A[0, 1]
                                self._constraint_coefficients['b'][disc_id][step][i] = b[0]
    
    def define_parameters(self, params):
        """Define symbolic parameters for the constraints."""
        # Define parameters for horizon + 1 stages (0 to horizon inclusive)
        for disc_id in range(self.num_discs):
            for step in range(self.horizon_length + 1):
                for i in range(self.max_constraints_per_disc):
                    base_name = f"disc_{disc_id}_safe_horizon_constraint_{i}_step_{step}"
                    params.add(f"{base_name}_a1")
                    params.add(f"{base_name}_a2")
                    params.add(f"{base_name}_b")
    
    def get_constraints(self, symbolic_state, params, stage_idx):
        """Build symbolic constraint expressions."""
        if stage_idx == 0:
            return []
        
        constraints = []
        pos_x = symbolic_state.get("x")
        pos_y = symbolic_state.get("y")
        slack = symbolic_state.get("slack") if self.use_slack else 0.0
        
        for disc_id in range(self.num_discs):
            for i in range(self.max_constraints_per_disc):
                base_name = f"disc_{disc_id}_safe_horizon_constraint_{i}_step_{stage_idx}"
                a1 = params.get(f"{base_name}_a1")
                a2 = params.get(f"{base_name}_a2")
                b = params.get(f"{base_name}_b")
                
                # Constraint form: a1*x + a2*y <= b + slack
                constraint_expr = a1 * pos_x + a2 * pos_y - b - slack
                constraints.append(constraint_expr)
        
        return constraints
    
    def get_lower_bound(self):
        """Get lower bounds for constraints."""
        return [-np.inf] * self.num_discs * self.max_constraints_per_disc
    
    def get_upper_bound(self):
        """Get upper bounds for constraints."""
        return [0.0] * self.num_discs * self.max_constraints_per_disc
    
    def set_parameters(self, parameter_manager, data, step):
        """Set parameter values for the current step."""
        # Set parameters for all stages including final stage (horizon)
        # Use step-1 for constraint coefficients (constraints computed for next step)
        # For step 0, use dummy values since no constraints computed yet
        # For step >= horizon_length, use last computed constraints
        for disc_id in range(self.num_discs):
            for i in range(self.max_constraints_per_disc):
                base_name = f"disc_{disc_id}_safe_horizon_constraint_{i}_step_{step}"
                
                if step == 0:
                    # Use non-zero dummy values to avoid constant constraints
                    # These will be used when stage_idx=1 is evaluated
                    a1_val = 1.0
                    a2_val = 0.0
                    b_val = 1000.0  # Large value to make constraint non-binding
                elif step <= self.horizon_length:
                    # Use computed constraints (step-1 because constraints are for next step)
                    coeff_step = min(step - 1, self.horizon_length - 1)
                    a1_val = self._constraint_coefficients['a1'][disc_id][coeff_step][i]
                    a2_val = self._constraint_coefficients['a2'][disc_id][coeff_step][i]
                    b_val = self._constraint_coefficients['b'][disc_id][coeff_step][i]
                else:
                    # Use last computed constraints for steps beyond horizon
                    a1_val = self._constraint_coefficients['a1'][disc_id][self.horizon_length - 1][i]
                    a2_val = self._constraint_coefficients['a2'][disc_id][self.horizon_length - 1][i]
                    b_val = self._constraint_coefficients['b'][disc_id][self.horizon_length - 1][i]
                
                parameter_manager.set_parameter(f"{base_name}_a1", a1_val)
                parameter_manager.set_parameter(f"{base_name}_a2", a2_val)
                parameter_manager.set_parameter(f"{base_name}_b", b_val)
    
    def get_constraint_info(self) -> Dict:
        """Get information about current constraints."""
        if not self.best_solver:
            return {"status": "No solver available"}
        
        return self.best_solver.scenario_module.get_constraint_info()
    
    def validate_parameters(self) -> bool:
        """Validate constraint parameters."""
        if not self.best_solver:
            return True
        
        return self.best_solver.scenario_module.validate_parameters()
    
    def is_data_ready(self, data: Data, missing_data: str = "") -> bool:
        """Check if required data is available."""
        if not self.scenario_solvers:
            return False
        
        # Check using the first solver
        return self.scenario_solvers[0].scenario_module.is_data_ready(data)
    
    def on_data_received(self, data: Data, data_name: str = ""):
        """Process incoming data."""
        try:
            if not (hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles):
                return
            
            # Validate prediction types
            for obs in data.dynamic_obstacles:
                if not hasattr(obs, 'prediction') or obs.prediction is None:
                    continue
                
                if (hasattr(obs.prediction, 'type') and
                    obs.prediction.type == PredictionType.DETERMINISTIC):
                    LOG_WARN("Using deterministic prediction with Safe Horizon Constraints")
                    LOG_WARN("Set process_noise to non-zero value to add uncertainty")
                    return
            
            # Process obstacle data
            def worker(solver_wrapper):
                try:
                    sampler = solver_wrapper.scenario_module.get_sampler()
                    if sampler and hasattr(sampler, 'integrate_and_translate_to_mean_and_variance'):
                        timestep = getattr(self.solver, 'timestep', 0.1)
                        sampler.integrate_and_translate_to_mean_and_variance(
                            data.dynamic_obstacles, timestep
                        )
                except Exception as e:
                    LOG_WARN(f"Error processing obstacle data for solver {solver_wrapper.solver_id}: {e}")
            
            # Parallelize data processing
            max_workers = min(4, len(self.scenario_solvers))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(worker, self.scenario_solvers)
        
        except Exception as e:
            LOG_WARN(f"Error in on_data_received: {e}")
    
    def reset(self):
        """Reset constraint state."""
        try:
            super().reset()
            
            # Reset constraint-specific state
            self.best_solver = None
            self.optimization_time = 0
            self.feasible_solutions = 0
            
            # Reset constraint coefficients
            self._constraint_coefficients = self._initialize_constraint_storage()
            
            # Reset all scenario solvers
            for solver_wrapper in self.scenario_solvers:
                solver_wrapper.exit_code = 0
                if hasattr(solver_wrapper.scenario_module, 'reset'):
                    solver_wrapper.scenario_module.reset()
            
            LOG_DEBUG("Safe Horizon Constraint reset")
            
        except Exception as e:
            LOG_WARN(f"Error in reset: {e}")
