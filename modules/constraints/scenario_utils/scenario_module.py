"""
Main scenario module for safe horizon constraints.
"""
from tkinter import NO
import numpy as np
from typing import List, Dict
from planning.types import Data, Scenario, ScenarioStatus, ScenarioSolveStatus, SupportSubsample
from modules.constraints.scenario_utils.math_utils import (
    Polytope, ScenarioConstraint, compute_sample_size, linearize_collision_constraint,
    construct_free_space_polytope, validate_polytope_feasibility
)
from modules.constraints.scenario_utils.sampler import ScenarioSampler, MonteCarloValidator
from utils.utils import LOG_DEBUG, LOG_WARN, LOG_INFO


class DiscManager:
    """Manages constraints for a single disc of the robot."""
    
    def __init__(self, disc_id: int, robot_radius: float, max_constraints: int):
        self.disc_id = disc_id
        self.robot_radius = robot_radius
        self.max_constraints = max_constraints
        self.polytopes = []  # One polytope per time step
        self.active_constraints = []
        
    def add_polytope(self, polytope: Polytope, time_step: int):
        """Add a polytope for a specific time step."""
        while len(self.polytopes) <= time_step:
            self.polytopes.append(Polytope([]))
        
        self.polytopes[time_step] = polytope
        
    def get_constraints_for_step(self, time_step: int) -> List[ScenarioConstraint]:
        """Get constraints for a specific time step.
        
        Reference: scenario_module - returns constraints from polytope halfspaces
        The polytope represents the feasible free space, and constraints are extracted
        from the halfspaces that define this polytope.
        
        CRITICAL: Only return the most important constraints (up to max_constraints)
        to avoid over-constraining the optimization problem.
        """
        if time_step >= len(self.polytopes):
            return []
        
        polytope = self.polytopes[time_step]
        constraints = []
        
        # Reference: scenario_module - limit constraints to avoid over-constraining
        # The polytope may have many halfspaces, but we only need the most critical ones
        # Typically, the polytope optimization already selects the most important constraints
        num_halfspaces = len(polytope.halfspaces)
        num_to_use = min(num_halfspaces, self.max_constraints)
        
        for i in range(num_to_use):
            halfspace = polytope.halfspaces[i]
            
            # Convert halfspace to scenario constraint
            A = halfspace.A
            b = halfspace.b
            
            if A.shape[0] > 0 and A.shape[1] >= 2:
                # CRITICAL: Retrieve obstacle position AND obstacle_idx from polytope if stored
                # Reference: C++ mpc_planner - obstacle positions are needed for symbolic constraint computation
                # The constraint normal (a1, a2) will be recomputed symbolically using predicted robot position
                obstacle_pos = None
                obstacle_radius = None
                obstacle_idx = 0  # Default to 0 if not found
                if hasattr(polytope, '_obstacle_positions') and i in polytope._obstacle_positions:
                    obs_info = polytope._obstacle_positions[i]
                    obstacle_pos = obs_info['position']
                    obstacle_radius = obs_info['radius']
                    obstacle_idx = obs_info.get('obstacle_idx', 0)  # Get obstacle_idx if stored
                    LOG_DEBUG(f"Retrieved obstacle info for constraint {i}: pos=({obstacle_pos[0]:.3f}, {obstacle_pos[1]:.3f}), radius={obstacle_radius:.3f}, obstacle_idx={obstacle_idx}")
                else:
                    LOG_WARN(f"Warning: No obstacle position stored for constraint {i} in polytope at time_step {time_step} "
                            f"(has _obstacle_positions: {hasattr(polytope, '_obstacle_positions')}, "
                            f"keys: {list(polytope._obstacle_positions.keys()) if hasattr(polytope, '_obstacle_positions') else 'N/A'})")
                
                constraint = ScenarioConstraint(
                    a1=A[0, 0], a2=A[0, 1], b=b[0],
                    scenario_idx=i, obstacle_idx=obstacle_idx, time_step=time_step,  # Use retrieved obstacle_idx
                    obstacle_pos=obstacle_pos,  # Store obstacle position for symbolic computation
                    obstacle_radius=obstacle_radius
                )
                constraints.append(constraint)
        
        return constraints


class SafeHorizonModule:
    """Main module for safe horizon scenario-based constraints."""
    
    def __init__(self, solver, config: Dict):
        self.solver = solver
        self.config = config
        
        # Parameters
        self.epsilon_p = config.get("epsilon_p", 0.1)  # Constraint violation probability
        self.beta = config.get("beta", 0.01)  # Confidence level
        self.n_bar = config.get("n_bar", 10)  # Support dimension
        
        # Get halfspace offset from config (matching linearized_constraints.py)
        self.halfspace_offset = float(config.get("linearized_constraints", {}).get("halfspace_offset", 0.0))
        if self.halfspace_offset is None:
            self.halfspace_offset = 0.0
        self.num_removal = config.get("num_removal", 0)  # Number of scenarios to remove
        self.robot_radius = config.get("robot_radius", 0.5)
        self.horizon_length = config.get("horizon_length", 10)
        # CRITICAL: Must have enough constraints to cover all obstacles
        # Reference: C++ scenario_module - uses enough constraints to ensure obstacle diversity
        self.max_constraints_per_disc = config.get("max_constraints_per_disc", 5)  # Increased to cover all obstacles
        self.num_discs = config.get("num_discs", 1)
        self.num_scenarios = config.get("num_scenarios", 100)  # Number of scenarios to sample (for diagnostics)
        
        # Components
        self.sampler = ScenarioSampler(
            num_scenarios=self.num_scenarios,
            enable_outlier_removal=config.get("enable_outlier_removal", True)
        )
        self.validator = MonteCarloValidator()
        
        # State
        self.disc_manager = []
        for i in range(self.num_discs):
            self.disc_manager.append(DiscManager(i, self.robot_radius, self.max_constraints_per_disc))
        
        self.scenarios = []
        self.support_subsample = SupportSubsample()
        self.status = ScenarioStatus.NONE
        self.solve_status = ScenarioSolveStatus.SUCCESS

        self.support_estimator = GreedySupportEstimator(self.n_bar, tolerance=config.get("support_estimator_tolerance", 1e-4))
        self.current_risk_bound = None
        LOG_DEBUG("SafeHorizonModule initialized")
    
    def update(self, data: Data):
        """Update module with new data.
        
        Reference: scenario_module - samples scenarios from obstacle predictions.
        This is called before optimize() to populate scenarios.
        """
        try:
            # Check if we have dynamic obstacles
            if not (hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles):
                LOG_WARN("No dynamic obstacles in data")
                self.scenarios = []
                return
            
            # Sample scenarios from obstacle predictions
            # Reference: scenario_module - samples full trajectories for each scenario
            self.scenarios = self.sampler.sample_scenarios(
                data.dynamic_obstacles, self.horizon_length, 
                getattr(self.solver, 'timestep', 0.1)
            )
            
            LOG_DEBUG(f"Updated with {len(self.scenarios)} scenarios")
            
        except Exception as e:
            LOG_WARN(f"Error updating SafeHorizonModule: {e}")
            return
    
    def optimize(self, data: Data) -> int:
        """
        Optimize scenario constraints.
        
        Reference: C++ mpc_planner - scenario optimization happens after warmstart is initialized
        The warmstart trajectory is used as reference for constraint linearization.
        This matches linearized_constraints.py pattern where update() gets reference trajectory.
        
        Returns:
            1 if successful, -1 if failed
        """
        try:
            LOG_DEBUG("Starting scenario optimization")

            self.support_estimator.reset()
            
            # CRITICAL: Ensure warmstart is initialized before optimization
            # Reference: C++ mpc_planner - warmstart must be initialized before constraint linearization
            # The warmstart provides the reference trajectory for linearizing constraints
            # This matches linearized_constraints.py where update() is called after initialize_rollout()
            if hasattr(self, 'solver') and self.solver is not None:
                if not (hasattr(self.solver, 'warmstart_intiailized') and self.solver.warmstart_intiailized):
                    LOG_WARN("SafeHorizonModule.optimize: Warmstart not initialized! This may cause reference position issues.")
                    # Try to initialize warmstart if state is available
                    if data is not None and hasattr(data, 'state') and data.state is not None:
                        try:
                            LOG_DEBUG("Attempting to initialize warmstart from data.state")
                            self.solver.initialize_rollout(data.state, data)
                        except Exception as e:
                            LOG_DEBUG(f"Could not initialize warmstart: {e}")
            
            # Update with current data (samples scenarios)
            self.update(data)
            
            if not self.scenarios:
                LOG_WARN("No scenarios available for optimization")
                return -1
            
            # Compute sample size
            sample_size = self.compute_sample_size()
            LOG_DEBUG(f"Computed sample size: {sample_size}")
            
            # Get reference trajectory states for linearization (matching linearized_constraints.py)
            # Reference: linearized_constraints.py uses reference trajectory for linearization
            # CRITICAL: If reference trajectory is empty, create states from warmstart values directly
            # This matches the C++ pattern: _solver->getEgoPrediction(k, "x") returns warmstart if no solution
            ref_states = []
            if hasattr(self.solver, 'get_reference_trajectory'):
                try:
                    ref_traj = self.solver.get_reference_trajectory()
                    if ref_traj is not None and hasattr(ref_traj, 'get_states'):
                        ref_states = ref_traj.get_states()
                except Exception as e:
                    LOG_DEBUG(f"Could not get reference trajectory for optimization: {e}")
            
            # CRITICAL FIX: If reference trajectory is empty, use warmstart values directly
            # This matches the C++ pattern: _solver->getEgoPrediction(k, "x") returns warmstart if no solution
            # Warmstart values already contain the predicted trajectory from previous solve or initialization
            # Reference: linearized_constraints.py lines 100-148 - uses warmstart values directly
            has_warmstart = (hasattr(self.solver, 'warmstart_values') and 
                           self.solver.warmstart_values is not None and
                           isinstance(self.solver.warmstart_values, dict) and
                           'x' in self.solver.warmstart_values and 
                           'y' in self.solver.warmstart_values)
            
            # PRIORITY 1: Use warmstart values directly (matching C++ getEgoPrediction pattern)
            # Warmstart values already contain the forward-propagated trajectory from previous solve
            # This is the correct C++ reference pattern: getEgoPrediction() returns warmstart
            # Reference: linearized_constraints.py lines 100-148
            if not ref_states and has_warmstart:
                try:
                    from planning.types import State
                    dynamics_model = self.solver._get_dynamics_model()
                    if dynamics_model is not None:
                        horizon_val = self.solver.horizon if self.solver.horizon is not None else 10
                        ws_vals = self.solver.warmstart_values
                        
                        # Create states from warmstart values (matching linearized_constraints.py lines 128-146)
                        if 'x' in ws_vals and 'y' in ws_vals:
                            x_vals = ws_vals['x']
                            y_vals = ws_vals['y']
                            psi_vals = ws_vals.get('psi', [0.0] * (horizon_val + 1))
                            v_vals = ws_vals.get('v', [0.0] * (horizon_val + 1))
                            
                            # Ensure arrays are the right length
                            if hasattr(x_vals, '__len__') and hasattr(y_vals, '__len__'):
                                for k in range(min(horizon_val + 1, len(x_vals), len(y_vals))):
                                    state_k = State(model_type=dynamics_model)
                                    # Extract numeric values from warmstart
                                    import casadi as cd
                                    x_val = x_vals[k]
                                    y_val = y_vals[k]
                                    if isinstance(x_val, (cd.MX, cd.SX)):
                                        try:
                                            x_val = float(cd.DM(x_val))
                                        except:
                                            x_val = 0.0
                                    else:
                                        x_val = float(x_val)
                                    if isinstance(y_val, (cd.MX, cd.SX)):
                                        try:
                                            y_val = float(cd.DM(y_val))
                                        except:
                                            y_val = 0.0
                                    else:
                                        y_val = float(y_val)
                                    
                                    state_k.set("x", x_val)
                                    state_k.set("y", y_val)
                                    if k < len(psi_vals):
                                        psi_val = psi_vals[k]
                                        if isinstance(psi_val, (cd.MX, cd.SX)):
                                            try:
                                                psi_val = float(cd.DM(psi_val))
                                            except:
                                                psi_val = 0.0
                                        else:
                                            psi_val = float(psi_val)
                                        state_k.set("psi", psi_val)
                                    if k < len(v_vals):
                                        v_val = v_vals[k]
                                        if isinstance(v_val, (cd.MX, cd.SX)):
                                            try:
                                                v_val = float(cd.DM(v_val))
                                            except:
                                                v_val = 0.0
                                        else:
                                            v_val = float(v_val)
                                        state_k.set("v", v_val)
                                    ref_states.append(state_k)
                                LOG_INFO(f"Created {len(ref_states)} reference states from warmstart values (matching linearized_constraints.py)")
                            else:
                                LOG_WARN(f"Warmstart x or y is not an array: x={type(x_vals)}, y={type(y_vals)}")

                except Exception as e:
                    LOG_WARN(f"Could not create reference states from warmstart: {e}")
                    import traceback
                    LOG_DEBUG(f"Traceback: {traceback.format_exc()}")
            
            # Get current state for step 0 (matching linearized_constraints.py)
            # Reference: linearized_constraints.py update_step() uses current state for step 0
            current_state = None
            if data is not None and hasattr(data, 'state') and data.state is not None:
                current_state = data.state
                LOG_INFO(f"optimize: Retrieved current_state from data.state: x={current_state.get('x') if hasattr(current_state, 'has') and current_state.has('x') else 'N/A'}, y={current_state.get('y') if hasattr(current_state, 'has') and current_state.has('y') else 'N/A'}")
            else:
                LOG_WARN(f"optimize: data.state is not available! data={data}, hasattr(data, 'state')={hasattr(data, 'state') if data is not None else 'N/A'}, data.state={data.state if data is not None and hasattr(data, 'state') else 'N/A'}")
            
            # Process scenarios for each disc and time step
            for disc_id in range(self.num_discs):
                for step in range(self.horizon_length):
                    # Get reference robot position for this step
                    # For step 0, use current state; for step > 0, use reference trajectory
                    # Reference: linearized_constraints.py - step 0 uses current state, future steps use reference trajectory
                    reference_robot_pos = None
                    
                    if step == 0 and current_state is not None:
                        # Use current state position for step 0 (matching linearized_constraints.py)
                        try:
                            if hasattr(current_state, 'has') and current_state.has('x') and current_state.has('y'):
                                x_val = current_state.get('x')
                                y_val = current_state.get('y')
                                
                                # CRITICAL: Extract numeric value if symbolic (CasADi variable)
                                # For linearization, we need numeric values, not symbolic
                                try:
                                    import casadi as cd
                                    if isinstance(x_val, (cd.MX, cd.SX)):
                                        # Try to get numeric value from solver's initial values or warmstart
                                        if hasattr(self, 'solver') and self.solver is not None:
                                            if hasattr(self.solver, 'warmstart_values') and 'x' in self.solver.warmstart_values:
                                                x_val = float(self.solver.warmstart_values['x'][0]) if len(self.solver.warmstart_values['x']) > 0 else 0.0
                                            else:
                                                x_val = 0.0
                                        else:
                                            x_val = 0.0
                                    else:
                                        x_val = float(x_val)
                                    
                                    if isinstance(y_val, (cd.MX, cd.SX)):
                                        if hasattr(self, 'solver') and self.solver is not None:
                                            if hasattr(self.solver, 'warmstart_values') and 'y' in self.solver.warmstart_values:
                                                y_val = float(self.solver.warmstart_values['y'][0]) if len(self.solver.warmstart_values['y']) > 0 else 0.0
                                            else:
                                                y_val = 0.0
                                        else:
                                            y_val = 0.0
                                    else:
                                        y_val = float(y_val)
                                except:
                                    # Fallback: try direct conversion
                                    try:
                                        x_val = float(x_val)
                                        y_val = float(y_val)
                                    except:
                                        x_val = 0.0
                                        y_val = 0.0
                                
                                reference_robot_pos = np.array([x_val, y_val])
                                LOG_INFO(f"Step {step}: Using current state position ({reference_robot_pos[0]:.3f}, {reference_robot_pos[1]:.3f}) for linearization")
                        except Exception as e:
                            LOG_WARN(f"Could not get current state position for step 0: {e}")
                            import traceback
                            LOG_DEBUG(f"Traceback: {traceback.format_exc()}")
                            reference_robot_pos = None
                    
                    # For future steps, use reference trajectory states (matching linearized_constraints.py)
                    # Reference: C++ mpc_planner - _solver->getEgoPrediction(k, "x") returns reference trajectory
                    # The reference trajectory is created from forward propagation if no solution is available
                    # This matches linearized_constraints.py lines 284-291 exactly
                    if reference_robot_pos is None and step > 0:
                        # CRITICAL FIX: Use forward-propagated trajectory states (created above)
                        # This ensures each step uses a different position, not the same current position
                        if step < len(ref_states):
                            try:
                                ref_state = ref_states[step]
                                if hasattr(ref_state, 'has') and ref_state.has('x') and ref_state.has('y'):
                                    x_val = ref_state.get('x')
                                    y_val = ref_state.get('y')
                                    # Extract numeric value (may be symbolic from propagation)
                                    import casadi as cd
                                    if isinstance(x_val, (cd.MX, cd.SX)):
                                        try:
                                            x_val = float(cd.DM(x_val))
                                        except:
                                            # Fallback: use warmstart if available
                                            if has_warmstart and 'x' in self.solver.warmstart_values and step < len(self.solver.warmstart_values['x']):
                                                x_val = float(self.solver.warmstart_values['x'][step])
                                            else:
                                                x_val = 0.0
                                    else:
                                        x_val = float(x_val)
                                    
                                    if isinstance(y_val, (cd.MX, cd.SX)):
                                        try:
                                            y_val = float(cd.DM(y_val))
                                        except:
                                            # Fallback: use warmstart if available
                                            if has_warmstart and 'y' in self.solver.warmstart_values and step < len(self.solver.warmstart_values['y']):
                                                y_val = float(self.solver.warmstart_values['y'][step])
                                            else:
                                                y_val = 0.0
                                    else:
                                        y_val = float(y_val)
                                    
                                    reference_robot_pos = np.array([x_val, y_val])
                                    LOG_INFO(f"Step {step}: Using forward-propagated reference trajectory position ({reference_robot_pos[0]:.3f}, {reference_robot_pos[1]:.3f}) for linearization")
                            except Exception as e:
                                LOG_DEBUG(f"Could not get reference trajectory position for step {step}: {e}")
                        else:
                            # Fallback: reference trajectory too short, use last available state
                            # This matches linearized_constraints.py lines 292-299
                            if len(ref_states) > 0:
                                try:
                                    last_state = ref_states[-1]
                                    if hasattr(last_state, 'has') and last_state.has('x') and last_state.has('y'):
                                        x_val = float(last_state.get('x'))
                                        y_val = float(last_state.get('y'))
                                        reference_robot_pos = np.array([x_val, y_val])
                                        LOG_WARN(f"Step {step}: Reference trajectory too short (len={len(ref_states)}), using LAST trajectory state: ({reference_robot_pos[0]:.3f}, {reference_robot_pos[1]:.3f})")
                                except Exception as e:
                                    LOG_DEBUG(f"Could not get last trajectory state: {e}")
                            
                            # Final fallback: use current state position
                            if reference_robot_pos is None and current_state is not None and hasattr(current_state, 'has') and current_state.has('x') and current_state.has('y'):
                                try:
                                    x_val = current_state.get('x')
                                    y_val = current_state.get('y')
                                    # Extract numeric value
                                    import casadi as cd
                                    if isinstance(x_val, (cd.MX, cd.SX)):
                                        x_val = float(cd.DM(x_val)) if hasattr(cd, 'DM') else 0.0
                                    else:
                                        x_val = float(x_val)
                                    if isinstance(y_val, (cd.MX, cd.SX)):
                                        y_val = float(cd.DM(y_val)) if hasattr(cd, 'DM') else 0.0
                                    else:
                                        y_val = float(y_val)
                                    reference_robot_pos = np.array([x_val, y_val])
                                    LOG_WARN(f"Step {step}: Reference trajectory too short (len={len(ref_states)}), using CURRENT state position: ({reference_robot_pos[0]:.3f}, {reference_robot_pos[1]:.3f})")
                                except Exception as e:
                                    LOG_DEBUG(f"Could not get current state position for fallback: {e}")
                    
                    # Final fallback: use warmstart directly if reference states not available
                    # This should rarely happen if warmstart is properly initialized
                    if reference_robot_pos is None:
                        has_ws = (hasattr(self, 'solver') and self.solver is not None and
                                 hasattr(self.solver, 'warmstart_values') and 
                                 self.solver.warmstart_values is not None and
                                 isinstance(self.solver.warmstart_values, dict) and
                                 'x' in self.solver.warmstart_values and 
                                 'y' in self.solver.warmstart_values)
                        if has_ws:
                            ws_vals = self.solver.warmstart_values
                            try:
                                x_ws = ws_vals['x']
                                y_ws = ws_vals['y']
                                if isinstance(x_ws, (list, np.ndarray)) and len(x_ws) > step:
                                    import casadi as cd
                                    x_val = x_ws[step]
                                    y_val = y_ws[step]
                                    if isinstance(x_val, (cd.MX, cd.SX)):
                                        x_val = float(cd.DM(x_val))
                                    else:
                                        x_val = float(x_val)
                                    if isinstance(y_val, (cd.MX, cd.SX)):
                                        y_val = float(cd.DM(y_val))
                                    else:
                                        y_val = float(y_val)
                                    reference_robot_pos = np.array([x_val, y_val])
                                    LOG_INFO(f"Step {step}: Using warmstart position directly ({reference_robot_pos[0]:.3f}, {reference_robot_pos[1]:.3f}) for linearization")
                            except Exception as e:
                                LOG_DEBUG(f"Could not get warmstart position directly: {e}")
                        
                        # Last resort: use (0, 0) - this should never happen if warmstart is initialized
                        if reference_robot_pos is None:
                            reference_robot_pos = np.array([0.0, 0.0])
                            has_ws_check = (hasattr(self, 'solver') and self.solver is not None and
                                           hasattr(self.solver, 'warmstart_values') and 
                                           self.solver.warmstart_values is not None and
                                           isinstance(self.solver.warmstart_values, dict) and
                                           'x' in self.solver.warmstart_values and 
                                           'y' in self.solver.warmstart_values)
                            LOG_WARN(f"Step {step}: No reference position available, using (0, 0) - this may cause infeasibility! "
                                   f"(ref_states len={len(ref_states)}, warmstart available={has_ws_check}, "
                                   f"warmstart_keys={list(self.solver.warmstart_values.keys()) if hasattr(self.solver, 'warmstart_values') and self.solver.warmstart_values else 'N/A'})")
                    
                    # Process scenarios for this step with reference position
                    self._process_scenarios_for_step(disc_id, step, data, reference_robot_pos)
            
               # After optimization, estimate support from solution
            if hasattr(self.solver, 'get_solution_trajectory'):
                solution_traj = self.solver.get_solution_trajectory()
                if solution_traj:
                    # Get all constraints
                    all_constraints = []
                    for disc_id in range(self.num_discs):
                        for step in range(self.horizon_length):
                            constraints = self.disc_manager[disc_id].get_constraints_for_step(step)
                            all_constraints.extend(constraints)
                   
                    # Estimate support
                    support_size = self.support_estimator.update_from_solution(
                        all_constraints, solution_traj
                    )
                   
                    LOG_INFO(f"Support estimation: {support_size} active constraints "
                            f"(limit: {self.n_bar})")
                   
                    # Check if support exceeded
                    if self.support_estimator.check_support_exceeded():
                        LOG_WARN(f"⚠️ Support exceeded! {support_size} > {self.n_bar}")
                        self.solve_status = ScenarioSolveStatus.SUPPORT_EXCEEDED
                       
                        # Compute adjusted risk bound
                        self.current_risk_bound = self.compute_risk_bound_after_removal(
                            n_support=support_size,
                            n_samples=len(self.scenarios),
                            n_removed=self.num_removal,
                            beta=self.beta
                        )
                        LOG_WARN(f"Adjusted risk bound: ε = {self.current_risk_bound:.4f}")
           
            self.status = ScenarioStatus.SUCCESS
            LOG_DEBUG("Scenario optimization completed successfully")
            return 1
            
        except Exception as e:
            LOG_WARN(f"Error in scenario optimization: {e}")
            self.status = ScenarioStatus.INFEASIBLE
            return -1
    
    def compute_sample_size(self) -> int:
        """Compute required sample size for scenario optimization."""
        return compute_sample_size(self.epsilon_p, self.beta, self.n_bar)

     
    def get_certified_risk_bound(self) -> Tuple[float, bool]:
        """
        Get the certified risk bound and whether it's valid.
       
        Returns:
            Tuple of (epsilon, is_certified)
            - epsilon: Upper bound on collision probability
            - is_certified: True if support didn't exceed limit
        """
        stats = self.support_estimator.get_statistics()
       
        if stats['exceeded']:
            # Support exceeded - risk bound is not certified
            epsilon = self.current_risk_bound if self.current_risk_bound else 1.0
            return epsilon, False
        else:
            # Support within limit - original risk bound holds
            return self.epsilon_p, True

    
    def _process_scenarios_for_step(self, disc_id: int, step: int, data: Data, reference_robot_pos: np.ndarray = None):
        """
        Process scenarios for a specific disc and time step.
        
        Reference: scenario_module - processes scenarios to create polytope constraints
        Each scenario represents a possible obstacle future, and constraints are formulated
        to ensure the robot avoids all scenario obstacles.
        
        Args:
            disc_id: Disc ID
            step: Time step
            data: Data object (may contain reference trajectory or current state)
            reference_robot_pos: Reference robot position for linearization (from reference trajectory or current state)
        """
        # Get all scenarios (each scenario represents a full trajectory)
        # For each scenario, extract the obstacle position at this time step
        # Reference: scenario_module - scenarios represent full trajectories, extract position at each step
        step_scenarios = []
        for scenario in self.scenarios:
            # Check if this scenario has a trajectory attribute
            if hasattr(scenario, 'trajectory') and scenario.trajectory and len(scenario.trajectory) > step:
                # Use position from trajectory at this time step
                step_scenario = Scenario(scenario.idx_, scenario.obstacle_idx_)
                step_scenario.position = scenario.trajectory[step]
                step_scenario.radius = scenario.radius
                step_scenario.time_step = step
                step_scenarios.append(step_scenario)
            else:
                # Use initial scenario position (for scenarios without trajectory or if step is beyond trajectory length)
                # This handles both old-style scenarios (per-time-step) and new-style scenarios (full trajectories)
                step_scenario = Scenario(scenario.idx_, scenario.obstacle_idx_)
                step_scenario.position = scenario.position
                step_scenario.radius = scenario.radius
                step_scenario.time_step = step
                step_scenarios.append(step_scenario)
        
        if not step_scenarios:
            LOG_DEBUG(f"No step_scenarios for disc {disc_id}, step {step} (total scenarios: {len(self.scenarios)})")
            return
        
        LOG_DEBUG(f"Processing {len(step_scenarios)} scenarios for disc {disc_id}, step {step}")
        
        # Use provided reference robot position (from optimize() method)
        # If not provided, fallback to (0, 0)
        if reference_robot_pos is None:
            reference_robot_pos = np.array([0.0, 0.0])
            LOG_DEBUG(f"Using default reference position (0, 0) for step {step}")
        
        # CRITICAL: Select support set of size n_bar from scenarios
        # Reference: C++ mpc_planner - each time step has its own support set of size n_bar
        # The support set contains the scenarios that will be used to form constraints
        # This is a key aspect of scenario-based MPC: not all scenarios are used, only the support set
        support_scenarios = self._select_support_set(step_scenarios, reference_robot_pos, step)
        
        LOG_DEBUG(f"Selected {len(support_scenarios)} scenarios for support set (n_bar={self.n_bar}) from {len(step_scenarios)} total scenarios")
        
        # Record support set selection for diagnostics (if diagnostics enabled)
        if hasattr(self, 'diagnostics') and self.diagnostics is not None:
            self.diagnostics.record_support_set(disc_id, step, step_scenarios, 
                                               support_scenarios, reference_robot_pos)
        
        # Formulate collision constraints from SUPPORT SET scenarios only
        # Reference: C++ mpc_planner - constraints are formed only from support set scenarios
        constraints = self._formulate_collision_constraints(support_scenarios, disc_id, step, reference_robot_pos)
        
        # Record constraints for diagnostics (if diagnostics enabled)
        if hasattr(self, 'diagnostics') and self.diagnostics is not None:
            self.diagnostics.record_constraints(disc_id, step, constraints, reference_robot_pos)
        
        # Optional scenario removal with big-M relaxation
        if self.num_removal > 0:
            constraints = self.remove_scenarios_with_big_m(constraints, self.num_removal)
        
        # Construct free-space polytope from constraints
        # CRITICAL: Store obstacle positions AND obstacle_idx BEFORE constructing polytope
        # Reference: C++ mpc_planner - polytope construction preserves order, so constraint index i maps to halfspace index i
        obstacle_positions_map = {}
        for i, constraint in enumerate(constraints):
            if hasattr(constraint, 'obstacle_pos') and constraint.obstacle_pos is not None:
                obstacle_positions_map[i] = {
                    'position': np.array(constraint.obstacle_pos).copy() if isinstance(constraint.obstacle_pos, (list, np.ndarray)) else constraint.obstacle_pos.copy(),
                    'radius': constraint.obstacle_radius if hasattr(constraint, 'obstacle_radius') and constraint.obstacle_radius is not None else self.robot_radius,
                    'obstacle_idx': constraint.obstacle_idx if hasattr(constraint, 'obstacle_idx') and constraint.obstacle_idx is not None else 0
                }
                LOG_DEBUG(f"Stored obstacle info for constraint {i}: pos=({obstacle_positions_map[i]['position'][0]:.3f}, {obstacle_positions_map[i]['position'][1]:.3f}), radius={obstacle_positions_map[i]['radius']:.3f}, obstacle_idx={obstacle_positions_map[i]['obstacle_idx']}")
        
        polytope = construct_free_space_polytope(constraints)
        
        # CRITICAL: Store obstacle positions with polytope for symbolic constraint computation
        # Reference: C++ mpc_planner - constraints are computed symbolically using predicted robot position
        # The polytope halfspaces preserve the order of constraints, so index i maps directly
        polytope._obstacle_positions = obstacle_positions_map.copy()
        LOG_DEBUG(f"Stored {len(obstacle_positions_map)} obstacle positions with polytope for disc {disc_id}, step {step} (polytope has {len(polytope.halfspaces)} halfspaces)")
        
        # CRITICAL: Validate polytope feasibility at reference robot position
        # This ensures the constraints form a feasible region around the reference position
        if step == 0 and reference_robot_pos is not None:
            # Check if reference position satisfies all constraints
            ref_pos_feasible = True
            violations = []
            for i, constraint in enumerate(constraints):
                constraint_value = constraint.a1 * reference_robot_pos[0] + constraint.a2 * reference_robot_pos[1] - constraint.b
                if constraint_value > 1e-6:  # Violation
                    ref_pos_feasible = False
                    violations.append((i, constraint_value, constraint))
            
            if not ref_pos_feasible:
                LOG_WARN(f"⚠️  Polytope at disc {disc_id}, step {step}: Reference position ({reference_robot_pos[0]:.3f}, {reference_robot_pos[1]:.3f}) VIOLATES {len(violations)} constraints!")
                for idx, val, const in violations[:3]:
                    LOG_WARN(f"    Constraint {idx}: value={val:.6f} > 0, a1={const.a1:.4f}, a2={const.a2:.4f}, b={const.b:.4f}")
                LOG_WARN(f"  ⚠️  This will cause infeasibility! Constraints are linearized around an infeasible position!")
            else:
                LOG_INFO(f"✓ Polytope at disc {disc_id}, step {step}: Reference position ({reference_robot_pos[0]:.3f}, {reference_robot_pos[1]:.3f}) satisfies all {len(constraints)} constraints")
                
                # Check if goal direction is feasible (if goal is available)
                if data is not None and hasattr(data, 'goal') and data.goal is not None:
                    try:
                        goal_pos = np.array([float(data.goal[0]), float(data.goal[1])])
                        goal_dir = goal_pos - reference_robot_pos
                        goal_dist = np.linalg.norm(goal_dir)
                        if goal_dist > 1e-6:
                            goal_dir_normalized = goal_dir / goal_dist
                            # Check a point slightly in the goal direction
                            test_point = reference_robot_pos + goal_dir_normalized * 0.5  # 0.5m toward goal
                            test_feasible = True
                            test_violations = []
                            for i, constraint in enumerate(constraints):
                                constraint_value = constraint.a1 * test_point[0] + constraint.a2 * test_point[1] - constraint.b
                                if constraint_value > 1e-6:
                                    test_feasible = False
                                    test_violations.append((i, constraint_value))
                            
                            if not test_feasible:
                                LOG_WARN(f"  ⚠️  Goal direction from reference position VIOLATES {len(test_violations)} constraints!")
                                LOG_WARN(f"  ⚠️  Vehicle may not be able to move toward goal - polytope may be too restrictive!")
                            else:
                                LOG_DEBUG(f"  ✓ Goal direction is feasible - vehicle can move toward goal")
                    except Exception as e:
                        LOG_DEBUG(f"  Could not check goal direction feasibility: {e}")
        
        # Validate polytope feasibility
        # Note: Polytope validation might be too strict - constraints will be applied symbolically
        # in calculate_constraints(), so we allow polytopes even if validation fails
        # The actual feasibility will be determined by the MPC solver
        polytope_valid = validate_polytope_feasibility(polytope, [])
        if not polytope_valid:
            LOG_DEBUG(f"Polytope validation failed for disc {disc_id}, step {step}, but allowing (constraints will be applied symbolically)")
        
        # Add polytope to disc manager for THIS step only
        # CRITICAL: Polytope is associated with THIS step (stage) only - constraints are NOT accumulated
        # Reference: C++ mpc_planner - each stage has its own polytope from its own support set
        # The constraints will be applied symbolically in calculate_constraints() using predicted robot position
        self.disc_manager[disc_id].add_polytope(polytope, step)
        LOG_DEBUG(f"Added polytope for disc {disc_id}, step {step} with {len(polytope.halfspaces)} halfspaces (stage-specific)")
        
        # Track active constraints (from support set)
        self._track_active_constraints(constraints, disc_id, step)
    
    def _select_support_set(self, scenarios: List[Scenario], reference_robot_pos: np.ndarray, step: int) -> List[Scenario]:
        """
        Select support set of size n_bar from scenarios.
        
        Reference: C++ mpc_planner - support set selection is critical for scenario-based MPC
        The support set contains the scenarios that will be used to form constraints.
        Each time step has its own support set.
        
        Selection strategy (matching reference):
        1. Prioritize scenarios that are closest to the reference trajectory (most likely to cause collisions)
        2. Ensure diversity in obstacle positions (avoid selecting all scenarios from same obstacle)
        3. Limit to n_bar scenarios to maintain feasibility
        
        Args:
            scenarios: List of all scenarios for this time step
            reference_robot_pos: Reference robot position for this time step
            step: Time step index
        
        Returns:
            List of selected scenarios (support set) of size at most n_bar
        """
        if len(scenarios) <= self.n_bar:
            # If we have fewer scenarios than n_bar, use all of them
            LOG_DEBUG(f"Step {step}: Using all {len(scenarios)} scenarios (n_bar={self.n_bar})")
            return scenarios
        
        # Strategy: Select scenarios closest to reference trajectory (most critical for collision avoidance)
        # Compute distance from reference position to each scenario's obstacle position
        scenario_distances = []
        for scenario in scenarios:
            # Get obstacle position for this scenario at this time step
            if hasattr(scenario, 'trajectory') and scenario.trajectory and step < len(scenario.trajectory):
                obstacle_pos = np.array([float(scenario.trajectory[step][0]), float(scenario.trajectory[step][1])])
            else:
                obstacle_pos = np.array([float(scenario.position[0]), float(scenario.position[1])])
            
            # Distance from reference robot position to obstacle
            dist = np.linalg.norm(obstacle_pos - reference_robot_pos)
            scenario_distances.append((dist, scenario, obstacle_pos))
        
        # Sort by distance (closer obstacles = higher priority for support set)
        scenario_distances.sort(key=lambda x: x[0])
        
        # Select support set: prioritize closest scenarios but ensure diversity
        # Reference: C++ mpc_planner - support set should include diverse scenarios
        selected = []
        selected_obstacle_indices = set()
        
        # First pass: select one scenario per obstacle (if possible) to ensure diversity
        # CRITICAL: Ensure ALL obstacles are represented in the support set if possible
        # This ensures constraints are shown for all obstacles, not just the closest ones
        for dist, scenario, obstacle_pos in scenario_distances:
            if len(selected) >= self.n_bar:
                break
            
            obstacle_idx = scenario.obstacle_idx_
            # Prioritize: if we haven't selected a scenario from this obstacle yet, add it
            # This ensures all obstacles get at least one constraint in the support set
            if obstacle_idx not in selected_obstacle_indices:
                selected.append(scenario)
                selected_obstacle_indices.add(obstacle_idx)
        
        # Second pass: fill remaining slots with closest scenarios (even if from same obstacles)
        # This ensures we use all n_bar slots while maintaining diversity
        for dist, scenario, obstacle_pos in scenario_distances:
            if len(selected) >= self.n_bar:
                break
            
            # Add scenario if not already selected
            if scenario not in selected:
                selected.append(scenario)
        
        # Ensure we have exactly n_bar scenarios (or all if fewer available)
        selected = selected[:self.n_bar]
        
        LOG_DEBUG(f"Step {step}: Selected support set of {len(selected)} scenarios from {len(scenarios)} total "
                 f"(covering {len(selected_obstacle_indices)} obstacles)")
        
        return selected
    
    def _formulate_collision_constraints(self, scenarios: List[Scenario], 
                                       disc_id: int, step: int, reference_robot_pos: np.ndarray = None) -> List[ScenarioConstraint]:
        """
        Formulate collision constraints from scenarios.
        
        Reference: mpc_planner - constraints are linearized around obstacle positions
        The constraint a1*x + a2*y <= b ensures distance >= safety_margin.
        
        Key insight: Constraints are computed once from scenarios and then applied symbolically
        in calculate_constraints(). The (a1, a2, b) values are pre-computed here.
        
        Args:
            scenarios: List of scenarios for this time step
            disc_id: Disc ID
            step: Time step
            reference_robot_pos: Reference robot position for linearization (from reference trajectory or warmstart)
        """
        constraints = []
        
        # Use provided reference robot position or default to (0, 0)
        if reference_robot_pos is None:
            reference_robot_pos = np.array([0.0, 0.0])
            LOG_WARN(f"_formulate_collision_constraints: reference_robot_pos is None, using (0, 0) - this may cause infeasibility!")
        else:
            LOG_DEBUG(f"_formulate_collision_constraints: Using reference_robot_pos=({reference_robot_pos[0]:.3f}, {reference_robot_pos[1]:.3f})")
        
        LOG_DEBUG(f"_formulate_collision_constraints: Processing {len(scenarios)} scenarios for disc {disc_id}, step {step}")
        
        for scenario in scenarios:
            # Get obstacle position from scenario for this time step
            # If scenario has trajectory, use position at this step; otherwise use initial position
            if hasattr(scenario, 'trajectory') and scenario.trajectory and step < len(scenario.trajectory):
                obstacle_pos = np.array([float(scenario.trajectory[step][0]), float(scenario.trajectory[step][1])])
            else:
                obstacle_pos = np.array([float(scenario.position[0]), float(scenario.position[1])])
            
            obstacle_radius = float(scenario.radius)
            
            # Compute constraint parameters directly (matching linearized_constraints.py exactly)
            # Reference: linearized_constraints.py lines 776-798
            # CRITICAL: The constraint normal (a1, a2) should point FROM robot TO obstacle
            # The constraint is: a1*x + a2*y <= b, where b = a1*obstacle_x + a2*obstacle_y - safety_margin
            # This ensures: a1*(x - obstacle_x) + a2*(y - obstacle_y) <= -safety_margin
            # Since (a1, a2) points FROM robot TO obstacle, this keeps robot at least safety_margin away
            
            # Direction vector FROM robot TO obstacle
            # CRITICAL FIX: Use reference robot position (from reference trajectory or warmstart) instead of (0, 0)
            # This ensures constraints are linearized around a feasible robot position, matching linearized_constraints.py
            diff_x = obstacle_pos[0] - reference_robot_pos[0]  # obstacle - robot
            diff_y = obstacle_pos[1] - reference_robot_pos[1]
            dist = np.sqrt(diff_x**2 + diff_y**2)
            dist = max(dist, 1e-6)  # Avoid division by zero
            
            # Normalized direction vector (points FROM robot TO obstacle)
            # Matching linearized_constraints.py line 783-784
            a1 = diff_x / dist
            a2 = diff_y / dist
            
            # Safety margin: robot_radius + obstacle_radius + halfspace_offset
            # Matching linearized_constraints.py line 931: safe_distance = robot_radius + target_obstacle_radius + self.halfspace_offset
            safety_margin = self.robot_radius + obstacle_radius + self.halfspace_offset
            
            # Compute b: a·obs_pos - safety_margin
            # Matching linearized_constraints.py line 794: b_sym = a1_sym * obs_pos[0] + a2_sym * obs_pos[1] - safe_distance
            # This ensures: a1*x + a2*y <= a1*obstacle_x + a2*obstacle_y - safety_margin
            # Which means: a1*(x - obstacle_x) + a2*(y - obstacle_y) <= -safety_margin
            # Since (a1, a2) = (obstacle - robot) / ||obstacle - robot||, this enforces distance >= safety_margin
            b = a1 * obstacle_pos[0] + a2 * obstacle_pos[1] - safety_margin
            
            # Store obstacle position and radius for visualization
            constraint = ScenarioConstraint(
                a1, a2, b, 
                scenario.idx_, scenario.obstacle_idx_, step,
                obstacle_pos=obstacle_pos.copy(),
                obstacle_radius=obstacle_radius
            )
            constraints.append(constraint)
            
            # Log first few constraints for debugging
            if len(constraints) <= 3:
                dist_ref_to_obs = np.sqrt((reference_robot_pos[0] - obstacle_pos[0])**2 + (reference_robot_pos[1] - obstacle_pos[1])**2)
                constraint_value_at_ref = a1 * reference_robot_pos[0] + a2 * reference_robot_pos[1] - b
                LOG_DEBUG(f"  Constraint {len(constraints)-1}: obstacle=({obstacle_pos[0]:.3f}, {obstacle_pos[1]:.3f}), "
                         f"ref_robot=({reference_robot_pos[0]:.3f}, {reference_robot_pos[1]:.3f}), "
                         f"dist={dist_ref_to_obs:.3f}m, a1={a1:.4f}, a2={a2:.4f}, b={b:.4f}, "
                         f"value_at_ref={constraint_value_at_ref:.6f} {'[VIOLATION]' if constraint_value_at_ref > 1e-6 else '[OK]'}")
        
        return constraints
    
    def _track_active_constraints(self, constraints: List[ScenarioConstraint], 
                                disc_id: int, step: int):
        """
        Track active constraints and support.
        
        CRITICAL: In the C++ reference, the support set is tracked PER TIME STEP, not accumulated.
        Each time step has its own support set of size n_bar. The support subsample should track
        unique scenarios across all time steps, but the constraint count per step should be n_bar.
        
        Reference: C++ mpc_planner - support set is per time step, not accumulated.
        """
        # Add constraints to support subsample (for tracking unique scenarios across all steps)
        # But note: each step should have at most n_bar constraints
        for constraint in constraints:
            scenario = Scenario(constraint.scenario_idx, constraint.obstacle_idx)
            self.support_subsample.add(scenario)
        
        # Check support limits: total unique scenarios should not exceed n_bar * horizon_length
        # But per-step, we should have at most n_bar constraints
        # The warning about exceeding n_bar is expected if we're tracking across all steps
        # Instead, we should check if THIS step has more than n_bar constraints
        if len(constraints) > self.n_bar:
            LOG_WARN(f"Step {step}, disc {disc_id}: {len(constraints)} constraints exceed per-step limit {self.n_bar}")
            self.solve_status = ScenarioSolveStatus.SUPPORT_EXCEEDED
        else:
            LOG_DEBUG(f"Step {step}, disc {disc_id}: {len(constraints)} constraints (within limit {self.n_bar})")
    
    def remove_scenarios_with_big_m(self, constraints: List[ScenarioConstraint],
                                    num_removal: int,
                                    reference_pos: np.ndarray = None,
                                    big_m: float = 1e6) -> Tuple[List[ScenarioConstraint], List[int]]:
        """
        Remove scenarios using support-based analysis with big-M relaxation.
        
        Reference: Safe Horizon MPC paper, Algorithm 1 (Greedy Scenario Removal)
        
        The algorithm identifies which scenarios are "of support" (active at the solution)
        and removes the most constraining ones to improve feasibility while tracking
        the impact on the probabilistic guarantee.
        
        Strategy:
        1. Compute constraint "tightness" at reference position
        2. Identify scenarios that are binding (active constraints)
        3. Remove the most restrictive binding scenarios
        4. Track removed scenario indices for risk bound adjustment
        
        Args:
            constraints: List of scenario constraints
            num_removal: Maximum number of scenarios to remove (R in paper)
            reference_pos: Reference position for evaluating constraint tightness
            big_m: Big-M parameter for constraint relaxation
            
        Returns:
            Tuple of (remaining_constraints, removed_indices)
        """
        if num_removal <= 0 or len(constraints) <= num_removal:
            return constraints, []
        
        if reference_pos is None:
            reference_pos = np.array([0.0, 0.0])
        
        # =============================================================
        # STEP 1: Compute constraint tightness (slack) at reference position
        # Tightness = b - (a1*x + a2*y)
        # Smaller tightness = more constraining = higher priority for removal
        # =============================================================
        constraint_tightness = []
        for i, c in enumerate(constraints):
            # Constraint: a1*x + a2*y <= b
            # Slack = b - (a1*x + a2*y)  [positive means satisfied, negative means violated]
            constraint_value = c.a1 * reference_pos[0] + c.a2 * reference_pos[1]
            slack = c.b - constraint_value
            
            # Store (slack, index, constraint, is_binding)
            # A constraint is "binding" if slack is very small (near zero)
            is_binding = abs(slack) < 0.1  # Threshold for "active" constraint
            constraint_tightness.append({
                'index': i,
                'constraint': c,
                'slack': slack,
                'is_binding': is_binding,
                'violation': max(0, -slack)  # How much it's violated (if at all)
            })
        
        # =============================================================
        # STEP 2: Sort by tightness (most constraining first)
        # Priority: 1) Violated constraints, 2) Binding constraints, 3) Slack
        # =============================================================
        constraint_tightness.sort(key=lambda x: (
            -x['violation'],  # Violated constraints first (highest violation)
            -int(x['is_binding']),  # Then binding constraints
            x['slack']  # Then by slack (smallest slack = most constraining)
        ))
        
        # =============================================================
        # STEP 3: Remove most constraining scenarios
        # Reference: Paper Algorithm 1 - greedy removal of support scenarios
        # =============================================================
        removed_indices = []
        remaining_constraints = []
        
        for item in constraint_tightness:
            if len(removed_indices) < num_removal:
                # Remove this constraint (relax with big-M)
                removed_indices.append(item['index'])
                LOG_DEBUG(f"Removing scenario {item['index']}: slack={item['slack']:.4f}, "
                         f"binding={item['is_binding']}, violation={item['violation']:.4f}")
            else:
                remaining_constraints.append(item['constraint'])
        
        LOG_INFO(f"Scenario removal: removed {len(removed_indices)} of {len(constraints)} constraints "
                 f"(removed indices: {removed_indices})")
        
        return remaining_constraints, removed_indices

    def compute_risk_bound_after_removal(self, n_support: int, n_samples: int,
                                          n_removed: int, beta: float) -> float:
        """
        Compute the probability of constraint violation after scenario removal.
        
        Reference: Safe Horizon MPC paper, Theorem 1 with removal adjustment
        
        The effective support dimension increases by the number of removed scenarios.
        
        Args:
            n_support: Estimated support size (number of binding constraints)
            n_samples: Total number of scenarios sampled
            n_removed: Number of scenarios removed
            beta: Confidence level
            
        Returns:
            epsilon: Upper bound on probability of constraint violation
        """
        # Effective dimension = support + removed
        d = n_support + n_removed
        
        # Inverse of sample size formula to get epsilon
        # From: S >= 2/ε × (ln(1/β) + d)
        # We get: ε >= 2 × (ln(1/β) + d) / S
        epsilon = 2.0 * (np.log(1.0 / beta) + d) / n_samples
        
        LOG_DEBUG(f"Risk bound: epsilon={epsilon:.4f} (support={n_support}, "
                 f"removed={n_removed}, samples={n_samples})")
        
        return epsilon
    
    def get_constraint_info(self) -> Dict:
        """Get information about current constraints."""
        total_constraints = 0
        for disc_manager in self.disc_manager:
            for polytope in disc_manager.polytopes:
                total_constraints += len(polytope.halfspaces)
        
        return {
            "total_scenarios": len(self.scenarios),
            "total_constraints": total_constraints,
            "support_size": self.support_subsample.size_,
            "support_limit": self.n_bar,
            "status": self.status,
            "solve_status": self.solve_status
        }
    
    def validate_parameters(self) -> bool:
        """Validate module parameters."""
        if self.epsilon_p <= 0 or self.epsilon_p >= 1:
            LOG_WARN(f"Invalid epsilon_p: {self.epsilon_p}")
            return False
        
        if self.beta <= 0 or self.beta >= 1:
            LOG_WARN(f"Invalid beta: {self.beta}")
            return False
        
        if self.n_bar <= 0:
            LOG_WARN(f"Invalid n_bar: {self.n_bar}")
            return False
        
        if self.robot_radius <= 0:
            LOG_WARN(f"Invalid robot_radius: {self.robot_radius}")
            return False
        
        return True
    
    def is_data_ready(self, data: Data) -> bool:
        """Check if required data is available."""
        try:
            if not (hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles):
                return False
            
            # Check that obstacles exist and have prediction types set
            # Note: predictions.steps may not be populated yet (will be done by propagate_obstacles)
            # So we only check that obstacles have prediction types configured
            for obstacle in data.dynamic_obstacles:
                if not hasattr(obstacle, 'prediction') or not obstacle.prediction:
                    return False
                
                # Check prediction type (PredictionType is an Enum)
                if not hasattr(obstacle.prediction, 'type'):
                    return False
                
                from planning.types import PredictionType
                # Allow GAUSSIAN predictions (steps will be populated by propagate_obstacles)
                if obstacle.prediction.type != PredictionType.GAUSSIAN:
                    return False
            
            return True
            
        except Exception as e:
            LOG_WARN(f"Error checking data readiness: {e}")
            return False
    
    def reset(self):
        """Reset module state."""
        self.scenarios = []
        self.support_subsample.reset()
        self.status = ScenarioStatus.NONE
        self.solve_status = ScenarioSolveStatus.SUCCESS
        
        for disc_manager in self.disc_manager:
            disc_manager.polytopes = []
            disc_manager.active_constraints = []
        
        self.sampler.reset()
        LOG_DEBUG("SafeHorizonModule reset")


class ScenarioSolver:
    """Wrapper for scenario solver with safe horizon module."""
    
    def __init__(self, solver_id: int, solver):
        self.solver_id = solver_id
        self.solver = solver
        self.scenario_module = SafeHorizonModule(solver, {})
        self.exit_code = 0
        self.solver_timeout = 0.1
        
    def get_sampler(self):
        """Get the sampler from the scenario module."""
        return self.scenario_module.sampler


class GreedySupportEstimator:
    """
    Estimates the support set during SQP optimization iterations.
   
    Reference: Safe Horizon MPC paper, Section V.B
   
    The support set consists of scenarios whose constraints are "active" (binding)
    at the current solution. We estimate this during optimization by tracking
    which constraints have near-zero slack.
   
    Key insight: Support estimation during optimization avoids the need to solve
    S additional optimization problems after the main solve (which would be
    computationally intractable for real-time).
    """
   
    def __init__(self, n_bar: int, tolerance: float = 1e-4):
        """
        Args:
            n_bar: Maximum allowed support size
            tolerance: Threshold for considering a constraint "active"
        """
        self.n_bar = n_bar
        self.tolerance = tolerance
        self.support_set = set()  # Set of (scenario_idx, obstacle_idx) tuples
        self.constraint_slacks = {}  # Track slack values per constraint
        self.iteration_supports = []  # Support size per SQP iteration
       
    def reset(self):
        """Reset for new optimization."""
        self.support_set = set()
        self.constraint_slacks = {}
        self.iteration_supports = []
   
    def update_from_solution(self, constraints: List[ScenarioConstraint],
                             solution_trajectory: List[np.ndarray]) -> int:
        """
        Update support estimate from current SQP solution.
       
        Args:
            constraints: List of all scenario constraints
            solution_trajectory: Current solution trajectory [(x0,y0), (x1,y1), ...]
           
        Returns:
            Current estimated support size
        """
        self.support_set.clear()
       
        for constraint in constraints:
            # Get robot position at constraint's time step
            step = constraint.time_step
            if step >= len(solution_trajectory):
                continue
               
            robot_pos = solution_trajectory[step]
           
            # Compute constraint slack
            # Constraint: a1*x + a2*y <= b
            # Slack = b - (a1*x + a2*y)
            constraint_value = constraint.a1 * robot_pos[0] + constraint.a2 * robot_pos[1]
            slack = constraint.b - constraint_value
           
            # Store slack for analysis
            key = (constraint.scenario_idx, constraint.obstacle_idx, step)
            self.constraint_slacks[key] = slack
           
            # Constraint is "active" (in support) if slack is near zero
            if abs(slack) < self.tolerance:
                self.support_set.add((constraint.scenario_idx, constraint.obstacle_idx))
       
        support_size = len(self.support_set)
        self.iteration_supports.append(support_size)
       
        return support_size
   
    def check_support_exceeded(self) -> bool:
        """Check if support exceeds the allowed limit."""
        return len(self.support_set) > self.n_bar
   
    def get_support_scenarios(self) -> List[Tuple[int, int]]:
        """Get list of (scenario_idx, obstacle_idx) in support."""
        return list(self.support_set)
   
    def get_statistics(self) -> dict:
        """Get support estimation statistics."""
        return {
            'current_support': len(self.support_set),
            'max_support': self.n_bar,
            'exceeded': self.check_support_exceeded(),
            'iteration_history': self.iteration_supports.copy(),
            'num_active_constraints': len([s for s in self.constraint_slacks.values()
                                          if abs(s) < self.tolerance])
        }
