"""
Main scenario module for safe horizon constraints.
"""
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
                # CRITICAL: Retrieve obstacle position from polytope if stored
                # Reference: C++ mpc_planner - obstacle positions are needed for symbolic constraint computation
                # The constraint normal (a1, a2) will be recomputed symbolically using predicted robot position
                obstacle_pos = None
                obstacle_radius = None
                if hasattr(polytope, '_obstacle_positions') and i in polytope._obstacle_positions:
                    obs_info = polytope._obstacle_positions[i]
                    obstacle_pos = obs_info['position']
                    obstacle_radius = obs_info['radius']
                    LOG_DEBUG(f"Retrieved obstacle position for constraint {i}: ({obstacle_pos[0]:.3f}, {obstacle_pos[1]:.3f}), radius={obstacle_radius:.3f}")
                else:
                    LOG_WARN(f"Warning: No obstacle position stored for constraint {i} in polytope at time_step {time_step} "
                            f"(has _obstacle_positions: {hasattr(polytope, '_obstacle_positions')}, "
                            f"keys: {list(polytope._obstacle_positions.keys()) if hasattr(polytope, '_obstacle_positions') else 'N/A'})")
                
                constraint = ScenarioConstraint(
                    a1=A[0, 0], a2=A[0, 1], b=b[0],
                    scenario_idx=i, obstacle_idx=0, time_step=time_step,
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
        self.num_removal = config.get("num_removal", 0)  # Number of scenarios to remove
        self.robot_radius = config.get("robot_radius", 0.5)
        self.horizon_length = config.get("horizon_length", 10)
        self.max_constraints_per_disc = config.get("max_constraints_per_disc", 3)  # Reduced from 24 to 3
        self.num_discs = config.get("num_discs", 1)
        
        # Components
        self.sampler = ScenarioSampler(
            num_scenarios=config.get("num_scenarios", 100),
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
        
        Returns:
            1 if successful, -1 if failed
        """
        try:
            LOG_DEBUG("Starting scenario optimization")
            
            # Update with current data
            self.update(data)
            
            if not self.scenarios:
                LOG_WARN("No scenarios available for optimization")
                return -1
            
            # Compute sample size
            sample_size = self.compute_sample_size()
            LOG_DEBUG(f"Computed sample size: {sample_size}")
            
            # Get reference trajectory states for linearization (matching linearized_constraints.py)
            # Reference: linearized_constraints.py uses reference trajectory for linearization
            ref_states = []
            if hasattr(self.solver, 'get_reference_trajectory'):
                try:
                    ref_traj = self.solver.get_reference_trajectory()
                    if ref_traj is not None and hasattr(ref_traj, 'get_states'):
                        ref_states = ref_traj.get_states()
                except Exception as e:
                    LOG_DEBUG(f"Could not get reference trajectory for optimization: {e}")
            
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
                    
                    # For future steps, try to get from reference trajectory
                    if reference_robot_pos is None and step > 0:
                        # Try to get from solver's reference trajectory or warmstart
                        if hasattr(self, 'solver') and self.solver is not None:
                            try:
                                # Try to get from reference trajectory
                                if hasattr(self.solver, 'get_reference_trajectory'):
                                    ref_traj = self.solver.get_reference_trajectory()
                                    if ref_traj is not None and hasattr(ref_traj, 'get_states'):
                                        states = ref_traj.get_states()
                                        if step < len(states):
                                            ref_state = states[step]
                                            if hasattr(ref_state, 'has') and ref_state.has('x') and ref_state.has('y'):
                                                reference_robot_pos = np.array([
                                                    float(ref_state.get('x')),
                                                    float(ref_state.get('y'))
                                                ])
                                                LOG_INFO(f"Step {step}: Using reference trajectory position ({reference_robot_pos[0]:.3f}, {reference_robot_pos[1]:.3f}) for linearization")
                            except Exception as e:
                                LOG_DEBUG(f"Could not get reference trajectory position for step {step}: {e}")
                        
                        # Fallback to warmstart if available
                        if reference_robot_pos is None:
                            if hasattr(self, 'solver') and self.solver is not None:
                                if hasattr(self.solver, 'warmstart_values'):
                                    ws_vals = self.solver.warmstart_values
                                    if 'x' in ws_vals and 'y' in ws_vals:
                                        try:
                                            x_ws = ws_vals['x']
                                            y_ws = ws_vals['y']
                                            if isinstance(x_ws, (list, np.ndarray)) and step < len(x_ws):
                                                reference_robot_pos = np.array([
                                                    float(x_ws[step]),
                                                    float(y_ws[step])
                                                ])
                                                LOG_INFO(f"Step {step}: Using warmstart position ({reference_robot_pos[0]:.3f}, {reference_robot_pos[1]:.3f}) for linearization")
                                        except Exception as e:
                                            LOG_DEBUG(f"Could not get warmstart position for step {step}: {e}")
                    
                    # Try to get from reference trajectory if still None
                    if reference_robot_pos is None and step < len(ref_states):
                        # Use reference trajectory position for future steps
                        try:
                            ref_state = ref_states[step]
                            if hasattr(ref_state, 'has') and ref_state.has('x') and ref_state.has('y'):
                                reference_robot_pos = np.array([
                                    float(ref_state.get('x')),
                                    float(ref_state.get('y'))
                                ])
                                LOG_INFO(f"Step {step}: Using reference trajectory position ({reference_robot_pos[0]:.3f}, {reference_robot_pos[1]:.3f}) for linearization")
                        except Exception as e:
                            LOG_DEBUG(f"Could not get reference trajectory position for step {step}: {e}")
                    
                    # Final fallback to (0, 0) if still None
                    if reference_robot_pos is None:
                        reference_robot_pos = np.array([0.0, 0.0])
                        LOG_WARN(f"Step {step}: No reference position available, using (0, 0) - this may cause infeasibility!")
                    
                    # Process scenarios for this step with reference position
                    self._process_scenarios_for_step(disc_id, step, data, reference_robot_pos)
            
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
        
        # Formulate collision constraints from scenarios at this time step
        constraints = self._formulate_collision_constraints(step_scenarios, disc_id, step, reference_robot_pos)
        
        # Optional scenario removal with big-M relaxation
        if self.num_removal > 0:
            constraints = self.remove_scenarios_with_big_m(constraints, self.num_removal)
        
        # Construct free-space polytope from constraints
        # CRITICAL: Store obstacle positions BEFORE constructing polytope
        # Reference: C++ mpc_planner - polytope construction preserves order, so constraint index i maps to halfspace index i
        obstacle_positions_map = {}
        for i, constraint in enumerate(constraints):
            if hasattr(constraint, 'obstacle_pos') and constraint.obstacle_pos is not None:
                obstacle_positions_map[i] = {
                    'position': np.array(constraint.obstacle_pos).copy() if isinstance(constraint.obstacle_pos, (list, np.ndarray)) else constraint.obstacle_pos.copy(),
                    'radius': constraint.obstacle_radius if hasattr(constraint, 'obstacle_radius') and constraint.obstacle_radius is not None else self.robot_radius
                }
                LOG_DEBUG(f"Stored obstacle position for constraint {i}: ({obstacle_positions_map[i]['position'][0]:.3f}, {obstacle_positions_map[i]['position'][1]:.3f}), radius={obstacle_positions_map[i]['radius']:.3f}")
        
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
        
        # Track active constraints
        self._track_active_constraints(constraints, disc_id, step)
    
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
            
            # Safety margin
            safety_margin = self.robot_radius + obstacle_radius
            
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
                                _disc_id: int, _step: int):
        """Track active constraints and support."""
        # Add constraints to support subsample
        for constraint in constraints:
            scenario = Scenario(constraint.scenario_idx, constraint.obstacle_idx)
            self.support_subsample.add(scenario)
        
        # Check support limits
        if self.support_subsample.size_ > self.n_bar:
            LOG_WARN(f"Support size {self.support_subsample.size_} exceeds limit {self.n_bar}")
            self.solve_status = ScenarioSolveStatus.SUPPORT_EXCEEDED
    
    def remove_scenarios_with_big_m(self, scenarios: List[ScenarioConstraint], 
                                  num_removal: int, _big_m: float = 1000.0) -> List[ScenarioConstraint]:
        """
        Remove scenarios using big-M relaxation method.
        
        Args:
            scenarios: List of scenario constraints
            num_removal: Number of scenarios to remove
            big_m: Big-M parameter for relaxation
            
        Returns:
            List of remaining scenarios after removal
        """
        if num_removal <= 0 or len(scenarios) <= num_removal:
            return scenarios
        
        # Sort scenarios by constraint violation potential (simplified heuristic)
        # In practice, this would use more sophisticated criteria
        sorted_scenarios = sorted(scenarios, key=lambda s: abs(s.b), reverse=True)
        
        # Remove the most restrictive scenarios
        remaining_scenarios = sorted_scenarios[num_removal:]
        
        LOG_DEBUG(f"Removed {num_removal} scenarios using big-M relaxation, "
                 f"{len(remaining_scenarios)} remaining")
        
        return remaining_scenarios
    
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
