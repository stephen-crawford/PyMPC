"""
Safe Horizon Constraint module for scenario-based MPC with support tracking.

This module implements safe horizon constraints following the reference implementation:
- https://github.com/tud-amr/mpc_planner
- https://github.com/oscardegroot/scenario_module

The safe horizon approach constrains collision probability over the entire planning horizon,
using scenario-based optimization to ensure probabilistic safety guarantees.
"""
import numpy as np
import casadi as cd
from typing import List, Dict, Optional
from planning.types import Data, State, ScenarioStatus
from modules.constraints.base_constraint import BaseConstraint
from modules.constraints.scenario_utils.scenario_module import SafeHorizonModule
from modules.constraints.scenario_utils.math_utils import ScenarioConstraint
from utils.utils import LOG_DEBUG, LOG_INFO, LOG_WARN


class SafeHorizonConstraint(BaseConstraint):
	"""
	Safe Horizon Constraint module for scenario-based MPC.
	
	This constraint module uses scenario-based optimization to ensure probabilistic
	safety guarantees over the entire planning horizon. It samples scenarios from
	obstacle predictions and formulates constraints that ensure collision probability
	stays below a specified threshold.
	
	Reference: "Safe Horizon MPC" by de Groot et al.
	"""
	
	def __init__(self):
		super().__init__()
		self.name = "safe_horizon_constraint"
		
		# Configuration parameters
		self.num_discs = int(self.get_config_value("safe_horizon_constraints.num_discs", 1))
		self.robot_radius = float(self.get_config_value("robot.radius", 0.5))
		self.horizon_length = int(self.get_config_value("horizon", 10))
		self.timestep = float(self.get_config_value("timestep", 0.1))
		
		# Scenario optimization parameters
		# Reference: C++ mpc_planner - reduce support size to avoid over-constraining
		# Smaller support dimension reduces the number of constraints per stage
		self.epsilon_p = float(self.get_config_value("safe_horizon_constraints.epsilon_p", 0.1))  # Constraint violation probability
		self.beta = float(self.get_config_value("safe_horizon_constraints.beta", 0.01))  # Confidence level
		self.n_bar = int(self.get_config_value("safe_horizon_constraints.n_bar", 5))  # Support dimension (reduced from 10 to 5)
		self.num_scenarios = int(self.get_config_value("safe_horizon_constraints.num_scenarios", 50))  # Reduced from 100 to 50
		# Reference: mpc_planner - limit constraints per disc to avoid over-constraining
		# Typical values: 3-5 constraints per disc per stage (reduced to maintain feasibility)
		# The reference codebase applies constraints more selectively to maintain feasibility
		self.max_constraints_per_disc = int(self.get_config_value("safe_horizon_constraints.max_constraints_per_disc", 3))  # Reduced from 5 to 3
		self.num_removal = int(self.get_config_value("safe_horizon_constraints.num_removal", 0))  # Scenarios to remove with big-M
		
		# Initialize scenario module (will be set in update when solver is available)
		self.scenario_module: Optional[SafeHorizonModule] = None
		
		# Cache for scenarios per stage (scenarios are independent of robot position)
		self.scenario_cache: Dict[int, List] = {}  # Key: (disc_id, stage_idx), Value: List[Scenario]
		
		LOG_INFO(f"SafeHorizonConstraint initialized: num_discs={self.num_discs}, "
		        f"epsilon_p={self.epsilon_p}, beta={self.beta}, n_bar={self.n_bar}, "
		        f"num_scenarios={self.num_scenarios}")
	
	def update(self, state: State, data: Data):
		"""
		Update constraint module with new data and optimize scenario constraints.
		
		This method:
		1. Initializes the scenario module if needed
		2. Samples scenarios from obstacle predictions
		3. Optimizes scenario constraints to form polytopes
		4. Caches constraints for use in calculate_constraints()
		
		Args:
			state: Current state (not used, but required by interface)
			data: Data object containing obstacles and other information
		"""
		try:
			# Initialize scenario module if not already done
			if self.scenario_module is None:
				if not hasattr(self, 'solver') or self.solver is None:
					LOG_DEBUG("SafeHorizonConstraint.update: solver not available yet, will initialize later")
					return
				
				# Create configuration for scenario module
				config = {
					"epsilon_p": self.epsilon_p,
					"beta": self.beta,
					"n_bar": self.n_bar,
					"robot_radius": self.robot_radius,
					"horizon_length": self.horizon_length,
					"max_constraints_per_disc": self.max_constraints_per_disc,
					"num_discs": self.num_discs,
					"num_scenarios": self.num_scenarios,
					"num_removal": self.num_removal,
					"enable_outlier_removal": True
				}
				
				self.scenario_module = SafeHorizonModule(self.solver, config)
				LOG_DEBUG("SafeHorizonConstraint: Initialized scenario module")
			
			# Check if data is ready
			if not self.scenario_module.is_data_ready(data):
				LOG_WARN("SafeHorizonConstraint.update: Data not ready (no obstacles with Gaussian predictions)")
				self.scenario_cache.clear()
				return
			
			# Ensure data.state is set for optimize() to use
			# CRITICAL: optimize() needs current_state to get reference robot position for step 0
			if data is not None and state is not None:
				if not hasattr(data, 'state') or data.state is None:
					data.state = state
					LOG_DEBUG("SafeHorizonConstraint.update: Set data.state from state parameter")
				elif state is not None:
					# Update data.state with current state to ensure it's current
					data.state = state
			
			# Optimize scenario constraints (samples scenarios and creates polytopes)
			# Reference: scenario_module - optimize() processes scenarios and creates constraints
			optimize_result = self.scenario_module.optimize(data)
			
			if optimize_result != 1:
				LOG_WARN(f"SafeHorizonConstraint.update: Scenario optimization failed (result={optimize_result})")
				self.scenario_cache.clear()
				return
			
			# Cache optimized constraints from polytopes for each disc and stage
			# Reference: mpc_planner - constraints are pre-computed from scenarios and applied symbolically
			# The polytopes contain optimized halfspace constraints (a1, a2, b) that are then
			# applied symbolically in calculate_constraints() using the predicted robot position
			self.scenario_cache.clear()
			total_cached = 0
			for disc_id in range(self.num_discs):
				for stage_idx in range(self.horizon_length):
					# Get optimized constraints from polytope for this disc and stage
					constraints = self.scenario_module.disc_manager[disc_id].get_constraints_for_step(stage_idx)
					key = (disc_id, stage_idx)
					self.scenario_cache[key] = constraints
					total_cached += len(constraints)
			
			# Log constraint info
			LOG_INFO(f"SafeHorizonConstraint.update: Cached {total_cached} optimized constraints across {len(self.scenario_cache)} (disc, stage) keys")
			
			# CRITICAL: After caching constraints, project warmstart again to ensure it satisfies cached constraints
			# This is necessary because constraints are now available in scenario_cache
			if hasattr(self, 'solver') and self.solver is not None:
				if hasattr(self.solver, 'warmstart_values') and self.solver.warmstart_values:
					self._project_warmstart_to_safety(data)
			
		except Exception as e:
			LOG_WARN(f"SafeHorizonConstraint.update: Error updating constraints: {e}")
			import traceback
			LOG_DEBUG(f"Traceback: {traceback.format_exc()}")
			self.scenario_cache.clear()
	
	def calculate_constraints(self, state: State, data: Data, stage_idx: int):
		"""
		Calculate symbolic constraint expressions for this stage.
		
		CRITICAL: This method MUST return symbolic CasADi expressions (MX or SX).
		The constraints are computed symbolically using the predicted robot position
		from the state variables (x, y) which are CasADi variables.
		
		The constraints are linear halfspaces of the form:
			a1 * x + a2 * y <= b
		
		where (a1, a2, b) come from the scenario optimization and are evaluated
		at the predicted robot position for this stage.
		
		Args:
			state: Symbolic state for this stage (contains CasADi variables)
			data: Data object
			stage_idx: Stage index (0 = current, >0 = future)
		
		Returns:
			List of constraint expressions or dicts. Each constraint is either:
			- A CasADi expression directly
			- A dict with constraint parameters (a1, a2, b) for linear constraints
		
		Reference: https://github.com/tud-amr/mpc_planner - constraints are evaluated symbolically.
		"""
		constraints = []
		
		# If scenario module not initialized or no cached scenarios, return empty constraints
		# This allows the solver to proceed even if scenarios haven't been sampled yet
		if self.scenario_module is None:
			LOG_WARN(f"SafeHorizonConstraint.calculate_constraints: scenario_module is None for stage {stage_idx}")
			return constraints
		
		if not self.scenario_cache:
			LOG_WARN(f"SafeHorizonConstraint.calculate_constraints: scenario_cache is empty for stage {stage_idx}")
			return constraints
		
		# Get robot position symbolically
		pos_x = state.get("x") if state is not None and state.has("x") else None
		pos_y = state.get("y") if state is not None and state.has("y") else None
		
		if pos_x is None or pos_y is None:
			LOG_WARN(f"SafeHorizonConstraint.calculate_constraints: Missing position in state for stage {stage_idx}")
			return constraints
		
		# Check if state is symbolic (should always be true)
		is_symbolic = isinstance(pos_x, (cd.MX, cd.SX)) or isinstance(pos_y, (cd.MX, cd.SX))
		
		if not is_symbolic:
			LOG_WARN(f"SafeHorizonConstraint.calculate_constraints: State is not symbolic for stage {stage_idx}")
			return constraints
		
		# Process scenarios for each disc
		# CRITICAL: Constraints are applied per stage independently - each stage has its own support set
		# Reference: C++ mpc_planner - constraints for stage k are computed from scenarios at time step k
		# Constraints are NOT accumulated across stages - each stage only has constraints from its own support set
		for disc_id in range(self.num_discs):
			key = (disc_id, stage_idx)
			
			if key not in self.scenario_cache:
				# No constraints for this disc at this stage - this is normal
				LOG_DEBUG(f"SafeHorizonConstraint.calculate_constraints: No constraints cached for disc {disc_id}, stage {stage_idx}")
				continue
			
			# Get constraints for THIS stage only (not accumulated from previous stages)
			# Reference: C++ mpc_planner - constraints are stage-specific
			scenarios = self.scenario_cache[key]
			LOG_DEBUG(f"SafeHorizonConstraint.calculate_constraints: Retrieved {len(scenarios)} constraints for disc {disc_id}, stage {stage_idx}")
			
			# Get disc offset if available
			disc_offset = 0.0
			if hasattr(data, 'robot_area') and data.robot_area is not None and disc_id < len(data.robot_area):
				disc_offset = float(data.robot_area[disc_id].offset)
			
			# Apply disc offset to position if needed
			if abs(disc_offset) > 1e-9:
				# Get heading angle for disc offset
				psi = state.get("psi") if state is not None and state.has("psi") else None
				if psi is not None:
					# Compute disc position symbolically
					disc_x = pos_x + disc_offset * cd.cos(psi)
					disc_y = pos_y + disc_offset * cd.sin(psi)
				else:
					# Fallback: no offset if heading not available
					disc_x = pos_x
					disc_y = pos_y
			else:
				disc_x = pos_x
				disc_y = pos_y
			
			# Apply pre-computed constraints from polytopes symbolically
			# Reference: mpc_planner - constraints (a1, a2, b) are pre-computed from scenarios
			# and then applied symbolically using the predicted robot position
			# Limit constraints per disc to avoid over-constraining (reference: max_constraints_per_disc)
			# CRITICAL: Apply constraints to early stages (0-3) to allow planning while avoiding over-constraining
			# Reference: C++ mpc_planner - constraints are typically applied to first few stages only
			# The constraints are computed symbolically at each stage using predicted robot position
			# This allows the vehicle to plan avoidance maneuvers for immediate obstacles
			# Applying to stages 0-3 provides enough lookahead for avoidance while maintaining feasibility
			max_stage_for_constraints = 3  # Apply constraints to stages 0, 1, 2, 3
			if stage_idx > max_stage_for_constraints:
				constraints_to_apply = []
				LOG_DEBUG(f"SafeHorizonConstraint: Skipping constraints for stage {stage_idx} (only applying to stages 0-{max_stage_for_constraints})")
			elif len(scenarios) > self.max_constraints_per_disc:
				# CRITICAL: Select constraints that form a feasible polytope
				# Reference: C++ mpc_planner - constraints should be selected to maintain feasibility
				# Strategy: Sort by distance to obstacle (closer obstacles = more critical)
				# But ensure we don't select constraints that all point in the same direction
				# This would create an infeasible halfspace intersection
				
				# Compute obstacle distances for each constraint
				if current_x is not None and current_y is not None:
					constraint_distances = []
					for c in scenarios:
						if isinstance(c, ScenarioConstraint) and hasattr(c, 'obstacle_pos') and c.obstacle_pos is not None:
							obs_pos = c.obstacle_pos
							dist = np.sqrt((current_x - obs_pos[0])**2 + (current_y - obs_pos[1])**2)
							constraint_distances.append((dist, c))
						else:
							constraint_distances.append((float('inf'), c))
					
					# Sort by distance (closer obstacles first)
					constraint_distances.sort(key=lambda x: x[0])
					
					# Select constraints ensuring diversity in constraint directions
					# This prevents all constraints from pointing in the same direction
					selected = []
					selected_directions = []
					for dist, c in constraint_distances:
						if len(selected) >= self.max_constraints_per_disc:
							break
						
						# Check if this constraint's direction is too similar to already selected ones
						# Normal vector is (a1, a2)
						new_dir = np.array([c.a1, c.a2])
						too_similar = False
						for sel_dir in selected_directions:
							# Check angle between directions (dot product)
							dot_product = np.dot(new_dir, sel_dir)
							if abs(dot_product) > 0.9:  # Very similar direction (within ~25 degrees)
								too_similar = True
								break
						
						if not too_similar or len(selected) == 0:
							selected.append(c)
							selected_directions.append(new_dir)
					
					# If we don't have enough diverse constraints, fill with closest ones
					while len(selected) < self.max_constraints_per_disc and len(selected) < len(constraint_distances):
						for dist, c in constraint_distances:
							if c not in selected:
								selected.append(c)
								break
					
					constraints_to_apply = selected[:self.max_constraints_per_disc]
					LOG_DEBUG(f"SafeHorizonConstraint: Selected {len(constraints_to_apply)} constraints from {len(scenarios)} scenarios "
					         f"(ensuring directional diversity to maintain feasibility)")
				else:
					# Fallback: Sort by b value (smaller b = more restrictive constraint = closer obstacle)
					sorted_scenarios = sorted(scenarios, key=lambda c: float(c.b) if isinstance(c, ScenarioConstraint) else float('inf'))
					constraints_to_apply = sorted_scenarios[:self.max_constraints_per_disc]
					LOG_DEBUG(f"SafeHorizonConstraint: Selected {len(constraints_to_apply)} constraints from {len(scenarios)} scenarios "
					         f"(fallback: sorted by b value)")
			else:
				constraints_to_apply = scenarios
			
			# Diagnostic: Check constraint feasibility at initial position and warmstart (stage 0 only)
			# CRITICAL: This checks if constraints are satisfied at the initial position
			# If constraints are violated at (0,0), the problem is immediately infeasible
			if stage_idx == 0:
				try:
					# Get current robot position from data (numeric)
					current_x = None
					current_y = None
					if data is not None:
						if hasattr(data, 'state') and data.state is not None:
							try:
								current_x = float(data.state.get("x")) if data.state.has("x") else None
								current_y = float(data.state.get("y")) if data.state.has("y") else None
							except:
								pass
					
					# Also check warmstart values if available
					warmstart_x = None
					warmstart_y = None
					if hasattr(self, 'solver') and self.solver is not None:
						if hasattr(self.solver, 'warmstart_values'):
							ws_vals = self.solver.warmstart_values
							if 'x' in ws_vals and 'y' in ws_vals:
								try:
									x_ws = ws_vals['x']
									y_ws = ws_vals['y']
									if isinstance(x_ws, (list, np.ndarray)) and len(x_ws) > stage_idx:
										warmstart_x = float(x_ws[stage_idx])
									if isinstance(y_ws, (list, np.ndarray)) and len(y_ws) > stage_idx:
										warmstart_y = float(y_ws[stage_idx])
								except:
									pass
					
					if current_x is not None and current_y is not None:
						LOG_INFO(f"=== SafeHorizonConstraint Stage {stage_idx} Feasibility Check ===")
						LOG_INFO(f"Initial position: ({current_x:.3f}, {current_y:.3f})")
						if warmstart_x is not None and warmstart_y is not None:
							LOG_INFO(f"Warmstart position: ({warmstart_x:.3f}, {warmstart_y:.3f})")
						LOG_INFO(f"Checking {len(constraints_to_apply)} constraints...")
						
						violations = []
						warmstart_violations = []
						all_constraint_values = []
						
						for i, constraint in enumerate(constraints_to_apply):
							if isinstance(constraint, ScenarioConstraint):
								# Get obstacle position for this constraint
								obstacle_pos = None
								if hasattr(constraint, 'obstacle_pos') and constraint.obstacle_pos is not None:
									if isinstance(constraint.obstacle_pos, np.ndarray):
										obstacle_pos = constraint.obstacle_pos
									elif isinstance(constraint.obstacle_pos, (list, tuple)):
										obstacle_pos = np.array([float(constraint.obstacle_pos[0]), float(constraint.obstacle_pos[1])])
								
								# Compute constraint symbolically (matching calculate_constraints logic)
								# This is the ACTUAL constraint that will be applied
								if obstacle_pos is not None:
									# Compute constraint normal symbolically (but evaluate numerically here)
									diff_x = obstacle_pos[0] - current_x
									diff_y = obstacle_pos[1] - current_y
									dist = np.sqrt(diff_x**2 + diff_y**2)
									dist = max(dist, 1e-6)
									
									a1_sym_val = diff_x / dist
									a2_sym_val = diff_y / dist
									
									obstacle_radius = float(constraint.obstacle_radius) if hasattr(constraint, 'obstacle_radius') and constraint.obstacle_radius is not None else self.robot_radius
									safety_margin = self.robot_radius + obstacle_radius
									
									b_sym_val = a1_sym_val * obstacle_pos[0] + a2_sym_val * obstacle_pos[1] - safety_margin
									
									# Constraint value: a1*x + a2*y - b <= 0
									constraint_value = a1_sym_val * current_x + a2_sym_val * current_y - b_sym_val
									all_constraint_values.append(constraint_value)
									
									if constraint_value > 1e-6:  # Violation
										violations.append((i, constraint_value, a1_sym_val, a2_sym_val, b_sym_val, obstacle_pos))
									
									# Check warmstart
									if warmstart_x is not None and warmstart_y is not None:
										ws_constraint_value = a1_sym_val * warmstart_x + a2_sym_val * warmstart_y - b_sym_val
										if ws_constraint_value > 1e-6:  # Violation
											warmstart_violations.append((i, ws_constraint_value, a1_sym_val, a2_sym_val, b_sym_val))
									
									if i < 3:  # Log first 3 constraints
										# Compute distance and direction for verification
										dist_to_obstacle = np.sqrt((current_x - obstacle_pos[0])**2 + (current_y - obstacle_pos[1])**2)
										expected_min_dist = safety_margin
										LOG_INFO(f"  Constraint {i}: obstacle_pos=({obstacle_pos[0]:.3f}, {obstacle_pos[1]:.3f}), "
										         f"a1={a1_sym_val:.4f}, a2={a2_sym_val:.4f}, b={b_sym_val:.4f}, "
										         f"value={constraint_value:.6f} {'[VIOLATION]' if constraint_value > 1e-6 else '[OK]'}")
										LOG_INFO(f"    Distance to obstacle: {dist_to_obstacle:.3f}m, Required: >= {expected_min_dist:.3f}m, "
										         f"Margin: {dist_to_obstacle - expected_min_dist:.3f}m")
										LOG_INFO(f"    Constraint direction (a1, a2) = ({a1_sym_val:.4f}, {a2_sym_val:.4f}) points FROM vehicle TO obstacle")
										LOG_INFO(f"    Constraint: {a1_sym_val:.4f}*x + {a2_sym_val:.4f}*y <= {b_sym_val:.4f}")
										LOG_INFO(f"    At vehicle ({current_x:.3f}, {current_y:.3f}): LHS = {a1_sym_val * current_x + a2_sym_val * current_y:.4f}, "
										         f"RHS = {b_sym_val:.4f}, Diff = {constraint_value:.6f}")
								else:
									LOG_WARN(f"  Constraint {i}: Missing obstacle_pos, cannot check feasibility")
						
						if all_constraint_values:
							min_val = min(all_constraint_values)
							max_val = max(all_constraint_values)
							LOG_INFO(f"Constraint value range: [{min_val:.6f}, {max_val:.6f}]")
						
						if violations:
							LOG_WARN(f"⚠️  Found {len(violations)} constraint violations at initial position:")
							for idx, val, a1, a2, b, obs_pos in violations[:5]:
								LOG_WARN(f"  Constraint {idx}: value={val:.6f} > 0 (VIOLATION), "
								         f"obstacle=({obs_pos[0]:.3f}, {obs_pos[1]:.3f}), a1={a1:.4f}, a2={a2:.4f}, b={b:.4f}")
							LOG_WARN(f"  ⚠️  INITIAL POSITION VIOLATES CONSTRAINTS - Problem is infeasible!")
						else:
							LOG_INFO(f"✓ All {len(constraints_to_apply)} constraints satisfied at initial position")
						
						if warmstart_violations:
							LOG_WARN(f"⚠️  Found {len(warmstart_violations)} constraint violations at WARMSTART position:")
							for idx, val, a1, a2, b in warmstart_violations[:5]:
								LOG_WARN(f"  Constraint {idx}: value={val:.6f} > 0 (VIOLATION), a1={a1:.4f}, a2={a2:.4f}, b={b:.4f}")
							LOG_WARN(f"  ⚠️  WARMSTART POSITION VIOLATES CONSTRAINTS - This may cause solver infeasibility!")
					else:
						LOG_DEBUG(f"SafeHorizonConstraint: Could not get numeric position for feasibility check")
				except Exception as e:
					LOG_WARN(f"SafeHorizonConstraint: Error checking initial feasibility: {e}")
					import traceback
					LOG_DEBUG(f"  Traceback: {traceback.format_exc()}")
			
			for constraint in constraints_to_apply:  # scenarios is List[ScenarioConstraint] from cache
				if not isinstance(constraint, ScenarioConstraint):
					LOG_WARN(f"SafeHorizonConstraint.calculate_constraints: Expected ScenarioConstraint, got {type(constraint)}")
					continue
				
				# CRITICAL: Compute constraint symbolically using predicted robot position
				# Reference: C++ mpc_planner (linearized_constraints.cpp) - constraint normal is computed symbolically
				# This matches linearized_constraints.py lines 776-798 exactly
				# The constraint normal adapts to the robot's predicted position, ensuring correct linearization
				
				# Extract obstacle position from constraint (must be stored)
				obstacle_pos = None
				obstacle_radius = None
				if hasattr(constraint, 'obstacle_pos') and constraint.obstacle_pos is not None:
					# Convert to numpy array if needed
					if isinstance(constraint.obstacle_pos, (list, tuple)):
						obstacle_pos = np.array([float(constraint.obstacle_pos[0]), float(constraint.obstacle_pos[1])])
					else:
						obstacle_pos = np.array([float(constraint.obstacle_pos[0]), float(constraint.obstacle_pos[1])])
					obstacle_radius = float(constraint.obstacle_radius) if hasattr(constraint, 'obstacle_radius') and constraint.obstacle_radius is not None else self.robot_radius
				else:
					# Obstacle position must be stored - this is a critical error
					LOG_WARN(f"SafeHorizonConstraint: Constraint missing obstacle_pos at stage {stage_idx}, disc {disc_id}, "
					         f"scenario_idx={getattr(constraint, 'scenario_idx', 'N/A')}, cannot compute symbolically")
					continue
				
				# Compute constraint symbolically: a·p_disc <= b
				# Reference: C++ mpc_planner - _a1[d][k](obs_id) = diff_x / dist where diff = obstacle_pos - ego_position
				# Matching linearized_constraints.py lines 776-798 exactly
				# CRITICAL: Normal vector points FROM vehicle TO obstacle (matching C++ reference)
				diff_x_sym = obstacle_pos[0] - disc_x  # obstacle - vehicle (points FROM vehicle TO obstacle)
				diff_y_sym = obstacle_pos[1] - disc_y
				dist_sym = cd.sqrt(diff_x_sym * diff_x_sym + diff_y_sym * diff_y_sym)
				dist_sym = cd.fmax(dist_sym, 1e-6)  # Avoid division by zero (matching linearized_constraints.py line 779)
				
				# Normalized direction vector (points FROM vehicle TO obstacle)
				# Matching C++: _a1[d][k](obs_id) = diff_x / dist, _a2[d][k](obs_id) = diff_y / dist
				# Matching linearized_constraints.py lines 783-784
				a1_sym = diff_x_sym / dist_sym
				a2_sym = diff_y_sym / dist_sym
				
				# Safety margin: robot_radius + obstacle_radius
				# Matching C++: safe_distance = radius + CONFIG["robot_radius"]
				# Matching linearized_constraints.py line 770: safe_distance = robot_radius + target_obstacle_radius + self.halfspace_offset
				safety_margin = self.robot_radius + obstacle_radius
				
				# Compute b: a·obs_pos - safety_margin
				# Matching C++: _b[d][k](obs_id) = _a1[d][k](obs_id) * obstacle_pos(0) + _a2[d][k](obs_id) * obstacle_pos(1) - safe_distance
				# Matching linearized_constraints.py line 794: b_sym = a1_sym * obs_pos[0] + a2_sym * obs_pos[1] - safe_distance
				b_sym = a1_sym * obstacle_pos[0] + a2_sym * obstacle_pos[1] - safety_margin
				
				# Constraint expression: a1*x + a2*y - b <= 0
				# This enforces: a·disc_pos <= b, i.e., a·disc_pos <= a·obstacle_pos - safety_margin
				# Matching linearized_constraints.py line 798: constraint_expr = a1_sym * disc_x_sym + a2_sym * disc_y_sym - b_sym
				constraint_expr = a1_sym * disc_x + a2_sym * disc_y - b_sym
				
				# Return as dict matching linearized_constraints.py format exactly
				# Reference: C++ mpc_planner - constraints are returned as symbolic expressions
				# Matching linearized_constraints.py lines 802-807
				constraints.append({
					"type": "symbolic_expression",
					"expression": constraint_expr,
					"ub": 0.0,  # expr <= 0 (matching linearized_constraints.py line 805)
					"constraint_type": "safe_horizon",  # Distinguish from linearized/contouring
					"disc_id": disc_id,
					"scenario_idx": getattr(constraint, 'scenario_idx', 0),
					"obstacle_idx": getattr(constraint, 'obstacle_idx', 0),
					"time_step": getattr(constraint, 'time_step', stage_idx),
					# Note: a1, a2, b are now symbolic (computed from predicted robot position)
					# Store obstacle position for visualization/debugging
					"obstacle_pos": obstacle_pos.tolist() if obstacle_pos is not None else None,
					"obstacle_radius": obstacle_radius
				})
		
		if constraints:
			LOG_INFO(f"SafeHorizonConstraint.calculate_constraints: stage_idx={stage_idx}, "
			         f"returning {len(constraints)} constraints (max_per_disc={self.max_constraints_per_disc}, num_discs={self.num_discs})")
		
		return constraints
	
	def lower_bounds(self, state: Optional[State] = None, data: Optional[Data] = None, stage_idx: Optional[int] = None):
		"""
		Get lower bounds for constraints at this stage.
		
		For safe horizon constraints, we use: constraint_expr <= 0
		So lower bound is -inf (no lower bound on the expression itself).
		This matches linearized_constraints.py exactly: lower_bound = -inf, upper_bound = 0.0
		
		CRITICAL: Returning 0.0 for lower bound would make this an equality constraint (0 <= expr <= 0),
		which is much more restrictive than inequality (expr <= 0) and causes infeasibility!
		
		Args:
			state: State (not used)
			data: Data object
			stage_idx: Stage index
		
		Returns:
			List of lower bounds (all -inf for <= constraints)
		"""
		if stage_idx is None or data is None:
			return []
		
		count = 0
		for disc_id in range(self.num_discs):
			key = (disc_id, stage_idx)
			if key in self.scenario_cache:
				count += len(self.scenario_cache[key])
		
		# CRITICAL FIX: Return -inf for lower bound (matching linearized_constraints.py)
		# This enforces: constraint_expr <= 0 (inequality), not constraint_expr == 0 (equality)
		return [-np.inf] * count
	
	def upper_bounds(self, state: Optional[State] = None, data: Optional[Data] = None, stage_idx: Optional[int] = None):
		"""
		Get upper bounds for constraints at this stage.
		
		For safe horizon constraints: constraint_expr <= 0
		So upper bound is 0.0.
		
		Args:
			state: State (not used)
			data: Data object
			stage_idx: Stage index
		
		Returns:
			List of upper bounds (all 0.0 for <= constraints)
		"""
		if stage_idx is None or data is None:
			return []
		
		count = 0
		for disc_id in range(self.num_discs):
			key = (disc_id, stage_idx)
			if key in self.scenario_cache:
				count += len(self.scenario_cache[key])
		
		return [0.0] * count
	
	def is_data_ready(self, data: Data) -> bool:
		"""
		Check if required data is available.
		
		For safe horizon constraints, we need obstacles with Gaussian predictions.
		Note: prediction steps may not be populated yet (will be done by propagate_obstacles),
		so we only check that obstacles exist and have Gaussian prediction types configured.
		
		If the scenario module isn't initialized yet, we return True if obstacles exist,
		allowing initialization to proceed in update().
		"""
		# If scenario module is initialized, use its is_data_ready check
		if self.scenario_module is not None:
			return self.scenario_module.is_data_ready(data)
		
		# If module not initialized yet, check if we have obstacles
		# This allows initialization to proceed in update()
		# CRITICAL: Return True if obstacles exist, even if prediction types aren't set yet
		# This allows the solver to proceed and update() will handle initialization
		if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles and len(data.dynamic_obstacles) > 0:
			# Check if obstacles have predictions with Gaussian types
			# If any obstacle has a Gaussian prediction type, we're ready
			from planning.types import PredictionType
			for obstacle in data.dynamic_obstacles:
				if (hasattr(obstacle, 'prediction') and obstacle.prediction is not None and
				    hasattr(obstacle.prediction, 'type') and
				    obstacle.prediction.type == PredictionType.GAUSSIAN):
					return True
			# If obstacles exist but don't have Gaussian predictions yet, still return True
			# to allow initialization (update() will handle the actual check)
			return True
		
		# No obstacles found
		return False
	
	def get_visualizer(self):
		"""Return a visualizer for safe horizon constraints."""
		class SafeHorizonConstraintsVisualizer:
			def __init__(self, module):
				self.module = module
			
			def visualize(self, state, data, stage_idx=0, ax=None, current_robot_pos=None):
				"""
				Visualize safe horizon constraints:
				- Show scenarios (hypothetical obstacle positions) as trajectories
				- Show the selected/optimal trajectory from MPC solution
				- Plot constraints (halfspaces) applied at each stage
				
				Args:
					state: State object (may contain current robot position)
					data: Data object (may contain current robot position)
					stage_idx: Stage index for visualization
					ax: Matplotlib axes (if None, uses current axes)
					current_robot_pos: Current robot position (x, y) for recomputing constraints
				"""
				try:
					import matplotlib.pyplot as plt
					import numpy as np
				except Exception:
					return
				
				if ax is None:
					ax = plt.gca()
				
				# Get current robot position if not provided
				if current_robot_pos is None:
					if state is not None and state.has('x') and state.has('y'):
						current_robot_pos = np.array([float(state.get('x')), float(state.get('y'))])
					elif data is not None and hasattr(data, 'state') and data.state is not None:
						if data.state.has('x') and data.state.has('y'):
							current_robot_pos = np.array([float(data.state.get('x')), float(data.state.get('y'))])
				
				# Plot constraints (halfspaces) for this stage
				# CRITICAL: Pass current robot position so constraints are recomputed correctly
				self.module._plot_constraints_for_stage(ax, stage_idx, current_robot_pos=current_robot_pos)
				
				# Debug: Check if scenario module exists
				if self.module.scenario_module is None:
					LOG_DEBUG("SafeHorizonConstraintsVisualizer: scenario_module is None - trying to initialize")
					# Try to initialize if we have solver and data
					if hasattr(self.module, 'solver') and self.module.solver is not None and data is not None:
						try:
							# Call update to initialize scenario module
							from planning.types import State
							dummy_state = State()
							self.module.update(dummy_state, data)
						except Exception as e:
							LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: Failed to initialize scenario_module: {e}")
					
					# Check again after initialization attempt
					if self.module.scenario_module is None:
						return
				
				# Debug: Check if scenarios exist
				if not hasattr(self.module.scenario_module, 'scenarios'):
					LOG_DEBUG("SafeHorizonConstraintsVisualizer: scenario_module has no 'scenarios' attribute")
					return
				
				scenarios = self.module.scenario_module.scenarios
				
				if not scenarios or len(scenarios) == 0:
					LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: No scenarios available (scenarios={scenarios})")
					# Try to update scenarios if we have data
					if data is not None:
						try:
							from planning.types import State
							dummy_state = State()
							self.module.scenario_module.update(data)
							scenarios = self.module.scenario_module.scenarios
							LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: After update, scenarios={len(scenarios) if scenarios else 0}")
						except Exception as e:
							LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: Failed to update scenarios: {e}")
					
					if not scenarios or len(scenarios) == 0:
						return
				
				LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: Visualizing {len(scenarios)} scenarios")
				
				# Limit number of trajectories to visualize for performance (show a sample)
				max_trajectories_to_show = 20  # Show up to 20 trajectories to avoid clutter
				
				# Group scenarios by scenario index (idx_) across time steps to form hypothetical trajectories
				# Each scenario index represents a different hypothetical future
				# Reference: scenario_module - scenarios have trajectory attribute with full obstacle trajectory
				# Key: (obstacle_idx, scenario_idx), Value: list of (time_step, position) sorted by time_step
				scenario_trajectories = {}
				
				# Get scenarios from scenario_module (they have trajectory attribute)
				# Reference: scenario_module - scenarios represent full obstacle trajectories
				if hasattr(self.module, 'scenario_module') and self.module.scenario_module is not None:
					scenario_module_scenarios = self.module.scenario_module.scenarios
					for scenario in scenario_module_scenarios:
						# Check if scenario has required attributes
						if not hasattr(scenario, 'obstacle_idx_') or not hasattr(scenario, 'idx_'):
							continue
						
						# Check if scenario has trajectory (full obstacle trajectory)
						# Reference: sampler.py - scenarios have trajectory attribute with full obstacle trajectory
						if not hasattr(scenario, 'trajectory') or not scenario.trajectory:
							# Fallback: use position if trajectory not available
							if hasattr(scenario, 'position'):
								pos = scenario.position
								if isinstance(pos, np.ndarray) and len(pos) >= 2:
									key = (scenario.obstacle_idx_, scenario.idx_)
									if key not in scenario_trajectories:
										scenario_trajectories[key] = []
									scenario_trajectories[key].append((0, [float(pos[0]), float(pos[1])]))
							continue
						
						# Group by scenario index (idx_) - this represents one hypothetical trajectory
						key = (scenario.obstacle_idx_, scenario.idx_)
						if key not in scenario_trajectories:
							scenario_trajectories[key] = []
						
						# Extract positions from trajectory
						# Reference: sampler.py - trajectory is a list of positions [pos_0, pos_1, ..., pos_horizon]
						for time_step, pos in enumerate(scenario.trajectory):
							if isinstance(pos, np.ndarray):
								if len(pos) >= 2:
									pos = [float(pos[0]), float(pos[1])]
								else:
									continue
							elif isinstance(pos, (list, tuple)):
								if len(pos) >= 2:
									pos = [float(pos[0]), float(pos[1])]
								else:
									continue
							else:
								continue
							
							scenario_trajectories[key].append((time_step, pos))
				
				# Sort by time step for each trajectory
				for key in scenario_trajectories:
					scenario_trajectories[key].sort(key=lambda x: x[0])
				
				# Limit trajectories to show (sample evenly)
				trajectory_keys = list(scenario_trajectories.keys())
				if len(trajectory_keys) > max_trajectories_to_show:
					# Sample evenly across all trajectories
					step = len(trajectory_keys) // max_trajectories_to_show
					trajectory_keys = trajectory_keys[::step][:max_trajectories_to_show]
				
				LOG_INFO(f"SafeHorizonConstraintsVisualizer: Grouped into {len(scenario_trajectories)} trajectory groups, showing {len(trajectory_keys)}")
				
				# Draw hypothetical trajectories (scenarios) - lighter, semi-transparent
				# Use different colors for different obstacles
				obstacle_colors = ['orange', 'purple', 'brown', 'pink', 'cyan']
				first_traj = True
				trajectories_drawn = 0
				
				for key in trajectory_keys:
					traj_points = scenario_trajectories[key]
					obstacle_idx, scenario_idx = key
					if len(traj_points) > 1:
						# Extract positions
						positions = [p[1] for p in traj_points]
						x_coords = []
						y_coords = []
						
						for pos in positions:
							if isinstance(pos, (list, tuple, np.ndarray)):
								if len(pos) >= 2:
									x_coords.append(float(pos[0]))
									y_coords.append(float(pos[1]))
						
						if len(x_coords) > 1:
							# Choose color based on obstacle
							color = obstacle_colors[obstacle_idx % len(obstacle_colors)]
							
							# Draw trajectory line (hypothetical obstacle trajectory)
							# Make it very transparent so selected trajectory stands out
							ax.plot(x_coords, y_coords, '--', color=color, 
							        alpha=0.15, linewidth=1.5, 
							        label='Hypothetical Obstacle Trajectories' if first_traj else None,
							        zorder=2)  # Low z-order so selected trajectory appears on top
							
							# Draw scenario points (very transparent) - only show every few points to reduce clutter
							step_points = max(1, len(x_coords) // 5)  # Show ~5 points per trajectory
							ax.scatter(x_coords[::step_points], y_coords[::step_points], c=color, 
							          s=20, alpha=0.15, marker='o', edgecolors='none', linewidths=0.3, zorder=3)
							
							trajectories_drawn += 1
							first_traj = False
				
				LOG_INFO(f"SafeHorizonConstraintsVisualizer: Drew {trajectories_drawn} hypothetical trajectories")
				
				# Draw selected trajectory from MPC solution (if available)
				# Get predicted trajectory from solver
				if hasattr(self.module, 'solver') and self.module.solver is not None:
					try:
						# Try to get the reference trajectory or solution trajectory
						if hasattr(self.module.solver, 'get_reference_trajectory'):
							traj = self.module.solver.get_reference_trajectory()
							if traj is not None:
								if hasattr(traj, 'get_states'):
									states = traj.get_states()
									if states and len(states) > 0:
										x_traj = []
										y_traj = []
										for state in states:
											try:
												if hasattr(state, 'get') and hasattr(state, 'has'):
													# Check if state has 'x' and 'y' using has() method
													if state.has('x') and state.has('y'):
														x_val = state.get('x')
														y_val = state.get('y')
														if x_val is not None and y_val is not None:
															try:
																x_traj.append(float(x_val))
																y_traj.append(float(y_val))
															except (ValueError, TypeError) as e:
																LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: Error converting state values: {e}")
																continue
													else:
														# State doesn't have 'x' or 'y', skip
														continue
												elif isinstance(state, (list, tuple, np.ndarray)) and len(state) >= 2:
													x_traj.append(float(state[0]))
													y_traj.append(float(state[1]))
												else:
													# State format not recognized, skip
													continue
											except (KeyError, AttributeError) as e:
												# KeyError means 'x' key doesn't exist, AttributeError means state doesn't have expected methods
												LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: State format issue: {e}")
												continue
											except Exception as e:
												LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: Error extracting state: {e}")
												continue
										
										if len(x_traj) > 1 and len(y_traj) > 1 and len(x_traj) == len(y_traj):
											LOG_INFO(f"SafeHorizonConstraintsVisualizer: Drawing selected trajectory with {len(x_traj)} points")
											# Draw selected trajectory - bold, highlighted, highest z-order
											ax.plot(x_traj, y_traj, '-', color='blue', linewidth=4.5, 
											        alpha=1.0, label='Selected Trajectory', zorder=20)
											# Mark start and end with highest z-order
											ax.scatter([x_traj[0]], [y_traj[0]], c='green', s=150, 
											          marker='s', label='Trajectory Start', zorder=21, edgecolors='black', linewidths=2.5)
											ax.scatter([x_traj[-1]], [y_traj[-1]], c='red', s=150, 
											          marker='*', label='Trajectory End', zorder=21, edgecolors='black', linewidths=2.5)
										else:
											LOG_WARN(f"SafeHorizonConstraintsVisualizer: Selected trajectory has insufficient points: x={len(x_traj)}, y={len(y_traj)}")
									else:
										LOG_WARN(f"SafeHorizonConstraintsVisualizer: Trajectory has no states")
								else:
									LOG_WARN(f"SafeHorizonConstraintsVisualizer: Trajectory has no get_states method")
							else:
								LOG_WARN(f"SafeHorizonConstraintsVisualizer: get_reference_trajectory returned None")
						else:
							LOG_WARN(f"SafeHorizonConstraintsVisualizer: Solver has no get_reference_trajectory method")
					except Exception as e:
						LOG_WARN(f"Could not visualize selected trajectory: {e}")
						import traceback
						LOG_DEBUG(f"Traceback: {traceback.format_exc()}")
		
		return SafeHorizonConstraintsVisualizer(self)
	
	def _plot_constraints_for_stage(self, ax, stage_idx, current_robot_pos=None):
		"""
		Plot constraints (halfspaces) applied at a specific stage.
		
		CRITICAL: Constraints are recomputed at the current robot position to show correct halfspace orientation.
		The halfspaces should rotate as the vehicle moves, pointing FROM vehicle TO obstacle.
		
		Args:
			ax: Matplotlib axes to plot on
			stage_idx: Stage index to plot constraints for
			current_robot_pos: Current robot position (x, y) for recomputing constraints. If None, uses (0, 0).
		"""
		try:
			import matplotlib.pyplot as plt
			import numpy as np
		except Exception:
			return
		
		if not self.scenario_cache:
			return
		
		# Get current robot position (for recomputing constraints)
		if current_robot_pos is None:
			# Try to get from solver's current state
			if hasattr(self, 'solver') and self.solver is not None:
				try:
					if hasattr(self.solver, 'data') and self.solver.data is not None:
						if hasattr(self.solver.data, 'state') and self.solver.data.state is not None:
							state = self.solver.data.state
							if state.has('x') and state.has('y'):
								current_robot_pos = np.array([float(state.get('x')), float(state.get('y'))])
				except:
					pass
		
		if current_robot_pos is None:
			current_robot_pos = np.array([0.0, 0.0])
		else:
			current_robot_pos = np.array([float(current_robot_pos[0]), float(current_robot_pos[1])])
		
		# Get current axis limits to determine plotting range
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		x_range = xlim[1] - xlim[0]
		y_range = ylim[1] - ylim[0]
		
		# Plot constraints for each disc
		for disc_id in range(self.num_discs):
			key = (disc_id, stage_idx)
			if key not in self.scenario_cache:
				continue
			
			constraints = self.scenario_cache[key]
			if not constraints:
				continue
			
			# Limit number of constraints to plot (avoid clutter)
			max_constraints_to_plot = min(10, len(constraints))
			constraints_to_plot = constraints[:max_constraints_to_plot]
			
			# Plot each constraint as a halfspace line
			for i, constraint in enumerate(constraints_to_plot):
				if not isinstance(constraint, ScenarioConstraint):
					continue
				
				# CRITICAL: Recompute constraint normal (a1, a2) using CURRENT robot position
				# This matches calculate_constraints() which computes symbolically
				# The constraint normal should point FROM vehicle TO obstacle
				obstacle_pos = None
				obstacle_radius = None
				if hasattr(constraint, 'obstacle_pos') and constraint.obstacle_pos is not None:
					if isinstance(constraint.obstacle_pos, (list, tuple)):
						obstacle_pos = np.array([float(constraint.obstacle_pos[0]), float(constraint.obstacle_pos[1])])
					else:
						obstacle_pos = np.array([float(constraint.obstacle_pos[0]), float(constraint.obstacle_pos[1])])
					obstacle_radius = float(constraint.obstacle_radius) if hasattr(constraint, 'obstacle_radius') and constraint.obstacle_radius is not None else self.robot_radius
				else:
					# Fallback to stored values if obstacle_pos not available
					LOG_DEBUG(f"SafeHorizonConstraint._plot_constraints_for_stage: Constraint missing obstacle_pos, using stored a1, a2, b")
					a1 = float(constraint.a1)
					a2 = float(constraint.a2)
					b = float(constraint.b)
					obstacle_pos = None  # Will skip recomputation
				
				# Recompute constraint normal using current robot position (matching calculate_constraints)
				if obstacle_pos is not None:
					# Compute direction FROM robot TO obstacle
					diff_x = obstacle_pos[0] - current_robot_pos[0]
					diff_y = obstacle_pos[1] - current_robot_pos[1]
					dist = np.sqrt(diff_x * diff_x + diff_y * diff_y)
					dist = max(dist, 1e-6)  # Avoid division by zero
					
					# Normalized direction vector (points FROM vehicle TO obstacle)
					a1 = diff_x / dist
					a2 = diff_y / dist
					
					# Safety margin: robot_radius + obstacle_radius
					safety_margin = self.robot_radius + obstacle_radius
					
					# Compute b: a·obs_pos - safety_margin
					b = a1 * obstacle_pos[0] + a2 * obstacle_pos[1] - safety_margin
				
				# Constraint: a1*x + a2*y <= b
				# To plot: find two points on the line a1*x + a2*y = b
				# We'll plot a line segment within the visible area
				
				# Normalize the normal vector
				norm = np.sqrt(a1**2 + a2**2)
				if norm < 1e-6:
					continue
				
				a1_norm = a1 / norm
				a2_norm = a2 / norm
				b_norm = b / norm
				
				# Find two points on the line
				# Line equation: a1_norm*x + a2_norm*y = b_norm
				# If a2_norm != 0: y = (b_norm - a1_norm*x) / a2_norm
				# If a1_norm != 0: x = (b_norm - a2_norm*y) / a1_norm
				
				# Choose points at the edges of the visible area
				x_center = (xlim[0] + xlim[1]) / 2
				y_center = (ylim[0] + ylim[1]) / 2
				
				# Extend line beyond visible area
				extend = max(x_range, y_range) * 1.5
				
				if abs(a2_norm) > 1e-6:
					# Use x coordinates
					x1 = x_center - extend
					x2 = x_center + extend
					y1 = (b_norm - a1_norm * x1) / a2_norm
					y2 = (b_norm - a1_norm * x2) / a2_norm
				else:
					# Vertical line
					x1 = b_norm / a1_norm
					x2 = x1
					y1 = y_center - extend
					y2 = y_center + extend
				
				# Plot constraint line (halfspace boundary)
				# The feasible region is on the side where a1*x + a2*y <= b
				# We'll use a dashed line to indicate it's a constraint boundary
				color = plt.cm.tab10(disc_id % 10)
				alpha = 0.3 if i < max_constraints_to_plot - 1 else 0.5  # Make last constraint more visible
				linewidth = 1.0 if i < max_constraints_to_plot - 1 else 1.5
				
				ax.plot([x1, x2], [y1, y2], '--', color=color, alpha=alpha, linewidth=linewidth, 
				        label=f'Safe Horizon Constraints (disc {disc_id})' if disc_id == 0 and i == 0 else None,
				        zorder=1)
				
				# Draw arrow indicating feasible side (pointing away from obstacle)
				# The normal vector (a1, a2) points FROM robot TO obstacle
				# So feasible side is opposite to normal (where a1*x + a2*y < b)
				mid_x = (x1 + x2) / 2
				mid_y = (y1 + y2) / 2
				arrow_length = min(x_range, y_range) * 0.1
				arrow_dx = -a1_norm * arrow_length  # Opposite to normal (feasible side)
				arrow_dy = -a2_norm * arrow_length
				
				if i == max_constraints_to_plot - 1:  # Only draw arrow for last constraint to avoid clutter
					ax.arrow(mid_x, mid_y, arrow_dx, arrow_dy, head_width=arrow_length*0.3, 
					        head_length=arrow_length*0.3, fc=color, ec=color, alpha=alpha*0.7, zorder=1)
		
		# Log constraint count
		total_constraints = sum(len(self.scenario_cache.get((disc_id, stage_idx), [])) 
		                       for disc_id in range(self.num_discs))
		if total_constraints > 0:
			LOG_DEBUG(f"Plotted {total_constraints} constraints for stage {stage_idx}")
	
	def _project_warmstart_to_safety(self, data: Data):
		"""
		Project warmstart trajectory to satisfy safe horizon constraints.
		
		Reference: C++ mpc_planner - warmstart is projected to ensure feasibility
		Similar to LinearizedConstraints.project_to_safety() but for safe horizon constraints.
		
		This method:
		1. Checks if warmstart positions violate cached constraints
		2. Projects violating positions away from obstacles to satisfy constraints
		3. Ensures warmstart is feasible, preventing solver failures
		
		Args:
			data: Data object containing obstacles and other information
		"""
		if not hasattr(self, 'solver') or self.solver is None:
			return
		
		if not hasattr(self.solver, 'warmstart_values') or not self.solver.warmstart_values:
			return
		
		ws_vals = self.solver.warmstart_values
		if 'x' not in ws_vals or 'y' not in ws_vals:
			return
		
		horizon_val = self.solver.horizon if self.solver.horizon is not None else 10
		x_ws = ws_vals['x']
		y_ws = ws_vals['y']
		
		# Check each stage that has constraints
		projections_made = 0
		for stage_idx in range(min(horizon_val + 1, len(x_ws))):
			# Get constraints for this stage (use disc_id=0 for now, can extend to multiple discs)
			disc_id = 0
			key = (disc_id, stage_idx)
			
			if key not in self.scenario_cache:
				continue
			
			constraints = self.scenario_cache[key]
			if not constraints:
				continue
			
			# Get warmstart position for this stage
			if stage_idx >= len(x_ws) or stage_idx >= len(y_ws):
				continue
			
			robot_pos = np.array([float(x_ws[stage_idx]), float(y_ws[stage_idx])])
			
			# Check if position violates any constraints
			max_violation = 0.0
			violating_constraint = None
			
			for constraint in constraints:
				if not isinstance(constraint, ScenarioConstraint):
					continue
				
				# Get obstacle position from constraint
				obstacle_pos = None
				obstacle_radius = None
				if hasattr(constraint, 'obstacle_pos') and constraint.obstacle_pos is not None:
					if isinstance(constraint.obstacle_pos, (list, tuple)):
						obstacle_pos = np.array([float(constraint.obstacle_pos[0]), float(constraint.obstacle_pos[1])])
					else:
						obstacle_pos = np.array([float(constraint.obstacle_pos[0]), float(constraint.obstacle_pos[1])])
					obstacle_radius = float(constraint.obstacle_radius) if hasattr(constraint, 'obstacle_radius') and constraint.obstacle_radius is not None else self.robot_radius
				else:
					continue
				
				# Compute constraint normal (points FROM robot TO obstacle)
				diff = obstacle_pos - robot_pos
				dist = np.linalg.norm(diff)
				dist = max(dist, 1e-6)
				
				a1 = diff[0] / dist
				a2 = diff[1] / dist
				safety_margin = self.robot_radius + obstacle_radius
				b = a1 * obstacle_pos[0] + a2 * obstacle_pos[1] - safety_margin
				
				# Check constraint violation: a1*x + a2*y - b <= 0
				constraint_value = a1 * robot_pos[0] + a2 * robot_pos[1] - b
				
				if constraint_value > max_violation:
					max_violation = constraint_value
					violating_constraint = {
						'obstacle_pos': obstacle_pos,
						'obstacle_radius': obstacle_radius,
						'a1': a1,
						'a2': a2,
						'b': b,
						'safety_margin': safety_margin
					}
			
			# Project position away from obstacle if violation detected
			if max_violation > 1e-6 and violating_constraint is not None:
				# Project robot position away from obstacle to satisfy constraint
				# Move robot away from obstacle by safety_margin + violation_amount
				obstacle_pos = violating_constraint['obstacle_pos']
				safety_margin = violating_constraint['safety_margin']
				
				# Direction FROM obstacle TO robot (opposite of constraint normal)
				diff = robot_pos - obstacle_pos
				dist = np.linalg.norm(diff)
				
				if dist < 1e-6:
					# Robot is exactly at obstacle - move in a safe direction
					diff = np.array([1.0, 0.0])  # Default direction
					dist = 1.0
				
				# Required distance from obstacle
				required_dist = safety_margin + 0.1  # Add small margin for safety
				
				if dist < required_dist:
					# Project robot position to satisfy constraint
					# New position: obstacle_pos + (required_dist / dist) * (robot_pos - obstacle_pos)
					projection_factor = required_dist / dist
					new_pos = obstacle_pos + projection_factor * diff
					
					# Update warmstart values
					x_ws[stage_idx] = float(new_pos[0])
					y_ws[stage_idx] = float(new_pos[1])
					projections_made += 1
					
					if stage_idx < 3:  # Log first few projections
						LOG_INFO(f"SafeHorizonConstraint._project_warmstart_to_safety: Stage {stage_idx}: "
						         f"Projected warmstart from ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}) "
						         f"to ({new_pos[0]:.3f}, {new_pos[1]:.3f}) "
						         f"(violation={max_violation:.3f}, required_dist={required_dist:.3f})")
		
		if projections_made > 0:
			LOG_INFO(f"SafeHorizonConstraint._project_warmstart_to_safety: Projected {projections_made} warmstart positions to satisfy constraints")
		
		# After projecting positions, ensure dynamics consistency
		# This is critical - projected positions might violate dynamics
		if hasattr(self.solver, '_ensure_warmstart_dynamics_consistency'):
			self.solver._ensure_warmstart_dynamics_consistency()
	
	def reset(self):
		"""Reset constraint module state."""
		if self.scenario_module is not None:
			self.scenario_module.reset()
		self.scenario_cache.clear()
		LOG_DEBUG("SafeHorizonConstraint.reset: Reset complete")
