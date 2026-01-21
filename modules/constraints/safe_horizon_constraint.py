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
	
	def __init__(self, settings=None):
		super().__init__(settings=settings)
		self.name = "safe_horizon_constraint"
		
		# Configuration parameters
		self.num_discs = int(self.get_config_value("safe_horizon_constraints.num_discs", 1))
		self.robot_radius = float(self.get_config_value("robot.radius", 0.5))
		self.horizon_length = int(self.get_config_value("horizon", 10))
		self.timestep = float(self.get_config_value("timestep", 0.1))
		
		# Get halfspace offset from obstacles (additional safety margin for constraints)
		# This offset is added to the safe_distance when computing constraint b values
		# Matching linearized_constraints.py line 39
		self.halfspace_offset = self.get_config_value("linearized_constraints.halfspace_offset", 0.0)
		if self.halfspace_offset is None:
			self.halfspace_offset = 0.0
		else:
			self.halfspace_offset = float(self.halfspace_offset)
		LOG_DEBUG(f"SafeHorizonConstraint: halfspace_offset={self.halfspace_offset:.3f}m")
		
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
		# CRITICAL: max_constraints_per_disc must be >= number of obstacles to ensure all obstacles are covered
		# Reference: C++ mpc_planner - typically uses 3-5 constraints per disc to cover multiple obstacles
		self.max_constraints_per_disc = int(self.get_config_value("safe_horizon_constraints.max_constraints_per_disc", 5))  # Increased to cover all obstacles
		self.num_removal = int(self.get_config_value("safe_horizon_constraints.num_removal", 0))  # Scenarios to remove with big-M
		
		# Initialize scenario module (will be set in update when solver is available)
		self.scenario_module: Optional[SafeHorizonModule] = None
		
		# Cache for scenarios per stage (scenarios are independent of robot position)
		self.scenario_cache: Dict[int, List] = {}  # Key: (disc_id, stage_idx), Value: List[Scenario]
		
		# Diagnostics output (optional, enabled via enable_diagnostics flag)
		self.diagnostics = None
		self.enable_diagnostics = bool(self.get_config_value("safe_horizon_constraints.enable_diagnostics", False))
		self.safe_horizon_steps = int(self.get_config_value("safe_horizon_constraints.safe_horizon_steps", 8))

		# Scenario optimization parameters
		self.epsilon_p = float(self.get_config_value("safe_horizon_constraints.epsilon_p", 0.05))
		self.beta = float(self.get_config_value("safe_horizon_constraints.beta", 0.01))
		self.n_bar = int(self.get_config_value("safe_horizon_constraints.n_bar", 10))
		self.num_removal = int(self.get_config_value("safe_horizon_constraints.num_removal", 1))

		# Adaptive mode sampling configuration (following guide.md)
		# When enabled, scenarios are sampled based on observed mode history
		self.enable_adaptive_mode_sampling = bool(self.get_config_value(
			"safe_horizon_constraints.enable_adaptive_mode_sampling", False))
		self.mode_weight_type = str(self.get_config_value(
			"safe_horizon_constraints.mode_weight_type", "frequency"))  # uniform, recency, frequency
		self.mode_recency_decay = float(self.get_config_value(
			"safe_horizon_constraints.mode_recency_decay", 0.9))
		self.mode_prior_type = str(self.get_config_value(
			"safe_horizon_constraints.mode_prior_type", "constant"))  # constant (C1) or switching (C2)

		# Compute required sample size using paper formula
		from modules.constraints.scenario_utils.math_utils import compute_sample_size
		self.num_scenarios = compute_sample_size(
			epsilon_p=self.epsilon_p,
			beta=self.beta,
			n_bar=self.n_bar,
			num_removal=self.num_removal
    	)

		LOG_INFO(f"SafeHorizonConstraint: Computed sample size S={self.num_scenarios} "
				f"(ε={self.epsilon_p}, β={self.beta}, n_bar={self.n_bar}, R={self.num_removal})")

		if self.enable_adaptive_mode_sampling:
			LOG_INFO(f"SafeHorizonConstraint: Adaptive mode sampling ENABLED "
					f"(weight_type={self.mode_weight_type}, prior_type={self.mode_prior_type})")

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
					"enable_outlier_removal": True,
					"timestep": self.timestep,
					# Adaptive mode sampling configuration (following guide.md)
					"enable_adaptive_mode_sampling": self.enable_adaptive_mode_sampling,
					"mode_weight_type": self.mode_weight_type,
					"mode_recency_decay": self.mode_recency_decay,
					"mode_prior_type": self.mode_prior_type
				}

				self.scenario_module = SafeHorizonModule(self.solver, config)
				
				# CRITICAL VALIDATION: Check if max_constraints_per_disc is sufficient for the number of obstacles
				# Reference: C++ scenario_module requires enough constraints to cover all obstacles
				if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles:
					num_obstacles = len(data.dynamic_obstacles)
					if self.max_constraints_per_disc < num_obstacles:
						LOG_WARN(f"⚠️  CRITICAL: max_constraints_per_disc ({self.max_constraints_per_disc}) < num_obstacles ({num_obstacles})!")
						LOG_WARN(f"    This may cause collisions as not all obstacles can be constrained!")
						LOG_WARN(f"    Recommended: Set max_constraints_per_disc >= {num_obstacles} in config")
						# Raise an error to prevent the test from continuing with insufficient constraints
						raise ValueError(
							f"SafeHorizonConstraint: max_constraints_per_disc ({self.max_constraints_per_disc}) "
							f"must be >= number of obstacles ({num_obstacles}) to ensure collision avoidance. "
							f"Set safe_horizon_constraints.max_constraints_per_disc >= {num_obstacles} in config."
						)
					else:
						LOG_INFO(f"SafeHorizonConstraint: max_constraints_per_disc ({self.max_constraints_per_disc}) >= num_obstacles ({num_obstacles}) ✓")
				
				# Initialize diagnostics if enabled (after scenario module is created)
				if self.enable_diagnostics:
					from modules.constraints.safe_horizon_diagnostics import SafeHorizonDiagnostics
					# Get output folder from solver if available
					output_folder = getattr(self.solver, 'output_folder', None)
					if output_folder is None:
						# Try to get from data or use default
						output_folder = getattr(data, 'output_folder', None)
					if output_folder is None or not isinstance(output_folder, str):
						# Use default if still None or not a string
						output_folder = 'test_outputs'
					# Ensure output folder exists
					import os
					os.makedirs(output_folder, exist_ok=True)
					self.diagnostics = SafeHorizonDiagnostics(output_folder)
					self.scenario_module.diagnostics = self.diagnostics
					LOG_INFO(f"SafeHorizonConstraint: Diagnostics enabled, output folder: {output_folder}")
			
			# CRITICAL: Update diagnostics output folder if it changed (e.g., new test run)
			# This ensures diagnostics are always saved to the correct test output folder
			if self.enable_diagnostics:
				current_output_folder = getattr(self.solver, 'output_folder', None) if hasattr(self, 'solver') and self.solver is not None else None
				if current_output_folder is None:
					current_output_folder = getattr(data, 'output_folder', None)
				if current_output_folder is None or not isinstance(current_output_folder, str):
					current_output_folder = 'test_outputs'
				
				# Initialize diagnostics if not already done
				if self.diagnostics is None:
					from modules.constraints.safe_horizon_diagnostics import SafeHorizonDiagnostics
					import os
					os.makedirs(current_output_folder, exist_ok=True)
					self.diagnostics = SafeHorizonDiagnostics(current_output_folder)
					self.scenario_module.diagnostics = self.diagnostics
					LOG_INFO(f"SafeHorizonConstraint: Diagnostics initialized, output folder: {current_output_folder}")
				# Update output folder if it changed
				elif self.diagnostics.output_folder != current_output_folder:
					LOG_INFO(f"SafeHorizonConstraint: Updating diagnostics output folder from {self.diagnostics.output_folder} to {current_output_folder}")
					self.diagnostics.output_folder = current_output_folder
					import os
					os.makedirs(current_output_folder, exist_ok=True)
				
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
			
			# Record scenario sampling for diagnostics (before optimization)
			if self.diagnostics is not None:
				from planning.types import propagate_obstacles
				# Ensure obstacles are propagated before recording
				if data.has("dynamic_obstacles") and data.dynamic_obstacles:
					propagate_obstacles(data, dt=self.timestep, horizon=self.horizon_length)
				self.diagnostics.record_scenario_sampling(
					self.scenario_module, 
					data.dynamic_obstacles if data.has("dynamic_obstacles") else [],
					self.timestep
				)
				self.diagnostics.record_obstacle_trajectories(
					data.dynamic_obstacles if data.has("dynamic_obstacles") else [],
					self.horizon_length,
					self.timestep
				)
			
			# CRITICAL: Reset scenario module state before optimization to prevent compounding
			# This ensures constraints are recomputed from scratch at each MPC iteration
			# Reference: C++ mpc_planner - scenario module is reset at each iteration
			self.scenario_module.reset()
			
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
					# CRITICAL FIX: Removed unused variables current_x/current_y that referenced undefined 'x'
					constraints = self.scenario_module.disc_manager[disc_id].get_constraints_for_step(stage_idx)
					key = (disc_id, stage_idx)
					self.scenario_cache[key] = constraints
					total_cached += len(constraints)
			
			# Log constraint info
			LOG_INFO(f"SafeHorizonConstraint.update: Cached {total_cached} optimized constraints across {len(self.scenario_cache)} (disc, stage) keys")
			
			# Add verification summary for diagnostics
			if self.diagnostics is not None:
				self.diagnostics.add_verification_summary(self.scenario_module)
			
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
			# CRITICAL: Apply constraints to early stages (0-4) to allow planning while avoiding over-constraining
			# Reference: C++ mpc_planner - constraints are typically applied to first few stages only
			# The constraints are computed symbolically at each stage using predicted robot position
			# CRITICAL: Constraint horizon must be long enough to plan avoidance maneuvers
			# At 3 m/s velocity and 0.1s timestep, each stage is 0.3m of lookahead
			# With obstacles moving at ~1.7 m/s, we need at least 8 stages (0.8s) to react
			# Reference: C++ mpc_planner - typically uses full horizon for collision constraints
			max_stage_for_constraints = min(self.safe_horizon_steps, self.horizon_length - 1)  # Apply constraints to stages 0-8 for better lookahead
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
									safety_margin = self.robot_radius + obstacle_radius + self.halfspace_offset
									
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
				
				# CRITICAL: Get obstacle position from the scenario's trajectory at the current stage
				# The constraint was formulated from a scenario, which has a trajectory representing
				# the obstacle's predicted positions at different time steps. We must use the scenario's
				# trajectory position at this stage, not from data.dynamic_obstacles, to maintain consistency.
				# Reference: C++ mpc_planner - constraints use scenario trajectory positions
				obstacle_pos = None
				obstacle_radius = None
				
				# First, try to get obstacle position from the scenario's trajectory at this stage
				# The scenario module stores scenarios with trajectories, and constraints reference scenarios
				if self.scenario_module is not None and hasattr(self.scenario_module, 'scenarios'):
					scenario_idx = getattr(constraint, 'scenario_idx', None)
					obstacle_idx = getattr(constraint, 'obstacle_idx', None)
					if scenario_idx is not None and obstacle_idx is not None:
						# Find the scenario that matches this constraint
						for scenario in self.scenario_module.scenarios:
							if (hasattr(scenario, 'idx_') and int(scenario.idx_) == int(scenario_idx) and
							    hasattr(scenario, 'obstacle_idx_') and int(scenario.obstacle_idx_) == int(obstacle_idx)):
								# Get obstacle position from scenario's trajectory at this stage
								if hasattr(scenario, 'trajectory') and scenario.trajectory and stage_idx < len(scenario.trajectory):
									obstacle_pos = np.array([float(scenario.trajectory[stage_idx][0]), float(scenario.trajectory[stage_idx][1])])
									obstacle_radius = float(scenario.radius) if hasattr(scenario, 'radius') else self.robot_radius
									LOG_DEBUG(f"SafeHorizonConstraint: Using scenario {scenario_idx} trajectory position at stage {stage_idx}: ({obstacle_pos[0]:.3f}, {obstacle_pos[1]:.3f})")
									break
								elif hasattr(scenario, 'position'):
									# Fallback: use initial scenario position if trajectory not available
									obstacle_pos = np.array([float(scenario.position[0]), float(scenario.position[1])])
									obstacle_radius = float(scenario.radius) if hasattr(scenario, 'radius') else self.robot_radius
									LOG_DEBUG(f"SafeHorizonConstraint: Using scenario {scenario_idx} initial position (no trajectory): ({obstacle_pos[0]:.3f}, {obstacle_pos[1]:.3f})")
									break
				
				# Fallback: use cached obstacle position from constraint (from when it was formulated)
				# This is the obstacle position at the time step when the constraint was created
				if obstacle_pos is None:
					if hasattr(constraint, 'obstacle_pos') and constraint.obstacle_pos is not None:
						# Convert to numpy array if needed
						if isinstance(constraint.obstacle_pos, (list, tuple)):
							obstacle_pos = np.array([float(constraint.obstacle_pos[0]), float(constraint.obstacle_pos[1])])
						else:
							obstacle_pos = np.array([float(constraint.obstacle_pos[0]), float(constraint.obstacle_pos[1])])
						obstacle_radius = float(constraint.obstacle_radius) if hasattr(constraint, 'obstacle_radius') and constraint.obstacle_radius is not None else self.robot_radius
						LOG_DEBUG(f"SafeHorizonConstraint: Using cached obstacle position from constraint (scenario_idx={getattr(constraint, 'scenario_idx', 'N/A')}, stage={getattr(constraint, 'time_step', 'N/A')})")
					else:
						# Obstacle position must be available - this is a critical error
						LOG_WARN(f"SafeHorizonConstraint: Cannot get obstacle position at stage {stage_idx}, disc {disc_id}, "
						         f"obstacle_idx={getattr(constraint, 'obstacle_idx', 'N/A')}, scenario_idx={getattr(constraint, 'scenario_idx', 'N/A')}, "
						         f"cannot compute symbolically")
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
				
				# Safety margin: robot_radius + obstacle_radius + halfspace_offset
				# Matching C++: safe_distance = radius + CONFIG["robot_radius"] + halfspace_offset
				# Matching linearized_constraints.py line 931: safe_distance = robot_radius + target_obstacle_radius + self.halfspace_offset
				safety_margin = self.robot_radius + obstacle_radius + self.halfspace_offset
				
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
				# CRITICAL: Pass current robot position and data so constraints are recomputed correctly
				# and obstacle positions are taken from data at current stage
				self.module._plot_constraints_for_stage(ax, stage_idx, current_robot_pos=current_robot_pos, data=data)
				
				# Plot linearized halfspaces for ALL sampled scenario obstacles (not just support set)
				# This shows all the constraints that could be applied, not just the ones selected for optimization
				self.module._plot_all_scenario_halfspaces(ax, stage_idx, current_robot_pos=current_robot_pos, data=data)
				
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
				
				# Limit trajectories to show - ensure diversity by sampling from each obstacle
				# Group scenarios by obstacle and mode for better diversity
				trajectory_keys = list(scenario_trajectories.keys())

				# Get mode info for each scenario to enable mode-diverse sampling
				mode_info = {}  # key -> mode_id
				if hasattr(self.module, 'scenario_module') and self.module.scenario_module is not None:
					for scenario in self.module.scenario_module.scenarios:
						if hasattr(scenario, 'obstacle_idx_') and hasattr(scenario, 'idx_'):
							key = (scenario.obstacle_idx_, scenario.idx_)
							if hasattr(scenario, 'mode_id'):
								mode_info[key] = scenario.mode_id

				# Group keys by obstacle, then by mode
				keys_by_obstacle_mode = {}  # (obstacle_idx, mode_id) -> [keys]
				keys_by_obstacle = {}
				for key in trajectory_keys:
					obstacle_idx = key[0]
					mode_id = mode_info.get(key, 'unknown')

					if obstacle_idx not in keys_by_obstacle:
						keys_by_obstacle[obstacle_idx] = []
					keys_by_obstacle[obstacle_idx].append(key)

					obs_mode_key = (obstacle_idx, mode_id)
					if obs_mode_key not in keys_by_obstacle_mode:
						keys_by_obstacle_mode[obs_mode_key] = []
					keys_by_obstacle_mode[obs_mode_key].append(key)

				# Sample diverse trajectories: pick from different modes for each obstacle
				selected_keys = []
				num_obstacles = len(keys_by_obstacle)
				per_obstacle = max(1, max_trajectories_to_show // max(1, num_obstacles))

				for obstacle_idx in sorted(keys_by_obstacle.keys()):
					# Get modes for this obstacle
					obstacle_modes = {}
					for (obs_idx, mode_id), keys in keys_by_obstacle_mode.items():
						if obs_idx == obstacle_idx:
							obstacle_modes[mode_id] = keys

					# Sample from each mode proportionally
					if len(obstacle_modes) > 0:
						per_mode = max(1, per_obstacle // len(obstacle_modes))
						for mode_id, keys in obstacle_modes.items():
							if len(keys) <= per_mode:
								selected_keys.extend(keys)
							else:
								# Sample spread across the range
								step = max(1, len(keys) // per_mode)
								sampled = keys[::step][:per_mode]
								selected_keys.extend(sampled)

				trajectory_keys = selected_keys[:max_trajectories_to_show]

				# Log mode diversity in shown trajectories
				shown_modes = {}
				for key in trajectory_keys:
					mode = mode_info.get(key, 'unknown')
					shown_modes[mode] = shown_modes.get(mode, 0) + 1

				LOG_INFO(f"SafeHorizonConstraintsVisualizer: Showing {len(trajectory_keys)} trajectories with mode diversity: {shown_modes}")
				
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
				# Get predicted trajectory from solver - try multiple methods
				if hasattr(self.module, 'solver') and self.module.solver is not None:
					try:
						x_traj = []
						y_traj = []
						
						# Method 1: Try to get directly from warmstart_values (most reliable fallback)
						if hasattr(self.module.solver, 'warmstart_values') and self.module.solver.warmstart_values:
							ws = self.module.solver.warmstart_values
							if 'x' in ws and 'y' in ws:
								try:
									x_vals = ws['x']
									y_vals = ws['y']
									for i in range(min(len(x_vals), len(y_vals))):
										x_traj.append(float(x_vals[i]))
										y_traj.append(float(y_vals[i]))
									if len(x_traj) > 1:
										LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: Got trajectory from warmstart_values: {len(x_traj)} points")
								except Exception as e:
									LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: Error extracting warmstart: {e}")
						
						# Method 2: Try to get from reference trajectory (if warmstart didn't work)
						if len(x_traj) <= 1 and hasattr(self.module.solver, 'get_reference_trajectory'):
							traj = self.module.solver.get_reference_trajectory()
							if traj is not None:
								if hasattr(traj, 'get_states'):
									states = traj.get_states()
									if states and len(states) > 0:
										LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: Got {len(states)} states from trajectory")
										x_traj = []  # Reset in case partial extraction failed
										y_traj = []
										for state in states:
											try:
												if hasattr(state, 'get') and hasattr(state, 'has'):
													# Check if state has 'x' and 'y' using has() method
													# State.get() returns 0.0 if variable not found, so check has() first
													if state.has('x') and state.has('y'):
														try:
															x_val = state.get('x')
															y_val = state.get('y')
														except (KeyError, AttributeError) as get_err:
															LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: Error getting x/y from state: {get_err}, state type={type(state)}")
															continue
														
														if x_val is not None and y_val is not None:
															try:
																# Handle CasADi symbolic values - try to evaluate if needed
																# State values from solution.value() should be numeric, but handle edge cases
																if isinstance(x_val, (int, float, np.number)):
																	x_traj.append(float(x_val))
																elif hasattr(x_val, '__float__'):
																	x_traj.append(float(x_val))
																elif hasattr(x_val, 'value'):  # CasADi DM/MX
																	x_traj.append(float(x_val.value()))
																else:
																	# Try direct conversion
																	x_traj.append(float(x_val))
																
																if isinstance(y_val, (int, float, np.number)):
																	y_traj.append(float(y_val))
																elif hasattr(y_val, '__float__'):
																	y_traj.append(float(y_val))
																elif hasattr(y_val, 'value'):  # CasADi DM/MX
																	y_traj.append(float(y_val.value()))
																else:
																	# Try direct conversion
																	y_traj.append(float(y_val))
															except (ValueError, TypeError, AttributeError) as e:
																LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: Error converting state values: {e}, x_val type={type(x_val)}, y_val type={type(y_val)}")
																continue
													else:
														# State doesn't have 'x' or 'y', skip
														LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: State missing x or y: has('x')={state.has('x') if hasattr(state, 'has') else 'N/A'}, has('y')={state.has('y') if hasattr(state, 'has') else 'N/A'}")
														continue
												elif isinstance(state, (list, tuple, np.ndarray)) and len(state) >= 2:
													x_traj.append(float(state[0]))
													y_traj.append(float(state[1]))
												else:
													# State format not recognized, skip
													LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: State format not recognized: type={type(state)}")
													continue
											except (KeyError, AttributeError) as e:
												# KeyError means 'x' key doesn't exist, AttributeError means state doesn't have expected methods
												LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: State format issue: {e}, state type={type(state)}")
												continue
											except Exception as e:
												LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: Error extracting state: {e}, state type={type(state)}")
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
											LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: Method 2 trajectory has insufficient points: x={len(x_traj)}, y={len(y_traj)}")
									else:
										LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: Trajectory has no states (empty)")
								else:
									LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: Trajectory has no get_states method")
							else:
								LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: get_reference_trajectory returned None")
						
						# Draw trajectory if we got valid points from either method
						if len(x_traj) > 1 and len(y_traj) > 1 and len(x_traj) == len(y_traj):
							LOG_INFO(f"SafeHorizonConstraintsVisualizer: Drawing predicted trajectory with {len(x_traj)} points")
							# Draw predicted trajectory - bold blue line
							ax.plot(x_traj, y_traj, '-', color='blue', linewidth=3.0, 
							        alpha=0.9, label='Predicted Trajectory', zorder=20)
							# Mark start with green square
							ax.scatter([x_traj[0]], [y_traj[0]], c='lime', s=120, 
							          marker='o', label='Trajectory Start', zorder=21, edgecolors='black', linewidths=1.5)
							# Mark end with red star
							ax.scatter([x_traj[-1]], [y_traj[-1]], c='red', s=150, 
							          marker='*', label='Trajectory End', zorder=21, edgecolors='black', linewidths=1.5)
						else:
							LOG_DEBUG(f"SafeHorizonConstraintsVisualizer: No valid trajectory to draw (x={len(x_traj)}, y={len(y_traj)})")
					except Exception as e:
						LOG_DEBUG(f"Could not visualize selected trajectory: {e}")
		
		return SafeHorizonConstraintsVisualizer(self)
	
	def _plot_constraints_for_stage(self, ax, stage_idx, current_robot_pos=None, data=None):
		"""
		Plot constraints (halfspaces) applied at a specific stage.
		
		CRITICAL: Constraints are recomputed at the current robot position to show correct halfspace orientation.
		The halfspaces should rotate as the vehicle moves, pointing FROM vehicle TO obstacle.
		Obstacle positions are taken from data.dynamic_obstacles at the current stage to ensure they move with obstacles.
		
		Args:
			ax: Matplotlib axes to plot on
			stage_idx: Stage index to plot constraints for
			current_robot_pos: Current robot position (x, y) for recomputing constraints. If None, uses (0, 0).
			data: Data object containing dynamic obstacles with predictions (optional, for getting current obstacle positions)
		"""
		try:
			import matplotlib.pyplot as plt
			import numpy as np
		except Exception:
			return
		
		if not self.scenario_cache:
			LOG_DEBUG(f"SafeHorizonConstraint._plot_constraints_for_stage: scenario_cache is empty for stage {stage_idx} (cache keys: {list(self.scenario_cache.keys())})")
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
				if stage_idx == 0 and disc_id == 0:  # Only log for first disc/stage to avoid spam
					LOG_DEBUG(f"SafeHorizonConstraint._plot_constraints_for_stage: Key {key} not in scenario_cache (available keys: {list(self.scenario_cache.keys())[:10]})")
				continue
			
			constraints = self.scenario_cache[key]
			if not constraints:
				if stage_idx == 0 and disc_id == 0:  # Only log for first disc/stage to avoid spam
					LOG_DEBUG(f"SafeHorizonConstraint._plot_constraints_for_stage: Key {key} found but constraints list is empty")
				continue
			
			# CRITICAL: Only plot ONE constraint per obstacle to avoid duplicate halfspace lines
			# Multiple constraints for the same obstacle would create overlapping lines
			# Track which obstacles we've already plotted for this disc/stage
			plotted_obstacles = set()
			
			if stage_idx == 0 and disc_id == 0:  # Debug logging for first disc/stage
				LOG_DEBUG(f"SafeHorizonConstraint._plot_constraints_for_stage: Found {len(constraints)} constraints for key {key}")
			
			# Color palette for different obstacles (matching linearized constraints visualization)
			obstacle_colors = ['red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
			
			# Plot each constraint as a halfspace line (one per obstacle)
			for i, constraint in enumerate(constraints):
				if not isinstance(constraint, ScenarioConstraint):
					continue
				
				# Get obstacle ID from constraint (for color assignment and duplicate detection)
				# ScenarioConstraint stores obstacle_idx which maps to the original obstacle
				if hasattr(constraint, 'obstacle_idx') and constraint.obstacle_idx is not None:
					obstacle_id = int(constraint.obstacle_idx)
				else:
					# Fallback: use constraint index
					obstacle_id = i % len(obstacle_colors)
				
				# CRITICAL: Skip if we've already plotted a halfspace for this obstacle
				# This prevents duplicate/overlapping halfspace lines for the same obstacle
				if obstacle_id in plotted_obstacles:
					continue
				plotted_obstacles.add(obstacle_id)
				
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
					LOG_DEBUG(f"SafeHorizonConstraint._plot_constraints_for_stage: Constraint {i} missing obstacle_pos, using stored a1, a2, b (obstacle_idx={obstacle_id})")
					a1 = float(constraint.a1)
					a2 = float(constraint.a2)
					b = float(constraint.b)
					obstacle_pos = None  # Will skip recomputation
				
				# CRITICAL: Get obstacle position at CURRENT stage from obstacle's predicted trajectory
				# This ensures visualization moves with obstacles (matching calculate_constraints logic)
				# First try to get from data.dynamic_obstacles if available
				obstacle_pos_current = None
				obstacle_radius_current = None
				
				# Try to get obstacle position from data at current stage (matching calculate_constraints)
				if data is not None and hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles:
					obstacle_idx = getattr(constraint, 'obstacle_idx', None)
					if obstacle_idx is not None and obstacle_idx < len(data.dynamic_obstacles):
						obstacle = data.dynamic_obstacles[obstacle_idx]
						if obstacle is not None:
							# Get obstacle position at current stage from prediction
							if hasattr(obstacle, 'prediction') and obstacle.prediction is not None:
								if hasattr(obstacle.prediction, 'steps') and obstacle.prediction.steps:
									if stage_idx < len(obstacle.prediction.steps):
										pred_step = obstacle.prediction.steps[stage_idx]
										if hasattr(pred_step, 'position') and pred_step.position is not None:
											obstacle_pos_current = np.array([float(pred_step.position[0]), float(pred_step.position[1])])
											obstacle_radius_current = float(pred_step.radius) if hasattr(pred_step, 'radius') and pred_step.radius is not None else (float(obstacle.radius) if hasattr(obstacle, 'radius') else self.robot_radius)
							
							# Fallback: use current obstacle position if prediction not available
							if obstacle_pos_current is None and hasattr(obstacle, 'position') and obstacle.position is not None:
								obstacle_pos_current = np.array([float(obstacle.position[0]), float(obstacle.position[1])])
								obstacle_radius_current = float(obstacle.radius) if hasattr(obstacle, 'radius') else self.robot_radius
				
				# Fallback: use stored obstacle_pos from constraint (for backward compatibility)
				if obstacle_pos_current is None and obstacle_pos is not None:
					obstacle_pos_current = obstacle_pos.copy()
					obstacle_radius_current = obstacle_radius
				
				# Recompute constraint normal using current robot position (matching calculate_constraints)
				if obstacle_pos_current is not None:
					# Compute direction FROM robot TO obstacle
					diff_x = obstacle_pos_current[0] - current_robot_pos[0]
					diff_y = obstacle_pos_current[1] - current_robot_pos[1]
					dist = np.sqrt(diff_x * diff_x + diff_y * diff_y)
					dist = max(dist, 1e-6)  # Avoid division by zero
					
					# Normalized direction vector (points FROM vehicle TO obstacle)
					a1 = diff_x / dist
					a2 = diff_y / dist
					
					# Safety margin: robot_radius + obstacle_radius + halfspace_offset
					safety_margin = self.robot_radius + obstacle_radius_current + self.halfspace_offset
					
					# Compute b: a·obs_pos - safety_margin
					b = a1 * obstacle_pos_current[0] + a2 * obstacle_pos_current[1] - safety_margin
				
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
				
				# Plot constraint line (halfspace boundary) - similar to linearized constraints
				# The feasible region is on the side where a1*x + a2*y <= b
				# Use obstacle-specific color (matching linearized constraints visualization)
				color = obstacle_colors[obstacle_id % len(obstacle_colors)]
				alpha = 0.6  # Match linearized constraints alpha
				linewidth = 1.0
				
				# Calculate shorter line segment (similar to linearized constraints)
				# If obstacle position is available, draw line segment near obstacle
				if obstacle_pos_current is not None:
					# Calculate line center point (where constraint line intersects vehicle-to-obstacle line)
					vehicle_to_obstacle = obstacle_pos_current - current_robot_pos
					vehicle_to_obstacle_dist = np.linalg.norm(vehicle_to_obstacle)
					
					if vehicle_to_obstacle_dist > 1e-6:
						# Find intersection point on vehicle-to-obstacle line
						A_dot_vehicle = a1_norm * current_robot_pos[0] + a2_norm * current_robot_pos[1]
						t = b_norm - A_dot_vehicle
						line_center_point = current_robot_pos + t * np.array([a1_norm, a2_norm])
						
						# Use shorter line segment (similar to linearized constraints)
						line_length = max(3.0, vehicle_to_obstacle_dist * 0.6)  # At least 3m, or 60% of distance
						
						# Direction along the line (perpendicular to A)
						dir_x = -a2_norm
						dir_y = a1_norm
						
						# Draw line segment centered at line_center
						x1 = line_center_point[0] - dir_x * line_length / 2
						y1 = line_center_point[1] - dir_y * line_length / 2
						x2 = line_center_point[0] + dir_x * line_length / 2
						y2 = line_center_point[1] + dir_y * line_length / 2
						
						# Draw constraint line with obstacle-specific color
						ax.plot([x1, x2], [y1, y2], '--', color=color, alpha=alpha, linewidth=linewidth, 
						        label=f'Safe Horizon Constraint (obstacle {obstacle_id})' if disc_id == 0 and i == 0 and obstacle_id == 0 else None,
						        zorder=1)
						
						# Add arrow showing restriction direction (away from obstacle, toward allowed region)
						arrow_length = 1.0  # Smaller arrow
						arrow_mid_x = line_center_point[0]
						arrow_mid_y = line_center_point[1]
						arrow_dx = -a1_norm * arrow_length  # Opposite to normal (feasible side)
						arrow_dy = -a2_norm * arrow_length
						
						ax.annotate('', xy=(arrow_mid_x + arrow_dx, arrow_mid_y + arrow_dy),
						          xytext=(arrow_mid_x, arrow_mid_y),
						          arrowprops=dict(arrowstyle='->', color=color, 
						                        lw=1.5, alpha=alpha, zorder=2))
					else:
						# Fallback: vehicle and obstacle are at same position, use full line
						# (x1, x2, y1, y2 already computed above from line equation)
						ax.plot([x1, x2], [y1, y2], '--', color=color, alpha=alpha, linewidth=linewidth, 
						        label=f'Safe Horizon Constraint (obstacle {obstacle_id})' if disc_id == 0 and i == 0 and obstacle_id == 0 else None,
						        zorder=1)
				else:
					# Fallback: use full line across plot if obstacle position not available
					# (x1, x2, y1, y2 already computed above from line equation)
					ax.plot([x1, x2], [y1, y2], '--', color=color, alpha=alpha, linewidth=linewidth, 
					        label=f'Safe Horizon Constraint (obstacle {obstacle_id})' if disc_id == 0 and i == 0 and obstacle_id == 0 else None,
					        zorder=1)
					
					# Draw arrow indicating feasible side
					mid_x = (x1 + x2) / 2
					mid_y = (y1 + y2) / 2
					arrow_length = min(x_range, y_range) * 0.1
					arrow_dx = -a1_norm * arrow_length  # Opposite to normal (feasible side)
					arrow_dy = -a2_norm * arrow_length
					
					if i == max_constraints_to_plot - 1:  # Only draw arrow for last constraint to avoid clutter
						ax.arrow(mid_x, mid_y, arrow_dx, arrow_dy, head_width=arrow_length*0.3, 
						        head_length=arrow_length*0.3, fc=color, ec=color, alpha=alpha*0.7, zorder=1)
		
		# Log constraint count with detailed debugging
		total_constraints = sum(len(self.scenario_cache.get((disc_id, stage_idx), [])) 
		                       for disc_id in range(self.num_discs))
		if total_constraints > 0:
			LOG_DEBUG(f"SafeHorizonConstraint._plot_constraints_for_stage: Plotted {total_constraints} constraints for stage {stage_idx} (discs: {self.num_discs})")
		else:
			LOG_DEBUG(f"SafeHorizonConstraint._plot_constraints_for_stage: No constraints to plot for stage {stage_idx} (scenario_cache keys: {list(self.scenario_cache.keys())})")
	
	def _plot_all_scenario_halfspaces(self, ax, stage_idx, current_robot_pos=None, data=None):
		"""
		Plot linearized halfspace constraints for ALL sampled scenario obstacles.
		
		This shows all the constraints that could be applied from the sampled scenarios,
		not just the ones selected for the support set. This gives a complete picture
		of all possible obstacle positions and their corresponding constraints.
		
		CRITICAL: Excludes scenarios that are already in the support set (to avoid duplicate visualization).
		
		Args:
			ax: Matplotlib axes to plot on
			stage_idx: Stage index to plot constraints for
			current_robot_pos: Current robot position (x, y) for computing constraints. If None, uses (0, 0).
			data: Data object containing dynamic obstacles with predictions (optional, for getting current obstacle positions)
		"""
		try:
			import matplotlib.pyplot as plt
			import numpy as np
		except Exception:
			return
		
		# Check if scenario module exists
		if self.scenario_module is None or not hasattr(self.scenario_module, 'scenarios'):
			return
		
		scenarios = self.scenario_module.scenarios
		if not scenarios or len(scenarios) == 0:
			return
		
		# CRITICAL: Track obstacle_idx values that are already drawn by _plot_constraints_for_stage
		# We need to exclude by OBSTACLE INDEX, not (scenario, obstacle) tuples, because
		# multiple scenarios can exist for the same obstacle. If we only check (scenario, obstacle),
		# we'd draw the same obstacle multiple times from different scenarios - causing double lines!
		support_set_obstacles = set()
		for disc_id in range(self.num_discs):
			key = (disc_id, stage_idx)
			if key in self.scenario_cache:
				constraints = self.scenario_cache[key]
				for constraint in constraints:
					if isinstance(constraint, ScenarioConstraint):
						if hasattr(constraint, 'obstacle_idx') and constraint.obstacle_idx is not None:
							support_set_obstacles.add(int(constraint.obstacle_idx))
		
		# Get current robot position
		if current_robot_pos is None:
			current_robot_pos = np.array([0.0, 0.0])
		else:
			current_robot_pos = np.array([float(current_robot_pos[0]), float(current_robot_pos[1])])
		
		# Get current axis limits
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		x_range = xlim[1] - xlim[0]
		y_range = ylim[1] - ylim[0]
		
		# Color palette for different obstacles (matching linearized constraints visualization)
		obstacle_colors = ['red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
		
		# Limit number of scenarios to visualize to avoid clutter (show a sample)
		max_scenarios_to_plot = min(50, len(scenarios))  # Show up to 50 scenarios
		
		# Plot linearized halfspace for each scenario obstacle at this stage
		# EXCLUDE obstacles that are already drawn by _plot_constraints_for_stage (to avoid duplicate lines)
		plotted_count = 0
		plotted_obstacles_in_all_scenarios = set()  # Track obstacles already drawn in this function
		for i, scenario in enumerate(scenarios):
			if plotted_count >= max_scenarios_to_plot:
				break
			
			# Check if this obstacle is already drawn by support set (in _plot_constraints_for_stage)
			if hasattr(scenario, 'obstacle_idx_'):
				obstacle_id = int(scenario.obstacle_idx_)
				if obstacle_id in support_set_obstacles:
					# Skip this obstacle - it's already being plotted by _plot_constraints_for_stage
					continue
				# Also skip if we've already drawn this obstacle in this function
				if obstacle_id in plotted_obstacles_in_all_scenarios:
					continue
				plotted_obstacles_in_all_scenarios.add(obstacle_id)
			# Get obstacle position from scenario at current stage
			obstacle_pos = None
			obstacle_radius = None
			obstacle_idx = None
			
			# Get obstacle index from scenario
			if hasattr(scenario, 'obstacle_idx_'):
				obstacle_idx = int(scenario.obstacle_idx_)
			else:
				continue  # Skip scenarios without obstacle_idx_
			
			# Increment plotted count (only count scenarios we actually plot)
			plotted_count += 1
			
			# Get obstacle position at current stage from scenario trajectory
			if hasattr(scenario, 'trajectory') and scenario.trajectory and stage_idx < len(scenario.trajectory):
				# Use position from trajectory at this time step
				obstacle_pos = np.array([float(scenario.trajectory[stage_idx][0]), float(scenario.trajectory[stage_idx][1])])
				obstacle_radius = float(scenario.radius) if hasattr(scenario, 'radius') else self.robot_radius
			elif hasattr(scenario, 'position'):
				# Fallback: use initial scenario position
				obstacle_pos = np.array([float(scenario.position[0]), float(scenario.position[1])])
				obstacle_radius = float(scenario.radius) if hasattr(scenario, 'radius') else self.robot_radius
			else:
				continue
			
			# Also try to get from data.dynamic_obstacles at current stage (for moving obstacles)
			if data is not None and hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles:
				if obstacle_idx < len(data.dynamic_obstacles):
					obstacle = data.dynamic_obstacles[obstacle_idx]
					if obstacle is not None:
						# Get obstacle position at current stage from prediction
						if hasattr(obstacle, 'prediction') and obstacle.prediction is not None:
							if hasattr(obstacle.prediction, 'steps') and obstacle.prediction.steps:
								if stage_idx < len(obstacle.prediction.steps):
									pred_step = obstacle.prediction.steps[stage_idx]
									if hasattr(pred_step, 'position') and pred_step.position is not None:
										obstacle_pos = np.array([float(pred_step.position[0]), float(pred_step.position[1])])
										obstacle_radius = float(pred_step.radius) if hasattr(pred_step, 'radius') and pred_step.radius is not None else (float(obstacle.radius) if hasattr(obstacle, 'radius') else self.robot_radius)
			
			if obstacle_pos is None:
				continue
			
			# Compute linearized halfspace constraint (matching linearized_constraints.py)
			# Direction vector FROM robot TO obstacle
			diff_x = obstacle_pos[0] - current_robot_pos[0]
			diff_y = obstacle_pos[1] - current_robot_pos[1]
			dist = np.sqrt(diff_x**2 + diff_y**2)
			dist = max(dist, 1e-6)  # Avoid division by zero
			
			# Normalized direction vector (points FROM vehicle TO obstacle)
			a1 = diff_x / dist
			a2 = diff_y / dist
			
			# Safety margin: robot_radius + obstacle_radius + halfspace_offset
			safety_margin = self.robot_radius + obstacle_radius + self.halfspace_offset
			
			# Compute b: a·obs_pos - safety_margin
			b = a1 * obstacle_pos[0] + a2 * obstacle_pos[1] - safety_margin
			
			# Normalize the normal vector
			norm = np.sqrt(a1**2 + a2**2)
			if norm < 1e-6:
				continue
			
			a1_norm = a1 / norm
			a2_norm = a2 / norm
			b_norm = b / norm
			
			# Calculate constraint line position (similar to linearized constraints visualization)
			vehicle_to_obstacle = obstacle_pos - current_robot_pos
			vehicle_to_obstacle_dist = np.linalg.norm(vehicle_to_obstacle)
			
			if vehicle_to_obstacle_dist > 1e-6:
				# Find intersection point on vehicle-to-obstacle line
				A_dot_vehicle = a1_norm * current_robot_pos[0] + a2_norm * current_robot_pos[1]
				t = b_norm - A_dot_vehicle
				line_center_point = current_robot_pos + t * np.array([a1_norm, a2_norm])
				
				# Use shorter line segment (similar to linearized constraints)
				line_length = max(2.0, vehicle_to_obstacle_dist * 0.5)  # At least 2m, or 50% of distance
				
				# Direction along the line (perpendicular to A)
				dir_x = -a2_norm
				dir_y = a1_norm
				
				# Draw line segment centered at line_center
				x1 = line_center_point[0] - dir_x * line_length / 2
				y1 = line_center_point[1] - dir_y * line_length / 2
				x2 = line_center_point[0] + dir_x * line_length / 2
				y2 = line_center_point[1] + dir_y * line_length / 2
			else:
				# Fallback: use full line
				extend = max(x_range, y_range) * 0.3
				x_center = (xlim[0] + xlim[1]) / 2
				y_center = (ylim[0] + ylim[1]) / 2
				
				if abs(a2_norm) > 1e-6:
					x1 = x_center - extend
					x2 = x_center + extend
					y1 = (b_norm - a1_norm * x1) / a2_norm
					y2 = (b_norm - a1_norm * x2) / a2_norm
				else:
					x1 = b_norm / a1_norm
					x2 = x1
					y1 = y_center - extend
					y2 = y_center + extend
			
			# Use obstacle-specific color (matching linearized constraints visualization)
			color = obstacle_colors[obstacle_idx % len(obstacle_colors)]
			alpha = 0.3  # More transparent than support set constraints to show they're potential constraints
			linewidth = 0.8
			
			# Draw constraint line with obstacle-specific color
			ax.plot([x1, x2], [y1, y2], '--', color=color, alpha=alpha, linewidth=linewidth, zorder=0)
			
			# Optionally add small arrow (less prominent than support set constraints)
			if i % 10 == 0:  # Only show arrow for every 10th scenario to reduce clutter
				arrow_length = 0.5
				if vehicle_to_obstacle_dist > 1e-6:
					arrow_mid_x = line_center_point[0]
					arrow_mid_y = line_center_point[1]
				else:
					arrow_mid_x = (x1 + x2) / 2
					arrow_mid_y = (y1 + y2) / 2
				arrow_dx = -a1_norm * arrow_length
				arrow_dy = -a2_norm * arrow_length
				
				ax.annotate('', xy=(arrow_mid_x + arrow_dx, arrow_mid_y + arrow_dy),
				          xytext=(arrow_mid_x, arrow_mid_y),
				          arrowprops=dict(arrowstyle='->', color=color, 
				                        lw=1.0, alpha=alpha*0.7, zorder=0))
		
		if stage_idx == 0:  # Only log for stage 0 to avoid spam
			LOG_DEBUG(f"SafeHorizonConstraint._plot_all_scenario_halfspaces: Plotted {plotted_count} scenario halfspaces for stage {stage_idx} (excluded obstacles {support_set_obstacles} from support set)")
	
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
				safety_margin = self.robot_radius + obstacle_radius + self.halfspace_offset
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
