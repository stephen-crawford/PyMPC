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
		self.epsilon_p = float(self.get_config_value("safe_horizon_constraints.epsilon_p", 0.1))  # Constraint violation probability
		self.beta = float(self.get_config_value("safe_horizon_constraints.beta", 0.01))  # Confidence level
		self.n_bar = int(self.get_config_value("safe_horizon_constraints.n_bar", 10))  # Support dimension
		self.num_scenarios = int(self.get_config_value("safe_horizon_constraints.num_scenarios", 100))
		self.max_constraints_per_disc = int(self.get_config_value("safe_horizon_constraints.max_constraints_per_disc", 24))
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
				LOG_DEBUG("SafeHorizonConstraint.update: Data not ready (no obstacles with Gaussian predictions)")
				self.scenario_cache.clear()
				return
			
			# Sample scenarios (this populates self.scenario_module.scenarios)
			self.scenario_module.update(data)
			
			if not self.scenario_module.scenarios:
				LOG_DEBUG("SafeHorizonConstraint.update: No scenarios sampled")
				self.scenario_cache.clear()
				return
			
			# Cache scenarios by disc and stage (scenarios are independent of robot position)
			# We'll formulate constraints symbolically in calculate_constraints()
			self.scenario_cache.clear()
			for disc_id in range(self.num_discs):
				for stage_idx in range(self.horizon_length):
					# Get scenarios for this time step
					step_scenarios = [s for s in self.scenario_module.scenarios if s.time_step == stage_idx]
					key = (disc_id, stage_idx)
					self.scenario_cache[key] = step_scenarios
			
			# Log scenario info
			LOG_DEBUG(f"SafeHorizonConstraint.update: Cached scenarios - "
			         f"total_scenarios={len(self.scenario_module.scenarios)}")
			
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
		if self.scenario_module is None or not self.scenario_cache:
			LOG_DEBUG(f"SafeHorizonConstraint.calculate_constraints: No scenarios available for stage {stage_idx}")
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
		for disc_id in range(self.num_discs):
			key = (disc_id, stage_idx)
			
			if key not in self.scenario_cache:
				continue
			
			scenarios = self.scenario_cache[key]
			
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
			
			# Formulate collision constraints symbolically from scenarios
			for scenario in scenarios:
				# Get obstacle position from scenario (numeric, fixed)
				obstacle_x = float(scenario.position[0])
				obstacle_y = float(scenario.position[1])
				obstacle_radius = float(scenario.radius)
				
				# Formulate constraint symbolically using predicted robot position
				# The constraint is: distance(robot_pos, obstacle_pos) >= robot_radius + obstacle_radius
				# Linearized at predicted position: nx * dx + ny * dy >= safety_margin
				# Where dx = obstacle_x - robot_x, dy = obstacle_y - robot_y
				# And nx, ny = (dx, dy) / distance (normalized direction from robot to obstacle)
				
				# Compute direction vector symbolically
				dx = obstacle_x - disc_x  # Symbolic: obstacle_x - robot_x
				dy = obstacle_y - disc_y  # Symbolic: obstacle_y - robot_y
				distance_sq = dx * dx + dy * dy  # Symbolic distance squared
				
				# Safety margin
				safety_margin = self.robot_radius + obstacle_radius
				
				# Normalize direction (handle division by zero with epsilon)
				epsilon = 1e-6
				distance_safe = cd.sqrt(distance_sq + epsilon * epsilon)
				nx = dx / distance_safe  # Normalized direction from robot to obstacle
				ny = dy / distance_safe
				
				# Constraint: distance >= safety_margin
				# Linearized: nx * dx + ny * dy >= safety_margin
				# Which is: nx * dx + ny * dy - safety_margin >= 0
				# For CasADi form (constraint_expr <= 0): safety_margin - (nx * dx + ny * dy) <= 0
				# This ensures: nx * dx + ny * dy >= safety_margin (distance >= safety_margin)
				constraint_expr = safety_margin - (nx * dx + ny * dy)
				
				# Return as dict for solver to process
				constraints.append({
					"type": "linear_halfspace",
					"expression": constraint_expr,
					"disc_id": disc_id,
					"scenario_idx": scenario.idx_,
					"obstacle_idx": scenario.obstacle_idx_,
					"time_step": scenario.time_step
				})
		
		if constraints:
			LOG_DEBUG(f"SafeHorizonConstraint.calculate_constraints: stage_idx={stage_idx}, "
			         f"returning {len(constraints)} constraints")
		
		return constraints
	
	def lower_bounds(self, state: Optional[State] = None, data: Optional[Data] = None, stage_idx: Optional[int] = None):
		"""
		Get lower bounds for constraints at this stage.
		
		For safe horizon constraints, we use: constraint_expr <= 0
		So lower bound is -inf (no lower bound on the expression itself).
		However, the solver expects bounds on the constraint value, so we return 0.0
		to enforce: constraint_expr <= 0
		
		Args:
			state: State (not used)
			data: Data object
			stage_idx: Stage index
		
		Returns:
			List of lower bounds (all 0.0 for <= constraints)
		"""
		if stage_idx is None or data is None:
			return []
		
		count = 0
		for disc_id in range(self.num_discs):
			key = (disc_id, stage_idx)
			if key in self.scenario_cache:
				count += len(self.scenario_cache[key])
		
		return [0.0] * count
	
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
	
	def reset(self):
		"""Reset constraint module state."""
		if self.scenario_module is not None:
			self.scenario_module.reset()
		self.scenario_cache.clear()
		LOG_DEBUG("SafeHorizonConstraint.reset: Reset complete")
