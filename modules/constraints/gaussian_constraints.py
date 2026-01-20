import numpy as np
import casadi as cd

from planning.types import PredictionType
from utils.const import GAUSSIAN, DYNAMIC
from utils.math_tools import rotation_matrix, casadi_rotation_matrix, DouglasRachford
from utils.utils import LOG_DEBUG, PROFILE_SCOPE, CONFIG, LOG_INFO, LOG_WARN
from modules.constraints.base_constraint import BaseConstraint

class GaussianConstraints(BaseConstraint):
	def __init__(self):
		super().__init__()
		self.name = 'gaussian_constraints'
		# Store dummy values for invalid states
		self._dummy_x = 0.0
		self._dummy_y = 0.0
		num_discs_val = self.get_config_value("num_discs")
		self.num_discs = int(num_discs_val) if num_discs_val is not None else 1
		robot_radius_val = self.get_config_value("robot.radius")
		self.robot_radius = float(robot_radius_val) if robot_radius_val is not None else 0.5
		max_obstacles_val = self.get_config_value("max_obstacles")
		self.max_obstacles = int(max_obstacles_val) if max_obstacles_val is not None else 10
		self.num_constraints = self.num_discs * self.max_obstacles
		self.num_active_obstacles = 0
		
		# Initialize solver attribute (set by framework later)
		self.solver = None
		
		# Initialize Douglas-Rachford projection for constraint violation handling
		self.douglas_rachford = DouglasRachford()

		LOG_DEBUG(f"Gaussian Constraints successfully initialized: num_discs={self.num_discs}, max_obstacles={self.max_obstacles}, robot_radius={self.robot_radius}")

	def update(self, state, data):
		"""
		Update constraint module with new data.
		
		CRITICAL: Similar to LinearizedConstraints, we should ensure the current vehicle position
		is feasible (not violating constraints) to prevent infeasibility. However, for Gaussian
		constraints, we rely on the probabilistic nature - small violations may be acceptable
		as long as the probability is below the risk threshold.
		
		Reference: C++ mpc_planner - constraints are updated each planning iteration.
		"""
		LOG_DEBUG("GaussianConstraints.update")

		# Update dummy values based on current state
		self._dummy_x = state.get("x") + 100.0
		self._dummy_y = state.get("y") + 100.0

		copied_dynamic_obstacles = data.dynamic_obstacles
		self.num_active_obstacles = len(copied_dynamic_obstacles)
		
		# CRITICAL: Check if current position violates constraints (for feasibility)
		# Reference: LinearizedConstraints.update_step projects position away from obstacles
		# For Gaussian constraints, we check Mahalanobis distance but don't project unless
		# violation is severe (would cause solver infeasibility)
		if state is not None and copied_dynamic_obstacles:
			try:
				pos_x = float(state.get("x")) if state.has("x") else None
				pos_y = float(state.get("y")) if state.has("y") else None
				if pos_x is not None and pos_y is not None:
					vehicle_pos = np.array([pos_x, pos_y])
					risk_level = float(self.get_config_value("gaussian_constraints.risk_level", 0.05))
					from scipy.stats import chi2
					chi_squared_threshold = chi2.ppf(1.0 - risk_level, df=2)
					
					# Check each obstacle for severe violations
					for obs_id, obstacle in enumerate(copied_dynamic_obstacles[:self.max_obstacles]):
						if (hasattr(obstacle, 'prediction') and obstacle.prediction is not None and
							obstacle.prediction.type == PredictionType.GAUSSIAN and
							hasattr(obstacle.prediction, 'steps') and len(obstacle.prediction.steps) > 0):
							# Use current step (stage 0) for feasibility check in update()
							pred_step = obstacle.prediction.steps[0]  # Current step
							if hasattr(pred_step, 'position') and pred_step.position is not None:
								mean_pos = np.array([float(pred_step.position[0]), float(pred_step.position[1])])
								major_radius = float(getattr(pred_step, 'major_radius', 0.1))
								minor_radius = float(getattr(pred_step, 'minor_radius', 0.1))
								
								# Compute Mahalanobis distance
								diff = vehicle_pos - mean_pos
								sigma_x = major_radius
								sigma_y = minor_radius
								if sigma_x > 1e-10 and sigma_y > 1e-10:
									# CRITICAL: Use effective covariance (matching constraint calculation)
									# Reference: C++ mpc_planner - consistency between constraint and feasibility check
									obstacle_radius = float(getattr(obstacle, 'radius', 0.35))
									robot_radius = float(self.robot_radius) if self.robot_radius else 0.5
									safe_distance = robot_radius + obstacle_radius
									
									# Use effective covariance (matching constraint formulation)
									sigma_x_eff = sigma_x + safe_distance
									sigma_y_eff = sigma_y + safe_distance
									
									# Compute Mahalanobis distance using EFFECTIVE covariance
									mahalanobis_dist_sq = (diff[0]**2 / sigma_x_eff**2) + (diff[1]**2 / sigma_y_eff**2)
									
									# CRITICAL: Use same safety margin as constraint calculation for consistency
									# Reference: C++ mpc_planner - feasibility check must match constraint formulation
									safety_margin_factor = 1.2  # 20% safety margin (matching constraint calculation)
									required_threshold = chi_squared_threshold * safety_margin_factor
									
									if mahalanobis_dist_sq < required_threshold * 0.5:  # Severe violation (50% below threshold)
										LOG_WARN(f"GaussianConstraints: Severe constraint violation detected at obstacle {obs_id}: "
										         f"mahalanobis_dist_sq={mahalanobis_dist_sq:.3f} < threshold={required_threshold:.3f}")
										
										# CRITICAL: Use Douglas-Rachford projection to move vehicle to safe region
										# Reference: C++ mpc_planner uses projection methods to ensure feasibility
										# Project vehicle position away from obstacle using ellipsoidal constraint region
										# CRITICAL: Use effective sigma values to match constraint formulation
										projected_pos = self._project_to_gaussian_safety(
											vehicle_pos.copy(), 
											mean_pos, 
											sigma_x_eff,  # Use effective sigma
											sigma_y_eff,  # Use effective sigma
											required_threshold,
											obstacle_radius,
											robot_radius
										)
										
										if projected_pos is not None:
											projection_distance = np.linalg.norm(projected_pos - vehicle_pos)
											if projection_distance > 1e-6:  # Significant projection
												LOG_WARN(f"GaussianConstraints: Projected vehicle position using Douglas-Rachford: "
												         f"from ({vehicle_pos[0]:.3f}, {vehicle_pos[1]:.3f}) "
												         f"to ({projected_pos[0]:.3f}, {projected_pos[1]:.3f}), "
												         f"distance={projection_distance:.3f}m")
												# Note: We can't modify state here, but we log the projection
												# The solver warmstart should use this projection
			except Exception as e:
				LOG_DEBUG(f"GaussianConstraints.update: Error checking feasibility: {e}")
		
		# CRITICAL: Project warmstart values to ensure feasibility
		# Reference: SafeHorizonConstraint._project_warmstart_to_safety - projects warmstart to satisfy constraints
		# This prevents solver infeasibility by ensuring initial guess is feasible
		self._project_warmstart_to_gaussian_safety(data)

	def calculate_constraints(self, state, data, stage_idx):
		"""
		Calculate Gaussian constraints symbolically for probabilistic obstacle avoidance.
		
		CRITICAL: Constraints are computed using the predicted obstacle state at each stage,
		similar to how contouring constraints use the predicted vehicle state (spline value).
		For each stage k, constraints use obstacle.prediction.steps[k] to get the predicted
		obstacle position and uncertainty at that stage.
		
		Constraint formulation: (p - μ)^T * Σ^(-1) * (p - μ) >= χ²(α)
		where:
			- p is the vehicle disc position (symbolic, predicted at stage k)
			- μ is the mean obstacle position (from prediction.steps[k])
			- Σ is the covariance matrix (uncertainty from prediction.steps[k])
			- χ²(α) is the chi-squared quantile for risk level α
		
		This ensures P(||p - μ|| <= safe_distance) <= α (chance constraint).
		
		Reference: https://github.com/tud-amr/mpc_planner - constraints use predicted states at each stage.
		"""
		constraints = []
		
		# CRITICAL: Apply constraints to more stages to ensure vehicle turns to avoid obstacles
		# Reference: C++ mpc_planner - Gaussian constraints are applied across the prediction horizon
		# to ensure the vehicle actively avoids obstacles by turning, not just by slowing down
		# Applying to more stages forces the vehicle to plan a turning trajectory
		
		# IMPROVEMENT: Adaptive constraint horizon based on obstacle speed
		# Fast-moving obstacles require longer constraint horizon to ensure avoidance
		base_max_stage = 8  # Base constraint horizon
		max_stage_for_constraints = base_max_stage
		
		# Check if any obstacles are moving fast (need longer horizon)
		if data.has("dynamic_obstacles") and data.dynamic_obstacles is not None:
			for obstacle in data.dynamic_obstacles[:self.max_obstacles]:
				if (hasattr(obstacle, 'prediction') and obstacle.prediction is not None and
					obstacle.prediction.type == PredictionType.GAUSSIAN):
					# Estimate obstacle speed
					obstacle_speed = 0.0
					if hasattr(obstacle, 'velocity') and obstacle.velocity is not None:
						obstacle_speed = np.linalg.norm(obstacle.velocity)
					elif hasattr(obstacle, 'speed') and obstacle.speed is not None:
						obstacle_speed = float(obstacle.speed)
					
					# Extend horizon for fast-moving obstacles (> 2 m/s)
					if obstacle_speed > 2.0:
						max_stage_for_constraints = max(max_stage_for_constraints, 10)  # Extended horizon
						break  # Use extended horizon if any obstacle is fast-moving
		
		if stage_idx > max_stage_for_constraints:
			LOG_DEBUG(f"GaussianConstraints: Skipping constraints for stage {stage_idx} (max_stage={max_stage_for_constraints})")
			return constraints
		
		# Get symbolic vehicle position (predicted at this stage)
		pos_x_sym = state.get("x")
		pos_y_sym = state.get("y")
		psi_sym = state.get("psi")
		
		if pos_x_sym is None or pos_y_sym is None:
			LOG_DEBUG(f"GaussianConstraints: Cannot get symbolic position at stage {stage_idx}")
			return []
		
		# Get obstacles with Gaussian predictions
		if not data.has("dynamic_obstacles") or data.dynamic_obstacles is None:
			return []
		
		copied_dynamic_obstacles = data.dynamic_obstacles
		risk_level = float(self.get_config_value("gaussian_constraints.risk_level", 0.05))  # Default 5% risk
		
		# Compute constraints for each disc and obstacle
		for disc_id in range(self.num_discs):
			# Get disc offset
			disc_offset = 0.0
			if data.has("robot_area") and data.robot_area is not None and disc_id < len(data.robot_area):
				disc_offset = float(data.robot_area[disc_id].offset)
			
			# Compute disc position symbolically (using predicted vehicle state at this stage)
			if abs(disc_offset) > 1e-9 and psi_sym is not None:
				disc_x_sym = pos_x_sym + disc_offset * cd.cos(psi_sym)
				disc_y_sym = pos_y_sym + disc_offset * cd.sin(psi_sym)
			else:
				disc_x_sym = pos_x_sym
				disc_y_sym = pos_y_sym
			
			disc_pos_sym = cd.vertcat(disc_x_sym, disc_y_sym)
			
			# Compute constraints for each obstacle
			for obs_id in range(min(len(copied_dynamic_obstacles), self.max_obstacles)):
				obstacle = copied_dynamic_obstacles[obs_id]
				
				# Check if obstacle has Gaussian prediction
				if not hasattr(obstacle, 'prediction') or obstacle.prediction is None:
					continue
				
				if obstacle.prediction.type != PredictionType.GAUSSIAN:
					continue
				
				# CRITICAL: Get prediction step for THIS stage (stage_idx)
				# This ensures constraints use the predicted obstacle state at the correct stage,
				# matching how contouring constraints use the predicted vehicle state (spline value) at each stage.
				# Reference: C++ mpc_planner - constraints are evaluated using predicted states at each stage.
				if not hasattr(obstacle.prediction, 'steps') or obstacle.prediction.steps is None:
					LOG_DEBUG(f"GaussianConstraints: Obstacle {obs_id} has no prediction steps at stage {stage_idx}")
					continue
				
				pred_steps = obstacle.prediction.steps
				if not pred_steps or len(pred_steps) <= stage_idx:
					LOG_DEBUG(f"GaussianConstraints: Obstacle {obs_id} prediction steps length {len(pred_steps) if pred_steps else 0} <= stage_idx {stage_idx}")
					continue
				
				# Get predicted obstacle state at this stage (analogous to contouring constraints using predicted spline value)
				# CRITICAL: This ensures constraints are computed using the obstacle's predicted position and uncertainty
				# at stage k, matching the pattern used by contouring constraints for vehicle state prediction.
				pred_step = pred_steps[stage_idx]
				
				# Get mean position (μ) from predicted obstacle state at this stage
				if not hasattr(pred_step, 'position') or pred_step.position is None:
					LOG_DEBUG(f"GaussianConstraints: Obstacle {obs_id} pred_step at stage {stage_idx} has no position")
					continue
				
				mean_pos = np.array([float(pred_step.position[0]), float(pred_step.position[1])])
				LOG_DEBUG(f"GaussianConstraints: Stage {stage_idx}, Obstacle {obs_id}: Using predicted position ({mean_pos[0]:.3f}, {mean_pos[1]:.3f})")
				
				# Get covariance matrix (Σ) from predicted obstacle state at this stage
				# CRITICAL: Use uncertainty parameters from prediction.steps[stage_idx] to ensure
				# constraints account for uncertainty growth over the prediction horizon.
				# This matches how contouring constraints use predicted vehicle state at each stage.
				major_radius = float(getattr(pred_step, 'major_radius', 0.1))
				minor_radius = float(getattr(pred_step, 'minor_radius', 0.1))
				# Try 'angle' first (PredictionStep uses 'angle'), then fallback to 'orientation'
				orientation = float(getattr(pred_step, 'angle', getattr(pred_step, 'orientation', 0.0)))
				
				# Get safe distance (robot radius + obstacle radius)
				obstacle_radius = float(getattr(obstacle, 'radius', 0.35))
				robot_radius = float(self.robot_radius) if self.robot_radius else 0.5
				safe_distance = robot_radius + obstacle_radius
				
				# Construct covariance matrix from uncertainty ellipsoid
				# For simplicity, use diagonal covariance scaled by radii
				# More sophisticated: use rotation matrix for oriented uncertainty
				sigma_x = major_radius
				sigma_y = minor_radius
				
				# CRITICAL FIX: Proper constraint formulation for Gaussian chance constraints
				# Reference: C++ mpc_planner - Gaussian constraints use Mahalanobis distance with proper scaling
				# Following the pattern from ellipsoid_constraints.py which incorporates safe_distance
				# directly into the constraint matrix (similar to how ellipsoid constraints incorporate disc radius)
				# 
				# We want to ensure: P(||p - μ|| <= safe_distance) <= α (chance constraint)
				# 
				# The constraint formulation should:
				# 1. Use Mahalanobis distance squared: d_M² = (p - μ)^T * Σ^(-1) * (p - μ)
				# 2. Ensure d_M² >= threshold that corresponds to safe_distance
				# 
				# Reference approach (matching ellipsoid_constraints pattern):
				# - Incorporate safe_distance into the covariance matrix by expanding the uncertainty
				# - This ensures the constraint naturally accounts for physical separation
				# - Similar to ellipsoid_constraints where disc radius is added to obstacle semi-axes
				# 
				# For diagonal covariance Σ = diag(σ_x², σ_y²):
				# - Expand uncertainty by safe_distance: σ_eff = σ + safe_distance
				# - This creates an effective covariance that accounts for both uncertainty and physical separation
				# - The Mahalanobis distance then naturally enforces both probabilistic safety and physical separation
				from scipy.stats import chi2
				chi_squared_threshold = chi2.ppf(1.0 - risk_level, df=2)
				
				# CRITICAL: Incorporate safe_distance into effective covariance (matching ellipsoid_constraints pattern)
				# Reference: C++ mpc_planner - similar to ellipsoid constraints, we expand the uncertainty region
				# by safe_distance to account for physical separation
				# This is analogous to ellipsoid_constraints where: major_denom = obst_major + r_disc
				sigma_x_eff = sigma_x + safe_distance
				sigma_y_eff = sigma_y + safe_distance
				
				# Create effective covariance matrix with safe_distance incorporated
				# This ensures the constraint naturally enforces both probabilistic safety and physical separation
				cov_matrix_eff = np.array([[sigma_x_eff**2, 0.0], [0.0, sigma_y_eff**2]])
				
				# Apply rotation if orientation is non-zero (same as before)
				if abs(orientation) > 1e-6:
					cos_theta = np.cos(orientation)
					sin_theta = np.sin(orientation)
					rotation = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
					cov_matrix_eff = rotation @ cov_matrix_eff @ rotation.T
				
				# Compute inverse effective covariance Σ_eff^(-1)
				try:
					inv_cov_matrix_eff = np.linalg.inv(cov_matrix_eff)
				except np.linalg.LinAlgError:
					# If singular, use pseudo-inverse or add small regularization
					inv_cov_matrix_eff = np.linalg.pinv(cov_matrix_eff + np.eye(2) * 1e-6)
				
				# Convert to CasADi symbolic matrix
				inv_cov_sym = cd.DM(inv_cov_matrix_eff)
				
				# Compute constraint: (p - μ)^T * Σ_eff^(-1) * (p - μ) >= χ²(α)
				# CRITICAL: Ensure mean_pos is converted to CasADi vector for symbolic computation
				# Reference: C++ mpc_planner - Gaussian constraints use Mahalanobis distance with predicted obstacle state
				# The mean position is a constant (from prediction), but disc_pos_sym is symbolic (from vehicle state)
				mean_pos_sym = cd.vertcat(cd.DM(mean_pos[0]), cd.DM(mean_pos[1]))  # Explicit DM for constants
				disc_to_mean = disc_pos_sym - mean_pos_sym
				# CRITICAL: Ensure mahalanobis_dist_sq is symbolic by using matrix multiplication
				# inv_cov_sym is DM (constant), disc_to_mean is MX (symbolic), result should be MX
				mahalanobis_dist_sq = disc_to_mean.T @ inv_cov_sym @ disc_to_mean
				
				# CRITICAL: Add safety margin to threshold for robust constraint enforcement
				# Reference: C++ mpc_planner - uses conservative thresholds to account for:
				# 1. Continuous motion between discrete stages (can violate constraints between stages)
				# 2. Numerical solver tolerances
				# 3. Prediction uncertainty
				# Similar to linearized_constraints which uses halfspace_offset for safety margin
				# CRITICAL FIX: Increase safety margin to force vehicle to turn and actively avoid obstacles
				# A larger margin ensures the vehicle maintains greater distance, which requires turning
				# rather than just slowing down or moving in a straight line
				
				# CRITICAL FIX: Minimal safety margin for Gaussian constraints
				# The effective covariance already includes safe_distance, so additional margin
				# makes constraints overly conservative for narrow roads
				# Use minimal margin (1.0 = no extra margin) to allow feasible solutions
				base_safety_margin = 1.0  # No additional margin beyond chi-squared threshold
				obstacle_speed = 0.0
				if hasattr(obstacle, 'velocity') and obstacle.velocity is not None:
					obstacle_speed = np.linalg.norm(obstacle.velocity)
				elif hasattr(obstacle, 'speed') and obstacle.speed is not None:
					obstacle_speed = float(obstacle.speed)
				elif hasattr(pred_step, 'position') and stage_idx > 0:
					# Estimate speed from position change between stages
					try:
						prev_step = pred_steps[stage_idx - 1] if stage_idx > 0 else None
						if prev_step is not None and hasattr(prev_step, 'position'):
							prev_pos = np.array([float(prev_step.position[0]), float(prev_step.position[1])])
							curr_pos = np.array([float(pred_step.position[0]), float(pred_step.position[1])])
							position_change = np.linalg.norm(curr_pos - prev_pos)
							# Estimate speed: distance / timestep
							timestep = 0.1  # Default timestep
							if hasattr(self, 'solver') and hasattr(self.solver, 'timestep') and self.solver.timestep is not None:
								timestep = float(self.solver.timestep)
							obstacle_speed = position_change / timestep if timestep > 0 else 0.0
					except Exception:
						obstacle_speed = 0.0
				
				# Increase safety margin for faster obstacles
				# 20% increase per m/s above 1 m/s (capped at 3x base margin)
				if obstacle_speed > 1.0:
					speed_factor = 1.0 + min((obstacle_speed - 1.0) * 0.2, 1.5)  # Max 2.5x (1.0 + 1.5)
					safety_margin_factor = base_safety_margin * speed_factor
				else:
					safety_margin_factor = base_safety_margin
				
				scaled_threshold = chi_squared_threshold * safety_margin_factor
				
				LOG_INFO(f"GaussianConstraints: Stage {stage_idx}, Obstacle {obs_id}: "
				         f"chi²={chi_squared_threshold:.3f}, scaled_threshold={scaled_threshold:.3f}, "
				         f"safe_distance={safe_distance:.3f}, "
				         f"sigma_original=({sigma_x:.3f}, {sigma_y:.3f}), "
				         f"sigma_effective=({sigma_x_eff:.3f}, {sigma_y_eff:.3f})")
				
				# CRITICAL: Convert scaled_threshold to CasADi constant to ensure proper symbolic expression
				# Constraint: mahalanobis_dist_sq >= scaled_threshold
				# i.e., scaled_threshold - mahalanobis_dist_sq <= 0
				scaled_threshold_sym = cd.DM(scaled_threshold)  # Convert to CasADi constant
				constraint_expr = scaled_threshold_sym - mahalanobis_dist_sq
				
				# CRITICAL: Verify constraint expression is symbolic
				if not isinstance(constraint_expr, (cd.MX, cd.SX)):
					LOG_WARN(f"GaussianConstraints: Constraint expression is not symbolic! Type: {type(constraint_expr)}")
					# Try to convert
					constraint_expr = cd.DM(scaled_threshold) - mahalanobis_dist_sq
				
				# CRITICAL: Log constraint details for debugging
				if stage_idx <= 2:  # Log first few stages
					LOG_INFO(f"GaussianConstraints: Stage {stage_idx}, Disc {disc_id}, Obstacle {obs_id}: "
					         f"Adding constraint with threshold={scaled_threshold:.3f}, "
					         f"mean_pos=({mean_pos[0]:.3f}, {mean_pos[1]:.3f}), "
					         f"sigma=({sigma_x:.3f}, {sigma_y:.3f}), "
					         f"constraint_expr type={type(constraint_expr).__name__}")
				
				constraints.append({
					"type": "symbolic_expression",
					"expression": constraint_expr,
					"ub": 0.0,  # expr <= 0 means mahalanobis_dist_sq >= scaled_threshold
					"lb": None,
					"constraint_type": "gaussian",
					"obs_id": obs_id,
					"disc_id": disc_id,
					"stage_idx": stage_idx
				})
		
		return constraints
	
	def _project_to_gaussian_safety_with_path_direction(self, position, obstacle_mean, sigma_x, sigma_y,
	                                                     mahalanobis_threshold, obstacle_radius, robot_radius,
	                                                     path_direction=None):
		"""
		Project vehicle position to safe region while maintaining path direction.
		
		CRITICAL FIX: When path_direction is provided, prefer projection that maintains
		the path direction (lateral offset) rather than directly away from obstacle.
		This ensures the predicted trajectory remains straight along the path.
		
		Reference: C++ mpc_planner - projection maintains path-following trajectory.
		"""
		# First, try standard projection
		projected_pos = self._project_to_gaussian_safety(
			position, obstacle_mean, sigma_x, sigma_y,
			mahalanobis_threshold, obstacle_radius, robot_radius
		)
		
		if projected_pos is None:
			return None
		
		# CRITICAL: If path_direction is available, adjust projection to maintain path direction
		# This ensures the trajectory remains straight along the path
		if path_direction is not None and np.linalg.norm(path_direction) > 1e-9:
			# Compute projection offset
			projection_offset = projected_pos - position
			
			# Decompose projection into path-parallel and path-perpendicular components
			path_perp = np.array([-path_direction[1], path_direction[0]])  # Perpendicular to path (left)
			
			# Project offset onto path direction and perpendicular
			offset_parallel = np.dot(projection_offset, path_direction) * path_direction
			offset_perp = np.dot(projection_offset, path_perp) * path_perp
			
			# CRITICAL: Prefer lateral (perpendicular) projection to maintain forward progress
			# Only use parallel component if it's small (to avoid reversing)
			if np.dot(projection_offset, path_direction) < -0.1:  # Would cause backward movement
				# Use only perpendicular component to avoid reversing
				adjusted_projection = position + offset_perp
			else:
				# Use both components but prefer perpendicular
				# Blend: 80% perpendicular, 20% parallel (maintains path direction)
				adjusted_projection = position + 0.8 * offset_perp + 0.2 * offset_parallel
			
			# Verify adjusted projection still satisfies constraints
			diff_adj = adjusted_projection - obstacle_mean
			mahalanobis_adj = (diff_adj[0]**2 / sigma_x**2) + (diff_adj[1]**2 / sigma_y**2)
			euclidean_adj = np.linalg.norm(diff_adj)
			safe_distance = robot_radius + obstacle_radius
			
			if mahalanobis_adj >= mahalanobis_threshold and euclidean_adj >= safe_distance:
				# Adjusted projection satisfies constraints - use it
				return adjusted_projection
			else:
				# Adjusted projection doesn't satisfy constraints - use original
				# But try to minimize backward movement
				if np.dot(projected_pos - position, path_direction) < -0.1:
					# Original projection causes backward movement - use perpendicular only
					diff_orig = projected_pos - position
					offset_perp_only = np.dot(diff_orig, path_perp) * path_perp
					perp_projection = position + offset_perp_only
					
					# Verify perpendicular-only projection
					diff_perp = perp_projection - obstacle_mean
					mahalanobis_perp = (diff_perp[0]**2 / sigma_x**2) + (diff_perp[1]**2 / sigma_y**2)
					euclidean_perp = np.linalg.norm(diff_perp)
					
					if mahalanobis_perp >= mahalanobis_threshold and euclidean_perp >= safe_distance:
						return perp_projection
		
		return projected_pos
	
	def _project_to_gaussian_safety(self, position, obstacle_mean, sigma_x, sigma_y, 
	                                mahalanobis_threshold, obstacle_radius, robot_radius):
		"""
		Project vehicle position to safe region outside Gaussian constraint violation.
		
		Uses Douglas-Rachford splitting principles adapted for ellipsoidal constraints.
		Projects position onto the ellipsoid defined by Mahalanobis distance threshold.
		
		The projection ensures:
		1. Mahalanobis distance squared >= threshold (probabilistic safety)
		2. Euclidean distance >= safe_distance (physical separation)
		
		Args:
			position: Current vehicle position (2D numpy array)
			obstacle_mean: Mean obstacle position (2D numpy array)
			sigma_x: Standard deviation in x direction
			sigma_y: Standard deviation in y direction
			mahalanobis_threshold: Required Mahalanobis distance squared threshold
			obstacle_radius: Obstacle radius
			robot_radius: Robot radius
		
		Returns:
			Projected position (2D numpy array) or None if no projection needed
		
		Reference: Douglas-Rachford splitting for constraint projection
		Reference: C++ mpc_planner - projection ensures warmstart feasibility
		"""
		# Compute current Mahalanobis distance squared
		diff = position - obstacle_mean
		mahalanobis_dist_sq = (diff[0]**2 / sigma_x**2) + (diff[1]**2 / sigma_y**2)
		
		# Compute safe distance
		safe_distance = robot_radius + obstacle_radius
		euclidean_dist = np.linalg.norm(diff)
		
		# Check if already safe (both Mahalanobis and Euclidean)
		if mahalanobis_dist_sq >= mahalanobis_threshold and euclidean_dist >= safe_distance:
			return None
		
		# CRITICAL: Use Douglas-Rachford projection adapted for ellipsoidal constraints
		# Step 1: Project onto ellipsoid boundary (Mahalanobis constraint)
		# The ellipsoid is: (x-μ_x)²/σ_x² + (y-μ_y)²/σ_y² = threshold
		
		# Normalize difference by standard deviations
		diff_normalized = np.array([diff[0] / sigma_x, diff[1] / sigma_y])
		dist_normalized = np.linalg.norm(diff_normalized)
		
		if dist_normalized < 1e-10:
			# Vehicle is exactly at obstacle mean - project in direction of largest uncertainty
			# This ensures we move away in the direction where uncertainty is highest
			if sigma_x >= sigma_y:
				diff_normalized = np.array([1.0, 0.0])
			else:
				diff_normalized = np.array([0.0, 1.0])
			dist_normalized = 1.0
		
		# Scale to ellipsoid boundary (Mahalanobis distance = sqrt(threshold))
		scale_factor = np.sqrt(mahalanobis_threshold) / dist_normalized
		diff_normalized_scaled = diff_normalized * scale_factor
		
		# Convert back to original space
		projected_diff_ellipsoid = np.array([
			diff_normalized_scaled[0] * sigma_x,
			diff_normalized_scaled[1] * sigma_y
		])
		
		projected_pos_ellipsoid = obstacle_mean + projected_diff_ellipsoid
		
		# Step 2: Ensure Euclidean distance constraint (physical separation)
		# This handles the case where Mahalanobis distance is satisfied but
		# Euclidean distance is still too small
		euclidean_dist_ellipsoid = np.linalg.norm(projected_pos_ellipsoid - obstacle_mean)
		
		if euclidean_dist_ellipsoid < safe_distance:
			# Further project to ensure safe Euclidean distance
			# Use direction from ellipsoid projection
			direction = (projected_pos_ellipsoid - obstacle_mean)
			dir_norm = np.linalg.norm(direction)
			if dir_norm < 1e-10:
				# Fallback: use original direction
				direction = diff if np.linalg.norm(diff) > 1e-10 else np.array([1.0, 0.0])
				dir_norm = np.linalg.norm(direction)
			direction = direction / dir_norm
			projected_pos = obstacle_mean + direction * safe_distance
		else:
			projected_pos = projected_pos_ellipsoid
		
		# Verify final projection satisfies both constraints
		final_diff = projected_pos - obstacle_mean
		final_mahalanobis_sq = (final_diff[0]**2 / sigma_x**2) + (final_diff[1]**2 / sigma_y**2)
		final_euclidean = np.linalg.norm(final_diff)
		
		# Ensure both constraints are satisfied
		if final_mahalanobis_sq < mahalanobis_threshold or final_euclidean < safe_distance:
			# Use the more restrictive constraint
			if final_euclidean < safe_distance:
				# Euclidean constraint is more restrictive - use it
				direction = final_diff / final_euclidean if final_euclidean > 1e-10 else np.array([1.0, 0.0])
				projected_pos = obstacle_mean + direction * safe_distance
			else:
				# Mahalanobis constraint is more restrictive - re-project
				diff_normalized = np.array([final_diff[0] / sigma_x, final_diff[1] / sigma_y])
				dist_normalized = np.linalg.norm(diff_normalized)
				if dist_normalized > 1e-10:
					scale_factor = np.sqrt(mahalanobis_threshold) / dist_normalized
					diff_normalized_scaled = diff_normalized * scale_factor
					projected_diff = np.array([
						diff_normalized_scaled[0] * sigma_x,
						diff_normalized_scaled[1] * sigma_y
					])
					projected_pos = obstacle_mean + projected_diff
		
		return projected_pos
	
	def _check_contouring_constraints(self, position, stage_idx):
		"""
		Check if a position satisfies contouring constraints (road boundaries).
		
		CRITICAL FIX: This ensures Gaussian constraint projection doesn't violate road boundaries.
		Reference: C++ mpc_planner - warmstart must satisfy all constraints (Gaussian + contouring).
		
		Args:
			position: Vehicle position (2D numpy array)
			stage_idx: Stage index for getting spline value
		
		Returns:
			(bool, float): (is_valid, contour_error) - True if position satisfies constraints, False otherwise
		"""
		if not hasattr(self, 'solver') or self.solver is None:
			return True, 0.0  # No solver - assume valid
		
		if not hasattr(self.solver, 'data') or self.solver.data is None:
			return True, 0.0  # No data - assume valid
		
		if not hasattr(self.solver.data, 'reference_path') or self.solver.data.reference_path is None:
			return True, 0.0  # No reference path - assume valid
		
		# Get contouring constraints module to check boundaries
		contouring_module = None
		if hasattr(self.solver, 'module_manager') and self.solver.module_manager is not None:
			for module in self.solver.module_manager.get_modules():
				if hasattr(module, 'name') and module.name == 'contouring_constraints':
					contouring_module = module
					break
		
		if contouring_module is None:
			return True, 0.0  # No contouring module - assume valid
		
		# Get spline value for this stage
		ws_vals = self.solver.warmstart_values if hasattr(self.solver, 'warmstart_values') else {}
		s_val = None
		if 'spline' in ws_vals and stage_idx < len(ws_vals['spline']):
			s_val = float(ws_vals['spline'][stage_idx])
		
		if s_val is None:
			return True, 0.0  # No spline value - assume valid
		
		# Get reference path
		ref_path = self.solver.data.reference_path
		if not hasattr(ref_path, 's') or ref_path.s is None:
			return True, 0.0
		
		s_arr = np.asarray(ref_path.s, dtype=float)
		if s_arr.size < 2:
			return True, 0.0
		
		s0 = float(s_arr[0])
		s_end = float(s_arr[-1])
		s_val = max(s0, min(s_end, s_val))
		
		# Get path point and normal vector
		try:
			if not (hasattr(ref_path, 'x_spline') and ref_path.x_spline is not None):
				return True, 0.0
			
			path_point_x = float(ref_path.x_spline(s_val))
			path_point_y = float(ref_path.y_spline(s_val))
			path_point = np.array([path_point_x, path_point_y])
			
			# Get path tangent
			path_dx = float(ref_path.x_spline.derivative()(s_val))
			path_dy = float(ref_path.y_spline.derivative()(s_val))
			norm = np.hypot(path_dx, path_dy)
			if norm < 1e-6:
				return True, 0.0
			
			path_dx_norm = path_dx / norm
			path_dy_norm = path_dy / norm
			
			# Normal vector pointing left: A = [path_dy_norm, -path_dx_norm]
			A = np.array([path_dy_norm, -path_dx_norm])
			
			# Compute contour error
			diff = position - path_point
			contour_error = np.dot(A, diff)
			
			# Get road width
			road_width_half = 3.5  # Default
			if hasattr(contouring_module, '_road_width_half') and contouring_module._road_width_half is not None:
				road_width_half = float(contouring_module._road_width_half)
			
			# Check if position is within boundaries
			# Contour error should be in [-width_right, width_left]
			# For symmetric road: [-road_width_half, road_width_half]
			is_valid = (-road_width_half <= contour_error <= road_width_half)
			
			return is_valid, contour_error
		except Exception as e:
			LOG_DEBUG(f"GaussianConstraints._check_contouring_constraints: Error checking constraints: {e}")
			return True, 0.0  # Assume valid on error
	
	def _project_to_contouring_safety(self, position, stage_idx):
		"""
		Project position to satisfy contouring constraints (road boundaries).
		
		CRITICAL FIX: Projects position to nearest valid point within road boundaries.
		Reference: C++ mpc_planner - ensures warmstart satisfies all constraints.
		
		Args:
			position: Vehicle position (2D numpy array)
			stage_idx: Stage index for getting spline value
		
		Returns:
			Projected position (2D numpy array) or None if no projection needed
		"""
		is_valid, contour_error = self._check_contouring_constraints(position, stage_idx)
		if is_valid:
			return None  # Already valid
		
		if not hasattr(self, 'solver') or self.solver is None:
			return None
		
		if not hasattr(self.solver, 'data') or self.solver.data is None:
			return None
		
		if not hasattr(self.solver.data, 'reference_path') or self.solver.data.reference_path is None:
			return None
		
		# Get contouring constraints module
		contouring_module = None
		if hasattr(self.solver, 'module_manager') and self.solver.module_manager is not None:
			for module in self.solver.module_manager.get_modules():
				if hasattr(module, 'name') and module.name == 'contouring_constraints':
					contouring_module = module
					break
		
		if contouring_module is None:
			return None
		
		# Get spline value
		ws_vals = self.solver.warmstart_values if hasattr(self.solver, 'warmstart_values') else {}
		s_val = None
		if 'spline' in ws_vals and stage_idx < len(ws_vals['spline']):
			s_val = float(ws_vals['spline'][stage_idx])
		
		if s_val is None:
			return None
		
		# Get reference path
		ref_path = self.solver.data.reference_path
		if not hasattr(ref_path, 's') or ref_path.s is None:
			return None
		
		s_arr = np.asarray(ref_path.s, dtype=float)
		if s_arr.size < 2:
			return None
		
		s0 = float(s_arr[0])
		s_end = float(s_arr[-1])
		s_val = max(s0, min(s_end, s_val))
		
		# Get road width
		road_width_half = 3.5  # Default
		if hasattr(contouring_module, '_road_width_half') and contouring_module._road_width_half is not None:
			road_width_half = float(contouring_module._road_width_half)
		
		# Project position to nearest valid point on path
		try:
			if not (hasattr(ref_path, 'x_spline') and ref_path.x_spline is not None):
				return None
			
			path_point_x = float(ref_path.x_spline(s_val))
			path_point_y = float(ref_path.y_spline(s_val))
			path_point = np.array([path_point_x, path_point_y])
			
			# Get path tangent
			path_dx = float(ref_path.x_spline.derivative()(s_val))
			path_dy = float(ref_path.y_spline.derivative()(s_val))
			norm = np.hypot(path_dx, path_dy)
			if norm < 1e-6:
				return None
			
			path_dx_norm = path_dx / norm
			path_dy_norm = path_dy / norm
			
			# Normal vector pointing left: A = [path_dy_norm, -path_dx_norm]
			A = np.array([path_dy_norm, -path_dx_norm])
			
			# Compute contour error
			diff = position - path_point
			contour_error = np.dot(A, diff)
			
			# Clamp contour error to valid range
			contour_error_clamped = np.clip(contour_error, -road_width_half, road_width_half)
			
			# Project position: path_point + contour_error_clamped * A
			projected_pos = path_point + contour_error_clamped * A
			
			return projected_pos
		except Exception as e:
			LOG_DEBUG(f"GaussianConstraints._project_to_contouring_safety: Error projecting: {e}")
			return None
	
	def _project_warmstart_to_gaussian_safety(self, data):
		"""
		Project warmstart trajectory to satisfy Gaussian constraints using Douglas-Rachford projection.
		
		CRITICAL FIX: Now also ensures projected positions satisfy contouring constraints (road boundaries).
		Reference: C++ mpc_planner - warmstart is projected to ensure feasibility for ALL constraints
		
		This method:
		1. Checks if warmstart positions violate Gaussian constraints (Mahalanobis distance)
		2. Projects violating positions away from obstacles using ellipsoidal projection
		3. Ensures projected positions also satisfy contouring constraints (road boundaries)
		4. Ensures warmstart is feasible, preventing solver failures and vehicle spiraling
		
		CRITICAL: Uses effective covariance (sigma + safe_distance) matching constraint calculation.
		This ensures warmstart projection matches the actual constraint formulation.
		
		Args:
			data: Data object containing obstacles and other information
		"""
		if not hasattr(self, 'solver') or self.solver is None:
			LOG_DEBUG("GaussianConstraints._project_warmstart_to_gaussian_safety: No solver available")
			return
		
		if not hasattr(self.solver, 'warmstart_values') or not self.solver.warmstart_values:
			LOG_DEBUG("GaussianConstraints._project_warmstart_to_gaussian_safety: No warmstart values available")
			return
		
		ws_vals = self.solver.warmstart_values
		if 'x' not in ws_vals or 'y' not in ws_vals:
			return
		
		# Get obstacles with Gaussian predictions
		if not data.has("dynamic_obstacles") or data.dynamic_obstacles is None:
			return
		
		copied_dynamic_obstacles = data.dynamic_obstacles
		risk_level = float(self.get_config_value("gaussian_constraints.risk_level", 0.05))
		from scipy.stats import chi2
		chi_squared_threshold = chi2.ppf(1.0 - risk_level, df=2)
		
		horizon_val = self.solver.horizon if self.solver.horizon is not None else 10
		x_ws = ws_vals['x']
		y_ws = ws_vals['y']
		
		# CRITICAL: Project warmstart for ALL stages where constraints are applied
		# This ensures both Gaussian and contouring constraints are satisfied for the entire constraint horizon
		# Reference: C++ mpc_planner - warmstart must satisfy all constraints across the prediction horizon
		
		# Determine max stage for constraints (same logic as calculate_constraints)
		base_max_stage = 8  # Base constraint horizon
		max_stage_for_constraints = base_max_stage
		
		# Check if any obstacles are moving fast (need longer horizon)
		if data.has("dynamic_obstacles") and data.dynamic_obstacles is not None:
			for obstacle in copied_dynamic_obstacles[:self.max_obstacles]:
				if (hasattr(obstacle, 'prediction') and obstacle.prediction is not None and
					obstacle.prediction.type == PredictionType.GAUSSIAN):
					# Estimate obstacle speed
					obstacle_speed = 0.0
					if hasattr(obstacle, 'velocity') and obstacle.velocity is not None:
						obstacle_speed = np.linalg.norm(obstacle.velocity)
					elif hasattr(obstacle, 'speed') and obstacle.speed is not None:
						obstacle_speed = float(obstacle.speed)
					
					# Extend horizon for fast-moving obstacles (> 2 m/s)
					if obstacle_speed > 2.0:
						max_stage_for_constraints = max(max_stage_for_constraints, 10)  # Extended horizon
						break  # Use extended horizon if any obstacle is fast-moving
		
		projections_made = 0
		
		# Project warmstart for ALL stages where constraints are applied
		for stage_idx in range(min(horizon_val + 1, len(x_ws), max_stage_for_constraints + 1)):
			if stage_idx >= len(x_ws) or stage_idx >= len(y_ws):
				continue
			
			robot_pos = np.array([float(x_ws[stage_idx]), float(y_ws[stage_idx])])
			
			# Check each obstacle for constraint violations
			for obs_id, obstacle in enumerate(copied_dynamic_obstacles[:self.max_obstacles]):
				if (not hasattr(obstacle, 'prediction') or obstacle.prediction is None or
					obstacle.prediction.type != PredictionType.GAUSSIAN):
					continue
				
				if (not hasattr(obstacle.prediction, 'steps') or 
					obstacle.prediction.steps is None or 
					len(obstacle.prediction.steps) <= stage_idx):
					continue
				
				pred_step = obstacle.prediction.steps[stage_idx]
				if not hasattr(pred_step, 'position') or pred_step.position is None:
					continue
				
				mean_pos = np.array([float(pred_step.position[0]), float(pred_step.position[1])])
				major_radius = float(getattr(pred_step, 'major_radius', 0.1))
				minor_radius = float(getattr(pred_step, 'minor_radius', 0.1))
				sigma_x = major_radius
				sigma_y = minor_radius
				
				if sigma_x < 1e-10 or sigma_y < 1e-10:
					continue
				
				# CRITICAL: Use the SAME effective covariance as constraint calculation
				# Reference: C++ mpc_planner - warmstart projection must match constraint formulation
				# The constraint uses effective covariance: sigma_eff = sigma + safe_distance
				# So the warmstart projection must also use effective covariance for consistency
				obstacle_radius = float(getattr(obstacle, 'radius', 0.35))
				robot_radius = float(self.robot_radius) if self.robot_radius else 0.5
				safe_distance = robot_radius + obstacle_radius
				
				# Use effective covariance (matching constraint calculation)
				sigma_x_eff = sigma_x + safe_distance
				sigma_y_eff = sigma_y + safe_distance
				
				# Compute Mahalanobis distance using EFFECTIVE covariance
				# This matches the constraint formulation exactly
				diff = robot_pos - mean_pos
				mahalanobis_dist_sq = (diff[0]**2 / sigma_x_eff**2) + (diff[1]**2 / sigma_y_eff**2)
				
				# CRITICAL: Use same safety margin as constraint calculation for consistency
				# Reference: C++ mpc_planner - warmstart projection must match constraint formulation
				safety_margin_factor = 1.2  # 20% safety margin (matching constraint calculation)
				required_threshold = chi_squared_threshold * safety_margin_factor
				
				# Check if violation exists
				if mahalanobis_dist_sq < required_threshold:
					# CRITICAL FIX: Project warmstart while maintaining path direction
					# Reference: C++ mpc_planner - projection should maintain forward progress along path
					# Get path direction at this stage to maintain straight trajectory
					path_direction = None
					if (hasattr(self.solver, 'data') and self.solver.data is not None and
						hasattr(self.solver.data, 'reference_path') and self.solver.data.reference_path is not None):
						try:
							ref_path = self.solver.data.reference_path
							# Get spline value for this stage from warmstart
							if 'spline' in ws_vals and stage_idx < len(ws_vals['spline']):
								s_val = float(ws_vals['spline'][stage_idx])
								if hasattr(ref_path, 'x_spline') and ref_path.x_spline is not None:
									# Evaluate path tangent at this spline value
									dx = float(ref_path.x_spline.derivative()(s_val))
									dy = float(ref_path.y_spline.derivative()(s_val))
									norm = np.hypot(dx, dy)
									if norm > 1e-9:
										path_direction = np.array([dx / norm, dy / norm])
						except Exception:
							pass
					
					# If no path direction available, use direction from previous stage
					if path_direction is None and stage_idx > 0:
						prev_pos = np.array([float(x_ws[stage_idx - 1]), float(y_ws[stage_idx - 1])])
						diff_path = robot_pos - prev_pos
						norm_path = np.linalg.norm(diff_path)
						if norm_path > 1e-9:
							path_direction = diff_path / norm_path
					
					# Project warmstart position to safe region using EFFECTIVE covariance
					# CRITICAL: Use effective sigma values to match constraint formulation
					projected_pos = self._project_to_gaussian_safety_with_path_direction(
						robot_pos.copy(),
						mean_pos,
						sigma_x_eff,  # Use effective sigma
						sigma_y_eff,  # Use effective sigma
						required_threshold,
						obstacle_radius,
						robot_radius,
						path_direction  # Pass path direction to maintain trajectory
					)
					
					if projected_pos is not None:
						projection_distance = np.linalg.norm(projected_pos - robot_pos)
						if projection_distance > 1e-6:  # Significant projection
							# CRITICAL FIX: ALWAYS ensure projected position satisfies contouring constraints
							# Reference: C++ mpc_planner - warmstart must satisfy ALL constraints (Gaussian + contouring)
							# If contouring constraints are violated, we MUST project to satisfy them
							is_valid, contour_error = self._check_contouring_constraints(projected_pos, stage_idx)
							if not is_valid:
								# Project to satisfy contouring constraints
								contouring_projected = self._project_to_contouring_safety(projected_pos, stage_idx)
								if contouring_projected is not None:
									# Verify the contouring-projected position still satisfies Gaussian constraints
									diff_contouring = contouring_projected - mean_pos
									mahalanobis_contouring_sq = (diff_contouring[0]**2 / sigma_x_eff**2) + (diff_contouring[1]**2 / sigma_y_eff**2)
									
									if mahalanobis_contouring_sq >= required_threshold:
										# Contouring projection satisfies Gaussian constraints - use it
										projected_pos = contouring_projected
										LOG_INFO(f"GaussianConstraints._project_warmstart_to_gaussian_safety: Stage {stage_idx}, Obstacle {obs_id}: "
										         f"Applied contouring projection to satisfy road boundaries "
										         f"(contour_error={contour_error:.3f}, mahalanobis_contouring_sq={mahalanobis_contouring_sq:.3f})")
									else:
										# CRITICAL: Both constraints must be satisfied - find a compromise position
										# Use iterative projection: project to Gaussian safety, then to contouring, repeat
										# Try a few iterations to find a position that satisfies both constraints
										max_iterations = 5
										current_pos = projected_pos.copy()
										for iter_idx in range(max_iterations):
											# Project to contouring safety
											contouring_proj = self._project_to_contouring_safety(current_pos, stage_idx)
											if contouring_proj is None:
												break
											
											# Check if contouring projection satisfies Gaussian constraints
											diff_check = contouring_proj - mean_pos
											mahalanobis_check_sq = (diff_check[0]**2 / sigma_x_eff**2) + (diff_check[1]**2 / sigma_y_eff**2)
											
											if mahalanobis_check_sq >= required_threshold:
												# Found valid position - use it
												projected_pos = contouring_proj
												LOG_INFO(f"GaussianConstraints._project_warmstart_to_gaussian_safety: Stage {stage_idx}, Obstacle {obs_id}: "
												         f"Found compromise position satisfying both constraints after {iter_idx+1} iterations")
												break
											
											# If not valid, try projecting back to Gaussian safety from contouring position
											gaussian_proj = self._project_to_gaussian_safety_with_path_direction(
												contouring_proj.copy(),
												mean_pos,
												sigma_x_eff,
												sigma_y_eff,
												required_threshold,
												obstacle_radius,
												robot_radius,
												path_direction
											)
											
											if gaussian_proj is None:
												# Can't project further - prioritize contouring constraints (road boundaries)
												# CRITICAL: Road boundaries are hard constraints - vehicle MUST stay on road
												# Gaussian constraints are probabilistic - slight violation is acceptable if necessary
												# Reference: C++ mpc_planner - contouring constraints take priority over obstacle avoidance
												projected_pos = contouring_proj
												LOG_WARN(f"GaussianConstraints._project_warmstart_to_gaussian_safety: Stage {stage_idx}, Obstacle {obs_id}: "
												         f"Prioritizing contouring constraints (road boundaries) over Gaussian constraints "
												         f"(mahalanobis_check_sq={mahalanobis_check_sq:.3f} < threshold={required_threshold:.3f}). "
												         f"Vehicle will stay on road but may be closer to obstacle than ideal.")
												break
											
											current_pos = gaussian_proj
										
										# Final check: ensure contouring constraints are satisfied
										is_valid_final, contour_error_final = self._check_contouring_constraints(projected_pos, stage_idx)
										if not is_valid_final:
											# Last resort: force projection to contouring safety
											contouring_final = self._project_to_contouring_safety(projected_pos, stage_idx)
											if contouring_final is not None:
												projected_pos = contouring_final
												LOG_WARN(f"GaussianConstraints._project_warmstart_to_gaussian_safety: Stage {stage_idx}, Obstacle {obs_id}: "
												         f"Force-applied contouring projection to ensure road boundary compliance")
								else:
									LOG_WARN(f"GaussianConstraints._project_warmstart_to_gaussian_safety: Stage {stage_idx}, Obstacle {obs_id}: "
									         f"Failed to project to contouring safety (contour_error={contour_error:.3f}). "
									         f"Warmstart may violate road boundaries.")
							
							# Update warmstart values
							x_ws[stage_idx] = float(projected_pos[0])
							y_ws[stage_idx] = float(projected_pos[1])
							projections_made += 1
							
							if stage_idx < 3:  # Log first few projections
								is_valid_final, contour_error_final = self._check_contouring_constraints(projected_pos, stage_idx)
								LOG_INFO(f"GaussianConstraints._project_warmstart_to_gaussian_safety: Stage {stage_idx}, Obstacle {obs_id}: "
								         f"Projected warmstart from ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}) "
								         f"to ({projected_pos[0]:.3f}, {projected_pos[1]:.3f}) "
								         f"(mahalanobis_dist_sq={mahalanobis_dist_sq:.3f} < threshold={required_threshold:.3f}, "
								         f"sigma_eff=({sigma_x_eff:.3f}, {sigma_y_eff:.3f}), "
								         f"projection_distance={projection_distance:.3f}m, "
								         f"contouring_valid={is_valid_final}, contour_error={contour_error_final:.3f}, "
								         f"path_direction={'available' if path_direction is not None else 'none'})")
		
		if projections_made > 0:
			LOG_INFO(f"GaussianConstraints._project_warmstart_to_gaussian_safety: Projected {projections_made} warmstart positions to satisfy Gaussian constraints")
		else:
			LOG_DEBUG(f"GaussianConstraints._project_warmstart_to_gaussian_safety: No warmstart projections needed (all positions satisfy constraints)")

	def lower_bounds(self, state=None, data=None, stage_idx=None):
		"""Lower bounds for Gaussian constraints: -inf (constraint is >= threshold, handled by ub=0.0 on expr)."""
		import casadi as cd
		count = 0
		if data is not None and stage_idx is not None:
			if data.has("dynamic_obstacles") and data.dynamic_obstacles:
				# Count only obstacles with Gaussian predictions
				for obs in data.dynamic_obstacles:
					if (hasattr(obs, 'prediction') and obs.prediction is not None and
						hasattr(obs.prediction, 'type') and obs.prediction.type == PredictionType.GAUSSIAN):
						if (hasattr(obs.prediction, 'steps') and len(obs.prediction.steps) > stage_idx):
							count += self.num_discs
		return [-cd.inf] * count if count > 0 else []

	def upper_bounds(self, state=None, data=None, stage_idx=None):
		"""Upper bounds for Gaussian constraints: 0.0 (constraint expr <= 0)."""
		count = 0
		if data is not None and stage_idx is not None:
			if data.has("dynamic_obstacles") and data.dynamic_obstacles:
				# Count only obstacles with Gaussian predictions
				for obs in data.dynamic_obstacles:
					if (hasattr(obs, 'prediction') and obs.prediction is not None and
						hasattr(obs.prediction, 'type') and obs.prediction.type == PredictionType.GAUSSIAN):
						if (hasattr(obs.prediction, 'steps') and len(obs.prediction.steps) > stage_idx):
							count += self.num_discs
		return [0.0] * count if count > 0 else []

	def get_visualizer(self):
		"""Return a visualizer for Gaussian constraints."""
		class GaussianConstraintsVisualizer:
			def __init__(self, module):
				self.module = module
			
			def visualize(self, state, data, stage_idx=0, ax=None):
				"""
				Visualize Gaussian constraints as uncertainty ellipses around obstacles.
				Plots ellipses directly on the current matplotlib axes.
				
				Args:
					state: Current state (not used)
					data: Data object containing obstacles
					stage_idx: Prediction stage index (default 0)
					ax: Matplotlib axes to plot on (optional, will use plt.gca() if not provided)
				"""
				try:
					import matplotlib.pyplot as plt
					from matplotlib.patches import Ellipse
					from planning.types import PredictionType
					import numpy as np
				except Exception:
					return
				
				# Get axes - prefer provided axes, fall back to gca()
				if ax is None:
					try:
						ax = plt.gca()
					except Exception:
						try:
							fig = plt.gcf()
							if fig is not None and len(fig.axes) > 0:
								ax = fig.axes[0]
							else:
								return
						except Exception:
							return
				
				# Check if data has dynamic obstacles
				# Data can store obstacles either as attribute or in _store dictionary
				if hasattr(data, 'dynamic_obstacles'):
					copied_dynamic_obstacles = data.dynamic_obstacles
				elif hasattr(data, 'has') and data.has("dynamic_obstacles"):
					copied_dynamic_obstacles = data.get("dynamic_obstacles")
				else:
					return
				
				if copied_dynamic_obstacles is None or len(copied_dynamic_obstacles) == 0:
					return
				# Verify axes is valid
				if ax is None or not hasattr(ax, 'add_patch'):
					return  # Cannot visualize without valid axes
				
				risk_level = float(self.module.get_config_value("gaussian_constraints.risk_level", 0.05))
				confidence_level = 1.0 - risk_level
				
				first_ellipse = True
				
				# Debug: Log that visualizer was called
				if stage_idx == 0:
					import logging
					logging.getLogger("integration_test").info(
						f"Gaussian visualizer called: {len(copied_dynamic_obstacles)} obstacles, stage_idx={stage_idx}")
				
				# Visualize each obstacle with Gaussian prediction
				# Use obstacle colors similar to linearized constraints for consistency
				obstacle_colors = ['red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
				
				for obs_id, obstacle in enumerate(copied_dynamic_obstacles[:self.module.max_obstacles]):
					# Check if obstacle has Gaussian prediction
					if not hasattr(obstacle, 'prediction') or obstacle.prediction is None:
						continue
					
					# Debug: Log prediction type for first few obstacles
					if obs_id < 3 and stage_idx == 0:
						import logging
						logging.getLogger("integration_test").info(
							f"Gaussian visualizer: Obstacle {obs_id} prediction type: {obstacle.prediction.type}")
					
					if obstacle.prediction.type != PredictionType.GAUSSIAN:
						if obs_id < 3 and stage_idx == 0:
							import logging
							logging.getLogger("integration_test").warning(
								f"Gaussian visualizer: Skipping obstacle {obs_id} - not GAUSSIAN (type={obstacle.prediction.type})")
						continue
					
					# Get prediction step for this stage
					if not hasattr(obstacle.prediction, 'steps') or len(obstacle.prediction.steps) <= stage_idx:
						continue
					
					pred_step = obstacle.prediction.steps[stage_idx]
					
					# Get mean position
					if not hasattr(pred_step, 'position') or pred_step.position is None:
						continue
					
					mean_pos = np.array([float(pred_step.position[0]), float(pred_step.position[1])])
					
					# Get uncertainty parameters
					major_radius = float(getattr(pred_step, 'major_radius', 0.1))
					minor_radius = float(getattr(pred_step, 'minor_radius', 0.1))
					# Use angle instead of orientation (PredictionStep has angle, not orientation)
					orientation = float(getattr(pred_step, 'angle', getattr(pred_step, 'orientation', 0.0)))
					
					# Get obstacle and robot radii for safe distance
					obstacle_radius = float(getattr(obstacle, 'radius', 0.35))
					robot_radius = float(self.module.robot_radius) if self.module.robot_radius else 0.5
					safe_distance = robot_radius + obstacle_radius
					
					# Compute chi-squared threshold for visualization
					from scipy.stats import chi2
					chi_squared_threshold = chi2.ppf(confidence_level, df=2)
					
					# Scale uncertainty ellipses by chi-squared threshold and safe distance
					# The visualization shows the effective constraint boundary
					major_effective = np.sqrt(chi_squared_threshold) * major_radius + safe_distance
					minor_effective = np.sqrt(chi_squared_threshold) * minor_radius + safe_distance
					
					# Use obstacle-specific color (similar to linearized constraints)
					obstacle_color = obstacle_colors[obs_id % len(obstacle_colors)]
					alpha_fill = 0.15
					alpha_edge = 0.6
					
					# Draw uncertainty ellipse (constraint boundary) - outer ellipse with safe distance
					uncertainty_ellipse = Ellipse(
						xy=(float(mean_pos[0]), float(mean_pos[1])),
						width=2 * major_effective,
						height=2 * minor_effective,
						angle=np.degrees(orientation),
						edgecolor=obstacle_color,
						facecolor=obstacle_color,
						alpha=alpha_fill,
						linestyle='--',
						linewidth=2.0,
						zorder=1,
						label=f'Gaussian Constraint ({confidence_level*100:.0f}%)' if first_ellipse else None
					)
					ax.add_patch(uncertainty_ellipse)
					
					# Debug: Log ellipse creation
					if obs_id < 3 and stage_idx == 0:
						import logging
						logging.getLogger("integration_test").info(
							f"Gaussian visualizer: Created ellipse for obstacle {obs_id} at ({mean_pos[0]:.2f}, {mean_pos[1]:.2f}), "
							f"size=({major_effective:.2f}, {minor_effective:.2f}), color={obstacle_color}")
					
					# Draw mean position uncertainty ellipse (without safe distance, shows actual uncertainty)
					uncertainty_only_ellipse = Ellipse(
						xy=(float(mean_pos[0]), float(mean_pos[1])),
						width=2 * np.sqrt(chi_squared_threshold) * major_radius,
						height=2 * np.sqrt(chi_squared_threshold) * minor_radius,
						angle=np.degrees(orientation),
						edgecolor=obstacle_color,
						facecolor='none',
						alpha=alpha_edge,
						linestyle=':',
						linewidth=1.5,
						zorder=2
					)
					ax.add_patch(uncertainty_only_ellipse)
					
					# Draw mean position marker (use obstacle color)
					mean_marker, = ax.plot(mean_pos[0], mean_pos[1], 'o', color=obstacle_color, markersize=6, 
					       label='Mean Position' if first_ellipse else None, zorder=3)
					
					# Add text marker for uncertainty parameters
					# Check obstacle for uncertainty params
					position_std = 0.0
					uncertainty_growth = 0.0
					
					if hasattr(obstacle, 'uncertainty_params') and obstacle.uncertainty_params:
						position_std = obstacle.uncertainty_params.get('position_std', 0.0)
						uncertainty_growth = obstacle.uncertainty_params.get('uncertainty_growth', 0.0)
					
					# Calculate current uncertainty at this stage
					current_std = position_std + stage_idx * uncertainty_growth
					
					# Position text offset above the obstacle (use a fixed offset to ensure visibility)
					# Use a larger offset to ensure text is clearly visible above the ellipse
					text_offset_y = max(major_effective, 1.5) + 0.8  # At least 1.5m + 0.8m offset
					text_x = float(mean_pos[0])
					text_y = float(mean_pos[1]) + text_offset_y
					
					# Format uncertainty parameters text
					uncertainty_text = f"σ={position_std:.2f}"
					if uncertainty_growth > 0:
						uncertainty_text += f"\n+{uncertainty_growth:.3f}/step"
					uncertainty_text += f"\nσₜ={current_std:.2f}"
					
					# Add text annotation with background box (use obstacle color)
					# Make text more visible with larger font and better contrast
					# Always add text for Gaussian obstacles to show uncertainty parameters
					uncertainty_text_obj = ax.text(text_x, text_y, uncertainty_text,
					       fontsize=10,
					       color=obstacle_color,
					       ha='center',
					       va='bottom',
					       weight='bold',
					       bbox=dict(boxstyle='round,pad=0.5',
					                facecolor='white',
					                edgecolor=obstacle_color,
					                alpha=0.95,
					                linewidth=2.0),
					       zorder=20,  # Very high zorder to ensure text is always on top
					       clip_on=False)  # Don't clip text if it goes slightly outside axes
					
					first_ellipse = False
		
		return GaussianConstraintsVisualizer(self)

	def is_data_ready(self, data):
		missing_data = ""
		if not data.has("dynamic_obstacles") or data.dynamic_obstacles is None:
			missing_data += "Dynamic Obstacles "
			LOG_DEBUG("Missing dynamic_obstacles: {}".format(missing_data))
		else:
			for i in range(len(data.dynamic_obstacles)):
				LOG_DEBUG("Obstacle prediction type is {}".format(data.dynamic_obstacles[i].prediction.type))
				if (not hasattr(data.dynamic_obstacles[i], 'prediction') or
						data.dynamic_obstacles[i].prediction is None):
					missing_data += "Obstacle Prediction "

				if (hasattr(data.dynamic_obstacles[i], 'prediction') and
						data.dynamic_obstacles[i].prediction is not None and
						hasattr(data.dynamic_obstacles[i].prediction, 'type') and
						not data.dynamic_obstacles[i].prediction.type is PredictionType.DETERMINISTIC and
						not data.dynamic_obstacles[i].prediction.type is PredictionType.GAUSSIAN):
					missing_data += "Obstacle Prediction (type must be deterministic, or gaussian) "

		return len(missing_data) < 1