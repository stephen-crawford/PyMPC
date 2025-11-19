import numpy as np
import casadi as cd

from planning.types import PredictionType
from utils.const import GAUSSIAN, DYNAMIC
from utils.math_tools import exponential_quantile, rotation_matrix, casadi_rotation_matrix
from utils.utils import LOG_DEBUG, PROFILE_SCOPE, CONFIG, LOG_INFO
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

		LOG_DEBUG(f"Gaussian Constraints successfully initialized: num_discs={self.num_discs}, max_obstacles={self.max_obstacles}, robot_radius={self.robot_radius}")

	def update(self, state, data):
		LOG_DEBUG("GaussianConstraints.update")

		# Update dummy values based on current state
		self._dummy_x = state.get("x") + 100.0
		self._dummy_y = state.get("y") + 100.0

		copied_dynamic_obstacles = data.dynamic_obstacles
		self.num_active_obstacles = len(copied_dynamic_obstacles)

	def calculate_constraints(self, state, data, stage_idx):
		"""
		Calculate Gaussian constraints symbolically for probabilistic obstacle avoidance.
		
		Constraint formulation: (p - μ)^T * Σ^(-1) * (p - μ) >= χ²(α)
		where:
			- p is the vehicle disc position
			- μ is the mean obstacle position (from prediction)
			- Σ is the covariance matrix (uncertainty)
			- χ²(α) is the chi-squared quantile for risk level α
		
		This ensures P(||p - μ|| <= safe_distance) <= α (chance constraint).
		
		Reference: https://github.com/tud-amr/mpc_planner
		"""
		constraints = []
		
		# Get symbolic vehicle position
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
			
			# Compute disc position symbolically
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
				
				# Get prediction step for this stage
				if not hasattr(obstacle.prediction, 'steps') or len(obstacle.prediction.steps) <= stage_idx:
					continue
				
				pred_step = obstacle.prediction.steps[stage_idx]
				
				# Get mean position (μ)
				if not hasattr(pred_step, 'position') or pred_step.position is None:
					continue
				
				mean_pos = np.array([float(pred_step.position[0]), float(pred_step.position[1])])
				
				# Get covariance matrix (Σ)
				# Use major/minor radii to construct covariance
				major_radius = float(getattr(pred_step, 'major_radius', 0.1))
				minor_radius = float(getattr(pred_step, 'minor_radius', 0.1))
				orientation = float(getattr(pred_step, 'orientation', 0.0))
				
				# Construct covariance matrix from uncertainty ellipsoid
				# For simplicity, use diagonal covariance scaled by radii
				# More sophisticated: use rotation matrix for oriented uncertainty
				sigma_x = major_radius
				sigma_y = minor_radius
				
				# Create covariance matrix Σ
				cov_matrix = np.array([[sigma_x**2, 0.0], [0.0, sigma_y**2]])
				
				# Apply rotation if orientation is non-zero
				if abs(orientation) > 1e-6:
					cos_theta = np.cos(orientation)
					sin_theta = np.sin(orientation)
					rotation = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
					cov_matrix = rotation @ cov_matrix @ rotation.T
				
				# Compute inverse covariance Σ^(-1)
				try:
					inv_cov_matrix = np.linalg.inv(cov_matrix)
				except np.linalg.LinAlgError:
					# If singular, use pseudo-inverse or add small regularization
					inv_cov_matrix = np.linalg.pinv(cov_matrix + np.eye(2) * 1e-6)
				
				# Convert to CasADi symbolic matrix
				inv_cov_sym = cd.DM(inv_cov_matrix)
				
				# Get safe distance (robot radius + obstacle radius)
				obstacle_radius = float(getattr(obstacle, 'radius', 0.35))
				robot_radius = float(self.robot_radius) if self.robot_radius else 0.5
				safe_distance = robot_radius + obstacle_radius
				
				# Compute chi-squared quantile for risk level α
				# For 2D: χ²(α) with 2 degrees of freedom
				from scipy.stats import chi2
				chi_squared_threshold = chi2.ppf(1.0 - risk_level, df=2)
				
				# Adjust threshold to account for safe distance
				# Scale threshold by safe_distance^2 to ensure physical separation
				scaled_threshold = chi_squared_threshold * (safe_distance**2) / (sigma_x * sigma_y)
				
				# Compute constraint: (p - μ)^T * Σ^(-1) * (p - μ) >= scaled_threshold
				disc_to_mean = disc_pos_sym - cd.vertcat(mean_pos[0], mean_pos[1])
				mahalanobis_dist_sq = disc_to_mean.T @ inv_cov_sym @ disc_to_mean
				
				# Constraint: mahalanobis_dist_sq >= scaled_threshold
				# i.e., scaled_threshold - mahalanobis_dist_sq <= 0
				constraint_expr = scaled_threshold - mahalanobis_dist_sq
				
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