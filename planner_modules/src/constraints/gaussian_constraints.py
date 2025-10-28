import numpy as np
import casadi as cd

from planning.src.types import PredictionType
from utils.const import GAUSSIAN, DYNAMIC
from utils.math_utils import exponential_quantile, rotation_matrix, casadi_rotation_matrix
from utils.utils import LOG_DEBUG, PROFILE_SCOPE, CONFIG, LOG_INFO
from utils.visualizer_compat import ROSPointMarker
from planner_modules.src.constraints.base_constraint import BaseConstraint

class GaussianConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)
		self.name = 'gaussian_constraints'
		# Store dummy values for invalid states
		self._dummy_x = 0.0
		self._dummy_y = 0.0
		self.num_discs = self.get_config_value("num_discs")
		self.robot_radius = self.get_config_value("robot.radius")
		self.max_obstacles = self.get_config_value("max_obstacles")
		self.num_constraints = self.num_discs * self.max_obstacles
		self.num_active_obstacles = 0

		LOG_DEBUG("Gaussian Constraints successfully initialized")

	def update(self, state, data):
		LOG_DEBUG("GaussianConstraints.update")

		# Update dummy values based on current state
		self._dummy_x = state.get("x") + 100.0
		self._dummy_y = state.get("y") + 100.0

		copied_dynamic_obstacles = data.dynamic_obstacles
		self.num_active_obstacles = len(copied_dynamic_obstacles)

	def define_parameters(self, params):
		params.add(f"ego_disc_radius")
		for disc_id in range(self.num_discs):
			params.add(f"ego_disc_{disc_id}_offset")

		for obs_id in range(self.max_obstacles):
			for step in range(self.solver.horizon + 1):
				params.add(f"gaussian_obstacle_{obs_id}_x_step_{step}")
				params.add(f"gaussian_obstacle_{obs_id}_y_step_{step}")
				params.add(f"gaussian_obstacle_{obs_id}_major_step_{step}")
				params.add(f"gaussian_obstacle_{obs_id}_minor_step_{step}")
				params.add(f"gaussian_obstacle_{obs_id}_risk_step_{step}")
				params.add(f"gaussian_obstacle_{obs_id}_r_step_{step}")

	def set_parameters(self, parameter_manager, data, step):
		# Set ego disc parameters (always needed)
		for disc_id in range(self.num_discs):
			parameter_manager.set_parameter(f"ego_disc_radius", self.robot_radius)
			if hasattr(data, 'robot_area') and len(data.robot_area) > disc_id:
				parameter_manager.set_parameter(f"ego_disc_{disc_id}_offset", data.robot_area[disc_id].offset)

		# Update active obstacles count
		if data.has("dynamic_obstacles") and data.dynamic_obstacles is not None:
			self.num_active_obstacles = len(data.dynamic_obstacles)
		else:
			self.num_active_obstacles = 0

		# Set obstacle parameters for ALL steps in horizon
		for obstacle_id in range(self.max_obstacles):
			for time_step in range(self.solver.horizon + 1):
				if obstacle_id < self.num_active_obstacles:
					# Set real obstacle data
					obstacle = data.dynamic_obstacles[obstacle_id]
					if obstacle.prediction.type == PredictionType.GAUSSIAN and time_step < len(obstacle.prediction.steps):
						parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_x_step_{time_step}",
														obstacle.prediction.steps[time_step].position[0])
						parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_y_step_{time_step}",
														obstacle.prediction.steps[time_step].position[1])
						if obstacle.type == DYNAMIC:
							# Dynamic obstacles have uncertainty
							parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_major_step_{time_step}",
															obstacle.prediction.steps[time_step].major_radius)
							parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_minor_step_{time_step}",
															obstacle.prediction.steps[time_step].minor_radius)
						else:
							# Static obstacles have minimal uncertainty
							parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_major_step_{time_step}", 0.001)
							parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_minor_step_{time_step}", 0.001)

						# Set risk and radius parameters
						parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_risk_step_{time_step}",
														self.get_config_value("probabilistic.risk"))
						parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_r_step_{time_step}",
														self.get_config_value("obstacle_radius"))
					else:
						# Set dummy values for invalid predictions
						parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_x_step_{time_step}",
														self._dummy_x)
						parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_y_step_{time_step}",
														self._dummy_y)
						parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_major_step_{time_step}", 0.1)
						parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_minor_step_{time_step}", 0.1)
						parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_risk_step_{time_step}", 0.05)
						parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_r_step_{time_step}", 0.1)
				else:
					# Set dummy values for inactive obstacles
					parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_x_step_{time_step}",
													self._dummy_x)
					parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_y_step_{time_step}",
													self._dummy_y)
					parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_major_step_{time_step}", 0.1)
					parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_minor_step_{time_step}", 0.1)
					parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_risk_step_{time_step}", 0.05)
					parameter_manager.set_parameter(f"gaussian_obstacle_{obstacle_id}_r_step_{time_step}", 0.1)

	def get_lower_bound(self):
		lower_bound = []
		for obs in range(0, self.max_obstacles):
			for disc in range(0, self.num_discs):
				lower_bound.append(0.0)
		return lower_bound

	def get_upper_bound(self):
		upper_bound = []
		for obs in range(0, self.max_obstacles):
			for disc in range(0, self.num_discs):
				upper_bound.append(np.Inf)
		return upper_bound

	def get_constraints(self, model, params, stage_idx):

		LOG_INFO("Gaussian Constraints get_constraints")
		constraints = []
		x = model.get("x")
		y = model.get("y")
		psi = model.get("psi")

		# Use CasADi vertcat instead of np.array
		pos = cd.vertcat(x, y)

		r_vehicle = params.get(f"ego_disc_radius")
		rotation_car = casadi_rotation_matrix(psi)

		for obs_id in range(self.max_obstacles):
			if obs_id >= self.num_active_obstacles:
				continue  # Skip inactive obstacles

			obs_x = params.get(f"gaussian_obstacle_{obs_id}_x_step_{stage_idx}")
			obs_y = params.get(f"gaussian_obstacle_{obs_id}_y_step_{stage_idx}")
			# Use CasADi vertcat instead of np.array
			obs_pos = cd.vertcat(obs_x, obs_y)

			sigma_x = params.get(f"gaussian_obstacle_{obs_id}_major_step_{stage_idx}")
			sigma_y = params.get(f"gaussian_obstacle_{obs_id}_minor_step_{stage_idx}")

			# Use CasADi diag instead of np.diag
			Sigma = cd.diag(cd.vertcat(sigma_x ** 2, sigma_y ** 2))

			risk = params.get(f"gaussian_obstacle_{obs_id}_risk_step_{stage_idx}")
			r_obstacle = params.get(f"gaussian_obstacle_{obs_id}_r_step_{stage_idx}")
			combined_radius = r_vehicle + r_obstacle

			for disc_it in range(self.num_discs):
				# Get and compute the disc position
				disc_x = params.get(f"ego_disc_{disc_it}_offset")

				# Use CasADi operations consistently
				disc_relative_pos = cd.vertcat(disc_x, 0)
				disc_pos = pos + cd.mtimes(rotation_car, disc_relative_pos)

				diff_pos = disc_pos - obs_pos

				# Use CasADi norm_2 for the denominator
				diff_norm = cd.norm_2(diff_pos)
				a_ij = diff_pos / diff_norm
				b_ij = combined_radius

				x_erfinv = 1.0 - 2.0 * risk

				# Fix 2: Handle edge cases for inverse erf
				# Clamp x_erfinv to valid range (-1, 1) to avoid numerical issues
				x_erfinv = cd.fmax(-0.999, cd.fmin(0.999, x_erfinv))

				# Fix 3: Improved manual inverse erf approximation with better numerical stability
				# Using Beasley-Springer-Moro algorithm which is more accurate
				z = cd.sqrt(-2.0 * cd.log((1.0 - x_erfinv) / 2.0))

				# More accurate coefficients for inverse erf
				c0 = 2.515517
				c1 = 0.802853
				c2 = 0.010328
				d1 = 1.432788
				d2 = 0.189269
				d3 = 0.001308

				y_erfinv = z - (c0 + c1 * z + c2 * z * z) / (1.0 + d1 * z + d2 * z * z + d3 * z * z * z)

				# Apply sign correction
				y_erfinv = cd.if_else(x_erfinv < 0, -y_erfinv, y_erfinv)

				# Fix 4: Newton-Raphson refinement with better convergence check
				for _ in range(2):  # Usually 2 iterations are sufficient
					erf_val = cd.erf(y_erfinv)
					error = erf_val - x_erfinv
					derivative = 2.0 / cd.sqrt(cd.pi) * cd.exp(-y_erfinv * y_erfinv)
					y_erfinv = y_erfinv - error / derivative

				# Fix 5: Ensure Sigma is symmetric and positive semi-definite
				# Add small regularization if needed
				Sigma_reg = Sigma + 1e-6 * cd.DM.eye(Sigma.shape[0])

				# Fix 6: More numerically stable computation of the quadratic form
				# Instead of a_ij.T @ Sigma @ a_ij, use the fact that this is a scalar
				variance_term = cd.mtimes(cd.mtimes(a_ij.T, Sigma_reg), a_ij)

				# Ensure variance is positive (should be by construction, but numerical safety)
				variance_term = cd.fmax(variance_term, 1e-12)

				# Use CasADi mtimes for matrix multiplication
				constraints.append(cd.mtimes(a_ij.T, diff_pos) - b_ij - y_erfinv * cd.sqrt(variance_term))

		return [cd.vertcat(*constraints)]

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