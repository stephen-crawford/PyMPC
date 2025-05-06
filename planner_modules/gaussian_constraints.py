import numpy as np
import casadi as cd
from utils.const import CONSTRAINT, GAUSSIAN, DYNAMIC
from utils.utils import LOG_DEBUG, PROFILE_SCOPE, exponential_quantile, CONFIG, rotation_matrix
from utils.visualizer import ROSPointMarker
from planner_modules.base_constraint import BaseConstraint


class GaussianConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)
		self.name = 'gaussian_constraints'
		# Store dummy values for invalid states
		self._dummy_x = 100.0
		self._dummy_y = 100.0

		LOG_DEBUG("Gaussian Constraints successfully initialized")

	def update(self, state, data, module_data):
		LOG_DEBUG("GaussianConstraints.update")

		# Update dummy values based on current state
		self._dummy_x = state.get("x") + 100.0
		self._dummy_y = state.get("y") + 100.0

	def define_parameters(self, params):
		params.add("ego_disc_radius")

		for disc_id in range(self.n_discs):
			params.add(f"ego_disc_{disc_id}_offset", bundle_name="ego_disc_offset")

		for obs_id in range(self.max_obstacles):
			params.add(f"gaussian_obst_{obs_id}_x", bundle_name="gaussian_obst_x")
			params.add(f"gaussian_obst_{obs_id}_y", bundle_name="gaussian_obst_y")
			params.add(f"gaussian_obst_{obs_id}_major", bundle_name="gaussian_obst_major")
			params.add(f"gaussian_obst_{obs_id}_minor", bundle_name="gaussian_obst_minor")
			params.add(f"gaussian_obst_{obs_id}_risk", bundle_name="gaussian_obst_risk")
			params.add(f"gaussian_obst_{obs_id}_r", bundle_name="gaussian_obst_r")

	def set_parameters(self, data, module_data, k):
		# Set robot parameters
		self.set_solver_parameter("ego_disc_radius", self.get_config_value("robot.radius"), k)

		for d in range(self.get_config_value("n_discs")):
			if hasattr(data, 'robot_area') and len(data.robot_area) > d:
				self.set_solver_parameter("ego_disc_offset", data.robot_area[d].offset, k, d)

		if k == 0:  # Dummies
			for i in range(data.dynamic_obstacles.size()):
				self.set_solver_parameter("gaussian_obstacle_x", self._dummy_x, k, i)
				self.set_solver_parameter("gaussian_obstacle_y", self._dummy_y, k, i)
				self.set_solver_parameter("gaussian_obstacle_major", 0.1, k, i)
				self.set_solver_parameter("gaussian_obstacle_minor", 0.1, k, i)
				self.set_solver_parameter("gaussian_obstacle_risk", 0.05, k, i)
				self.set_solver_parameter("gaussian_obstacle_r", 0.1, k, i)
			return

		if k == 1:
			LOG_DEBUG("GaussianConstraints::set_parameters")

		# Set obstacle parameters
		for i in range(data.dynamic_obstacles.size()):
			obstacle = data.dynamic_obstacles[i]

			if obstacle.prediction.type == GAUSSIAN:
				# Set position parameters
				self.set_solver_parameter(
					"gaussian_obstacle_x",
					obstacle.prediction.modes[0][k - 1].position[0],
					k, i
				)
				self.set_solver_parameter(
					"gaussian_obstacle_y",
					obstacle.prediction.modes[0][k - 1].position[1],
					k, i
				)

				if obstacle.type == DYNAMIC:
					# Dynamic obstacles have uncertainty
					self.set_solver_parameter(
						"gaussian_obstacle_major",
						obstacle.prediction.modes[0][k - 1].major_radius,
						k, i
					)
					self.set_solver_parameter(
						"gaussian_obstacle_minor",
						obstacle.prediction.modes[0][k - 1].minor_radius,
						k, i
					)
				else:
					# Static obstacles have minimal uncertainty
					self.set_solver_parameter("gaussian_obstacle_major", 0.001, k, i)
					self.set_solver_parameter("gaussian_obstacle_minor", 0.001, k, i)

				# Set risk and radius parameters
				self.set_solver_parameter(
					"gaussian_obstacle_risk",
					self.get_config_value("probabilistic.risk"),
					k, i
				)
				self.set_solver_parameter(
					"gaussian_obstacle_r",
					self.get_config_value("obstacle_radius"),
					k, i
				)

		if k == 1:
			LOG_DEBUG("GaussianConstraints::set_parameters Done")

	def get_lower_bound(self):
		lower_bound = []
		for obs in range(0, self.max_obstacles):
			for disc in range(0, self.n_discs):
				lower_bound.append(0.0)
		return lower_bound

	def get_upper_bound(self):
		upper_bound = []
		for obs in range(0, self.max_obstacles):
			for disc in range(0, self.n_discs):
				upper_bound.append(np.Inf)
		return upper_bound

	def get_constraints(self, model, params, settings, stage_idx):
		constraints = []
		x = model.get("x")
		y = model.get("y")
		psi = model.get("psi")
		pos = np.array([x, y])

		# area = model.system.area
		r_vehicle = params.get("ego_disc_radius")
		rotation_car = rotation_matrix(psi)

		for obs_id in range(self.max_obstacles):

			obs_x = params.get(f"gaussian_obst_{obs_id}_x")
			obs_y = params.get(f"gaussian_obst_{obs_id}_y")
			obs_pos = np.array([obs_x, obs_y])

			# params.add(f"gaussian_obst_{obs_id}_psi")
			sigma_x = params.get(f"gaussian_obst_{obs_id}_major")
			sigma_y = params.get(f"gaussian_obst_{obs_id}_minor")
			Sigma = np.diag([sigma_x ** 2, sigma_y ** 2])

			risk = params.get(f"gaussian_obst_{obs_id}_risk")

			r_obstacle = params.get(f"gaussian_obst_{obs_id}_r")
			combined_radius = r_vehicle + r_obstacle

			for disc_it in range(self.n_discs):
				# Get and compute the disc position
				disc_x = params.get(f"ego_disc_{disc_it}_offset")
				disc_relative_pos = np.array([disc_x, 0])
				disc_pos = pos + rotation_car.dot(disc_relative_pos)

				diff_pos = disc_pos - obs_pos

				a_ij = diff_pos / cd.sqrt(diff_pos.dot(diff_pos))
				b_ij = combined_radius

				x_erfinv = 1.0 - 2.0 * risk

				# Manual inverse erf, because somehow lacking from casadi...
				# From here: http://casadi.sourceforge.net/v1.9.0/api/internal/d4/d99/casadi_calculus_8hpp_source.html
				z = cd.sqrt(-cd.log((1.0 - x_erfinv) / 2.0))
				y_erfinv = (((1.641345311 * z + 3.429567803) * z - 1.624906493) * z - 1.970840454) / (
							(1.637067800 * z + 3.543889200) * z + 1.0)

				y_erfinv = y_erfinv - (cd.erf(y_erfinv) - x_erfinv) / (
							2.0 / cd.sqrt(cd.pi) * cd.exp(-y_erfinv * y_erfinv))
				y_erfinv = y_erfinv - (cd.erf(y_erfinv) - x_erfinv) / (
							2.0 / cd.sqrt(cd.pi) * cd.exp(-y_erfinv * y_erfinv))

				constraints.append(a_ij.T @ cd.SX(diff_pos) - b_ij - y_erfinv * cd.sqrt(2.0 * a_ij.T @ Sigma @ a_ij))
		return constraints

	def is_data_ready(self, data, missing_data):
		if data.dynamic_obstacles.size() != self.get_config_value("max_obstacles"):
			missing_data += "Obstacles "
			return False

		for i in range(data.dynamic_obstacles.size()):
			if data.dynamic_obstacles[i].prediction.modes.empty():
				missing_data += "Obstacle Prediction "
				return False

			if data.dynamic_obstacles[i].prediction.type != GAUSSIAN:
				missing_data += "Obstacle Prediction (Type is not Gaussian) "
				return False

		return True

	def visualize(self, data, module_data):
		if not self.get_config_value("debug_visuals", CONFIG.get("debug_visuals", False)):
			return

		PROFILE_SCOPE("GaussianConstraints::Visualize")
		LOG_DEBUG("GaussianConstraints.visualize")

		# Create publisher for visualization
		publisher = self.create_visualization_publisher("obstacles", ROSPointMarker)
		ellipsoid = publisher.get_new_point_marker("CYLINDER")

		for obstacle in data.dynamic_obstacles:
			k = 1
			while k < self.solver.N:
				if k - 1 >= len(obstacle.prediction.modes[0]):
					break

				ellipsoid.set_color_int(k, self.solver.N, 0.5)

				# Calculate chi-square value for confidence ellipse
				if obstacle.type == DYNAMIC:
					chi = exponential_quantile(0.5, 1.0 - self.get_config_value("probabilistic.risk"))
				else:
					chi = 0.0

				# Set the scale of the visualization
				major_radius = obstacle.prediction.modes[0][k - 1].major_radius
				minor_radius = obstacle.prediction.modes[0][k - 1].minor_radius
				pos = obstacle.prediction.modes[0][k - 1].position

				ellipsoid.set_scale(
					2 * (major_radius * np.sqrt(chi) + obstacle.radius),
					2 * (minor_radius * np.sqrt(chi) + obstacle.radius),
					0.005
				)

				ellipsoid.add_point_marker(pos)

				k += self.get_config_value("visualization.draw_every")

		publisher.publish()