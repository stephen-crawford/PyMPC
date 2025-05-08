import numpy as np
import casadi as cd
from utils.const import GAUSSIAN, DYNAMIC
from utils.utils import LOG_DEBUG, PROFILE_SCOPE, exponential_quantile, CONFIG, rotation_matrix
from utils.visualizer import ROSPointMarker
from planner_modules.src.constraints.base_constraint import BaseConstraint


class GaussianConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)
		self.name = 'gaussian_constraints'
		# Store dummy values for invalid states
		self._dummy_x = 100.0
		self._dummy_y = 100.0
		self.num_discs = self.get_config_value("num_discs")
		self.robot_radius = self.get_config_value("robot.radius")
		self.max_obstacles = self.get_config_value("max_obstacles")

		LOG_DEBUG("Gaussian Constraints successfully initialized")

	def update(self, state, data, module_data):
		LOG_DEBUG("GaussianConstraints.update")

		# Update dummy values based on current state
		self._dummy_x = state.get("x") + 100.0
		self._dummy_y = state.get("y") + 100.0

	def define_parameters(self, params):

		for disc_id in range(self.num_discs):
			params.add(f"ego_disc_{disc_id}_offset")

		for obs_id in range(self.max_obstacles):
			params.add(f"gaussian_obst_{obs_id}_x")
			params.add(f"gaussian_obst_{obs_id}_y")
			params.add(f"gaussian_obst_{obs_id}_major")
			params.add(f"gaussian_obst_{obs_id}_minor")
			params.add(f"gaussian_obst_{obs_id}_risk")
			params.add(f"gaussian_obst_{obs_id}_r")

	def set_parameters(self, parameter_manager, data, module_data, k):
		for d in range(self.num_discs):
			# Set solver parameter for ego disc radius
			parameter_manager.set(f"ego_disc{d}_radius", self.robot_radius)

			# Set solver parameter for ego disc offset
			if hasattr(data, 'robot_area') and len(data.robot_area) > d:
				parameter_manager.set(f"ego_disc{d}_offset", data.robot_area[d].offset)

		if k == 0:  # Dummies
			for obstacle_id in range(data.dynamic_obstacles.size()):
				parameter_manager.set(f"gaussian_obstacle_{obstacle_id}_x", self._dummy_x)
				parameter_manager.set(f"gaussian_obstacle_{obstacle_id}_y", self._dummy_y)
				parameter_manager.set(f"gaussian_obstacle_{obstacle_id}_major", 0.1)
				parameter_manager.set(f"gaussian_obstacle_{obstacle_id}_minor", 0.1)
				parameter_manager.set(f"gaussian_obstacle_{obstacle_id}_risk", 0.05)
				parameter_manager.set(f"gaussian_obstacle_{obstacle_id}_r", 0.1)
			return

		if k == 1:
			LOG_DEBUG("GaussianConstraints::set_parameters")

		# Set obstacle parameters
		for obstacle_id in range(data.dynamic_obstacles.size()):
			obstacle = data.dynamic_obstacles[obstacle_id]

			if obstacle.prediction.type == GAUSSIAN:
				# Set position parameters
				parameter_manager.set(f"gaussian_obstacle_{obstacle_id}_x", obstacle.prediction.modes[0][k - 1].position[0])
				parameter_manager.set(f"gaussian_obstacle_{obstacle_id}_y", obstacle.prediction.modes[0][k - 1].position[1])

				if obstacle.type == DYNAMIC:
					# Dynamic obstacles have uncertainty
					parameter_manager.set(f"gaussian_obstacle_{obstacle_id}_major", obstacle.prediction.modes[0][k - 1].major_radius)
					parameter_manager.set(f"gaussian_obstacle_{obstacle_id}_minor", obstacle.prediction.modes[0][k - 1].minor_radius)
				else:
					# Static obstacles have minimal uncertainty
					parameter_manager.set(f"gaussian_obstacle_{obstacle_id}_major", 0.001)
					parameter_manager.set(f"gaussian_obstacle_{obstacle_id}_minor", 0.001)

				# Set risk and radius parameters
				parameter_manager.set(f"gaussian_obstacle_{obstacle_id}_risk", self.get_config_value("probabilistic.risk"))
				parameter_manager.set(f"gaussian_obstacle_{obstacle_id}_r", self.get_config_value("obstacle_radius"))

		if k == 1:
			LOG_DEBUG("GaussianConstraints::set_parameters Done")

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

			for disc_it in range(self.num_discs):
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

	def is_data_ready(self, data):
		missing_data = ""
		if data.dynamic_obstacles.size() != self.get_config_value("max_obstacles"):
			missing_data += "Obstacles "


		for i in range(data.dynamic_obstacles.size()):
			if data.dynamic_obstacles[i].prediction.modes.empty():
				missing_data += "Obstacle Prediction "


			if data.dynamic_obstacles[i].prediction.type != GAUSSIAN:
				missing_data += "Obstacle Prediction (Type is not Gaussian) "


		return len(missing_data) < 1

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
			while k < self.solver.horizon:
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