import logging
import numpy as np
from utils.const import CONSTRAINT, DETERMINISTIC, GAUSSIAN
from utils.utils import read_config_file, LOG_DEBUG, exponential_quantile
from utils.visualizer import ROSLine, ROSPointMarker

CONFIG = read_config_file()


class EllipsoidConstraints:
	def __init__(self, solver):
		self.solver = solver
		self.module_type = CONSTRAINT
		self.name = "ellipsoid_constraints"
		LOG_DEBUG("Initializing Ellipsoid Constraints")

		self.num_segments = CONFIG["contouring"]["num_segments"]
		self.n_discs = CONFIG["n_discs"]
		self._robot_radius = CONFIG["robot"]["radius"]
		self.risk = CONFIG["probabilistic"]["risk"]

		# Dummy values for invalid states
		self._dummy_x = 50.0
		self._dummy_y = 50.0

		LOG_DEBUG("Ellipsoid Constraints successfully initialized")

	def update(self, state, data, module_data):
		LOG_DEBUG("EllipsoidConstraints.update")

		# Update dummy values based on current state
		self._dummy_x = state.get("x") + 50.0
		self._dummy_y = state.get("y") + 50.0

	def set_parameters(self, data, module_data, k):
		for d in range(self.n_discs):
			# Set solver parameter for ego disc radius
			self.set_solver_parameter_ego_disc_radius(k, self.solver.params, self._robot_radius)

			# Set solver parameter for ego disc offset
			if hasattr(data, 'robot_area') and len(data.robot_area) > d:
				self.set_solver_parameter_ego_disc_offset(k, self.solver.params, data.robot_area[d].offset, d)

		if k == 0:  # Dummies
			# Set dummy values for obstacles at k=0
			for i in range(data.dynamic_obstacles.size()):
				self.set_solver_parameter_ellipsoid_obst_x(k, self.solver.params, self._dummy_x, i)
				self.set_solver_parameter_ellipsoid_obst_y(k, self.solver.params, self._dummy_y, i)
				self.set_solver_parameter_ellipsoid_obst_psi(k, self.solver.params, 0.0, i)
				self.set_solver_parameter_ellipsoid_obst_r(k, self.solver.params, 0.1, i)
				self.set_solver_parameter_ellipsoid_obst_major(k, self.solver.params, 0.0, i)
				self.set_solver_parameter_ellipsoid_obst_minor(k, self.solver.params, 0.0, i)
				self.set_solver_parameter_ellipsoid_obst_chi(k, self.solver.params, 1.0, i)
			return

		if k == 1:
			LOG_DEBUG("EllipsoidConstraints::set_parameters")

		# Set parameters for each dynamic obstacle
		for i in range(data.dynamic_obstacles.size()):
			obstacle = data.dynamic_obstacles[i]
			mode = obstacle.prediction.modes[0]

			# The first prediction step is index 1 of the optimization problem
			# k-1 maps to the predictions for this stage
			self.set_solver_parameter_ellipsoid_obst_x(k, self.solver.params, mode[k - 1].position[0], i)
			self.set_solver_parameter_ellipsoid_obst_y(k, self.solver.params, mode[k - 1].position[1], i)
			self.set_solver_parameter_ellipsoid_obst_psi(k, self.solver.params, mode[k - 1].angle, i)
			self.set_solver_parameter_ellipsoid_obst_r(k, self.solver.params, obstacle.radius, i)

			if obstacle.prediction.type == DETERMINISTIC:
				# For deterministic obstacles, set zeros for uncertainty ellipse
				self.set_solver_parameter_ellipsoid_obst_major(k, self.solver.params, 0.0, i)
				self.set_solver_parameter_ellipsoid_obst_minor(k, self.solver.params, 0.0, i)
				self.set_solver_parameter_ellipsoid_obst_chi(k, self.solver.params, 1.0, i)

			elif obstacle.prediction.type == GAUSSIAN:
				# Calculate chi-square quantile for desired risk level
				chi = exponential_quantile(0.5, 1.0 - self.risk)

				# Set uncertainty ellipse parameters
				self.set_solver_parameter_ellipsoid_obst_major(k, self.solver.params, mode[k - 1].major_radius, i)
				self.set_solver_parameter_ellipsoid_obst_minor(k, self.solver.params, mode[k - 1].minor_radius, i)
				self.set_solver_parameter_ellipsoid_obst_chi(k, self.solver.params, chi, i)

		if k == 1:
			LOG_DEBUG("EllipsoidConstraints::set_parameters Done")

	def is_data_ready(self, data, missing_data):
		if not hasattr(data, 'robot_area') or len(data.robot_area) == 0:
			missing_data += "Robot area "
			return False

		if data.dynamic_obstacles.size() != CONFIG["max_obstacles"]:
			missing_data += "Obstacles "
			return False

		for i in range(data.dynamic_obstacles.size()):
			if data.dynamic_obstacles[i].prediction.empty():
				missing_data += "Obstacle Prediction "
				return False

			if (data.dynamic_obstacles[i].prediction.type != GAUSSIAN and
					data.dynamic_obstacles[i].prediction.type != DETERMINISTIC):
				missing_data += "Obstacle Prediction (Type is incorrect) "
				return False

		return True

	def visualize(self, data, module_data):
		if not CONFIG["debug_visuals"]:
			return

		LOG_DEBUG("EllipsoidConstraints.Visualize")

		# Create publisher for ellipsoid visualization
		ellipsoid_publisher = ROSLine(self.name + "/ellipsoids")

		# For each prediction step
		for k in range(1, self.solver.N):
			# For each obstacle
			for i in range(data.dynamic_obstacles.size()):
				obstacle = data.dynamic_obstacles[i]
				if k - 1 >= len(obstacle.prediction.modes[0]):
					continue

				mode = obstacle.prediction.modes[0]
				position = np.array([mode[k - 1].position[0], mode[k - 1].position[1]])
				angle = mode[k - 1].angle

				# Create ellipse visualization
				line = ellipsoid_publisher.add_new_line()
				line.set_scale(0.1)
				line.set_color_int(i, data.dynamic_obstacles.size())

				# Draw basic circle for deterministic obstacles
				if obstacle.prediction.type == DETERMINISTIC:
					self._draw_circle(line, position, obstacle.radius)
				# Draw uncertainty ellipse for probabilistic obstacles
				elif obstacle.prediction.type == GAUSSIAN:
					major_radius = mode[k - 1].major_radius
					minor_radius = mode[k - 1].minor_radius
					chi = exponential_quantile(0.5, 1.0 - self.risk)

					self._draw_ellipse(line, position, angle,
					                   major_radius * np.sqrt(chi),
					                   minor_radius * np.sqrt(chi))

		ellipsoid_publisher.publish()

	def _draw_circle(self, line, center, radius, num_points=20):
		"""Helper method to draw a circle"""
		theta = np.linspace(0, 2 * np.pi, num_points)
		prev_point = None

		for t in theta:
			point = center + radius * np.array([np.cos(t), np.sin(t)])

			if prev_point is not None:
				line.add_line((prev_point[0], prev_point[1], 0),
				              (point[0], point[1], 0))

			prev_point = point

		# Close the loop
		first_point = center + radius * np.array([np.cos(theta[0]), np.sin(theta[0])])
		line.add_line((prev_point[0], prev_point[1], 0),
		              (first_point[0], first_point[1], 0))

	def _draw_ellipse(self, line, center, angle, a, b, num_points=20):
		"""Helper method to draw an ellipse"""
		theta = np.linspace(0, 2 * np.pi, num_points)

		# Rotation matrix
		R = np.array([[np.cos(angle), -np.sin(angle)],
		              [np.sin(angle), np.cos(angle)]])

		prev_point = None

		for t in theta:
			# Ellipse point in local coordinates
			local_point = np.array([a * np.cos(t), b * np.sin(t)])

			# Rotate and translate to global coordinates
			point = center + R @ local_point

			if prev_point is not None:
				line.add_line((prev_point[0], prev_point[1], 0),
				              (point[0], point[1], 0))

			prev_point = point

		# Close the loop
		local_first = np.array([a * np.cos(theta[0]), b * np.sin(theta[0])])
		first_point = center + R @ local_first
		line.add_line((prev_point[0], prev_point[1], 0),
		              (first_point[0], first_point[1], 0))

	# Parameter setter methods
	def set_solver_parameter_ego_disc_radius(self, k, params, value):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "ego_disc_radius", value, k, settings=CONFIG)

	def set_solver_parameter_ego_disc_offset(self, k, params, value, d):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "ego_disc_offset", value, k, index=d, settings=CONFIG)

	def set_solver_parameter_ellipsoid_obst_x(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "ellipsoid_obst_x", value, k, index=i, settings=CONFIG)

	def set_solver_parameter_ellipsoid_obst_y(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "ellipsoid_obst_y", value, k, index=i, settings=CONFIG)

	def set_solver_parameter_ellipsoid_obst_psi(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "ellipsoid_obst_psi", value, k, index=i, settings=CONFIG)

	def set_solver_parameter_ellipsoid_obst_r(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "ellipsoid_obst_r", value, k, index=i, settings=CONFIG)

	def set_solver_parameter_ellipsoid_obst_major(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "ellipsoid_obst_major", value, k, index=i, settings=CONFIG)

	def set_solver_parameter_ellipsoid_obst_minor(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "ellipsoid_obst_minor", value, k, index=i, settings=CONFIG)

	def set_solver_parameter_ellipsoid_obst_chi(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "ellipsoid_obst_chi", value, k, index=i, settings=CONFIG)