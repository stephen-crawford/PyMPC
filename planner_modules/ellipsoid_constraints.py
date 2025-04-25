import logging
import numpy as np
from utils.const import CONSTRAINT, DETERMINISTIC, GAUSSIAN
from utils.utils import LOG_DEBUG, exponential_quantile, CONFIG
from utils.visualizer import ROSLine, ROSPointMarker

from planner_modules.base_constraint import BaseConstraint


class EllipsoidConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)
		self.name = 'ellipsoid_constraints'
		self.num_segments = self.get_config_value("contouring.num_segments")
		self.n_discs = self.get_config_value("n_discs")
		self._robot_radius = self.get_config_value("robot.radius")
		self.risk = self.get_config_value("probabilistic.risk")

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
			self.set_solver_parameter("ego_disc_radius", self._robot_radius, k)

			# Set solver parameter for ego disc offset
			if hasattr(data, 'robot_area') and len(data.robot_area) > d:
				self.set_solver_parameter("ego_disc_offset", data.robot_area[d].offset, k, d)

		if k == 0:  # Dummies
			# Set dummy values for obstacles at k=0
			for i in range(data.dynamic_obstacles.size()):
				self.set_solver_parameter("ellipsoid_obst_x", self._dummy_x, k, i)
				self.set_solver_parameter("ellipsoid_obst_y", self._dummy_y, k, i)
				self.set_solver_parameter("ellipsoid_obst_psi", 0.0, k, i)
				self.set_solver_parameter("ellipsoid_obst_r", 0.1, k, i)
				self.set_solver_parameter("ellipsoid_obst_major", 0.0, k, i)
				self.set_solver_parameter("ellipsoid_obst_minor", 0.0, k, i)
				self.set_solver_parameter("ellipsoid_obst_chi", 1.0, k, i)
			return

		if k == 1:
			LOG_DEBUG("EllipsoidConstraints::set_parameters")

		# Set parameters for each dynamic obstacle
		for i in range(data.dynamic_obstacles.size()):
			obstacle = data.dynamic_obstacles[i]
			mode = obstacle.prediction.modes[0]

			# The first prediction step is index 1 of the optimization problem
			# k-1 maps to the predictions for this stage
			self.set_solver_parameter("ellipsoid_obst_x", mode[k - 1].position[0], k, i)
			self.set_solver_parameter("ellipsoid_obst_y", mode[k - 1].position[1], k, i)
			self.set_solver_parameter("ellipsoid_obst_psi", mode[k - 1].angle, k, i)
			self.set_solver_parameter("ellipsoid_obst_r", obstacle.radius, k, i)

			if obstacle.prediction.type == DETERMINISTIC:
				# For deterministic obstacles, set zeros for uncertainty ellipse
				self.set_solver_parameter("ellipsoid_obst_major", 0.0, k, i)
				self.set_solver_parameter("ellipsoid_obst_minor", 0.0, k, i)
				self.set_solver_parameter("ellipsoid_obst_chi", 1.0, k, i)

			elif obstacle.prediction.type == GAUSSIAN:
				# Calculate chi-square quantile for desired risk level
				chi = exponential_quantile(0.5, 1.0 - self.risk)

				# Set uncertainty ellipse parameters
				self.set_solver_parameter("ellipsoid_obst_major", mode[k - 1].major_radius, k, i)
				self.set_solver_parameter("ellipsoid_obst_minor", mode[k - 1].minor_radius, k, i)
				self.set_solver_parameter("ellipsoid_obst_chi", chi, k, i)

		if k == 1:
			LOG_DEBUG("EllipsoidConstraints::set_parameters Done")

	def is_data_ready(self, data, missing_data):
		print("Checking if data is ready...")
		required_fields = ["robot_area", "dynamic_obstacles"]
		missing_fields = self.check_data_availability(data, required_fields)

		if missing_fields:
			missing_data += " ".join(missing_fields) + " "
			return False

		if data.dynamic_obstacles.size() != self.get_config_value("max_obstacles"):
			missing_data += "Obstacles "
			print("Found missing obstacles, " + str(self.get_config_value("max_obstacles")))
			return False

		for i in range(data.dynamic_obstacles.size()):
			if data.dynamic_obstacles[i].prediction.empty():
				print("Found missing obstacle pred")
				missing_data += "Obstacle Prediction "
				return False

			if (data.dynamic_obstacles[i].prediction.type != GAUSSIAN and
					data.dynamic_obstacles[i].prediction.type != DETERMINISTIC):
				missing_data += "Obstacle Prediction (Type is incorrect) "
				print("Found wrong predictive type")
				return False

		return True

	def visualize(self, data, module_data):
		super().visualize(data, module_data)

		# Create publisher for ellipsoid visualization
		ellipsoid_publisher = self.create_visualization_publisher("ellipsoids", ROSLine)

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