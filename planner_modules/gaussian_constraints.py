import numpy as np
from numpy import sqrt
from utils.const import CONSTRAINT, GAUSSIAN, DYNAMIC
from utils.utils import LOG_DEBUG, PROFILE_SCOPE, exponential_quantile, CONFIG
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
					2 * (major_radius * sqrt(chi) + obstacle.radius),
					2 * (minor_radius * sqrt(chi) + obstacle.radius),
					0.005
				)

				ellipsoid.add_point_marker(pos)

				k += self.get_config_value("visualization.draw_every")

		publisher.publish()