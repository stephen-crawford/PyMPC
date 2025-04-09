import logging
import numpy as np
from numpy import sqrt
from utils.const import CONSTRAINT, GAUSSIAN, DYNAMIC
from utils.utils import read_config_file, LOG_DEBUG, PROFILE_SCOPE, exponential_quantile
from utils.visualizer import ROSPointMarker

CONFIG = read_config_file()


class GaussianConstraints:
	def __init__(self, solver):
		self.solver = solver
		self.module_type = CONSTRAINT
		self.name = "gaussian_constraints"

		LOG_DEBUG("Initializing Gaussian Constraints")

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
		self.set_solver_parameter_ego_disc_radius(k, self.solver.params, CONFIG["robot"]["radius"])

		for d in range(CONFIG["n_discs"]):
			if hasattr(data, 'robot_area') and len(data.robot_area) > d:
				self.set_solver_parameter_ego_disc_offset(k, self.solver.params, data.robot_area[d].offset, d)

		if k == 0:  # Dummies
			for i in range(data.dynamic_obstacles.size()):
				self.set_solver_parameter_gaussian_obstacle_x(k, self.solver.params, self._dummy_x, i)
				self.set_solver_parameter_gaussian_obstacle_y(k, self.solver.params, self._dummy_y, i)
				self.set_solver_parameter_gaussian_obstacle_major(k, self.solver.params, 0.1, i)
				self.set_solver_parameter_gaussian_obstacle_minor(k, self.solver.params, 0.1, i)
				self.set_solver_parameter_gaussian_obstacle_risk(k, self.solver.params, 0.05, i)
				self.set_solver_parameter_gaussian_obstacle_r(k, self.solver.params, 0.1, i)
			return

		if k == 1:
			LOG_DEBUG("GaussianConstraints::set_parameters")

		# Set obstacle parameters
		for i in range(data.dynamic_obstacles.size()):
			obstacle = data.dynamic_obstacles[i]

			if obstacle.prediction.type == GAUSSIAN:
				# Set position parameters
				self.set_solver_parameter_gaussian_obstacle_x(
					k, self.solver.params,
					obstacle.prediction.modes[0][k - 1].position[0], i
				)
				self.set_solver_parameter_gaussian_obstacle_y(
					k, self.solver.params,
					obstacle.prediction.modes[0][k - 1].position[1], i
				)

				if obstacle.type == DYNAMIC:
					# Dynamic obstacles have uncertainty
					self.set_solver_parameter_gaussian_obstacle_major(
						k, self.solver.params,
						obstacle.prediction.modes[0][k - 1].major_radius, i
					)
					self.set_solver_parameter_gaussian_obstacle_minor(
						k, self.solver.params,
						obstacle.prediction.modes[0][k - 1].minor_radius, i
					)
				else:
					# Static obstacles have minimal uncertainty
					self.set_solver_parameter_gaussian_obstacle_major(k, self.solver.params, 0.001, i)
					self.set_solver_parameter_gaussian_obstacle_minor(k, self.solver.params, 0.001, i)

				# Set risk and radius parameters
				self.set_solver_parameter_gaussian_obstacle_risk(
					k, self.solver.params,
					CONFIG["probabilistic"]["risk"], i
				)
				self.set_solver_parameter_gaussian_obstacle_r(
					k, self.solver.params,
					CONFIG["obstacle_radius"], i
				)

		if k == 1:
			LOG_DEBUG("GaussianConstraints::set_parameters Done")

	def is_data_ready(self, data, missing_data):
		if data.dynamic_obstacles.size() != CONFIG["max_obstacles"]:
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
		if not CONFIG["debug_visuals"]:
			return

		PROFILE_SCOPE("GaussianConstraints::Visualize")
		LOG_DEBUG("GaussianConstraints.visualize")

		# Create publisher for visualization
		publisher = ROSPointMarker(self.name + "/obstacles")
		ellipsoid = publisher.get_new_point_marker("CYLINDER")

		for obstacle in data.dynamic_obstacles:
			k = 1
			while k < self.solver.N:
				if k - 1 >= len(obstacle.prediction.modes[0]):
					break

				ellipsoid.set_color_int(k, self.solver.N, 0.5)

				# Calculate chi-square value for confidence ellipse
				if obstacle.type == DYNAMIC:
					chi = exponential_quantile(0.5, 1.0 - CONFIG["probabilistic"]["risk"])
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

				k += CONFIG["visualization"]["draw_every"]

		publisher.publish()

	# Parameter setter methods
	def set_solver_parameter_ego_disc_radius(self, k, params, value):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "ego_disc_radius", value, k, settings=CONFIG)

	def set_solver_parameter_ego_disc_offset(self, k, params, value, d):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "ego_disc_offset", value, k, index=d, settings=CONFIG)

	def set_solver_parameter_gaussian_obstacle_x(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "gaussian_obstacle_x", value, k, index=i, settings=CONFIG)

	def set_solver_parameter_gaussian_obstacle_y(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "gaussian_obstacle_y", value, k, index=i, settings=CONFIG)

	def set_solver_parameter_gaussian_obstacle_major(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "gaussian_obstacle_major", value, k, index=i, settings=CONFIG)

	def set_solver_parameter_gaussian_obstacle_minor(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "gaussian_obstacle_minor", value, k, index=i, settings=CONFIG)

	def set_solver_parameter_gaussian_obstacle_risk(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "gaussian_obstacle_risk", value, k, index=i, settings=CONFIG)

	def set_solver_parameter_gaussian_obstacle_r(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "gaussian_obstacle_r", value, k, index=i, settings=CONFIG)