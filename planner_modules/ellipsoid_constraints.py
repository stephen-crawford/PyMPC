import logging
import numpy as np
import casadi as cd

from utils.utils import rotation_matrix
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
		self.max_obstacles = self.get_config_value("max_obstacles")
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

	def define_parameters(self, params):
		params.add("ego_disc_radius")

		for disc_id in range(self.n_discs):
			params.add(f"ego_disc_{disc_id}_offset", bundle_name="ego_disc_offset")

		for obs_id in range(self.max_obstacles):
			params.add(f"ellipsoid_obst_{obs_id}_x", bundle_name="ellipsoid_obst_x")
			params.add(f"ellipsoid_obst_{obs_id}_y", bundle_name="ellipsoid_obst_y")
			params.add(f"ellipsoid_obst_{obs_id}_psi", bundle_name="ellipsoid_obst_psi")
			params.add(f"ellipsoid_obst_{obs_id}_major", bundle_name="ellipsoid_obst_major")
			params.add(f"ellipsoid_obst_{obs_id}_minor", bundle_name="ellipsoid_obst_minor")
			params.add(f"ellipsoid_obst_{obs_id}_chi", bundle_name="ellipsoid_obst_chi")
			params.add(f"ellipsoid_obst_{obs_id}_r", bundle_name="ellipsoid_obst_r")

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

	def get_lower_bound(self):
		lower_bound = []
		for obs in range(self.max_obstacles):
			for disc in range(self.n_discs):
				lower_bound.append(1.0)
		return lower_bound

	def get_upper_bound(self):
		upper_bound = []
		for obs in range(self.max_obstacles):
			for disc in range(self.n_discs):
				upper_bound.append(np.inf)
		return upper_bound

	def get_constraints(self, model, params, settings, stage_idx):
		constraints = []
		pos_x = model.get("x")
		pos_y = model.get("y")
		pos = np.array([pos_x, pos_y])

		try:
			psi = model.get("psi")
		except:
			psi = 0.0

		rotation_car = rotation_matrix(psi)

		r_disc = params.get("ego_disc_radius")

		# Constraint for dynamic obstacles
		for obs_id in range(self.max_obstacles):
			obst_x = params.get(f"ellipsoid_obst_{obs_id}_x")
			obst_y = params.get(f"ellipsoid_obst_{obs_id}_y")
			obstacle_cog = np.array([obst_x, obst_y])

			obst_psi = params.get(f"ellipsoid_obst_{obs_id}_psi")
			obst_major = params.get(f"ellipsoid_obst_{obs_id}_major")
			obst_minor = params.get(f"ellipsoid_obst_{obs_id}_minor")
			obst_r = params.get(f"ellipsoid_obst_{obs_id}_r")

			# multiplier for the risk when obst_major, obst_major only denote the covariance
			# (i.e., the smaller the risk, the larger the ellipsoid).
			# This value should already be a multiplier (using exponential cdf).
			chi = params.get(f"ellipsoid_obst_{obs_id}_chi")

			# Compute ellipse matrix
			obst_major *= cd.sqrt(chi)
			obst_minor *= cd.sqrt(chi)
			ab = cd.SX(2, 2)
			ab[0, 0] = 1.0 / ((obst_major + r_disc + obst_r) * (obst_major + r_disc + obst_r))
			ab[0, 1] = 0.0
			ab[1, 0] = 0.0
			ab[1, 1] = 1.0 / ((obst_minor + r_disc + obst_r) * (obst_minor + r_disc + obst_r))

			obstacle_rotation = cd.SX(rotation_matrix(obst_psi))
			obstacle_ellipse_matrix = obstacle_rotation.T @ ab @ obstacle_rotation

			for disc_it in range(self.n_discs):
				# Get and compute the disc position
				disc_x = params.get(f"ego_disc_{disc_it}_offset")
				disc_relative_pos = np.array([disc_x, 0])
				disc_pos = pos + rotation_car @ disc_relative_pos

				# construct the constraint and append it
				disc_to_obstacle = cd.SX(disc_pos - obstacle_cog)
				c_disc_obstacle = disc_to_obstacle.T @ obstacle_ellipse_matrix @ disc_to_obstacle
				constraints.append(c_disc_obstacle)  # + slack)

		return constraints

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