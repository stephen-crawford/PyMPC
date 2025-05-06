import numpy as np
from utils.const import CONSTRAINT, DETERMINISTIC, GAUSSIAN
from utils.utils import LOG_DEBUG, rotation_matrix, rotation_matrix
from utils.visualizer import visualize_linear_constraint
from planner_modules.base_constraint import BaseConstraint


class LinearizedConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)
		self.name = "linearized_constraints"  # Override the name from BaseConstraint

		LOG_DEBUG("Initializing Linearized Constraints")

		self.n_discs = int(self.get_config_value("n_discs"))  # Is overwritten to 1 for topology constraints
		self._n_other_halfspaces = self.get_config_value("linearized_constraints.add_halfspaces")
		self._max_obstacles = self.get_config_value("max_obstacles")
		self.num_constraints = self._max_obstacles + self._n_other_halfspaces
		self.nh = self.num_constraints
		self.use_slack = self.get_config_value("linearized_constraints.use_slack")


		# Initialize arrays
		self._a1 = [[[0 for _ in range(self.num_constraints)] for _ in range(self.get_config_value("N"))] for _ in
					range(int(self.get_config_value("n_discs")))]
		self._a2 = [[[0 for _ in range(self.num_constraints)] for _ in range(self.get_config_value("N"))] for _ in
					range(int(self.get_config_value("n_discs")))]
		self._b = [[[0 for _ in range(self.num_constraints)] for _ in range(self.get_config_value("N"))] for _ in
				   range(int(self.get_config_value("n_discs")))]

		self._num_obstacles = 0
		self._use_guidance = False

		# Store dummy values for invalid states
		self._dummy_a1 = 0.0
		self._dummy_a2 = 0.0
		self._dummy_b = 100.0

		LOG_DEBUG("Linearized Constraints successfully initialized")

	def setTopologyConstraints(self):
		self.n_discs = 1  # Only one disc is used for the topology constraints
		self._use_guidance = True

	def update(self, state, data, module_data):
		LOG_DEBUG("LinearizedConstraints.update")

		self._dummy_b = state.get("x") + 100.0

		# Thread safe copy of obstacles
		copied_obstacles = data.dynamic_obstacles
		self._num_obstacles = copied_obstacles.size()

		# For all stages
		for k in range(1, self.solver.N):
			for d in range(self.n_discs):
				# Get ego position
				pos = np.array([
					self.solver.get_ego_prediction(k, "x"),
					self.solver.get_ego_prediction(k, "y")
				])

				if not self._use_guidance:  # Use discs and their positions
					disc = data.robot_area[d]
					disc_pos = disc.get_position(pos, self.solver.get_ego_prediction(k, "psi"))
					self.project_to_safety(copied_obstacles, k, disc_pos)  # Ensure position is collision-free
					pos = disc_pos
				else:  # Use the robot position
					self.project_to_safety(copied_obstacles, k, pos)  # Ensure position is collision-free

				# For all obstacles
				for obs_id in range(copied_obstacles.size()):
					copied_obstacle = copied_obstacles[obs_id]
					obstacle_pos = np.array([
						copied_obstacle.prediction.modes[0][k - 1].position[0],
						copied_obstacle.prediction.modes[0][k - 1].position[1]
					])

					diff_x = obstacle_pos[0] - pos[0]
					diff_y = obstacle_pos[1] - pos[1]
					dist = np.linalg.norm(obstacle_pos - pos)

					# Compute the components of A for this obstacle (normalized normal vector)
					self._a1[d][k][obs_id] = diff_x / dist
					self._a2[d][k][obs_id] = diff_y / dist

					# Compute b (evaluate point on the collision circle)
					radius = 1e-3 if self._use_guidance else copied_obstacle.radius
					self._b[d][k][obs_id] = (self._a1[d][k][obs_id] * obstacle_pos[0] +
											 self._a2[d][k][obs_id] * obstacle_pos[1] -
											 (radius + self.get_config_value("n_discs")))

				# Handle static obstacles
				if not module_data.static_obstacles.empty():
					if module_data.static_obstacles[k].size() < self._n_other_halfspaces:
						LOG_DEBUG(
							f"{self._n_other_halfspaces} halfspaces expected, but {module_data.static_obstacles[k].size()} are present")

					num_halfspaces = min(module_data.static_obstacles[k].size(), self._n_other_halfspaces)
					for h in range(num_halfspaces):
						obs_id = copied_obstacles.size() + h
						self._a1[d][k][obs_id] = module_data.static_obstacles[k][h].A[0]
						self._a2[d][k][obs_id] = module_data.static_obstacles[k][h].A[1]
						self._b[d][k][obs_id] = module_data.static_obstacles[k][h].b

		LOG_DEBUG("LinearizedConstraints.update done")

	def project_to_safety(self, copied_obstacles, k, pos):
		if copied_obstacles.empty():  # There is no anchor
			return

		# Project to a collision free position if necessary, considering all the obstacles
		for _ in range(3):  # At most 3 iterations
			for obstacle in copied_obstacles:
				radius = 1e-3 if self._use_guidance else obstacle.radius

				# Douglas-Rachford projection method
				self.douglas_rachford_projection(
					pos,
					obstacle.prediction.modes[0][k - 1].position,
					copied_obstacles[0].prediction.modes[0][k - 1].position,
					radius + self.get_config_value("n_discs"),
					pos
				)

	def douglas_rachford_projection(self, pos, obstacle_pos, anchor_pos, radius, result):
		# Implementation of Douglas-Rachford projection algorithm
		# (This is a placeholder - actual implementation would depend on your system)
		pass

	def define_parameters(self, params):

		for disc_id in range(self.n_discs):
			params.add(f"ego_disc_{disc_id}_offset", bundle_name="ego_disc_offset")

			for index in range(self.max_obstacles):
				params.add(self.constraint_name(index, disc_id) + "_a1", bundle_name="lin_constraint_a1")
				params.add(self.constraint_name(index, disc_id) + "_a2", bundle_name="lin_constraint_a2")
				params.add(self.constraint_name(index, disc_id) + "_b", bundle_name="lin_constraint_b")

	def constraint_name(self, index, disc_id):
		return f"disc_{disc_id}_lin_constraint_{index}"

	def get_lower_bound(self):
		lower_bound = []
		for index in range(0, self.num_constraints):
			lower_bound.append(-np.inf)
		return lower_bound

	def get_upper_bound(self):
		upper_bound = []
		for index in range(0, self.num_constraints):
			upper_bound.append(0.0)
		return upper_bound

	def get_constraints(self, model, params, settings, stage_idx):
		constraints = []

		# States
		pos_x = model.get("x")
		pos_y = model.get("y")
		pos = np.array([pos_x, pos_y])
		psi = model.get("psi")

		try:
			if self.use_slack:
				slack = model.get("slack")
			else:
				slack = 0.0
		except:
			slack = 0.0

		rotation_car = rotation_matrix(psi)
		for disc_it in range(self.n_discs):
			disc_x = params.get(f"ego_disc_{disc_it}_offset")
			disc_relative_pos = np.array([disc_x, 0])
			disc_pos = pos + rotation_car.dot(disc_relative_pos)

			for index in range(self.max_obstacles):
				a1 = params.get(self.constraint_name(index, disc_it) + "_a1")
				a2 = params.get(self.constraint_name(index, disc_it) + "_a2")
				b = params.get(self.constraint_name(index, disc_it) + "_b")

				constraints.append(a1 * disc_pos[0] + a2 * disc_pos[1] - (b + slack))

		return constraints

	def set_parameters(self, data, module_data, k):
		constraint_counter = 0  # Maps disc and obstacle index to a single index

		if k == 0:
			for i in range(self._max_obstacles + self._n_other_halfspaces):
				self.set_solver_parameter("lin_constraint_a1", self._dummy_a1, k, constraint_counter)
				self.set_solver_parameter("lin_constraint_a2", self._dummy_a2, k, constraint_counter)
				self.set_solver_parameter("lin_constraint_b", self._dummy_b, k, constraint_counter)
				constraint_counter += 1
			return

		for d in range(self.n_discs):
			if not self._use_guidance:
				self.set_solver_parameter("ego_disc_offset", data.robot_area[d].offset, k, d)

			# Set actual constraints
			for i in range(data.dynamic_obstacles.size() + self._n_other_halfspaces):
				self.set_solver_parameter("lin_constraint_a1", self._a1[d][k][i], k, constraint_counter)
				self.set_solver_parameter("lin_constraint_a2", self._a2[d][k][i], k, constraint_counter)
				self.set_solver_parameter("lin_constraint_b", self._b[d][k][i], k, constraint_counter)
				constraint_counter += 1

			# Set dummy constraints for remaining slots
			for i in range(data.dynamic_obstacles.size() + self._n_other_halfspaces,
						   self._max_obstacles + self._n_other_halfspaces):
				self.set_solver_parameter("lin_constraint_a1", self._dummy_a1, k, constraint_counter)
				self.set_solver_parameter("lin_constraint_a2", self._dummy_a2, k, constraint_counter)
				self.set_solver_parameter("lin_constraint_b", self._dummy_b, k, constraint_counter)
				constraint_counter += 1

	def is_data_ready(self, data, missing_data):
		required_fields = ["dynamic_obstacles", "robot_area"]
		missing_fields = self.check_data_availability(data, required_fields)

		if not self.report_missing_data(missing_fields, missing_data):
			return False

		if data.dynamic_obstacles.size() != self._max_obstacles:
			missing_data += "Obstacles "
			return False

		for i in range(data.dynamic_obstacles.size()):
			if data.dynamic_obstacles[i].prediction.empty():
				missing_data += "Obstacle Prediction "
				return False

			if (data.dynamic_obstacles[i].prediction.type != DETERMINISTIC and
					data.dynamic_obstacles[i].prediction.type != GAUSSIAN):
				missing_data += "Obstacle Prediction (type must be deterministic, or gaussian) "
				return False

		return True

	def visualize(self, data, module_data):
		super().visualize(data, module_data)

		if self._use_guidance and not self.get_config_value("debug_visuals", False):
			return

		for k in range(1, self.solver.N):
			for i in range(data.dynamic_obstacles.size()):
				# Determine if this is the last visualization to publish
				is_last = (k == self.solver.N - 1 and i == data.dynamic_obstacles.size() - 1)

				visualize_linear_constraint(
					self._a1[0][k][i],
					self._a2[0][k][i],
					self._b[0][k][i],
					k,
					self.solver.N,
					self.name,
					is_last
				)

	def reset(self):
		super().reset()
		# Reset constraint-specific state
		self._num_obstacles = 0

		# Re-initialize arrays with zeros
		for d in range(self.get_config_value("n_discs")):
			for k in range(self.get_config_value("N")):
				for i in range(self.num_constraints):
					self._a1[d][k][i] = 0
					self._a2[d][k][i] = 0
					self._b[d][k][i] = 0