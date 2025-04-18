import numpy as np
from utils.const import CONSTRAINT, DETERMINISTIC, GAUSSIAN
from utils.utils import read_config_file, LOG_DEBUG
from utils.visualizer import *

CONFIG = read_config_file()

class LinearizedConstraints:
	def __init__(self, solver):
		self.solver = solver
		self.module_type = CONSTRAINT
		self.name = "linearized_constraints"

		LOG_DEBUG("Initializing Linearized Constraints")

		self.n_discs = CONFIG["n_discs"]  # Is overwritten to 1 for topology constraints
		self._n_other_halfspaces = CONFIG["linearized_constraints"]["add_halfspaces"]
		self._max_obstacles = CONFIG["max_obstacles"]
		self.n_constraints = self._max_obstacles + self._n_other_halfspaces

		# Initialize arrays
		self._a1 = [[[0 for _ in range(self.n_constraints)] for _ in range(CONFIG["N"])] for _ in
		            range(CONFIG["n_discs"])]
		self._a2 = [[[0 for _ in range(self.n_constraints)] for _ in range(CONFIG["N"])] for _ in
		            range(CONFIG["n_discs"])]
		self._b = [[[0 for _ in range(self.n_constraints)] for _ in range(CONFIG["N"])] for _ in
		           range(CONFIG["n_discs"])]

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
					                         (radius + CONFIG["n_discs"]))

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
					radius + CONFIG["n_discs"],
					pos
				)

	def douglas_rachford_projection(self, pos, obstacle_pos, anchor_pos, radius, result):
		# Implementation of Douglas-Rachford projection algorithm
		# (This is a placeholder - actual implementation would depend on your system)
		# In the original code this seems to be imported from elsewhere
		pass

	def set_parameters(self, data, module_data, k):
		constraint_counter = 0  # Maps disc and obstacle index to a single index

		if k == 0:
			for i in range(self._max_obstacles + self._n_other_halfspaces):
				self.set_solver_parameter_lin_constraint_a1(k, self.solver.params, self._dummy_a1, constraint_counter)
				self.set_solver_parameter_lin_constraint_a2(k, self.solver.params, self._dummy_a2, constraint_counter)
				self.set_solver_parameter_lin_constraint_b(k, self.solver.params, self._dummy_b, constraint_counter)
				constraint_counter += 1
			return

		for d in range(self.n_discs):
			if not self._use_guidance:
				self.set_solver_parameter_ego_disc_offset(k, self.solver.params, data.robot_area[d].offset, d)

			# Set actual constraints
			for i in range(data.dynamic_obstacles.size() + self._n_other_halfspaces):
				self.set_solver_parameter_lin_constraint_a1(k, self.solver.params, self._a1[d][k][i],
				                                            constraint_counter)
				self.set_solver_parameter_lin_constraint_a2(k, self.solver.params, self._a2[d][k][i],
				                                            constraint_counter)
				self.set_solver_parameter_lin_constraint_b(k, self.solver.params, self._b[d][k][i], constraint_counter)
				constraint_counter += 1

			# Set dummy constraints for remaining slots
			for i in range(data.dynamic_obstacles.size() + self._n_other_halfspaces,
			               self._max_obstacles + self._n_other_halfspaces):
				self.set_solver_parameter_lin_constraint_a1(k, self.solver.params, self._dummy_a1, constraint_counter)
				self.set_solver_parameter_lin_constraint_a2(k, self.solver.params, self._dummy_a2, constraint_counter)
				self.set_solver_parameter_lin_constraint_b(k, self.solver.params, self._dummy_b, constraint_counter)
				constraint_counter += 1

	def is_data_ready(self, data, missing_data):
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
		if self._use_guidance and not CONFIG["debug_visuals"]:
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

	# Parameter setter methods
	def set_solver_parameter_lin_constraint_a1(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "lin_constraint_a1", value, k, index=i, settings=CONFIG)

	def set_solver_parameter_lin_constraint_a2(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "lin_constraint_a2", value, k, index=i, settings=CONFIG)

	def set_solver_parameter_lin_constraint_b(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "lin_constraint_b", value, k, index=i, settings=CONFIG)

	def set_solver_parameter_ego_disc_offset(self, k, params, value, d):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "ego_disc_offset", value, k, index=d, settings=CONFIG)