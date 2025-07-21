import numpy as np

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planning.src.types import Data, State, PredictionType
from utils.const import DETERMINISTIC, GAUSSIAN
from utils.math_utils import rotation_matrix
from utils.utils import LOG_DEBUG, LOG_WARN


class LinearizedConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)
		self.name = "linearized_constraints"

		LOG_DEBUG("Initializing Linearized Constraints")

		self.num_discs = int(self.get_config_value("num_discs"))
		self.num_other_halfspaces = self.get_config_value("linearized_constraints.add_halfspaces")
		self.max_obstacles = self.get_config_value("max_obstacles")
		self.num_constraints = self.max_obstacles + self.num_other_halfspaces
		self.nh = self.num_constraints
		self.use_slack = self.get_config_value("linearized_constraints.use_slack")

		if self.get_config_value("scenario_constraints.num_constraints"):
			self.num_constraints = self.get_config_value("scenario_constraints.num_constraints")
			self.use_slack = self.get_config_value("scenario_constraints.use_slack", True)

		# Initialize arrays with proper dimensions
		horizon = self.solver.horizon
		LOG_DEBUG("Horizon: {}".format(horizon))
		LOG_DEBUG("Num constraints: {}".format(self.num_constraints))
		LOG_DEBUG("Num discs: {}".format(self.num_discs))
		self._a1 = [[[0.0 for _ in range(self.num_constraints)] for _ in range(horizon)] for _ in range(self.num_discs)]
		self._a2 = [[[0.0 for _ in range(self.num_constraints)] for _ in range(horizon)] for _ in range(self.num_discs)]
		self._b = [[[0.0 for _ in range(self.num_constraints)] for _ in range(horizon)] for _ in range(self.num_discs)]
		LOG_DEBUG("_a1 initialized to : " + str(self._a1))
		LOG_DEBUG("_a2 initialized to : " + str(self._a2))

		self.num_obstacles = 0
		self.use_guidance = False

		# Store dummy values for invalid states
		self._dummy_a1 = 0.0
		self._dummy_a2 = 0.0
		self._dummy_b = 100.0

		LOG_DEBUG("Linearized Constraints successfully initialized")

	def set_topology_constraints(self):
		self.num_discs = 1  # Only one disc is used for the topology constraints
		self.use_guidance = True

	def update(self, state: State, data: Data):
		LOG_DEBUG("LinearizedConstraints.update")

		self._dummy_b = state.get("x") + 100.0

		# Thread safe copy of obstacles
		if not data.has("dynamic_obstacles") or data.dynamic_obstacles is None:
			LOG_WARN("No dynamic obstacles available")
			return

		copied_obstacles = data.dynamic_obstacles
		self.num_obstacles = len(copied_obstacles)
		LOG_DEBUG("Copied obstacles: {}".format(copied_obstacles))
		LOG_DEBUG("Num obstacles: {}".format(self.num_obstacles))
		# For all stages
		for k in range(1, self.solver.horizon):
			# Get reference trajectory position
			ref_states = self.solver.get_reference_trajectory().get_states()
			LOG_DEBUG("fetched reference states: {}".format(ref_states))
			if k >= len(ref_states):
				continue

			pos = np.array([
				ref_states[k].get("x"),
				ref_states[k].get("y")
			])

			for d in range(self.num_discs):
				if not self.use_guidance:  # Use discs and their positions
					if not data.has("robot_area") or data.robot_area is None or d >= len(data.robot_area):
						LOG_WARN(f"Robot area not available for disc {d}")
						continue
					disc = data.robot_area[d]
					disc_pos = disc.get_position(pos, ref_states[k].get("psi"))

					self.project_to_safety(copied_obstacles, k, disc_pos)
					pos = disc_pos
				else:  # Use the robot position
					self.project_to_safety(copied_obstacles, k, pos)

				# For all obstacles
				for obs_id in range(min(self.num_obstacles, self.max_obstacles)):
					copied_obstacle = copied_obstacles[obs_id]

					# Check if prediction exists and has enough timesteps
					if (not hasattr(copied_obstacle, 'prediction') or
							copied_obstacle.prediction is None or
							not hasattr(copied_obstacle.prediction, 'modes') or
							len(copied_obstacle.prediction.modes) == 0):
						LOG_WARN(f"Obstacle {obs_id} prediction not available for timestep {k}")
						continue

					obstacle_pos = np.array([
						copied_obstacle.position[0],
						copied_obstacle.position[1]
					])

					diff_x = obstacle_pos[0] - pos[0]
					diff_y = obstacle_pos[1] - pos[1]
					dist = np.linalg.norm(obstacle_pos - pos)

					# Avoid division by zero
					if dist < 1e-6:
						dist = 1e-6

					# Compute the components of A for this obstacle (normalized normal vector)
					self._a1[d][k][obs_id] = diff_x / dist
					self._a2[d][k][obs_id] = diff_y / dist
					LOG_DEBUG(f"a1 for {obs_id} set to {self._a1[d][k][obs_id]}, a2 for {obs_id} set to {self._a2[d][k][obs_id]}")
					# Compute b (evaluate point on the collision circle)
					radius = 1e-3 if self.use_guidance else copied_obstacle.radius
					disc_radius = self.get_config_value("disc_radius", 1.0)
					self._b[d][k][obs_id] = (self._a1[d][k][obs_id] * obstacle_pos[0] +
											 self._a2[d][k][obs_id] * obstacle_pos[1] -
											 (radius + disc_radius))
					LOG_DEBUG(f"b for {obs_id} set to {self._b[d][k][obs_id]}")
				# Handle static obstacles
				if data.has("static_obstacles") and data.static_obstacles is not None:
					if len(data.static_obstacles) < self.num_other_halfspaces:
						LOG_DEBUG(
							f"{self.num_other_halfspaces} halfspaces expected, but {len(data.static_obstacles)} are present")

						num_halfspaces = min(data.static_obstacles[k].size(), self.num_other_halfspaces)
						for h in range(num_halfspaces):
							obs_id = copied_obstacles.size() + h
							if obs_id < self.num_constraints:
								self._a1[d][k][obs_id] = data.static_obstacles[k][h].A[0]
								self._a2[d][k][obs_id] = data.static_obstacles[k][h].A[1]
								self._b[d][k][obs_id] = data.static_obstacles[k][h].b

		LOG_DEBUG("LinearizedConstraints.update done")

	def project_to_safety(self, copied_obstacles, k, pos):
		if len(copied_obstacles) == 0:  # There is no anchor
			return

		# Project to a collision free position if necessary, considering all the obstacles
		for _ in range(3):  # At most 3 iterations
			for obstacle in copied_obstacles:
				if (not hasattr(obstacle, 'prediction') or
						obstacle.prediction is None or
						not hasattr(obstacle.prediction, 'modes') or
						len(obstacle.prediction.modes) == 0):
					continue

				radius = 1e-3 if self.use_guidance else obstacle.radius
				disc_radius = self.get_config_value("disc_radius", 1.0)

				# Douglas-Rachford projection method
				self.douglas_rachford_projection(
					pos,
					obstacle.position[:2],
					radius + disc_radius,
					pos
				)

	def douglas_rachford_projection(self, pos, obstacle_pos, radius, result):
		direction = pos - obstacle_pos
		norm = np.linalg.norm(direction)
		if norm < radius:
			if norm < 1e-6:  # Avoid division by zero
				# Move in a random direction
				direction = np.array([1.0, 0.0])
				norm = 1.0
			correction = (radius - norm) * direction / norm
			result[:] = pos + correction

	def define_parameters(self, params):
		for disc_id in range(self.num_discs):
			params.add(f"ego_disc_{disc_id}_offset")

			for index in range(self.max_obstacles):
				params.add(self.constraint_name(index, disc_id) + "_a1")
				params.add(self.constraint_name(index, disc_id) + "_a2")
				params.add(self.constraint_name(index, disc_id) + "_b")
		LOG_DEBUG("Finished definition of parameters in LinearizedConstraints")

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

	def get_constraints(self, symbolic_state, params, stage_idx):
		"""
		Fixed method signature to match BaseConstraint interface
		"""
		constraints = []

		# States
		pos_x = symbolic_state.get("x")
		pos_y = symbolic_state.get("y")
		pos = np.array([pos_x, pos_y])
		psi = symbolic_state.get("psi")

		try:
			if self.use_slack:
				slack = symbolic_state.get("slack")
			else:
				slack = 0.0
		except:
			slack = 0.0

		rotation_car = rotation_matrix(psi)
		for disc_id in range(self.num_discs):
			disc_x = params.get(f"ego_disc_{disc_id}_offset")
			disc_relative_pos = np.array([disc_x, 0])
			disc_pos = pos + rotation_car.dot(disc_relative_pos)

			for index in range(self.max_obstacles):
				a1 = params.get(self.constraint_name(index, disc_id) + "_a1")
				a2 = params.get(self.constraint_name(index, disc_id) + "_a2")
				b = params.get(self.constraint_name(index, disc_id) + "_b")

				expr = a1 * disc_pos[0] + a2 * disc_pos[1] - (b + slack)
				# ✅ Skip constant constraints
				if hasattr(expr, "is_constant") and expr.is_constant():
					continue
				constraints.append(expr)

		return constraints

	def set_parameters(self, parameter_manager, data, k):
		# Initialize all parameters with dummy values first
		LOG_DEBUG("Setting parameters for linearized constraints")
		for disc_id in range(self.num_discs):
			if not self.use_guidance:
				if data.has("robot_area") and data.robot_area is not None and disc_id < len(data.robot_area):
					parameter_manager.set_parameter(f"ego_disc_{disc_id}_offset", data.robot_area[disc_id].offset)
				else:
					parameter_manager.set_parameter(f"ego_disc_{disc_id}_offset", 0.0)

			for i in range(self.max_obstacles):
				if k == 0:
					# Set dummy values for initial stage
					parameter_manager.set_parameter(self.constraint_name(i, disc_id) + "_a1", self._dummy_a1)
					parameter_manager.set_parameter(self.constraint_name(i, disc_id) + "_a2", self._dummy_a2)
					parameter_manager.set_parameter(self.constraint_name(i, disc_id) + "_b", self._dummy_b)
				else:
					# invalid constraint states
					if i >= self.num_obstacles:
						parameter_manager.set_parameter(self.constraint_name(i, disc_id) + "_a1", self._dummy_a1)
						parameter_manager.set_parameter(self.constraint_name(i, disc_id) + "_a2", self._dummy_a2)
						parameter_manager.set_parameter(self.constraint_name(i, disc_id) + "_b", self._dummy_b)
					else:
						# Set actual constraint values
						parameter_manager.set_parameter(self.constraint_name(i, disc_id) + "_a1", self._a1[disc_id][k][i])
						parameter_manager.set_parameter(self.constraint_name(i, disc_id) + "_a2", self._a2[disc_id][k][i])
						parameter_manager.set_parameter(self.constraint_name(i, disc_id) + "_b", self._b[disc_id][k][i])

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
		LOG_DEBUG("Missing data in linearized constraints: {}".format(missing_data))
		LOG_DEBUG("Obstacles: {}".format(data.dynamic_obstacles))
		return len(missing_data) < 1

	def reset(self):
		super().reset()
		# Reset constraint-specific state
		self.num_obstacles = 0

		# Re-initialize arrays with zeros
		for d in range(self.num_discs):
			for k in range(self.solver.horizon):
				for i in range(self.num_constraints):
					self._a1[d][k][i] = 0.0
					self._a2[d][k][i] = 0.0
					self._b[d][k][i] = 0.0
