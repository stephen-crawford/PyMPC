import numpy as np

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planning.src.data_prep import remove_distant_obstacles, filter_distant_obstacles
from planning.src.types import Data, State, PredictionType
from utils.const import DETERMINISTIC, GAUSSIAN
from utils.math_utils import rotation_matrix, DouglasRachford
from utils.utils import LOG_DEBUG, LOG_WARN


class LinearizedConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)
		self.name = "linearized_constraints"

		LOG_DEBUG("Initializing Linearized Constraints")

		self.num_discs = int(self.get_config_value("num_discs"))
		self.num_other_halfspaces = self.get_config_value("linearized_constraints.add_halfspaces")
		self.max_obstacles = self.get_config_value("max_obstacles")
		self.filter_distant_obstacles = self.get_config_value("linearized_constraints.filter_distant_obstacles")
		self.max_num_constraints = self.max_obstacles + self.num_other_halfspaces

		self.use_guidance = False
		self.use_slack = self.get_config_value("linearized_constraints.use_slack")

		self.use_topology_constraints = self.get_config_value("linearized_constraints.use_topology_constraints")
		if self.use_topology_constraints:
			self.num_discs = 1  # Only one disc is used for the topology constraints
			self.use_guidance = True

		self.disc_radius = self.get_config_value("disc_radius", 1.0)

		self._a1 = [None] * self.num_discs
		self._a2 = [None] * self.num_discs
		self._b = [None] * self.num_discs

		for disc_id in range(self.num_discs):
			self._a1[disc_id] = [None] * self.solver.horizon
			self._a2[disc_id] = [None] * self.solver.horizon
			self._b[disc_id] = [None] * self.solver.horizon
			for step in range(self.solver.horizon):
				self._a1[disc_id][step] = [None] * self.max_num_constraints
				self._a2[disc_id][step] = [None] * self.max_num_constraints
				self._b[disc_id][step] = [None] * self.max_num_constraints

		self.num_active_obstacles = 0

		# Store dummy values for invalid states
		self._dummy_a1 = 0.0
		self._dummy_a2 = 0.0
		self._dummy_b = 100.0

		LOG_DEBUG("Linearized Constraints successfully initialized")

	def update(self, state: State, data: Data):
		LOG_DEBUG("LinearizedConstraints.update")

		self._dummy_b = state.get("x") + 100.0

		# Thread safe copy of obstacles
		if not data.has("dynamic_obstacles") or data.dynamic_obstacles is None:
			LOG_WARN("No dynamic obstacles available")
			return

		copied_dynamic_obstacles = data.dynamic_obstacles
		if self.filter_distant_obstacles:
			copied_dynamic_obstacles = filter_distant_obstacles(data.dynamic_obstacles, state, 5)
		self.num_active_obstacles = len(copied_dynamic_obstacles)
		ref_states = self.solver.get_reference_trajectory().get_states() # This gets a horizon length trajectory of the ego robot
		LOG_DEBUG("fetched reference states: {}".format(ref_states))

		for step in range(1, self.solver.horizon):

			ego_position = np.array([
				ref_states[step].get("x"),
				ref_states[step].get("y")
			])

			ego_psi = ref_states[step].get("psi")

			for disc_id in range(self.num_discs):

				if not self.use_guidance:  # Use discs and their positions
					if not data.has("robot_area") or data.robot_area is None or disc_id >= len(data.robot_area):
						LOG_WARN(f"Robot area not available for disc {disc_id}")
						continue

					active_disc = data.robot_area[disc_id]
					disc_position = active_disc.get_position(ego_position, ego_psi)
					self.project_to_safety(copied_dynamic_obstacles, step, disc_position)
					ego_position = disc_position

				else:  # Use the robot position
					self.project_to_safety(copied_dynamic_obstacles, step, ego_position)

				# For all obstacles
				for obs_id in range(len(copied_dynamic_obstacles)):
					target_obstacle = copied_dynamic_obstacles[obs_id]
					LOG_DEBUG("copied obstacle step prediction: {}".format(target_obstacle.prediction.steps))

					target_obstacle_pos = np.array([
						target_obstacle.prediction.steps[step - 1].position[0],
						target_obstacle.prediction.steps[step - 1].position[1]
					])

					LOG_DEBUG("Obstacle position is: {}".format(target_obstacle_pos))

					# FIX 3: Normal vector points FROM ego TO obstacle (matching C++)
					diff_x = target_obstacle_pos[0] - ego_position[0]  # FROM ego TO obstacle
					diff_y = target_obstacle_pos[1] - ego_position[1]  # FROM ego TO obstacle
					dist = np.linalg.norm(target_obstacle_pos - ego_position)

					# Normal vector pointing toward obstacle center
					self._a1[disc_id][step][obs_id] = float(diff_x / dist)
					self._a2[disc_id][step][obs_id] = float(diff_y / dist)

					target_obstacle_radius = 1e-3 if self.use_guidance else target_obstacle.radius

					self._b[disc_id][step][obs_id] = (self._a1[disc_id][step][obs_id] * target_obstacle_pos[0] +
													  self._a2[disc_id][step][obs_id] * target_obstacle_pos[1] -
													  (target_obstacle_radius + self.disc_radius))

					LOG_DEBUG(f"b for {obs_id} set to {self._b[disc_id][step][obs_id]}")


				if data.has("static_obstacles") and data.static_obstacles is not None:

						max_num_halfspaces = min(len(data.static_obstacles), self.num_other_halfspaces)

						for halfspace_id in range(max_num_halfspaces):
							target_obs_id = len(copied_dynamic_obstacles) + halfspace_id
							if target_obs_id < self.max_num_constraints:
								self._a1[disc_id][step][target_obs_id] = float(data.static_obstacles[halfspace_id].A[0])
								self._a2[disc_id][step][target_obs_id] = float(data.static_obstacles[halfspace_id].A[1])
								self._b[disc_id][step][target_obs_id] = float(data.static_obstacles[halfspace_id].b)

		LOG_DEBUG("LinearizedConstraints.update done")

	def project_to_safety(self, copied_obstacles, step, pos):
		dr = DouglasRachford()
		if len(copied_obstacles) == 0:  # There is no anchor
			return

		# Project to a collision free position if necessary, considering all the obstacles
		for i in range(3):  # At most 3 iterations
			for obstacle in copied_obstacles:

				radius = 1e-3 if self.use_guidance else obstacle.radius
				robot_radius = self.get_config_value("robot_radius", 1.0)

				anchor_pos = np.array([
					copied_obstacles[0].prediction.steps[step].position[0],
					copied_obstacles[0].prediction.steps[step].position[1]
				])

				obstacle_pos = np.array([obstacle.prediction.steps[step].position[0], obstacle.prediction.steps[step].position[1]])

				dr.douglas_rachford_projection(
					pos,  # p: point to project (modified in-place)
					obstacle_pos,  # delta: obstacle position
					anchor_pos,  # anchor: anchor position
					radius + robot_radius,  # r: collision radius
					pos
				)

	def get_lower_bound(self):
		lower_bound = []
		for index in range(0, self.num_active_obstacles):
			lower_bound.append(-np.inf)
		return lower_bound

	def get_upper_bound(self):
		upper_bound = []
		for index in range(0, self.num_active_obstacles):
			upper_bound.append(0.0)
		return upper_bound

	def get_constraints(self, symbolic_state, params, stage_idx):
		LOG_DEBUG("LinearizedConstraints.get_constraints called with stage_idx: {}".format(stage_idx))
		constraints = []

		if stage_idx >= self.solver.horizon:
			return constraints

		# States
		pos_x = symbolic_state.get("x")
		pos_y = symbolic_state.get("y")
		pos = np.array([pos_x, pos_y])
		psi = symbolic_state.get("psi")

		try:
			slack = symbolic_state.get("slack") if self.use_slack else 0.0
		except:
			slack = 0.0

		rotation_car = rotation_matrix(psi)

		for disc_id in range(self.num_discs):
			disc_x = params.get(f"ego_disc_{disc_id}_offset")
			disc_relative_pos = np.array([disc_x, 0])
			disc_pos = pos + rotation_car.dot(disc_relative_pos)

			for index in range(self.num_active_obstacles):
				a1 = params.get(self.constraint_name(index, stage_idx, disc_id) + "_a1")
				a2 = params.get(self.constraint_name(index, stage_idx, disc_id) + "_a2")
				b = params.get(self.constraint_name(index, stage_idx, disc_id) + "_b")

				# Skip dummy/invalid constraints
				if abs(a1) < 1e-6 and abs(a2) < 1e-6:
					LOG_DEBUG(f"Skipping dummy constraint for obstacle {index}, disc {disc_id}")
					continue

				constraint_expr = a1 * disc_pos[0] + a2 * disc_pos[1] - (b + slack)
				constraints.append(constraint_expr)

		return constraints

	def define_parameters(self, params):
		"""Define parameters for the ParameterManager"""
		# Define ego disc offset parameters
		for disc_id in range(self.num_discs):
			if not self.use_guidance:
				params.add(f"ego_disc_{disc_id}_offset")

		# Define constraint parameters for each disc, step, and obstacle
		for disc_id in range(self.num_discs):
			for step in range(self.solver.horizon + 1):
				for obstacle_id in range(self.max_obstacles + self.num_other_halfspaces):
					# Create unique parameter names
					base_name = self.constraint_name(obstacle_id, step, disc_id)
					params.add(f"{base_name}_a1")
					params.add(f"{base_name}_a2")
					params.add(f"{base_name}_b")

		LOG_DEBUG("Finished definition of parameters in LinearizedConstraints")

	def constraint_name(self, index, step, disc_id=None):
		"""Generate constraint parameter names"""
		if disc_id is None:
			return f"lin_constraint_{index}_step_{step}"
		return f"disc_{disc_id}_lin_constraint_{index}_step_{step}"

	def set_parameters(self, parameter_manager, data, step):
		"""Set parameters using the ParameterManager"""

		if step == 0:
			# Set dummy values for step 0
			for disc_id in range(self.num_discs):
				for constraint_id in range(self.max_obstacles + self.num_other_halfspaces):
					base_name = self.constraint_name(constraint_id, step, disc_id)
					parameter_manager.set_parameter(f"{base_name}_a1", self._dummy_a1)
					parameter_manager.set_parameter(f"{base_name}_a2", self._dummy_a2)
					parameter_manager.set_parameter(f"{base_name}_b", self._dummy_b)
			return

		for disc_id in range(self.num_discs):
			# Set ego disc offset if not using guidance
			if not self.use_guidance:
				if data.has("robot_area") and data.robot_area is not None and disc_id < len(data.robot_area):
					offset_value = data.robot_area[disc_id].offset
				else:
					offset_value = 0.0
				parameter_manager.set_parameter(f"ego_disc_{disc_id}_offset", offset_value)

			# Get number of active obstacles
			num_dynamic_obstacles = len(data.dynamic_obstacles) if data.has(
				"dynamic_obstacles") and data.dynamic_obstacles is not None else 0

			# Set parameters for active dynamic obstacles + static halfspaces
			for obstacle_id in range(num_dynamic_obstacles + self.num_other_halfspaces):
				base_name = self.constraint_name(obstacle_id, step, disc_id)

				if obstacle_id < num_dynamic_obstacles + self.num_other_halfspaces:
					# Use actual constraint values if available
					if (obstacle_id < len(self._a1[disc_id][step]) and
							self._a1[disc_id][step][obstacle_id] is not None):

						a1_val = float(self._a1[disc_id][step][obstacle_id])
						a2_val = float(self._a2[disc_id][step][obstacle_id])
						b_val = float(self._b[disc_id][step][obstacle_id])
					else:
						# Use dummy values if constraint data not available
						a1_val = self._dummy_a1
						a2_val = self._dummy_a2
						b_val = self._dummy_b
				else:
					# Use dummy values for inactive constraints
					a1_val = self._dummy_a1
					a2_val = self._dummy_a2
					b_val = self._dummy_b

				# Set the parameter values
				parameter_manager.set_parameter(f"{base_name}_a1", a1_val)
				parameter_manager.set_parameter(f"{base_name}_a2", a2_val)
				parameter_manager.set_parameter(f"{base_name}_b", b_val)

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
		if len(missing_data) > 0:
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
				for i in range(self.num_active_obstacles):
					self._a1[d][k][i] = self._dummy_a1
					self._a2[d][k][i] = self._dummy_a2
					self._b[d][k][i] = self._dummy_b
