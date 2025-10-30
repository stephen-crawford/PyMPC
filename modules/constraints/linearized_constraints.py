import numpy as np

from modules.constraints.base_constraint import BaseConstraint
from planning.types import Data, State, PredictionType
from utils.const import DETERMINISTIC, GAUSSIAN
from utils.math_tools import rotation_matrix
from utils.utils import LOG_DEBUG, LOG_WARN


class LinearizedConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__()
		self.name = "linearized_constraints"

		LOG_DEBUG("Initializing Linearized Constraints")

		self.solver = solver
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
		# Note: filter_distant_obstacles import adjusted elsewhere if needed
		# if self.filter_distant_obstacles:
		# 	copied_dynamic_obstacles = filter_distant_obstacles(data.dynamic_obstacles, state, 5)
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

					# Normal vector points FROM ego TO obstacle
					diff_x = target_obstacle_pos[0] - ego_position[0]
					diff_y = target_obstacle_pos[1] - ego_position[1]
					dist = np.linalg.norm(target_obstacle_pos - ego_position)
					if dist < 1e-6:
						dist = 1e-6
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
		# Placeholder projection if needed; left as-is
		return

	def calculate_constraints(self, state: State, data: Data, stage_idx: int):
		"""Return structured linear constraints for the solver to convert.
		Each constraint is a dict: {a1,a2,b,disc_offset}.
		"""
		constraints = []
		if stage_idx >= self.solver.horizon:
			return constraints

		for disc_id in range(self.num_discs):
			# Resolve disc offset from robot_area if available
			disc_offset = 0.0
			if not self.use_guidance and data.has("robot_area") and data.robot_area is not None and disc_id < len(data.robot_area):
				disc_offset = float(data.robot_area[disc_id].offset)

			for index in range(self.num_active_obstacles + self.num_other_halfspaces):
				a1 = self._a1[disc_id][stage_idx][index] if self._a1[disc_id][stage_idx][index] is not None else self._dummy_a1
				a2 = self._a2[disc_id][stage_idx][index] if self._a2[disc_id][stage_idx][index] is not None else self._dummy_a2
				b = self._b[disc_id][stage_idx][index] if self._b[disc_id][stage_idx][index] is not None else self._dummy_b
				# Skip degenerate
				if abs(a1) < 1e-9 and abs(a2) < 1e-9:
					continue
				constraints.append({"a1": float(a1), "a2": float(a2), "b": float(b), "disc_offset": disc_offset})
		return constraints

	def lower_bounds(self):
		# All constraints are of form a^T p - b >= 0 → lb = 0
		return [0.0] * (self.num_active_obstacles + self.num_other_halfspaces)

	def upper_bounds(self):
		# Upper bounds 0 for linearized constraints in this form
		return [np.inf] * (self.num_active_obstacles + self.num_other_halfspaces)

	def is_data_ready(self, data):
		missing_data = ""
		if not data.has("dynamic_obstacles") or data.dynamic_obstacles is None:
			missing_data += "Dynamic Obstacles "
			LOG_DEBUG("Missing dynamic_obstacles: {}".format(missing_data))
		return len(missing_data) < 1

	def reset(self):
		super().reset()
		self.num_obstacles = 0
		for d in range(self.num_discs):
			for k in range(self.solver.horizon):
				for i in range(self.num_active_obstacles):
					self._a1[d][k][i] = self._dummy_a1
					self._a2[d][k][i] = self._dummy_a2
					self._b[d][k][i] = self._dummy_b
