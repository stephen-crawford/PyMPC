import numpy as np

from modules.constraints.base_constraint import BaseConstraint
from planning.types import Data, State, PredictionType
from utils.const import DETERMINISTIC, GAUSSIAN
from utils.math_tools import rotation_matrix
from utils.utils import LOG_DEBUG, LOG_WARN



class LinearizedConstraints(BaseConstraint):
	def __init__(self):
		super().__init__()
		self.name = "linearized_constraints"

		LOG_DEBUG("Initializing Linearized Constraints")

		# Solver will be set by framework later
		self.solver = None
		num_discs_val = self.get_config_value("num_discs")
		self.num_discs = int(num_discs_val) if num_discs_val is not None else 1
		self.num_other_halfspaces = self.get_config_value("linearized_constraints.add_halfspaces") or 0
		self.max_obstacles = self.get_config_value("max_obstacles") or 10
		self.filter_distant_obstacles = self.get_config_value("linearized_constraints.filter_distant_obstacles") or False
		self.max_num_constraints = self.max_obstacles + self.num_other_halfspaces

		self.use_guidance = False
		self.use_slack = self.get_config_value("linearized_constraints.use_slack")

		self.use_topology_constraints = self.get_config_value("linearized_constraints.use_topology_constraints")
		if self.use_topology_constraints:
			self.num_discs = 1  # Only one disc is used for the topology constraints
			self.use_guidance = True

		self.disc_radius = self.get_config_value("disc_radius", 1.0)

		# Use horizon with fallback - solver will be set later by framework
		horizon_val = 10  # Default, will be updated when solver is set

		self._a1 = [None] * self.num_discs
		self._a2 = [None] * self.num_discs
		self._b = [None] * self.num_discs

		for disc_id in range(self.num_discs):
			self._a1[disc_id] = [None] * horizon_val
			self._a2[disc_id] = [None] * horizon_val
			self._b[disc_id] = [None] * horizon_val
			for step in range(horizon_val):
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

		horizon_val = self.solver.horizon if (hasattr(self.solver, 'horizon') and self.solver.horizon is not None) else 10
		
		# Compute constraints for all stages including stage 0 (current vehicle position)
		# C++ code only does k >= 1, but we need constraints at stage 0 to prevent the vehicle from
		# violating halfspaces at its current position
		for step in range(0, horizon_val):
			# For stage 0, use current state position; for stage >= 1, use reference trajectory
			if step == 0:
				# Use current vehicle position from state (matching C++: state.get("x"))
				ego_position = np.array([
					state.get("x"),
					state.get("y")
				])
				ego_psi = state.get("psi")
			else:
				# Use reference trajectory position (matching C++: _solver->getEgoPrediction(k, "x"))
				if step < len(ref_states):
					ego_position = np.array([
						ref_states[step].get("x"),
						ref_states[step].get("y")
					])
					ego_psi = ref_states[step].get("psi")
				else:
					# Fallback to state if reference trajectory is not long enough
					ego_position = np.array([
						state.get("x"),
						state.get("y")
					])
					ego_psi = state.get("psi")

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

				# For all obstacles (matching C++ implementation)
				for obs_id in range(len(copied_dynamic_obstacles)):
					target_obstacle = copied_dynamic_obstacles[obs_id]
					
					# Get obstacle position
					# For stage 0, use current obstacle position (step 0)
					# For stage >= 1, use predicted position at step k-1 (C++: obstacle.prediction.modes[0][k-1].position)
					if step == 0:
						# Use current obstacle position
						if len(target_obstacle.prediction.steps) > 0:
							target_obstacle_pos = np.array([
								target_obstacle.prediction.steps[0].position[0],
								target_obstacle.prediction.steps[0].position[1]
							])
						else:
							# Fallback to obstacle's initial position
							target_obstacle_pos = np.array([
								target_obstacle.position[0],
								target_obstacle.position[1]
							])
					else:
						# Use predicted position at step k-1
						if step - 1 < len(target_obstacle.prediction.steps):
							target_obstacle_pos = np.array([
								target_obstacle.prediction.steps[step - 1].position[0],
								target_obstacle.prediction.steps[step - 1].position[1]
							])
						else:
							# Fallback to last available prediction
							if len(target_obstacle.prediction.steps) > 0:
								last_step = len(target_obstacle.prediction.steps) - 1
								target_obstacle_pos = np.array([
									target_obstacle.prediction.steps[last_step].position[0],
									target_obstacle.prediction.steps[last_step].position[1]
								])
							else:
								target_obstacle_pos = np.array([
									target_obstacle.position[0],
									target_obstacle.position[1]
								])

					# Compute difference vector FROM ego TO obstacle (C++: diff_x = obstacle_pos(0) - pos(0))
					diff_x = target_obstacle_pos[0] - ego_position[0]
					diff_y = target_obstacle_pos[1] - ego_position[1]
					
					# Compute distance (C++: dist = (obstacle_pos - pos).norm())
					dist = np.linalg.norm(target_obstacle_pos - ego_position)
					
					# Normalize direction vector (C++: _a1[d][k](obs_id) = diff_x / dist)
					if dist < 1e-6:
						# Avoid division by zero - use arbitrary direction
						diff_x, diff_y = 1.0, 0.0
						dist = 1.0
					
					self._a1[disc_id][step][obs_id] = float(diff_x / dist)
					self._a2[disc_id][step][obs_id] = float(diff_y / dist)

					# Get obstacle radius (C++: radius = _use_guidance ? 1e-3 : obstacle.radius)
					target_obstacle_radius = 1e-3 if self.use_guidance else target_obstacle.radius
					
					# Get robot radius from config (C++: CONFIG["robot_radius"])
					robot_radius = self.get_config_value("robot_radius", 0.5)
					if robot_radius is None:
						robot_radius = self.disc_radius  # Fallback to disc_radius
					else:
						robot_radius = float(robot_radius)

					# Compute b value (C++: _b[d][k](obs_id) = _a1[d][k](obs_id) * obstacle_pos(0) + 
					#                                    _a2[d][k](obs_id) * obstacle_pos(1) - 
					#                                    (radius + CONFIG["robot_radius"]))
					self._b[disc_id][step][obs_id] = (self._a1[disc_id][step][obs_id] * target_obstacle_pos[0] +
													  self._a2[disc_id][step][obs_id] * target_obstacle_pos[1] -
													  (target_obstacle_radius + robot_radius))

					LOG_DEBUG(f"b for {obs_id} set to {self._b[disc_id][step][obs_id]}")

				# Handle static obstacles (matching C++: module_data.static_obstacles[k])
				# In Python, data.static_obstacles is a list of StaticObstacle objects, one per stage
				# Each StaticObstacle has a halfspaces attribute which is a list of Halfspace objects
				if data.has("static_obstacles") and data.static_obstacles is not None:
					# Get static obstacle for this stage (C++: module_data.static_obstacles[k])
					if step < len(data.static_obstacles) and data.static_obstacles[step] is not None:
						static_obstacle = data.static_obstacles[step]
						
						# Get halfspaces for this stage (C++: module_data.static_obstacles[k].size())
						if hasattr(static_obstacle, 'halfspaces') and static_obstacle.halfspaces is not None:
							num_halfspaces = min(len(static_obstacle.halfspaces), self.num_other_halfspaces)
							
							# Process each halfspace (C++: for (int h = 0; h < num_halfspaces; h++))
							for halfspace_id in range(num_halfspaces):
								target_obs_id = len(copied_dynamic_obstacles) + halfspace_id
								if target_obs_id < self.max_num_constraints:
									halfspace = static_obstacle.halfspaces[halfspace_id]
									# Extract A and b from halfspace (C++: module_data.static_obstacles[k][h].A(0), etc.)
									if hasattr(halfspace, 'A') and hasattr(halfspace, 'b'):
										A = halfspace.A
										if isinstance(A, np.ndarray) and A.size >= 2:
											self._a1[disc_id][step][target_obs_id] = float(A[0])
											self._a2[disc_id][step][target_obs_id] = float(A[1])
											self._b[disc_id][step][target_obs_id] = float(halfspace.b)
										elif hasattr(halfspace, 'a1') and hasattr(halfspace, 'a2'):
											# Alternative format: halfspace has a1, a2, b attributes
											self._a1[disc_id][step][target_obs_id] = float(halfspace.a1)
											self._a2[disc_id][step][target_obs_id] = float(halfspace.a2)
											self._b[disc_id][step][target_obs_id] = float(halfspace.b)

		LOG_DEBUG("LinearizedConstraints.update done")

	def project_to_safety(self, copied_obstacles, step, pos):
		# Placeholder projection if needed; left as-is
			return

	def calculate_constraints(self, state: State, data: Data, stage_idx: int):
		"""Return structured linear constraints for the solver to convert.
		Each constraint is a dict: {a1,a2,b,disc_offset}.
		"""
		constraints = []
		horizon_val = self.solver.horizon if (hasattr(self.solver, 'horizon') and self.solver.horizon is not None) else 10
		if stage_idx >= horizon_val:
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

	def lower_bounds(self, state=None, data=None, stage_idx=None):
		# For linearized constraints: expr = a1*x + a2*y - b
		# Matching reference implementation: lower_bound = -inf, upper_bound = 0.0
		# This enforces: a1*x + a2*y - b <= 0, i.e., a1*x + a2*y <= b
		count = len(self.calculate_constraints(state, data, stage_idx)) if (data is not None and stage_idx is not None) else 0
		return [-np.inf] * count

	def upper_bounds(self, state=None, data=None, stage_idx=None):
		# For linearized constraints: expr = a1*x + a2*y - b
		# Matching reference implementation: lower_bound = -inf, upper_bound = 0.0
		# This enforces: a1*x + a2*y - b <= 0, i.e., a1*x + a2*y <= b
		count = len(self.calculate_constraints(state, data, stage_idx)) if (data is not None and stage_idx is not None) else 0
		return [0.0] * count

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
			horizon_val = self.solver.horizon if (hasattr(self.solver, 'horizon') and self.solver.horizon is not None) else 10
			for k in range(horizon_val):
				for i in range(self.num_active_obstacles):
					self._a1[d][k][i] = self._dummy_a1
					self._a2[d][k][i] = self._dummy_a2
					self._b[d][k][i] = self._dummy_b
