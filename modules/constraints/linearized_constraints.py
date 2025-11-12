import numpy as np

from modules.constraints.base_constraint import BaseConstraint
from planning.types import Data, State, PredictionType
from utils.const import DETERMINISTIC, GAUSSIAN
from utils.math_tools import rotation_matrix
from utils.utils import LOG_DEBUG, LOG_INFO, LOG_WARN



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
		
		# Get reference trajectory - this should be the warmstart trajectory
		# CRITICAL: If reference trajectory is empty, use warmstart values directly
		ref_trajectory = self.solver.get_reference_trajectory()
		ref_states = ref_trajectory.get_states() if ref_trajectory else []
		
		# If reference trajectory is empty, try to create states from warmstart values
		if len(ref_states) == 0:
			LOG_WARN("LinearizedConstraints.update: Reference trajectory is EMPTY! Attempting to use warmstart values directly.")
			has_warmstart = hasattr(self.solver, 'warmstart_values') and self.solver.warmstart_values
			LOG_INFO(f"LinearizedConstraints.update: warmstart_values available: {has_warmstart}")
			if has_warmstart:
				ws_keys = list(self.solver.warmstart_values.keys()) if self.solver.warmstart_values else []
				LOG_INFO(f"LinearizedConstraints.update: warmstart_values keys: {ws_keys}")
				if 'x' in self.solver.warmstart_values:
					x_len = len(self.solver.warmstart_values['x']) if hasattr(self.solver.warmstart_values['x'], '__len__') else 'N/A'
					LOG_INFO(f"LinearizedConstraints.update: warmstart x length: {x_len}")
				
				# Create reference states from warmstart values
				horizon_val = self.solver.horizon if (hasattr(self.solver, 'horizon') and self.solver.horizon is not None) else 10
				from planning.types import State
				from planning.dynamic_models import DynamicsModel
				# Get dynamics model from data
				dynamics_model = None
				if data is not None and hasattr(data, 'dynamics_model'):
					dynamics_model = data.dynamics_model
				elif hasattr(self.solver, '_get_dynamics_model'):
					dynamics_model = self.solver._get_dynamics_model()
				
				LOG_INFO(f"LinearizedConstraints.update: dynamics_model available: {dynamics_model is not None}")
				
				if dynamics_model and 'x' in self.solver.warmstart_values and 'y' in self.solver.warmstart_values:
					ref_states = []
					x_vals = self.solver.warmstart_values['x']
					y_vals = self.solver.warmstart_values['y']
					psi_vals = self.solver.warmstart_values.get('psi', [0.0] * (horizon_val + 1))
					
					# Ensure arrays are the right length
					if not hasattr(x_vals, '__len__'):
						LOG_WARN(f"LinearizedConstraints.update: warmstart x is not an array: {type(x_vals)}")
					else:
						for k in range(min(horizon_val + 1, len(x_vals))):
							ref_state = State(model_type=dynamics_model)
							ref_state.set("x", float(x_vals[k]))
							ref_state.set("y", float(y_vals[k]))
							if k < len(psi_vals):
								ref_state.set("psi", float(psi_vals[k]))
							if 'v' in self.solver.warmstart_values and k < len(self.solver.warmstart_values['v']):
								ref_state.set("v", float(self.solver.warmstart_values['v'][k]))
							ref_states.append(ref_state)
						LOG_INFO(f"LinearizedConstraints.update: Created {len(ref_states)} reference states from warmstart values")
				else:
					missing = []
					if not dynamics_model:
						missing.append("dynamics_model")
					if 'x' not in self.solver.warmstart_values:
						missing.append("warmstart x")
					if 'y' not in self.solver.warmstart_values:
						missing.append("warmstart y")
					LOG_WARN(f"LinearizedConstraints.update: Cannot create reference states from warmstart - missing: {missing}")
			else:
				LOG_WARN("LinearizedConstraints.update: No warmstart values available either!")
		
		LOG_INFO(f"LinearizedConstraints.update: Using {len(ref_states)} reference states for constraint computation")
		if len(ref_states) > 0:
			# Log first few reference states to verify they're correct
			for i in range(min(3, len(ref_states))):
				ref_state = ref_states[i]
				ref_x = ref_state.get("x") if ref_state.has("x") else None
				ref_y = ref_state.get("y") if ref_state.has("y") else None
				LOG_INFO(f"  Reference state {i}: x={ref_x:.3f}, y={ref_y:.3f}" if ref_x is not None and ref_y is not None else f"  Reference state {i}: x={ref_x}, y={ref_y}")

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
				LOG_INFO(f"LinearizedConstraints.update: step={step} (stage 0), using CURRENT state position: ({ego_position[0]:.3f}, {ego_position[1]:.3f}), psi={ego_psi:.3f}")
			else:
				# Use reference trajectory position (matching C++: _solver->getEgoPrediction(k, "x"))
				if step < len(ref_states):
					ego_position = np.array([
						ref_states[step].get("x"),
						ref_states[step].get("y")
					])
					ego_psi = ref_states[step].get("psi")
					LOG_INFO(f"LinearizedConstraints.update: step={step}, using REFERENCE TRAJECTORY position: ({ego_position[0]:.3f}, {ego_position[1]:.3f}), psi={ego_psi:.3f}")
				else:
					# Fallback to state if reference trajectory is not long enough
					ego_position = np.array([
						state.get("x"),
						state.get("y")
					])
					ego_psi = state.get("psi")
					LOG_WARN(f"LinearizedConstraints.update: step={step}, reference trajectory too short (len={len(ref_states)}), using CURRENT state position: ({ego_position[0]:.3f}, {ego_position[1]:.3f})")
				
				# CRITICAL FIX: Project reference trajectory position away from obstacles
				# Matching reference implementation: linearization must happen at feasible points (outside obstacles)
				# The reference trajectory may go through obstacles, making linearization ineffective.
				# Project the position away from nearby obstacles to ensure constraints are linearized
				# at feasible positions. This is critical for solver convergence.
				robot_radius = self.get_config_value("robot_radius", 0.5) or self.disc_radius
				safety_margin = 0.3  # Increased safety margin for better feasibility
				
				# Find closest obstacle and project away from it
				closest_obs_id = None
				closest_dist = float('inf')
				closest_obs_pos = None
				closest_min_safe_distance = None
				
				for obs_id in range(len(copied_dynamic_obstacles)):
					obstacle = copied_dynamic_obstacles[obs_id]
					# Get obstacle position (use current position for static obstacles)
					if hasattr(obstacle, 'velocity') and obstacle.velocity is not None:
						vel_norm = np.linalg.norm(obstacle.velocity)
						is_static = vel_norm < 1e-3
					else:
						is_static = True  # Assume static if no velocity
					
					if is_static:
						obs_pos = np.array([obstacle.position[0], obstacle.position[1]])
					else:
						# For dynamic obstacles, use predicted position at step-1
						if step - 1 < len(obstacle.prediction.steps):
							obs_pos = np.array([
								obstacle.prediction.steps[step - 1].position[0],
								obstacle.prediction.steps[step - 1].position[1]
							])
						else:
							obs_pos = np.array([obstacle.position[0], obstacle.position[1]])
					
					# Compute distance to obstacle
					diff = ego_position - obs_pos
					dist = np.linalg.norm(diff)
					
					# Get combined radius (robot + obstacle)
					obstacle_radius = obstacle.radius if hasattr(obstacle, 'radius') else 0.35
					min_safe_distance = robot_radius + obstacle_radius + safety_margin
					
					# Track closest obstacle
					if dist < closest_dist:
						closest_dist = dist
						closest_obs_id = obs_id
						closest_obs_pos = obs_pos
						closest_min_safe_distance = min_safe_distance
				
				# Project away from closest obstacle if too close (even if just slightly inside)
				# This ensures linearization point is always feasible
				if closest_obs_id is not None and closest_dist < closest_min_safe_distance:
					if closest_dist < 1e-6:
						# Ego position is exactly at obstacle center - project in arbitrary direction
						# Use path tangent direction if available, otherwise use [1, 0]
						if hasattr(self.solver, 'data') and self.solver.data is not None:
							if hasattr(self.solver.data, 'reference_path') and self.solver.data.reference_path is not None:
								try:
									# Try to get path tangent at this step
									ref_states = self.solver.get_reference_trajectory().get_states()
									if step < len(ref_states):
										ref_psi = ref_states[step].get("psi")
										direction = np.array([np.cos(ref_psi), np.sin(ref_psi)])
									else:
										direction = np.array([1.0, 0.0])
								except:
									direction = np.array([1.0, 0.0])
							else:
								direction = np.array([1.0, 0.0])
						else:
							direction = np.array([1.0, 0.0])
					else:
						# Normalize direction vector (away from obstacle)
						direction = (ego_position - closest_obs_pos) / closest_dist
					
					# Project to safe distance (ensure we're outside the obstacle)
					ego_position = closest_obs_pos + direction * closest_min_safe_distance
					LOG_DEBUG(f"  Projected reference position at step {step} away from obstacle {closest_obs_id}: "
					         f"distance {closest_dist:.3f}m -> {closest_min_safe_distance:.3f}m")

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
					# For static obstacles (zero velocity), always use the initial position
					# For dynamic obstacles, use predicted position at step k-1 (C++: obstacle.prediction.modes[0][k-1].position)
					# Check if obstacle is static by checking if velocity is zero or if all prediction steps have the same position
					is_static = False
					if hasattr(target_obstacle, 'velocity') and target_obstacle.velocity is not None:
						vel_norm = np.linalg.norm(target_obstacle.velocity)
						is_static = vel_norm < 1e-3
					elif len(target_obstacle.prediction.steps) > 1:
						# Check if all prediction steps have the same position (static obstacle)
						first_pos = target_obstacle.prediction.steps[0].position
						all_same = all(np.linalg.norm(step.position - first_pos) < 1e-3 
						                for step in target_obstacle.prediction.steps[1:])
						is_static = all_same
					
					if is_static:
						# For static obstacles, always use the initial position (obstacle doesn't move)
						target_obstacle_pos = np.array([
							target_obstacle.position[0],
							target_obstacle.position[1]
						])
						LOG_DEBUG(f"  Static obstacle {obs_id} at step {step}: using fixed position ({target_obstacle_pos[0]:.2f}, {target_obstacle_pos[1]:.2f})")
					else:
						# For dynamic obstacles, prioritize actual current position for stage 0
						# For future stages, use predicted positions but ensure they're reasonable
						if step == 0:
							# For stage 0, always use actual current obstacle position
							# This ensures constraints remain accurate even when obstacles turn rapidly
							target_obstacle_pos = np.array([
								target_obstacle.position[0],
								target_obstacle.position[1]
							])
							LOG_DEBUG(f"  Dynamic obstacle {obs_id} at step {step}: using actual current position ({target_obstacle_pos[0]:.2f}, {target_obstacle_pos[1]:.2f})")
						else:
							# For future stages, use predicted position at step k-1
							# But validate that prediction is reasonable (not too far from current position)
							current_obstacle_pos = np.array([
								target_obstacle.position[0],
								target_obstacle.position[1]
							])
							
							# Try to get predicted position
							predicted_pos = None
							if step - 1 < len(target_obstacle.prediction.steps):
								predicted_pos = np.array([
									target_obstacle.prediction.steps[step - 1].position[0],
									target_obstacle.prediction.steps[step - 1].position[1]
								])
							elif len(target_obstacle.prediction.steps) > 0:
								# Use last available prediction
								last_step = len(target_obstacle.prediction.steps) - 1
								predicted_pos = np.array([
									target_obstacle.prediction.steps[last_step].position[0],
									target_obstacle.prediction.steps[last_step].position[1]
								])
							
							# Validate prediction is reasonable (not moving faster than solver can handle)
							# Maximum reasonable distance: max_velocity * timestep * step
							# Use conservative estimate: 3 m/s * 0.1s * step = 0.3 * step meters
							if predicted_pos is not None:
								prediction_distance = np.linalg.norm(predicted_pos - current_obstacle_pos)
								max_reasonable_distance = 0.3 * step  # Conservative: 3 m/s * 0.1s * step
								
								if prediction_distance > max_reasonable_distance:
									# Prediction seems unreasonable (obstacle moving too fast)
									# Use extrapolated position based on current velocity instead
									if hasattr(target_obstacle, 'velocity') and target_obstacle.velocity is not None:
										# Limit velocity to reasonable maximum (3 m/s)
										vel = np.array(target_obstacle.velocity)
										vel_norm = np.linalg.norm(vel)
										if vel_norm > 3.0:
											vel = vel / vel_norm * 3.0
										
										# Extrapolate position: current_pos + velocity * timestep * step
										timestep = 0.1  # Default timestep (should match solver timestep)
										if hasattr(self.solver, 'timestep') and self.solver.timestep is not None:
											timestep = float(self.solver.timestep)
										target_obstacle_pos = current_obstacle_pos + vel * timestep * step
										LOG_DEBUG(f"  Dynamic obstacle {obs_id} at step {step}: prediction distance {prediction_distance:.2f}m > max {max_reasonable_distance:.2f}m, using velocity-extrapolated position ({target_obstacle_pos[0]:.2f}, {target_obstacle_pos[1]:.2f})")
									else:
										# No velocity available, use current position
										target_obstacle_pos = current_obstacle_pos
										LOG_DEBUG(f"  Dynamic obstacle {obs_id} at step {step}: prediction unreasonable, no velocity available, using current position ({target_obstacle_pos[0]:.2f}, {target_obstacle_pos[1]:.2f})")
								else:
									# Prediction is reasonable, use it
									target_obstacle_pos = predicted_pos
									LOG_DEBUG(f"  Dynamic obstacle {obs_id} at step {step}: using validated prediction step {step-1} position ({target_obstacle_pos[0]:.2f}, {target_obstacle_pos[1]:.2f})")
							else:
								# No prediction available, extrapolate from current position and velocity
								if hasattr(target_obstacle, 'velocity') and target_obstacle.velocity is not None:
									vel = np.array(target_obstacle.velocity)
									vel_norm = np.linalg.norm(vel)
									if vel_norm > 3.0:
										vel = vel / vel_norm * 3.0
									
									timestep = 0.1
									if hasattr(self.solver, 'timestep') and self.solver.timestep is not None:
										timestep = float(self.solver.timestep)
									target_obstacle_pos = current_obstacle_pos + vel * timestep * step
									LOG_DEBUG(f"  Dynamic obstacle {obs_id} at step {step}: no prediction available, using velocity-extrapolated position ({target_obstacle_pos[0]:.2f}, {target_obstacle_pos[1]:.2f})")
								else:
									# No velocity available, use current position
									target_obstacle_pos = current_obstacle_pos
									LOG_DEBUG(f"  Dynamic obstacle {obs_id} at step {step}: no prediction or velocity, using current position ({target_obstacle_pos[0]:.2f}, {target_obstacle_pos[1]:.2f})")

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

					# Log constraint computation details for verification
					LOG_INFO(f"  Step {step}, disc {disc_id}, obstacle {obs_id}: ego_pos=({ego_position[0]:.3f}, {ego_position[1]:.3f}), "
					        f"obs_pos=({target_obstacle_pos[0]:.3f}, {target_obstacle_pos[1]:.3f}), "
					        f"dist={dist:.3f}, a1={self._a1[disc_id][step][obs_id]:.4f}, a2={self._a2[disc_id][step][obs_id]:.4f}, "
					        f"b={self._b[disc_id][step][obs_id]:.4f}")

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
			LOG_DEBUG(f"LinearizedConstraints.calculate_constraints: stage_idx={stage_idx} >= horizon={horizon_val}, returning empty constraints")
			return constraints

		LOG_INFO(f"LinearizedConstraints.calculate_constraints: stage_idx={stage_idx}, num_discs={self.num_discs}, num_active_obstacles={self.num_active_obstacles}, num_other_halfspaces={self.num_other_halfspaces}")

		for disc_id in range(self.num_discs):
			# Resolve disc offset from robot_area if available
			disc_offset = 0.0
			if not self.use_guidance and data.has("robot_area") and data.robot_area is not None and disc_id < len(data.robot_area):
				disc_offset = float(data.robot_area[disc_id].offset)

			disc_constraints = 0
			for index in range(self.num_active_obstacles + self.num_other_halfspaces):
				a1 = self._a1[disc_id][stage_idx][index] if self._a1[disc_id][stage_idx][index] is not None else self._dummy_a1
				a2 = self._a2[disc_id][stage_idx][index] if self._a2[disc_id][stage_idx][index] is not None else self._dummy_a2
				b = self._b[disc_id][stage_idx][index] if self._b[disc_id][stage_idx][index] is not None else self._dummy_b
				# Skip degenerate constraints (dummy constraints have a1=0, a2=0)
				if abs(a1) < 1e-9 and abs(a2) < 1e-9:
					continue
				# Add constraint - ensure it's for a valid obstacle index
				# For actual obstacles (index < num_active_obstacles), always add
				# For other halfspaces (index >= num_active_obstacles), also add
				constraint_type = "obstacle" if index < self.num_active_obstacles else "halfspace"
				constraints.append({"a1": float(a1), "a2": float(a2), "b": float(b), "disc_offset": disc_offset})
				disc_constraints += 1
				# Log constraint details for all stages
				# Also evaluate constraint at current vehicle position to check if it's violated
				constraint_violated = False
				constraint_value = None
				if stage_idx == 0 and state is not None:
					try:
						# Get current vehicle position
						vehicle_x = float(state.get("x")) if state.has("x") else None
						vehicle_y = float(state.get("y")) if state.has("y") else None
						if vehicle_x is not None and vehicle_y is not None:
							# Evaluate constraint: a1*x + a2*y <= b
							constraint_value = a1 * vehicle_x + a2 * vehicle_y
							constraint_violated = constraint_value > b
							if constraint_violated:
								LOG_WARN(f"  ⚠️  Linearized constraint stage {stage_idx}, disc {disc_id}, {constraint_type} {index}: "
								        f"VIOLATED! constraint_value={constraint_value:.4f} > b={b:.4f} "
								        f"at vehicle position ({vehicle_x:.3f}, {vehicle_y:.3f})")
							else:
								LOG_INFO(f"  ✓ Linearized constraint stage {stage_idx}, disc {disc_id}, {constraint_type} {index}: "
								        f"a1={a1:.4f}, a2={a2:.4f}, b={b:.4f}, disc_offset={disc_offset:.4f}, "
								        f"constraint_value={constraint_value:.4f} <= b={b:.4f} at vehicle ({vehicle_x:.3f}, {vehicle_y:.3f})")
						else:
							LOG_INFO(f"  Linearized constraint stage {stage_idx}, disc {disc_id}, {constraint_type} {index}: "
							        f"a1={a1:.4f}, a2={a2:.4f}, b={b:.4f}, disc_offset={disc_offset:.4f}")
					except Exception as e:
						LOG_INFO(f"  Linearized constraint stage {stage_idx}, disc {disc_id}, {constraint_type} {index}: "
						        f"a1={a1:.4f}, a2={a2:.4f}, b={b:.4f}, disc_offset={disc_offset:.4f} (could not evaluate: {e})")
				else:
					LOG_INFO(f"  Linearized constraint stage {stage_idx}, disc {disc_id}, {constraint_type} {index}: "
					        f"a1={a1:.4f}, a2={a2:.4f}, b={b:.4f}, disc_offset={disc_offset:.4f}")
			
			if disc_constraints == 0:
				LOG_DEBUG(f"  Disc {disc_id} at stage {stage_idx}: no valid constraints (all degenerate)")

		LOG_INFO(f"LinearizedConstraints.calculate_constraints: stage_idx={stage_idx}, returning {len(constraints)} constraints")
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
