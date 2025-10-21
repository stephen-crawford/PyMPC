import casadi as cd
import numpy as np

from .base_constraint import BaseConstraint
from pympc.utils.math_utils import chi_square_quantile
from pympc.utils.utils import LOG_DEBUG


class EllipsoidConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)
		self.name = 'ellipsoid_constraints'
		self.num_segments = self.get_config_value("contouring.num_segments")

		self.num_discs = self.get_config_value("num_discs")
		self.robot_radius = self.get_config_value("robot.radius")
		self.risk = self.get_config_value("probabilistic.risk")

		self.max_obstacles = self.get_config_value("max_obstacles")
		self.num_constraints = self.max_obstacles * self.num_discs

		# Dummy values for invalid states
		self._dummy_x = 50.0
		self._dummy_y = 50.0

		LOG_DEBUG("Ellipsoid Constraints successfully initialized")

	def get_visualization_overlay(self):
		"""Best-effort overlay for ellipsoids (parameters live in manager; keep None)."""
		return None

	def update(self, state, data):
		LOG_DEBUG("EllipsoidConstraints.update")

		# Update dummy values based on current state
		self._dummy_x = state.get("x") + 50.0
		self._dummy_y = state.get("y") + 50.0

	def constraint_name(self, index, step, disc_id=None):
		"""Generate constraint parameter names"""
		if disc_id is None:
			return f"ellipsoid_obs_{index}_step_{step}"
		return f"disc_{disc_id}_ellipsoid_constraint_{index}_step_{step}"

	def define_parameters(self, params):
		params.add(f"ego_disc_radius")

		for disc_id in range(self.num_discs):
			params.add(f"ego_disc_{disc_id}_offset")

		for obs_id in range(self.max_obstacles):
			for step in range(self.solver.horizon + 1):
				base_name = self.constraint_name(obs_id, step)
				params.add(f"{base_name}_x")
				params.add(f"{base_name}_y")
				params.add(f"{base_name}_psi")
				params.add(f"{base_name}_major")
				params.add(f"{base_name}_minor")
				params.add(f"{base_name}_chi")
				params.add(f"{base_name}_r")

	def set_parameters(self, parameter_manager, data, step):
		LOG_DEBUG(
			f"set_parameters called with step={step}, num_obstacles={len(data.dynamic_obstacles) if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles else 0}")

		parameter_manager.set_parameter("ego_disc_radius", self.robot_radius)

		for disc_id in range(self.num_discs):
			if disc_id >= len(data.robot_area):
				LOG_DEBUG(f"ERROR: disc_id {disc_id} >= len(robot_area) {len(data.robot_area)}")
				break
			parameter_manager.set_parameter(f"ego_disc_{disc_id}_offset", data.robot_area[disc_id].offset)

		# Set parameters for ALL obstacles and ALL steps to avoid missing parameter errors
		for obs_id in range(self.max_obstacles):
			for step_idx in range(self.solver.horizon + 1):
				base_name = self.constraint_name(obs_id, step_idx)

				# Default dummy values
				default_values = {
					"x": self._dummy_x, "y": self._dummy_y,
					"psi": 0.0, "r": 0.1,
					"major": 0.1, "minor": 0.1, "chi": 1.0
				}

				# Set dummy values first
				for name, value in default_values.items():
					parameter_manager.set_parameter(f"{base_name}_{name}", value)

				# Override with actual data if available
				if (hasattr(data, 'dynamic_obstacles') and
						data.dynamic_obstacles and
						obs_id < len(data.dynamic_obstacles)):

					current_obstacle = data.dynamic_obstacles[obs_id]

					if (current_obstacle.prediction is not None and
							len(current_obstacle.prediction.steps) > 0):

						# Use the appropriate prediction step (or last available if not enough steps)
						prediction_step_idx = min(step_idx, len(current_obstacle.prediction.steps) - 1)

						LOG_DEBUG(
							f"Setting real data for obstacle {obs_id}, step {step_idx}, using prediction step {prediction_step_idx}")

						# Set actual obstacle data
						parameter_manager.set_parameter(f"{base_name}_x",
														current_obstacle.prediction.steps[prediction_step_idx].position[
															0])
						parameter_manager.set_parameter(f"{base_name}_y",
														current_obstacle.prediction.steps[prediction_step_idx].position[
															1])
						parameter_manager.set_parameter(f"{base_name}_psi",
														current_obstacle.prediction.steps[prediction_step_idx].angle)
						parameter_manager.set_parameter(f"{base_name}_r", current_obstacle.radius)

						# Compute uncertainty scaling
						chi = chi_square_quantile(dof=2, alpha=1.0 - self.risk)

						if current_obstacle.prediction.type == PredictionType.DETERMINISTIC:
							major, minor = 0.1, 0.1

						elif current_obstacle.prediction.type == PredictionType.GAUSSIAN:
							major = max(0.1,
										current_obstacle.prediction.steps[prediction_step_idx].major_radius * np.sqrt(
											chi))
							minor = max(0.1,
										current_obstacle.prediction.steps[prediction_step_idx].minor_radius * np.sqrt(
											chi))

						elif current_obstacle.prediction.type == PredictionType.NONGAUSSIAN:
							major = max(0.1,
										current_obstacle.prediction.steps[prediction_step_idx].major_radius * np.sqrt(
											chi) * 1.5)
							minor = max(0.1,
										current_obstacle.prediction.steps[prediction_step_idx].minor_radius * np.sqrt(
											chi) * 1.5)
						else:
							major, minor = 0.1, 0.1

						# Update with actual values
						parameter_manager.set_parameter(f"{base_name}_major", major)
						parameter_manager.set_parameter(f"{base_name}_minor", minor)

	def get_lower_bound(self):
		lower_bound = []
		for obs in range(self.max_obstacles):
			for disc in range(self.num_discs):
				lower_bound.append(1.0)
		return lower_bound

	def get_upper_bound(self):
		upper_bound = []
		for obs in range(self.max_obstacles):
			for disc in range(self.num_discs):
				upper_bound.append(np.inf)
		return upper_bound

	def get_constraints(self, symbolic_state, parameter_manager, stage_idx):
		constraints = []
		pos_x = symbolic_state.get("x")
		pos_y = symbolic_state.get("y")
		pos = cd.vertcat(pos_x, pos_y)

		try:
			psi = symbolic_state.get("psi")
		except:
			psi = 0.0

		# Build rotation matrix directly with CasADi operations
		cos_psi = cd.cos(psi)
		sin_psi = cd.sin(psi)
		rotation_car = cd.vertcat(
			cd.horzcat(cos_psi, -sin_psi),
			cd.horzcat(sin_psi, cos_psi)
		)

		r_disc = parameter_manager.get("ego_disc_radius")

		# Constraint for dynamic obstacles
		for obs_id in range(self.max_obstacles):
			base_name = self.constraint_name(obs_id, stage_idx)
			obst_x = parameter_manager.get(f"{base_name}_x")
			obst_y = parameter_manager.get(f"{base_name}_y")
			obstacle_cog = cd.vertcat(obst_x, obst_y)

			obst_psi = parameter_manager.get(f"{base_name}_psi")
			obst_major = parameter_manager.get(f"{base_name}_major")
			obst_minor = parameter_manager.get(f"{base_name}_minor")
			obst_r = parameter_manager.get(f"{base_name}_r")

			# Chi scaling is already applied in set_parameters, so don't apply again
			chi = parameter_manager.get(f"{base_name}_chi")

			# Create ellipse matrix using proper CasADi syntax
			major_denom = obst_major + r_disc + obst_r
			minor_denom = obst_minor + r_disc + obst_r

			# Handle potential division by zero
			major_inv_sq = cd.if_else(major_denom > 1e-6, 1.0 / (major_denom * major_denom), 1e6)
			minor_inv_sq = cd.if_else(minor_denom > 1e-6, 1.0 / (minor_denom * minor_denom), 1e6)

			# Create diagonal matrix properly
			ab = cd.diag(cd.vertcat(major_inv_sq, minor_inv_sq))

			# Build obstacle rotation matrix
			cos_obst_psi = cd.cos(obst_psi)
			sin_obst_psi = cd.sin(obst_psi)
			obstacle_rotation = cd.vertcat(
				cd.horzcat(cos_obst_psi, -sin_obst_psi),
				cd.horzcat(sin_obst_psi, cos_obst_psi)
			)

			# Compute ellipse matrix: R^T * A * R
			obstacle_ellipse_matrix = obstacle_rotation.T @ ab @ obstacle_rotation

			for disc_id in range(self.num_discs):
				# Get disc offset
				disc_x = parameter_manager.get(f"ego_disc_{disc_id}_offset")
				disc_relative_pos = cd.vertcat(disc_x, 0)

				# Compute disc position in global frame
				disc_pos = pos + rotation_car @ disc_relative_pos

				# Compute constraint: (p - c)^T * Q * (p - c) >= 1
				disc_to_obstacle = disc_pos - obstacle_cog
				c_disc_obstacle = disc_to_obstacle.T @ obstacle_ellipse_matrix @ disc_to_obstacle
				constraints.append(c_disc_obstacle)

		return constraints

	def is_data_ready(self, data):
		missing_data = ""
		if not data.has("dynamic_obstacles"):
			missing_data += "Obstacles "

		for i in range(len(data.dynamic_obstacles)):
			if data.dynamic_obstacles[i].prediction.empty():
				print("Found missing obstacle pred: " + str(data.dynamic_obstacles[i].prediction.empty()))

			if (not (data.dynamic_obstacles[i].prediction.type == PredictionType.GAUSSIAN) and
					not (data.dynamic_obstacles[i].prediction.type == PredictionType.DETERMINISTIC)):
				missing_data += "Obstacle Prediction (Type is incorrect) "

		LOG_DEBUG("Missing data for ellipsoid constraints has length: " + str(len(missing_data)))
		LOG_DEBUG("Missing data for ellipsoid constraints is: " + str(missing_data))
		return len(missing_data) < 1
