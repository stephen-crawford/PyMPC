import casadi as cd
import numpy as np

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planning.src.types import PredictionType, PredictionStep
from solver.src.parameter_manager import ParameterManager
from utils.const import DETERMINISTIC, GAUSSIAN
from utils.math_utils import exponential_quantile, rotation_matrix, chi_square_quantile
from utils.utils import LOG_DEBUG


class EllipsoidConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)
		self.name = 'ellipsoid_constraints'
		self.num_segments = self.get_config_value("contouring.num_segments")
		self.num_discs = self.get_config_value("num_discs")
		self.robot_radius = self.get_config_value("robot.radius")
		self.max_obstacles = self.get_config_value("max_obstacles")
		self.risk = self.get_config_value("probabilistic.risk")

		# Dummy values for invalid states
		self._dummy_x = 50.0
		self._dummy_y = 50.0

		LOG_DEBUG("Ellipsoid Constraints successfully initialized")

	def update(self, state, data):
		LOG_DEBUG("EllipsoidConstraints.update")

		# Update dummy values based on current state
		self._dummy_x = state.get("x") + 50.0
		self._dummy_y = state.get("y") + 50.0

	def define_parameters(self, params):
		params.add(f"ego_disc_radius")

		for disc_id in range(self.num_discs):
			params.add(f"ego_disc_{disc_id}_offset")


		for obs_id in range(self.max_obstacles):
			params.add(f"ellipsoid_obst_{obs_id}_x")
			params.add(f"ellipsoid_obst_{obs_id}_y")
			params.add(f"ellipsoid_obst_{obs_id}_psi")
			params.add(f"ellipsoid_obst_{obs_id}_major")
			params.add(f"ellipsoid_obst_{obs_id}_minor")
			params.add(f"ellipsoid_obst_{obs_id}_chi")
			params.add(f"ellipsoid_obst_{obs_id}_r")

	def set_parameters(self, parameter_manager, data, k):
		parameter_manager.set_parameter("ego_disc_radius", self.robot_radius)

		for d in range(self.num_discs):
			if hasattr(data, 'robot_area') and len(data.robot_area) > d:
				parameter_manager.set_parameter(f"ego_disc_{d}_offset", data.robot_area[d].offset)

		if k == 0:  # Dummy values
			for i in range(self.max_obstacles):
				for name, value in [
					("x", self._dummy_x), ("y", self._dummy_y),
					("psi", 0.0), ("r", 0.1),
					("major", 0.0), ("minor", 0.0), ("chi", 1.0)
				]:
					parameter_manager.set_parameter(f"ellipsoid_obst_{i}_{name}", value)
			return

		for i, obstacle in enumerate(data.dynamic_obstacles):
			pred = obstacle.prediction
			if pred is None or len(pred.steps) == 0:
				continue

			step = pred.steps[0]

			# Set obstacle base position & orientation
			parameter_manager.set_parameter(f"ellipsoid_obst_{i}_x", step.position[0])
			parameter_manager.set_parameter(f"ellipsoid_obst_{i}_y", step.position[1])
			parameter_manager.set_parameter(f"ellipsoid_obst_{i}_psi", step.angle)
			parameter_manager.set_parameter(f"ellipsoid_obst_{i}_r", obstacle.radius)

			# Compute uncertainty scaling
			chi = chi_square_quantile(dof=2, alpha=1.0 - self.risk)

			if pred.type == PredictionType.DETERMINISTIC:
				major, minor = 0.0, 0.0

			elif pred.type == PredictionType.GAUSSIAN:
				# Apply chi scaling for confidence region
				major = step.major_radius * np.sqrt(chi)
				minor = step.minor_radius * np.sqrt(chi)

			elif pred.type == PredictionType.NONGAUSSIAN:
				major = step.major_radius * np.sqrt(chi) * 1.5
				minor = step.minor_radius * np.sqrt(chi) * 1.5
			else:
				major, minor = 0.0, 0.0

			# Store parameters
			parameter_manager.set_parameter(f"ellipsoid_obst_{i}_major", major)
			parameter_manager.set_parameter(f"ellipsoid_obst_{i}_minor", minor)
			parameter_manager.set_parameter(f"ellipsoid_obst_{i}_chi", chi)

			# Debug
			print(f"[Ellipsoid] Obstacle {i}: major={major:.3f}, minor={minor:.3f}, chi={chi:.3f}")

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

	def get_constraints(self, model, parameter_manager: ParameterManager, stage_idx):
		constraints = []
		pos_x = model.get("x")
		pos_y = model.get("y")
		# Use CasADi vertcat instead of np.array for symbolic variables
		pos = cd.vertcat(pos_x, pos_y)

		try:
			psi = model.get("psi")
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
			obst_x = parameter_manager.get(f"ellipsoid_obst_{obs_id}_x")
			obst_y = parameter_manager.get(f"ellipsoid_obst_{obs_id}_y")
			# Use CasADi vertcat instead of np.array for parameters that might be symbolic
			obstacle_cog = cd.vertcat(obst_x, obst_y)

			obst_psi = parameter_manager.get(f"ellipsoid_obst_{obs_id}_psi")
			obst_major = parameter_manager.get(f"ellipsoid_obst_{obs_id}_major")
			obst_minor = parameter_manager.get(f"ellipsoid_obst_{obs_id}_minor")
			obst_r = parameter_manager.get(f"ellipsoid_obst_{obs_id}_r")

			# multiplier for the risk when obst_major, obst_major only denote the covariance
			# (i.e., the smaller the risk, the larger the ellipsoid).
			# This value should already be a multiplier (using exponential cdf).
			chi = parameter_manager.get(f"ellipsoid_obst_{obs_id}_chi")

			# Compute ellipse matrix
			obst_major *= cd.sqrt(chi)
			obst_minor *= cd.sqrt(chi)
			ab = cd.MX(2, 2)
			ab[0, 0] = 1.0 / ((obst_major + r_disc + obst_r) * (obst_major + r_disc + obst_r))
			ab[0, 1] = 0.0
			ab[1, 0] = 0.0
			ab[1, 1] = 1.0 / ((obst_minor + r_disc + obst_r) * (obst_minor + r_disc + obst_r))

			# Build obstacle rotation matrix directly with CasADi operations
			cos_obst_psi = cd.cos(obst_psi)
			sin_obst_psi = cd.sin(obst_psi)
			obstacle_rotation = cd.vertcat(
				cd.horzcat(cos_obst_psi, -sin_obst_psi),
				cd.horzcat(sin_obst_psi, cos_obst_psi)
			)
			obstacle_ellipse_matrix = obstacle_rotation.T @ ab @ obstacle_rotation

			for disc_id in range(self.num_discs):
				# Get and compute the disc position
				disc_x = parameter_manager.get(f"ego_disc_{disc_id}_offset")
				# Use CasADi vertcat instead of np.array for the relative position
				disc_relative_pos = cd.vertcat(disc_x, 0)

				# Simple matrix multiplication - CasADi should handle this correctly now
				disc_pos = pos + rotation_car @ disc_relative_pos

				# construct the constraint and append it
				disc_to_obstacle = disc_pos - obstacle_cog
				# Chain matrix multiplication
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
