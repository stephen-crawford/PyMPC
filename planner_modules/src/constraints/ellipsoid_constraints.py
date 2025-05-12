import casadi as cd
import numpy as np

from planner_modules.src.constraints.base_constraint import BaseConstraint
from solver.src.parameter_manager import ParameterManager
from utils.const import DETERMINISTIC, GAUSSIAN
from utils.math_utils import exponential_quantile, rotation_matrix
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

	def update(self, state, data, module_data):
		LOG_DEBUG("EllipsoidConstraints.update")

		# Update dummy values based on current state
		self._dummy_x = state.get("x") + 50.0
		self._dummy_y = state.get("y") + 50.0

	def define_parameters(self, params):

		for disc_id in range(self.num_discs):
			params.add(f"ego_disc_{disc_id}_offset")
			params.add(f"ego_disc_{disc_id}_radius")

		for obs_id in range(self.max_obstacles):
			params.add(f"ellipsoid_obst_{obs_id}_x")
			params.add(f"ellipsoid_obst_{obs_id}_y")
			params.add(f"ellipsoid_obst_{obs_id}_psi")
			params.add(f"ellipsoid_obst_{obs_id}_major")
			params.add(f"ellipsoid_obst_{obs_id}_minor")
			params.add(f"ellipsoid_obst_{obs_id}_chi")
			params.add(f"ellipsoid_obst_{obs_id}_r")

	def set_parameters(self, parameter_manager, data, module_data, k):
		print("num discs is", self.num_discs)
		for d in range(self.num_discs):

			# Set solver parameter for ego disc radius
			parameter_manager.set_parameter(f"ego_disc_{d}_radius", self.robot_radius)

			# Set solver parameter for ego disc offset
			if hasattr(data, 'robot_area') and len(data.robot_area) > d:
				parameter_manager.set_parameter(f"ego_disc_{d}_offset", data.robot_area[d].offset)

		if k == 0:  # Dummies
			# Set dummy values for obstacles at k=0
			for i in range(data.dynamic_obstacles.size()):
				parameter_manager.set_parameter(f"ellipsoid_obst_{i}_x", self._dummy_x)
				parameter_manager.set_parameter(f"ellipsoid_obst_{i}_y", self._dummy_y)
				parameter_manager.set_parameter(f"ellipsoid_obst_{i}_psi", 0.0)
				parameter_manager.set_parameter(f"ellipsoid_obst_{i}_r", 0.1)
				parameter_manager.set_parameter(f"ellipsoid_obst_{i}_major", 0.0)
				parameter_manager.set_parameter(f"ellipsoid_obst_{i}_minor", 0.0)
				parameter_manager.set_parameter(f"ellipsoid_obst_{i}_chi", 1.0)
			return

		if k == 1:
			LOG_DEBUG("EllipsoidConstraints::set_parameters")

		# Set parameters for each dynamic obstacle
		for i in range(data.dynamic_obstacles.size()):
			obstacle = data.dynamic_obstacles[i]
			mode = obstacle.prediction.modes[0]

			# The first prediction step is index 1 of the optimization problem
			# k-1 maps to the predictions for this stage
			parameter_manager.set_parameter(f"ellipsoid_obst_{i}_x", mode[k - 1].position[0])
			parameter_manager.set_parameter(f"ellipsoid_obst_{i}_y", mode[k - 1].position[1])
			parameter_manager.set_parameter(f"ellipsoid_obst_{i}_psi", mode[k - 1].angle)
			parameter_manager.set_parameter(f"ellipsoid_obst_{i}_r", obstacle.radius)

			if obstacle.prediction.type == DETERMINISTIC:
				# For deterministic obstacles, set zeros for uncertainty ellipse
				parameter_manager.set_parameter(f"ellipsoid_obst_{i}_major", 0.0)
				parameter_manager.set_parameter(f"ellipsoid_obst_{i}_minor", 0.0)
				parameter_manager.set_parameter(f"ellipsoid_obst_{i}_chi", 1.0)

			elif obstacle.prediction.type == GAUSSIAN:
				# Calculate chi-square quantile for desired risk level
				chi = exponential_quantile(0.5, 1.0 - self.risk)

				# Set uncertainty ellipse parameters
				parameter_manager.set_parameter(f"ellipsoid_obst_{i}_major", mode[k - 1].major_radius)
				parameter_manager.set_parameter(f"ellipsoid_obst_{i}_minor", mode[k - 1].minor_radius)
				parameter_manager.set_parameter(f"ellipsoid_obst_{i}_chi", chi)

		if k == 1:
			LOG_DEBUG("EllipsoidConstraints::set_parameters Done")

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

	def get_constraints(self, model, parameter_manager: ParameterManager, settings, stage_idx):
		constraints = []
		pos_x = model.get("x")
		pos_y = model.get("y")
		pos = np.array([pos_x, pos_y])

		try:
			psi = model.get("psi")
		except:
			psi = 0.0

		rotation_car = rotation_matrix(psi)

		r_disc = parameter_manager.get("ego_disc_radius")

		# Constraint for dynamic obstacles
		for obs_id in range(self.max_obstacles):
			obst_x = parameter_manager.get(f"ellipsoid_obst_{obs_id}_x")
			obst_y = parameter_manager.get(f"ellipsoid_obst_{obs_id}_y")
			obstacle_cog = np.array([obst_x, obst_y])

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
			ab = cd.SX(2, 2)
			ab[0, 0] = 1.0 / ((obst_major + r_disc + obst_r) * (obst_major + r_disc + obst_r))
			ab[0, 1] = 0.0
			ab[1, 0] = 0.0
			ab[1, 1] = 1.0 / ((obst_minor + r_disc + obst_r) * (obst_minor + r_disc + obst_r))

			obstacle_rotation = cd.SX(rotation_matrix(obst_psi))
			obstacle_ellipse_matrix = obstacle_rotation.T @ ab @ obstacle_rotation

			for disc_id in range(self.num_discs):
				# Get and compute the disc position
				disc_x = parameter_manager.get(f"ego_disc_{disc_id}_offset")
				disc_relative_pos = np.array([disc_x, 0])
				disc_pos = pos + rotation_car @ disc_relative_pos

				# construct the constraint and append it
				disc_to_obstacle = cd.SX(disc_pos - obstacle_cog)
				c_disc_obstacle = disc_to_obstacle.T @ obstacle_ellipse_matrix @ disc_to_obstacle
				constraints.append(c_disc_obstacle)  # + slack)

		return constraints

	def is_data_ready(self, data):
		missing_data = ""
		if len(data.dynamic_obstacles) != self.get_config_value("max_obstacles"):
			missing_data += "Obstacles "


		for i in range(len(data.dynamic_obstacles)):
			if data.dynamic_obstacles[i].prediction.empty():
				print("Found missing obstacle pred: " + str(data.dynamic_obstacles[i].prediction.empty()))

			if (not (data.dynamic_obstacles[i].prediction.type.equals(GAUSSIAN)) and
					not (data.dynamic_obstacles[i].prediction.type.equals(DETERMINISTIC))):
				missing_data += "Obstacle Prediction (Type is incorrect) "


		return len(missing_data) < 1
