import numpy as np
import casadi as cd

from planning.types import PredictionType
from utils.const import GAUSSIAN, DYNAMIC
from utils.math_tools import exponential_quantile, rotation_matrix, casadi_rotation_matrix
from utils.utils import LOG_DEBUG, PROFILE_SCOPE, CONFIG, LOG_INFO
from modules.constraints.base_constraint import BaseConstraint

class GaussianConstraints(BaseConstraint):
	def __init__(self):
		super().__init__()
		self.name = 'gaussian_constraints'
		# Store dummy values for invalid states
		self._dummy_x = 0.0
		self._dummy_y = 0.0
		self.num_discs = self.get_config_value("num_discs")
		self.robot_radius = self.get_config_value("robot.radius")
		self.max_obstacles = self.get_config_value("max_obstacles")
		self.num_constraints = self.num_discs * self.max_obstacles
		self.num_active_obstacles = 0

		LOG_DEBUG("Gaussian Constraints successfully initialized")

	def update(self, state, data):
		LOG_DEBUG("GaussianConstraints.update")

		# Update dummy values based on current state
		self._dummy_x = state.get("x") + 100.0
		self._dummy_y = state.get("y") + 100.0

		copied_dynamic_obstacles = data.dynamic_obstacles
		self.num_active_obstacles = len(copied_dynamic_obstacles)

	def calculate_constraints(self, state, data, stage_idx):
		# TODO: convert probabilistic constraints to structured form; placeholder returns []
			return []

	def lower_bounds(self):
		return [0.0] * (self.num_discs * self.num_active_obstacles)

	def upper_bounds(self):
		return [np.inf] * (self.num_discs * self.num_active_obstacles)

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

		return len(missing_data) < 1