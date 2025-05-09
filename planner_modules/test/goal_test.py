import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planner_modules.src.objectives.goal_objective import GoalObjective
from utils.const import GAUSSIAN, DETERMINISTIC, OBJECTIVE

CONFIG_MOCK = {
	"contouring": {
		"num_segments": 10
	},
	"num_discs": 2,
	"robot": {
		"radius": 0.5
	},
	"probabilistic": {
		"risk": 0.05
	},
	"max_obstacles": 3,
	"debug_visuals": False,
	"goal_weight": 0.5
}

def get_mocked_config(key, default=None):
	"""Static method to handle config mocking"""
	keys = key.split('.')
	cfg = CONFIG_MOCK
	try:
		for k in keys:
			cfg = cfg[k]
		return cfg
	except (KeyError, TypeError):
		return default


class TestGoalObjective(unittest.TestCase):
	"""Test suite for EllipsoidConstraints class"""

	def setUp(self):
		"""Set up test fixtures before each test"""

		patcher = patch('planner_modules.src.objectives.base_objective.BaseObjective.get_config_value',
						side_effect=get_mocked_config)
		self.mock_get_config = patcher.start()
		self.addCleanup(patcher.stop)

		"""Set up test fixtures before each test"""
		# Create mock solver
		self.solver = MagicMock()
		self.solver.horizon = 10
		self.solver.params = MagicMock()
		# Create instance of the class under test
		self.goal_objective = GoalObjective(self.solver)

	def test_initialization(self):
		"""Test proper initialization of EllipsoidConstraints"""
		self.assertEqual(self.goal_objective.module_type, OBJECTIVE)
		self.assertEqual(self.goal_objective.name, "goal_objective")
		self.assertEqual(self.goal_objective.goal_weight, CONFIG_MOCK["goal_weight"])

	def test_update(self):
		"""Test update method with valid data"""
		# Setup
		state = MagicMock()
		state.get.side_effect = lambda key: 10.0 if key == "x" else 20.0 if key == "y" else 0.0

		data = MagicMock()
		module_data = MagicMock()

		# Call method under test
		self.goal_objective.update(state, data, module_data)


	def test_set_parameters_k0(self):
		"""Test set_parameters method for k=0 (dummies)"""
		# Setup

		k = 0
		data = MagicMock()
		data.robot_area = [MagicMock(), MagicMock()]
		data.robot_area[0].offset = np.array([0.5, 0.3])
		data.robot_area[1].offset = np.array([1.0, 0.0])

		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = 2

		module_data = MagicMock()

		# Call method under test
		self.goal_objective.set_parameters(self.solver.params, data, module_data, 0)
		assert (self.solver.params.set_parameter.call_count == 3)

	def test_set_parameters_k1_deterministic(self):
		"""Test set_parameters method for k=1 with deterministic obstacles"""
		# Setup
		k = 1
		data = MagicMock()
		data.robot_area = [MagicMock(), MagicMock()]
		data.robot_area[0].offset = np.array([0.5, 0.3])
		data.robot_area[1].offset = np.array([1.0, 0.0])

		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = 2

		# Setup dynamic obstacles
		obstacle = MagicMock()
		obstacle.radius = 1.0
		obstacle.prediction.type = DETERMINISTIC

		# Setup prediction mode
		mode = MagicMock()
		mode.__getitem__.return_value = MagicMock()
		mode.__getitem__.return_value.position = [5.0, 6.0]
		mode.__getitem__.return_value.angle = 0.5

		obstacle.prediction.modes = [mode]

		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = 1
		data.dynamic_obstacles.__getitem__.return_value = obstacle

		module_data = MagicMock()

		module_data = MagicMock()

		# Call method under test
		self.goal_objective.set_parameters(self.solver.params, data, module_data, 1)
		assert (self.solver.params.set_parameter.call_count == 3)

	def test_set_parameters_k1_gaussian(self):
		"""Test set_parameters method for k=1 with gaussian obstacles"""
		# Setup

		k = 1
		data = MagicMock()
		data.robot_area = [MagicMock(), MagicMock()]
		data.robot_area[0].offset = np.array([0.5, 0.3])
		data.robot_area[1].offset = np.array([1.0, 0.0])

		# Setup dynamic obstacles
		obstacle = MagicMock()
		obstacle.radius = 1.0
		obstacle.prediction.type = GAUSSIAN

		# Setup prediction mode
		mode = MagicMock()
		mode_k0 = MagicMock()
		mode_k0.position = [5.0, 6.0]
		mode_k0.angle = 0.5
		# Add major and minor radius for gaussian obstacle
		mode_k0.major_radius = 2.0
		mode_k0.minor_radius = 1.5

		# Configure mode to return mode_k0 when indexed with (k-1)
		mode.__getitem__.return_value = mode_k0

		obstacle.prediction.modes = [mode]

		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = 1
		data.dynamic_obstacles.__getitem__.return_value = obstacle

		module_data = MagicMock()

		module_data = MagicMock()

		# Call method under test
		self.goal_objective.set_parameters(self.solver.params, data, module_data, 1)
		assert (self.solver.params.set_parameter.call_count == 3)

	def test_is_data_ready(self):
		"""Test is_data_ready method"""
		data = MagicMock()
		dyn_obstacle = MagicMock()
		dyn_obstacle.prediction = MagicMock()
		dyn_obstacle.predction.type = GAUSSIAN
		dyn_obstacle.prediction.modes = [GAUSSIAN]
		dyn_obstacle.prediction.empty.return_value = False
		data.dynamic_obstacles = [dyn_obstacle, dyn_obstacle, dyn_obstacle]

		result = self.goal_objective.is_data_ready(data)
		self.assertTrue(result)

class TestSystemIntegration(unittest.TestCase):
	"""Test integration between EllipsoidConstraints and Planner"""

	def setUp(self):
		"""Set up test fixtures before each test"""
		# Create mock solver
		self.solver = MagicMock()
		self.solver.horizon = 10
		self.solver.params = MagicMock()

		patcher = patch('planner_modules.src.objectives.base_objective.BaseObjective.get_config_value',
						side_effect=get_mocked_config)
		self.mock_get_config = patcher.start()
		self.addCleanup(patcher.stop)

		"""Set up test fixtures before each test"""
		# Create mock solver
		self.solver = MagicMock()
		self.solver.horizon = 10
		self.solver.params = MagicMock()
		# Create instance of the class under test
		self.goal_objective = GoalObjective(self.solver)
		# Create mock planner
		self.planner = MagicMock()
		self.planner.modules = [self.goal_objective]

		# Add create_visualization_publisher mock
		self.goal_objective.create_visualization_publisher = MagicMock()

	@patch('utils.utils.LOG_DEBUG')
	def test_planner_integration(self, mock_log_debug):
		"""Test if module properly interacts with planner"""
		# Setup mocks for planner's solve_mpc method
		data = MagicMock()
		state = MagicMock()
		module_data = MagicMock()

		# Setup data ready mocks
		with patch.object(self.goal_objective, 'is_data_ready', return_value=True), \
				patch.object(self.goal_objective, 'update') as mock_update, \
				patch.object(self.goal_objective, 'set_parameters') as mock_set_params:

			# Mock planner.solve_mpc similar to the actual implementation
			# Update modules
			for module in self.planner.modules:
				module.update(state, data, module_data)

			# Set parameters for each prediction step
			for k in range(self.solver.horizon):
				for module in self.planner.modules:
					module.set_parameters(data, module_data, k)

			# Assertions
			mock_update.assert_called_once_with(state, data, module_data)

			# Module should have set_parameters called N times
			self.assertEqual(mock_set_params.call_count, self.solver.horizon)

if __name__ == '__main__':
	unittest.main()