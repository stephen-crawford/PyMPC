import unittest
import numpy as np
from unittest.mock import MagicMock, patch, call, ANY

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planner_modules.src.constraints.gaussian_constraints import GaussianConstraints
from planning.src.types import Data
# Import modules to test
from utils.const import CONSTRAINT, GAUSSIAN, DYNAMIC

# Manually patch CONFIG to avoid dependency issues in testing
CONFIG_MOCK = {
	"robot": {
		"radius": 0.5
	},
	"num_discs": 2,
	"max_obstacles": 3,
	"obstacle_radius": 0.8,
	"probabilistic": {
		"risk": 0.05
	},
	"visualization": {
		"draw_every": 2
	},
	"debug_visuals": False
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

class TestGaussianConstraints(unittest.TestCase):
	"""Test suite for GaussianConstraints class"""

	def setUp(self):
		"""Set up test fixtures before each test"""
		# Create mock solver
		self.solver = MagicMock()
		self.solver.N = 10
		self.solver.params = MagicMock()

		# Apply the patch before creating the class
		patcher = patch('planner_modules.src.constraints.base_constraint.BaseConstraint.get_config_value',
						side_effect=get_mocked_config)
		self.mock_get_config = patcher.start()
		self.addCleanup(patcher.stop)

		self.gaussian_constraints = GaussianConstraints(self.solver)

	def test_initialization(self):
		"""Test proper initialization of GaussianConstraints"""
		self.assertEqual(self.gaussian_constraints.module_type, CONSTRAINT)
		self.assertEqual(self.gaussian_constraints.name, "gaussian_constraints")
		self.assertEqual(self.gaussian_constraints._dummy_x, 100.0)
		self.assertEqual(self.gaussian_constraints._dummy_y, 100.0)

	def test_update(self):
		"""Test update method with valid data"""
		# Setup
		state = MagicMock()
		state.get.side_effect = lambda key: 10.0 if key == "x" else 20.0 if key == "y" else 0.0

		data = MagicMock()
		module_data = MagicMock()

		# Call method under test
		self.gaussian_constraints.update(state, data, module_data)

		# Assertions
		self.assertEqual(self.gaussian_constraints._dummy_x, 110.0)  # 10.0 + 100.0
		self.assertEqual(self.gaussian_constraints._dummy_y, 120.0)  # 20.0 + 100.0

	def test_set_parameters_k0(self):
		"""Test set_parameters method for k=0 (dummies)"""
		# Setup
		data = MagicMock()
		data.robot_area = [MagicMock(), MagicMock()]
		data.robot_area[0].offset = np.array([0.5, 0.3])
		data.robot_area[1].offset = np.array([1.0, 0.0])

		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = 2

		module_data = MagicMock()

		# Set dummy values
		self.gaussian_constraints._dummy_x = 110.0
		self.gaussian_constraints._dummy_y = 120.0

		self.gaussian_constraints.set_parameters(self.solver.params, data, module_data, 0)
		assert (self.solver.params.set_parameter.call_count == 14)

	def test_set_parameters_k1_dynamic(self):
		"""Test set_parameters method for k=1 with dynamic gaussian obstacles"""
		# Setup
		k = 1
		data = MagicMock()
		data.robot_area = [MagicMock()]
		data.robot_area[0].offset = np.array([0.5, 0.3])

		# Setup dynamic obstacles with Gaussian prediction
		obstacle = MagicMock()
		obstacle.type = DYNAMIC
		obstacle.prediction.type = GAUSSIAN

		# Setup prediction mode
		mode = MagicMock()
		mode_k0 = MagicMock()
		mode_k0.position = [5.0, 6.0]
		mode_k0.major_radius = 2.0
		mode_k0.minor_radius = 1.0
		mode.__getitem__.return_value = mode_k0

		modes_list = MagicMock()
		modes_list.__getitem__.return_value = mode
		obstacle.prediction.modes = modes_list

		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = 1
		data.dynamic_obstacles.__getitem__.return_value = obstacle

		module_data = MagicMock()
		self.gaussian_constraints.set_parameters(self.solver.params, data, module_data, 1)
		assert (self.solver.params.set_parameter.call_count == 9)


	def test_set_parameters_k1_static(self):
		"""Test set_parameters method for k=1 with static gaussian obstacles"""
		# Setup
		k = 1
		data = MagicMock()
		data.robot_area = [MagicMock()]
		data.robot_area[0].offset = np.array([0.5, 0.3])

		# Setup static obstacle with Gaussian prediction
		obstacle = MagicMock()
		obstacle.type = "STATIC"  # Not DYNAMIC
		obstacle.prediction.type = GAUSSIAN

		# Setup prediction mode
		mode = MagicMock()
		mode_k0 = MagicMock()
		mode_k0.position = [5.0, 6.0]
		mode_k0.major_radius = 2.0  # These values should be ignored for static obstacles
		mode_k0.minor_radius = 1.0  # These values should be ignored for static obstacles
		mode.__getitem__.return_value = mode_k0

		modes_list = MagicMock()
		modes_list.__getitem__.return_value = mode
		obstacle.prediction.modes = modes_list

		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = 1
		data.dynamic_obstacles.__getitem__.return_value = obstacle

		module_data = MagicMock()

		self.gaussian_constraints.set_parameters(self.solver.params, data, module_data, 9)

	def test_is_data_ready(self):
		"""Test is_data_ready method"""
		# Test when obstacles count does not match max_obstacles
		data = Data()
		obst = MagicMock()
		obsts = [obst, obst]
		data.set("dynamic_obstacles", obsts)
		missing_data = ""

		result = self.gaussian_constraints.is_data_ready(data)
		self.assertFalse(result)

		# Test when obstacle prediction modes are empty
		data.set("dynamic_obstacles", [obst, obst, obst])
		obst.prediction.modes.empty.return_value = True
		missing_data = ""

		result = self.gaussian_constraints.is_data_ready(data)
		self.assertFalse(result)

		# Test when obstacle prediction type is not Gaussian
		obst.prediction.modes.empty.return_value = False
		obst.prediction.type = "NOT_GAUSSIAN"  # Not GAUSSIAN
		missing_data = ""

		result = self.gaussian_constraints.is_data_ready(data)
		self.assertFalse(result)

		# Test when data is ready
		obst.prediction.type = GAUSSIAN
		missing_data = ""

		result = self.gaussian_constraints.is_data_ready(data)
		self.assertTrue(result)

class TestSystemIntegration(unittest.TestCase):
	"""Test integration between GaussianConstraints and Planner"""

	def setUp(self):
		"""Set up test fixtures before each test"""
		# Create mock solver
		self.solver = MagicMock()
		self.solver.horizon = 10
		self.solver.params = MagicMock()

		# Patch get_config_value to use our CONFIG_MOCK
		with patch.object(BaseConstraint, 'get_config_value') as mock_get_config_value:
			def get_mocked_config(key, default=None):
				keys = key.split('.')
				cfg = CONFIG_MOCK

				try:
					for k in keys:
						cfg = cfg[k]
					return cfg
				except (KeyError, TypeError):
					return default

			mock_get_config_value.side_effect = get_mocked_config

			# Create instance of the class under test
			from planner_modules.src.constraints.gaussian_constraints import GaussianConstraints
			self.gaussian_constraints = GaussianConstraints(self.solver)

		# Create mock planning
		self.planner = MagicMock()
		self.planner.modules = [self.gaussian_constraints]

		# Add create_visualization_publisher mock
		self.gaussian_constraints.create_visualization_publisher = MagicMock()

	@patch('utils.utils.LOG_DEBUG')
	def test_planner_integration(self, mock_log_debug):
		"""Test if module properly interacts with planning"""
		# Setup mocks for planning's solve_mpc method
		data = MagicMock()
		state = MagicMock()
		module_data = MagicMock()

		# Setup data ready mocks
		with patch.object(self.gaussian_constraints, 'is_data_ready', return_value=True), \
				patch.object(self.gaussian_constraints, 'update') as mock_update, \
				patch.object(self.gaussian_constraints, 'set_parameters') as mock_set_params:

			# Mock planning.solve_mpc similar to the actual implementation
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