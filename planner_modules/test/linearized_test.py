import unittest
import numpy as np
from unittest.mock import MagicMock, patch, call, ANY

from planner_modules.src.constraints.linearized_constraints import LinearizedConstraints
# Import modules to test
from utils.const import CONSTRAINT, DETERMINISTIC, GAUSSIAN

# Manually patch CONFIG to avoid dependency issues in testing
CONFIG_MOCK = {
	"num_discs": 2,
	"horizon": 10,
	"max_obstacles": 3,
	"linearized_constraints": {
		"add_halfspaces": 2,
		"max_constraints": 20
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


class TestLinearizedConstraints(unittest.TestCase):
	"""Test suite for LinearizedConstraints class"""

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

		self.linearized_constraints = LinearizedConstraints(self.solver)

	def test_initialization(self):
		"""Test proper initialization of LinearizedConstraints"""
		self.assertEqual(self.linearized_constraints.module_type, CONSTRAINT)
		self.assertEqual(self.linearized_constraints.name, "linearized_constraints")
		self.assertEqual(self.linearized_constraints.num_discs, CONFIG_MOCK["num_discs"])
		self.assertEqual(self.linearized_constraints.num_other_halfspaces,
		                 CONFIG_MOCK["linearized_constraints"]["add_halfspaces"])
		self.assertEqual(self.linearized_constraints.max_obstacles, CONFIG_MOCK["max_obstacles"])
		self.assertEqual(self.linearized_constraints.num_constraints,
		                 CONFIG_MOCK["max_obstacles"] + CONFIG_MOCK["linearized_constraints"]["add_halfspaces"])
		self.assertEqual(self.linearized_constraints._dummy_a1, 0.0)
		self.assertEqual(self.linearized_constraints._dummy_a2, 0.0)
		self.assertEqual(self.linearized_constraints._dummy_b, 100.0)

	def test_set_topology_constraints(self):
		"""Test setTopologyConstraints method"""
		# Call method under test
		self.linearized_constraints.set_topology_constraints()

		# Verify the changes
		self.assertEqual(self.linearized_constraints.num_discs, 1)
		self.assertTrue(self.linearized_constraints.use_guidance)

	def test_update(self):
		"""Test update method with valid data"""
		# Setup
		state = MagicMock()
		state.get.side_effect = lambda key: 10.0 if key == "x" else 0.0

		data = MagicMock()
		data.robot_area = [MagicMock(), MagicMock()]
		data.robot_area[0].get_position.return_value = np.array([1.0, 2.0])
		data.robot_area[1].get_position.return_value = np.array([3.0, 4.0])

		# Setup dynamic obstacles
		obstacle = MagicMock()
		obstacle.prediction.modes = [MagicMock()]
		obstacle.prediction.modes[0].dummy = 1  # Just to prevent attribute errors
		obstacle.radius = 0.5

		# Setup position predictions
		obstacle.prediction.modes[0].__getitem__.return_value = MagicMock()
		obstacle.prediction.modes[0][0].position = np.array([5.0, 6.0])

		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = 1
		data.dynamic_obstacles.empty.return_value = False
		data.dynamic_obstacles.__getitem__.return_value = obstacle

		module_data = MagicMock()
		module_data.static_obstacles = MagicMock()
		module_data.static_obstacles.empty.return_value = True

		# Mock solver's get_ego_prediction method
		self.solver.get_ego_prediction.side_effect = lambda k, param: 1.0 if param == "x" else 2.0 if param == "y" else 0.0

		# Mock douglas_rachford_projection method
		with patch.object(self.linearized_constraints, "project_to_safety") as mock_projection:
			# Call method under test
			self.linearized_constraints.update(state, data, module_data)

			# Assertions
			self.assertEqual(self.linearized_constraints._dummy_b, 110.0)  # 10.0 + 100.0

			# Check that douglas_rachford_projection was called for dynamic obstacles
			mock_projection.assert_called()

			# Check values were set correctly for a1, a2, b
			# For disc 0, k=1, obstacle 0 (normalized vector from robot to obstacle)
			# Position of obstacle is [5.0, 6.0], robot disc is at [1.0, 2.0]
			# Diff = [4.0, 4.0], dist = sqrt(32) = 5.66
			# a1 = 4.0/5.66, a2 = 4.0/5.66, b = a1*5.0 + a2*6.0 - (0.5 + 2)
			diff = np.array([4.0, 4.0])
			dist = np.linalg.norm(diff)
			expected_a1 = diff[0] / dist
			expected_a2 = diff[1] / dist
			expected_b = expected_a1 * 5.0 + expected_a2 * 6.0 - (0.5 + CONFIG_MOCK["num_discs"])

			# Allow small floating point differences
			self.assertAlmostEqual(self.linearized_constraints._a1[0][1][0], expected_a1, places=5)
			self.assertAlmostEqual(self.linearized_constraints._a2[0][1][0], expected_a2, places=5)
			self.assertAlmostEqual(self.linearized_constraints._b[0][1][0], expected_b, places=5)

	def test_update_with_static_obstacles(self):
		"""Test update method with static obstacles"""
		# Setup
		state = MagicMock()
		state.get.side_effect = lambda key: 10.0 if key == "x" else 0.0

		data = MagicMock()
		data.robot_area = [MagicMock(), MagicMock()]
		data.robot_area[0].get_position.return_value = np.array([1.0, 2.0])
		data.robot_area[1].get_position.return_value = np.array([3.0, 4.0])

		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = 0
		data.dynamic_obstacles.empty.return_value = True

		# Setup static obstacles for k=1
		static_obstacle = MagicMock()
		static_obstacle.A = np.array([0.7, 0.7])  # Normal vector
		static_obstacle.b = 5.0  # Offset

		static_obstacles_k1 = MagicMock()
		static_obstacles_k1.size.return_value = 1
		static_obstacles_k1.__getitem__.return_value = static_obstacle

		module_data = MagicMock()
		module_data.static_obstacles = MagicMock()
		module_data.static_obstacles.empty.return_value = False
		module_data.static_obstacles.__getitem__.return_value = static_obstacles_k1

		# Mock solver's get_ego_prediction method
		self.solver.get_ego_prediction.side_effect = lambda k, param: 1.0 if param == "x" else 2.0 if param == "y" else 0.0

		# Mock douglas_rachford_projection method
		with patch.object(self.linearized_constraints, 'project_to_safety') as mock_projection:
			# Call method under test
			self.linearized_constraints.update(state, data, module_data)

			# Check that static obstacle constraints were added
			self.assertEqual(self.linearized_constraints._a1[0][1][0], 0.7)
			self.assertEqual(self.linearized_constraints._a2[0][1][0], 0.7)
			self.assertEqual(self.linearized_constraints._b[0][1][0], 5.0)

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

		self.linearized_constraints.set_parameters(self.solver.params, data, module_data, 0)
		assert (self.solver.params.set_parameter.call_count == 3 * self.linearized_constraints.num_discs * (self.linearized_constraints.max_obstacles + self.linearized_constraints.num_other_halfspaces))

	def test_set_parameters_k1(self):
		"""Test set_parameters method for k=1 with constraints"""
		# Setup
		k = 1
		data = MagicMock()
		data.robot_area = [MagicMock(), MagicMock()]
		data.robot_area[0].offset = np.array([0.5, 0.3])
		data.robot_area[1].offset = np.array([1.0, 0.0])

		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = 1

		module_data = MagicMock()

		# Set constraint values for testing
		self.linearized_constraints._a1[0][k][0] = 0.7
		self.linearized_constraints._a2[0][k][0] = 0.7
		self.linearized_constraints._b[0][k][0] = 5.0

		self.linearized_constraints._a1[1][k][0] = 0.8
		self.linearized_constraints._a2[1][k][0] = 0.6
		self.linearized_constraints._b[1][k][0] = 4.0

		# Call method under test
		self.linearized_constraints.set_parameters(self.solver.params, data, module_data, 1)
		assert (self.solver.params.set_parameter.call_count == self.linearized_constraints.num_discs + 3 * self.linearized_constraints.num_discs * (
					self.linearized_constraints.max_obstacles + self.linearized_constraints.num_other_halfspaces))

	def test_is_data_ready(self):
		"""Test is_data_ready method"""
		# Test when obstacles count does not match max_obstacles
		data = MagicMock()
		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = 2  # Not equal to CONFIG_MOCK["max_obstacles"]
		missing_data = ""

		result = self.linearized_constraints.is_data_ready(data)
		self.assertFalse(result)

		# Test when obstacle prediction is empty
		data.dynamic_obstacles.size.return_value = CONFIG_MOCK["max_obstacles"]
		obstacle = MagicMock()
		obstacle.prediction.empty.return_value = True
		data.dynamic_obstacles.__getitem__.return_value = obstacle
		missing_data = ""

		result = self.linearized_constraints.is_data_ready(data)
		self.assertFalse(result)

		# Test when obstacle prediction type is not deterministic or gaussian
		obstacle.prediction.empty.return_value = False
		obstacle.prediction.type = "INVALID_TYPE"  # Not DETERMINISTIC or GAUSSIAN
		missing_data = ""

		result = self.linearized_constraints.is_data_ready(data)
		self.assertFalse(result)

		# Test when data is ready with deterministic prediction
		obstacle.prediction.type = DETERMINISTIC
		missing_data = ""

		result = self.linearized_constraints.is_data_ready(data)
		self.assertTrue(result)

		# Test when data is ready with gaussian prediction
		obstacle.prediction.type = GAUSSIAN
		missing_data = ""

		result = self.linearized_constraints.is_data_ready(data)
		self.assertTrue(result)

	def test_planner_integration(self):
		"""Test if module properly interacts with planner"""
		# Setup mocks for planner's solve_mpc method
		data = MagicMock()
		state = MagicMock()
		module_data = MagicMock()
		self.planner = MagicMock()
		self.planner.modules = [self.linearized_constraints]

		# Setup data ready mocks
		with patch.object(self.linearized_constraints, 'is_data_ready', return_value=True), \
				patch.object(self.linearized_constraints, 'update') as mock_update, \
				patch.object(self.linearized_constraints, 'set_parameters') as mock_set_params:

			# Mock planner.solve_mpc similar to the actual implementation
			# Update modules
			for module in self.planner.modules:
				module.update(state, data, module_data)

			# Set parameters for each prediction step
			for k in range(self.solver.N):
				for module in self.planner.modules:
					module.set_parameters(data, module_data, k)

			# Assertions
			mock_update.assert_called_once_with(state, data, module_data)

			# Module should have set_parameters called N times
			self.assertEqual(mock_set_params.call_count, self.solver.N)


if __name__ == '__main__':
	unittest.main()