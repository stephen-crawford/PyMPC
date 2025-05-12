import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planner_modules.src.constraints.decomp_constraints import DecompConstraints
# Import modules to test
from utils.const import CONSTRAINT

# Manually patch CONFIG to avoid dependency issues in testing
CONFIG_MOCK = {
	"params": MagicMock(),
	"contouring": {
		"num_segments": 10
	},
	"decomp": {
		"range": 10.0,
		"max_constraints": 8
	},
	"road": {
		"width": 3.5,
		"two_way": False
	},
	"robot": {
		"width": 0.8
	},
	"N": 10,
	"n_discs": 1,
	"visualization": {
		"draw_every": 1
	},
	"debug_visuals": False
}


def get_mocked_config(key, default=None):
	keys = key.split('.')
	cfg = CONFIG_MOCK

	try:
		for k in keys:
			cfg = cfg[k]
		return cfg
	except (KeyError, TypeError):
		return default

class TestDecompConstraints(unittest.TestCase):
	"""Test suite for DecompConstraints class"""

	def setUp(self):
		"""Set up test fixtures before each test"""

		# Create mock solver
		self.solver = MagicMock()
		self.solver.horizon = 10
		self.solver.params = MagicMock()
		self.solver.dt = 0.1

		# Apply the patch before creating the class
		patcher = patch('planner_modules.src.constraints.base_constraint.BaseConstraint.get_config_value',
						side_effect=get_mocked_config)
		self.mock_get_config = patcher.start()
		self.addCleanup(patcher.stop)

		# Create mock for EllipsoidDecomp2D
		self.mock_decomp_util = MagicMock()

		# Patch the EllipsoidDecomp2D class
		self.ellip_patcher = patch('utils.math.EllipsoidDecomp2D', return_value=self.mock_decomp_util)
		self.mock_decomp_class = self.ellip_patcher.start()
		self.addCleanup(self.ellip_patcher.stop)

		self.decomp_constraints = DecompConstraints(self.solver)


	def test_initialization(self):
		"""Test proper initialization of DecompConstraints"""
		self.assertEqual(self.decomp_constraints.module_type, CONSTRAINT)
		self.assertEqual(self.decomp_constraints.name, "decomp_constraints")
		self.assertEqual(self.decomp_constraints.get_num_segments, CONFIG_MOCK["contouring"]["num_segments"])
		self.assertEqual(self.decomp_constraints.range, CONFIG_MOCK["decomp"]["range"])
		self.assertEqual(self.decomp_constraints._max_constraints, CONFIG_MOCK["decomp"]["max_constraints"])
		self.assertEqual(len(self.decomp_constraints.occ_pos), 0)

	@patch('planner_modules.src.constraints.decomp_constraints.PROFILE_SCOPE')
	def test_update(self, mock_profile_scope):
		"""Test update method with valid data"""
		# Setup
		state = MagicMock()
		state.get.return_value = 0.0  # Spline parameter

		data = MagicMock()
		module_data = MagicMock()
		module_data.path = MagicMock()
		module_data.path.get_point.return_value = np.array([1.0, 2.0])

		# Mock get_occupied_grid_cells and set_obs methods
		self.decomp_constraints.get_occupied_grid_cells = MagicMock(return_value=True)
		self.decomp_constraints.occ_pos = [np.array([3.0, 4.0]), np.array([5.0, 6.0])]

		# Create mock constraint
		mock_constraint = MagicMock()
		mock_constraint.A_ = np.array([[1.0, 2.0], [3.0, 4.0]])
		mock_constraint.b_ = np.array([5.0, 6.0])

		# Mock decomp_util methods
		self.mock_decomp_util.get_polyhedrons.return_value = [MagicMock()]
		self.mock_decomp_util.set_constraints.side_effect = lambda constraints, val: constraints.append(mock_constraint)

		self.decomp_constraints.decomp_util = self.mock_decomp_util

		# Predict velocity
		self.solver.get_ego_prediction.return_value = 1.0  # Velocity

		# Call method under test
		self.decomp_constraints.update(state, data, module_data)

		# Assertions
		self.decomp_constraints.get_occupied_grid_cells.assert_called_once_with(data)
		self.mock_decomp_util.set_obs.assert_called_once_with(self.decomp_constraints.occ_pos)
		self.mock_decomp_util.dilate.assert_called_once()
		self.mock_decomp_util.set_constraints.assert_called_once()
		self.mock_decomp_util.get_polyhedrons.assert_called_once()

		# Check constraint values were stored correctly
		self.assertEqual(self.decomp_constraints.a1[0][1][0], 1.0)
		self.assertEqual(self.decomp_constraints.a2[0][1][0], 2.0)
		self.assertEqual(self.decomp_constraints.b[0][1][0], 5.0)
		self.assertEqual(self.decomp_constraints.a1[0][1][1], 3.0)
		self.assertEqual(self.decomp_constraints.a2[0][1][1], 4.0)
		self.assertEqual(self.decomp_constraints.b[0][1][1], 6.0)

	def test_get_occupied_grid_cells(self):
		"""Test get_occupied_grid_cells method"""
		# Setup mock costmap
		costmap = MagicMock()
		costmap.get_size_in_cells_x.return_value = 3
		costmap.get_size_in_cells_y.return_value = 2

		# Define cost values
		def mock_get_cost(i, j):
			# Return non-zero cost only for specific cells
			if i == 1 and j == 1:
				return 100
			return 0

		costmap.getCost = mock_get_cost

		# Define map to world conversion
		costmap.map_to_world.return_value = (10.0, 20.0)

		data = MagicMock()
		data.costmap = costmap

		# Call method under test
		result = self.decomp_constraints.get_occupied_grid_cells(data)

		# Assertions
		self.assertTrue(result)
		self.assertEqual(len(self.decomp_constraints.occ_pos), 1)
		np.testing.assert_array_equal(self.decomp_constraints.occ_pos[0], np.array([10.0, 20.0]))

	def test_set_parameters_k0(self):
		"""Test set_parameters method with boundary data"""
		data = MagicMock()
		module_data = MagicMock()

		self.decomp_constraints.set_parameters(self.solver.params, data, module_data, 0)
		assert (self.solver.params.set_parameter.call_count == 3 * CONFIG_MOCK["decomp"]["max_constraints"])

	def test_set_parameters_k1(self):
		"""Test set_parameters method for k=1 (real constraints)"""
		# Setup
		k = 1
		data = MagicMock()
		data.robot_area = [MagicMock()]
		data.robot_area[0].offset = np.array([0.5, 0.3])
		module_data = MagicMock()

		# Set up some constraints
		self.decomp_constraints.a1[0][1][0] = 1.1
		self.decomp_constraints.a2[0][1][0] = 2.2
		self.decomp_constraints.b[0][1][0] = 3.3

		# Call method under test
		self.decomp_constraints.set_parameters(self.solver.params, data, module_data, 1)
		assert (self.solver.params.set_parameter.call_count == 1 + 3 * CONFIG_MOCK["decomp"]["max_constraints"])

	def test_is_data_ready(self):
		"""Test is_data_ready method"""
		# Test when data is not ready
		data = MagicMock()
		data.costmap = None
		missing_data = ""

		result = self.decomp_constraints.is_data_ready(data)
		self.assertFalse(result)

		# Test when data is ready
		data = ["costmap"]
		missing_data = ""

		result = self.decomp_constraints.is_data_ready(data)
		self.assertTrue(result)

class TestSystemIntegration(unittest.TestCase):
	"""Test integration between DecompConstraints and Planner"""

	@patch.object(BaseConstraint, 'get_config_value')
	def setUp(self, mock_get_config_value):
		"""Set up test fixtures before each test"""
		# Create mock solver and data
		self.solver = MagicMock()
		self.solver.horizon = 10
		self.solver.params = MagicMock()
		self.solver.timestep = 0.1

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

		# Create mock for EllipsoidDecomp2D
		self.mock_decomp_util = MagicMock()

		# Patch the EllipsoidDecomp2D class
		self.patcher = patch('utils.math.EllipsoidDecomp2D', return_value=self.mock_decomp_util)
		self.mock_decomp_class = self.patcher.start()

		# Create instance of the class under test
		self.decomp_constraints = DecompConstraints(self.solver)

		# Create mock planning
		self.planner = MagicMock()
		self.planner.modules = [self.decomp_constraints]

	def tearDown(self):
		self.patcher.stop()

	def test_planner_integration(self):
		"""Test if module properly interacts with planning"""
		# Setup mocks for planning's solve_mpc method
		data = MagicMock()
		state = MagicMock()
		module_data = MagicMock()

		# Setup data ready mocks
		with patch.object(self.decomp_constraints, 'is_data_ready', return_value=True), \
				patch.object(self.decomp_constraints, 'update') as mock_update, \
				patch.object(self.decomp_constraints, 'set_parameters') as mock_set_params:

			# Mock planning.solve_mpc similar to the actual implementation
			# Update modules
			for module in self.planner.modules:
				module.update(state, data, module_data)

			# Set parameters for each prediction step
			for k in range(self.solver.horizon):
				for module in self.planner.modules:
					module.set_parameters(self.solver.params, data, module_data, k)

			# Assertions
			mock_update.assert_called_once_with(state, data, module_data)

			# Module should have set_parameters called N times
			self.assertEqual(mock_set_params.call_count, self.solver.horizon)


if __name__ == '__main__':
	unittest.main()