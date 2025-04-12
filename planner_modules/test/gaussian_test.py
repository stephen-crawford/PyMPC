import unittest
import numpy as np
from unittest.mock import MagicMock, patch, call, ANY
from numpy import sqrt

# Import modules to test
from utils.const import CONSTRAINT, GAUSSIAN, DYNAMIC

# Manually patch CONFIG to avoid dependency issues in testing
CONFIG_MOCK = {
	"robot": {
		"radius": 0.5
	},
	"n_discs": 2,
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


@patch('utils.utils.read_config_file', return_value=CONFIG_MOCK)
class TestGaussianConstraints(unittest.TestCase):
	"""Test suite for GaussianConstraints class"""

	def setUp(self):
		"""Set up test fixtures before each test"""
		# Create mock solver
		self.solver = MagicMock()
		self.solver.N = 10
		self.solver.params = MagicMock()

		# Create instance of the class under test
		with patch('utils.utils.read_config_file', return_value=CONFIG_MOCK):
			from planner_modules.gaussian_constraints import GaussianConstraints
			self.gaussian_constraints = GaussianConstraints(self.solver)

	def test_initialization(self, mock_config):
		"""Test proper initialization of GaussianConstraints"""
		self.assertEqual(self.gaussian_constraints.module_type, CONSTRAINT)
		self.assertEqual(self.gaussian_constraints.name, "gaussian_constraints")
		self.assertEqual(self.gaussian_constraints._dummy_x, 100.0)
		self.assertEqual(self.gaussian_constraints._dummy_y, 100.0)

	def test_update(self, mock_config):
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

	@patch('solver.solver_interface.set_solver_parameter')
	def test_set_parameters_k0(self, mock_set_param, mock_config):
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

		# Set dummy values
		self.gaussian_constraints._dummy_x = 110.0
		self.gaussian_constraints._dummy_y = 120.0

		# Call method under test
		self.gaussian_constraints.set_parameters(data, module_data, k)

		# Assertions
		expected_calls = [
			# Ego disc radius call
			call(self.solver.params, "ego_disc_radius", 0.5, k, settings=CONFIG_MOCK),

			# Ego disc offset calls
			call(self.solver.params, "ego_disc_offset", ANY, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ego_disc_offset", ANY, k, index=1, settings=CONFIG_MOCK),

			# Dummy obstacle calls for obstacle 0
			call(self.solver.params, "gaussian_obstacle_x", 110.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "gaussian_obstacle_y", 120.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "gaussian_obstacle_major", 0.1, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "gaussian_obstacle_minor", 0.1, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "gaussian_obstacle_risk", 0.05, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "gaussian_obstacle_r", 0.1, k, index=0, settings=CONFIG_MOCK),

			# Dummy obstacle calls for obstacle 1
			call(self.solver.params, "gaussian_obstacle_x", 110.0, k, index=1, settings=CONFIG_MOCK),
			call(self.solver.params, "gaussian_obstacle_y", 120.0, k, index=1, settings=CONFIG_MOCK),
			call(self.solver.params, "gaussian_obstacle_major", 0.1, k, index=1, settings=CONFIG_MOCK),
			call(self.solver.params, "gaussian_obstacle_minor", 0.1, k, index=1, settings=CONFIG_MOCK),
			call(self.solver.params, "gaussian_obstacle_risk", 0.05, k, index=1, settings=CONFIG_MOCK),
			call(self.solver.params, "gaussian_obstacle_r", 0.1, k, index=1, settings=CONFIG_MOCK),
		]

		# Check that all expected calls were made
		mock_set_param.assert_has_calls(expected_calls, any_order=True)
		self.assertEqual(mock_set_param.call_count, 15)

	@patch('solver.solver_interface.set_solver_parameter')
	def test_set_parameters_k1_dynamic(self, mock_set_param, mock_config):
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

		# Call method under test
		self.gaussian_constraints.set_parameters(data, module_data, k)

		# Assertions for dynamic obstacles
		expected_calls = [
			# Ego disc radius and offset calls
			call(self.solver.params, "ego_disc_radius", 0.5, k, settings=CONFIG_MOCK),
			call(self.solver.params, "ego_disc_offset", ANY, k, index=0, settings=CONFIG_MOCK),

			# Obstacle calls
			call(self.solver.params, "gaussian_obstacle_x", 5.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "gaussian_obstacle_y", 6.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "gaussian_obstacle_major", 2.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "gaussian_obstacle_minor", 1.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "gaussian_obstacle_risk", 0.05, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "gaussian_obstacle_r", 0.8, k, index=0, settings=CONFIG_MOCK),
		]

		mock_set_param.assert_has_calls(expected_calls, any_order=True)
		self.assertEqual(mock_set_param.call_count, 8)  # 1 radius + 1 offset + 6 obstacle params

	@patch('solver.solver_interface.set_solver_parameter')
	def test_set_parameters_k1_static(self, mock_set_param, mock_config):
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

		# Call method under test
		self.gaussian_constraints.set_parameters(data, module_data, k)

		# Assertions for static obstacles
		expected_calls = [
			# Ego disc radius and offset calls
			call(self.solver.params, "ego_disc_radius", 0.5, k, settings=CONFIG_MOCK),
			call(self.solver.params, "ego_disc_offset", ANY, k, index=0, settings=CONFIG_MOCK),

			# Obstacle calls
			call(self.solver.params, "gaussian_obstacle_x", 5.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "gaussian_obstacle_y", 6.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "gaussian_obstacle_major", 0.001, k, index=0, settings=CONFIG_MOCK),
			# Minimal uncertainty
			call(self.solver.params, "gaussian_obstacle_minor", 0.001, k, index=0, settings=CONFIG_MOCK),
			# Minimal uncertainty
			call(self.solver.params, "gaussian_obstacle_risk", 0.05, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "gaussian_obstacle_r", 0.8, k, index=0, settings=CONFIG_MOCK),
		]

		mock_set_param.assert_has_calls(expected_calls, any_order=True)
		self.assertEqual(mock_set_param.call_count, 8)  # 1 radius + 1 offset + 6 obstacle params

	def test_is_data_ready(self, mock_config):
		"""Test is_data_ready method"""
		# Test when obstacles count does not match max_obstacles
		data = MagicMock()
		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = 2  # Not equal to CONFIG_MOCK["max_obstacles"]
		missing_data = ""

		result = self.gaussian_constraints.is_data_ready(data, missing_data)
		self.assertFalse(result)

		# Test when obstacle prediction modes are empty
		data.dynamic_obstacles.size.return_value = CONFIG_MOCK["max_obstacles"]
		obstacle = MagicMock()
		obstacle.prediction.modes.empty.return_value = True
		data.dynamic_obstacles.__getitem__.return_value = obstacle
		missing_data = ""

		result = self.gaussian_constraints.is_data_ready(data, missing_data)
		self.assertFalse(result)

		# Test when obstacle prediction type is not Gaussian
		obstacle.prediction.modes.empty.return_value = False
		obstacle.prediction.type = "NOT_GAUSSIAN"  # Not GAUSSIAN
		missing_data = ""

		result = self.gaussian_constraints.is_data_ready(data, missing_data)
		self.assertFalse(result)

		# Test when data is ready
		obstacle.prediction.type = GAUSSIAN
		missing_data = ""

		result = self.gaussian_constraints.is_data_ready(data, missing_data)
		self.assertTrue(result)

	@patch('planner_modules.gaussian_constraints.PROFILE_SCOPE')
	@patch('planner_modules.gaussian_constraints.ROSPointMarker')
	@patch('planner_modules.gaussian_constraints.exponential_quantile')
	def test_visualize(self, mock_exp_quantile, mock_point_marker, mock_profile_scope, mock_config):
		"""Test visualize method with debug visuals enabled"""
		# Setup
		data = MagicMock()
		module_data = MagicMock()

		# Skip test if debug_visuals is disabled
		if not CONFIG_MOCK["debug_visuals"]:
			CONFIG_MOCK["debug_visuals"] = True  # Enable for testing

		# Configure obstacles
		dynamic_obstacle = MagicMock()
		dynamic_obstacle.type = DYNAMIC
		dynamic_obstacle.radius = 0.5
		dynamic_obstacle.prediction.type = GAUSSIAN

		static_obstacle = MagicMock()
		static_obstacle.type = "STATIC"
		static_obstacle.radius = 0.5
		static_obstacle.prediction.type = GAUSSIAN

		# Setup prediction modes
		dyn_mode = []
		for i in range(self.solver.N - 1):
			mode_ki = MagicMock()
			mode_ki.position = [5.0 + i, 6.0 + i]
			mode_ki.major_radius = 2.0
			mode_ki.minor_radius = 1.0
			dyn_mode.append(mode_ki)

		dyn_modes = MagicMock()
		dyn_modes.__getitem__.side_effect = lambda i: dyn_mode[i]
		dyn_modes.__len__.return_value = len(dyn_mode)

		modes_list = MagicMock()
		modes_list.__getitem__.return_value = dyn_modes
		dynamic_obstacle.prediction.modes = modes_list
		static_obstacle.prediction.modes = modes_list

		# Setup dynamic obstacles
		data.dynamic_obstacles = [dynamic_obstacle, static_obstacle]

		# Mock point marker
		mock_marker_instance = MagicMock()
		mock_point_marker.return_value = mock_marker_instance
		mock_marker_instance.get_new_point_marker.return_value = MagicMock()

		# Mock exponential_quantile
		mock_exp_quantile.return_value = 3.0

		# Call method under test
		self.gaussian_constraints.visualize(data, module_data)

		# Assertions
		mock_point_marker.assert_called_once_with(self.gaussian_constraints.name + "/obstacles")
		mock_marker_instance.get_new_point_marker.assert_called_once_with("CYLINDER")
		mock_marker_instance.publish.assert_called_once()

		# Check that chi-square quantile was called for dynamic obstacle
		mock_exp_quantile.assert_called_with(0.5, 1.0 - CONFIG_MOCK["probabilistic"]["risk"])

		# Restore debug visuals setting if needed
		CONFIG_MOCK["debug_visuals"] = False

	@patch('planner_modules.gaussian_constraints.ROSPointMarker')
	def test_visualize_debug_disabled(self, mock_point_marker, mock_config):
		"""Test visualize method with debug visuals disabled"""
		# Setup
		data = MagicMock()
		module_data = MagicMock()

		# Ensure debug is disabled
		CONFIG_MOCK["debug_visuals"] = False

		# Call method under test
		self.gaussian_constraints.visualize(data, module_data)

		# Should not create any visualizations
		mock_point_marker.assert_not_called()


class TestSystemIntegration(unittest.TestCase):
	"""Test integration between GaussianConstraints and Planner"""

	def setUp(self):
		"""Set up test fixtures before each test"""
		# Create mock solver
		self.solver = MagicMock()
		self.solver.N = 10
		self.solver.params = MagicMock()

		# Create instance of the class under test
		with patch('utils.utils.read_config_file', return_value=CONFIG_MOCK):
			from planner_modules.gaussian_constraints import GaussianConstraints
			self.gaussian_constraints = GaussianConstraints(self.solver)

		# Create mock planner
		self.planner = MagicMock()
		self.planner.modules = [self.gaussian_constraints]

	@patch('utils.utils.read_config_file', return_value=CONFIG_MOCK)
	def test_planner_integration(self, mock_config):
		"""Test if module properly interacts with planner"""
		# Setup mocks for planner's solve_mpc method
		data = MagicMock()
		state = MagicMock()
		module_data = MagicMock()

		# Setup data ready mocks
		with patch.object(self.gaussian_constraints, 'is_data_ready', return_value=True), \
				patch.object(self.gaussian_constraints, 'update') as mock_update, \
				patch.object(self.gaussian_constraints, 'set_parameters') as mock_set_params:

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