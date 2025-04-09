import unittest
import numpy as np
from unittest.mock import MagicMock, patch, call

# Import modules to test
from utils.const import CONSTRAINT, GAUSSIAN, DETERMINISTIC

# Manually patch CONFIG to avoid dependency issues in testing
CONFIG_MOCK = {
	"contouring": {
		"num_segments": 10
	},
	"n_discs": 2,
	"robot": {
		"radius": 0.5
	},
	"probabilistic": {
		"risk": 0.05
	},
	"max_obstacles": 3,
	"debug_visuals": False
}


@patch('utils.utils.read_config_file', return_value=CONFIG_MOCK)
class TestEllipsoidConstraints(unittest.TestCase):
	"""Test suite for EllipsoidConstraints class"""

	def setUp(self):
		"""Set up test fixtures before each test"""
		# Create mock solver
		self.solver = MagicMock()
		self.solver.N = 10
		self.solver.params = MagicMock()
		self.solver.dt = 0.1

		# Create instance of the class under test
		with patch('utils.utils.read_config_file', return_value=CONFIG_MOCK):
			from planner_modules.ellipsoid_constraints import EllipsoidConstraints
			self.ellipsoid_constraints = EllipsoidConstraints(self.solver)

	def test_initialization(self, mock_config):
		"""Test proper initialization of EllipsoidConstraints"""
		self.assertEqual(self.ellipsoid_constraints.module_type, CONSTRAINT)
		self.assertEqual(self.ellipsoid_constraints.name, "ellipsoid_constraints")
		self.assertEqual(self.ellipsoid_constraints.num_segments, CONFIG_MOCK["contouring"]["num_segments"])
		self.assertEqual(self.ellipsoid_constraints.n_discs, CONFIG_MOCK["n_discs"])
		self.assertEqual(self.ellipsoid_constraints._robot_radius, CONFIG_MOCK["robot"]["radius"])
		self.assertEqual(self.ellipsoid_constraints.risk, CONFIG_MOCK["probabilistic"]["risk"])
		self.assertEqual(self.ellipsoid_constraints._dummy_x, 50.0)
		self.assertEqual(self.ellipsoid_constraints._dummy_y, 50.0)

	def test_update(self, mock_config):
		"""Test update method with valid data"""
		# Setup
		state = MagicMock()
		state.get.side_effect = lambda key: 10.0 if key == "x" else 20.0 if key == "y" else 0.0

		data = MagicMock()
		module_data = MagicMock()

		# Call method under test
		self.ellipsoid_constraints.update(state, data, module_data)

		# Assertions
		self.assertEqual(self.ellipsoid_constraints._dummy_x, 60.0)  # 10.0 + 50.0
		self.assertEqual(self.ellipsoid_constraints._dummy_y, 70.0)  # 20.0 + 50.0

	@patch('planner_modules.ellipsoid_constraints.set_solver_parameter')
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

		# Call method under test
		self.ellipsoid_constraints.set_parameters(data, module_data, k)

		# Assertions
		expected_calls = [
			# Ego disc radius calls
			call(self.solver.params, "ego_disc_radius", 0.5, k, settings=CONFIG_MOCK),
			call(self.solver.params, "ego_disc_radius", 0.5, k, settings=CONFIG_MOCK),

			# Ego disc offset calls
			call(self.solver.params, "ego_disc_offset", np.array([0.5, 0.3]), k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ego_disc_offset", np.array([1.0, 0.0]), k, index=1, settings=CONFIG_MOCK),

			# Dummy obstacle calls for obstacle 0
			call(self.solver.params, "ellipsoid_obst_x", 50.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_y", 50.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_psi", 0.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_r", 0.1, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_major", 0.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_minor", 0.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_chi", 1.0, k, index=0, settings=CONFIG_MOCK),

			# Dummy obstacle calls for obstacle 1
			call(self.solver.params, "ellipsoid_obst_x", 50.0, k, index=1, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_y", 50.0, k, index=1, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_psi", 0.0, k, index=1, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_r", 0.1, k, index=1, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_major", 0.0, k, index=1, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_minor", 0.0, k, index=1, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_chi", 1.0, k, index=1, settings=CONFIG_MOCK),
		]

		# Check that all expected calls were made
		mock_set_param.assert_has_calls(expected_calls, any_order=True)
		self.assertEqual(mock_set_param.call_count, 17)  # 2 radius + 2 offset + 7 params * 2 obstacles

	@patch('planner_modules.ellipsoid_constraints.set_solver_parameter')
	@patch('utils.utils.exponential_quantile')
	def test_set_parameters_k1_deterministic(self, mock_exp_quantile, mock_set_param, mock_config):
		"""Test set_parameters method for k=1 with deterministic obstacles"""
		# Setup
		k = 1
		data = MagicMock()
		data.robot_area = [MagicMock()]
		data.robot_area[0].offset = np.array([0.5, 0.3])

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

		# Call method under test
		self.ellipsoid_constraints.set_parameters(data, module_data, k)

		# Assertions for deterministic obstacles
		expected_calls = [
			# Ego disc radius and offset calls
			call(self.solver.params, "ego_disc_radius", 0.5, k, settings=CONFIG_MOCK),
			call(self.solver.params, "ego_disc_offset", np.array([0.5, 0.3]), k, index=0, settings=CONFIG_MOCK),

			# Obstacle calls
			call(self.solver.params, "ellipsoid_obst_x", 5.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_y", 6.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_psi", 0.5, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_r", 1.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_major", 0.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_minor", 0.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_chi", 1.0, k, index=0, settings=CONFIG_MOCK),
		]

		mock_set_param.assert_has_calls(expected_calls, any_order=True)
		self.assertEqual(mock_set_param.call_count, 9)  # 1 radius + 1 offset + 7 obstacle params
		# Ensure exponential_quantile was not called for deterministic obstacles
		mock_exp_quantile.assert_not_called()

	@patch('planner_modules.ellipsoid_constraints.set_solver_parameter')
	@patch('utils.utils.exponential_quantile')
	def test_set_parameters_k1_gaussian(self, mock_exp_quantile, mock_set_param, mock_config):
		"""Test set_parameters method for k=1 with Gaussian obstacles"""
		# Setup
		k = 1
		data = MagicMock()
		data.robot_area = [MagicMock()]
		data.robot_area[0].offset = np.array([0.5, 0.3])

		# Setup dynamic obstacles with Gaussian prediction
		obstacle = MagicMock()
		obstacle.radius = 1.0
		obstacle.prediction.type = GAUSSIAN

		# Setup prediction mode
		mode = MagicMock()
		mode_k0 = MagicMock()
		mode_k0.position = [5.0, 6.0]
		mode_k0.angle = 0.5
		mode_k0.major_radius = 2.0
		mode_k0.minor_radius = 1.0
		mode.__getitem__.return_value = mode_k0

		obstacle.prediction.modes = [mode]

		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = 1
		data.dynamic_obstacles.__getitem__.return_value = obstacle

		module_data = MagicMock()

		# Mock exponential_quantile to return a constant value
		mock_exp_quantile.return_value = 3.0

		# Call method under test
		self.ellipsoid_constraints.set_parameters(data, module_data, k)

		# Assertions for Gaussian obstacles
		expected_calls = [
			# Ego disc radius and offset calls
			call(self.solver.params, "ego_disc_radius", 0.5, k, settings=CONFIG_MOCK),
			call(self.solver.params, "ego_disc_offset", np.array([0.5, 0.3]), k, index=0, settings=CONFIG_MOCK),

			# Obstacle calls
			call(self.solver.params, "ellipsoid_obst_x", 5.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_y", 6.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_psi", 0.5, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_r", 1.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_major", 2.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_minor", 1.0, k, index=0, settings=CONFIG_MOCK),
			call(self.solver.params, "ellipsoid_obst_chi", 3.0, k, index=0, settings=CONFIG_MOCK),
		]

		mock_set_param.assert_has_calls(expected_calls, any_order=True)
		self.assertEqual(mock_set_param.call_count, 9)  # 1 radius + 1 offset + 7 obstacle params

		# Ensure exponential_quantile was called with correct parameters
		mock_exp_quantile.assert_called_once_with(0.5, 1.0 - CONFIG_MOCK["probabilistic"]["risk"])

	def test_is_data_ready(self, mock_config):
		"""Test is_data_ready method"""
		# Test when robot area is missing
		data = MagicMock()
		data.robot_area = []
		missing_data = ""

		result = self.ellipsoid_constraints.is_data_ready(data, missing_data)
		self.assertFalse(result)
		self.assertEqual(missing_data, "Robot area ")

		# Test when obstacles count does not match max_obstacles
		data.robot_area = [MagicMock()]
		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = 2  # Not equal to CONFIG_MOCK["max_obstacles"]
		missing_data = ""

		result = self.ellipsoid_constraints.is_data_ready(data, missing_data)
		self.assertFalse(result)
		self.assertEqual(missing_data, "Obstacles ")

		# Test when obstacle prediction is empty
		data.dynamic_obstacles.size.return_value = CONFIG_MOCK["max_obstacles"]
		obstacle = MagicMock()
		obstacle.prediction.empty.return_value = True
		data.dynamic_obstacles.__getitem__.return_value = obstacle
		missing_data = ""

		result = self.ellipsoid_constraints.is_data_ready(data, missing_data)
		self.assertFalse(result)
		self.assertEqual(missing_data, "Obstacle Prediction ")

		# Test when obstacle prediction type is incorrect
		obstacle.prediction.empty.return_value = False
		obstacle.prediction.type = "INVALID_TYPE"  # Not GAUSSIAN or DETERMINISTIC
		missing_data = ""

		result = self.ellipsoid_constraints.is_data_ready(data, missing_data)
		self.assertFalse(result)
		self.assertEqual(missing_data, "Obstacle Prediction (Type is incorrect) ")

		# Test when data is ready
		obstacle.prediction.type = GAUSSIAN
		missing_data = ""

		result = self.ellipsoid_constraints.is_data_ready(data, missing_data)
		self.assertTrue(result)

	@patch('planner_modules.ellipsoid_constraints.ROSLine')
	@patch('planner_modules.ellipsoid_constraints.ROSPointMarker')
	def test_visualize_debug_disabled(self, mock_point_marker, mock_line, mock_config):
		"""Test visualize method with debug visuals disabled"""
		# Setup
		data = MagicMock()
		module_data = MagicMock()

		# Call method under test with debug disabled
		self.ellipsoid_constraints.visualize(data, module_data)

		# Should not create any visualizations
		mock_line.assert_not_called()
		mock_point_marker.assert_not_called()

	@patch('planner_modules.ellipsoid_constraints.ROSLine')
	def test_visualize_debug_enabled(self, mock_line, mock_config):
		"""Test visualize method with debug visuals enabled"""
		# Setup
		data = MagicMock()
		module_data = MagicMock()

		# Configure obstacles
		deterministic_obstacle = MagicMock()
		deterministic_obstacle.radius = 1.0
		deterministic_obstacle.prediction.type = DETERMINISTIC

		gaussian_obstacle = MagicMock()
		gaussian_obstacle.radius = 1.0
		gaussian_obstacle.prediction.type = GAUSSIAN

		# Setup prediction modes
		det_mode = MagicMock()
		det_mode_k0 = MagicMock()
		det_mode_k0.position = [5.0, 6.0]
		det_mode_k0.angle = 0.5
		det_mode.__getitem__.return_value = det_mode_k0
		deterministic_obstacle.prediction.modes = [det_mode]

		gauss_mode = MagicMock()
		gauss_mode_k0 = MagicMock()
		gauss_mode_k0.position = [7.0, 8.0]
		gauss_mode_k0.angle = 0.7
		gauss_mode_k0.major_radius = 2.0
		gauss_mode_k0.minor_radius = 1.0
		gauss_mode.__getitem__.return_value = gauss_mode_k0
		gaussian_obstacle.prediction.modes = [gauss_mode]

		# Setup dynamic obstacles
		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = 2
		data.dynamic_obstacles.__getitem__.side_effect = [deterministic_obstacle, gaussian_obstacle]

		# Mock line instance and methods
		mock_line_instance = MagicMock()
		mock_line.return_value = mock_line_instance
		mock_line_instance.add_new_line.return_value = MagicMock()

		# Enable debug visuals
		CONFIG_MOCK["debug_visuals"] = True

		# Patch exponential_quantile
		with patch('utils.utils.exponential_quantile', return_value=3.0):
			# Call method under test
			self.ellipsoid_constraints.visualize(data, module_data)

		# Assertions
		mock_line.assert_called_once_with(self.ellipsoid_constraints.name + "/ellipsoids")
		mock_line_instance.publish.assert_called_once()

		# Should add lines for both obstacles (deterministic and gaussian)
		self.assertEqual(mock_line_instance.add_new_line.call_count, 2)

		# Restore debug visuals setting
		CONFIG_MOCK["debug_visuals"] = False

	def test_helper_draw_circle(self, mock_config):
		"""Test _draw_circle helper method"""
		# Setup
		line = MagicMock()
		center = np.array([1.0, 2.0])
		radius = 3.0

		# Call method under test
		self.ellipsoid_constraints._draw_circle(line, center, radius, num_points=4)

		# Should add lines to connect the points (4 points = 4 lines to close the loop)
		self.assertEqual(line.add_line.call_count, 4)

	def test_helper_draw_ellipse(self, mock_config):
		"""Test _draw_ellipse helper method"""
		# Setup
		line = MagicMock()
		center = np.array([1.0, 2.0])
		angle = 0.5
		a = 3.0  # major axis
		b = 2.0  # minor axis

		# Call method under test
		self.ellipsoid_constraints._draw_ellipse(line, center, angle, a, b, num_points=4)

		# Should add lines to connect the points (4 points = 4 lines to close the loop)
		self.assertEqual(line.add_line.call_count, 4)


class TestSystemIntegration(unittest.TestCase):
	"""Test integration between EllipsoidConstraints and Planner"""

	def setUp(self):
		"""Set up test fixtures before each test"""
		# Create mock solver
		self.solver = MagicMock()
		self.solver.N = 10
		self.solver.params = MagicMock()

		# Create instance of the class under test
		with patch('utils.utils.read_config_file', return_value=CONFIG_MOCK):
			from planner_modules.ellipsoid_constraints import EllipsoidConstraints
			self.ellipsoid_constraints = EllipsoidConstraints(self.solver)

		# Create mock planner
		self.planner = MagicMock()
		self.planner._modules = [self.ellipsoid_constraints]

	@patch('utils.utils.read_config_file', return_value=CONFIG_MOCK)
	def test_planner_integration(self, mock_config):
		"""Test if module properly interacts with planner"""
		# Setup mocks for planner's solve_mpc method
		data = MagicMock()
		state = MagicMock()
		module_data = MagicMock()

		# Setup data ready mocks
		with patch.object(self.ellipsoid_constraints, 'is_data_ready', return_value=True), \
				patch.object(self.ellipsoid_constraints, 'update') as mock_update, \
				patch.object(self.ellipsoid_constraints, 'set_parameters') as mock_set_params:

			# Mock planner.solve_mpc similar to the actual implementation
			# Update modules
			for module in self.planner._modules:
				module.update(state, data, module_data)

			# Set parameters for each prediction step
			for k in range(self.solver.N):
				for module in self.planner._modules:
					module.set_parameters(data, module_data, k)

			# Assertions
			mock_update.assert_called_once_with(state, data, module_data)

			# Module should have set_parameters called N times
			self.assertEqual(mock_set_params.call_count, self.solver.N)


if __name__ == '__main__':
	unittest.main()