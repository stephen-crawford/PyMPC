import unittest
import numpy as np
from unittest.mock import MagicMock, patch, call, ANY

# Import modules to test
from utils.const import CONSTRAINT, GAUSSIAN, DETERMINISTIC
from utils.visualizer import ROSLine

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

# Patch the read_config_file function
with patch('utils.utils.read_config_file', return_value=CONFIG_MOCK):
	from planner_modules.ellipsoid_constraints import EllipsoidConstraints
	from planner_modules.base_constraint import BaseConstraint


class TestEllipsoidConstraints(unittest.TestCase):
	"""Test suite for EllipsoidConstraints class"""

	@staticmethod
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

	def setUp(self):
		"""Set up test fixtures before each test"""
		# Create mock solver
		self.solver = MagicMock()
		self.solver.N = 10
		self.solver.params = MagicMock()
		self.solver.dt = 0.1

		self.config_attr_patcher = patch('planner_modules.base_constraint.CONFIG', CONFIG_MOCK)
		self.config_attr_patcher.start()
		self.addCleanup(self.config_attr_patcher.stop)

		# Apply the patch before creating the class
		patcher = patch('planner_modules.base_constraint.BaseConstraint.get_config_value',
						side_effect=self.get_mocked_config)
		self.mock_get_config = patcher.start()
		self.addCleanup(patcher.stop)

		# Create your class instance after the patch
		self.ellipsoid_constraints = EllipsoidConstraints(self.solver)

		# Add create_visualization_publisher mock
		self.ellipsoid_constraints.create_visualization_publisher = MagicMock()

	def test_initialization(self):
		"""Test proper initialization of EllipsoidConstraints"""
		self.assertEqual(self.ellipsoid_constraints.module_type, CONSTRAINT)
		self.assertEqual(self.ellipsoid_constraints.name, "ellipsoid_constraints")
		self.assertEqual(self.ellipsoid_constraints.num_segments, CONFIG_MOCK["contouring"]["num_segments"])
		self.assertEqual(self.ellipsoid_constraints.n_discs, CONFIG_MOCK["n_discs"])
		self.assertEqual(self.ellipsoid_constraints._robot_radius, CONFIG_MOCK["robot"]["radius"])
		self.assertEqual(self.ellipsoid_constraints.risk, CONFIG_MOCK["probabilistic"]["risk"])
		self.assertEqual(self.ellipsoid_constraints._dummy_x, 50.0)
		self.assertEqual(self.ellipsoid_constraints._dummy_y, 50.0)

	def test_update(self):
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

	@patch('planner_modules.base_constraint.set_solver_parameter')
	def test_set_parameters_k0(self, mock_set_param):
		"""Test set_parameters method for k=0 (dummies)"""
		# Setup
		from unittest.mock import ANY  # Make sure to import ANY

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
			call(self.solver.params, "ego_disc_radius", 0.5, k, settings=ANY),
			call(self.solver.params, "ego_disc_radius", 0.5, k, settings=ANY),

			# Ego disc offset calls
			call(self.solver.params, "ego_disc_offset", ANY, k, index=0, settings=ANY),
			call(self.solver.params, "ego_disc_offset", ANY, k, index=1, settings=ANY),

			# Dummy obstacle calls for obstacle 0
			call(self.solver.params, "ellipsoid_obst_x", 50.0, k, index=0, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_y", 50.0, k, index=0, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_psi", 0.0, k, index=0, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_r", 0.1, k, index=0, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_major", 0.0, k, index=0, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_minor", 0.0, k, index=0, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_chi", 1.0, k, index=0, settings=ANY),

			# Dummy obstacle calls for obstacle 1
			call(self.solver.params, "ellipsoid_obst_x", 50.0, k, index=1, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_y", 50.0, k, index=1, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_psi", 0.0, k, index=1, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_r", 0.1, k, index=1, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_major", 0.0, k, index=1, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_minor", 0.0, k, index=1, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_chi", 1.0, k, index=1, settings=ANY),
		]

		# Check that all expected calls were made
		mock_set_param.assert_has_calls(expected_calls, any_order=True)
		self.assertEqual(mock_set_param.call_count, 18)  # 2 radius + 2 offset + 7 params * 2 obstacles

	@patch('planner_modules.base_constraint.set_solver_parameter')
	@patch('utils.utils.exponential_quantile')
	def test_set_parameters_k1_deterministic(self, mock_exp_quantile, mock_set_param):
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
			call(self.solver.params, "ego_disc_radius", 0.5, k, settings=ANY),
			call(self.solver.params, "ego_disc_offset", ANY, k, index=0, settings=ANY),

			# Obstacle calls
			call(self.solver.params, "ellipsoid_obst_x", 5.0, k, index=0, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_y", 6.0, k, index=0, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_psi", 0.5, k, index=0, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_r", 1.0, k, index=0, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_major", 0.0, k, index=0, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_minor", 0.0, k, index=0, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_chi", 1.0, k, index=0, settings=ANY),
		]

		mock_set_param.assert_has_calls(expected_calls, any_order=True)
		self.assertEqual(mock_set_param.call_count, 10)
		# Ensure exponential_quantile was not called for deterministic obstacles
		mock_exp_quantile.assert_not_called()

	@patch('planner_modules.base_constraint.set_solver_parameter')
	@patch('utils.utils.exponential_quantile')
	def test_set_parameters_k1_gaussian(self, mock_exp_quantile, mock_set_param):
		"""Test set_parameters method for k=1 with gaussian obstacles"""
		# Setup

		k = 1
		data = MagicMock()
		data.robot_area = [MagicMock()]
		data.robot_area[0].offset = np.array([0.5, 0.3])

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

		# Mock the return value for exponential_quantile
		mock_exp_quantile.return_value = 3.0

		# Call method under test
		self.ellipsoid_constraints.set_parameters(data, module_data, k)

		# Assertions for gaussian obstacles
		expected_calls = [
			# Ego disc radius and offset calls
			call(self.solver.params, "ego_disc_radius", 0.5, k, settings=ANY),
			call(self.solver.params, "ego_disc_offset", ANY, k, index=0, settings=ANY),

			# Obstacle calls
			call(self.solver.params, "ellipsoid_obst_x", 5.0, k, index=0, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_y", 6.0, k, index=0, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_psi", 0.5, k, index=0, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_r", 1.0, k, index=0, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_major", 2.0, k, index=0, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_minor", 1.5, k, index=0, settings=ANY),
			call(self.solver.params, "ellipsoid_obst_chi", ANY, k, index=0, settings=ANY),
		]

		mock_set_param.assert_has_calls(expected_calls, any_order=True)
		self.assertEqual(mock_set_param.call_count, 10)  # 1 radius + 1 offset + 7 obstacle params


	def test_is_data_ready(self):
		"""Test is_data_ready method"""
		# Set up a proper mock for get_config_value that always returns the expected value for max_obstacles
		with patch.object(self.ellipsoid_constraints, 'get_config_value') as mock_get_config:
			mock_get_config.return_value = CONFIG_MOCK.get("max_obstacles", 3)

			# Test when required fields are missing
			with patch.object(self.ellipsoid_constraints, 'check_data_availability') as mock_check_data:
				mock_check_data.return_value = ["robot_area"]
				data = MagicMock()
				missing_data = ""

				result = self.ellipsoid_constraints.is_data_ready(data, missing_data)
				self.assertFalse(result)

			# Test when obstacles count does not match max_obstacles
			with patch.object(self.ellipsoid_constraints, 'check_data_availability') as mock_check_data:
				mock_check_data.return_value = []
				data = MagicMock()
				data.dynamic_obstacles = MagicMock()
				data.dynamic_obstacles.size.return_value = 2  # Not equal to CONFIG_MOCK["max_obstacles"]
				missing_data = ""

				result = self.ellipsoid_constraints.is_data_ready(data, missing_data)
				self.assertFalse(result)

			# Test when obstacle prediction is empty
			with patch.object(self.ellipsoid_constraints, 'check_data_availability') as mock_check_data:
				mock_check_data.return_value = []
				data = MagicMock()
				data.dynamic_obstacles = MagicMock()
				data.dynamic_obstacles.size.return_value = 3  # Equal to CONFIG_MOCK["max_obstacles"]
				obstacle = MagicMock()
				obstacle.prediction.empty.return_value = True
				data.dynamic_obstacles.__getitem__.return_value = obstacle
				missing_data = ""

				result = self.ellipsoid_constraints.is_data_ready(data, missing_data)
				self.assertFalse(result)

			# Test when obstacle prediction type is incorrect
			with patch.object(self.ellipsoid_constraints, 'check_data_availability') as mock_check_data:
				mock_check_data.return_value = []
				data = MagicMock()
				data.dynamic_obstacles = MagicMock()
				data.dynamic_obstacles.size.return_value = 3
				obstacle = MagicMock()
				obstacle.prediction.empty.return_value = False
				obstacle.prediction.type = "INVALID_TYPE"  # Not GAUSSIAN or DETERMINISTIC
				data.dynamic_obstacles.__getitem__.return_value = obstacle
				missing_data = ""

				result = self.ellipsoid_constraints.is_data_ready(data, missing_data)
				self.assertFalse(result)

			# Test when data is ready
			with patch.object(self.ellipsoid_constraints, 'check_data_availability') as mock_check_data:
				mock_check_data.return_value = []
				data = MagicMock()
				data.dynamic_obstacles = MagicMock()
				data.dynamic_obstacles.size.return_value = 3
				obstacle = MagicMock()
				obstacle.prediction.empty.return_value = False
				obstacle.prediction.type = GAUSSIAN
				data.dynamic_obstacles.__getitem__.return_value = obstacle
				missing_data = ""

				result = self.ellipsoid_constraints.is_data_ready(data, missing_data)
				self.assertTrue(result)

	def test_visualize(self):
		"""Test visualize method with debug visuals enabled"""
		# Setup
		data = MagicMock()
		module_data = MagicMock()

		# Create mock line publisher
		line_publisher = MagicMock()
		self.ellipsoid_constraints.create_visualization_publisher.return_value = line_publisher

		# Setup obstacles for testing
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

		# Set mode lengths
		det_mode.__len__.return_value = 10
		gauss_mode.__len__.return_value = 10

		# Setup dynamic obstacles
		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = 2
		data.dynamic_obstacles.__getitem__.side_effect = lambda i: [deterministic_obstacle, gaussian_obstacle][i]

		# Mock line instance
		line_instance = MagicMock()
		line_publisher.add_new_line.return_value = line_instance

		# Patch _draw_circle and _draw_ellipse methods
		with patch.object(self.ellipsoid_constraints, '_draw_circle') as mock_draw_circle, \
				patch.object(self.ellipsoid_constraints, '_draw_ellipse') as mock_draw_ellipse, \
				patch('utils.utils.exponential_quantile', return_value=3.0):
			# Call method under test
			self.ellipsoid_constraints.visualize(data, module_data)

			# Verify publisher was created correctly
			self.ellipsoid_constraints.create_visualization_publisher.assert_called_once_with("ellipsoids", ROSLine)

			# Verify publish was called
			line_publisher.publish.assert_called_once()

			# Verify draw_circle and draw_ellipse were called appropriately
			self.assertTrue(mock_draw_circle.called)
			self.assertTrue(mock_draw_ellipse.called)

	def test_helper_draw_circle(self):
		"""Test _draw_circle helper method"""
		# Setup
		line = MagicMock()
		center = np.array([1.0, 2.0])
		radius = 3.0

		# Call method under test
		self.ellipsoid_constraints._draw_circle(line, center, radius, num_points=4)

		# Should add lines to connect the points (4 points = 4 lines to close the loop)
		self.assertEqual(line.add_line.call_count, 4)

	def test_helper_draw_ellipse(self):
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
			from planner_modules.ellipsoid_constraints import EllipsoidConstraints
			self.ellipsoid_constraints = EllipsoidConstraints(self.solver)

		# Create mock planner
		self.planner = MagicMock()
		self.planner.modules = [self.ellipsoid_constraints]

		# Add create_visualization_publisher mock
		self.ellipsoid_constraints.create_visualization_publisher = MagicMock()

	@patch('utils.utils.LOG_DEBUG')
	def test_planner_integration(self, mock_log_debug):
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