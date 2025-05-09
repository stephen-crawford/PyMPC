import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np

from planner_modules.src.objectives.path_reference_velocity_objective import PathReferenceVelocityObjective
from utils.const import OBJECTIVE

CONFIG_MOCK = {
	"params": MagicMock(),
	"contouring": {
		"num_segments": 10,
		"get_num_segments": 10,
		"add_road_constraints": True,
		"dynamic_velocity_reference": False
	},
	"road": {
		"width": 3.5,
		"two_way": False
	},
	"robot": {
		"width": 0.8
	},
	"weights": {
		"contour": 1.0,
		"lag": 0.5,
		"terminal_angle": 0.1,
		"terminal_contouring": 2.0,
		"reference_velocity": 1.0,
		"velocity": 0.8
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


class TestPathReferenceVelocity(unittest.TestCase):

	def setUp(self):
		"""Set up test fixtures before each test"""
		# Create mock solver
		self.solver = MagicMock()
		self.solver.horizon = 10
		self.solver.params = MagicMock()

		# Mock the config getter method
		patcher = patch('planner_modules.src.objectives.base_objective.BaseObjective.get_config_value',
						side_effect=get_mocked_config)
		self.mock_get_config = patcher.start()
		self.addCleanup(patcher.stop)

		# Create a mock for the spline that will be used in various tests
		self.mock_spline = MagicMock()

		# Create the instance of the class under test
		self.prv = PathReferenceVelocityObjective(self.solver)

		# Set up mocks for tests
		self.mock_state = MagicMock()
		self.mock_data = MagicMock()
		self.mock_module_data = MagicMock()
		self.mock_module_data.path_velocity = None
		self.mock_module_data.current_path_segment = 0

	def test_initialization(self):
		"""Test the initialization of PathReferenceVelocity."""
		self.assertEqual(self.prv.name, "path_reference_velocity")
		self.assertEqual(self.prv.module_type, OBJECTIVE)
		self.assertIsNone(self.prv.velocity_spline)

	def test_update_with_existing_spline(self):
		"""Test the update method when a velocity spline exists."""
		# Setup
		self.prv.velocity_spline = self.mock_spline

		# Execute
		self.prv.update(self.mock_state, self.mock_data, self.mock_module_data)

		# Verify
		self.assertEqual(self.mock_module_data.path_velocity, self.prv.velocity_spline)

	def test_update_without_existing_spline(self):
		"""Test the update method when no velocity spline exists."""
		# Setup - velocity_spline is None by default

		# Execute
		self.prv.update(self.mock_state, self.mock_data, self.mock_module_data)

		# Verify that nothing changes
		self.assertIsNone(self.mock_module_data.path_velocity)

	def test_update_with_existing_module_data_path_velocity(self):
		"""Test the update method when module_data already has path_velocity."""
		# Setup
		existing_velocity = MagicMock()
		self.prv.velocity_spline = self.mock_spline
		self.mock_module_data.path_velocity = existing_velocity

		# Execute
		self.prv.update(self.mock_state, self.mock_data, self.mock_module_data)

		# Verify that nothing changes
		self.assertEqual(self.mock_module_data.path_velocity, existing_velocity)

	def test_on_data_received_with_velocity(self):
		"""Test receiving reference path data with velocity."""
		# Setup
		self.mock_data.reference_path.has_velocity.return_value = True
		self.mock_data.reference_path.s = np.array([0, 1, 2])
		self.mock_data.reference_path.v = np.array([10, 20, 30])

		# Since we're seeing that a new TkSpline is being created inside the function,
		# we need to make sure our mock is the one being used
		with patch('planner_modules.src.objectives.path_reference_velocity_objective.TkSpline',
				   return_value=self.mock_spline) as mock_tk_class:
			# Execute
			self.prv.on_data_received(self.mock_data, "reference_path")

			# Verify
			mock_tk_class.assert_called_once()
			self.mock_spline.set_points.assert_called_once_with(
				self.mock_data.reference_path.s, self.mock_data.reference_path.v
			)
			self.assertEqual(self.prv.velocity_spline, self.mock_spline)

	def test_on_data_received_without_velocity(self):
		"""Test receiving reference path data without velocity."""
		# Setup
		self.mock_data.reference_path.has_velocity.return_value = False

		# Execute
		self.prv.on_data_received(self.mock_data, "reference_path")

		# Verify that TkSpline is not created
		self.mock_spline.assert_not_called()
		# verify velocity_spline is not set
		self.assertIsNone(self.prv.velocity_spline)

	def test_on_data_received_wrong_data_name(self):
		"""Test receiving data with incorrect data name."""
		# Execute
		self.prv.on_data_received(self.mock_data, "wrong_data_name")

		# Verify that no spline operations occur
		self.mock_spline.assert_not_called()

	def test_set_parameters_with_velocity_spline(self):
		"""Test setting parameters with a velocity spline."""
		# Setup
		"""Test setting parameters with a velocity spline."""
		# Setup
		self.prv.velocity_spline = self.mock_spline
		self.mock_spline.m_x_ = MagicMock()
		self.mock_spline.m_x_.size.return_value = 10
		self.mock_spline.get_parameters.side_effect = [
			(1.0, 2.0, 3.0, 4.0),
			(5.0, 6.0, 7.0, 8.0),
			(9.0, 10.0, 11.0, 12.0),
			(13.0, 14.0, 15.0, 16.0),
			(17.0, 18.0, 19.0, 20.0),
			(21.0, 22.0, 23.0, 24.0),
			(25.0, 26.0, 27.0, 28.0),
			(29.0, 30.0, 31.0, 32.0),
			(33.0, 34.0, 35.0, 36.0),
			(37.0, 38.0, 39.0, 40.0)
		]
		self.mock_data.reference_path = MagicMock()
		self.mock_data.reference_path.has_velocity.return_value = True

		# Create a mock for the parameter manager
		param_manager = MagicMock()

		# Execute
		self.prv.set_parameters(param_manager, self.mock_data, self.mock_module_data, 0)

		# Verify - note the parameter names are different in the actual class
		expected_calls = [
			call("spline_0_va", 1.0),
			call("spline_0_vb", 2.0),
			call("spline_0_vc", 3.0),
			call("spline_0_vd", 4.0),
			call("spline_1_va", 5.0),
			call("spline_1_vb", 6.0),
			call("spline_1_vc", 7.0),
			call("spline_1_vd", 8.0),
			call("spline_2_va", 9.0),
			call("spline_2_vb", 10.0),
			call("spline_2_vc", 11.0),
			call("spline_2_vd", 12.0)
		]
		param_manager.set_parameter.assert_has_calls(expected_calls)

	def test_set_parameters_without_velocity_spline(self):
		"""Test setting parameters without a velocity spline."""
		"""Test setting parameters without a velocity spline."""
		# Setup
		self.mock_data.reference_path = MagicMock()
		self.mock_data.reference_path.has_velocity.return_value = False

		# Create a mock for the parameter manager
		param_manager = MagicMock()

		# Execute
		self.prv.set_parameters(param_manager, self.mock_data, self.mock_module_data, 0)

		# Verify - zeros for a,b,c and reference_velocity for d
		expected_calls = [
			call("spline_0_va", 0.0),
			call("spline_0_vb", 0.0),
			call("spline_0_vc", 0.0),
			call("spline_0_vd", 1.0),  # From weights.reference_velocity = 1.0
			call("spline_1_va", 0.0),
			call("spline_1_vb", 0.0),
			call("spline_1_vc", 0.0),
			call("spline_1_vd", 1.0),
			call("spline_2_va", 0.0),
			call("spline_2_vb", 0.0),
			call("spline_2_vc", 0.0),
			call("spline_2_vd", 1.0)
		]
		param_manager.set_parameter.assert_has_calls(expected_calls)

	def test_set_parameters_with_k_not_zero(self):
		"""Test setting parameters with k != 0."""
		# Setup
		self.mock_data.reference_path = MagicMock()
		self.mock_data.reference_path.has_velocity.return_value = False

		# Create a mock for the parameter manager
		param_manager = MagicMock()

		# Execute
		self.prv.set_parameters(param_manager, self.mock_data, self.mock_module_data, 1)

		# Verify that reference_velocity is 0.0 for k != 0
		expected_calls = [
			call("spline_0_va", 0.0),
			call("spline_0_vb", 0.0),
			call("spline_0_vc", 0.0),
			call("spline_0_vd", 0.0),
			call("spline_1_va", 0.0),
			call("spline_1_vb", 0.0),
			call("spline_1_vc", 0.0),
			call("spline_1_vd", 0.0),
		]
		param_manager.set_parameter.assert_has_calls(expected_calls)

	def test_set_parameters_out_of_bounds(self):
		"""Test setting parameters with path segment out of bounds."""
		# Setup
		self.prv.velocity_spline = self.mock_spline
		self.mock_spline.m_x_ = MagicMock()
		self.mock_spline.m_x_.size.return_value = 2  # Only 2 points
		self.mock_data.reference_path = MagicMock()
		self.mock_data.reference_path.has_velocity.return_value = True
		self.mock_module_data.current_path_segment = 1  # Makes indices exceed size

		# Create a mock for the parameter manager
		param_manager = MagicMock()

		# Execute
		self.prv.set_parameters(param_manager, self.mock_data, self.mock_module_data, 0)

		# Verify - all segments should get 0s due to being out of bounds
		out_of_bounds_calls = [
			call("spline_0_va", 0.0),
			call("spline_0_vb", 0.0),
			call("spline_0_vc", 0.0),
			call("spline_0_vd", 0.0),
			call("spline_1_va", 0.0),
			call("spline_1_vb", 0.0),
			call("spline_1_vc", 0.0),
			call("spline_1_vd", 0.0),
		]

		param_manager.set_parameter.assert_has_calls(out_of_bounds_calls)

	@patch('utils.visualizer.VISUALS')
	def test_visualize_with_empty_path(self, mock_visuals):
		"""Test visualization with empty reference path."""
		# Setup
		self.mock_data.reference_path.empty.return_value = True

		# Execute
		self.prv.visualize(self.mock_data, self.mock_module_data)

		# Verify
		mock_visuals.get_publisher.assert_not_called()

	@patch('utils.visualizer.VISUALS')
	def test_visualize_with_empty_s(self, mock_visuals):
		"""Test visualization with empty s array."""
		# Setup
		self.mock_data.reference_path.empty.return_value = False
		self.mock_data.reference_path.s.empty.return_value = True

		# Execute
		self.prv.visualize(self.mock_data, self.mock_module_data)

		# Verify
		mock_visuals.get_publisher.assert_not_called()

	@patch('utils.visualizer.VISUALS')
	def test_visualize_with_debug_visuals_off(self, mock_visuals):
		"""Test visualization with debug_visuals turned off."""
		# Setup
		self.mock_data.reference_path.empty.return_value = False
		self.mock_data.reference_path.s.empty.return_value = False
		CONFIG_MOCK["debug_visuals"] = False

		# Execute
		self.prv.visualize(self.mock_data, self.mock_module_data)

		mock_visuals.get_publisher.assert_not_called()


if __name__ == '__main__':
	unittest.main()