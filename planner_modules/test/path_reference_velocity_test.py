import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np

from planner.src.types import TwoDimensionalSpline
from utils.const import OBJECTIVE
from utils.utils import read_config_file


class TestPathReferenceVelocity(unittest.TestCase):
	@patch('utils.utils.read_config_file')
	@patch('utils.utils.LOG_DEBUG')
	def setUp(self, mock_log_debug, mock_read_config):
		# Configure mocks
		self.mock_config = {
			"contouring": {"get_num_segments": 3},
			"weights": {"reference_velocity": 10.0},
			"debug_visuals": True
		}
		mock_read_config.return_value = self.mock_config

		# Import here to use the mocked config
		from path_reference_velocity import PathReferenceVelocity

		# Create mock solver
		self.mock_solver = MagicMock()
		self.mock_solver.params = {}

		# Initialize class under test
		self.prv = PathReferenceVelocity(self.mock_solver)

		# Verify initialization logs
		mock_log_debug.assert_any_call("Initializing Path Reference Velocity")
		mock_log_debug.assert_any_call("Path Reference Velocity successfully initialized")

		# Create mock data structures for tests
		self.mock_state = MagicMock()
		self.mock_data = MagicMock()
		self.mock_module_data = MagicMock()
		self.mock_module_data.path_velocity = None
		self.mock_module_data.current_path_segment = 0

	def test_initialization(self):
		"""Test the initialization of PathReferenceVelocity."""
		self.assertEqual(self.prv.name, "path_reference_velocity")
		self.assertEqual(self.prv.module_type, OBJECTIVE)
		self.assertEqual(self.prv.n_segments, 3)
		self.assertIsNone(self.prv.velocity_spline)

	def test_update_with_existing_spline(self):
		"""Test the update method when a velocity spline exists."""
		# Setup
		self.prv.velocity_spline = MagicMock()

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
		self.prv.velocity_spline = MagicMock()
		self.mock_module_data.path_velocity = MagicMock()
		existing_velocity = self.mock_module_data.path_velocity

		# Execute
		self.prv.update(self.mock_state, self.mock_data, self.mock_module_data)

		# Verify that nothing changes
		self.assertEqual(self.mock_module_data.path_velocity, existing_velocity)

	@patch('utils.utils.LOG_DEBUG')
	def test_on_data_received_with_velocity(self, mock_log_debug):
		"""Test receiving reference path data with velocity."""
		# Setup
		self.prv.velocity_spline = MagicMock()
		self.mock_data.reference_path = MagicMock()
		self.mock_data.reference_path.hasVelocity.return_value = True
		self.mock_data.reference_path.s = np.array([0, 1, 2])
		self.mock_data.reference_path.v = np.array([10, 20, 30])

		# Execute
		self.prv.on_data_received(self.mock_data, "reference_path")

		# Verify
		mock_log_debug.assert_called_with("Received Reference Path")
		self.prv.velocity_spline.set_points.assert_called_once_with(
			self.mock_data.reference_path.s, self.mock_data.reference_path.v
		)

	@patch('utils.utils.LOG_DEBUG')
	def test_on_data_received_without_velocity(self, mock_log_debug):
		"""Test receiving reference path data without velocity."""
		# Setup
		self.prv.velocity_spline = MagicMock()
		self.mock_data.reference_path = MagicMock()
		self.mock_data.reference_path.hasVelocity.return_value = False

		# Execute
		self.prv.on_data_received(self.mock_data, "reference_path")

		# Verify
		mock_log_debug.assert_called_with("Received Reference Path")
		self.prv.velocity_spline.set_points.assert_not_called()

	@patch('utils.utils.LOG_DEBUG')
	def test_on_data_received_wrong_data_name(self, mock_log_debug):
		"""Test receiving data with incorrect data name."""
		# Setup
		self.prv.velocity_spline = MagicMock()

		# Execute
		self.prv.on_data_received(self.mock_data, "wrong_data_name")

		# Verify
		mock_log_debug.assert_not_called()
		self.prv.velocity_spline.set_points.assert_not_called()

	@patch('path_reference_velocity.set_solver_parameter')
	@patch('utils.utils.LOG_DEBUG')
	def test_set_parameters_with_velocity_spline(self, mock_log_debug, mock_set_param):
		"""Test setting parameters with a velocity spline."""
		# Setup
		self.prv.velocity_spline = MagicMock()
		self.prv.velocity_spline.m_x_.size.return_value = 10
		self.prv.velocity_spline.get_parameters.side_effect = [
			(1.0, 2.0, 3.0, 4.0),
			(5.0, 6.0, 7.0, 8.0),
			(9.0, 10.0, 11.0, 12.0)
		]
		self.mock_data.reference_path = MagicMock()
		self.mock_data.reference_path.hasVelocity.return_value = True

		# Execute
		self.prv.set_parameters(self.mock_data, self.mock_module_data, 0)

		# Verify
		mock_log_debug.assert_called_with("Using spline-based reference velocity")
		expected_calls = [
			call(self.mock_solver.params, "spline_va", 1.0, 0, 0),
			call(self.mock_solver.params, "spline_vb", 2.0, 0, 0),
			call(self.mock_solver.params, "spline_vc", 3.0, 0, 0),
			call(self.mock_solver.params, "spline_vd", 4.0, 0, 0),
			call(self.mock_solver.params, "spline_va", 5.0, 0, 1),
			call(self.mock_solver.params, "spline_vb", 6.0, 0, 1),
			call(self.mock_solver.params, "spline_vc", 7.0, 0, 1),
			call(self.mock_solver.params, "spline_vd", 8.0, 0, 1),
			call(self.mock_solver.params, "spline_va", 9.0, 0, 2),
			call(self.mock_solver.params, "spline_vb", 10.0, 0, 2),
			call(self.mock_solver.params, "spline_vc", 11.0, 0, 2),
			call(self.mock_solver.params, "spline_vd", 12.0, 0, 2)
		]
		mock_set_param.assert_has_calls(expected_calls)

	@patch('path_reference_velocity.set_solver_parameter')
	def test_set_parameters_without_velocity_spline(self, mock_set_param):
		"""Test setting parameters without a velocity spline."""
		# Setup
		self.mock_data.reference_path = MagicMock()
		self.mock_data.reference_path.hasVelocity.return_value = False

		# Execute
		self.prv.set_parameters(self.mock_data, self.mock_module_data, 0)

		# Verify
		expected_calls = [
			call(self.mock_solver.params, "spline_va", 0.0, 0, 0),
			call(self.mock_solver.params, "spline_vb", 0.0, 0, 0),
			call(self.mock_solver.params, "spline_vc", 0.0, 0, 0),
			call(self.mock_solver.params, "spline_vd", 10.0, 0, 0),
			call(self.mock_solver.params, "spline_va", 0.0, 0, 1),
			call(self.mock_solver.params, "spline_vb", 0.0, 0, 1),
			call(self.mock_solver.params, "spline_vc", 0.0, 0, 1),
			call(self.mock_solver.params, "spline_vd", 10.0, 0, 1),
			call(self.mock_solver.params, "spline_va", 0.0, 0, 2),
			call(self.mock_solver.params, "spline_vb", 0.0, 0, 2),
			call(self.mock_solver.params, "spline_vc", 0.0, 0, 2),
			call(self.mock_solver.params, "spline_vd", 10.0, 0, 2)
		]
		mock_set_param.assert_has_calls(expected_calls)

	@patch('path_reference_velocity.set_solver_parameter')
	def test_set_parameters_with_k_not_zero(self, mock_set_param):
		"""Test setting parameters with k != 0."""
		# Setup
		self.mock_data.reference_path = MagicMock()
		self.mock_data.reference_path.hasVelocity.return_value = False

		# Execute
		self.prv.set_parameters(self.mock_data, self.mock_module_data, 1)

		# Verify that reference_velocity is 0.0 for k != 0
		expected_calls = [
			call(self.mock_solver.params, "spline_va", 0.0, 1, 0),
			call(self.mock_solver.params, "spline_vb", 0.0, 1, 0),
			call(self.mock_solver.params, "spline_vc", 0.0, 1, 0),
			call(self.mock_solver.params, "spline_vd", 0.0, 1, 0),
			# ... similar for other segments
		]
		mock_set_param.assert_has_calls(expected_calls[:4])  # Just check first segment

	@patch('path_reference_velocity.set_solver_parameter')
	@patch('utils.utils.LOG_DEBUG')
	def test_set_parameters_out_of_bounds(self, mock_log_debug, mock_set_param):
		"""Test setting parameters with path segment out of bounds."""
		# Setup
		self.prv.velocity_spline = MagicMock()
		self.prv.velocity_spline.m_x_.size.return_value = 2  # Only 2 points
		self.prv.velocity_spline.get_parameters.side_effect = [
			(1.0, 2.0, 3.0, 4.0),  # For index 0
		]
		self.mock_data.reference_path = MagicMock()
		self.mock_data.reference_path.hasVelocity.return_value = True
		self.mock_module_data.current_path_segment = 1  # Makes index 1, 2, 3 - only 1 is valid

		# Execute
		self.prv.set_parameters(self.mock_data, self.mock_module_data, 0)

		# Verify - all out of bounds should set zeros
		out_of_bounds_calls = [
			call(self.mock_solver.params, "spline_va", 0.0, 0, 0),  # Index 1 is out of bounds
			call(self.mock_solver.params, "spline_vb", 0.0, 0, 0),
			call(self.mock_solver.params, "spline_vc", 0.0, 0, 0),
			call(self.mock_solver.params, "spline_vd", 0.0, 0, 0),
			call(self.mock_solver.params, "spline_va", 0.0, 0, 1),  # Index 2 is out of bounds
			call(self.mock_solver.params, "spline_vb", 0.0, 0, 1),
			call(self.mock_solver.params, "spline_vc", 0.0, 0, 1),
			call(self.mock_solver.params, "spline_vd", 0.0, 0, 1),
			call(self.mock_solver.params, "spline_va", 0.0, 0, 2),  # Index 3 is out of bounds
			call(self.mock_solver.params, "spline_vb", 0.0, 0, 2),
			call(self.mock_solver.params, "spline_vc", 0.0, 0, 2),
			call(self.mock_solver.params, "spline_vd", 0.0, 0, 2)
		]
		mock_set_param.assert_has_calls(out_of_bounds_calls)

	@patch('utils.visualizer.VISUALS')
	@patch('utils.utils.LOG_DEBUG')
	def test_visualize_with_empty_path(self, mock_log_debug, mock_visuals):
		"""Test visualization with empty reference path."""
		# Setup
		self.mock_data.reference_path.empty.return_value = True

		# Execute
		self.prv.visualize(self.mock_data, self.mock_module_data)

		# Verify
		mock_log_debug.assert_not_called()
		mock_visuals.get_publisher.assert_not_called()

	@patch('utils.visualizer.VISUALS')
	@patch('utils.utils.LOG_DEBUG')
	def test_visualize_with_empty_s(self, mock_log_debug, mock_visuals):
		"""Test visualization with empty s array."""
		# Setup
		self.mock_data.reference_path.empty.return_value = False
		self.mock_data.reference_path.s.empty.return_value = True

		# Execute
		self.prv.visualize(self.mock_data, self.mock_module_data)

		# Verify
		mock_log_debug.assert_not_called()
		mock_visuals.get_publisher.assert_not_called()

	@patch('utils.visualizer.VISUALS')
	@patch('utils.utils.LOG_DEBUG')
	def test_visualize_with_debug_visuals_off(self, mock_log_debug, mock_visuals):
		"""Test visualization with debug_visuals turned off."""
		# Setup
		self.mock_data.reference_path.empty.return_value = False
		self.mock_data.reference_path.s.empty.return_value = False
		self.mock_config["debug_visuals"] = False

		# Execute
		self.prv.visualize(self.mock_data, self.mock_module_data)

		# Verify
		mock_log_debug.assert_not_called()
		mock_visuals.get_publisher.assert_not_called()

	@patch('planner.src.types.TwoDimensionalSpline')
	@patch('utils.visualizer.VISUALS')
	@patch('utils.utils.LOG_DEBUG')
	def test_visualize_with_valid_data(self, mock_log_debug, mock_visuals, mock_spline_class):
		"""Test visualization with valid data."""
		# Setup
		self.mock_data.reference_path.empty.return_value = False
		self.mock_data.reference_path.s.empty.return_value = False
		self.mock_config["debug_visuals"] = True

		mock_publisher = MagicMock()
		mock_line = MagicMock()
		mock_visuals.get_publisher.return_value = mock_publisher
		mock_publisher.get_new_line.return_value = mock_line

		mock_spline_instance = MagicMock()
		mock_spline_class.return_value = mock_spline_instance
		mock_spline_instance.get_point.side_effect = [[1, 1], [2, 2], [3, 3]]

		self.prv.velocity_spline = MagicMock()
		self.prv.velocity_spline.m_x_.back.return_value = 3
		self.prv.velocity_spline.operator().side_effect = [1.0, 2.0, 3.0]

		# Execute
		self.prv.visualize(self.mock_data, self.mock_module_data)

		# Verify
		mock_log_debug.assert_called_with("PathReferenceVelocity.Visualize")
		mock_visuals.get_publisher.assert_called_with("path_velocity")
		mock_publisher.get_new_line.assert_called_once()
		mock_line.set_scale.assert_called_with(0.25, 0.25, 0.1)

		# Check that TwoDimensionalSpline was created correctly
		mock_spline_class.assert_called_once_with(
			self.mock_data.reference_path.x,
			self.mock_data.reference_path.y,
			self.mock_data.reference_path.s
		)

		# Verify line was added and published
		self.assertEqual(mock_line.add_line.call_count, 2)  # For s=1 and s=2
		mock_publisher.publish.assert_called_once()


if __name__ == '__main__':
	unittest.main()