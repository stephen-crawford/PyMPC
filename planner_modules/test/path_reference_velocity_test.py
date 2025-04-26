import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np

from planner.src.types import TwoDimensionalSpline
from utils.const import OBJECTIVE
from utils.utils import read_config_file

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

# Patch the read_config_file function
with patch('utils.utils.read_config_file', return_value=CONFIG_MOCK):
	from planner_modules.path_reference_velocity import PathReferenceVelocity
	from planner_modules.base_constraint import BaseConstraint


class TestPathReferenceVelocity(unittest.TestCase):

	@staticmethod
	def get_mocked_config(key, default=None):
		"""Static method to handle config mocking"""
		print("Trying to get key: " + str(key))
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

		self.config_attr_patcher = patch('planner_modules.base_constraint.CONFIG', CONFIG_MOCK)
		self.config_attr_patcher.start()
		self.addCleanup(self.config_attr_patcher.stop)

		# Apply the patch before creating the class
		patcher = patch('planner_modules.path_reference_velocity.PathReferenceVelocity.get_config_value',
						side_effect=self.get_mocked_config)
		self.mock_get_config = patcher.start()

		self.addCleanup(patcher.stop)
		# Initialize class under test
		self.prv = PathReferenceVelocity(self.solver)

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
		self.assertEqual(self.prv.n_segments, 10)
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

	def test_on_data_received_with_velocity(self):
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
		self.prv.velocity_spline.set_points.assert_called_once_with(
			self.mock_data.reference_path.s, self.mock_data.reference_path.v
		)

	def test_on_data_received_without_velocity(self):
		"""Test receiving reference path data without velocity."""
		# Setup
		self.prv.velocity_spline = MagicMock()
		self.mock_data.reference_path = MagicMock()
		self.mock_data.reference_path.hasVelocity.return_value = False

		# Execute
		self.prv.on_data_received(self.mock_data, "reference_path")

		self.prv.velocity_spline.set_points.assert_not_called()

	def test_on_data_received_wrong_data_name(self):
		"""Test receiving data with incorrect data name."""
		# Setup
		self.prv.velocity_spline = MagicMock()

		# Execute
		self.prv.on_data_received(self.mock_data, "wrong_data_name")

		self.prv.velocity_spline.set_points.assert_not_called()

	def test_set_parameters_with_velocity_spline(self):
		"""Test setting parameters with a velocity spline."""
		# Setup
		self.prv.velocity_spline = MagicMock()
		self.prv.velocity_spline.m_x_.size.return_value = 10
		self.prv.velocity_spline.get_parameters.side_effect = [
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
		self.mock_data.reference_path.hasVelocity.return_value = True

		# Patch the instance method directly
		self.prv.set_solver_param = MagicMock()

		# Execute
		self.prv.set_parameters(self.mock_data, self.mock_module_data, 0)

		# Verify
		expected_calls = [
			call(self.solver.params, "spline_va", 1.0, 0, 0),
			call(self.solver.params, "spline_vb", 2.0, 0, 0),
			call(self.solver.params, "spline_vc", 3.0, 0, 0),
			call(self.solver.params, "spline_vd", 4.0, 0, 0),
			call(self.solver.params, "spline_va", 5.0, 0, 1),
			call(self.solver.params, "spline_vb", 6.0, 0, 1),
			call(self.solver.params, "spline_vc", 7.0, 0, 1),
			call(self.solver.params, "spline_vd", 8.0, 0, 1),
			call(self.solver.params, "spline_va", 9.0, 0, 2),
			call(self.solver.params, "spline_vb", 10.0, 0, 2),
			call(self.solver.params, "spline_vc", 11.0, 0, 2),
			call(self.solver.params, "spline_vd", 12.0, 0, 2)
		]
		self.prv.set_solver_param.assert_has_calls(expected_calls)

	def test_set_parameters_without_velocity_spline(self):
		"""Test setting parameters without a velocity spline."""
		# Setup
		self.mock_data.reference_path = MagicMock()
		self.mock_data.reference_path.hasVelocity.return_value = False

		# Patch the instance method directly
		self.prv.set_solver_param = MagicMock()

		# Execute
		self.prv.set_parameters(self.mock_data, self.mock_module_data, 0)

		# Verify
		expected_calls = [
			call(self.solver.params, "spline_va", 0.0, 0, 0),
			call(self.solver.params, "spline_vb", 0.0, 0, 0),
			call(self.solver.params, "spline_vc", 0.0, 0, 0),
			call(self.solver.params, "spline_vd", 1.0, 0, 0),  # From weights.reference_velocity = 1.0
			call(self.solver.params, "spline_va", 0.0, 0, 1),
			call(self.solver.params, "spline_vb", 0.0, 0, 1),
			call(self.solver.params, "spline_vc", 0.0, 0, 1),
			call(self.solver.params, "spline_vd", 1.0, 0, 1),
			call(self.solver.params, "spline_va", 0.0, 0, 2),
			call(self.solver.params, "spline_vb", 0.0, 0, 2),
			call(self.solver.params, "spline_vc", 0.0, 0, 2),
			call(self.solver.params, "spline_vd", 1.0, 0, 2)
		]
		self.prv.set_solver_param.assert_has_calls(expected_calls)

	def test_set_parameters_with_k_not_zero(self):
		"""Test setting parameters with k != 0."""
		# Setup
		self.mock_data.reference_path = MagicMock()
		self.mock_data.reference_path.hasVelocity.return_value = False

		# Patch the instance method directly
		self.prv.set_solver_param = MagicMock()

		# Execute
		self.prv.set_parameters(self.mock_data, self.mock_module_data, 1)

		# Verify that reference_velocity is 0.0 for k != 0
		expected_calls = [
			call(self.solver.params, "spline_va", 0.0, 1, 0),
			call(self.solver.params, "spline_vb", 0.0, 1, 0),
			call(self.solver.params, "spline_vc", 0.0, 1, 0),
			call(self.solver.params, "spline_vd", 0.0, 1, 0),
			call(self.solver.params, "spline_va", 0.0, 1, 1),
			call(self.solver.params, "spline_vb", 0.0, 1, 1),
			call(self.solver.params, "spline_vc", 0.0, 1, 1),
			call(self.solver.params, "spline_vd", 0.0, 1, 1),
			call(self.solver.params, "spline_va", 0.0, 1, 2),
			call(self.solver.params, "spline_vb", 0.0, 1, 2),
			call(self.solver.params, "spline_vc", 0.0, 1, 2),
			call(self.solver.params, "spline_vd", 0.0, 1, 2)
		]
		self.prv.set_solver_param.assert_has_calls(expected_calls)

	def test_set_parameters_out_of_bounds(self):
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

		# Patch the instance method directly
		self.prv.set_solver_param = MagicMock()

		# Execute
		self.prv.set_parameters(self.mock_data, self.mock_module_data, 0)

		# Verify - all out of bounds should set zeros
		out_of_bounds_calls = [
			call(self.solver.params, "spline_va", 0.0, 0, 0),  # Index 1 is out of bounds
			call(self.solver.params, "spline_vb", 0.0, 0, 0),
			call(self.solver.params, "spline_vc", 0.0, 0, 0),
			call(self.solver.params, "spline_vd", 0.0, 0, 0),
			call(self.solver.params, "spline_va", 0.0, 0, 1),  # Index 2 is out of bounds
			call(self.solver.params, "spline_vb", 0.0, 0, 1),
			call(self.solver.params, "spline_vc", 0.0, 0, 1),
			call(self.solver.params, "spline_vd", 0.0, 0, 1),
			call(self.solver.params, "spline_va", 0.0, 0, 2),  # Index 3 is out of bounds
			call(self.solver.params, "spline_vb", 0.0, 0, 2),
			call(self.solver.params, "spline_vc", 0.0, 0, 2),
			call(self.solver.params, "spline_vd", 0.0, 0, 2)
		]
		self.prv.set_solver_param.assert_has_calls(out_of_bounds_calls)

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