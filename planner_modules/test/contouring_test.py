import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.interpolate import CubicSpline

# Import modules to test
from utils.const import OBJECTIVE, CONSTRAINT

# Manually patch CONFIG to avoid dependency issues in testing
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

CONFIG_MOCK["params"].parameter_bundles = {
    "contour": [0, 1, 2],
    "lag": [1],
    "terminal_angle": [1],
    "terminal_contouring": [1]# Example bundle index list
}
CONFIG_MOCK["params"].length.return_value = 10

# Patch the read_config_file function
with patch('utils.utils.read_config_file', return_value=CONFIG_MOCK):
    from planner_modules.contouring_constraints import ContouringConstraints
    from planner_modules.contouring import Contouring


class TestContouringConstraints(unittest.TestCase):
    """Test suite for ContouringConstraints class"""

    def setUp(self):
        """Set up test fixtures before each test"""
        # Create mock solver
        self.solver = MagicMock()
        self.solver.N = 10
        self.solver.params = MagicMock()
        # Create instance of the class under test
        self.contouring_constraints = ContouringConstraints(self.solver)

    def test_initialization(self):
        """Test proper initialization of ContouringConstraints"""
        self.assertEqual(self.contouring_constraints.module_type, CONSTRAINT)
        self.assertEqual(self.contouring_constraints.name, "contouring_constraints")
        self.assertEqual(self.contouring_constraints.num_segments, CONFIG_MOCK["contouring"]["num_segments"])
        self.assertIsNone(self.contouring_constraints.width_left)
        self.assertIsNone(self.contouring_constraints.width_right)

    def test_update_with_existing_widths(self):
        """Test update method when widths exist"""
        # Setup
        module_data = MagicMock()
        module_data.path_width_left = None
        module_data.path_width_right = None

        # Create mock splines
        mock_x = np.linspace(0, 10, 10)
        mock_y = np.sin(mock_x)
        self.contouring_constraints.width_left = CubicSpline(mock_x, mock_y)
        self.contouring_constraints.width_right = CubicSpline(mock_x, mock_y)

        # Call method under test
        self.contouring_constraints.update(MagicMock(), MagicMock(), module_data)

        # Assertions
        self.assertEqual(module_data.path_width_left, self.contouring_constraints.width_left)
        self.assertEqual(module_data.path_width_right, self.contouring_constraints.width_right)

    def test_on_data_received_with_boundaries(self):
        """Test on_data_received with valid boundary data"""
        # Setup mock data
        data = MagicMock()
        data.reference_path.x = np.linspace(0, 10, 10)
        data.reference_path.y = np.sin(data.reference_path.x)
        data.reference_path.s = np.linspace(0, 10, 10)

        data.left_bound.empty.return_value = False
        data.left_bound.x = data.reference_path.x + 1
        data.left_bound.y = data.reference_path.y + 1

        data.right_bound.empty.return_value = False
        data.right_bound.x = data.reference_path.x - 1
        data.right_bound.y = data.reference_path.y - 1

        # Call method under test
        self.contouring_constraints.on_data_received(data, "reference_path")

        # Assertions
        self.assertIsNotNone(self.contouring_constraints.width_left)
        self.assertIsNotNone(self.contouring_constraints.width_right)
        self.assertTrue(hasattr(self.contouring_constraints.width_left, 'm_x_'))
        self.assertTrue(hasattr(self.contouring_constraints.width_right, 'm_x_'))
        self.assertTrue(hasattr(self.contouring_constraints.width_left, 'get_parameters'))
        self.assertTrue(hasattr(self.contouring_constraints.width_right, 'get_parameters'))


    @patch('planner_modules.contouring_constraints.set_solver_parameter')
    def test_set_parameters(self, mock_set_param):
        """Test set_parameters method with boundary data"""
        # Setup
        k = 1
        module_data = MagicMock()
        module_data.current_path_segment = 0

        # Create mock splines with proper methods
        mock_x = np.linspace(0, 10, 10)
        mock_y = np.sin(mock_x)
        self.contouring_constraints.width_left = CubicSpline(mock_x, mock_y)
        self.contouring_constraints.width_right = CubicSpline(mock_x, mock_y)

        # Add custom attributes to match implementation
        self.contouring_constraints.width_left.m_x_ = mock_x
        self.contouring_constraints.width_right.m_x_ = mock_x

        # Add custom method to get parameters
        def mock_get_parameters(index, a, b, c, d):
            return 1.0, 2.0, 3.0, 4.0 # Mock coefficients

        self.contouring_constraints.width_left.get_parameters = mock_get_parameters
        self.contouring_constraints.width_right.get_parameters = mock_get_parameters

        # Call method under test
        self.contouring_constraints.set_parameters(MagicMock(), module_data, k)

        # Assertions - check that parameters were set for each segment
        expected_calls = self.contouring_constraints.num_segments * 8  # 8 params per segment
        self.assertEqual(mock_set_param.call_count, expected_calls)

    def test_is_data_ready(self):
        """Test is_data_ready method"""
        # Test when data is not ready
        data = MagicMock()
        data.left_bound.empty.return_value = True
        data.right_bound.empty.return_value = False
        missing_data = ""

        result = self.contouring_constraints.is_data_ready(data, missing_data)
        self.assertFalse(result)

        # Test when data is ready
        data.left_bound.empty.return_value = False
        data.right_bound.empty.return_value = False

        result = self.contouring_constraints.is_data_ready(data, missing_data)
        self.assertTrue(result)


class TestContouring(unittest.TestCase):
    """Test suite for Contouring class"""

    def setUp(self):
        """Set up test fixtures before each test"""
        # Create mock solver
        self.solver = MagicMock()
        self.solver.N = 10
        self.solver.params = MagicMock()
        # Create instance of the class under test
        self.contouring = Contouring(self.solver)

    def test_initialization(self):
        """Test proper initialization of Contouring"""
        self.assertEqual(self.contouring.module_type, OBJECTIVE)
        self.assertEqual(self.contouring.name, "contouring")
        self.assertEqual(self.contouring.get_num_segments, CONFIG_MOCK["contouring"]["get_num_segments"])
        self.assertEqual(self.contouring.add_road_constraints, CONFIG_MOCK["contouring"]["add_road_constraints"])
        self.assertEqual(self.contouring.two_way_road, CONFIG_MOCK["road"]["two_way"])
        self.assertIsNone(self.contouring.spline)
        self.assertEqual(self.contouring.closest_segment, 0)

    def test_update_with_spline(self):
        """Test update method with valid spline"""
        # Setup mock spline
        mock_spline = MagicMock()
        mock_spline.find_closest_point.return_value = (5.0, 2)
        self.contouring.spline = mock_spline

        # Setup state and module_data
        state = MagicMock()
        state.getPos.return_value = np.array([1.0, 2.0])
        module_data = MagicMock()
        module_data.path = None

        # Call method under test
        self.contouring.update(state, MagicMock(), module_data)

        # Assertions
        state.set.assert_called_once_with("spline", 5.0)
        self.assertEqual(module_data.path, mock_spline)
        self.assertEqual(self.contouring.closest_segment, 2)

    def test_update_without_spline(self):
        """Test update method without spline initialization"""
        # Setup
        self.contouring.spline = None
        state = MagicMock()
        module_data = MagicMock()

        # Call method under test
        self.contouring.update(state, MagicMock(), module_data)

        # Assertions - should not update state
        state.set.assert_not_called()

    @patch('planner_modules.contouring.set_solver_parameter')
    def test_set_parameters(self, mock_set_param):
        """Test set_parameters method"""
        # Setup mock spline
        mock_spline = MagicMock()
        mock_spline.get_num_segments.return_value = 20
        mock_spline.get_parameters.return_value = (
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)  # ax, bx, cx, dx, ay, by, cy, dy
        mock_spline.get_segment_start.return_value = 0.0

        self.contouring.spline = mock_spline
        self.contouring.closest_segment = 3

        # Mock the set_spline_parameters method on the instance
        with patch.object(self.contouring, 'set_spline_parameters') as mock_set_spline_params:
            # Call method under test - first for k=0 then for k=1
            self.contouring.set_parameters(MagicMock(), MagicMock(), 0)
            self.contouring.set_parameters(MagicMock(), MagicMock(), 1)

            # Verify set_spline_parameters was called twice (once for k=0, once for k=1)
            self.assertEqual(mock_set_spline_params.call_count, 2)

    def test_on_data_received(self):
        """Test on_data_received with valid reference path"""
        # Setup mock data
        data = MagicMock()
        data.reference_path.x.empty.return_value = False
        data.reference_path.y.empty.return_value = False
        data.reference_path.s.empty.return_value = True

        # Setup mock for TwoDimensionalSpline
        mock_spline = MagicMock()
        with patch('planner_modules.contouring.TwoDimensionalSpline', return_value=mock_spline):
            # Call method under test
            self.contouring.on_data_received(data, "reference_path")

            # Assertions
            self.assertEqual(self.contouring.spline, mock_spline)
            self.assertEqual(self.contouring.closest_segment, 0)

    def test_on_data_received_with_bounds(self):
        """Test on_data_received with valid reference path and boundaries"""
        # Setup mock data
        data = MagicMock()
        data.reference_path.x.empty.return_value = False
        data.reference_path.y.empty.return_value = False
        data.reference_path.s.empty.return_value = False
        data.left_bound.empty.return_value = False
        data.right_bound.empty.return_value = False

        # Setup mocks for TwoDimensionalSpline
        mock_spline = MagicMock()
        mock_spline.getTVector.return_value = np.linspace(0, 10, 10)
        mock_left_bound = MagicMock()
        mock_right_bound = MagicMock()

        with patch('planner_modules.contouring.TwoDimensionalSpline') as mock_spline_class:
            mock_spline_class.side_effect = [mock_spline, mock_left_bound, mock_right_bound]

            # Enable road constraints
            self.contouring.add_road_constraints = True

            # Call method under test
            self.contouring.on_data_received(data, "reference_path")

            # Assertions
            self.assertEqual(self.contouring.spline, mock_spline)
            self.assertEqual(self.contouring.bound_left, mock_left_bound)
            self.assertEqual(self.contouring.bound_right, mock_right_bound)

    def test_is_data_ready(self):
        """Test is_data_ready method"""
        # Test when data is not ready
        data = MagicMock()
        data.reference_path.x.empty.return_value = True
        missing_data = ""

        result = self.contouring.is_data_ready(data, missing_data)
        self.assertFalse(result)
        self.assertIn("", missing_data)

        # Test when data is ready
        data.reference_path.x.empty.return_value = False
        missing_data = ""

        result = self.contouring.is_data_ready(data, missing_data)
        self.assertTrue(result)

    def test_is_objective_reached(self):
        """Test is_objective_reached method"""
        # Setup state and spline
        state = MagicMock()
        state.getPos.return_value = np.array([5.0, 5.0])

        mock_spline = MagicMock()
        mock_spline.parameter_length.return_value = 10.0
        mock_spline.get_point.return_value = np.array([5.5, 5.5])  # Distance is 0.71

        # Case 1: No spline
        self.contouring.spline = None
        self.assertFalse(self.contouring.is_objective_reached(state, MagicMock()))

        # Case 2: Not close enough
        self.contouring.spline = mock_spline
        with patch('planner_modules.contouring.distance', return_value=1.5):
            self.assertFalse(self.contouring.is_objective_reached(state, MagicMock()))

        # Case 3: Close enough
        with patch('planner_modules.contouring.distance', return_value=0.5):
            self.assertTrue(self.contouring.is_objective_reached(state, MagicMock()))

    def test_construct_road_constraints_from_centerline(self):
        """Test construction of road constraints from centerline"""
        # Setup
        data = MagicMock()
        data.robot_area = [MagicMock()]
        data.robot_area[0].radius = 0.4  # Half of robot width

        module_data = MagicMock()
        module_data.static_obstacles.empty.return_value = True
        module_data.static_obstacles.size.return_value = self.solver.N

        mock_spline = MagicMock()
        mock_spline.get_point.return_value = np.array([5.0, 5.0])
        mock_spline.get_orthogonal.return_value = np.array([0.0, 1.0])  # Pointing up

        self.contouring.spline = mock_spline
        self.solver.get_ego_prediction.return_value = 3.0  # s value

        # Call method under test
        self.contouring.construct_road_constraints_from_centerline(data, module_data)

        # Assertions
        self.assertEqual(module_data.static_obstacles.resize.call_count, 1)

        expected_calls = 3 * self.solver.N + self.solver.N  # 3 for main loop + 1 for reserve loop
        self.assertEqual(module_data.static_obstacles.__getitem__.call_count, expected_calls,
                         f"Expected {expected_calls} accesses but got {module_data.static_obstacles.__getitem__.call_count}")

    def test_reset(self):
        """Test reset method"""
        # Setup mock spline
        mock_spline = MagicMock()
        self.contouring.spline = mock_spline
        self.contouring.closest_segment = 5

        # Call method under test
        self.contouring.reset()

        # Assertions
        mock_spline.reset.assert_called_once()
        self.assertEqual(self.contouring.closest_segment, 0)


class TestSystemIntegration(unittest.TestCase):
    """Test integration between Contouring, ContouringConstraints, and Planner"""

    def setUp(self):
        """Set up test fixtures before each test"""
        # Create mock solver and data
        self.solver = MagicMock()
        self.solver.N = 10
        self.solver.params = MagicMock()

        # Create instances
        self.contouring = Contouring(self.solver)
        self.contouring_constraints = ContouringConstraints(self.solver)

        # Create mock planner
        self.planner = MagicMock()
        self.planner.modules = [self.contouring, self.contouring_constraints]

    @patch('utils.utils.read_config_file', return_value=CONFIG_MOCK)
    def test_planner_integration(self, mock_config):
        """Test if modules properly interact with planner"""
        # Setup mocks for planner's solve_mpc method
        data = MagicMock()
        state = MagicMock()
        module_data = MagicMock()

        # Setup data ready mocks
        with patch.object(self.contouring, 'is_data_ready', return_value=True), \
                patch.object(self.contouring_constraints, 'is_data_ready', return_value=True), \
                patch.object(self.contouring, 'update') as mock_cont_update, \
                patch.object(self.contouring_constraints, 'update') as mock_cons_update, \
                patch.object(self.contouring, 'set_parameters') as mock_cont_set_params, \
                patch.object(self.contouring_constraints, 'set_parameters') as mock_cons_set_params:

            # Mock planner.solve_mpc similar to the actual implementation
            # Update modules
            for module in self.planner.modules:
                module.update(state, data, module_data)

            # Set parameters for each prediction step
            for k in range(self.solver.N):
                for module in self.planner.modules:
                    module.set_parameters(data, module_data, k)

            # Assertions
            mock_cont_update.assert_called_once_with(state, data, module_data)
            mock_cons_update.assert_called_once_with(state, data, module_data)

            # Each module should have set_parameters called N times
            self.assertEqual(mock_cont_set_params.call_count, self.solver.N)
            self.assertEqual(mock_cons_set_params.call_count, self.solver.N)

    def test_data_sharing_between_modules(self):
        """Test if data is properly shared between modules"""
        # Setup mock data and module_data
        data = MagicMock()
        module_data = MagicMock()
        module_data.path = None
        module_data.path_width_left = None
        module_data.path_width_right = None
        state = MagicMock()

        # Setup splines in contouring
        mock_spline = MagicMock()
        mock_spline.find_closest_point.return_value = (5.0, 2)
        self.contouring.spline = mock_spline

        # Setup width splines in contouring_constraints
        mock_x = np.linspace(0, 10, 10)
        mock_y = np.sin(mock_x)
        self.contouring_constraints.width_left = CubicSpline(mock_x, mock_y)
        self.contouring_constraints.width_right = CubicSpline(mock_x, mock_y)

        # First update contouring - it should set the path in module_data
        self.contouring.update(state, data, module_data)

        # Then update contouring_constraints - it should set width splines
        self.contouring_constraints.update(state, data, module_data)

        # Assertions
        self.assertEqual(module_data.path, mock_spline)
        self.assertEqual(module_data.path_width_left, self.contouring_constraints.width_left)
        self.assertEqual(module_data.path_width_right, self.contouring_constraints.width_right)


if __name__ == '__main__':
    unittest.main()