import unittest
import numpy as np
from unittest.mock import MagicMock, patch, call, ANY
import numpy.testing as npt

from planner_modules.base_constraint import BaseConstraint
# Import modules to test
from utils.const import CONSTRAINT

# Manually patch CONFIG to avoid dependency issues in testing
CONFIG_MOCK = {
    "contouring": {
        "num_segments": 10
    },
    "decomp": {
        "range": 10.0,
        "max_constraints": 8
    },
    "N": 10,
    "n_discs": 1,
    "visualization": {
        "draw_every": 1
    },
    "debug_visuals": False
}

with patch('utils.utils.read_config_file', return_value=CONFIG_MOCK):
    from planner_modules.decomp_constraints import DecompConstraints
    from planner_modules.ellipsoid_constraints import EllipsoidConstraints
    from planner_modules.base_constraint import BaseConstraint

# Apply the patch before any imports
@patch('utils.utils.read_config_file', return_value=CONFIG_MOCK)
class TestDecompConstraints(unittest.TestCase):
    """Test suite for DecompConstraints class"""

    @patch.object(BaseConstraint, 'get_config_value')
    def setUp(self, mock_get_config_value):
        """Set up test fixtures before each test"""
        # Create mock solver
        self.solver = MagicMock()
        self.solver.N = 10
        self.solver.params = MagicMock()
        self.solver.dt = 0.1

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
        self.patcher = patch('utils.utils.EllipsoidDecomp2D', return_value=self.mock_decomp_util)
        self.mock_decomp_class = self.patcher.start()

        self.decomp_constraints = DecompConstraints(self.solver)

    def tearDown(self):
        self.patcher.stop()

    def test_initialization(self, mock_config=CONFIG_MOCK):
        """Test proper initialization of DecompConstraints"""
        self.assertEqual(self.decomp_constraints.module_type, CONSTRAINT)
        self.assertEqual(self.decomp_constraints.name, "decomp_constraints")
        self.assertEqual(self.decomp_constraints.get_num_segments, CONFIG_MOCK["contouring"]["num_segments"])
        self.assertEqual(self.decomp_constraints.range, CONFIG_MOCK["decomp"]["range"])
        self.assertEqual(self.decomp_constraints._max_constraints, CONFIG_MOCK["decomp"]["max_constraints"])
        self.assertEqual(len(self.decomp_constraints.occ_pos), 0)

    @patch('planner_modules.decomp_constraints.PROFILE_SCOPE')
    def test_update(self, mock_profile_scope, mock_config=CONFIG_MOCK):
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

    def test_get_occupied_grid_cells(self, mock_config=CONFIG_MOCK):
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

    def test_set_parameters_k0(self, mock_config=CONFIG_MOCK):
        """Test set_parameters method for k=0 (dummies)"""
        # Setup
        k = 0
        data = MagicMock()
        data.robot_area = [MagicMock()]
        data.robot_area[0].offset = np.array([0.5, 0.3])
        module_data = MagicMock()

        # Patch set_solver_parameter
        with patch('solver.solver_interface.set_solver_parameter') as mock_set_param:
            self.decomp_constraints.set_parameters(data, module_data, k)

            # Assert that the offset call was made — but use ANY for the array
            expected_calls = [
                call(self.solver.params, "ego_disc_offset", ANY, k, index=0, settings=CONFIG_MOCK),
            ]

            for i in range(CONFIG_MOCK["decomp"]["max_constraints"]):
                expected_calls.extend([
                    call(self.solver.params, "decomp_a1", 0.0, k, index=i, settings=CONFIG_MOCK),
                    call(self.solver.params, "decomp_a2", 0.0, k, index=i, settings=CONFIG_MOCK),
                    call(self.solver.params, "decomp_b", 100.0, k, index=i, settings=CONFIG_MOCK),
                ])

            mock_set_param.assert_has_calls(expected_calls)

            # Manually check the numpy array argument
            actual_offset_call = mock_set_param.call_args_list[0]
            np.testing.assert_array_equal(actual_offset_call.args[2], np.array([0.5, 0.3]))

            self.assertEqual(mock_set_param.call_count, 1 + 3 * CONFIG_MOCK["decomp"]["max_constraints"])

    def test_set_parameters_k1(self, mock_config=CONFIG_MOCK):
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

        # Patch set_solver_parameter
        with patch('solver.solver_interface.set_solver_parameter') as mock_set_param:
            # Call method under test
            self.decomp_constraints.set_parameters(data, module_data, k)

            # Assertions
            expected_calls = [
                # Ego disc offset call
                call(self.solver.params, "ego_disc_offset", ANY, k, index=0, settings=CONFIG_MOCK),
            ]

            # Add constraint calls
            for i in range(CONFIG_MOCK["decomp"]["max_constraints"]):
                a1_val = 1.1 if i == 0 else 0.0
                a2_val = 2.2 if i == 0 else 0.0
                b_val = 3.3 if i == 0 else 0.0

                expected_calls.extend([
                    call(self.solver.params, "decomp_a1", a1_val, k, index=i, settings=CONFIG_MOCK),
                    call(self.solver.params, "decomp_a2", a2_val, k, index=i, settings=CONFIG_MOCK),
                    call(self.solver.params, "decomp_b", b_val, k, index=i, settings=CONFIG_MOCK),
                ])

            mock_set_param.assert_has_calls(expected_calls)
            self.assertEqual(mock_set_param.call_count, 1 + 3 * CONFIG_MOCK["decomp"]["max_constraints"])

    def test_is_data_ready(self, mock_config=CONFIG_MOCK):
        """Test is_data_ready method"""
        # Test when data is not ready
        data = MagicMock()
        data.costmap = None
        missing_data = ""

        result = self.decomp_constraints.is_data_ready(data, missing_data)
        self.assertFalse(result)

        # Test when data is ready
        data.costmap = MagicMock()
        missing_data = ""

        result = self.decomp_constraints.is_data_ready(data, missing_data)
        self.assertTrue(result)

    @patch('planner_modules.decomp_constraints.ROSLine')
    @patch('planner_modules.decomp_constraints.ROSPointMarker')
    def test_visualize(self, mock_point_marker, mock_line, mock_config=CONFIG_MOCK):
        """Test visualize method"""
        # Setup
        data = MagicMock()
        module_data = MagicMock()

        # Mock line and point marker
        mock_line_instance = MagicMock()
        mock_line.return_value = mock_line_instance
        mock_line_instance.add_new_line.return_value = MagicMock()

        mock_point_instance = MagicMock()
        mock_point_marker.return_value = mock_point_instance
        mock_point_instance.get_new_point_marker.return_value = MagicMock()

        # Setup polyhedrons
        vertices = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
        poly = MagicMock()
        poly.vertices = vertices
        self.mock_decomp_util.get_polyhedrons.return_value = [poly]

        # Enable debug visuals
        with patch.dict(CONFIG_MOCK, {"debug_visuals": True}):
            # Call method under test
            self.decomp_constraints.visualize(data, module_data)

            # Assertions
            mock_line.assert_called_once_with(self.decomp_constraints.name + "/free_space")
            mock_point_marker.assert_called_with(self.decomp_constraints.name + "/map")

        # Test visualization with debug visuals disabled
        mock_line.reset_mock()
        mock_point_marker.reset_mock()

        with patch.dict(CONFIG_MOCK, {"debug_visuals": False}):
            self.decomp_constraints.visualize(data, module_data)

            # Should not publish any visualizations
            mock_line_instance.publish.assert_not_called()
            mock_point_instance.publish.assert_not_called()

    def test_project_to_safety(self, mock_config=CONFIG_MOCK):
        """Test project_to_safety method"""
        # Setup
        pos = np.array([1.0, 2.0])

        # Test with empty occ_pos
        self.decomp_constraints.occ_pos = []
        self.decomp_constraints.project_to_safety(pos)

        # Test with non-empty occ_pos
        self.decomp_constraints.occ_pos = [np.array([3.0, 4.0])]
        self.decomp_constraints.project_to_safety(pos)
        # This is a placeholder test since the method is not implemented
        # We're just verifying it doesn't crash


@patch('utils.utils.read_config_file', return_value=CONFIG_MOCK)
class TestSystemIntegration(unittest.TestCase):
    """Test integration between DecompConstraints and Planner"""

    def setUp(self, mock_config=CONFIG_MOCK):
        """Set up test fixtures before each test"""
        # Create mock solver and data
        self.solver = MagicMock()
        self.solver.N = 10
        self.solver.params = MagicMock()
        self.solver.dt = 0.1

        # Create mock for EllipsoidDecomp2D
        self.mock_decomp_util = MagicMock()

        # Patch the EllipsoidDecomp2D class
        self.patcher = patch('utils.utils.EllipsoidDecomp2D', return_value=self.mock_decomp_util)
        self.mock_decomp_class = self.patcher.start()

        # Create instance of the class under test
        with patch('utils.utils.read_config_file', return_value=CONFIG_MOCK):
            from planner_modules.decomp_constraints import DecompConstraints
            self.decomp_constraints = DecompConstraints(self.solver)

        # Create mock planner
        self.planner = MagicMock()
        self.planner.modules = [self.decomp_constraints]

    def tearDown(self):
        self.patcher.stop()

    def test_planner_integration(self, mock_config=CONFIG_MOCK):
        """Test if module properly interacts with planner"""
        # Setup mocks for planner's solve_mpc method
        data = MagicMock()
        state = MagicMock()
        module_data = MagicMock()

        # Setup data ready mocks
        with patch.object(self.decomp_constraints, 'is_data_ready', return_value=True), \
                patch.object(self.decomp_constraints, 'update') as mock_update, \
                patch.object(self.decomp_constraints, 'set_parameters') as mock_set_params:

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