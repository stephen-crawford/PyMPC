import unittest
from unittest import mock
from unittest.mock import MagicMock, patch
import numpy as np

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planner_modules.src.constraints.guidance_constraints import GuidanceConstraints, GuidancePlanner
from planning.src.types import Data
from utils.const import CONSTRAINT

# Manually patch CONFIG to avoid dependency issues in testing
CONFIG_MOCK = {
	"control_frequency": 10.0,
	"max_obstacles": 5,
	"guidance": {
		"num_other_halfspaces": 3,
		"longitudinal_goals": 5,
		"vertical_goals": 5
	},
	"num_paths": 2,
	"use_tmpc": False,
	"weights": {
		"reference_velocity": 5.0,
		"guidance_position": 1.0,
		"guidance_velocity": 0.5
	},
	"road": {
		"width": 3.5
	},
	"num_discs": 0.1,
	"horizon": 10
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


class TestGuidanceConstraints(unittest.TestCase):
	"""Test suite for GuidanceConstraints class"""

	def setUp(self):
		"""Set up test fixtures before each test"""
		# Create mock solver
		self.solver = MagicMock()
		self.solver.horizon = 10
		self.solver.dt = 0.1
		self.solver.params = MagicMock()

		# Apply the patch before creating the class
		patcher = patch('planner_modules.src.constraints.base_constraint.BaseConstraint.get_config_value',
						side_effect=get_mocked_config)
		self.mock_get_config = patcher.start()
		self.addCleanup(patcher.stop)

		# Patch the GuidancePlanner class
		self.planner_patcher = patch('planner_modules.src.constraints.guidance_constraints.GuidancePlanner')
		self.mock_planner_class = self.planner_patcher.start()
		self.mock_planner = MagicMock()
		self.mock_planner_class.return_value = self.mock_planner
		self.addCleanup(self.planner_patcher.stop)

		# Create guidance constraints instance
		self.guidance_constraints = GuidanceConstraints(self.solver)

	def test_initialization(self):
		"""Test proper initialization of GuidanceConstraints"""
		self.assertEqual(self.guidance_constraints.module_type, CONSTRAINT)
		self.assertEqual(self.guidance_constraints.name, "guidance_constraints")
		self.assertEqual(self.guidance_constraints.planning_frequency, CONFIG_MOCK["control_frequency"])
		self.assertEqual(self.guidance_constraints.control_frequency, CONFIG_MOCK["control_frequency"])
		self.assertEqual(self.guidance_constraints.max_obstacles, CONFIG_MOCK["max_obstacles"])
		self.assertEqual(self.guidance_constraints.num_other_halfspaces, CONFIG_MOCK["guidance"]["num_other_halfspaces"])
		self.assertEqual(self.guidance_constraints.num_paths, CONFIG_MOCK["num_paths"])

		# Check that planners were created correctly
		self.assertEqual(len(self.guidance_constraints.planners), CONFIG_MOCK["num_paths"])
		self.mock_planner_class.assert_called_with(self.solver)

	def test_create_guidance_planner(self):
		"""Test creation of guidance planning"""
		# Reset the mock to clear previous calls
		self.mock_planner_class.reset_mock()

		# Call method under test
		planner = self.guidance_constraints._create_guidance_planner()

		# Assert planning was created with correct parameters
		self.mock_planner_class.assert_called_once_with(self.solver)
		self.assertEqual(planner.longitudinal_goals, CONFIG_MOCK["guidance"]["longitudinal_goals"])
		self.assertEqual(planner.vertical_goals, CONFIG_MOCK["guidance"]["vertical_goals"])

	def test_update_no_path(self):
		"""Test update method when path is not available"""
		# Setup
		state = MagicMock()
		data = MagicMock()
		module_data = MagicMock()
		module_data.path = None

		# Call method under test
		self.guidance_constraints.update(state, data, module_data)

		# Verify that function returns early
		self.mock_planner.update.assert_not_called()

	def test_update_with_path(self):
		"""Test update method with valid path data"""
		# Setup
		state = MagicMock()
		state.get_pos.return_value = [1.0, 2.0]
		state.get.side_effect = lambda \
			key: 0.5 if key == "psi" else 3.0 if key == "v" else 2.0 if key == "spline" else None

		data = MagicMock()
		module_data = MagicMock()
		module_data.path = MagicMock()
		module_data.path_velocity = MagicMock(return_value=5.0)
		module_data.path_width_left = MagicMock(return_value=1.5)
		module_data.path_width_right = MagicMock(return_value=1.5)

		# Setup mock for static obstacles
		module_data.static_obstacles = [[MagicMock()]]
		module_data.static_obstacles[0][0].A = np.array([[1.0, 0.0], [0.0, 1.0]])
		module_data.static_obstacles[0][0].b = np.array([10.0, 10.0])

		# Setup mock for planning
		self.mock_planner.update.return_value = None
		self.mock_planner.trajectories = [MagicMock()]
		self.mock_planner.succeeded.return_value = True

		# Mock optimize method
		self.guidance_constraints.optimize = MagicMock(return_value=1)
		self.guidance_constraints.set_goals = MagicMock()

		# Call method under test
		self.guidance_constraints.update(state, data, module_data)


		# Check planning setup
		self.mock_planner.set_start.assert_called_with([1.0, 2.0], 0.5, 3.0)
		self.mock_planner.set_reference_velocity.assert_called_with(5.0)
		self.mock_planner.load_static_obstacles.assert_called()
		self.mock_planner.update.assert_called()

		# Verify optimize was called
		self.guidance_constraints.optimize.assert_called_with(state, data, module_data)

	def test_set_goals(self):
		"""Test setting goals for guidance planning"""
		# Setup
		state = MagicMock()
		state.get.return_value = 2.0  # spline

		module_data = MagicMock()
		module_data.path = MagicMock()
		module_data.path.get_point.return_value = np.array([3.0, 4.0])
		module_data.path.get_orthogonal.return_value = np.array([0.0, 1.0])
		module_data.path.get_path_angle.return_value = 0.0

		module_data.path_velocity = MagicMock(return_value=5.0)
		module_data.path_width_left = MagicMock(return_value=1.5)
		module_data.path_width_right = MagicMock(return_value=1.5)

		planner = MagicMock()
		planner.horizon = 10
		planner.space_time_point_num_states.return_value = 3

		# Call method under test
		self.guidance_constraints.set_goals(state, module_data, planner)

		# Verify planning.set_goals was called with correct parameters
		planner.set_goals.assert_called_once()
		# Check first argument is a list of goals (tuples of position and cost)
		goals = planner.set_goals.call_args[0][0]
		self.assertIsInstance(goals, list)

		# Check first goal format
		self.assertIsInstance(goals[0], tuple)
		self.assertEqual(len(goals[0]), 2)  # (position, cost)

		# Check position format
		position = goals[0][0]
		self.assertIsInstance(position, list)
		self.assertEqual(len(position), 3)  # [x, y, angle]

	def test_define_parameters(self):
		"""Test defining parameters for the solver"""
		# Setup
		params = MagicMock()

		# Call method under test
		self.guidance_constraints.define_parameters(params)

		# Verify parameters were defined
		self.assertEqual(params.add.call_count, 4 * self.solver.horizon)  # 4 params per timestep

		# Check parameter names
		expected_calls = []
		for k in range(self.solver.horizon):
			expected_calls.extend([
				mock.call(f"guidance_x_{k}"),
				mock.call(f"guidance_y_{k}"),
				mock.call(f"guidance_vx_{k}"),
				mock.call(f"guidance_vy_{k}")
			])

		params.add.assert_has_calls(expected_calls, any_order=True)

	def test_set_parameters_k0(self):
		"""Test setting parameter values for the solver"""
		# Setup
		data = MagicMock()
		module_data = MagicMock()

		# Create a mock trajectory
		trajectory = MagicMock()
		trajectory_spline = MagicMock()
		trajectory_spline.get_point.return_value = [10.0, 20.0]
		trajectory_spline.get_velocity.return_value = [2.0, 3.0]
		trajectory.spline.get_trajectory.return_value = trajectory_spline

		self.guidance_constraints.trajectories = [trajectory, trajectory, trajectory, trajectory]

		# Set selected trajectory
		self.guidance_constraints.selected_trajectory = trajectory

		self.guidance_constraints.set_parameters(self.solver.params, data, module_data, 0)
		assert (self.solver.params.set_parameter.call_count == 16) # 4 per trajectory or 4 per horizon timestep whichever is smaller


	def test_optimize(self):
		"""Test optimization to find best trajectory"""
		# Setup
		state = MagicMock()
		state.get_pos.return_value = [0.0, 0.0]

		data = MagicMock()
		module_data = MagicMock()

		# Create mock trajectories
		trajectory1 = MagicMock()
		trajectory1.cost = 10.0
		trajectory1_spline = MagicMock()
		trajectory1_spline.get_point.return_value = [1.0, 1.0]
		trajectory1.spline.get_trajectory.return_value = trajectory1_spline

		trajectory2 = MagicMock()
		trajectory2.cost = 5.0
		trajectory2_spline = MagicMock()
		trajectory2_spline.get_point.return_value = [2.0, 2.0]
		trajectory2.spline.get_trajectory.return_value = trajectory2_spline

		self.guidance_constraints.trajectories = [trajectory1, trajectory2]

		# Mock planners
		planner1 = MagicMock()
		planner1.succeeded.return_value = True
		planner2 = MagicMock()
		planner2.succeeded.return_value = True
		self.guidance_constraints.planners = [planner1, planner2]

		# Call method under test
		result = self.guidance_constraints.optimize(state, data, module_data)

		# Verify result and selected trajectory
		self.assertEqual(result, 1)
		self.assertEqual(self.guidance_constraints.selected_trajectory, trajectory2)
		self.assertEqual(self.guidance_constraints.best_planner_index, 1)

	def test_optimize_no_successful_trajectories(self):
		"""Test optimization when no successful trajectories are available"""
		# Setup
		state = MagicMock()
		data = MagicMock()
		module_data = MagicMock()

		# Mock planners with no successful solutions
		planner = MagicMock()
		planner.succeeded.return_value = False
		self.guidance_constraints.planners = [planner]

		# Call method under test
		result = self.guidance_constraints.optimize(state, data, module_data)

		# Verify result
		self.assertEqual(result, 0)
		self.assertIsNone(self.guidance_constraints.selected_trajectory)

	def test_calculate_trajectory_cost(self):
		"""Test calculation of trajectory cost"""
		# Setup
		state = MagicMock()
		state.get_pos.return_value = [0.0, 0.0]

		# Create mock trajectory
		trajectory = MagicMock()
		trajectory.cost = 10
		trajectory_spline = MagicMock()
		trajectory_spline.get_point.return_value = [3.0, 4.0]  # Distance of 5 from origin
		trajectory.spline.get_trajectory.return_value = trajectory_spline
		trajectory.previously_selected = False

		# Call method under test
		cost = self.guidance_constraints.calculate_trajectory_cost(trajectory, state)

		# Verify cost calculation (base cost + distance * 0.5)
		self.assertEqual(cost, 12.5)

		# Test with previously selected trajectory
		trajectory.previously_selected = True
		cost = self.guidance_constraints.calculate_trajectory_cost(trajectory, state)

		# Verify cost calculation with discount for previously selected trajectory
		self.assertEqual(cost, (10.0 + 5 * 0.5) * 0.8)

	def test_lower_and_upper_bounds(self):
		"""Test constraint bounds"""
		lower_bounds = self.guidance_constraints.lower_bounds()
		upper_bounds = self.guidance_constraints.upper_bounds()

		self.assertEqual(len(lower_bounds), 2)
		self.assertEqual(len(upper_bounds), 2)
		self.assertEqual(lower_bounds, [-float('inf'), -float('inf')])
		self.assertEqual(upper_bounds, [0.0, 0.0])

	def test_calculate_constraints(self):
		"""Test calculation of guidance constraints"""
		# Setup
		params = MagicMock()
		params.get.side_effect = lambda name: {
			"guidance_x_0": 10.0,
			"guidance_y_0": 20.0,
			"guidance_vx_0": 2.0,
			"guidance_vy_0": 3.0
		}[name]

		settings = MagicMock()
		model = MagicMock()
		model.get.side_effect = lambda name: {
			"x": 11.0,
			"y": 22.0,
			"vx": 2.5,
			"vy": 3.5,
			"slack": 0.1
		}[name]
		model.vars = ["x", "y", "vx", "vy", "slack"]
		settings.model = model

		# Call method under test
		constraints = self.guidance_constraints.calculate_constraints(params, settings, 0)

		# Verify constraints
		self.assertEqual(len(constraints), 2)

		# Verify position constraint
		pos_weight = CONFIG_MOCK["weights"]["guidance_position"]
		pos_error_x = 11.0 - 10.0
		pos_error_y = 22.0 - 20.0
		expected_pos_constraint = pos_weight * (pos_error_x ** 2 + pos_error_y ** 2) - 0.1
		self.assertEqual(constraints[0], expected_pos_constraint)

		# Verify velocity constraint
		vel_weight = CONFIG_MOCK["weights"]["guidance_velocity"]
		vel_error_x = 2.5 - 2.0
		vel_error_y = 3.5 - 3.0
		expected_vel_constraint = vel_weight * (vel_error_x ** 2 + vel_error_y ** 2) - 0.1
		self.assertEqual(constraints[1], expected_vel_constraint)

	def test_is_data_ready(self):
		"""Test is_data_ready method"""
		# Test when data is not ready
		data = Data()

		result = self.guidance_constraints.is_data_ready(data)
		self.assertFalse(result)

		# Test when data is ready

		data.set("path", MagicMock())

		result = self.guidance_constraints.is_data_ready(data)
		self.assertTrue(result)

	def test_on_data_received(self):
		"""Test on_data_received method for handling dynamic obstacles"""
		# Setup
		data = MagicMock()
		obstacle = MagicMock()
		obstacle.index = 1
		obstacle.position = [5.0, 6.0]
		obstacle.radius = 1.0

		prediction_mode = MagicMock()
		prediction_mode.position = [7.0, 8.0]
		obstacle.prediction.modes = [[prediction_mode]]

		data.dynamic_obstacles = [obstacle]
		data.robot_area = [MagicMock()]
		data.robot_area[0].radius = 0.5

		# Call method under test
		self.guidance_constraints.on_data_received(data, "dynamic obstacles")

		# Verify planning method calls
		expected_obstacles = [
			(1, [[5.0, 6.0], [7.0, 8.0]], 1.5)  # index, positions, combined radius
		]

		self.mock_planner.load_obstacles.assert_called_with(expected_obstacles)

	def test_reset(self):
		"""Test reset method"""
		# Setup
		self.guidance_constraints.trajectories = [MagicMock()]
		self.guidance_constraints.selected_trajectory = MagicMock()

		# Call method under test
		self.guidance_constraints.reset()

		# Verify reset behavior
		self.assertEqual(self.guidance_constraints.trajectories, [])
		self.assertIsNone(self.guidance_constraints.selected_trajectory)
		self.mock_planner.reset.assert_called()


class TestGuidancePlanner(unittest.TestCase):
	"""Test suite for GuidancePlanner class"""

	def setUp(self):
		"""Set up test fixtures before each test"""
		# Create mock solver
		self.solver = MagicMock()
		self.solver.horizon = 10
		self.solver.dt = 0.1

		# Create planning instance directly (no need to patch)
		with patch('planner_modules.src.constraints.guidance_constraints.Planner'):
			self.planner = GuidancePlanner(self.solver)

	def test_initialization(self):
		"""Test proper initialization of GuidancePlanner"""
		self.assertEqual(self.planner._planning_frequency, 10.0)
		self.assertEqual(self.planner._dt, 0.1)
		self.assertEqual(self.planner.current_pos.tolist(), [0.0, 0.0])
		self.assertEqual(self.planner.current_psi, 0.0)
		self.assertEqual(self.planner.current_v, 0.0)
		self.assertEqual(self.planner._reference_velocity, 0.0)
		self.assertIsNone(self.planner.reference_path)
		self.assertEqual(self.planner.start_s, 0.0)
		self.assertEqual(self.planner.width_left, 0.0)
		self.assertEqual(self.planner.width_right, 0.0)
		self.assertEqual(self.planner.static_obstacles, [])
		self.assertEqual(self.planner.dynamic_obstacles, [])
		self.assertEqual(self.planner.goals, [])
		self.assertEqual(self.planner.trajectories, [])
		self.assertEqual(self.planner.selected_trajectory_id, -1)
		self.assertEqual(self.planner.horizon, 10)
		self.assertEqual(self.planner.longitudinal_goals, 5)
		self.assertEqual(self.planner.vertical_goals, 5)

	def test_set_start(self):
		"""Test setting start state"""
		self.planner.set_start([1.0, 2.0], 0.5, 3.0)
		np.testing.assert_array_equal(self.planner.current_pos, np.array([1.0, 2.0]))
		self.assertEqual(self.planner.current_psi, 0.5)
		self.assertEqual(self.planner.current_v, 3.0)

	def test_set_reference_velocity(self):
		"""Test setting reference velocity"""
		self.planner.set_reference_velocity(5.0)
		self.assertEqual(self.planner._reference_velocity, 5.0)

	def test_load_static_obstacles(self):
		"""Test loading static obstacles"""
		halfspaces = [
			(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([10.0, 10.0]))
		]
		self.planner.load_static_obstacles(halfspaces)
		self.assertEqual(self.planner.static_obstacles, halfspaces)

	def test_load_obstacles(self):
		"""Test loading dynamic obstacles"""
		obstacles = [
			(1, [[5.0, 6.0], [7.0, 8.0]], 1.5)
		]
		self.planner.load_obstacles(obstacles)
		self.assertEqual(self.planner.dynamic_obstacles, obstacles)

	def test_set_goals(self):
		"""Test setting goals"""
		goals = [
			([1.0, 2.0, 0.0], 5.0),
			([3.0, 4.0, 0.5], 3.0)
		]
		self.planner.set_goals(goals)
		self.assertEqual(self.planner.goals, goals)

	def test_load_reference_path(self):
		"""Test loading reference path"""
		path = MagicMock()
		self.planner.load_reference_path(2.0, path, 1.5, 1.5)
		self.assertEqual(self.planner.reference_path, path)
		self.assertEqual(self.planner.start_s, 2.0)
		self.assertEqual(self.planner.width_left, 1.5)
		self.assertEqual(self.planner.width_right, 1.5)

	@patch('planner_modules.src.constraints.guidance_constraints.LOG_DEBUG')
	def test_update_no_goals(self, mock_log_debug):
		"""Test update when no goals are set"""
		self.planner.update()
		mock_log_debug.assert_called_with("No goals set, cannot update guidance")
		self.assertFalse(self.planner._success)
		self.assertEqual(self.planner.trajectories, [])

	@patch('planner_modules.src.constraints.guidance_constraints.LOG_DEBUG')
	def test_update_with_goals(self, mock_log_debug):
		"""Test update with valid goals"""
		# Setup goals
		self.planner.goals = [
			([1.0, 2.0, 0.0], 5.0),
			([3.0, 4.0, 0.5], 3.0)
		]

		# Mock trajectory generation
		with patch.object(self.planner, 'generate_trajectories') as mock_generate:
			mock_generate.side_effect = lambda: setattr(self.planner, 'trajectories', [MagicMock()])

			# Call method under test
			self.planner.update()

			# Verify trajectory generation was called
			mock_generate.assert_called_once()
			self.assertTrue(self.planner._success)

	def test_generate_trajectories(self):
		"""Test generation of trajectories"""
		# Setup goals
		self.planner.goals = [
			([1.0, 2.0, 0.0], 5.0),
			([3.0, 4.0, 0.5], 3.0)
		]

		# Mock trajectory creation
		with patch.object(self.planner, 'create_trajectory_to_goal') as mock_create:
			mock_create.return_value = MagicMock()

			# Call method under test
			self.planner.generate_trajectories()

			# Verify trajectory creation
			self.assertEqual(mock_create.call_count, 2)
			self.assertEqual(len(self.planner.trajectories), 2)

	def test_create_trajectory_to_goal(self):
		"""Test creation of trajectory to a goal"""
		# Setup
		self.planner.current_pos = np.array([0.0, 0.0])
		self.planner.current_psi = 0.0
		self.planner._reference_velocity = 5.0

		# Call method under test
		trajectory = self.planner.create_trajectory_to_goal([3.0, 4.0, 0.0], 1, 5.0)

		# Verify trajectory properties
		self.assertEqual(trajectory.topology_class, 1)
		self.assertEqual(trajectory.color, 1)
		self.assertFalse(trajectory.previously_selected)
		self.assertEqual(trajectory.cost, 5.0)

		# Test trajectory functions
		traj_spline = trajectory.spline.get_trajectory()
		np.testing.assert_array_almost_equal(traj_spline.get_point(0.0), np.array([0.0, 0.0]))
		np.testing.assert_array_almost_equal(traj_spline.get_point(0.5),
											 np.array([1.5, 2.0]))  # 50% along path to [3,4]
		np.testing.assert_array_almost_equal(traj_spline.get_velocity(0.5), np.array([3.0, 4.0]) / 5.0 * 5.0)

	def test_succeeded(self):
		"""Test succeeded method"""
		self.planner._success = False
		self.assertFalse(self.planner.succeeded())

		self.planner._success = True
		self.assertTrue(self.planner.succeeded())

	def test_number_of_guidance_trajectories(self):
		"""Test number_of_guidance_trajectories method"""
		self.planner.trajectories = []
		self.assertEqual(self.planner.number_of_guidance_trajectories(), 0)

		self.planner.trajectories = [MagicMock(), MagicMock()]
		self.assertEqual(self.planner.number_of_guidance_trajectories(), 2)

	def test_get_guidance_trajectory(self):
		"""Test getting a guidance trajectory"""
		# Setup
		trajectory1 = MagicMock()
		trajectory1.topology_class = 1
		self.planner.trajectories = [trajectory1]

		# Valid index
		result = self.planner.get_guidance_trajectory(0)
		self.assertEqual(result, trajectory1)

		# Invalid index
		result = self.planner.get_guidance_trajectory(1)
		self.assertEqual(result.topology_class, -1)
		self.assertEqual(result.color, -1)
		self.assertFalse(result.previously_selected)

	def test_space_time_point_num_states(self):
		"""Test space_time_point_num_states method"""
		# For 2D goals
		self.planner.goals = [
			([1.0, 2.0], 5.0)
		]
		self.assertEqual(self.planner.space_time_point_num_states(), 2)

		# For 3D goals
		self.planner.goals = [
			([1.0, 2.0, 0.5], 5.0)
		]
		self.assertEqual(self.planner.space_time_point_num_states(), 3)

	def test_override_selected_trajectory(self):
		"""Test overriding selected trajectory"""
		self.planner.selected_trajectory_id = -1
		self.planner.override_selected_trajectory(2)
		self.assertEqual(self.planner.selected_trajectory_id, 2)

	def test_reset(self):
		"""Test reset method"""
		# Setup
		self.planner.trajectories = [MagicMock()]
		self.planner.selected_trajectory_id = 1
		self.planner._success = True
		self.planner.goals = [MagicMock()]

		# Call method under test
		self.planner.reset()

		# Verify reset behavior
		self.assertEqual(self.planner.trajectories, [])
		self.assertEqual(self.planner.selected_trajectory_id, -1)
		self.assertFalse(self.planner._success)
		self.assertEqual(self.planner.goals, [])

class TestSystemIntegration(unittest.TestCase):
	"""Test integration between GaussianConstraints and Planner"""

	def setUp(self):
		"""Set up test fixtures before each test"""
		# Create mock solver
		self.solver = MagicMock()
		self.solver.horizon = 10
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
			from planner_modules.src.constraints.guidance_constraints import GuidanceConstraints
			self.guidance_constraints = GuidanceConstraints(self.solver)

		# Create mock planning
		self.planner = MagicMock()
		self.planner.modules = [self.guidance_constraints]

		# Add create_visualization_publisher mock
		self.guidance_constraints.create_visualization_publisher = MagicMock()

	@patch('utils.utils.LOG_DEBUG')
	def test_planner_integration(self, mock_log_debug):
		"""Test if module properly interacts with planning"""
		# Setup mocks for planning's solve_mpc method
		data = MagicMock()
		state = MagicMock()
		module_data = MagicMock()

		# Setup data ready mocks
		with patch.object(self.guidance_constraints, 'is_data_ready', return_value=True), \
				patch.object(self.guidance_constraints, 'update') as mock_update, \
				patch.object(self.guidance_constraints, 'set_parameters') as mock_set_params:

			# Mock planning.solve_mpc similar to the actual implementation
			# Update modules
			for module in self.planner.modules:
				module.update(state, data, module_data)

			# Set parameters for each prediction step
			for k in range(self.solver.horizon):
				for module in self.planner.modules:
					module.set_parameters(data, module_data, k)

			# Assertions
			mock_update.assert_called_once_with(state, data, module_data)

			# Module should have set_parameters called N times
			self.assertEqual(mock_set_params.call_count, self.solver.horizon)

if __name__ == '__main__':
	unittest.main()