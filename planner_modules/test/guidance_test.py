import time
import unittest
import numpy as np
from unittest.mock import MagicMock, patch, call
from math import atan2

# Import modules to test
from utils.const import CONSTRAINT, OBJECTIVE

# Manually patch CONFIG to avoid dependency issues in testing
CONFIG_MOCK = {
	"control_frequency": 10.0,
	"debug_visuals": False,
	"shift_previous_solution_forward": True,
	"enable_output": True,
	"max_obstacles": 20,
	"n_discs": 2,
	"road": {
		"width": 8.0
	},
	"weights": {
		"reference_velocity": 5.0
	},
	"t-mpc": {
		"use_t-mpc+=1": True,
		"enable_constraints": True,
		"warmstart_with_mpc_solution": True,
		"highlight_selected": True
	},
	"global_guidance": {
		"n_paths": 3,
		"N": 10,
		"longitudinal_goals": 5,
		"vertical_goals": 5,
		"selection_weight_consistency": 0.8
	},
	"guidance":
		{
		"n_other_halfspaces": 2
	}
}

# Patch the read_config_file function
with patch('utils.utils.read_config_file', return_value=CONFIG_MOCK):
	from planner_modules.guidance_constraints import GuidanceConstraints, GlobalGuidance
	from planner_modules.base_constraint import BaseConstraint


class TestGuidanceConstraints(unittest.TestCase):
	"""Test suite for GuidanceConstraints class"""

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
		self.solver.dt = 0.1
		self.solver.copy = MagicMock(return_value=MagicMock())  # Returns a new mock

		self.config_attr_patcher = patch('planner_modules.base_constraint.CONFIG', CONFIG_MOCK)
		self.config_attr_patcher.start()
		self.addCleanup(self.config_attr_patcher.stop)

		# Apply the patch before creating the class
		patcher = patch('planner_modules.base_constraint.BaseConstraint.get_config_value',
						side_effect=self.get_mocked_config)
		self.mock_get_config = patcher.start()
		self.addCleanup(patcher.stop)

		# Mock the GlobalGuidance class
		self.global_guidance_patcher = patch('planner_modules.guidance_constraints.GlobalGuidance')
		self.mock_global_guidance = self.global_guidance_patcher.start()

		# Create a mock instance of GlobalGuidance to be returned by constructor
		self.mock_global_guidance_instance = MagicMock()
		self.mock_global_guidance_instance.get_config.return_value = MagicMock(n_paths=3, N=10, longitudinal_goals=5,
		                                                                       vertical_goals=5,
		                                                                       selection_weight_consistency=0.8)
		self.mock_global_guidance_instance.get_guidance_trajectory.return_value = MagicMock(topology_class=1, color=1,
		                                                                                    previously_selected=False,
		                                                                                    spline=MagicMock())
		self.mock_global_guidance_instance.succeeded.return_value = True
		self.mock_global_guidance_instance.number_of_guidance_trajectories.return_value = 3
		self.mock_global_guidance.return_value = self.mock_global_guidance_instance

		# Import after patching
		from planner_modules.guidance_constraints import GuidanceConstraints
		self.guidance_constraints = GuidanceConstraints(self.solver)



	def tearDown(self):
		"""Clean up after each test"""
		self.global_guidance_patcher.stop()

	def test_initialization(self):
		"""Test proper initialization of GuidanceConstraints"""
		self.assertEqual(self.guidance_constraints.module_type, CONSTRAINT)
		self.assertEqual(self.guidance_constraints.name, "guidance_constraints")
		self.assertEqual(len(self.guidance_constraints.planners), 4)  # 3 path planners + 1 for T-MPC+=1
		self.assertEqual(self.guidance_constraints._planning_time, 0.1)  # 1/control_frequency

		# Check if global guidance was properly initialized
		self.mock_global_guidance.assert_called_once()
		self.mock_global_guidance_instance.set_planning_frequency.assert_called_once_with(
			CONFIG_MOCK["control_frequency"])

		# Check planner initialization
		for i, planner in enumerate(self.guidance_constraints.planners[:3]):
			self.assertEqual(planner.id, i)
			self.assertFalse(planner.is_original_planner)
			self.assertFalse(planner.taken)
			self.assertFalse(planner.existing_guidance)
			self.assertFalse(planner.disabled)

		# Check T-MPC+=1 planner (should be the last one)
		tmpc_planner = self.guidance_constraints.planners[-1]
		self.assertEqual(tmpc_planner.id, 3)
		self.assertTrue(tmpc_planner.is_original_planner)

	def test_update_with_path_data(self):
		"""Test update method with valid path data"""
		# Setup
		state = MagicMock()
		state.get_pos.return_value = np.array([10.0, 20.0])
		state.get.side_effect = lambda \
			key: 1.5 if key == "psi" else 5.0 if key == "v" else 30.0 if key == "spline" else 0.0

		data = MagicMock()

		module_data = MagicMock()
		module_data.path = MagicMock()
		module_data.path_velocity = MagicMock(return_value=5.0)
		module_data.path_width_left = MagicMock(return_value=2.0)
		module_data.path_width_right = MagicMock(return_value=2.0)
		module_data.static_obstacles = [(MagicMock(A=np.array([[1, 0], [0, 1]]), b=np.array([5, 5])))]

		# Call method under test
		self.guidance_constraints.update(state, data, module_data)

		# Assertions
		self.mock_global_guidance_instance.set_start.assert_called_once_with(state.get_pos(), state.get("psi"),
		                                                                     state.get("v"))
		self.mock_global_guidance_instance.set_reference_velocity.assert_called_once_with(5.0)
		self.mock_global_guidance_instance.load_static_obstacles.assert_called_once()
		self.mock_global_guidance_instance.update.assert_called_once()

	def test_update_no_path_data(self):
		"""Test update method with no path data"""
		# Setup
		state = MagicMock()
		data = MagicMock()
		module_data = MagicMock()
		module_data.path = None

		# Call method under test
		self.guidance_constraints.update(state, data, module_data)

		# Assertions - nothing should happen with global guidance
		self.mock_global_guidance_instance.set_start.assert_not_called()
		self.mock_global_guidance_instance.update.assert_not_called()

	def test_set_goals(self):
		"""Test set_goals method"""
		# Setup
		state = MagicMock()
		state.get.side_effect = lambda key: 10.0 if key == "spline" else 0.0

		module_data = MagicMock()
		module_data.path = MagicMock()
		module_data.path.get_point.return_value = np.array([5.0, 5.0])
		module_data.path.get_orthogonal.return_value = np.array([0.0, 1.0])
		module_data.path.get_path_angle.return_value = 0.0

		module_data.path_velocity = MagicMock(return_value=2.0)
		module_data.path_width_left = MagicMock(return_value=2.0)
		module_data.path_width_right = MagicMock(return_value=2.0)

		# Configure solver and global guidance for test
		self.guidance_constraints.solver.dt = 0.1
		self.mock_global_guidance_instance.get_config.return_value = MagicMock(N=10, longitudinal_goals=3,
		                                                                       vertical_goals=3, n_paths=3)
		self.mock_global_guidance_instance.space_time_point_num_states.return_value = 2

		# Call method under test
		self.guidance_constraints.set_goals(state, module_data)

		# Assertions
		self.mock_global_guidance_instance.set_goals.assert_called_once()

		# Check that goals were created properly - we can't check exact values but can verify call was made
		goals_arg = self.mock_global_guidance_instance.set_goals.call_args[0][0]
		self.assertTrue(isinstance(goals_arg, list))
		self.assertTrue(len(goals_arg) > 0)

	def test_set_goals_no_path_data(self):
		"""Test set_goals method with no path velocity data"""
		# Setup
		state = MagicMock()
		state.get.side_effect = lambda key: 10.0 if key == "spline" else 0.0

		module_data = MagicMock()
		module_data.path = MagicMock()
		module_data.path_velocity = None
		module_data.path_width_left = None
		module_data.path_width_right = None

		# Call method under test
		self.guidance_constraints.set_goals(state, module_data)

		# Assertions - should fall back to loading reference path directly
		self.mock_global_guidance_instance.load_reference_path.assert_called_once()
		self.mock_global_guidance_instance.set_goals.assert_not_called()

	@patch('time.time', return_value=10.0)
	def test_optimize(self, mock_time):
		"""Test optimize method with successful guidance"""
		# Setup
		state = MagicMock()
		data = MagicMock()
		data.planning_start_time = 9.95  # 0.05 seconds used already
		module_data = MagicMock()

		# Configure planners
		for i, planner in enumerate(self.guidance_constraints.planners):
			planner.local_solver = MagicMock()
			planner.local_solver._info = MagicMock(pobj=100.0 - i * 10.0)
			planner.local_solver.solve.return_value = 1
			planner.result = MagicMock(exit_code=1, success=True)
			planner.guidance_constraints = MagicMock()
			planner.safety_constraints = MagicMock()

		# Ensure we have guidance trajectories
		self.mock_global_guidance_instance.number_of_guidance_trajectories.return_value = 3
		self.mock_global_guidance_instance.get_guidance_trajectory.side_effect = lambda i: MagicMock(
			topology_class=i, color=i, previously_selected=False,
			spline=MagicMock(get_trajectory=MagicMock(return_value=MagicMock(
				get_point=MagicMock(return_value=np.array([i * 1.0, i * 2.0])),
				get_velocity=MagicMock(return_value=np.array([1.0, 0.0]))
			)))
		)

		# Mock the initialize_solver_with_guidance method
		with patch.object(self.guidance_constraints, 'initialize_solver_with_guidance') as mock_init:
			# Call method under test
			result = self.guidance_constraints.optimize(state, data, module_data)

			# Assertions
			self.assertEqual(result, 1)  # Should return exit_code=1 (success)

			# Check that solvers were configured with proper timeouts
			for planner in self.guidance_constraints.planners:
				self.assertAlmostEqual(planner.local_solver.params.solver_timeout, 0.044)  # 0.1 - 0.05 - 0.006

			# Check that best planner was selected (lowest objective, which is planners[3] with 70.0)
			self.assertEqual(self.guidance_constraints.best_planner_index_, 3)

			# Check that guidance trajectory selection was communicated back
			self.mock_global_guidance_instance.override_selected_trajectory.assert_called_once()

	def test_initialize_solver_with_guidance(self):
		"""Test initialize_solver_with_guidance method"""
		# Setup
		planner = MagicMock()
		planner.id = 1
		planner.local_solver = MagicMock()
		planner.local_solver.N = 10
		planner.local_solver.dt = 0.1

		# Setup guidance trajectory
		mock_trajectory = MagicMock()
		mock_trajectory.get_point.side_effect = lambda t: np.array([t * 2.0, t * 3.0])
		mock_trajectory.get_velocity.side_effect = lambda t: np.array([2.0, 3.0])

		mock_spline = MagicMock()
		mock_spline.get_trajectory.return_value = mock_trajectory

		self.mock_global_guidance_instance.get_guidance_trajectory.return_value = MagicMock(
			spline=mock_spline
		)

		# Call method under test
		self.guidance_constraints.initialize_solver_with_guidance(planner)

		# Assertions - check if solver was properly initialized with trajectory points
		expected_calls_x = [call(k, "x", k * 0.1 * 2.0) for k in range(10)]
		expected_calls_y = [call(k, "y", k * 0.1 * 3.0) for k in range(10)]

		planner.local_solver.set_ego_prediction.assert_has_calls(expected_calls_x + expected_calls_y, any_order=True)

		# Also should set heading and velocity
		for k in range(10):
			planner.local_solver.set_ego_prediction.assert_any_call(k, "psi", atan2(3.0, 2.0))
			planner.local_solver.set_ego_prediction.assert_any_call(k, "v", np.linalg.norm(np.array([2.0, 3.0])))

	def test_find_best_planner(self):
		"""Test find_best_planner method"""
		# Setup planners with different objectives
		self.guidance_constraints.planners = []
		for i in range(4):
			planner = MagicMock()
			planner.disabled = False
			planner.result = MagicMock()
			planner.result.success = True
			planner.result.objective = 100.0 - i * 20.0  # 100, 80, 60, 40
			self.guidance_constraints.planners.append(planner)

		# Disable one planner
		self.guidance_constraints.planners[2].disabled = True

		# Make one planner unsuccessful
		self.guidance_constraints.planners[0].result.success = False

		# Call method under test
		best_index = self.guidance_constraints.find_best_planner()

		# Should select planner 3 with objective 40.0 (lowest successful, non-disabled)
		self.assertEqual(best_index, 3)

		# Test with all planners disabled or unsuccessful
		for planner in self.guidance_constraints.planners:
			planner.disabled = True

		best_index = self.guidance_constraints.find_best_planner()
		self.assertEqual(best_index, -1)  # No valid planner found

	@patch('planner_modules.guidance_constraints.visualize_trajectory')
	def test_visualize(self, mock_vis_traj):
		"""Test visualize method"""
		# Setup
		data = MagicMock()
		module_data = MagicMock()

		# Setup planners with results
		self.guidance_constraints.planners = []
		for i in range(3):
			planner = MagicMock()
			planner.disabled = False
			planner.result = MagicMock(success=True, color=i)
			planner.is_original_planner = (i == 2)  # Make last one the original planner
			planner.local_solver = MagicMock()
			planner.local_solver.N = 5
			planner.local_solver.get_output.side_effect = lambda k, var: k * 1.0 if var == "x" else k * 2.0
			planner.local_solver.get_ego_prediction.side_effect = lambda k, var: k * 1.0 if var == "x" else k * 2.0
			planner.guidance_constraints = MagicMock()
			planner.safety_constraints = MagicMock()
			self.guidance_constraints.planners.append(planner)

		# Set the best planner index
		self.guidance_constraints.best_planner_index_ = 1

		# Setup global guidance visualization
		self.mock_global_guidance_instance.get_config.return_value = MagicMock(n_paths=3)

		# Enable debug visuals for coverage
		CONFIG_MOCK["debug_visuals"] = True

		# Call method under test
		self.guidance_constraints.visualize(data, module_data)

		# Assertions
		self.mock_global_guidance_instance.visualize.assert_called_once_with(True, -1)

		# Should only visualize constraints for the first planner
		self.guidance_constraints.planners[0].guidance_constraints.visualize.assert_called_once()
		self.guidance_constraints.planners[0].safety_constraints.visualize.assert_called_once()
		self.guidance_constraints.planners[1].guidance_constraints.visualize.assert_not_called()

		# Should visualize trajectories for all successful planners
		expected_calls = 3  # One for each successful planner
		if CONFIG_MOCK["debug_visuals"]:
			expected_calls *= 2  # Also visualize warmstarts

		self.assertEqual(mock_vis_traj.call_count, expected_calls)

		# Reset debug visuals
		CONFIG_MOCK["debug_visuals"] = False

	def test_is_data_ready(self):
		"""Test is_data_ready method"""
		# Setup
		data = MagicMock()
		missing_data = ""

		# Set up the first planner with mocked constraint modules
		first_planner = self.guidance_constraints.planners[0]
		first_planner.guidance_constraints = MagicMock()
		first_planner.safety_constraints = MagicMock()

		# Test when all data is ready
		first_planner.guidance_constraints.is_data_ready.return_value = True
		first_planner.safety_constraints.is_data_ready.return_value = True

		result = self.guidance_constraints.is_data_ready(data, missing_data)
		self.assertTrue(result)

		# Test when guidance constraints data is not ready
		first_planner.guidance_constraints.is_data_ready.return_value = False

		result = self.guidance_constraints.is_data_ready(data, missing_data)
		self.assertFalse(result)

		# Test when safety constraints data is not ready
		first_planner.guidance_constraints.is_data_ready.return_value = True
		first_planner.safety_constraints.is_data_ready.return_value = False

		result = self.guidance_constraints.is_data_ready(data, missing_data)
		self.assertFalse(result)

	def test_on_data_received_dynamic_obstacles(self):
		"""Test on_data_received method with dynamic obstacles"""
		# Setup
		data = MagicMock()
		obstacle = MagicMock()
		obstacle.index = 1
		obstacle.position = [5.0, 6.0]
		obstacle.radius = 1.0
		obstacle.prediction = MagicMock()
		obstacle.prediction.modes = [[MagicMock(position=[5.5, 6.5])]]
		data.dynamic_obstacles = [obstacle]
		data.robot_area = [MagicMock(radius=0.5)]

		# Create mock planner with mocked safety_constraints
		mock_planner = MagicMock()
		mock_planner.safety_constraints = MagicMock()
		mock_planner.safety_constraints.on_data_received = MagicMock()

		self.guidance_constraints.planners = [mock_planner]

		# Call method under test
		self.guidance_constraints.on_data_received(data, "dynamic obstacles")

		# Assertions
		mock_planner.safety_constraints.on_data_received.assert_called_once_with(data, "dynamic obstacles")

		self.mock_global_guidance_instance.load_obstacles.assert_called_once()
		obstacles_arg = self.mock_global_guidance_instance.load_obstacles.call_args[0][0]
		self.assertEqual(len(obstacles_arg), 1)
		self.assertEqual(obstacles_arg[0][0], 1)
		self.assertEqual(len(obstacles_arg[0][1]), 2)
		self.assertEqual(obstacles_arg[0][2], 1.5)

	def test_on_data_received_goal(self):
		"""Test on_data_received method with goal data"""
		# Setup
		data = MagicMock()

		# Call method under test
		self.guidance_constraints.on_data_received(data, "goal")

		# Currently this just logs a message, so there's nothing to assert
		pass

	def test_reset(self):
		"""Test reset method"""
		# Setup planners
		for planner in self.guidance_constraints.planners:
			planner.local_solver = MagicMock()

		# Call method under test
		self.guidance_constraints.reset()

		# Assertions
		self.mock_global_guidance_instance.reset.assert_called_once()
		for planner in self.guidance_constraints.planners:
			planner.local_solver.reset.assert_called_once()

	def test_save_data(self):
		"""Test save_data method"""
		# Setup
		data_saver = MagicMock()

		# Setup planners with results
		for i, planner in enumerate(self.guidance_constraints.planners):
			planner.result = MagicMock(success=(i % 2 == 0), objective=100.0 - i * 10.0)
			planner.is_original_planner = (i == 1)
			planner.local_solver = MagicMock()
			if planner.result.success:
				planner.local_solver._info = MagicMock(pobj=100.0 - i * 10.0)

		# Set best planner index
		self.guidance_constraints.best_planner_index_ = 2

		# Mock the last runtime
		self.mock_global_guidance_instance.get_last_runtime.return_value = 0.05

		# Call method under test
		self.guidance_constraints.save_data(data_saver)

		# Assertions
		data_saver.add_data.assert_any_call("runtime_guidance", 0.05)
		data_saver.add_data.assert_any_call("best_planner_idx", 2)
		data_saver.add_data.assert_any_call("gmpcc_objective", 80.0)

		# Check that each planner's objective was saved
		for i in range(4):
			if i % 2 == 0:  # Success
				data_saver.add_data.assert_any_call(f"objective_{i}", 100.0 - i * 10.0)
			else:  # Failure
				data_saver.add_data.assert_any_call(f"objective_{i}", -1)

		# Check that original planner was identified
		data_saver.add_data.assert_any_call("original_planner_id", 1)
		data_saver.add_data.assert_any_call("lmpcc_objective", -1)  # Original planner has no success

		# Check that global guidance save_data was called
		self.mock_global_guidance_instance.save_data.assert_called_once_with(data_saver)

	@patch('planner_modules.guidance_constraints.GlobalGuidance')
	def test_planner_integration(self, mock_global_guidance):
		"""Test if module properly interacts with planner"""

		# Setup mocks for planner's solve_mpc method
		data = MagicMock()
		data.planning_start_time = 0.0
		state = MagicMock()
		module_data = MagicMock()
		module_data.path = MagicMock()
		self.planner = MagicMock()
		self.planner.modules = [self.guidance_constraints]

		# Create mock GlobalGuidance instance
		mock_global_guidance_instance = MagicMock()

		# Return the mocked instance when GlobalGuidance is initialized
		mock_global_guidance.return_value = mock_global_guidance_instance

		# Setup data ready mocks
		with patch.object(self.guidance_constraints, 'is_data_ready', return_value=True), \
				patch.object(self.guidance_constraints, 'update') as mock_update, \
				patch.object(self.guidance_constraints, 'optimize') as mock_optimize, \
				patch.object(self.guidance_constraints, 'set_parameters') as mock_set_params:

			# Mock optimize to indicate success
			mock_optimize.return_value = 1

			# Mock the planner execution similar to actual code
			# Update modules
			for module in self.planner.modules:
				module.update(state, data, module_data)

			# Optimize the guidance
			for module in self.planner.modules:
				if hasattr(module, 'optimize'):
					module.optimize(state, data, module_data)

			# Set parameters for each prediction step
			for k in range(self.solver.N):
				for module in self.planner.modules:
					module.set_parameters(data, module_data, k)

			# Assertions
			mock_update.assert_called_once_with(state, data, module_data)
			mock_optimize.assert_called_once_with(state, data, module_data)

			# Module should have set_parameters called N times
			self.assertEqual(mock_set_params.call_count, self.solver.N)


class TestGlobalGuidance(unittest.TestCase):
	"""Test suite for the GlobalGuidance class"""

	def setUp(self):
		"""Set up common test environment"""
		self.guidance = GlobalGuidance()

		# Set up standard test parameters
		self.start_pos = np.array([0.0, 0.0])
		self.start_psi = 0.0
		self.start_v = 5.0
		self.ref_velocity = 5.0

		# Sample goals for testing
		self.test_goals = [
			([10.0, 0.0, 0.0], 0.0),  # Straight ahead goal, zero cost
			([10.0, 5.0, 0.1], 5.0),  # Goal to the right, cost 5
			([10.0, -5.0, -0.1], 5.0),  # Goal to the left, cost 5
		]

		# Set up a simple reference path for testing
		class SimplePath:
			def get_point(self, s):
				return np.array([s, 0.0])

			def get_orthogonal(self, s):
				return np.array([0.0, 1.0])

			def get_path_angle(self, s):
				return 0.0

		self.test_path = SimplePath()

	def test_initialization(self):
		"""Test that the GlobalGuidance initializes with correct default values"""
		self.assertEqual(self.guidance._planning_frequency, 10.0)
		self.assertEqual(self.guidance._dt, 0.1)
		self.assertEqual(len(self.guidance._static_obstacles), 0)
		self.assertEqual(len(self.guidance._dynamic_obstacles), 0)
		self.assertEqual(len(self.guidance._goals), 0)
		self.assertEqual(len(self.guidance._trajectories), 0)
		self.assertEqual(self.guidance._selected_trajectory_id, -1)
		self.assertFalse(self.guidance._success)

	def test_set_planning_frequency(self):
		"""Test setting planning frequency correctly updates dt"""
		test_freq = 20.0
		self.guidance.set_planning_frequency(test_freq)
		self.assertEqual(self.guidance._planning_frequency, test_freq)
		self.assertEqual(self.guidance._dt, 1.0 / test_freq)

	def test_set_start(self):
		"""Test setting the start state"""
		self.guidance.set_start(self.start_pos, self.start_psi, self.start_v)
		np.testing.assert_array_equal(self.guidance._current_pos, self.start_pos)
		self.assertEqual(self.guidance._current_psi, self.start_psi)
		self.assertEqual(self.guidance._current_v, self.start_v)

	def test_set_reference_velocity(self):
		"""Test setting reference velocity"""
		self.guidance.set_reference_velocity(self.ref_velocity)
		self.assertEqual(self.guidance._reference_velocity, self.ref_velocity)

	def test_load_static_obstacles(self):
		"""Test loading static obstacles"""
		obstacles = [
			(np.array([[1.0, 0.0]]), np.array([5.0])),  # A = [1, 0], b = [5]
			(np.array([[0.0, 1.0]]), np.array([5.0])),  # A = [0, 1], b = [5]
		]
		self.guidance.load_static_obstacles(obstacles)
		self.assertEqual(len(self.guidance._static_obstacles), 2)
		self.assertEqual(self.guidance._static_obstacles, obstacles)

	def test_load_reference_path(self):
		"""Test loading reference path"""
		start_s = 0.0
		width_left = 3.0
		width_right = 3.0

		self.guidance.load_reference_path(start_s, self.test_path, width_left, width_right)

		self.assertEqual(self.guidance._reference_path, self.test_path)
		self.assertEqual(self.guidance._start_s, start_s)
		self.assertEqual(self.guidance._width_left, width_left)
		self.assertEqual(self.guidance._width_right, width_right)

	def test_set_goals(self):
		"""Test setting goals"""
		self.guidance.set_goals(self.test_goals)
		self.assertEqual(len(self.guidance._goals), len(self.test_goals))
		self.assertEqual(self.guidance._goals, self.test_goals)

	def test_update_with_no_goals(self):
		"""Test update with no goals set"""
		# Setup
		start_time = time.time()
		self.guidance.update()
		end_time = time.time()

		# Check results
		self.assertFalse(self.guidance._success)
		self.assertEqual(len(self.guidance._trajectories), 0)
		self.assertTrue(self.guidance._runtime > 0)
		self.assertTrue(self.guidance._runtime <= (end_time - start_time))

	def test_update_with_goals(self):
		"""Test update with goals set"""
		# Setup
		self.guidance.set_start(self.start_pos, self.start_psi, self.start_v)
		self.guidance.set_reference_velocity(self.ref_velocity)
		self.guidance.set_goals(self.test_goals)

		# Execute
		self.guidance.update()

		# Check results
		self.assertTrue(self.guidance._success)
		self.assertTrue(len(self.guidance._trajectories) > 0)
		self.assertLessEqual(len(self.guidance._trajectories), self.guidance._config.n_paths)

	def test_get_guidance_trajectory(self):
		"""Test getting a specific guidance trajectory"""
		# Setup
		self.guidance.set_start(self.start_pos, self.start_psi, self.start_v)
		self.guidance.set_reference_velocity(self.ref_velocity)
		self.guidance.set_goals(self.test_goals)
		self.guidance.update()

		# Valid index
		traj = self.guidance.get_guidance_trajectory(0)
		self.assertIsNotNone(traj)
		self.assertTrue(hasattr(traj, 'topology_class'))
		self.assertTrue(hasattr(traj, 'color'))
		self.assertTrue(hasattr(traj, 'spline'))

		# Invalid index (should return a default trajectory)
		invalid_traj = self.guidance.get_guidance_trajectory(100)
		self.assertIsNotNone(invalid_traj)
		self.assertTrue(hasattr(invalid_traj, 'topology_class'))
		self.assertEqual(invalid_traj.topology_class, -1)

	def test_generate_trajectories(self):
		"""Test the trajectory generation logic"""
		# Setup
		self.guidance.set_start(self.start_pos, self.start_psi, self.start_v)
		self.guidance.set_reference_velocity(self.ref_velocity)
		self.guidance.set_goals(self.test_goals)

		# Execute by calling the internal method directly
		self.guidance._generate_trajectories()

		# Check results
		self.assertTrue(len(self.guidance._trajectories) > 0)

		# Check trajectory properties
		for traj in self.guidance._trajectories:
			self.assertTrue(hasattr(traj, 'topology_class'))
			self.assertTrue(hasattr(traj, 'color'))
			self.assertTrue(hasattr(traj, 'spline'))

			# Get and test the trajectory function
			trajectory_spline = traj.spline.get_trajectory()
			self.assertTrue(callable(trajectory_spline.get_point))
			self.assertTrue(callable(trajectory_spline.get_velocity))

			# Test the get_point and get_velocity functions
			point = trajectory_spline.get_point(0.0)
			self.assertEqual(len(point), 2)  # Should return [x, y]

			velocity = trajectory_spline.get_velocity(0.0)
			self.assertEqual(len(velocity), 2)  # Should return [vx, vy]

	def test_create_trajectory_to_goal(self):
		"""Test creating a trajectory to a specific goal"""
		# Setup
		goal_point = [10.0, 5.0, 0.1]  # x, y, psi
		traj_id = 0
		self.guidance.set_start(self.start_pos, self.start_psi, self.start_v)

		# Execute
		traj = self.guidance._create_trajectory_to_goal(goal_point, traj_id)

		# Check trajectory properties
		self.assertEqual(traj.topology_class, traj_id)
		self.assertEqual(traj.color, traj_id)

		# Get and test the trajectory function
		trajectory_spline = traj.spline.get_trajectory()

		# Test the start point
		start_point = trajectory_spline.get_point(0.0)
		np.testing.assert_array_almost_equal(start_point, self.start_pos)

		# Test a middle point
		middle_point = trajectory_spline.get_point(self.guidance._config.N * self.guidance._dt / 2)
		self.assertTrue(middle_point[0] > self.start_pos[0])  # Should be moving toward goal

		# Test the end point
		end_point = trajectory_spline.get_point(self.guidance._config.N * self.guidance._dt)
		goal_pos = np.array(goal_point[:2])
		# The end point should be in the direction of the goal
		# (May not reach the goal exactly due to trajectory duration limitations)
		direction = goal_pos - self.start_pos
		end_direction = end_point - self.start_pos
		# Check if moving in same direction as goal
		dot_product = np.dot(direction, end_direction)
		self.assertTrue(dot_product > 0)

	def test_override_selected_trajectory(self):
		"""Test overriding the selected trajectory"""
		guidance_id = 2
		is_original_planner = True

		self.guidance.override_selected_trajectory(guidance_id, is_original_planner)

		self.assertEqual(self.guidance._selected_trajectory_id, guidance_id)
		self.assertEqual(self.guidance._original_planner_selected, is_original_planner)

	def test_reset(self):
		"""Test resetting the guidance planner"""
		# Setup - create some state
		self.guidance.set_goals(self.test_goals)
		self.guidance.update()
		self.guidance.override_selected_trajectory(0, False)

		# Reset
		self.guidance.reset()

		# Check reset state
		self.assertEqual(len(self.guidance._trajectories), 0)
		self.assertEqual(self.guidance._selected_trajectory_id, -1)
		self.assertFalse(self.guidance._success)
		self.assertEqual(len(self.guidance._goals), 0)

	def test_load_obstacles(self):
		"""Test loading dynamic obstacles"""
		obstacles = [
			(1, [np.array([1.0, 1.0]), np.array([2.0, 2.0])], 1.0),  # id, positions, radius
			(2, [np.array([3.0, 3.0]), np.array([4.0, 4.0])], 0.8),  # id, positions, radius
		]
		extra_data = {}

		self.guidance.load_obstacles(obstacles, extra_data)

		self.assertEqual(len(self.guidance._dynamic_obstacles), len(obstacles))
		self.assertEqual(self.guidance._dynamic_obstacles, obstacles)

	def test_space_time_point_num_states(self):
		"""Test determining number of states for space-time points"""
		# Default (no goals) should return 2 (x, y)
		self.assertEqual(self.guidance.space_time_point_num_states(), 2)

		# With goals that have 3 components (x, y, psi)
		self.guidance.set_goals(self.test_goals)  # Our test goals have 3 components
		self.assertEqual(self.guidance.space_time_point_num_states(), 3)

		# With goals that have only 2 components
		goals_with_2_components = [([1.0, 2.0], 0.0), ([3.0, 4.0], 1.0)]
		self.guidance.set_goals(goals_with_2_components)
		self.assertEqual(self.guidance.space_time_point_num_states(), 2)

	def test_propagate_nodes_flag(self):
		"""Test disabling node propagation"""
		# Default should be True
		self.assertTrue(self.guidance._propagate_nodes)

		# Disable
		self.guidance.do_not_propagate_nodes()
		self.assertFalse(self.guidance._propagate_nodes)

	def test_get_last_runtime(self):
		"""Test getting the last planning runtime"""
		# Setup - run an update to set runtime
		self.guidance.set_start(self.start_pos, self.start_psi, self.start_v)
		self.guidance.set_goals(self.test_goals)
		self.guidance.update()

		# Get runtime
		runtime = self.guidance.get_last_runtime()
		self.assertTrue(runtime > 0)
		self.assertEqual(runtime, self.guidance._runtime)


if __name__ == '__main__':
	unittest.main()