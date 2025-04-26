import unittest
import numpy as np
from unittest.mock import MagicMock, patch, call, ANY
from datetime import datetime, timedelta

# Import modules to test
from utils.const import CONSTRAINT, DETERMINISTIC, GAUSSIAN

# Manually patch CONFIG to avoid dependency issues in testing
CONFIG_MOCK = {
	"N": 10,
	"dt": 0.1,
	"max_obstacles": 3,
	"control_frequency": 10.0,
	"scenario_constraints": {
		"parallelsolvers": 3,
		"enable_safe_horizon": True,
		"visualize_all": False
	},
	"debug_visuals": False
}

# Patch the read_config_file function
with patch('utils.utils.read_config_file', return_value=CONFIG_MOCK):
	from planner_modules.scenario_constraints import ScenarioConstraints
	from planner_modules.base_constraint import BaseConstraint


class TestScenarioConstraints(unittest.TestCase):
	"""Test suite for ScenarioConstraints class"""

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
		self.solver._info = MagicMock()
		self.solver.output = MagicMock()

		# Apply CONFIG patch to BaseConstraint
		self.config_attr_patcher = patch('planner_modules.base_constraint.CONFIG', CONFIG_MOCK)
		self.config_attr_patcher.start()
		self.addCleanup(self.config_attr_patcher.stop)

		# Apply the patch before creating the class
		patcher = patch('planner_modules.base_constraint.BaseConstraint.get_config_value',
						side_effect=self.get_mocked_config)
		self.mock_get_config = patcher.start()
		self.addCleanup(patcher.stop)

		# Create mocks for ScenarioSolver and ScenarioModule
		self.scenario_solver_patcher = patch('planner_modules.scenario_constraints.ScenarioSolver')
		self.mock_scenario_solver = self.scenario_solver_patcher.start()
		self.addCleanup(self.scenario_solver_patcher.stop)

		# Configure solver mock to return a properly configured solver instance
		self.mock_solver_instances = []

		def create_solver_mock(solver_id):
			mock_instance = MagicMock()
			mock_instance.solver = MagicMock()
			mock_instance.solver._info = MagicMock()
			mock_instance.solver._info.pobj = 100.0 + solver_id * 10  # Different costs for different solvers
			mock_instance.scenario_module = MagicMock()
			mock_instance.exit_code = 0
			mock_instance.N = CONFIG_MOCK["N"]
			mock_instance.dt = CONFIG_MOCK["dt"]
			mock_instance.solver_id = solver_id
			self.mock_solver_instances.append(mock_instance)
			return mock_instance

		self.mock_scenario_solver.side_effect = create_solver_mock

		# Create mock for ScenarioConfig
		self.scenario_config_patcher = patch('planner_modules.scenario_constraints.ScenarioConfig')
		self.mock_scenario_config = self.scenario_config_patcher.start()
		self.addCleanup(self.scenario_config_patcher.stop)

		# Configure ScenarioConfig mock
		mock_config_instance = MagicMock()
		mock_config_instance.enable_safe_horizon_ = True
		self.mock_scenario_config.return_value = mock_config_instance

		# Create instance of the class under test
		self.scenario_constraints = ScenarioConstraints(self.solver)

		# Mock visualize_trajectory function
		self.visualize_trajectory_patcher = patch('planner_modules.base_constraint.BaseConstraint.visualize_trajectory')
		self.mock_visualize_trajectory = self.visualize_trajectory_patcher.start()
		self.addCleanup(self.visualize_trajectory_patcher.stop)

	def test_initialization(self):
		"""Test proper initialization of ScenarioConstraints"""
		self.assertEqual(self.scenario_constraints.module_type, CONSTRAINT)
		self.assertEqual(self.scenario_constraints.name, "scenario_constraints")
		self.assertEqual(self.scenario_constraints._planning_time, 0.1)  # 1.0 / CONFIG["control_frequency"]
		self.assertEqual(len(self.scenario_constraints._scenariosolvers),
						 CONFIG_MOCK["scenario_constraints"]["parallelsolvers"])
		self.assertIsNone(self.scenario_constraints._best_solver)

	def test_update(self):
		"""Test update method"""
		# Setup
		state = MagicMock()
		data = MagicMock()
		module_data = MagicMock()

		# Call method under test
		self.scenario_constraints.update(state, data, module_data)

		# Check that each scenario solver was updated
		for solver in self.mock_solver_instances:
			solver.scenario_module.update.assert_called_once_with(data, module_data)

	@patch('planner_modules.scenario_constraints.datetime')
	def test_optimize_with_feasible_solutions(self, mock_datetime):
		"""Test optimize method with feasible solutions"""
		# Setup
		state = MagicMock()
		data = MagicMock()
		module_data = MagicMock()

		# Setup time mocking
		start_time = datetime(2023, 1, 1, 12, 0, 0)
		current_time = datetime(2023, 1, 1, 12, 0, 0, 50000)  # 50ms elapsed
		data.planning_start_time = start_time
		mock_datetime.now.return_value = current_time

		for solver in self.mock_solver_instances:
			solver.solver = MagicMock()
			solver.solver.params = MagicMock()
			solver.solver._info = MagicMock()
			solver.solver.output = MagicMock()
			solver.scenario_module = MagicMock()

		# Now set specific behavior
		self.mock_solver_instances[0].scenario_module.optimize.return_value = 1
		self.mock_solver_instances[0].solver._info.pobj = 100.0

		self.mock_solver_instances[1].scenario_module.optimize.return_value = 1
		self.mock_solver_instances[1].solver._info.pobj = 50.0

		self.mock_solver_instances[2].scenario_module.optimize.return_value = 0
		self.mock_solver_instances[2].solver._info.pobj = 9999.0

		# IMPORTANT: Make the optimize() method actually use your mock solvers
		self.scenario_constraints._scenariosolvers = self.mock_solver_instances

		# IMPORTANT: Make the optimize() method actually use your mock solvers
		self.scenario_constraints._scenariosolvers = self.mock_solver_instances

		# Mock main solver
		self.scenario_constraints.solver = MagicMock()
		self.scenario_constraints.solver.output = MagicMock()
		self.scenario_constraints.solver._info = MagicMock()
		self.scenario_constraints.solver.params = MagicMock()

		# Mock set_openmp_params
		with patch.object(self.scenario_constraints, 'set_openmp_params') as mock_openmp:
			# Call method under test
			result = self.scenario_constraints.optimize(state, data, module_data)

			# Assertions
			# Check OpenMP params were set correctly
			mock_openmp.assert_any_call(nested=1, max_active_levels=2, dynamic=0)
			mock_openmp.assert_any_call(dynamic=1)

			# Check solver timeout was set correctly (100ms planning time - 50ms elapsed - 8ms buffer)
			expected_timeout = 0.042  # 100ms - 50ms - 8ms
			for solver in self.mock_solver_instances:
				self.assertAlmostEqual(solver.solver.params.solver_timeout, expected_timeout, places=3)

			# Check optimization was run for each solver
			for solver in self.mock_solver_instances:
				solver.scenario_module.optimize.assert_called_once_with(data)
				solver.solver.load_warmstart.assert_called_once()

			# Check best solver was selected (lowest cost with successful exit code)
			print("Best solver is:" +  str(self.scenario_constraints._best_solver))
			print("Expected sovler is: " + str(self.mock_solver_instances[1]))
			self.assertEqual(self.scenario_constraints._best_solver, self.mock_solver_instances[1])

			# Check best solution was loaded into main solver
			self.assertIs(self.scenario_constraints.solver.output, self.scenario_constraints._best_solver.solver.output)
			self.assertIs(self.scenario_constraints.solver._info, self.scenario_constraints._best_solver.solver._info)
			self.assertIs(self.scenario_constraints.solver.params, self.scenario_constraints._best_solver.solver.params)

			# Check return value is the exit code of the best solver
			self.assertEqual(result, 1)

	@patch('planner_modules.scenario_constraints.datetime')
	def test_optimize_with_no_feasible_solutions(self, mock_datetime):
		"""Test optimize method with no feasible solutions"""
		# Setup
		state = MagicMock()
		data = MagicMock()
		module_data = MagicMock()

		# Setup time mocking
		start_time = datetime(2023, 1, 1, 12, 0, 0)
		current_time = datetime(2023, 1, 1, 12, 0, 0, 50000)  # 50ms elapsed
		data.planning_start_time = start_time
		mock_datetime.now.return_value = current_time

		# Setup all solvers to fail
		for solver in self.mock_solver_instances:
			solver.scenario_module.optimize.return_value = 0

		# Mock set_openmp_params
		with patch.object(self.scenario_constraints, 'set_openmp_params'):
			# Call method under test
			result = self.scenario_constraints.optimize(state, data, module_data)

			# Check that no best solver was selected
			self.assertIsNone(self.scenario_constraints._best_solver)

			# Return value should be the exit code of the first solver
			self.assertEqual(result, 0)

	def test_on_data_received_obstacles(self):
		"""Test on_data_received method with dynamic obstacles"""
		# Setup
		data = MagicMock()

		# Setup obstacles with non-deterministic predictions
		obstacle = MagicMock()
		obstacle.prediction.type = GAUSSIAN  # Non-deterministic
		data.dynamic_obstacles = [obstacle]

		# Call method under test
		self.scenario_constraints.on_data_received(data, "dynamic obstacles")

		# Check that sampler was called for each solver
		for solver in self.mock_solver_instances:
			solver.scenario_module.get_sampler().integrate_and_translate_to_mean_and_variance.assert_called_once_with(
				data.dynamic_obstacles, solver.dt
			)

	def test_on_data_received_obstacles_deterministic(self):
		"""Test on_data_received method with deterministic obstacles (should assert)"""
		# Setup
		data = MagicMock()

		# Setup obstacles with deterministic predictions
		obstacle = MagicMock()
		obstacle.prediction.type = DETERMINISTIC  # This should cause an assertion error
		data.dynamic_obstacles = [obstacle]

		# Check that an assertion error is raised
		with self.assertRaises(AssertionError):
			self.scenario_constraints.on_data_received(data, "dynamic obstacles")

	def test_is_data_ready_success(self):
		"""Test is_data_ready method with valid data"""
		# Setup
		data = MagicMock()
		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = CONFIG_MOCK["max_obstacles"]

		# Setup obstacle with non-deterministic prediction
		obstacle = MagicMock()
		obstacle.prediction.empty.return_value = False
		obstacle.prediction.type = GAUSSIAN  # Non-deterministic
		data.dynamic_obstacles.__getitem__.return_value = obstacle

		# Setup scenario module to return ready
		self.mock_solver_instances[0].scenario_module.is_data_ready.return_value = True

		# Call method under test
		missing_data = ""
		result = self.scenario_constraints.is_data_ready(data, missing_data)

		# Assertions
		self.assertTrue(result)
		self.mock_solver_instances[0].scenario_module.is_data_ready.assert_called_once()

	def test_is_data_ready_wrong_obstacle_count(self):
		"""Test is_data_ready method with wrong obstacle count"""
		# Setup
		data = MagicMock()
		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = CONFIG_MOCK["max_obstacles"] - 1  # Not enough obstacles

		# Call method under test
		missing_data = ""
		result = self.scenario_constraints.is_data_ready(data, missing_data)

		# Assertions
		self.assertFalse(result)
		for solver in self.mock_solver_instances:
			solver.scenario_module.is_data_ready.assert_not_called()

	def test_is_data_ready_empty_prediction(self):
		"""Test is_data_ready method with empty prediction"""
		# Setup
		data = MagicMock()
		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = CONFIG_MOCK["max_obstacles"]

		# Setup obstacle with empty prediction
		obstacle = MagicMock()
		obstacle.prediction.empty.return_value = True
		data.dynamic_obstacles.__getitem__.return_value = obstacle

		# Call method under test
		missing_data = ""
		result = self.scenario_constraints.is_data_ready(data, missing_data)

		# Assertions
		self.assertFalse(result)
		for solver in self.mock_solver_instances:
			solver.scenario_module.is_data_ready.assert_not_called()

	def test_is_data_ready_deterministic_prediction(self):
		"""Test is_data_ready method with deterministic prediction"""
		# Setup
		data = MagicMock()
		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = CONFIG_MOCK["max_obstacles"]

		# Setup obstacle with deterministic prediction
		obstacle = MagicMock()
		obstacle.prediction.empty.return_value = False
		obstacle.prediction.type = DETERMINISTIC  # Not allowed for scenario constraints
		data.dynamic_obstacles.__getitem__.return_value = obstacle

		# Call method under test
		missing_data = ""
		result = self.scenario_constraints.is_data_ready(data, missing_data)

		# Assertions
		self.assertFalse(result)
		for solver in self.mock_solver_instances:
			solver.scenario_module.is_data_ready.assert_not_called()

	def test_is_data_ready_scenario_module_not_ready(self):
		"""Test is_data_ready method when scenario module is not ready"""
		# Setup
		data = MagicMock()
		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = CONFIG_MOCK["max_obstacles"]

		# Setup obstacle with non-deterministic prediction
		obstacle = MagicMock()
		obstacle.prediction.empty.return_value = False
		obstacle.prediction.type = GAUSSIAN  # Non-deterministic
		data.dynamic_obstacles.__getitem__.return_value = obstacle

		# Setup scenario module to return not ready
		self.mock_solver_instances[0].scenario_module.is_data_ready.return_value = False

		# Call method under test
		missing_data = ""
		result = self.scenario_constraints.is_data_ready(data, missing_data)

		# Assertions
		self.assertFalse(result)
		self.mock_solver_instances[0].scenario_module.is_data_ready.assert_called_once()

	def test_visualize(self):
		"""Test visualize method with best solver set"""
		# Setup
		data = MagicMock()
		module_data = MagicMock()

		# Set a best solver
		best_solver = self.mock_solver_instances[1]
		best_solver.exit_code = 1  # Success
		self.scenario_constraints._best_solver = best_solver

		# Setup trajectory points for both successful solvers
		self.mock_solver_instances[0].exit_code = 1

		# Mock get_output method properly for both successful solvers
		def get_output_side_effect(k, param):
			return k if param == "x" else k * 2

		self.mock_solver_instances[0].solver.get_output = MagicMock(side_effect=get_output_side_effect)
		self.mock_solver_instances[1].solver.get_output = MagicMock(side_effect=get_output_side_effect)

		# Call method under test
		self.scenario_constraints.visualize(data, module_data)

		# Check best solver visualize was called
		best_solver.scenario_module.visualize.assert_called_once_with(data)

		# Check trajectory was visualized for both successful solvers
		expected_calls = []
		for solver in self.mock_solver_instances:
			if solver.exit_code == 1:
				trajectory = []
				for k in range(CONFIG_MOCK["N"]):
					trajectory.append([k, k * 2])
				expected_calls.append(call(
					trajectory,
					"optimized_trajectories",
					scale=0.2,
					color_int=solver.solver_id
				))

		self.mock_visualize_trajectory.assert_has_calls(expected_calls, any_order=True)
		self.assertEqual(self.mock_visualize_trajectory.call_count, 2)  # Two successful solvers

	def test_visualize_with_no_best_solver(self):
		"""Test visualize method with no best solver"""
		# Setup
		data = MagicMock()
		module_data = MagicMock()

		# No best solver set
		self.scenario_constraints._best_solver = None

		# But we have a successful solver
		self.mock_solver_instances[0].exit_code = 1
		self.mock_solver_instances[0].solver.get_output.side_effect = lambda k, param: k if param == "x" else k * 2

		# Call method under test
		self.scenario_constraints.visualize(data, module_data)

		# No scenario module visualization should happen
		for solver in self.mock_solver_instances:
			solver.scenario_module.visualize.assert_not_called()

		# But successful trajectories should still be visualized
		self.mock_visualize_trajectory.assert_called_once()

	def test_reset(self):
		"""Test reset method"""
		# Setup
		# Set a best solver
		self.scenario_constraints._best_solver = self.mock_solver_instances[0]

		# Set exit codes
		for solver in self.mock_solver_instances:
			solver.exit_code = 1

		# Call method under test
		self.scenario_constraints.reset()

		# Assertions
		self.assertIsNone(self.scenario_constraints._best_solver)
		for solver in self.mock_solver_instances:
			self.assertEqual(solver.exit_code, 0)

	def test_set_openmp_params(self):
		"""Test set_openmp_params method"""
		# This is mostly a mock implementation in the code, so we just check it exists
		self.scenario_constraints.set_openmp_params(nested=1, max_active_levels=2, dynamic=0)
		self.scenario_constraints.set_openmp_params(dynamic=1)
	# No assertion needed, just checking it doesn't raise exceptions

	def test_planner_integration(self):
		"""Test if module properly interacts with planner"""
		# Setup mocks for planner's solve_mpc method
		self.planner = MagicMock()
		self.planner.modules = [self.scenario_constraints]
		data = MagicMock()
		state = MagicMock()
		module_data = MagicMock()

		# Setup data ready mocks
		with patch.object(self.scenario_constraints, 'is_data_ready', return_value=True), \
				patch.object(self.scenario_constraints, 'update') as mock_update, \
				patch.object(self.scenario_constraints, 'set_parameters') as mock_set_params, \
				patch.object(self.scenario_constraints, 'optimize') as mock_optimize:

			# Mock planner.solve_mpc similar to the actual implementation
			# Update modules
			for module in self.planner.modules:
				module.update(state, data, module_data)

			# Set parameters for each prediction step
			for k in range(self.solver.N):
				for module in self.planner.modules:
					module.set_parameters(data, module_data, k)

			# Optimize (specific to scenario constraints)
			self.scenario_constraints.optimize(state, data, module_data)

			# Assertions
			mock_update.assert_called_once_with(state, data, module_data)
			mock_set_params.assert_called()
			mock_optimize.assert_called_once_with(state, data, module_data)

if __name__ == '__main__':
	unittest.main()