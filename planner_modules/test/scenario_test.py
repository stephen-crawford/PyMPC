import unittest
import numpy as np
from unittest.mock import MagicMock, patch, call
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
		"parallel_solvers": 3,
		"enable_safe_horizon": True
	},
	"debug_visuals": False
}


@patch('utils.utils.read_config_file', return_value=CONFIG_MOCK)
class TestScenarioConstraints(unittest.TestCase):
	"""Test suite for ScenarioConstraints class"""

	def setUp(self):
		"""Set up test fixtures before each test"""
		# Create mock solver
		self.solver = MagicMock()
		self.solver.N = 10
		self.solver.params = MagicMock()
		self.solver._info = MagicMock()
		self.solver.output = MagicMock()
		self.solver._params = MagicMock()

		# Create mock for ScenarioSolver and ScenarioModule
		self.scenario_module_mock = MagicMock()
		self.scenario_sampler_mock = MagicMock()

		with patch('planner_modules.scenario_constraints.ScenarioSolver') as self.solver_mock, \
				patch('planner_modules.scenario_constraints.ScenarioConfig') as self.config_mock:
			# Set up scenario module mock
			self.solver_mock.return_value.scenario_module = self.scenario_module_mock
			self.scenario_module_mock.get_sampler.return_value = self.scenario_sampler_mock

			# Create instance of the class under test
			from planner_modules.scenario_constraints import ScenarioConstraints
			self.scenario_constraints = ScenarioConstraints(self.solver)

	def test_initialization(self, mock_config):
		"""Test proper initialization of ScenarioConstraints"""
		self.assertEqual(self.scenario_constraints.module_type, CONSTRAINT)
		self.assertEqual(self.scenario_constraints.name, "scenario_constraints")
		self.assertEqual(self.scenario_constraints._planning_time, 0.1)  # 1.0 / CONFIG["control_frequency"]
		self.assertEqual(len(self.scenario_constraints._scenario_solvers),
		                 CONFIG_MOCK["scenario_constraints"]["parallel_solvers"])
		self.assertIsNone(self.scenario_constraints._best_solver)

		# Check that ScenarioConfig was initialized
		self.config_mock.assert_called_once()
		self.config_mock.return_value.Init.assert_called_once()

		# Check that ScenarioSolver was created for each parallel solver
		self.assertEqual(self.solver_mock.call_count, CONFIG_MOCK["scenario_constraints"]["parallel_solvers"])
		for i in range(CONFIG_MOCK["scenario_constraints"]["parallel_solvers"]):
			self.solver_mock.assert_any_call(i)

	def test_update(self, mock_config):
		"""Test update method"""
		# Setup
		state = MagicMock()
		data = MagicMock()
		module_data = MagicMock()

		# Call method under test
		self.scenario_constraints.update(state, data, module_data)

		# Check that each scenario solver was updated
		for solver in self.scenario_constraints._scenario_solvers:
			self.assertEqual(solver.solver, self.solver)
			solver.scenario_module.update.assert_called_once_with(data, module_data)

	@patch('planner_modules.scenario_constraints.datetime')
	def test_optimize_with_feasible_solutions(self, mock_datetime, mock_config):
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

		# Setup solvers with different costs
		self.scenario_constraints._scenario_solvers[0].exit_code = 1  # Success
		self.scenario_constraints._scenario_solvers[0].solver._info.pobj = 100.0

		self.scenario_constraints._scenario_solvers[1].exit_code = 1  # Success
		self.scenario_constraints._scenario_solvers[1].solver._info.pobj = 50.0  # Best solution

		self.scenario_constraints._scenario_solvers[2].exit_code = 0  # Failed

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
			for solver in self.scenario_constraints._scenario_solvers:
				self.assertAlmostEqual(solver.solver._params.solver_timeout, expected_timeout, places=3)

			# Check optimization was run for each solver
			for solver in self.scenario_constraints._scenario_solvers:
				solver.scenario_module.optimize.assert_called_once_with(data)
				solver.solver.load_warmstart.assert_called_once()

			# Check best solver was selected (lowest cost with successful exit code)
			self.assertEqual(self.scenario_constraints._best_solver, self.scenario_constraints._scenario_solvers[1])

			# Check best solution was loaded into main solver
			self.assertEqual(self.solver.output, self.scenario_constraints._best_solver.solver.output)
			self.assertEqual(self.solver._info, self.scenario_constraints._best_solver.solver._info)
			self.assertEqual(self.solver._params, self.scenario_constraints._best_solver.solver._params)

			# Check return value is the exit code of the best solver
			self.assertEqual(result, 1)

	@patch('planner_modules.scenario_constraints.datetime')
	def test_optimize_with_no_feasible_solutions(self, mock_datetime, mock_config):
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
		for solver in self.scenario_constraints._scenario_solvers:
			solver.exit_code = 0  # Failed

		# Mock set_openmp_params
		with patch.object(self.scenario_constraints, 'set_openmp_params'):
			# Call method under test
			result = self.scenario_constraints.optimize(state, data, module_data)

			# Check that no best solver was selected
			self.assertIsNone(self.scenario_constraints._best_solver)

			# Return value should be the exit code of the first solver
			self.assertEqual(result, 0)

	def test_on_data_received_obstacles(self, mock_config):
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
		for solver in self.scenario_constraints._scenario_solvers:
			solver.scenario_module.get_sampler().integrate_and_translate_to_mean_and_variance.assert_called_once_with(
				data.dynamic_obstacles, solver.dt
			)

	def test_on_data_received_obstacles_deterministic(self, mock_config):
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

	def test_is_data_ready_success(self, mock_config):
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
		self.scenario_module_mock.is_data_ready.return_value = True

		# Call method under test
		missing_data = ""
		result = self.scenario_constraints.is_data_ready(data, missing_data)

		# Assertions
		self.assertTrue(result)
		self.scenario_module_mock.is_data_ready.assert_called_once()

	def test_is_data_ready_wrong_obstacle_count(self, mock_config):
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
		self.assertEqual(missing_data, "Obstacles ")
		self.scenario_module_mock.is_data_ready.assert_not_called()

	def test_is_data_ready_empty_prediction(self, mock_config):
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
		self.assertEqual(missing_data, "Obstacle Prediction ")
		self.scenario_module_mock.is_data_ready.assert_not_called()

	def test_is_data_ready_deterministic_prediction(self, mock_config):
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
		self.assertEqual(missing_data,
		                 "Uncertain Predictions (scenario-based control cannot use deterministic predictions) ")
		self.scenario_module_mock.is_data_ready.assert_not_called()

	def test_is_data_ready_scenario_module_not_ready(self, mock_config):
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
		self.scenario_module_mock.is_data_ready.return_value = False
		self.scenario_module_mock.is_data_ready.side_effect = lambda data, missing_data: False

		# Call method under test
		missing_data = ""
		result = self.scenario_constraints.is_data_ready(data, missing_data)

		# Assertions
		self.assertFalse(result)
		self.scenario_module_mock.is_data_ready.assert_called_once()

	@patch('planner_modules.scenario_constraints.visualize_trajectory')
	@patch('planner_modules.scenario_constraints.VISUALS')
	def test_visualize_with_best_solver(self, mock_visuals, mock_visualize_trajectory, mock_config):
		"""Test visualize method with best solver set"""
		# Setup
		data = MagicMock()
		module_data = MagicMock()

		# Set a best solver
		best_solver = self.scenario_constraints._scenario_solvers[1]
		best_solver.exit_code = 1  # Success
		self.scenario_constraints._best_solver = best_solver

		# Setup trajectory points
		best_solver.solver.get_output.side_effect = lambda k, param: k if param == "x" else k * 2

		# Call method under test
		self.scenario_constraints.visualize(data, module_data)

		# Assertions
		# Check best solver visualize was called
		best_solver.scenario_module.visualize.assert_called_once_with(data)

		# Check trajectory was visualized
		expected_trajectory = []
		for k in range(CONFIG_MOCK["N"]):
			expected_trajectory.append([k, k * 2])

		mock_visualize_trajectory.assert_called_with(
			expected_trajectory,
			f"{self.scenario_constraints.name}/optimized_trajectories",
			False,
			0.2,
			best_solver._solver_id,
			2 * len(self.scenario_constraints._scenario_solvers)
		)

		# Check visualization was published
		mock_visuals.missing_data.assert_called_once_with(f"{self.scenario_constraints.name}/optimized_trajectories")
		mock_visuals.missing_data.return_value.publish.assert_called_once()

	@patch('planner_modules.scenario_constraints.visualize_trajectory')
	@patch('planner_modules.scenario_constraints.VISUALS')
	def test_visualize_with_no_best_solver(self, mock_visuals, mock_visualize_trajectory, mock_config):
		"""Test visualize method with no best solver"""
		# Setup
		data = MagicMock()
		module_data = MagicMock()

		# No best solver set
		self.scenario_constraints._best_solver = None

		# But we have a successful solver
		self.scenario_constraints._scenario_solvers[0].exit_code = 1

		# Call method under test
		self.scenario_constraints.visualize(data, module_data)

		# No scenario module visualization should happen
		for solver in self.scenario_constraints._scenario_solvers:
			solver.scenario_module.visualize.assert_not_called()

		# Successful trajectories should still be visualized
		mock_visualize_trajectory.assert_called()
		mock_visuals.missing_data.assert_called_once()

	def test_set_openmp_params(self, mock_config):
		"""Test set_openmp_params method"""
		# This is mostly a mock implementation in the code, so we just check it exists
		self.scenario_constraints.set_openmp_params(nested=1, max_active_levels=2, dynamic=0)
		self.scenario_constraints.set_openmp_params(dynamic=1)
	# No assertion needed, just checking it doesn't raise exceptions


class TestSystemIntegration(unittest.TestCase):
	"""Test integration between ScenarioConstraints and Planner"""

	def setUp(self):
		"""Set up test fixtures before each test"""
		# Create mock solver
		self.solver = MagicMock()
		self.solver.N = 10
		self.solver.params = MagicMock()

		# Create mock for ScenarioSolver and ScenarioModule
		with patch('planner_modules.scenario_constraints.ScenarioSolver'), \
				patch('planner_modules.scenario_constraints.ScenarioConfig'):
			# Create instance of the class under test
			with patch('utils.utils.read_config_file', return_value=CONFIG_MOCK):
				from planner_modules.scenario_constraints import ScenarioConstraints
				self.scenario_constraints = ScenarioConstraints(self.solver)

		# Create mock planner
		self.planner = MagicMock()
		self.planner._modules = [self.scenario_constraints]

	@patch('utils.utils.read_config_file', return_value=CONFIG_MOCK)
	def test_planner_integration(self, mock_config):
		"""Test if module properly interacts with planner"""
		# Setup mocks for planner's solve_mpc method
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
			for module in self.planner._modules:
				module.update(state, data, module_data)

			# Set parameters for each prediction step
			for k in range(self.solver.N):
				for module in self.planner._modules:
					module.set_parameters(data, module_data, k)

			# Optimize (specific to scenario constraints)
			self.scenario_constraints.optimize(state, data, module_data)

			# Assertions
			mock_update.assert_called_once_with(state, data, module_data)
			mock_set_params.assert_called()
			mock_optimize.assert_called_once_with(state, data, module_data)


if __name__ == '__main__':
	unittest.main()