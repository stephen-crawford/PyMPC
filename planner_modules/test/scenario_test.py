import unittest
from unittest.mock import MagicMock, patch, call, ANY
from datetime import datetime

# Import modules to test
from utils.const import CONSTRAINT, DETERMINISTIC, GAUSSIAN

# Manually patch CONFIG to avoid dependency issues in testing
CONFIG_MOCK = {
	"horizon": 10,
	"timestep": 0.1,
	"max_obstacles": 3,
	"max_constraints": 20,
	"num_discs": 10,
	"control_frequency": 10.0,
	"scenario_constraints": {
		"parallel_solvers": 3,
		"enable_safe_horizon": True,
		"visualize_all": False,
		"use_slack": True,
		"num_scenarios": 5,
		"slack_penalty": 1000.0,
	},
	"debug_visuals": False,
	"linearized_constraints": {
		"add_halfspaces": 2
	}
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


class TestScenarioConstraints(unittest.TestCase):
	"""Test suite for ScenarioConstraints class"""

	def setUp(self):
		"""Set up test fixtures before each test"""
		# Create mock solver
		self.solver = MagicMock()
		self.solver.horizon = 10
		self.solver.timestep = 0.1
		self.solver.params = MagicMock()
		self.solver.info = MagicMock()
		self.solver.output = MagicMock()

		# Apply the patch before creating the class
		self.patcher = patch('planner_modules.src.constraints.base_constraint.BaseConstraint.get_config_value',
							 side_effect=get_mocked_config)
		self.mock_get_config = self.patcher.start()
		self.addCleanup(self.patcher.stop)

		# Create mocks for ScenarioSolver, ScenarioModule and ScenarioSampler classes
		self.scenario_solver_patcher = patch('planner_modules.src.constraints.scenario_constraints.ScenarioSolver')
		self.mock_scenario_solver_class = self.scenario_solver_patcher.start()
		self.addCleanup(self.scenario_solver_patcher.stop)

		self.scenario_module_patcher = patch('planner_modules.src.constraints.scenario_constraints.ScenarioModule')
		self.mock_scenario_module_class = self.scenario_module_patcher.start()
		self.addCleanup(self.scenario_module_patcher.stop)

		self.scenario_sampler_patcher = patch('planner_modules.src.constraints.scenario_constraints.ScenarioSampler')
		self.mock_scenario_sampler_class = self.scenario_sampler_patcher.start()
		self.addCleanup(self.scenario_sampler_patcher.stop)

		# Configure mock instances that will be created
		self.mock_solver_instances = []
		self.mock_module_instances = []
		self.mock_sampler_instance = MagicMock()

		# Set up behavior for ScenarioSolver constructor
		def create_solver_mock(solver_id, config):
			mock_instance = MagicMock()
			# Create a new scenario module mock for each solver
			mock_module = MagicMock()
			mock_sampler = MagicMock()
			mock_module.get_sampler.return_value = mock_sampler
			mock_instance.scenario_module = mock_module

			# Set up solver properties
			mock_instance.solver = MagicMock()
			mock_instance.solver.info = MagicMock()
			mock_instance.solver.info.pobj = 100.0 + solver_id * 10  # Different costs for different solvers
			mock_instance.exit_code = 0
			mock_instance.horizon = CONFIG_MOCK["horizon"]
			mock_instance.timestep = CONFIG_MOCK["timestep"]
			mock_instance.solver_id = solver_id
			mock_instance.tmp_config = MagicMock()

			self.mock_solver_instances.append(mock_instance)
			self.mock_module_instances.append(mock_module)
			return mock_instance

		self.mock_scenario_solver_class.side_effect = create_solver_mock

		# Now import the class and create an instance
		from planner_modules.src.constraints.scenario_constraints import ScenarioConstraints
		self.scenario_constraints = ScenarioConstraints(self.solver)
		self.scenario_constraints._deep_copy_solver = MagicMock(side_effect=lambda s: s)

		# Verify the number of solvers created
		self.assertEqual(len(self.mock_solver_instances),
						 CONFIG_MOCK["scenario_constraints"]["parallel_solvers"])

		# Replace the created solvers with our mocks for testing
		self.scenario_constraints.scenario_solvers = self.mock_solver_instances

	def test_initialization(self):
		"""Test proper initialization of ScenarioConstraints"""
		self.assertEqual(self.scenario_constraints.module_type, CONSTRAINT)
		self.assertEqual(self.scenario_constraints.name, "scenario_constraints")
		self.assertEqual(self.scenario_constraints.planning_time, 0.1)  # 1.0 / CONFIG["control_frequency"]
		self.assertEqual(self.scenario_constraints.enable_safe_horizon, True)
		self.assertEqual(len(self.scenario_constraints.scenario_solvers),
						 CONFIG_MOCK["scenario_constraints"]["parallel_solvers"])
		self.assertIsNone(self.scenario_constraints.best_solver)

	def test_update(self):
		"""Test update method"""
		# Setup
		state = MagicMock()
		data = MagicMock()
		module_data = MagicMock()

		# Configure dynamic obstacles with non-deterministic predictions
		obstacle = MagicMock()
		obstacle.prediction.type = GAUSSIAN
		data.dynamic_obstacles = [obstacle]

		# Call method under test
		self.scenario_constraints.update(state, data, module_data)

		# Check that each scenario solver was updated
		for solver_mock in self.mock_solver_instances:
			solver_mock.scenario_module.update.assert_called_once_with(data, module_data)

	def test_optimize_with_feasible_solutions(self):
		"""Test optimize method with feasible solutions"""

		# Setup
		state = MagicMock()
		data = MagicMock()
		module_data = MagicMock()

		# Setup time mocking
		start_time = datetime(2023, 1, 1, 12, 0, 0)
		data.planning_start_time = start_time

		# Configure solver mocks
		solver0 = MagicMock()
		solver1 = MagicMock()
		solver2 = MagicMock()

		solver0.scenario_module.optimize.return_value = 1  # Success
		solver0.solver.info.pobj = 100.0
		solver0.exit_code = 1
		solver0.solver_id = 0
		solver0.horizon = 1

		solver1.scenario_module.optimize.return_value = 1  # Success
		solver1.solver.info.pobj = 50.0
		solver1.exit_code = 1
		solver1.solver_id = 1
		solver1.horizon = 1

		solver2.scenario_module.optimize.return_value = 0  # Failure
		solver2.solver.info.pobj = 9999.0
		solver2.exit_code = 0
		solver2.solver_id = 2
		solver2.horizon = 1

		# Save references for assertions if needed
		self.mock_solver_instances = [solver0, solver1, solver2]

		# Inject into scenario_constraints
		self.scenario_constraints.scenario_solvers = [solver0, solver1, solver2]

		# Call the method under test
		result = self.scenario_constraints.optimize(state, data, module_data)

		# Optional assertions
		self.assertEqual(self.scenario_constraints.feasible_solutions, 2)
		self.assertEqual(self.scenario_constraints.best_solver.solver_id, 1)  # solver1 has lowest cost

		# Check optimization was run for each solver
		for solver in self.mock_solver_instances:
			solver.scenario_module.optimize.assert_called_once_with(data)
			solver.solver.load_warmstart.assert_called_once()

		# Check best solver was selected (lowest cost with successful exit code)
		self.assertEqual(self.scenario_constraints.best_solver, self.mock_solver_instances[1])

		# Check best solution was loaded into main solver
		self.assertIs(self.scenario_constraints.solver.output, self.scenario_constraints.best_solver.solver.output)
		self.assertIs(self.scenario_constraints.solver.info, self.scenario_constraints.best_solver.solver.info)
		self.assertIs(self.scenario_constraints.solver.tmp_config,
					  self.scenario_constraints.best_solver.solver.tmp_config)

		# Check return value is the exit code of the best solver
		self.assertEqual(result, 1)

	@patch('datetime.datetime')
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
			solver.solver.info.constraint_violation = 100.0 + solver.solver_id * 10

		# The third solver has the lowest constraint violation
		self.mock_solver_instances[2].solver.info.constraint_violation = 50.0

		# Call method under test
		result = self.scenario_constraints.optimize(state, data, module_data)

		# Check that no best solver was selected based on cost
		self.assertIsNone(self.scenario_constraints.best_solver)

		# Check that the solution with minimum constraint violation was used
		self.assertIs(self.scenario_constraints.solver.output, self.mock_solver_instances[2].solver.output)
		self.assertIs(self.scenario_constraints.solver.info, self.mock_solver_instances[2].solver.info)
		self.assertIs(self.scenario_constraints.solver.tmp_config, self.mock_solver_instances[2].solver.tmp_config)

		# Return value should be 0 indicating failure
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
		self.scenario_constraints.on_data_received(data)

		# Check that sampler was called for each solver when safe horizon is enabled
		for solver in self.mock_solver_instances:
			solver.scenario_module.get_sampler().integrate_and_translate_to_mean_and_variance.assert_called_once_with(
				data.dynamic_obstacles, solver.timestep
			)

	def test_on_data_received_obstacles_deterministic(self):
		"""Test on_data_received method with deterministic obstacles (should just return)"""
		# Setup
		data = MagicMock()

		# Setup obstacles with deterministic predictions
		obstacle = MagicMock()
		obstacle.prediction.type = DETERMINISTIC
		data.dynamic_obstacles = [obstacle]

		# Call method under test - should just return without error
		self.scenario_constraints.on_data_received(data, "dynamic obstacles")

		# No integration should be called
		for solver in self.mock_solver_instances:
			solver.scenario_module.get_sampler().integrate_and_translate_to_mean_and_variance.assert_not_called()

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
		for solver in self.mock_solver_instances:
			solver.scenario_module.is_data_ready.return_value = True

		# Call method under test
		result = self.scenario_constraints.is_data_ready(data)

		# Assertions
		self.assertTrue(result)
		for solver in self.mock_solver_instances:
			solver.scenario_module.is_data_ready.assert_called_once()

	def test_is_data_ready_too_many_obstacles(self):
		"""Test is_data_ready method with too many obstacles"""
		# Setup
		data = MagicMock()
		data.dynamic_obstacles = MagicMock()
		data.dynamic_obstacles.size.return_value = CONFIG_MOCK["max_obstacles"] + 1  # Too many obstacles

		# Call method under test
		result = self.scenario_constraints.is_data_ready(data)

		# Assertions
		self.assertFalse(result)

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
		result = self.scenario_constraints.is_data_ready(data)

		# Assertions
		self.assertFalse(result)

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
		result = self.scenario_constraints.is_data_ready(data)

		# Assertions
		self.assertFalse(result)

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
		for solver in self.mock_solver_instances:
			solver.scenario_module.is_data_ready.return_value = False

		# Call method under test
		result = self.scenario_constraints.is_data_ready(data)

		# Assertions
		self.assertFalse(result)
		self.mock_solver_instances[0].scenario_module.is_data_ready.assert_called_once()

	def test_reset(self):
		"""Test reset method"""
		# Setup
		# Set a best solver
		self.scenario_constraints.best_solver = self.mock_solver_instances[0]

		# Set exit codes
		for solver in self.mock_solver_instances:
			solver.exit_code = 1

		# Call method under test
		self.scenario_constraints.reset()

		# Assertions
		self.assertIsNone(self.scenario_constraints.best_solver)
		for solver in self.mock_solver_instances:
			self.assertEqual(solver.exit_code, 0)
			if hasattr(solver.scenario_module, 'reset'):
				solver.scenario_module.reset.assert_called_once()

	def test_planner_integration(self):
		"""Test if module properly interacts with planning"""
		# Setup mocks for planning's solve_mpc method
		self.planner = MagicMock()
		self.planner.modules = [self.scenario_constraints]
		data = MagicMock()
		state = MagicMock()
		module_data = MagicMock()

		# Setup data ready mocks
		with patch.object(self.scenario_constraints, 'is_data_ready', return_value=True), \
				patch.object(self.scenario_constraints, 'update') as mock_update, \
				patch.object(self.scenario_constraints, 'optimize') as mock_optimize:

			# Mock planning.solve_mpc similar to the actual implementation
			# Update modules
			for module in self.planner.modules:
				module.update(state, data, module_data)

			# Set parameters for each prediction step
			for k in range(self.solver.horizon):
				for module in self.planner.modules:
					if hasattr(module, 'set_parameters'):
						module.set_parameters(self.solver.params, data, module_data, k)

			# Optimize (specific to scenario constraints)
			self.scenario_constraints.optimize(state, data, module_data)

			# Assertions
			mock_update.assert_called_once_with(state, data, module_data)
			mock_optimize.assert_called_once_with(state, data, module_data)


if __name__ == '__main__':
	unittest.main()