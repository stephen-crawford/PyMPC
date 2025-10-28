"""
Example usage and integration test for Safe Horizon Constraint.
"""
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from planner_modules.src.constraints.safe_horizon_constraint import SafeHorizonConstraint
from planning.src.types import Data, DynamicObstacle, PredictionType, PredictionStep
from solver.src.casadi_solver import CasADiSolver


class TestSafeHorizonConstraint(unittest.TestCase):
    """Test suite for SafeHorizonConstraint class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock solver
        self.solver = MagicMock()
        self.solver.horizon = 10
        self.solver.timestep = 0.1
        self.solver.copy.return_value = self.solver
        
        # Mock configuration
        self.config = {
            "safe_horizon": {
                "epsilon_p": 0.1,
                "beta": 0.01,
                "n_bar": 10,
                "num_removal": 0,
                "max_constraints_per_disc": 24,
                "use_slack": True,
                "num_scenarios": 100,
                "enable_outlier_removal": True,
                "parallel_solvers": 2
            },
            "robot": {"radius": 0.5},
            "num_discs": 1,
            "control_frequency": 10.0
        }
        
        # Create constraint with mocked config
        with patch('planner_modules.src.constraints.safe_horizon_constraint.read_config_file', 
                  return_value=self.config):
            self.constraint = SafeHorizonConstraint(self.solver)
    
    def test_initialization(self):
        """Test constraint initialization."""
        self.assertEqual(self.constraint.name, "safe_horizon_constraint")
        self.assertEqual(self.constraint.epsilon_p, 0.1)
        self.assertEqual(self.constraint.beta, 0.01)
        self.assertEqual(self.constraint.n_bar, 10)
        self.assertEqual(self.constraint.robot_radius, 0.5)
        self.assertEqual(self.constraint.horizon_length, 10)
        self.assertTrue(len(self.constraint.scenario_solvers) > 0)
    
    def test_compute_sample_size(self):
        """Test sample size computation."""
        sample_size = self.constraint.compute_sample_size()
        self.assertGreater(sample_size, 0)
        self.assertIsInstance(sample_size, int)
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        self.assertTrue(self.constraint.validate_parameters())
    
    def test_data_readiness(self):
        """Test data readiness checking."""
        # Test with no obstacles
        data = Data()
        self.assertFalse(self.constraint.is_data_ready(data))
        
        # Test with valid obstacles
        data.dynamic_obstacles = self._create_test_obstacles()
        self.assertTrue(self.constraint.is_data_ready(data))
    
    def test_constraint_info(self):
        """Test constraint info retrieval."""
        info = self.constraint.get_constraint_info()
        self.assertIsInstance(info, dict)
        self.assertIn("status", info)
    
    def test_define_parameters(self):
        """Test parameter definition."""
        params = MagicMock()
        params.add = MagicMock()
        
        self.constraint.define_parameters(params)
        
        # Should add parameters for each disc, step, and constraint
        expected_calls = self.constraint.num_discs * self.constraint.horizon_length * self.constraint.max_constraints_per_disc * 3
        self.assertEqual(params.add.call_count, expected_calls)
    
    def test_constraint_bounds(self):
        """Test constraint bounds."""
        lower_bounds = self.constraint.get_lower_bound()
        upper_bounds = self.constraint.get_upper_bound()
        
        expected_size = self.constraint.num_discs * self.constraint.max_constraints_per_disc
        
        self.assertEqual(len(lower_bounds), expected_size)
        self.assertEqual(len(upper_bounds), expected_size)
        
        # All lower bounds should be -inf
        self.assertTrue(all(bound == -np.inf for bound in lower_bounds))
        
        # All upper bounds should be 0
        self.assertTrue(all(bound == 0.0 for bound in upper_bounds))
    
    def test_reset(self):
        """Test constraint reset."""
        # Set some state
        self.constraint.best_solver = MagicMock()
        self.constraint.optimization_time = 1.0
        self.constraint.feasible_solutions = 5
        
        # Reset
        self.constraint.reset()
        
        # Check state is reset
        self.assertIsNone(self.constraint.best_solver)
        self.assertEqual(self.constraint.optimization_time, 0)
        self.assertEqual(self.constraint.feasible_solutions, 0)
    
    def _create_test_obstacles(self):
        """Create test obstacles for testing."""
        obstacles = []
        
        # Create a dynamic obstacle with Gaussian prediction
        obstacle = DynamicObstacle(
            index=0,
            position=np.array([5.0, 5.0]),
            angle=0.0,
            radius=0.5
        )
        
        # Add prediction steps
        obstacle.prediction.type = PredictionType.GAUSSIAN
        obstacle.prediction.steps = []
        
        for i in range(5):
            step = PredictionStep(
                position=np.array([5.0 + i * 0.5, 5.0]),
                angle=0.0,
                major_radius=0.5,
                minor_radius=0.5
            )
            obstacle.prediction.steps.append(step)
        
        obstacles.append(obstacle)
        return obstacles


class SafeHorizonIntegrationExample:
    """Example of how to integrate Safe Horizon Constraint with MPC."""
    
    def __init__(self):
        self.solver = CasADiSolver()
        self.module_manager = self.solver.get_module_manager()
        
        # Create Safe Horizon Constraint
        self.safe_horizon_constraint = SafeHorizonConstraint(self.solver)
        
        # Add to module manager
        self.module_manager.add_module(self.safe_horizon_constraint)
    
    def run_example(self):
        """Run a simple example with Safe Horizon Constraint."""
        print("Safe Horizon Constraint Integration Example")
        print("=" * 50)
        
        # Create test data
        data = self._create_example_data()
        
        # Check data readiness
        if not self.safe_horizon_constraint.is_data_ready(data):
            print("Data not ready for Safe Horizon Constraint")
            return
        
        print("Data is ready for Safe Horizon Constraint")
        
        # Get constraint info
        info = self.safe_horizon_constraint.get_constraint_info()
        print(f"Constraint info: {info}")
        
        # Validate parameters
        if self.safe_horizon_constraint.validate_parameters():
            print("Parameters are valid")
        else:
            print("Parameter validation failed")
        
        # Prepare iteration
        x_init = np.array([0.0, 0.0, 0.0, 0.0])  # Example initial state
        self.safe_horizon_constraint.prepare_iteration(x_init, data)
        print("Iteration prepared")
        
        # Simulate optimization (would normally be called by MPC solver)
        result = self.safe_horizon_constraint.optimize(None, data)
        if result == 1:
            print("Safe Horizon optimization successful")
        else:
            print("Safe Horizon optimization failed")
        
        # Post-solve processing
        x_solution = [np.array([i * 0.1, i * 0.1, 0.0, 0.0]) for i in range(10)]
        u_solution = [np.array([0.1, 0.0]) for i in range(10)]
        self.safe_horizon_constraint.post_solve(x_solution, u_solution)
        print("Post-solve processing completed")
    
    def _create_example_data(self):
        """Create example data for testing."""
        from planning.src.types import Data, DynamicObstacle, PredictionType, PredictionStep
        
        data = Data()
        
        # Create dynamic obstacles
        obstacles = []
        
        # Obstacle 1: Moving straight
        obstacle1 = DynamicObstacle(
            index=0,
            position=np.array([5.0, 0.0]),
            angle=0.0,
            radius=0.5
        )
        obstacle1.prediction.type = PredictionType.GAUSSIAN
        obstacle1.prediction.steps = []
        
        for i in range(10):
            step = PredictionStep(
                position=np.array([5.0 + i * 0.2, 0.0]),
                angle=0.0,
                major_radius=0.5,
                minor_radius=0.5
            )
            obstacle1.prediction.steps.append(step)
        
        obstacles.append(obstacle1)
        
        # Obstacle 2: Moving diagonally
        obstacle2 = DynamicObstacle(
            index=1,
            position=np.array([0.0, 5.0]),
            angle=np.pi/4,
            radius=0.3
        )
        obstacle2.prediction.type = PredictionType.GAUSSIAN
        obstacle2.prediction.steps = []
        
        for i in range(10):
            step = PredictionStep(
                position=np.array([i * 0.15, 5.0 - i * 0.15]),
                angle=np.pi/4,
                major_radius=0.3,
                minor_radius=0.3
            )
            obstacle2.prediction.steps.append(step)
        
        obstacles.append(obstacle2)
        
        data.dynamic_obstacles = obstacles
        return data


if __name__ == "__main__":
    # Run unit tests
    print("Running Safe Horizon Constraint Tests")
    print("=" * 40)
    
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration example
    print("\n" + "=" * 40)
    example = SafeHorizonIntegrationExample()
    example.run_example()
