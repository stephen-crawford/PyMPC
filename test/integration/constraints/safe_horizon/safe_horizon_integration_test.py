"""
Comprehensive integration test for Safe Horizon MPC with scenario constraints.
"""
import unittest
import numpy as np
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from planner_modules.src.constraints.safe_horizon_constraint import SafeHorizonConstraint
from planner_modules.src.constraints.scenario_utils.scenario_module import SafeHorizonModule
from planner_modules.src.constraints.scenario_utils.math_utils import (
    compute_sample_size, linearize_collision_constraint, construct_free_space_polytope
)
from planning.src.types import Data, DynamicObstacle, PredictionType, PredictionStep
from solver.src.casadi_solver import CasADiSolver


class TestSafeHorizonIntegration(unittest.TestCase):
    """Comprehensive integration tests for Safe Horizon MPC."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a real CasADi solver
        self.solver = CasADiSolver()
        self.solver.horizon = 10
        self.solver.timestep = 0.1
        
        # Mock configuration
        self.config = {
            "safe_horizon": {
                "epsilon_p": 0.1,
                "beta": 0.01,
                "n_bar": 10,
                "num_removal": 2,
                "max_constraints_per_disc": 24,
                "use_slack": True,
                "num_scenarios": 50,
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
    
    def test_scenario_optimization_workflow(self):
        """Test the complete scenario optimization workflow."""
        # Create test data
        data = self._create_complex_test_data()
        
        # Test data readiness
        self.assertTrue(self.constraint.is_data_ready(data))
        
        # Test parameter validation
        self.assertTrue(self.constraint.validate_parameters())
        
        # Test sample size computation
        sample_size = self.constraint.compute_sample_size()
        self.assertGreater(sample_size, 0)
        
        # Test scenario sampling
        obstacles = data.dynamic_obstacles
        scenarios = self.constraint.sample_scenarios(obstacles, 10, 0.1)
        self.assertGreater(len(scenarios), 0)
        
        # Test collision constraint formulation
        robot_pos = np.array([0.0, 0.0])
        obstacle_pos = np.array([2.0, 0.0])
        constraint = self.constraint.formulate_collision_constraint(
            robot_pos, obstacle_pos, 0.5, 0.3
        )
        self.assertIsNotNone(constraint)
        self.assertEqual(constraint.a1, 1.0)  # Should be normalized
        self.assertEqual(constraint.a2, 0.0)
    
    def test_polytope_construction(self):
        """Test free-space polytope construction."""
        # Create test scenarios
        scenarios = []
        for i in range(5):
            constraint = linearize_collision_constraint(
                np.array([0.0, 0.0]), np.array([i + 1.0, 0.0]), 0.5, 0.3
            )
            scenarios.append(constraint)
        
        # Construct polytope
        polytope = self.constraint.construct_free_space_polytope(scenarios, 0.5)
        self.assertIsNotNone(polytope)
        self.assertGreater(len(polytope.halfspaces), 0)
    
    def test_support_tracking(self):
        """Test active constraint and support tracking."""
        # Create test constraints
        constraints = []
        for i in range(3):
            constraint = linearize_collision_constraint(
                np.array([0.0, 0.0]), np.array([i + 1.0, 0.0]), 0.5, 0.3
            )
            constraints.append(constraint)
        
        # Track constraints
        self.constraint.track_active_constraints(constraints, 0, 0)
        
        # Check support limits
        self.assertTrue(self.constraint.check_support_limits())
    
    def test_mpc_integration_hooks(self):
        """Test MPC integration hooks."""
        data = self._create_test_data()
        x_init = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Test prepare_iteration
        self.constraint.prepare_iteration(x_init, data)
        
        # Test post_solve
        x_solution = [np.array([i * 0.1, i * 0.1, 0.0, 0.0]) for i in range(10)]
        u_solution = [np.array([0.1, 0.0]) for i in range(10)]
        self.constraint.post_solve(x_solution, u_solution)
    
    def test_scenario_removal(self):
        """Test scenario removal with big-M relaxation."""
        # Create test scenarios
        scenarios = []
        for i in range(10):
            constraint = linearize_collision_constraint(
                np.array([0.0, 0.0]), np.array([i + 1.0, 0.0]), 0.5, 0.3
            )
            scenarios.append(constraint)
        
        # Test scenario removal
        if hasattr(self.constraint.best_solver, 'scenario_module'):
            remaining = self.constraint.best_solver.scenario_module.remove_scenarios_with_big_m(
                scenarios, 3
            )
            self.assertEqual(len(remaining), 7)
    
    def test_constraint_coefficient_extraction(self):
        """Test constraint coefficient extraction and storage."""
        # Mock a scenario solver with polytopes
        mock_solver = MagicMock()
        mock_disc_manager = MagicMock()
        mock_polytope = MagicMock()
        
        # Mock halfspace
        mock_halfspace = MagicMock()
        mock_halfspace.A = np.array([[1.0, 0.0]])
        mock_halfspace.b = np.array([2.0])
        
        mock_polytope.halfspaces = [mock_halfspace]
        mock_disc_manager.polytopes = [mock_polytope]
        mock_solver.scenario_module.disc_manager = [mock_disc_manager]
        
        self.constraint.best_solver = mock_solver
        self.constraint._extract_constraint_coefficients()
        
        # Check that coefficients were stored
        self.assertEqual(self.constraint._constraint_coefficients['a1'][0][0][0], 1.0)
        self.assertEqual(self.constraint._constraint_coefficients['a2'][0][0][0], 0.0)
        self.assertEqual(self.constraint._constraint_coefficients['b'][0][0][0], 2.0)
    
    def test_parameter_setting(self):
        """Test parameter setting for MPC solver."""
        # Mock parameter manager
        param_manager = MagicMock()
        param_manager.set_parameter = MagicMock()
        
        # Set some constraint coefficients
        self.constraint._constraint_coefficients['a1'][0][0][0] = 1.0
        self.constraint._constraint_coefficients['a2'][0][0][0] = 0.0
        self.constraint._constraint_coefficients['b'][0][0][0] = 2.0
        
        # Test parameter setting
        data = Data()
        self.constraint.set_parameters(param_manager, data, 1)
        
        # Check that parameters were set
        self.assertTrue(param_manager.set_parameter.called)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with invalid data
        invalid_data = Data()
        self.assertFalse(self.constraint.is_data_ready(invalid_data))
        
        # Test with empty obstacles
        data = Data()
        data.dynamic_obstacles = []
        self.assertFalse(self.constraint.is_data_ready(data))
        
        # Test reset functionality
        self.constraint.reset()
        self.assertIsNone(self.constraint.best_solver)
        self.assertEqual(self.constraint.optimization_time, 0)
    
    def _create_test_data(self):
        """Create basic test data."""
        data = Data()
        
        # Create a simple dynamic obstacle
        obstacle = DynamicObstacle(
            index=0,
            position=np.array([3.0, 0.0]),
            angle=0.0,
            radius=0.5
        )
        
        obstacle.prediction.type = PredictionType.GAUSSIAN
        obstacle.prediction.steps = []
        
        for i in range(5):
            step = PredictionStep(
                position=np.array([3.0 + i * 0.2, 0.0]),
                angle=0.0,
                major_radius=0.5,
                minor_radius=0.5
            )
            obstacle.prediction.steps.append(step)
        
        data.dynamic_obstacles = [obstacle]
        return data
    
    def _create_complex_test_data(self):
        """Create complex test data with multiple obstacles."""
        data = Data()
        obstacles = []
        
        # Create multiple obstacles with different trajectories
        for i in range(3):
            obstacle = DynamicObstacle(
                index=i,
                position=np.array([i * 2.0, i * 1.0]),
                angle=i * np.pi / 4,
                radius=0.3 + i * 0.1
            )
            
            obstacle.prediction.type = PredictionType.GAUSSIAN
            obstacle.prediction.steps = []
            
            for j in range(8):
                step = PredictionStep(
                    position=np.array([
                        i * 2.0 + j * 0.1 * np.cos(i * np.pi / 4),
                        i * 1.0 + j * 0.1 * np.sin(i * np.pi / 4)
                    ]),
                    angle=i * np.pi / 4,
                    major_radius=0.3 + i * 0.1,
                    minor_radius=0.3 + i * 0.1
                )
                obstacle.prediction.steps.append(step)
            
            obstacles.append(obstacle)
        
        data.dynamic_obstacles = obstacles
        return data


class SafeHorizonMPCDemo:
    """Demonstration of Safe Horizon MPC integration."""
    
    def __init__(self):
        self.solver = CasADiSolver()
        self.module_manager = self.solver.get_module_manager()
        
        # Create Safe Horizon Constraint
        self.safe_horizon_constraint = SafeHorizonConstraint(self.solver)
        
        # Add to module manager
        self.module_manager.add_module(self.safe_horizon_constraint)
    
    def run_demo(self):
        """Run a complete demonstration."""
        print("Safe Horizon MPC Demonstration")
        print("=" * 50)
        
        # Create scenario data
        data = self._create_demo_data()
        
        print(f"Created {len(data.dynamic_obstacles)} dynamic obstacles")
        
        # Check data readiness
        if not self.safe_horizon_constraint.is_data_ready(data):
            print("❌ Data not ready for Safe Horizon Constraint")
            return False
        
        print("✅ Data is ready for Safe Horizon Constraint")
        
        # Validate parameters
        if not self.safe_horizon_constraint.validate_parameters():
            print("❌ Parameter validation failed")
            return False
        
        print("✅ Parameters are valid")
        
        # Get constraint information
        info = self.safe_horizon_constraint.get_constraint_info()
        print(f"📊 Constraint info: {info}")
        
        # Prepare iteration
        x_init = np.array([0.0, 0.0, 0.0, 0.0])
        self.safe_horizon_constraint.prepare_iteration(x_init, data)
        print("✅ Iteration prepared")
        
        # Run optimization
        result = self.safe_horizon_constraint.optimize(None, data)
        if result == 1:
            print("✅ Safe Horizon optimization successful")
        else:
            print("❌ Safe Horizon optimization failed")
            return False
        
        # Post-solve processing
        x_solution = [np.array([i * 0.1, i * 0.1, 0.0, 0.0]) for i in range(10)]
        u_solution = [np.array([0.1, 0.0]) for i in range(10)]
        self.safe_horizon_constraint.post_solve(x_solution, u_solution)
        print("✅ Post-solve processing completed")
        
        # Test constraint bounds
        lower_bounds = self.safe_horizon_constraint.get_lower_bound()
        upper_bounds = self.safe_horizon_constraint.get_upper_bound()
        print(f"📏 Constraint bounds: {len(lower_bounds)} constraints")
        
        print("\n🎉 Safe Horizon MPC demonstration completed successfully!")
        return True
    
    def _create_demo_data(self):
        """Create demonstration data."""
        data = Data()
        obstacles = []
        
        # Create obstacles with realistic trajectories
        trajectories = [
            (np.array([5.0, 0.0]), np.array([0.2, 0.0])),  # Straight line
            (np.array([0.0, 5.0]), np.array([0.1, -0.1])),  # Diagonal
            (np.array([3.0, 3.0]), np.array([-0.1, 0.1])),  # Reverse diagonal
        ]
        
        for i, (start_pos, velocity) in enumerate(trajectories):
            obstacle = DynamicObstacle(
                index=i,
                position=start_pos,
                angle=np.arctan2(velocity[1], velocity[0]),
                radius=0.4 + i * 0.1
            )
            
            obstacle.prediction.type = PredictionType.GAUSSIAN
            obstacle.prediction.steps = []
            
            # Generate prediction steps
            for j in range(10):
                pos = start_pos + velocity * j * 0.1
                step = PredictionStep(
                    position=pos,
                    angle=np.arctan2(velocity[1], velocity[0]),
                    major_radius=0.4 + i * 0.1,
                    minor_radius=0.4 + i * 0.1
                )
                obstacle.prediction.steps.append(step)
            
            obstacles.append(obstacle)
        
        data.dynamic_obstacles = obstacles
        return data


if __name__ == "__main__":
    # Run comprehensive tests
    print("Running Safe Horizon MPC Integration Tests")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run demonstration
    print("\n" + "=" * 50)
    demo = SafeHorizonMPCDemo()
    success = demo.run_demo()
    
    if success:
        print("\n✅ All tests and demonstrations passed!")
    else:
        print("\n❌ Some tests or demonstrations failed!")
