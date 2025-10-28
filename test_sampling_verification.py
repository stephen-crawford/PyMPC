"""
Standalone test for Safe Horizon Constraint sampling functionality.
This test verifies that samples are properly taken from trajectory distributions.
"""
import unittest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Mock ROS dependencies
sys.modules['rclpy'] = MagicMock()
sys.modules['rclpy.node'] = MagicMock()
sys.modules['rclpy.qos'] = MagicMock()
sys.modules['geometry_msgs.msg'] = MagicMock()
sys.modules['nav_msgs.msg'] = MagicMock()
sys.modules['std_msgs.msg'] = MagicMock()

from planner_modules.src.constraints.scenario_utils.sampler import ScenarioSampler, MonteCarloValidator
from planner_modules.src.constraints.scenario_utils.math_utils import (
    compute_sample_size, linearize_collision_constraint, construct_free_space_polytope
)
from planning.src.types import DynamicObstacle, PredictionType, PredictionStep, Scenario


class TestScenarioSampling(unittest.TestCase):
    """Test scenario sampling functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sampler = ScenarioSampler(num_scenarios=100, enable_outlier_removal=True)
        self.validator = MonteCarloValidator(num_samples=1000)
    
    def test_sample_size_computation(self):
        """Test sample size computation using scenario optimization theory."""
        # Test with different parameters
        epsilon_p = 0.1
        beta = 0.01
        n_bar = 10
        
        sample_size = compute_sample_size(epsilon_p, beta, n_bar)
        
        # Should be a positive integer
        self.assertIsInstance(sample_size, int)
        self.assertGreater(sample_size, 0)
        
        # Should be reasonable (not too large)
        self.assertLess(sample_size, 10000)
        
        print(f"Sample size for ε={epsilon_p}, β={beta}, n̄={n_bar}: {sample_size}")
    
    def test_gaussian_scenario_sampling(self):
        """Test sampling from Gaussian obstacle predictions."""
        # Create a dynamic obstacle with Gaussian prediction
        obstacle = DynamicObstacle(
            index=0,
            position=np.array([5.0, 0.0]),
            angle=0.0,
            radius=0.5
        )
        
        obstacle.prediction.type = PredictionType.GAUSSIAN
        obstacle.prediction.steps = []
        
        # Create prediction steps with increasing uncertainty
        horizon_length = 5
        for i in range(horizon_length):
            step = PredictionStep(
                position=np.array([5.0 + i * 0.2, 0.0]),  # Moving forward
                angle=0.0,
                major_radius=0.5 + i * 0.1,  # Increasing uncertainty
                minor_radius=0.5 + i * 0.1
            )
            obstacle.prediction.steps.append(step)
        
        # Sample scenarios
        scenarios = self.sampler.sample_scenarios([obstacle], horizon_length, 0.1)
        
        # Verify we got scenarios
        self.assertGreater(len(scenarios), 0)
        print(f"Generated {len(scenarios)} scenarios from Gaussian prediction")
        
        # Verify scenario properties
        for scenario in scenarios[:5]:  # Check first 5 scenarios
            self.assertIsInstance(scenario, Scenario)
            self.assertEqual(scenario.obstacle_idx_, 0)
            self.assertIn(scenario.time_step, range(horizon_length))
            self.assertIsNotNone(scenario.position)
            self.assertEqual(len(scenario.position), 2)  # x, y coordinates
    
    def test_scenario_distribution_properties(self):
        """Test that scenarios follow the expected distribution properties."""
        # Create obstacle with known Gaussian parameters
        obstacle = DynamicObstacle(
            index=0,
            position=np.array([0.0, 0.0]),
            angle=0.0,
            radius=0.5
        )
        
        obstacle.prediction.type = PredictionType.GAUSSIAN
        obstacle.prediction.steps = []
        
        # Single prediction step with known mean and variance
        mean_pos = np.array([2.0, 1.0])
        sigma = 0.3
        
        step = PredictionStep(
            position=mean_pos,
            angle=0.0,
            major_radius=sigma,
            minor_radius=sigma
        )
        obstacle.prediction.steps.append(step)
        
        # Sample many scenarios
        num_samples = 1000
        self.sampler.num_scenarios = num_samples
        scenarios = self.sampler.sample_scenarios([obstacle], 1, 0.1)
        
        # Extract positions
        positions = np.array([s.position for s in scenarios if s.time_step == 0])
        
        # Verify statistical properties
        sample_mean = np.mean(positions, axis=0)
        sample_std = np.std(positions, axis=0)
        
        print(f"Expected mean: {mean_pos}")
        print(f"Sample mean: {sample_mean}")
        print(f"Expected std: {sigma}")
        print(f"Sample std: {sample_std}")
        
        # Mean should be close to expected (within 2 standard errors)
        mean_error = np.abs(sample_mean - mean_pos)
        self.assertLess(mean_error[0], 2 * sigma / np.sqrt(num_samples))
        self.assertLess(mean_error[1], 2 * sigma / np.sqrt(num_samples))
        
        # Standard deviation should be close to expected
        std_error = np.abs(sample_std - sigma)
        self.assertLess(std_error[0], sigma * 0.2)  # Within 20%
        self.assertLess(std_error[1], sigma * 0.2)
    
    def test_multiple_obstacles_sampling(self):
        """Test sampling from multiple obstacles."""
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
            
            # Create prediction steps
            for j in range(3):
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
        
        # Sample scenarios
        scenarios = self.sampler.sample_scenarios(obstacles, 3, 0.1)
        
        # Verify we got scenarios from all obstacles
        obstacle_indices = set(s.obstacle_idx_ for s in scenarios)
        self.assertEqual(obstacle_indices, {0, 1, 2})
        
        print(f"Generated {len(scenarios)} scenarios from {len(obstacles)} obstacles")
        print(f"Obstacle distribution: {[scenarios.count(s) for s in set(scenarios)]}")
    
    def test_outlier_removal(self):
        """Test outlier removal functionality."""
        # Create scenarios with some outliers
        scenarios = []
        
        # Normal scenarios around origin
        for i in range(50):
            scenario = Scenario(i, 0)
            scenario.position = np.random.normal([0.0, 0.0], 0.5, 2)
            scenario.time_step = 0
            scenario.radius = 0.5
            scenarios.append(scenario)
        
        # Add some outliers
        for i in range(10):
            scenario = Scenario(50 + i, 0)
            scenario.position = np.random.normal([0.0, 0.0], 3.0, 2)  # Far outliers
            scenario.time_step = 0
            scenario.radius = 0.5
            scenarios.append(scenario)
        
        # Apply outlier removal
        filtered_scenarios = self.sampler._remove_outliers(scenarios, outlier_threshold=2.0)
        
        # Should have fewer scenarios after filtering
        self.assertLess(len(filtered_scenarios), len(scenarios))
        self.assertGreater(len(filtered_scenarios), 0)
        
        print(f"Original scenarios: {len(scenarios)}")
        print(f"Filtered scenarios: {len(filtered_scenarios)}")
    
    def test_collision_constraint_formulation(self):
        """Test collision constraint formulation."""
        # Test with known positions
        robot_pos = np.array([0.0, 0.0])
        obstacle_pos = np.array([2.0, 0.0])
        robot_radius = 0.5
        obstacle_radius = 0.3
        
        constraint = linearize_collision_constraint(
            robot_pos, obstacle_pos, robot_radius, obstacle_radius
        )
        
        # Verify constraint properties
        self.assertEqual(constraint.a1, 1.0)  # Normalized direction
        self.assertEqual(constraint.a2, 0.0)
        self.assertAlmostEqual(constraint.b, 0.8)  # robot_radius + obstacle_radius
        
        print(f"Collision constraint: {constraint.a1}x + {constraint.a2}y <= {constraint.b}")
    
    def test_polytope_construction(self):
        """Test free-space polytope construction."""
        # Create test constraints
        constraints = []
        for i in range(5):
            constraint = linearize_collision_constraint(
                np.array([0.0, 0.0]), 
                np.array([i + 1.0, 0.0]), 
                0.5, 0.3
            )
            constraints.append(constraint)
        
        # Construct polytope
        polytope = construct_free_space_polytope(constraints)
        
        # Verify polytope properties
        self.assertIsNotNone(polytope)
        self.assertGreater(len(polytope.halfspaces), 0)
        self.assertEqual(len(polytope.halfspaces), len(constraints))
        
        print(f"Constructed polytope with {len(polytope.halfspaces)} halfspaces")
    
    def test_monte_carlo_validation(self):
        """Test Monte Carlo collision probability validation."""
        # Create simple robot trajectory
        robot_trajectory = [
            np.array([0.0, 0.0]),
            np.array([0.1, 0.0]),
            np.array([0.2, 0.0])
        ]
        
        # Create obstacle with prediction
        obstacle = DynamicObstacle(
            index=0,
            position=np.array([1.0, 0.0]),
            angle=0.0,
            radius=0.3
        )
        
        obstacle.prediction.type = PredictionType.GAUSSIAN
        obstacle.prediction.steps = []
        
        # Add prediction steps
        for i in range(3):
            step = PredictionStep(
                position=np.array([1.0 + i * 0.05, 0.0]),
                angle=0.0,
                major_radius=0.3,
                minor_radius=0.3
            )
            obstacle.prediction.steps.append(step)
        
        # Validate collision probability
        is_safe, probability = self.validator.validate_collision_probability(
            robot_trajectory, [obstacle], 0.5, 0.05
        )
        
        # Should be safe (low collision probability)
        self.assertTrue(is_safe)
        self.assertLess(probability, 0.1)
        
        print(f"Collision probability: {probability:.4f}, Safe: {is_safe}")


def run_sampling_verification():
    """Run comprehensive sampling verification."""
    print("=" * 60)
    print("SAFE HORIZON SAMPLING VERIFICATION")
    print("=" * 60)
    
    # Test sample size computation
    print("\n1. Testing Sample Size Computation:")
    epsilon_values = [0.05, 0.1, 0.2]
    beta_values = [0.01, 0.05, 0.1]
    n_bar_values = [5, 10, 20]
    
    for eps in epsilon_values:
        for beta in beta_values:
            for n_bar in n_bar_values:
                sample_size = compute_sample_size(eps, beta, n_bar)
                print(f"   ε={eps}, β={beta}, n̄={n_bar} → n={sample_size}")
    
    # Test scenario sampling
    print("\n2. Testing Scenario Sampling:")
    sampler = ScenarioSampler(num_scenarios=500, enable_outlier_removal=True)
    
    # Create test obstacle
    obstacle = DynamicObstacle(
        index=0,
        position=np.array([3.0, 2.0]),
        angle=np.pi/4,
        radius=0.4
    )
    
    obstacle.prediction.type = PredictionType.GAUSSIAN
    obstacle.prediction.steps = []
    
    # Create prediction steps with realistic uncertainty growth
    for i in range(5):
        step = PredictionStep(
            position=np.array([
                3.0 + i * 0.2 * np.cos(np.pi/4),
                2.0 + i * 0.2 * np.sin(np.pi/4)
            ]),
            angle=np.pi/4,
            major_radius=0.4 + i * 0.05,  # Growing uncertainty
            minor_radius=0.4 + i * 0.05
        )
        obstacle.prediction.steps.append(step)
    
    # Sample scenarios
    scenarios = sampler.sample_scenarios([obstacle], 5, 0.1)
    
    print(f"   Generated {len(scenarios)} scenarios")
    
    # Analyze distribution
    positions_by_step = {}
    for scenario in scenarios:
        if scenario.time_step not in positions_by_step:
            positions_by_step[scenario.time_step] = []
        positions_by_step[scenario.time_step].append(scenario.position)
    
    for step, positions in positions_by_step.items():
        positions = np.array(positions)
        mean_pos = np.mean(positions, axis=0)
        std_pos = np.std(positions, axis=0)
        print(f"   Step {step}: mean={mean_pos}, std={std_pos}")
    
    # Test collision constraint formulation
    print("\n3. Testing Collision Constraint Formulation:")
    robot_pos = np.array([0.0, 0.0])
    
    for i, scenario in enumerate(scenarios[:3]):
        constraint = linearize_collision_constraint(
            robot_pos, scenario.position, 0.5, scenario.radius
        )
        print(f"   Scenario {i}: {constraint.a1:.3f}x + {constraint.a2:.3f}y <= {constraint.b:.3f}")
    
    print("\n" + "=" * 60)
    print("SAMPLING VERIFICATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    # Run unit tests
    print("Running Safe Horizon Sampling Tests")
    print("=" * 40)
    
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run comprehensive verification
    run_sampling_verification()
