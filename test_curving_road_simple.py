#!/usr/bin/env python3
"""
Simple test runner for curving road scenarios without visualization.

This script runs comprehensive integration tests for a bicycle model
traveling along a curving road while avoiding dynamic obstacles
using different constraint types.
"""

import sys
import os
import numpy as np
import time

# Add the pympc module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pympc'))

from pympc import (
    MPCCPlanner, BicycleModel,
    LinearConstraints, EllipsoidConstraints, GaussianConstraints, ScenarioConstraints,
    ContouringObjective, GoalObjective
)


def create_curving_road():
    """Create a challenging curving road reference path."""
    # Create an S-curve road with varying curvature
    t = np.linspace(0, 6*np.pi, 100)
    x = 0.2 * t
    y = 2 * np.sin(0.2 * t) + 0.5 * np.sin(0.6 * t)
    
    return np.column_stack([x, y])


def create_dynamic_obstacles_ellipsoid(time_steps):
    """Create dynamic ellipsoid obstacles that cross the road."""
    obstacles = []
    
    # Create multiple moving obstacles
    for t in range(time_steps):
        # Obstacle 1: Moving across road
        x1 = 1.0 + 0.3 * t
        y1 = 1.0 + 0.8 * np.sin(0.1 * t)
        obstacles.append({
            'center': np.array([x1, y1]),
            'shape': np.array([0.6, 0.3]),
            'rotation': 0.0
        })
        
        # Obstacle 2: Oscillating obstacle
        x2 = 3.0 + 0.1 * t
        y2 = 2.0 + 1.0 * np.cos(0.15 * t)
        obstacles.append({
            'center': np.array([x2, y2]),
            'shape': np.array([0.5, 0.4]),
            'rotation': np.pi/6
        })
    
    return obstacles


def create_dynamic_obstacles_gaussian(time_steps):
    """Create dynamic Gaussian obstacles with uncertainty."""
    obstacles = []
    
    for t in range(time_steps):
        # Obstacle 1: Uncertain position
        x_mean = 2.0 + 0.2 * t
        y_mean = 1.5 + 0.5 * np.sin(0.12 * t)
        uncertainty = 0.1 + 0.02 * t  # Uncertainty increases with time
        
        obstacles.append({
            'mean': np.array([x_mean, y_mean]),
            'covariance': np.array([[uncertainty, 0.0], [0.0, uncertainty]]),
            'shape': np.array([0.5, 0.3])
        })
        
        # Obstacle 2: Correlated uncertainty
        x_mean = 4.0 + 0.15 * t
        y_mean = 2.5 + 0.4 * np.cos(0.08 * t)
        uncertainty = 0.08 + 0.03 * t
        
        obstacles.append({
            'mean': np.array([x_mean, y_mean]),
            'covariance': np.array([[uncertainty, 0.1*uncertainty], [0.1*uncertainty, uncertainty]]),
            'shape': np.array([0.4, 0.4])
        })
    
    return obstacles


def create_dynamic_scenarios(time_steps):
    """Create dynamic scenario obstacles."""
    scenarios = []
    
    # Scenario 1: Aggressive obstacles
    scenario1_obstacles = []
    for t in range(time_steps):
        # Fast moving obstacle
        x = 0.5 + 0.35 * t
        y = 1.5 + 0.6 * np.sin(0.15 * t)
        scenario1_obstacles.append({
            'center': np.array([x, y]),
            'shape': np.array([0.7, 0.4])
        })
        
        # Oscillating obstacle
        x = 3.0 + 0.1 * t
        y = 2.0 + 1.2 * np.cos(0.1 * t)
        scenario1_obstacles.append({
            'center': np.array([x, y]),
            'shape': np.array([0.5, 0.6])
        })
    
    # Scenario 2: Conservative obstacles
    scenario2_obstacles = []
    for t in range(time_steps):
        # Slow moving obstacle
        x = 1.5 + 0.2 * t
        y = 2.5 + 0.3 * np.sin(0.08 * t)
        scenario2_obstacles.append({
            'center': np.array([x, y]),
            'shape': np.array([0.6, 0.3])
        })
        
        # Stationary obstacle
        x = 4.0
        y = 1.0 + 0.2 * np.sin(0.05 * t)
        scenario2_obstacles.append({
            'center': np.array([x, y]),
            'shape': np.array([0.4, 0.5])
        })
    
    scenarios = [
        {
            'obstacles': scenario1_obstacles,
            'constraints': [],
            'probability': 0.5
        },
        {
            'obstacles': scenario2_obstacles,
            'constraints': [],
            'probability': 0.5
        }
    ]
    
    return scenarios


def setup_planner(road_path, horizon_length=25):
    """Set up basic planner with common settings."""
    dynamics = BicycleModel(dt=0.1)
    planner = MPCCPlanner(
        dynamics=dynamics,
        horizon_length=horizon_length,
        dt=0.1
    )
    
    # Add contouring objective
    contouring_obj = ContouringObjective(
        reference_path=road_path,
        progress_weight=1.0,
        contouring_weight=10.0,
        control_weight=0.1
    )
    planner.add_objective(contouring_obj)
    
    # Add state bounds
    state_min = np.array([-30, -15, -np.pi, 0, -0.6])
    state_max = np.array([30, 15, np.pi, 5, 0.6])
    
    linear_constraint = LinearConstraints()
    linear_constraint.add_state_bounds(state_min, state_max)
    planner.add_constraint(linear_constraint)
    
    return planner


def test_ellipsoid_constraints():
    """Test ellipsoid constraints on curving road."""
    print("=== Ellipsoid Constraints - Curving Road ===")
    
    # Create road and obstacles
    road_path = create_curving_road()
    obstacles = create_dynamic_obstacles_ellipsoid(25)
    
    # Set up planner
    planner = setup_planner(road_path, horizon_length=25)
    
    # Add ellipsoid constraints
    ellipsoid_constraint = EllipsoidConstraints(
        obstacles=obstacles,
        safety_margin=0.4
    )
    planner.add_constraint(ellipsoid_constraint)
    
    # Solve
    initial_state = np.array([0.0, 0.0, 0.0, 2.0, 0.0])
    start_time = time.time()
    solution = planner.solve(initial_state, reference_path=road_path)
    solve_time = time.time() - start_time
    
    if solution is not None and planner.is_feasible():
        print("✓ Ellipsoid constraints test passed")
        print(f"  Solve time: {solve_time:.3f} seconds")
        print(f"  Trajectory shape: {solution['states'].shape}")
        print(f"  Control shape: {solution['controls'].shape}")
        return True
    else:
        print("✗ Ellipsoid constraints test failed")
        return False


def test_gaussian_constraints():
    """Test Gaussian constraints on curving road."""
    print("=== Gaussian Constraints - Curving Road ===")
    
    # Create road and obstacles
    road_path = create_curving_road()
    uncertain_obstacles = create_dynamic_obstacles_gaussian(25)
    
    # Set up planner
    planner = setup_planner(road_path, horizon_length=25)
    
    # Add Gaussian constraints
    gaussian_constraint = GaussianConstraints(
        uncertain_obstacles=uncertain_obstacles,
        confidence_level=0.95,
        safety_margin=0.3
    )
    planner.add_constraint(gaussian_constraint)
    
    # Solve
    initial_state = np.array([0.0, 0.0, 0.0, 2.0, 0.0])
    start_time = time.time()
    solution = planner.solve(initial_state, reference_path=road_path)
    solve_time = time.time() - start_time
    
    if solution is not None and planner.is_feasible():
        print("✓ Gaussian constraints test passed")
        print(f"  Solve time: {solve_time:.3f} seconds")
        print(f"  Trajectory shape: {solution['states'].shape}")
        print(f"  Control shape: {solution['controls'].shape}")
        return True
    else:
        print("✗ Gaussian constraints test failed")
        return False


def test_scenario_constraints():
    """Test scenario constraints on curving road."""
    print("=== Scenario Constraints - Curving Road ===")
    
    # Create road and scenarios
    road_path = create_curving_road()
    scenarios = create_dynamic_scenarios(25)
    
    # Set up planner
    planner = setup_planner(road_path, horizon_length=25)
    
    # Add scenario constraints
    scenario_constraint = ScenarioConstraints(
        scenarios=scenarios,
        scenario_weights=[0.5, 0.5]
    )
    planner.add_constraint(scenario_constraint)
    
    # Solve
    initial_state = np.array([0.0, 0.0, 0.0, 2.0, 0.0])
    start_time = time.time()
    solution = planner.solve(initial_state, reference_path=road_path)
    solve_time = time.time() - start_time
    
    if solution is not None and planner.is_feasible():
        print("✓ Scenario constraints test passed")
        print(f"  Solve time: {solve_time:.3f} seconds")
        print(f"  Trajectory shape: {solution['states'].shape}")
        print(f"  Control shape: {solution['controls'].shape}")
        return True
    else:
        print("✗ Scenario constraints test failed")
        return False


def test_combined_constraints():
    """Test combined constraint types on curving road."""
    print("=== Combined Constraints - Curving Road ===")
    
    # Create road and obstacles
    road_path = create_curving_road()
    obstacles = create_dynamic_obstacles_ellipsoid(25)
    uncertain_obstacles = create_dynamic_obstacles_gaussian(25)
    scenarios = create_dynamic_scenarios(25)
    
    # Set up planner
    planner = setup_planner(road_path, horizon_length=25)
    
    # Add all constraint types
    ellipsoid_constraint = EllipsoidConstraints(
        obstacles=obstacles,
        safety_margin=0.3
    )
    planner.add_constraint(ellipsoid_constraint)
    
    gaussian_constraint = GaussianConstraints(
        uncertain_obstacles=uncertain_obstacles,
        confidence_level=0.95,
        safety_margin=0.2
    )
    planner.add_constraint(gaussian_constraint)
    
    scenario_constraint = ScenarioConstraints(
        scenarios=scenarios,
        scenario_weights=[0.5, 0.5]
    )
    planner.add_constraint(scenario_constraint)
    
    # Solve
    initial_state = np.array([0.0, 0.0, 0.0, 2.0, 0.0])
    start_time = time.time()
    solution = planner.solve(initial_state, reference_path=road_path)
    solve_time = time.time() - start_time
    
    if solution is not None and planner.is_feasible():
        print("✓ Combined constraints test passed")
        print(f"  Solve time: {solve_time:.3f} seconds")
        print(f"  Trajectory shape: {solution['states'].shape}")
        print(f"  Control shape: {solution['controls'].shape}")
        print(f"  Number of constraints: {planner.get_constraint_count()}")
        return True
    else:
        print("✗ Combined constraints test failed")
        return False


def test_goal_reaching():
    """Test goal reaching on curving road."""
    print("=== Goal Reaching - Curving Road ===")
    
    # Create road
    road_path = create_curving_road()
    obstacles = create_dynamic_obstacles_ellipsoid(25)
    
    # Set up planner with goal objective
    dynamics = BicycleModel(dt=0.1)
    planner = MPCCPlanner(
        dynamics=dynamics,
        horizon_length=25,
        dt=0.1
    )
    
    # Add goal objective
    goal_position = road_path[-1]
    goal_state = np.array([goal_position[0], goal_position[1], 0.0, 1.0, 0.0])
    
    goal_obj = GoalObjective(
        goal_state=goal_state,
        goal_weight=1.0,
        control_weight=0.1,
        terminal_weight=10.0
    )
    planner.add_objective(goal_obj)
    
    # Add state bounds
    state_min = np.array([-30, -15, -np.pi, 0, -0.6])
    state_max = np.array([30, 15, np.pi, 5, 0.6])
    
    linear_constraint = LinearConstraints()
    linear_constraint.add_state_bounds(state_min, state_max)
    planner.add_constraint(linear_constraint)
    
    # Add obstacles
    ellipsoid_constraint = EllipsoidConstraints(
        obstacles=obstacles,
        safety_margin=0.3
    )
    planner.add_constraint(ellipsoid_constraint)
    
    # Solve
    initial_state = np.array([0.0, 0.0, 0.0, 2.0, 0.0])
    start_time = time.time()
    solution = planner.solve(initial_state)
    solve_time = time.time() - start_time
    
    if solution is not None and planner.is_feasible():
        print("✓ Goal reaching test passed")
        print(f"  Solve time: {solve_time:.3f} seconds")
        print(f"  Trajectory shape: {solution['states'].shape}")
        print(f"  Control shape: {solution['controls'].shape}")
        
        # Check final position
        trajectory = solution['states']
        final_position = trajectory[:2, -1]
        goal_distance = np.linalg.norm(final_position - goal_position)
        print(f"  Final position: {final_position}")
        print(f"  Goal distance: {goal_distance:.3f}")
        
        return True
    else:
        print("✗ Goal reaching test failed")
        return False


def main():
    """Run all curving road tests."""
    print("Curving Road Integration Tests")
    print("=" * 50)
    print()
    
    tests = [
        test_ellipsoid_constraints,
        test_gaussian_constraints,
        test_scenario_constraints,
        test_combined_constraints,
        test_goal_reaching
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"Test failed: {e}")
            results.append(False)
            print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All curving road tests passed!")
    else:
        print("⚠ Some tests failed. Check error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
