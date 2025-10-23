#!/usr/bin/env python3
"""
Run MPC test suite.

This script runs the comprehensive test suite for MPC planning
with various dynamics models, constraints, and objectives.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_runner import MPCTestRunner, TestConfig, TestResult


def run_single_test(test_name: str, dynamics_type: str = "bicycle", 
                   constraint_type: str = "linear", num_obstacles: int = 3,
                   obstacle_type: str = "static") -> TestResult:
    """
    Run a single test.
    
    Args:
        test_name: Name of the test
        dynamics_type: Type of dynamics model
        constraint_type: Type of obstacle constraints
        num_obstacles: Number of obstacles
        obstacle_type: Type of obstacles
        
    Returns:
        Test result
    """
    config = TestConfig(
        test_name=test_name,
        dynamics_type=dynamics_type,
        constraint_type=constraint_type,
        num_obstacles=num_obstacles,
        obstacle_type=obstacle_type
    )
    
    runner = MPCTestRunner()
    result = runner.run_test(config)
    
    return result


def run_all_tests() -> None:
    """Run all test scenarios."""
    print("Running MPC Test Suite")
    print("=" * 50)
    
    runner = MPCTestRunner()
    results = runner.run_all_tests()
    
    print("\nTest Results Summary:")
    print("=" * 50)
    for result in results:
        status = "PASS" if result.success else "FAIL"
        print(f"{result.test_name}: {status} ({result.execution_time:.2f}s)")
        if not result.success:
            print(f"  Final error: {result.final_error:.2f}")
            print(f"  Constraint violations: {result.constraint_violations}")


def run_contouring_tests() -> None:
    """Run contouring control tests."""
    print("Running Contouring Control Tests")
    print("=" * 50)
    
    # Test 1: Overactuated system, no obstacles
    print("Test 1: Overactuated system, no obstacles")
    result1 = run_single_test("contouring_overactuated", "overactuated_unicycle", 
                             "linear", 0, "static")
    print(f"Result: {'PASS' if result1.success else 'FAIL'}")
    
    # Test 2: Car system, no obstacles
    print("Test 2: Car system, no obstacles")
    result2 = run_single_test("contouring_car", "bicycle", "linear", 0, "static")
    print(f"Result: {'PASS' if result2.success else 'FAIL'}")
    
    return [result1, result2]


def run_obstacle_avoidance_tests() -> None:
    """Run obstacle avoidance tests."""
    print("Running Obstacle Avoidance Tests")
    print("=" * 50)
    
    results = []
    
    # Test 3: Overactuated system, static obstacles, linear constraints
    print("Test 3: Overactuated system, static obstacles, linear constraints")
    result3 = run_single_test("static_linear_overactuated", "overactuated_unicycle", 
                             "linear", 3, "static")
    print(f"Result: {'PASS' if result3.success else 'FAIL'}")
    results.append(result3)
    
    # Test 4: Car system, static obstacles, linear constraints
    print("Test 4: Car system, static obstacles, linear constraints")
    result4 = run_single_test("static_linear_car", "bicycle", "linear", 3, "static")
    print(f"Result: {'PASS' if result4.success else 'FAIL'}")
    results.append(result4)
    
    # Test 5: Car system, static obstacles, ellipsoid constraints
    print("Test 5: Car system, static obstacles, ellipsoid constraints")
    result5 = run_single_test("static_ellipsoid_car", "bicycle", "ellipsoid", 3, "static")
    print(f"Result: {'PASS' if result5.success else 'FAIL'}")
    results.append(result5)
    
    # Test 6: Car system, dynamic obstacles, ellipsoid constraints
    print("Test 6: Car system, dynamic obstacles, ellipsoid constraints")
    result6 = run_single_test("dynamic_ellipsoid_car", "bicycle", "ellipsoid", 3, "dynamic")
    print(f"Result: {'PASS' if result6.success else 'FAIL'}")
    results.append(result6)
    
    # Test 7: Car system, dynamic obstacles, gaussian constraints
    print("Test 7: Car system, dynamic obstacles, gaussian constraints")
    result7 = run_single_test("dynamic_gaussian_car", "bicycle", "gaussian", 3, "dynamic")
    print(f"Result: {'PASS' if result7.success else 'FAIL'}")
    results.append(result7)
    
    # Test 8: Car system, dynamic obstacles, scenario constraints
    print("Test 8: Car system, dynamic obstacles, scenario constraints")
    result8 = run_single_test("dynamic_scenario_car", "bicycle", "scenario", 3, "dynamic")
    print(f"Result: {'PASS' if result8.success else 'FAIL'}")
    results.append(result8)
    
    return results


def run_robust_tests() -> None:
    """Run robust MPC tests with failure handling."""
    print("Running Robust MPC Tests")
    print("=" * 50)
    
    # Test car system with retry logic
    print("Testing car system with retry logic...")
    
    max_retries = 2
    for attempt in range(max_retries + 1):
        print(f"Attempt {attempt + 1}/{max_retries + 1}")
        
        result = run_single_test(f"robust_car_attempt_{attempt + 1}", "bicycle", 
                                "linear", 3, "static")
        
        if result.success:
            print(f"Car system succeeded on attempt {attempt + 1}")
            break
        else:
            print(f"Car system failed on attempt {attempt + 1}")
            if attempt == max_retries:
                print("Car system failed after all retries, switching to overactuated system")
                # Switch to overactuated system
                result = run_single_test("robust_overactuated_fallback", "overactuated_unicycle", 
                                      "linear", 3, "static")
                if result.success:
                    print("Overactuated system succeeded")
                else:
                    print("Overactuated system failed - check constraints and solver")
                    print("This indicates a problem with the MPC formulation")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MPC test suite')
    parser.add_argument('--test-type', choices=['all', 'contouring', 'obstacles', 'robust'], 
                       default='all', help='Type of tests to run')
    parser.add_argument('--output-dir', default='test_outputs', 
                       help='Output directory for test results')
    
    args = parser.parse_args()
    
    if args.test_type == 'all':
        run_all_tests()
    elif args.test_type == 'contouring':
        run_contouring_tests()
    elif args.test_type == 'obstacles':
        run_obstacle_avoidance_tests()
    elif args.test_type == 'robust':
        run_robust_tests()
    
    print("\nTest suite completed!")


if __name__ == "__main__":
    main()
