#!/usr/bin/env python3
"""
Simple example demonstrating how easy it is to set up MPC tests
with contouring objectives and arbitrary constraints.
"""

import sys
import os
import numpy as np

# Add the pympc module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pympc'))

from utils.demo_framework import create_demo_framework
from utils.test_config import TestConfigBuilder, PredefinedTestConfigs


def mpc_solver(config):
    """Simple MPC solver function."""
    try:
        from pympc import MPCCPlanner, BicycleModel, ContouringObjective, LinearConstraints, EllipsoidConstraints
        
        # Create dynamics
        dynamics = BicycleModel(dt=config.dt, wheelbase=2.5)
        
        # Create planner
        planner = MPCCPlanner(
            dynamics=dynamics,
            horizon_length=config.horizon_length,
            dt=config.dt
        )
        
        # Add objectives
        for obj_config in config.objectives:
            if obj_config['type'] == 'contouring':
                reference_path = np.array(obj_config['reference_path'])
                objective = ContouringObjective(
                    reference_path=reference_path,
                    progress_weight=obj_config.get('progress_weight', 1.0),
                    contouring_weight=obj_config.get('contouring_weight', 10.0),
                    control_weight=obj_config.get('control_weight', 0.1)
                )
                planner.add_objective(objective)
        
        # Add constraints
        for constr_config in config.constraints:
            if constr_config['type'] == 'linear':
                linear_constr = LinearConstraints()
                if constr_config.get('state_bounds'):
                    state_bounds = constr_config['state_bounds']
                    state_min = np.array(state_bounds['min'])
                    state_max = np.array(state_bounds['max'])
                    linear_constr.add_state_bounds(state_min, state_max)
                planner.add_constraint(linear_constr)
                
            elif constr_config['type'] == 'ellipsoid':
                obstacles = constr_config['obstacles']
                safety_margin = constr_config.get('safety_margin', 0.3)
                ellipsoid_constr = EllipsoidConstraints(
                    obstacles=obstacles,
                    safety_margin=safety_margin
                )
                planner.add_constraint(ellipsoid_constr)
        
        # Solve
        initial_state = np.array(config.initial_state)
        solution = planner.solve(initial_state)
        
        if solution is not None and planner.is_feasible():
            return {
                'success': True,
                'states': solution['states'],
                'controls': solution['controls'],
                'solve_time': planner.get_solve_time(),
                'iterations': planner.get_iterations(),
                'objective_value': planner.get_objective_value()
            }
        else:
            return {'success': False, 'error': 'Optimization failed'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}


def example_1_simple_contouring():
    """Example 1: Simple contouring control with obstacles."""
    print("=== Example 1: Simple Contouring Control ===")
    
    # Create demo framework
    demo = create_demo_framework(
        output_dir="simple_demo_outputs",
        enable_logging=True,
        enable_visualization=True,
        enable_performance_monitoring=True
    )
    
    # Create a curving road
    t = np.linspace(0, 4*np.pi, 100)
    x = 0.3 * t
    y = 2 * np.sin(0.3 * t)
    road_path = np.column_stack([x, y])
    
    # Create some obstacles
    obstacles = []
    for i in range(20):
        x_obs = 1.0 + 0.4 * i
        y_obs = 1.0 + 0.5 * np.sin(0.1 * i)
        obstacles.append({
            'center': np.array([x_obs, y_obs]),
            'shape': np.array([0.5, 0.3]),
            'rotation': 0.0
        })
    
    # Create test configuration
    config = (TestConfigBuilder("simple_contouring")
              .set_mpc_params(horizon_length=20, dt=0.1)
              .add_contouring_objective(road_path)
              .add_linear_constraints(
                  state_bounds={
                      'min': [-20, -10, -np.pi, 0, -0.6],
                      'max': [20, 10, np.pi, 4, 0.6]
                  }
              )
              .add_ellipsoid_constraints(obstacles, safety_margin=0.3)
              .set_initial_state([0.0, 0.0, 0.0, 1.5, 0.0])
              .build())
    
    # Run demo
    result = demo.run_demo(config, mpc_solver)
    
    print(f"Result: {'SUCCESS' if result['success'] else 'FAILED'}")
    if result['success']:
        print(f"Solve time: {result.get('solve_time', 0):.3f}s")
        print(f"Iterations: {result.get('results', {}).get('iterations', 0)}")
    
    # Cleanup
    demo.cleanup()
    
    return result


def example_2_predefined_configuration():
    """Example 2: Using predefined configurations."""
    print("=== Example 2: Predefined Configuration ===")
    
    # Create demo framework
    demo = create_demo_framework(
        output_dir="simple_demo_outputs",
        enable_logging=True,
        enable_visualization=True,
        enable_performance_monitoring=True
    )
    
    # Use predefined configuration
    config = PredefinedTestConfigs.curving_road_ellipsoid()
    
    # Run demo
    result = demo.run_demo(config, mpc_solver)
    
    print(f"Result: {'SUCCESS' if result['success'] else 'FAILED'}")
    if result['success']:
        print(f"Solve time: {result.get('solve_time', 0):.3f}s")
        print(f"Objective value: {result.get('results', {}).get('objective_value', 0):.3f}")
    
    # Cleanup
    demo.cleanup()
    
    return result


def example_3_multiple_constraints():
    """Example 3: Multiple constraint types."""
    print("=== Example 3: Multiple Constraint Types ===")
    
    # Create demo framework
    demo = create_demo_framework(
        output_dir="simple_demo_outputs",
        enable_logging=True,
        enable_visualization=True,
        enable_performance_monitoring=True
    )
    
    # Create road
    t = np.linspace(0, 6*np.pi, 100)
    x = 0.2 * t
    y = 2 * np.sin(0.2 * t) + 0.5 * np.sin(0.6 * t)
    road_path = np.column_stack([x, y])
    
    # Create ellipsoid obstacles
    ellipsoid_obstacles = []
    for i in range(25):
        x_obs = 1.0 + 0.3 * i
        y_obs = 1.0 + 0.8 * np.sin(0.1 * i)
        ellipsoid_obstacles.append({
            'center': np.array([x_obs, y_obs]),
            'shape': np.array([0.6, 0.3]),
            'rotation': 0.0
        })
    
    # Create uncertain obstacles
    uncertain_obstacles = []
    for i in range(25):
        x_mean = 2.0 + 0.2 * i
        y_mean = 1.5 + 0.5 * np.sin(0.12 * i)
        uncertainty = 0.1 + 0.02 * i
        uncertain_obstacles.append({
            'mean': np.array([x_mean, y_mean]),
            'covariance': np.array([[uncertainty, 0.0], [0.0, uncertainty]]),
            'shape': np.array([0.5, 0.3])
        })
    
    # Create configuration with multiple constraints
    config = (TestConfigBuilder("multiple_constraints")
              .set_mpc_params(horizon_length=25, dt=0.1)
              .add_contouring_objective(road_path)
              .add_linear_constraints(
                  state_bounds={
                      'min': [-30, -15, -np.pi, 0, -0.6],
                      'max': [30, 15, np.pi, 5, 0.6]
                  }
              )
              .add_ellipsoid_constraints(ellipsoid_obstacles, safety_margin=0.3)
              .add_gaussian_constraints(uncertain_obstacles, confidence_level=0.95)
              .set_initial_state([0.0, 0.0, 0.0, 2.0, 0.0])
              .build())
    
    # Run demo
    result = demo.run_demo(config, mpc_solver)
    
    print(f"Result: {'SUCCESS' if result['success'] else 'FAILED'}")
    if result['success']:
        print(f"Solve time: {result.get('solve_time', 0):.3f}s")
        print(f"Number of constraints: {len(config.constraints)}")
    
    # Cleanup
    demo.cleanup()
    
    return result


def example_4_benchmark_comparison():
    """Example 4: Benchmark comparison."""
    print("=== Example 4: Benchmark Comparison ===")
    
    # Create demo framework
    demo = create_demo_framework(
        output_dir="simple_demo_outputs",
        enable_logging=True,
        enable_visualization=True,
        enable_performance_monitoring=True
    )
    
    # Run multiple predefined demos
    demo_names = [
        "curving_road_ellipsoid",
        "curving_road_gaussian",
        "curving_road_scenario"
    ]
    
    results = demo.run_predefined_demos(demo_names, mpc_solver)
    
    # Print comparison
    print("Benchmark Results:")
    for result in results:
        status = "SUCCESS" if result['success'] else "FAILED"
        solve_time = result.get('solve_time', 0)
        print(f"  {result['demo_name']}: {status} ({solve_time:.3f}s)")
    
    # Generate comparison report
    comparison_report = demo.create_comparison_report(demo_names)
    
    print(f"\nComparison Summary:")
    print(f"  Total demos: {comparison_report['summary']['total_demos']}")
    print(f"  Success rate: {comparison_report['summary']['success_rate']:.1f}%")
    print(f"  Average solve time: {comparison_report['summary']['average_solve_time']:.3f}s")
    
    # Cleanup
    demo.cleanup()
    
    return results


def main():
    """Run all examples."""
    print("Simple MPC Demo Examples")
    print("=" * 40)
    print()
    
    examples = [
        ("Simple Contouring", example_1_simple_contouring),
        ("Predefined Configuration", example_2_predefined_configuration),
        ("Multiple Constraints", example_3_multiple_constraints),
        ("Benchmark Comparison", example_4_benchmark_comparison)
    ]
    
    results = {}
    
    for example_name, example_func in examples:
        print(f"\n{'-'*20} {example_name} {'-'*20}")
        try:
            result = example_func()
            results[example_name] = result
            print(f"✓ {example_name} completed")
        except Exception as e:
            print(f"✗ {example_name} failed: {e}")
            results[example_name] = None
    
    # Summary
    print(f"\n{'='*40}")
    print("SUMMARY")
    print(f"{'='*40}")
    
    successful_examples = sum(1 for r in results.values() if r is not None)
    total_examples = len(results)
    
    print(f"Total examples: {total_examples}")
    print(f"Successful examples: {successful_examples}")
    print(f"Success rate: {successful_examples/total_examples*100:.1f}%")
    
    if successful_examples == total_examples:
        print("🎉 All examples completed successfully!")
    else:
        print("⚠ Some examples failed. Check error messages above.")
    
    print(f"\nDemo outputs saved to: simple_demo_outputs/")
    print("Check the session directories for detailed logs, plots, and performance data.")
    
    return results


if __name__ == "__main__":
    results = main()
    sys.exit(0 if all(r is not None for r in results.values()) else 1)
