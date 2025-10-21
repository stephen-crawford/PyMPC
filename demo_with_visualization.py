#!/usr/bin/env python3
"""
Comprehensive demo script showcasing the MPC framework with visualization and logging.

This script demonstrates how to use the visualization and logging framework
to easily set up tests with contouring objectives and arbitrary constraints.
"""

import sys
import os
import numpy as np

# Add the pympc module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pympc'))

from pympc import (
    MPCCPlanner, BicycleModel,
    LinearConstraints, EllipsoidConstraints, GaussianConstraints, ScenarioConstraints,
    ContouringObjective, GoalObjective
)
from utils.demo_framework import create_demo_framework
from utils.test_config import (
    TestConfigBuilder
)


def create_curving_road():
    """Create a challenging curving road reference path."""
    t = np.linspace(0, 8*np.pi, 150)
    x = 0.2 * t
    y = 2 * np.sin(0.2 * t) + 0.5 * np.sin(0.6 * t) + 0.1 * np.sin(2 * t)
    return np.column_stack([x, y])


def create_dynamic_obstacles_ellipsoid(time_steps):
    """Create dynamic ellipsoid obstacles."""
    obstacles = []
    for t in range(time_steps):
        # Multiple moving obstacles
        x1 = 1.0 + 0.3 * t
        y1 = 1.0 + 0.8 * np.sin(0.1 * t)
        obstacles.append({
            'center': np.array([x1, y1]),
            'shape': np.array([0.6, 0.3]),
            'rotation': 0.0
        })
        
        x2 = 3.0 + 0.1 * t
        y2 = 2.0 + 1.0 * np.cos(0.15 * t)
        obstacles.append({
            'center': np.array([x2, y2]),
            'shape': np.array([0.5, 0.4]),
            'rotation': np.pi/6
        })
    
    return obstacles


def create_dynamic_obstacles_gaussian(time_steps):
    """Create dynamic Gaussian obstacles."""
    obstacles = []
    for t in range(time_steps):
        x_mean = 2.0 + 0.2 * t
        y_mean = 1.5 + 0.5 * np.sin(0.12 * t)
        uncertainty = 0.1 + 0.02 * t
        
        obstacles.append({
            'mean': np.array([x_mean, y_mean]),
            'covariance': np.array([[uncertainty, 0.0], [0.0, uncertainty]]),
            'shape': np.array([0.5, 0.3])
        })
        
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
        x = 0.5 + 0.35 * t
        y = 1.5 + 0.6 * np.sin(0.15 * t)
        scenario1_obstacles.append({
            'center': np.array([x, y]),
            'shape': np.array([0.7, 0.4])
        })
    
    # Scenario 2: Conservative obstacles
    scenario2_obstacles = []
    for t in range(time_steps):
        x = 1.5 + 0.2 * t
        y = 2.5 + 0.3 * np.sin(0.08 * t)
        scenario2_obstacles.append({
            'center': np.array([x, y]),
            'shape': np.array([0.6, 0.3])
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


def mpc_solver(config):
    """
    MPC solver function that takes a configuration and returns results.
    
    Args:
        config: Test configuration
        
    Returns:
        Dictionary with optimization results
    """
    try:
        # Create dynamics
        if config.dynamics_type == "bicycle":
            dynamics = BicycleModel(dt=config.dt, **config.dynamics_params)
        else:
            raise ValueError(f"Unknown dynamics type: {config.dynamics_type}")
        
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
                
            elif obj_config['type'] == 'goal':
                goal_state = np.array(obj_config['goal_state'])
                objective = GoalObjective(
                    goal_state=goal_state,
                    goal_weight=obj_config.get('goal_weight', 1.0),
                    control_weight=obj_config.get('control_weight', 0.1),
                    terminal_weight=obj_config.get('terminal_weight', 10.0)
                )
                planner.add_objective(objective)
        
        # Add constraints
        for constr_config in config.constraints:
            if constr_config['type'] == 'linear':
                linear_constr = LinearConstraints()
                
                # State bounds
                if constr_config.get('state_bounds'):
                    state_bounds = constr_config['state_bounds']
                    state_min = np.array(state_bounds['min'])
                    state_max = np.array(state_bounds['max'])
                    linear_constr.add_state_bounds(state_min, state_max)
                
                # Control bounds
                if constr_config.get('control_bounds'):
                    control_bounds = constr_config['control_bounds']
                    control_min = np.array(control_bounds['min'])
                    control_max = np.array(control_bounds['max'])
                    linear_constr.add_control_bounds(control_min, control_max)
                
                planner.add_constraint(linear_constr)
                
            elif constr_config['type'] == 'ellipsoid':
                obstacles = constr_config['obstacles']
                safety_margin = constr_config.get('safety_margin', 0.3)
                
                ellipsoid_constr = EllipsoidConstraints(
                    obstacles=obstacles,
                    safety_margin=safety_margin
                )
                planner.add_constraint(ellipsoid_constr)
                
            elif constr_config['type'] == 'gaussian':
                uncertain_obstacles = constr_config['uncertain_obstacles']
                confidence_level = constr_config.get('confidence_level', 0.95)
                safety_margin = constr_config.get('safety_margin', 0.2)
                
                gaussian_constr = GaussianConstraints(
                    uncertain_obstacles=uncertain_obstacles,
                    confidence_level=confidence_level,
                    safety_margin=safety_margin
                )
                planner.add_constraint(gaussian_constr)
                
            elif constr_config['type'] == 'scenario':
                scenarios = constr_config['scenarios']
                scenario_weights = constr_config.get('scenario_weights', None)
                
                scenario_constr = ScenarioConstraints(
                    scenarios=scenarios,
                    scenario_weights=scenario_weights
                )
                planner.add_constraint(scenario_constr)
        
        # Solve
        initial_state = np.array(config.initial_state)
        reference_path = None
        
        # Extract reference path if available
        if config.reference_path:
            # Handle different reference path formats
            pass  # Implementation depends on reference path format
        
        solution = planner.solve(initial_state, reference_path=reference_path)
        
        if solution is not None and planner.is_feasible():
            return {
                'success': True,
                'states': solution['states'],
                'controls': solution['controls'],
                'solve_time': planner.get_solve_time(),
                'iterations': planner.get_iterations(),
                'objective_value': planner.get_objective_value(),
                'constraint_violations': planner.get_constraint_violations(),
                'num_variables': planner.get_num_variables(),
                'num_constraints': planner.get_num_constraints(),
                'convergence_status': 'optimal'
            }
        else:
            return {
                'success': False,
                'error': 'Optimization failed or infeasible'
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def demo_basic_usage():
    """Demonstrate basic usage of the framework."""
    print("=== Basic Usage Demo ===")
    
    # Create demo framework
    demo_framework = create_demo_framework(
        output_dir="demo_outputs",
        enable_logging=True,
        enable_visualization=True,
        enable_performance_monitoring=True
    )
    
    # Create a simple test configuration
    road_path = create_curving_road()
    obstacles = create_dynamic_obstacles_ellipsoid(25)
    
    config = (TestConfigBuilder("basic_contouring_demo")
              .set_mpc_params(horizon_length=25, dt=0.1)
              .add_contouring_objective(road_path)
              .add_linear_constraints(
                  state_bounds={
                      'min': [-30, -15, -np.pi, 0, -0.6],
                      'max': [30, 15, np.pi, 5, 0.6]
                  }
              )
              .add_ellipsoid_constraints(obstacles, safety_margin=0.4)
              .set_initial_state([0.0, 0.0, 0.0, 2.0, 0.0])
              .build())
    
    # Run demo
    result = demo_framework.run_demo(config, mpc_solver)
    
    print(f"Demo result: {result['success']}")
    print(f"Solve time: {result.get('solve_time', 0):.3f}s")
    
    # Cleanup
    demo_framework.cleanup()
    
    return result


def demo_predefined_configurations():
    """Demonstrate predefined configurations."""
    print("=== Predefined Configurations Demo ===")
    
    # Create demo framework
    demo_framework = create_demo_framework(
        output_dir="demo_outputs",
        enable_logging=True,
        enable_visualization=True,
        enable_performance_monitoring=True
    )
    
    # Run predefined demos
    demo_names = [
        "curving_road_ellipsoid",
        "curving_road_gaussian", 
        "curving_road_scenario",
        "goal_reaching",
        "combined_constraints"
    ]
    
    results = demo_framework.run_predefined_demos(demo_names, mpc_solver)
    
    # Print results
    for result in results:
        print(f"{result['demo_name']}: {'SUCCESS' if result['success'] else 'FAILED'}")
        if result['success']:
            print(f"  Solve time: {result.get('solve_time', 0):.3f}s")
    
    # Generate session report
    session_report = demo_framework.generate_session_report()
    print(f"\nSession Summary:")
    print(f"  Total demos: {session_report['total_demos']}")
    print(f"  Success rate: {session_report['success_rate']:.1f}%")
    print(f"  Average solve time: {session_report['average_solve_time']:.3f}s")
    
    # Cleanup
    demo_framework.cleanup()
    
    return results


def demo_custom_configuration():
    """Demonstrate custom configuration creation."""
    print("=== Custom Configuration Demo ===")
    
    # Create demo framework
    demo_framework = create_demo_framework(
        output_dir="demo_outputs",
        enable_logging=True,
        enable_visualization=True,
        enable_performance_monitoring=True
    )
    
    # Create custom configuration
    road_path = create_curving_road()
    obstacles = create_dynamic_obstacles_ellipsoid(30)
    uncertain_obstacles = create_dynamic_obstacles_gaussian(30)
    scenarios = create_dynamic_scenarios(30)
    
    config = (TestConfigBuilder("custom_combined_demo")
              .set_mpc_params(horizon_length=30, dt=0.1)
              .add_contouring_objective(road_path, progress_weight=2.0, contouring_weight=15.0)
              .add_linear_constraints(
                  state_bounds={
                      'min': [-50, -25, -np.pi, 0, -0.8],
                      'max': [50, 25, np.pi, 8, 0.8]
                  }
              )
              .add_ellipsoid_constraints(obstacles, safety_margin=0.5)
              .add_gaussian_constraints(uncertain_obstacles, confidence_level=0.99)
              .add_scenario_constraints(scenarios, scenario_weights=[0.6, 0.4])
              .set_initial_state([0.0, 0.0, 0.0, 3.0, 0.0])
              .set_solver_options(ipopt={'max_iter': 200})
              .set_visualization_options(create_animation=True)
              .build())
    
    # Run demo
    result = demo_framework.run_demo(config, mpc_solver)
    
    print(f"Custom demo result: {result['success']}")
    if result['success']:
        print(f"  Solve time: {result.get('solve_time', 0):.3f}s")
        print(f"  Iterations: {result.get('results', {}).get('iterations', 0)}")
        print(f"  Objective value: {result.get('results', {}).get('objective_value', 0):.3f}")
    
    # Cleanup
    demo_framework.cleanup()
    
    return result


def demo_benchmark_suite():
    """Demonstrate benchmark suite."""
    print("=== Benchmark Suite Demo ===")
    
    # Create demo framework
    demo_framework = create_demo_framework(
        output_dir="demo_outputs",
        enable_logging=True,
        enable_visualization=True,
        enable_performance_monitoring=True
    )
    
    # Create benchmark configurations
    benchmark_configs = []
    
    # Different horizon lengths
    for horizon in [15, 20, 25, 30]:
        road_path = create_curving_road()
        obstacles = create_dynamic_obstacles_ellipsoid(horizon)
        
        config = (TestConfigBuilder(f"benchmark_horizon_{horizon}")
                  .set_mpc_params(horizon_length=horizon, dt=0.1)
                  .add_contouring_objective(road_path)
                  .add_linear_constraints(
                      state_bounds={
                          'min': [-30, -15, -np.pi, 0, -0.6],
                          'max': [30, 15, np.pi, 5, 0.6]
                      }
                  )
                  .add_ellipsoid_constraints(obstacles, safety_margin=0.4)
                  .set_initial_state([0.0, 0.0, 0.0, 2.0, 0.0])
                  .build())
        
        benchmark_configs.append(config)
    
    # Run benchmark suite
    benchmark_results = demo_framework.run_benchmark_suite(mpc_solver, benchmark_configs)
    
    print(f"Benchmark Results:")
    print(f"  Total tests: {benchmark_results['total_tests']}")
    print(f"  Success rate: {benchmark_results['success_rate']:.1f}%")
    print(f"  Average solve time: {benchmark_results['average_solve_time']:.3f}s")
    print(f"  Min solve time: {benchmark_results['min_solve_time']:.3f}s")
    print(f"  Max solve time: {benchmark_results['max_solve_time']:.3f}s")
    
    # Cleanup
    demo_framework.cleanup()
    
    return benchmark_results


def demo_performance_analysis():
    """Demonstrate performance analysis."""
    print("=== Performance Analysis Demo ===")
    
    # Create demo framework
    demo_framework = create_demo_framework(
        output_dir="demo_outputs",
        enable_logging=True,
        enable_visualization=True,
        enable_performance_monitoring=True
    )
    
    # Run multiple demos for analysis
    demo_names = [
        "curving_road_ellipsoid",
        "curving_road_gaussian",
        "curving_road_scenario",
        "combined_constraints"
    ]
    
    results = demo_framework.run_predefined_demos(demo_names, mpc_solver)
    
    # Create comparison report
    comparison_report = demo_framework.create_comparison_report(
        demo_names, 
        metrics=['solve_time', 'success', 'iterations', 'objective_value']
    )
    
    print(f"Comparison Report:")
    print(f"  Demos compared: {comparison_report['demo_names']}")
    print(f"  Total demos: {comparison_report['summary']['total_demos']}")
    print(f"  Successful demos: {comparison_report['summary']['successful_demos']}")
    print(f"  Average solve time: {comparison_report['summary']['average_solve_time']:.3f}s")
    
    # Generate session report
    session_report = demo_framework.generate_session_report()
    
    # Cleanup
    demo_framework.cleanup()
    
    return comparison_report


def main():
    """Run all demos."""
    print("MPC Framework Demo with Visualization and Logging")
    print("=" * 60)
    print()
    
    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Predefined Configurations", demo_predefined_configurations),
        ("Custom Configuration", demo_custom_configuration),
        ("Benchmark Suite", demo_benchmark_suite),
        ("Performance Analysis", demo_performance_analysis)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        print(f"\n{'='*20} {demo_name} {'='*20}")
        try:
            result = demo_func()
            results[demo_name] = result
            print(f"✓ {demo_name} completed successfully")
        except Exception as e:
            print(f"✗ {demo_name} failed: {e}")
            results[demo_name] = None
    
    # Summary
    print(f"\n{'='*60}")
    print("DEMO SUMMARY")
    print(f"{'='*60}")
    
    successful_demos = sum(1 for r in results.values() if r is not None)
    total_demos = len(results)
    
    print(f"Total demos: {total_demos}")
    print(f"Successful demos: {successful_demos}")
    print(f"Success rate: {successful_demos/total_demos*100:.1f}%")
    
    if successful_demos == total_demos:
        print("🎉 All demos completed successfully!")
    else:
        print("⚠ Some demos failed. Check error messages above.")
    
    print(f"\nDemo outputs saved to: demo_outputs/")
    print("Check the session directories for detailed logs, plots, and performance data.")
    
    return results


if __name__ == "__main__":
    results = main()
    sys.exit(0 if all(r is not None for r in results.values()) else 1)
