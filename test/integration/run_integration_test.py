"""
Integration Test Runner

This script provides a command-line interface for running integration tests
using the standardized framework.
"""
import argparse
import sys
import os
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig, create_reference_path


def create_test_config(args) -> TestConfig:
    """Create test configuration from command line arguments."""
    return TestConfig(
        reference_path=create_reference_path(args.path_type, args.path_length),
        objective_module=args.objective,
        constraint_modules=args.constraints,
        vehicle_dynamics=args.vehicle,
        num_obstacles=args.obstacles,
        obstacle_dynamics=[args.obstacle_dynamics] * args.obstacles,
        test_name=args.name,
        duration=args.duration,
        timestep=args.timestep
    )


def run_predefined_test(test_name: str):
    """Run predefined test configurations."""
    framework = IntegrationTestFramework()
    
    predefined_tests = {
        "safe_horizon_basic": TestConfig(
            reference_path=create_reference_path("straight", 20.0),
            objective_module="contouring",
            constraint_modules=["safe_horizon", "contouring"],
            vehicle_dynamics="bicycle",
            num_obstacles=3,
            obstacle_dynamics=["gaussian", "gaussian", "gaussian"],
            test_name="Safe Horizon Basic Test",
            duration=10.0,
            timestep=0.1
        ),
        
        "safe_horizon_advanced": TestConfig(
            reference_path=create_reference_path("curve", 25.0),
            objective_module="contouring",
            constraint_modules=["safe_horizon", "contouring", "gaussian"],
            vehicle_dynamics="bicycle",
            num_obstacles=4,
            obstacle_dynamics=["gaussian", "gaussian", "deterministic", "gaussian"],
            test_name="Safe Horizon Advanced Test",
            duration=15.0,
            timestep=0.1
        ),
        
        "gaussian_constraints": TestConfig(
            reference_path=create_reference_path("straight", 18.0),
            objective_module="goal",
            constraint_modules=["gaussian", "contouring"],
            vehicle_dynamics="unicycle",
            num_obstacles=2,
            obstacle_dynamics=["gaussian", "deterministic"],
            test_name="Gaussian Constraints Test",
            duration=8.0,
            timestep=0.1
        ),
        
        "multi_constraint": TestConfig(
            reference_path=create_reference_path("s_curve", 22.0),
            objective_module="contouring",
            constraint_modules=["safe_horizon", "gaussian", "contouring", "linear"],
            vehicle_dynamics="bicycle",
            num_obstacles=5,
            obstacle_dynamics=["gaussian", "gaussian", "deterministic", "gaussian", "gaussian"],
            test_name="Multi-Constraint Test",
            duration=12.0,
            timestep=0.1
        ),
        
        "comparison_test": TestConfig(
            reference_path=create_reference_path("curve", 20.0),
            objective_module="contouring",
            constraint_modules=["safe_horizon", "contouring"],
            vehicle_dynamics="bicycle",
            num_obstacles=3,
            obstacle_dynamics=["gaussian", "gaussian", "gaussian"],
            test_name="Safe Horizon Comparison Test",
            duration=10.0,
            timestep=0.1
        )
    }
    
    if test_name not in predefined_tests:
        print(f"Unknown predefined test: {test_name}")
        print(f"Available tests: {', '.join(predefined_tests.keys())}")
        return None
        
    config = predefined_tests[test_name]
    print(f"Running predefined test: {test_name}")
    print(f"Test name: {config.test_name}")
    print(f"Constraints: {', '.join(config.constraint_modules)}")
    print(f"Obstacles: {config.num_obstacles}")
    print()
    
    result = framework.run_test(config)
    return result


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="PyMPC Integration Test Runner")
    
    # Test selection
    parser.add_argument("--test", "-t", type=str, 
                       help="Run predefined test (safe_horizon_basic, safe_horizon_advanced, gaussian_constraints, multi_constraint, comparison_test)")
    
    # Custom test parameters
    parser.add_argument("--name", "-n", type=str, default="Custom Integration Test",
                       help="Test name")
    parser.add_argument("--objective", "-o", type=str, default="contouring",
                       choices=["contouring", "goal"],
                       help="Objective module type")
    parser.add_argument("--constraints", "-c", nargs="+", default=["safe_horizon", "contouring"],
                       choices=["safe_horizon", "contouring", "gaussian", "linear", "ellipsoid"],
                       help="Constraint module types")
    parser.add_argument("--vehicle", "-v", type=str, default="bicycle",
                       choices=["bicycle", "unicycle"],
                       help="Vehicle dynamics model")
    parser.add_argument("--obstacles", type=int, default=3,
                       help="Number of obstacles")
    parser.add_argument("--obstacle-dynamics", type=str, default="gaussian",
                       choices=["gaussian", "deterministic"],
                       help="Obstacle dynamics type")
    
    # Path parameters
    parser.add_argument("--path-type", type=str, default="straight",
                       choices=["straight", "curve", "s_curve"],
                       help="Reference path type")
    parser.add_argument("--path-length", type=float, default=20.0,
                       help="Reference path length")
    
    # Simulation parameters
    parser.add_argument("--duration", "-d", type=float, default=10.0,
                       help="Simulation duration (seconds)")
    parser.add_argument("--timestep", type=float, default=0.1,
                       help="Simulation timestep (seconds)")
    
    # Output options
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--list-tests", action="store_true",
                       help="List available predefined tests")
    
    args = parser.parse_args()
    
    # List available tests
    if args.list_tests:
        print("Available predefined tests:")
        print("  safe_horizon_basic     - Basic Safe Horizon constraint test")
        print("  safe_horizon_advanced  - Advanced Safe Horizon with multiple constraints")
        print("  gaussian_constraints   - Traditional Gaussian constraints test")
        print("  multi_constraint      - Multiple constraint types test")
        print("  comparison_test       - Safe Horizon vs traditional comparison")
        return
    
    # Run predefined test
    if args.test:
        result = run_predefined_test(args.test)
        if result is None:
            return 1
            
        if result.success:
            print("✅ Test completed successfully")
            print(f"📁 Output folder: {result.output_folder}")
            return 0
        else:
            print("❌ Test failed")
            print(f"📁 Output folder: {result.output_folder}")
            return 1
    
    # Run custom test
    print("Running custom integration test...")
    framework = IntegrationTestFramework()
    config = create_test_config(args)
    
    print(f"Test Configuration:")
    print(f"  Name: {config.test_name}")
    print(f"  Objective: {config.objective_module}")
    print(f"  Constraints: {', '.join(config.constraint_modules)}")
    print(f"  Vehicle: {config.vehicle_dynamics}")
    print(f"  Obstacles: {config.num_obstacles} ({config.obstacle_dynamics[0]})")
    print(f"  Path: {args.path_type} ({args.path_length}m)")
    print(f"  Duration: {config.duration}s")
    print()
    
    result = framework.run_test(config)
    
    if result.success:
        print("✅ Custom test completed successfully")
        print(f"📁 Output folder: {result.output_folder}")
        print(f"📊 Vehicle states: {len(result.vehicle_states)}")
        print(f"⏱️  Avg computation time: {sum(result.computation_times)/len(result.computation_times):.3f}s")
        print(f"⚠️  Constraint violations: {sum(result.constraint_violations)}")
        return 0
    else:
        print("❌ Custom test failed")
        print(f"📁 Output folder: {result.output_folder}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
