#!/usr/bin/env python3
"""
Simple Test Runner CLI for PyMPC.

Usage:
    # List available tests and constraint types
    python test/run_tests.py --list

    # Run a specific predefined scenario
    python test/run_tests.py --scenario obstacle_avoidance

    # Run with a specific constraint type
    python test/run_tests.py --scenario basic_straight --constraint-type gaussian

    # Compare all constraint types on a scenario
    python test/run_tests.py --scenario obstacle_avoidance --compare

    # Custom test
    python test/run_tests.py --obstacles 5 --duration 15 --constraint-type scenario
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test.simple_framework import (
    run_test, run_comparison_test,
    TestScenario, TestResult,
    get_scenario, list_scenarios, SCENARIOS
)
from main import list_constraint_types


def main():
    parser = argparse.ArgumentParser(
        description="PyMPC Simple Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Test selection
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        help="Predefined scenario to run (use --list to see available)"
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available scenarios and constraint types"
    )

    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare all constraint types on the given scenario"
    )

    # Custom test parameters
    parser.add_argument(
        "--constraint-type", "-t",
        type=str,
        default="linearized",
        help="Constraint type to use (default: linearized)"
    )

    parser.add_argument(
        "--obstacles", "-o",
        type=int,
        default=0,
        help="Number of obstacles (default: 0)"
    )

    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=10.0,
        help="Simulation duration in seconds (default: 10.0)"
    )

    parser.add_argument(
        "--path-type",
        type=str,
        default="straight",
        choices=["straight", "curved", "s-turn"],
        help="Reference path type (default: straight)"
    )

    parser.add_argument(
        "--path-length",
        type=float,
        default=20.0,
        help="Path length in meters (default: 20.0)"
    )

    parser.add_argument(
        "--obstacle-speed",
        type=float,
        default=1.0,
        help="Obstacle speed (default: 1.0)"
    )

    parser.add_argument(
        "--prediction-type",
        type=str,
        default="gaussian",
        choices=["gaussian", "deterministic"],
        help="Obstacle prediction type (default: gaussian)"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("Available Scenarios:")
        print("-" * 40)
        for name, scenario in SCENARIOS.items():
            print(f"  {name:20} - {scenario.num_obstacles} obstacles, "
                  f"{scenario.constraint_type} constraints")
        print()
        print("Available Constraint Types:")
        print("-" * 40)
        for ctype in list_constraint_types():
            print(f"  {ctype}")
        return 0

    # Get or create scenario
    if args.scenario:
        try:
            scenario = get_scenario(args.scenario)
            # Override with command line args if provided
            if args.constraint_type != "linearized":
                scenario.constraint_type = args.constraint_type
            if args.obstacles > 0:
                scenario.num_obstacles = args.obstacles
            if args.duration != 10.0:
                scenario.duration = args.duration
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    else:
        # Create custom scenario from args
        scenario = TestScenario(
            name="custom_test",
            constraint_type=args.constraint_type,
            path_type=args.path_type,
            path_length=args.path_length,
            num_obstacles=args.obstacles,
            obstacle_speed=args.obstacle_speed,
            prediction_type=args.prediction_type,
            duration=args.duration,
        )

    verbose = not args.quiet

    # Run comparison or single test
    if args.compare:
        results = run_comparison_test(scenario, verbose=verbose)
        all_passed = all(r.success for r in results.values())
        return 0 if all_passed else 1
    else:
        result = run_test(scenario, verbose=verbose)
        return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
