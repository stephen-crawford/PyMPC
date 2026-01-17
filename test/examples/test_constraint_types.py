#!/usr/bin/env python3
"""
Example: Test all constraint types on a simple obstacle avoidance scenario.

This demonstrates how to use the simplified test framework to compare
different constraint formulations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from test.simple_framework import (
    run_test, run_comparison_test, TestScenario
)
from main import list_constraint_types


def main():
    print("Testing all constraint types on obstacle avoidance scenario")
    print("=" * 60)

    # Create a base scenario
    scenario = TestScenario(
        name="constraint_comparison",
        path_type="straight",
        path_length=20.0,
        num_obstacles=3,
        obstacle_speed=0.5,
        prediction_type="gaussian",
        duration=10.0,
    )

    # Compare all available constraint types
    results = run_comparison_test(scenario, verbose=True)

    # Analyze results
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    fastest = min(results.items(), key=lambda x: x[1].avg_solve_time)
    safest = min(results.items(), key=lambda x: x[1].collision_count)

    print(f"Fastest solver: {fastest[0]} ({fastest[1].avg_solve_time*1000:.1f}ms avg)")
    print(f"Fewest collisions: {safest[0]} ({safest[1].collision_count} collisions)")

    # Check if all passed
    all_passed = all(r.success for r in results.values())
    if all_passed:
        print("\nAll constraint types completed successfully!")
        return 0
    else:
        failed = [name for name, r in results.items() if not r.success]
        print(f"\nFailed constraint types: {failed}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
