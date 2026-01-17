#!/usr/bin/env python3
"""
Example: Basic test scenarios.

Demonstrates simple usage of the test framework.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from test.simple_framework import run_test, TestScenario, get_scenario


def test_straight_path_no_obstacles():
    """Test basic straight path with no obstacles."""
    print("\n--- Test: Straight path, no obstacles ---")

    scenario = TestScenario(
        name="straight_no_obstacles",
        constraint_type="linearized",
        path_type="straight",
        path_length=15.0,
        num_obstacles=0,
        duration=8.0,
    )

    result = run_test(scenario, verbose=True)
    return result.success


def test_curved_path_no_obstacles():
    """Test curved path with no obstacles."""
    print("\n--- Test: Curved path, no obstacles ---")

    scenario = TestScenario(
        name="curved_no_obstacles",
        constraint_type="linearized",
        path_type="curved",
        path_length=20.0,
        num_obstacles=0,
        duration=10.0,
    )

    result = run_test(scenario, verbose=True)
    return result.success


def test_predefined_scenario():
    """Test using a predefined scenario."""
    print("\n--- Test: Predefined 'basic_straight' scenario ---")

    scenario = get_scenario("basic_straight")
    result = run_test(scenario, verbose=True)
    return result.success


def main():
    print("Running basic test examples")
    print("=" * 50)

    tests = [
        ("Straight path no obstacles", test_straight_path_no_obstacles),
        ("Curved path no obstacles", test_curved_path_no_obstacles),
        ("Predefined scenario", test_predefined_scenario),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    for name, passed, error in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if error:
            print(f"    Error: {error}")

    num_passed = sum(1 for _, p, _ in results if p)
    print(f"\n{num_passed}/{len(results)} tests passed")

    return 0 if num_passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
