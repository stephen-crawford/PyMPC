#!/usr/bin/env python3
"""
Diagnostic tests to identify the cause of high MPC failure rate.

This script tests various aspects of the constraint formulation to identify
why the linearized constraints are causing more solver failures.

Run with: python test_constraint_diagnosis.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import casadi as cd
from typing import List, Tuple, Dict
import json
import time

# Import modules
from modules.constraints.scenario_utils.math_utils import ScenarioConstraint
from modules.constraints.scenario_utils.scenario_module import SafeHorizonModule
from planning.types import State, Data, Scenario

print("=" * 70)
print("CONSTRAINT DIAGNOSIS: Identifying MPC Failure Causes")
print("=" * 70)

# =============================================================================
# Test 1: Verify constraint direction matches guide.md
# =============================================================================
def test_constraint_direction():
    """
    Verify: n points FROM ego TO obstacle (guide.md Eq. 17-18)

    The constraint should push the robot AWAY from the obstacle.
    """
    print("\n[TEST 1] Constraint Direction Verification")
    print("-" * 50)

    # Setup: robot at origin, obstacle at (5, 0)
    robot_pos = np.array([0.0, 0.0])
    obstacle_pos = np.array([5.0, 0.0])
    robot_radius = 0.5
    obstacle_radius = 0.3
    safety_margin = robot_radius + obstacle_radius

    # Compute constraint as per guide.md
    diff = obstacle_pos - robot_pos  # FROM robot TO obstacle
    dist = np.linalg.norm(diff)
    n = diff / dist  # Normal pointing FROM robot TO obstacle

    a1, a2 = n[0], n[1]
    b = np.dot(n, obstacle_pos) - safety_margin

    print(f"Robot: {robot_pos}, Obstacle: {obstacle_pos}")
    print(f"Normal n = (obstacle - robot) / ||...|| = ({a1:.4f}, {a2:.4f})")
    print(f"Constraint: {a1:.4f}*x + {a2:.4f}*y <= {b:.4f}")

    # Test: moving TOWARD obstacle should INCREASE constraint value (toward violation)
    test_points = [
        (0.0, 0.0, "origin"),
        (2.0, 0.0, "closer to obstacle"),
        (4.0, 0.0, "very close"),
        (4.5, 0.0, "inside safety margin"),
    ]

    print("\nConstraint values (positive = violation):")
    prev_value = None
    all_correct = True
    for x, y, desc in test_points:
        value = a1 * x + a2 * y - b
        direction = ""
        if prev_value is not None:
            if value > prev_value:
                direction = "↑ (increasing toward violation)"
            else:
                direction = "↓ (decreasing - WRONG!)"
                all_correct = False
        prev_value = value
        print(f"  ({x:.1f}, {y:.1f}) {desc}: {value:.4f} {direction}")

    if all_correct:
        print("✓ Constraint direction is CORRECT (increases toward obstacle)")
    else:
        print("✗ Constraint direction is WRONG!")

    return all_correct


# =============================================================================
# Test 2: Compare linearized vs symbolic constraint values
# =============================================================================
def test_linearization_accuracy():
    """
    Compare constraint values between:
    - Linearized constraint (pre-computed at reference)
    - "True" constraint (computed at actual position)

    Large differences indicate linearization error.
    """
    print("\n[TEST 2] Linearization Accuracy")
    print("-" * 50)

    # Reference position (where constraint is linearized)
    ref_robot_pos = np.array([0.0, 0.0])

    # Obstacle position
    obstacle_pos = np.array([5.0, 3.0])
    robot_radius = 0.5
    obstacle_radius = 0.3
    safety_margin = robot_radius + obstacle_radius

    # Compute LINEARIZED constraint at reference
    diff_ref = obstacle_pos - ref_robot_pos
    dist_ref = np.linalg.norm(diff_ref)
    n_ref = diff_ref / dist_ref
    a1_lin, a2_lin = n_ref[0], n_ref[1]
    b_lin = np.dot(n_ref, obstacle_pos) - safety_margin

    print(f"Reference robot position: {ref_robot_pos}")
    print(f"Obstacle position: {obstacle_pos}")
    print(f"Linearized constraint: {a1_lin:.4f}*x + {a2_lin:.4f}*y <= {b_lin:.4f}")

    # Test at various actual robot positions
    test_positions = [
        (0.0, 0.0, "at reference"),
        (1.0, 0.0, "1m from reference"),
        (2.0, 1.0, "2.2m from reference"),
        (3.0, 2.0, "3.6m from reference"),
        (4.0, 2.5, "4.7m from reference (close to obstacle)"),
    ]

    print("\nComparing linearized vs true constraint:")
    print(f"{'Position':<20} {'Linearized':<12} {'True':<12} {'Error':<12} {'Status'}")
    print("-" * 70)

    max_error = 0
    for x, y, desc in test_positions:
        actual_pos = np.array([x, y])

        # Linearized constraint value
        lin_value = a1_lin * x + a2_lin * y - b_lin

        # "True" constraint value (recomputed at actual position)
        diff_true = obstacle_pos - actual_pos
        dist_true = np.linalg.norm(diff_true)
        if dist_true > 1e-6:
            n_true = diff_true / dist_true
            # True constraint: n_true · (actual - obstacle) <= -safety_margin
            # Which is: n_true · actual <= n_true · obstacle - safety_margin
            true_value = np.dot(n_true, actual_pos) - (np.dot(n_true, obstacle_pos) - safety_margin)
        else:
            true_value = safety_margin  # At obstacle = violation

        error = abs(lin_value - true_value)
        max_error = max(max_error, error)

        # Check if both agree on feasibility
        lin_feasible = lin_value <= 0
        true_feasible = true_value <= 0
        status = "✓" if lin_feasible == true_feasible else "✗ MISMATCH"

        print(f"({x:.1f}, {y:.1f}) {desc:<15} {lin_value:>10.4f}  {true_value:>10.4f}  {error:>10.4f}  {status}")

    print(f"\nMax linearization error: {max_error:.4f}")

    if max_error < 0.5:
        print("✓ Linearization error is acceptable")
    else:
        print("⚠ Linearization error is significant - may cause issues far from reference")

    return max_error


# =============================================================================
# Test 3: Check for constraint conflicts
# =============================================================================
def test_constraint_conflicts():
    """
    Check if multiple constraints from different obstacles can conflict,
    creating an infeasible region.
    """
    print("\n[TEST 3] Constraint Conflict Detection")
    print("-" * 50)

    robot_pos = np.array([2.0, 2.0])
    robot_radius = 0.5

    # Multiple obstacles surrounding the robot
    obstacles = [
        (np.array([0.0, 2.0]), 0.3, "left"),
        (np.array([4.0, 2.0]), 0.3, "right"),
        (np.array([2.0, 0.0]), 0.3, "below"),
        (np.array([2.0, 4.0]), 0.3, "above"),
    ]

    print(f"Robot at: {robot_pos}")
    print("Obstacles surrounding robot:")

    constraints = []
    for obs_pos, obs_radius, name in obstacles:
        safety_margin = robot_radius + obs_radius
        diff = obs_pos - robot_pos
        dist = np.linalg.norm(diff)

        if dist < 1e-6:
            continue

        n = diff / dist
        a1, a2 = n[0], n[1]
        b = np.dot(n, obs_pos) - safety_margin

        # Check if current position satisfies constraint
        value = a1 * robot_pos[0] + a2 * robot_pos[1] - b
        status = "satisfied" if value <= 0 else "VIOLATED"

        print(f"  {name}: obs at {obs_pos}, dist={dist:.2f}m, constraint value={value:.4f} ({status})")
        constraints.append((a1, a2, b, name))

    # Check if there's a feasible point
    # For 4 constraints pointing inward, the feasible region is limited
    print("\nChecking feasible region...")

    # Sample points around robot position
    feasible_count = 0
    total_samples = 100
    for _ in range(total_samples):
        test_x = robot_pos[0] + np.random.uniform(-1, 1)
        test_y = robot_pos[1] + np.random.uniform(-1, 1)

        all_satisfied = True
        for a1, a2, b, _ in constraints:
            if a1 * test_x + a2 * test_y - b > 0:
                all_satisfied = False
                break

        if all_satisfied:
            feasible_count += 1

    feasibility_ratio = feasible_count / total_samples
    print(f"Feasibility ratio in [-1,1] box around robot: {feasibility_ratio:.1%}")

    if feasibility_ratio > 0.5:
        print("✓ Constraints leave sufficient feasible space")
    elif feasibility_ratio > 0.1:
        print("⚠ Constraints are tight but feasible")
    else:
        print("✗ Constraints may be over-constraining!")

    return feasibility_ratio


# =============================================================================
# Test 4: Verify constraint computation in scenario module
# =============================================================================
def test_scenario_module_constraints():
    """
    Test that the scenario module computes constraints correctly.
    """
    print("\n[TEST 4] Scenario Module Constraint Computation")
    print("-" * 50)

    # Create mock solver
    class MockSolver:
        def __init__(self):
            self.warmstart_values = {
                'x': [0.0] * 11,
                'y': [0.0] * 11,
            }
            self.timestep = 0.1
            self.horizon = 10

    config = {
        "epsilon_p": 0.05,
        "beta": 0.01,
        "n_bar": 5,
        "robot_radius": 0.5,
        "horizon_length": 10,
        "max_constraints_per_disc": 5,
        "num_discs": 1,
        "num_scenarios": 100,
        "num_removal": 0,
        "timestep": 0.1
    }

    module = SafeHorizonModule(MockSolver(), config)

    # Create test scenarios
    scenarios = []
    obstacle_pos = np.array([5.0, 0.0])
    scenario = Scenario(idx=0, obstacle_idx=0)
    scenario.position = obstacle_pos
    scenario.radius = 0.3
    scenarios.append(scenario)

    # Compute constraints
    reference_robot_pos = np.array([0.0, 0.0])
    constraints = module._formulate_collision_constraints(
        scenarios, disc_id=0, step=0, reference_robot_pos=reference_robot_pos
    )

    print(f"Reference robot position: {reference_robot_pos}")
    print(f"Obstacle position: {obstacle_pos}")
    print(f"Generated {len(constraints)} constraints")

    if constraints:
        c = constraints[0]
        print(f"\nConstraint parameters:")
        print(f"  a1 = {c.a1:.6f}")
        print(f"  a2 = {c.a2:.6f}")
        print(f"  b  = {c.b:.6f}")

        # Verify against expected values
        diff = obstacle_pos - reference_robot_pos
        dist = np.linalg.norm(diff)
        expected_n = diff / dist
        expected_b = np.dot(expected_n, obstacle_pos) - (0.5 + 0.3)  # robot_radius + obstacle_radius

        print(f"\nExpected values:")
        print(f"  a1 = {expected_n[0]:.6f}")
        print(f"  a2 = {expected_n[1]:.6f}")
        print(f"  b  = {expected_b:.6f}")

        # Check if they match
        a1_match = abs(c.a1 - expected_n[0]) < 1e-6
        a2_match = abs(c.a2 - expected_n[1]) < 1e-6
        b_match = abs(c.b - expected_b) < 1e-6

        if a1_match and a2_match and b_match:
            print("\n✓ Constraint computation is CORRECT")
            return True
        else:
            print("\n✗ Constraint computation MISMATCH!")
            return False
    else:
        print("✗ No constraints generated!")
        return False


# =============================================================================
# Test 5: Test constraint at solver warmstart positions
# =============================================================================
def test_warmstart_feasibility():
    """
    Test if constraints are feasible at warmstart positions.
    Infeasibility at warmstart can cause solver failures.
    """
    print("\n[TEST 5] Warmstart Feasibility Check")
    print("-" * 50)

    # Simulate a scenario where warmstart might be infeasible
    # Robot moving toward obstacle, warmstart from previous solution

    # Previous position (warmstart)
    warmstart_pos = np.array([3.0, 0.0])

    # Obstacle position
    obstacle_pos = np.array([5.0, 0.0])
    robot_radius = 0.5
    obstacle_radius = 0.3
    safety_margin = robot_radius + obstacle_radius

    # Constraint linearized at CURRENT position (which might differ from warmstart)
    current_pos = np.array([3.5, 0.0])  # Robot has moved closer

    print(f"Warmstart position: {warmstart_pos}")
    print(f"Current position (linearization point): {current_pos}")
    print(f"Obstacle position: {obstacle_pos}")

    # Compute constraint at current position
    diff = obstacle_pos - current_pos
    dist = np.linalg.norm(diff)
    n = diff / dist
    a1, a2 = n[0], n[1]
    b = np.dot(n, obstacle_pos) - safety_margin

    print(f"\nConstraint linearized at current: {a1:.4f}*x + {a2:.4f}*y <= {b:.4f}")

    # Check feasibility at warmstart
    warmstart_value = a1 * warmstart_pos[0] + a2 * warmstart_pos[1] - b
    warmstart_feasible = warmstart_value <= 0

    # Check feasibility at current
    current_value = a1 * current_pos[0] + a2 * current_pos[1] - b
    current_feasible = current_value <= 0

    print(f"\nFeasibility check:")
    print(f"  At warmstart {warmstart_pos}: value={warmstart_value:.4f} ({'feasible' if warmstart_feasible else 'INFEASIBLE'})")
    print(f"  At current {current_pos}: value={current_value:.4f} ({'feasible' if current_feasible else 'INFEASIBLE'})")

    # Key insight: if linearized at current but warmstart is behind current,
    # the constraint might be more restrictive at warmstart
    if not warmstart_feasible and current_feasible:
        print("\n⚠ WARNING: Constraint feasible at current but INFEASIBLE at warmstart!")
        print("  This can cause solver initialization issues.")
        return False
    elif warmstart_feasible:
        print("\n✓ Warmstart is feasible")
        return True
    else:
        print("\n⚠ Both positions infeasible - robot too close to obstacle")
        return False


# =============================================================================
# Test 6: Numerical stability of constraint parameters
# =============================================================================
def test_numerical_stability():
    """
    Test numerical stability when robot is very close to or far from obstacles.
    """
    print("\n[TEST 6] Numerical Stability")
    print("-" * 50)

    robot_pos = np.array([0.0, 0.0])
    robot_radius = 0.5
    obstacle_radius = 0.3
    safety_margin = robot_radius + obstacle_radius

    # Test at various distances
    distances = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    print(f"{'Distance':<12} {'a1':<12} {'a2':<12} {'b':<12} {'|n|':<12}")
    print("-" * 60)

    all_stable = True
    for dist in distances:
        obstacle_pos = np.array([dist, 0.0])

        diff = obstacle_pos - robot_pos
        actual_dist = np.linalg.norm(diff)

        if actual_dist < 1e-10:
            print(f"{dist:<12.4f} SKIP (too close)")
            continue

        n = diff / actual_dist
        a1, a2 = n[0], n[1]
        b = np.dot(n, obstacle_pos) - safety_margin

        # Check if normal is unit length
        n_norm = np.sqrt(a1**2 + a2**2)

        stable = abs(n_norm - 1.0) < 1e-6
        if not stable:
            all_stable = False

        print(f"{dist:<12.4f} {a1:<12.6f} {a2:<12.6f} {b:<12.4f} {n_norm:<12.6f} {'✓' if stable else '✗'}")

    if all_stable:
        print("\n✓ Constraint parameters are numerically stable")
    else:
        print("\n✗ Numerical instability detected!")

    return all_stable


# =============================================================================
# Run all tests
# =============================================================================
def run_all_tests():
    results = {}

    results['direction'] = test_constraint_direction()
    results['linearization_error'] = test_linearization_accuracy()
    results['conflict_ratio'] = test_constraint_conflicts()
    results['scenario_module'] = test_scenario_module_constraints()
    results['warmstart'] = test_warmstart_feasibility()
    results['numerical'] = test_numerical_stability()

    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    print("\nTest Results:")
    for name, result in results.items():
        if isinstance(result, bool):
            status = "✓ PASS" if result else "✗ FAIL"
        elif isinstance(result, float):
            status = f"Value: {result:.4f}"
        else:
            status = str(result)
        print(f"  {name}: {status}")

    # Save results to file for later analysis
    with open('constraint_diagnosis_results.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, (int, float, np.floating)) else v
                   for k, v in results.items()}, f, indent=2)
    print("\nResults saved to constraint_diagnosis_results.json")

    return results


if __name__ == "__main__":
    run_all_tests()
