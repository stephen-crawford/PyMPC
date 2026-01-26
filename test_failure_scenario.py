#!/usr/bin/env python3
"""
Test the exact failure scenario from the integration test.

Step 18 (SUCCESS): vehicle at (3.917, 1.934)
Step 19 (FAILURE): vehicle at (4.193, 2.036)

This script investigates what changes between these steps.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import casadi as cd

print("=" * 70)
print("FAILURE SCENARIO ANALYSIS")
print("=" * 70)

# From the integration test logs
step_18_pos = np.array([3.917, 1.934])
step_19_pos = np.array([4.193, 2.036])

print(f"\nStep 18 (SUCCESS): vehicle at {step_18_pos}")
print(f"Step 19 (FAILURE): vehicle at {step_19_pos}")
print(f"Movement: {step_19_pos - step_18_pos}")

# =============================================================================
# Test: Simulate constraint computation for both steps
# =============================================================================

def compute_constraints_for_obstacles(robot_pos, obstacles, robot_radius=0.5):
    """Compute constraints for given robot position and obstacles."""
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

        # Constraint value at robot position
        value = a1 * robot_pos[0] + a2 * robot_pos[1] - b

        constraints.append({
            'name': name,
            'a1': a1,
            'a2': a2,
            'b': b,
            'value_at_robot': value,
            'dist_to_obstacle': dist,
            'feasible': value <= 0
        })
    return constraints


# The test uses random obstacles, but let's create some plausible scenarios
# Based on the test configuration: 4 obstacles with unicycle dynamics

print("\n" + "=" * 70)
print("SCENARIO A: Obstacles that might cause step 19 to fail")
print("=" * 70)

# Scenario: obstacles positioned such that they're okay at step 18 but problematic at step 19
# This could happen if obstacle trajectories cross the robot's path

obstacles_scenario_a = [
    (np.array([5.0, 2.5]), 0.3, "obs_0"),  # Ahead and to the side
    (np.array([4.5, 1.5]), 0.3, "obs_1"),  # Close to path
    (np.array([6.0, 3.0]), 0.3, "obs_2"),  # Further ahead
    (np.array([3.5, 3.0]), 0.3, "obs_3"),  # To the side
]

print("\nObstacle positions:")
for obs_pos, obs_radius, name in obstacles_scenario_a:
    print(f"  {name}: {obs_pos}")

print("\n--- Step 18 Analysis ---")
constraints_18 = compute_constraints_for_obstacles(step_18_pos, obstacles_scenario_a)
for c in constraints_18:
    status = "✓" if c['feasible'] else "✗ VIOLATED"
    print(f"  {c['name']}: dist={c['dist_to_obstacle']:.3f}m, value={c['value_at_robot']:.4f} {status}")

print("\n--- Step 19 Analysis ---")
constraints_19 = compute_constraints_for_obstacles(step_19_pos, obstacles_scenario_a)
for c in constraints_19:
    status = "✓" if c['feasible'] else "✗ VIOLATED"
    print(f"  {c['name']}: dist={c['dist_to_obstacle']:.3f}m, value={c['value_at_robot']:.4f} {status}")

# =============================================================================
# KEY TEST: Linearization at different reference points
# =============================================================================

print("\n" + "=" * 70)
print("KEY TEST: Effect of Linearization Point")
print("=" * 70)

# When the robot moves from step 18 to step 19, the constraints are re-linearized
# at the new position. This changes the constraint parameters.

obstacle_pos = np.array([4.5, 1.5])  # Close obstacle
obs_radius = 0.3
robot_radius = 0.5
safety_margin = robot_radius + obs_radius

print(f"\nObstacle at: {obstacle_pos}")
print(f"Safety margin: {safety_margin}")

# Constraint linearized at step 18 position
diff_18 = obstacle_pos - step_18_pos
dist_18 = np.linalg.norm(diff_18)
n_18 = diff_18 / dist_18
a1_18, a2_18 = n_18[0], n_18[1]
b_18 = np.dot(n_18, obstacle_pos) - safety_margin

print(f"\nConstraint linearized at step 18 ({step_18_pos}):")
print(f"  {a1_18:.4f}*x + {a2_18:.4f}*y <= {b_18:.4f}")
print(f"  Value at step 18 pos: {a1_18 * step_18_pos[0] + a2_18 * step_18_pos[1] - b_18:.4f}")
print(f"  Value at step 19 pos: {a1_18 * step_19_pos[0] + a2_18 * step_19_pos[1] - b_18:.4f}")

# Constraint linearized at step 19 position
diff_19 = obstacle_pos - step_19_pos
dist_19 = np.linalg.norm(diff_19)
n_19 = diff_19 / dist_19
a1_19, a2_19 = n_19[0], n_19[1]
b_19 = np.dot(n_19, obstacle_pos) - safety_margin

print(f"\nConstraint linearized at step 19 ({step_19_pos}):")
print(f"  {a1_19:.4f}*x + {a2_19:.4f}*y <= {b_19:.4f}")
print(f"  Value at step 18 pos: {a1_19 * step_18_pos[0] + a2_19 * step_18_pos[1] - b_19:.4f}")
print(f"  Value at step 19 pos: {a1_19 * step_19_pos[0] + a2_19 * step_19_pos[1] - b_19:.4f}")

# Compare the constraint directions
angle_18 = np.arctan2(a2_18, a1_18) * 180 / np.pi
angle_19 = np.arctan2(a2_19, a1_19) * 180 / np.pi

print(f"\nConstraint normal directions:")
print(f"  Step 18: {angle_18:.1f}°")
print(f"  Step 19: {angle_19:.1f}°")
print(f"  Difference: {abs(angle_19 - angle_18):.1f}°")

# =============================================================================
# CRITICAL TEST: Check if constraints become infeasible at warmstart
# =============================================================================

print("\n" + "=" * 70)
print("CRITICAL: Warmstart Feasibility at Step 19")
print("=" * 70)

# At step 19, the solver starts with warmstart from step 18's solution
# But the constraints are linearized at step 19's position
# If warmstart (from step 18) violates constraints linearized at step 19, solver may fail

# Simulate multiple obstacles
obstacles_close = [
    (np.array([4.5, 1.5]), 0.3, "close_1"),
    (np.array([5.0, 2.5]), 0.3, "close_2"),
]

print(f"\nWarmstart position (from step 18 solution): {step_18_pos}")
print(f"Linearization point (step 19 current): {step_19_pos}")

print("\nChecking if warmstart satisfies constraints linearized at current position:")
all_feasible = True
for obs_pos, obs_radius, name in obstacles_close:
    safety_margin = robot_radius + obs_radius

    # Constraint linearized at step 19 position
    diff = obs_pos - step_19_pos
    dist = np.linalg.norm(diff)
    n = diff / dist
    a1, a2 = n[0], n[1]
    b = np.dot(n, obs_pos) - safety_margin

    # Value at warmstart (step 18 position)
    warmstart_value = a1 * step_18_pos[0] + a2 * step_18_pos[1] - b

    # Value at current (step 19 position)
    current_value = a1 * step_19_pos[0] + a2 * step_19_pos[1] - b

    ws_status = "feasible" if warmstart_value <= 0 else "INFEASIBLE"
    cur_status = "feasible" if current_value <= 0 else "INFEASIBLE"

    if warmstart_value > 0:
        all_feasible = False

    print(f"  {name}: warmstart={warmstart_value:.4f} ({ws_status}), current={current_value:.4f} ({cur_status})")

if all_feasible:
    print("\n✓ Warmstart is feasible for all constraints")
else:
    print("\n⚠ WARNING: Warmstart may be INFEASIBLE!")
    print("  This could cause solver initialization issues at step 19.")

# =============================================================================
# TEST: What if obstacle moved between steps?
# =============================================================================

print("\n" + "=" * 70)
print("OBSTACLE MOVEMENT EFFECT")
print("=" * 70)

# In the adaptive modes test, obstacles are moving
# Between step 18 and step 19 (0.1s), obstacle moves

obstacle_velocity = np.array([1.0, 0.5])  # Example velocity
dt = 0.1

obs_pos_18 = np.array([4.5, 1.5])
obs_pos_19 = obs_pos_18 + obstacle_velocity * dt

print(f"Obstacle velocity: {obstacle_velocity}")
print(f"Obstacle at step 18: {obs_pos_18}")
print(f"Obstacle at step 19: {obs_pos_19}")

# Constraint at step 18 (linearized at step 18 robot pos, obstacle at step 18 pos)
diff_18 = obs_pos_18 - step_18_pos
dist_18 = np.linalg.norm(diff_18)
n_18 = diff_18 / dist_18
b_18 = np.dot(n_18, obs_pos_18) - safety_margin
value_18 = n_18[0] * step_18_pos[0] + n_18[1] * step_18_pos[1] - b_18

# Constraint at step 19 (linearized at step 19 robot pos, obstacle at step 19 pos)
diff_19 = obs_pos_19 - step_19_pos
dist_19 = np.linalg.norm(diff_19)
n_19 = diff_19 / dist_19
b_19 = np.dot(n_19, obs_pos_19) - safety_margin
value_19 = n_19[0] * step_19_pos[0] + n_19[1] * step_19_pos[1] - b_19

print(f"\nStep 18: dist_to_obs={dist_18:.3f}m, constraint_value={value_18:.4f}")
print(f"Step 19: dist_to_obs={dist_19:.3f}m, constraint_value={value_19:.4f}")

if dist_19 < safety_margin:
    print(f"\n⚠ ROBOT IS INSIDE SAFETY MARGIN AT STEP 19!")
    print(f"  Distance: {dist_19:.3f}m < Safety margin: {safety_margin:.3f}m")

# =============================================================================
# HYPOTHESIS: The issue might be with how reference trajectory is computed
# =============================================================================

print("\n" + "=" * 70)
print("HYPOTHESIS: Reference Trajectory Issue")
print("=" * 70)

print("""
The solver failed at step 19 but succeeded at step 18.
Possible causes:

1. OBSTACLE PROXIMITY: An obstacle moved into a position that makes the
   constraints infeasible at step 19.

2. REFERENCE TRAJECTORY: The reference trajectory used for linearization
   might not be accurate, causing constraints to be too restrictive.

3. WARMSTART INCONSISTENCY: The warmstart from step 18 might not satisfy
   constraints that are linearized at step 19's position.

4. CUMULATIVE DRIFT: Small errors in linearization accumulate, eventually
   making the problem infeasible.

NEXT STEPS:
- Check the actual obstacle positions in the integration test logs
- Verify the reference trajectory computation
- Add slack variables to constraints to prevent hard infeasibility
""")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
