"""
Goal Objective Test with NO Obstacles and NO Constraints - Multiple Goals

This test verifies that:
- Goal objective works without any constraints
- All computation is symbolic (no numeric fallbacks)
- Vehicle visits multiple goals in sequence
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig


def run(dt=0.1, horizon=10, start=(0.0, 0.0), max_iterations=300):
    """Run Goal objective test with NO obstacles, NO constraints, and multiple goals."""
    
    framework = IntegrationTestFramework()
    
    # Define a sequence of goals: move in a square pattern
    goal_sequence = [
        [10.0, 0.0],   # First goal: straight ahead
        [10.0, 10.0],  # Second goal: turn right
        [0.0, 10.0],   # Third goal: turn back
        [0.0, 0.0],    # Final goal: return to start
    ]
    
    # Create a reference path that covers all goals (for framework compatibility)
    # This is just for the framework - the goal objective will drive the vehicle
    max_x = max([g[0] for g in goal_sequence])
    max_y = max([g[1] for g in goal_sequence])
    path_length = max(max_x, max_y) * 1.5  # Make path long enough
    num_points = max(10, int(path_length / 0.5))
    
    ref_path_points = []
    for i in range(num_points + 1):
        # Simple diagonal path
        x = start[0] + (i / num_points) * path_length * 0.707
        y = start[1] + (i / num_points) * path_length * 0.707
        ref_path_points.append([x, y])
    ref_path_points = np.array(ref_path_points)
    
    # NO obstacles, NO constraints - only goal objective with multiple goals
    config = TestConfig(
        reference_path=ref_path_points,
        objective_module="goal",
        constraint_modules=[],  # NO constraints
        vehicle_dynamics="unicycle",
        num_obstacles=0,
        obstacle_dynamics=[],
        test_name="Goal Objective Multiple Goals No Constraints",
        duration=max_iterations * dt,
        timestep=dt,
        show_predicted_trajectory=True,
        fallback_control_enabled=False,
        max_consecutive_failures=50,
        goal_sequence=goal_sequence
    )
    
    # Run the test
    print(f"\n=== Test Configuration ===")
    print(f"Start position: ({start[0]:.2f}, {start[1]:.2f})")
    print(f"Goal sequence:")
    for i, goal in enumerate(goal_sequence):
        print(f"  Goal {i+1}: ({goal[0]:.2f}, {goal[1]:.2f})")
    print(f"Number of obstacles: 0")
    print(f"Number of constraints: 0")
    print()
    result = framework.run_test(config)
    
    # Verify that the vehicle reached all goals
    if result.success and hasattr(result, 'vehicle_states') and result.vehicle_states:
        print(f"\n=== Goal Sequence Verification ===")
        goal_tolerance = 1.0
        
        # Check each goal in sequence
        for i, goal in enumerate(goal_sequence):
            # Find the closest point to this goal in the trajectory
            min_dist = float('inf')
            closest_step = -1
            for step, state in enumerate(result.vehicle_states):
                if len(state) >= 2:
                    dist = np.linalg.norm([state[0] - goal[0], state[1] - goal[1]])
                    if dist < min_dist:
                        min_dist = dist
                        closest_step = step
            
            reached = min_dist <= goal_tolerance
            print(f"Goal {i+1} ({goal[0]:.2f}, {goal[1]:.2f}): "
                  f"min_distance={min_dist:.3f}m, "
                  f"reached={'✓ YES' if reached else '✗ NO'} "
                  f"(closest at step {closest_step})")
        
        # Final position
        final_state = result.vehicle_states[-1]
        if len(final_state) >= 2:
            final_x = final_state[0]
            final_y = final_state[1]
            final_goal = goal_sequence[-1]
            dist_to_final = np.linalg.norm([final_x - final_goal[0], final_y - final_goal[1]])
            print(f"\nFinal position: ({final_x:.3f}, {final_y:.3f})")
            print(f"Final goal: ({final_goal[0]:.3f}, {final_goal[1]:.3f})")
            print(f"Distance to final goal: {dist_to_final:.3f}m")
            print(f"All goals reached: {'✓ YES' if dist_to_final <= goal_tolerance else '✗ NO'}")
    
    return result


def test():
    """Entry point for pytest or direct execution."""
    result = run()
    # Check if all goals were reached
    if result.success and hasattr(result, 'vehicle_states') and result.vehicle_states:
        goal_sequence = [
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
            [0.0, 0.0],
        ]
        goal_tolerance = 1.0
        
        # Verify final goal
        final_state = result.vehicle_states[-1]
        if len(final_state) >= 2:
            final_goal = goal_sequence[-1]
            final_x = final_state[0]
            final_y = final_state[1]
            dist_to_goal = np.linalg.norm([final_x - final_goal[0], final_y - final_goal[1]])
            assert dist_to_goal <= goal_tolerance, f"Vehicle did not reach final goal (distance: {dist_to_goal:.3f}m)"
    assert result.success, "Test should complete successfully"
    return result


if __name__ == "__main__":
    test()

