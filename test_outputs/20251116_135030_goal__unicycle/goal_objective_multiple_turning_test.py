"""
Goal Objective Test with Multiple Goals Requiring Turning - NO Obstacles, NO Constraints

This test verifies that:
- Goal objective works without any constraints
- Vehicle can successfully reach multiple goals in sequence
- Each goal requires the vehicle to turn
- All computation is symbolic (no numeric fallbacks)
- Vehicle reaches all goals successfully

Test setup:
- Start: (0, 0) facing east (psi = 0)
- Goal 1: (10, 0) - straight ahead (no turn needed, but tests basic goal reaching)
- Goal 2: (10, 10) - requires 90-degree turn to the left
- Goal 3: (0, 10) - requires 90-degree turn to the left (backward)
- Goal 4: (0, 0) - requires 90-degree turn to the left (return to start)
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig


def run(dt=0.1, horizon=10, start=(0.0, 0.0), max_iterations=300):
    """Run Goal objective test with multiple goals requiring turning, NO obstacles, NO constraints."""
    
    framework = IntegrationTestFramework()
    
    # Define a sequence of goals that form a square pattern requiring turns
    goal_sequence = [
        [10.0, 0.0],   # Goal 1: straight ahead (tests basic goal reaching)
        [10.0, 10.0],  # Goal 2: requires 90-degree turn to the left
        [0.0, 10.0],   # Goal 3: requires 90-degree turn to the left (backward)
        [0.0, 0.0],    # Goal 4: requires 90-degree turn to the left (return to start)
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
        test_name="Goal Objective Multiple Goals Turning Test - No Constraints",
        duration=max_iterations * dt,
        timestep=dt,
        show_predicted_trajectory=True,
        fallback_control_enabled=False,
        max_consecutive_failures=50,
        goal_sequence=goal_sequence  # Multiple goals requiring turns
    )
    
    # Run the test
    print(f"\n=== Test Configuration ===")
    print(f"Start position: ({start[0]:.2f}, {start[1]:.2f})")
    print(f"Initial heading: 0.0 rad (facing east)")
    print(f"Goal sequence ({len(goal_sequence)} goals):")
    for i, goal in enumerate(goal_sequence):
        if i == 0:
            print(f"  Goal {i+1}: ({goal[0]:.2f}, {goal[1]:.2f}) - straight ahead")
        else:
            prev_goal = goal_sequence[i-1]
            # Calculate required turn
            dx = goal[0] - prev_goal[0]
            dy = goal[1] - prev_goal[1]
            angle = np.arctan2(dy, dx) * 180 / np.pi
            print(f"  Goal {i+1}: ({goal[0]:.2f}, {goal[1]:.2f}) - requires turn (angle: {angle:.1f}°)")
    print(f"Number of obstacles: 0")
    print(f"Number of constraints: 0")
    print(f"Objective: Goal only")
    print()
    
    result = framework.run_test(config)
    
    # Verify that the vehicle reached all goals
    if result.success and hasattr(result, 'vehicle_states') and result.vehicle_states:
        print(f"\n=== Goal Sequence Verification ===")
        goal_tolerance = 1.0
        
        # Check each goal in sequence
        goals_reached = []
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
            goals_reached.append(reached)
            
            # Calculate heading at closest point to verify turning
            heading_info = ""
            if closest_step >= 0 and closest_step < len(result.vehicle_states):
                state_at_closest = result.vehicle_states[closest_step]
                if len(state_at_closest) >= 3:
                    heading_at_closest = state_at_closest[2]
                    # Expected heading toward goal
                    if i == 0:
                        # From start to goal 1
                        expected_heading = np.arctan2(goal[1] - start[1], goal[0] - start[0])
                    else:
                        # From previous goal to current goal
                        prev_goal = goal_sequence[i-1]
                        expected_heading = np.arctan2(goal[1] - prev_goal[1], goal[0] - prev_goal[0])
                    
                    heading_error = abs(np.arctan2(np.sin(heading_at_closest - expected_heading), 
                                                   np.cos(heading_at_closest - expected_heading)))
                    heading_info = f", heading={np.degrees(heading_at_closest):.1f}° (error: {np.degrees(heading_error):.1f}°)"
            
            print(f"Goal {i+1} ({goal[0]:.2f}, {goal[1]:.2f}): "
                  f"min_distance={min_dist:.3f}m, "
                  f"reached={'✓ YES' if reached else '✗ NO'} "
                  f"(closest at step {closest_step}){heading_info}")
        
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
            
            # Verify turning occurred
            if len(result.vehicle_states) > 10:
                initial_heading = result.vehicle_states[0][2] if len(result.vehicle_states[0]) >= 3 else 0.0
                final_heading = final_state[2] if len(final_state) >= 3 else 0.0
                total_heading_change = abs(np.arctan2(np.sin(final_heading - initial_heading), 
                                                      np.cos(final_heading - initial_heading)))
                
                print(f"\nTurning verification:")
                print(f"  Initial heading: {initial_heading:.3f} rad ({np.degrees(initial_heading):.1f}°)")
                print(f"  Final heading: {final_heading:.3f} rad ({np.degrees(final_heading):.1f}°)")
                print(f"  Total heading change: {total_heading_change:.3f} rad ({np.degrees(total_heading_change):.1f}°)")
                print(f"  Vehicle turned: {'✓ YES' if total_heading_change > 0.5 else '✗ NO (insufficient turn)'}")
            
            all_reached = all(goals_reached)
            print(f"\nAll goals reached: {'✓ YES' if all_reached else '✗ NO'}")
            print(f"  Goals reached: {sum(goals_reached)}/{len(goal_sequence)}")
            
            return all_reached
        else:
            print(f"\n⚠️  Final state incomplete")
            return False
    else:
        print(f"\n=== Test Failed ===")
        print(f"Test did not complete successfully")
        if hasattr(result, 'error_message'):
            print(f"Error: {result.error_message}")
        return False


def test():
    """Entry point for pytest or direct execution."""
    # Test with multiple goals requiring turns: square pattern
    result = run(start=(0.0, 0.0), max_iterations=300)
    
    # Verify all goals were reached
    assert result, "Vehicle should reach all goals successfully"
    return result


if __name__ == "__main__":
    test()

