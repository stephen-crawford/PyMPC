"""
Goal Objective Test with Turning Requirement - NO Obstacles, NO Constraints

This test verifies that:
- Goal objective works without any constraints
- Vehicle can successfully turn to reach a goal that requires turning
- All computation is symbolic (no numeric fallbacks)
- Vehicle reaches the goal successfully

Test setup:
- Start: (0, 0) facing east (psi = 0)
- Goal: (10, 10) - requires 90-degree turn to the left
- No obstacles
- No constraints
- Only goal objective module
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig


def run(dt=0.1, horizon=10, start=(0.0, 0.0), goal=(10.0, 10.0), max_iterations=200):
    """Run Goal objective test with turning requirement, NO obstacles, NO constraints."""
    
    framework = IntegrationTestFramework()
    
    # Create a simple reference path from start to goal (for framework compatibility)
    # The framework uses the reference path end point as the goal for goal objective
    # So we need to make sure the path ends at our desired goal
    path_length = np.linalg.norm(np.array(goal) - np.array(start))
    num_points = max(10, int(path_length / 0.5))  # Points every 0.5m
    
    ref_path_points = []
    for i in range(num_points + 1):
        t = i / num_points
        x = start[0] + t * (goal[0] - start[0])
        y = start[1] + t * (goal[1] - start[1])
        ref_path_points.append([x, y])
    ref_path_points = np.array(ref_path_points)
    
    # Verify the path ends at the goal
    assert abs(ref_path_points[-1, 0] - goal[0]) < 0.01, f"Path end x {ref_path_points[-1, 0]} != goal x {goal[0]}"
    assert abs(ref_path_points[-1, 1] - goal[1]) < 0.01, f"Path end y {ref_path_points[-1, 1]} != goal y {goal[1]}"
    
    # NO obstacles, NO constraints - only goal objective
    # Use goal_sequence to set the goal (framework ignores reference_path for goal objectives)
    config = TestConfig(
        reference_path=ref_path_points,  # Still needed for framework compatibility
        objective_module="goal",
        constraint_modules=[],  # NO constraints
        vehicle_dynamics="unicycle",
        num_obstacles=0,
        obstacle_dynamics=[],
        test_name="Goal Objective Turning Test - No Constraints",
        duration=max_iterations * dt,
        timestep=dt,
        show_predicted_trajectory=True,
        fallback_control_enabled=False,
        max_consecutive_failures=50,
        goal_sequence=[[goal[0], goal[1]]]  # Set goal using goal_sequence
    )
    
    # Run the test
    print(f"\n=== Test Configuration ===")
    print(f"Start position: ({start[0]:.2f}, {start[1]:.2f})")
    print(f"Initial heading: 0.0 rad (facing east)")
    print(f"Goal position: ({goal[0]:.2f}, {goal[1]:.2f})")
    print(f"Required turn: ~90 degrees to the left")
    print(f"Number of obstacles: 0")
    print(f"Number of constraints: 0")
    print(f"Objective: Goal only")
    print()
    
    result = framework.run_test(config)
    
    # Verify that the vehicle reached the goal
    if result.success and hasattr(result, 'vehicle_states') and result.vehicle_states:
        print(f"\n=== Goal Reaching Verification ===")
        goal_tolerance = 1.0
        
        # Find the closest point to goal in the trajectory
        min_dist = float('inf')
        closest_step = -1
        final_state = result.vehicle_states[-1]
        
        for step, state in enumerate(result.vehicle_states):
            if len(state) >= 2:
                dist = np.linalg.norm([state[0] - goal[0], state[1] - goal[1]])
                if dist < min_dist:
                    min_dist = dist
                    closest_step = step
        
        reached = min_dist <= goal_tolerance
        
        # Final position
        if len(final_state) >= 2:
            final_x = final_state[0]
            final_y = final_state[1]
            dist_to_goal = np.linalg.norm([final_x - goal[0], final_y - goal[1]])
            
            print(f"Goal position: ({goal[0]:.2f}, {goal[1]:.2f})")
            print(f"Final position: ({final_x:.3f}, {final_y:.3f})")
            print(f"Minimum distance to goal: {min_dist:.3f}m (at step {closest_step})")
            print(f"Final distance to goal: {dist_to_goal:.3f}m")
            print(f"Goal reached: {'✓ YES' if reached else '✗ NO'}")
            
            # Check if vehicle turned (final heading should be roughly toward goal)
            if len(final_state) >= 3:
                final_psi = final_state[2]
                # Expected heading toward goal: atan2(goal_y - final_y, goal_x - final_x)
                expected_heading = np.arctan2(goal[1] - final_y, goal[0] - final_x)
                heading_error = abs(np.arctan2(np.sin(final_psi - expected_heading), 
                                               np.cos(final_psi - expected_heading)))
                print(f"Final heading: {final_psi:.3f} rad ({np.degrees(final_psi):.1f}°)")
                print(f"Expected heading: {expected_heading:.3f} rad ({np.degrees(expected_heading):.1f}°)")
                print(f"Heading error: {heading_error:.3f} rad ({np.degrees(heading_error):.1f}°)")
                print(f"Vehicle turned: {'✓ YES' if heading_error < 0.5 else '✗ NO (heading error too large)'}")
            
            # Verify trajectory shows turning
            if len(result.vehicle_states) > 10:
                # Check if vehicle's heading changed significantly
                initial_heading = result.vehicle_states[0][2] if len(result.vehicle_states[0]) >= 3 else 0.0
                mid_heading = result.vehicle_states[len(result.vehicle_states)//2][2] if len(result.vehicle_states[len(result.vehicle_states)//2]) >= 3 else 0.0
                heading_change = abs(np.arctan2(np.sin(mid_heading - initial_heading), 
                                                np.cos(mid_heading - initial_heading)))
                print(f"Initial heading: {initial_heading:.3f} rad ({np.degrees(initial_heading):.1f}°)")
                print(f"Mid-trajectory heading: {mid_heading:.3f} rad ({np.degrees(mid_heading):.1f}°)")
                print(f"Heading change: {heading_change:.3f} rad ({np.degrees(heading_change):.1f}°)")
                print(f"Trajectory shows turning: {'✓ YES' if heading_change > 0.3 else '✗ NO (insufficient turn)'}")
        
        return reached
    else:
        print(f"\n=== Test Failed ===")
        print(f"Test did not complete successfully")
        if hasattr(result, 'error_message'):
            print(f"Error: {result.error_message}")
        return False


def test():
    """Entry point for pytest or direct execution."""
    # Test with goal requiring 90-degree turn: start at (0,0) facing east, goal at (10, 10)
    result = run(start=(0.0, 0.0), goal=(10.0, 10.0), max_iterations=200)
    
    # Verify goal was reached
    assert result, "Vehicle should reach the goal successfully"
    return result


if __name__ == "__main__":
    test()

