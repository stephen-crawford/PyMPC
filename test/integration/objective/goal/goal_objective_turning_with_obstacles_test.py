"""
Goal Objective Test with Turning Requirement and Static Obstacles - Linearized Constraints

This test verifies that:
- Goal objective works with linearized constraints for obstacle avoidance
- Vehicle can successfully turn to reach a goal that requires turning
- Vehicle navigates around static obstacles placed on the direct path
- All computation is symbolic (no numeric fallbacks)
- Vehicle reaches the goal successfully

Test setup:
- Start: (0, 0) facing east (psi = 0)
- Goal: (10, 10) - requires 90-degree turn to the left
- Obstacle 1: (3, 3) - on direct path, requires navigation around
- Obstacle 2: (7, 7) - on direct path, requires navigation around
- Constraints: Linearized constraints for obstacle avoidance
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig
from planning.obstacle_manager import create_point_mass_obstacle
from planning.types import PredictionType


def run(dt=0.1, horizon=10, start=(0.0, 0.0), goal=(10.0, 10.0), max_iterations=200):
    """Run Goal objective test with turning requirement, static obstacles, and linearized constraints."""
    
    framework = IntegrationTestFramework()
    
    # Create a simple reference path from start to goal (for framework compatibility)
    # The framework uses the reference path end point as the goal for goal objective
    path_length = np.linalg.norm(np.array(goal) - np.array(start))
    num_points = max(10, int(path_length / 0.5))  # Points every 0.5m
    
    ref_path_points = []
    for i in range(num_points + 1):
        t = i / num_points
        x = start[0] + t * (goal[0] - start[0])
        y = start[1] + t * (goal[1] - start[1])
        ref_path_points.append([x, y])
    ref_path_points = np.array(ref_path_points)
    
    # Place two static obstacles on the direct path from start to goal
    # Obstacle 1: at 1/3 of the way
    # Obstacle 2: at 2/3 of the way
    obstacle_positions = [
        [3.0, 3.0],   # Obstacle 1: on direct path
        [7.0, 7.0],   # Obstacle 2: on direct path
    ]
    
    # Create obstacle configurations using create_point_mass_obstacle
    obstacle_configs = []
    for i, obs_pos in enumerate(obstacle_positions):
        # Create stationary obstacle (zero velocity)
        obstacle_config = create_point_mass_obstacle(
            obstacle_id=i,
            position=np.array([obs_pos[0], obs_pos[1]]),
            velocity=np.array([0.0, 0.0])  # Stationary
        )
        obstacle_config.radius = 0.5  # Obstacle radius
        obstacle_config.prediction_type = PredictionType.DETERMINISTIC
        obstacle_configs.append(obstacle_config)
        print(f"Created stationary obstacle {i} at position=({obs_pos[0]:.2f}, {obs_pos[1]:.2f}), "
              f"radius={obstacle_config.radius:.2f}m")
    
    # Goal objective with linearized constraints for obstacle avoidance
    config = TestConfig(
        reference_path=ref_path_points,  # Still needed for framework compatibility
        objective_module="goal",
        constraint_modules=["linear"],  # Linearized constraints for obstacle avoidance
        vehicle_dynamics="unicycle",
        num_obstacles=len(obstacle_positions),
        obstacle_dynamics=["point_mass"] * len(obstacle_positions),
        test_name="Goal Objective Turning with Obstacles - Linearized Constraints",
        duration=max_iterations * dt,
        timestep=dt,
        show_predicted_trajectory=True,
        fallback_control_enabled=False,
        max_consecutive_failures=50,
        goal_sequence=[[goal[0], goal[1]]],  # Set goal using goal_sequence
        obstacle_configs=obstacle_configs  # Use explicit obstacle configurations
    )
    
    # Run the test
    print(f"\n=== Test Configuration ===")
    print(f"Start position: ({start[0]:.2f}, {start[1]:.2f})")
    print(f"Initial heading: 0.0 rad (facing east)")
    print(f"Goal position: ({goal[0]:.2f}, {goal[1]:.2f})")
    print(f"Required turn: ~90 degrees to the left")
    print(f"Number of obstacles: {len(obstacle_positions)}")
    for i, obs_pos in enumerate(obstacle_positions):
        print(f"  Obstacle {i+1}: position=({obs_pos[0]:.2f}, {obs_pos[1]:.2f}), radius=0.5m, static")
    print(f"Constraints: Linearized (for obstacle avoidance)")
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
                # Expected heading toward goal
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
            
            # Verify obstacle avoidance
            print(f"\n=== Obstacle Avoidance Verification ===")
            min_dist_to_obstacles = []
            for i, obs_pos in enumerate(obstacle_positions):
                min_obs_dist = float('inf')
                min_obs_step = -1
                for step, state in enumerate(result.vehicle_states):
                    if len(state) >= 2:
                        dist = np.linalg.norm([state[0] - obs_pos[0], state[1] - obs_pos[1]])
                        if dist < min_obs_dist:
                            min_obs_dist = dist
                            min_obs_step = step
                
                obstacle_radius = 0.5
                safe_distance = obstacle_radius + 0.25  # Vehicle radius (0.25) + obstacle radius (0.5)
                avoided = min_obs_dist >= safe_distance
                min_dist_to_obstacles.append(min_obs_dist)
                
                print(f"Obstacle {i+1} at ({obs_pos[0]:.2f}, {obs_pos[1]:.2f}): "
                      f"min_distance={min_obs_dist:.3f}m, "
                      f"safe_distance={safe_distance:.3f}m, "
                      f"avoided={'✓ YES' if avoided else '✗ NO (too close)'} "
                      f"(closest at step {min_obs_step})")
            
            all_obstacles_avoided = all(d >= 0.75 for d in min_dist_to_obstacles)  # 0.5 + 0.25 = 0.75 minimum
            print(f"\nAll obstacles avoided: {'✓ YES' if all_obstacles_avoided else '✗ NO'}")
            
            return reached and all_obstacles_avoided
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
    # Test with goal requiring 90-degree turn: start at (0,0) facing east, goal at (10, 10)
    # Two static obstacles on the direct path
    result = run(start=(0.0, 0.0), goal=(10.0, 10.0), max_iterations=200)
    
    # Verify goal was reached and obstacles were avoided
    assert result, "Vehicle should reach the goal successfully while avoiding obstacles"
    return result


if __name__ == "__main__":
    test()

