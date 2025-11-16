"""
Goal Objective Test with Multiple Goals Requiring Turning and Static Obstacles - Linearized Constraints

This test verifies that:
- Goal objective works with linearized constraints for obstacle avoidance
- Vehicle can successfully reach multiple goals in sequence
- Each goal requires the vehicle to turn
- Vehicle navigates around static obstacles placed between goals
- All computation is symbolic (no numeric fallbacks)
- Vehicle reaches all goals successfully

Test setup:
- Start: (0, 0) facing east (psi = 0)
- Goal 1: (10, 0) - straight ahead (tests basic goal reaching)
- Goal 2: (10, 10) - requires 90-degree turn to the left
- Goal 3: (0, 10) - requires 90-degree turn to the left (backward)
- Goal 4: (0, 0) - requires 90-degree turn to the left (return to start)
- Obstacle 1: Between start and Goal 1 (on direct path)
- Obstacle 2: Between Goal 1 and Goal 2 (on direct path)
- Obstacle 3: Between Goal 2 and Goal 3 (on direct path)
- Obstacle 4: Between Goal 3 and Goal 4 (on direct path)
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig
from planning.obstacle_manager import create_point_mass_obstacle
from planning.types import PredictionType


def run(dt=0.1, horizon=10, start=(0.0, 0.0), max_iterations=400):
    """Run Goal objective test with multiple goals requiring turning, static obstacles, and linearized constraints."""
    
    framework = IntegrationTestFramework()
    
    # Define a sequence of goals that form a square pattern requiring turns
    goal_sequence = [
        [10.0, 0.0],   # Goal 1: straight ahead (tests basic goal reaching)
        [10.0, 10.0],  # Goal 2: requires 90-degree turn to the left
        [0.0, 10.0],   # Goal 3: requires 90-degree turn to the left (backward)
        [0.0, 0.0],    # Goal 4: requires 90-degree turn to the left (return to start)
    ]
    
    # Place obstacles between each pair of goals (including start to first goal)
    # Obstacle positions: on the direct path between consecutive waypoints
    obstacle_positions = []
    
    # Obstacle 1: Between start and Goal 1 (at ~50% of the way)
    obs1_x = start[0] + 0.5 * (goal_sequence[0][0] - start[0])
    obs1_y = start[1] + 0.5 * (goal_sequence[0][1] - start[1])
    obstacle_positions.append([obs1_x, obs1_y])
    
    # Obstacle 2: Between Goal 1 and Goal 2 (at ~50% of the way)
    obs2_x = goal_sequence[0][0] + 0.5 * (goal_sequence[1][0] - goal_sequence[0][0])
    obs2_y = goal_sequence[0][1] + 0.5 * (goal_sequence[1][1] - goal_sequence[0][1])
    obstacle_positions.append([obs2_x, obs2_y])
    
    # Obstacle 3: Between Goal 2 and Goal 3 (at ~50% of the way)
    obs3_x = goal_sequence[1][0] + 0.5 * (goal_sequence[2][0] - goal_sequence[1][0])
    obs3_y = goal_sequence[1][1] + 0.5 * (goal_sequence[2][1] - goal_sequence[1][1])
    obstacle_positions.append([obs3_x, obs3_y])
    
    # Obstacle 4: Between Goal 3 and Goal 4 (at ~50% of the way)
    obs4_x = goal_sequence[2][0] + 0.5 * (goal_sequence[3][0] - goal_sequence[2][0])
    obs4_y = goal_sequence[2][1] + 0.5 * (goal_sequence[3][1] - goal_sequence[2][1])
    obstacle_positions.append([obs4_x, obs4_y])
    
    # Create a reference path that covers all goals (for framework compatibility)
    max_x = max([g[0] for g in goal_sequence])
    max_y = max([g[1] for g in goal_sequence])
    path_length = max(max_x, max_y) * 1.5
    num_points = max(10, int(path_length / 0.5))
    
    ref_path_points = []
    for i in range(num_points + 1):
        x = start[0] + (i / num_points) * path_length * 0.707
        y = start[1] + (i / num_points) * path_length * 0.707
        ref_path_points.append([x, y])
    ref_path_points = np.array(ref_path_points)
    
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
        print(f"Created stationary obstacle {i+1} at position=({obs_pos[0]:.2f}, {obs_pos[1]:.2f}), "
              f"radius={obstacle_config.radius:.2f}m")
    
    # Goal objective with linearized constraints for obstacle avoidance
    config = TestConfig(
        reference_path=ref_path_points,
        objective_module="goal",
        constraint_modules=["linear"],  # Linearized constraints for obstacle avoidance
        vehicle_dynamics="unicycle",
        num_obstacles=len(obstacle_positions),
        obstacle_dynamics=["point_mass"] * len(obstacle_positions),
        test_name="Goal Objective Multiple Goals Turning with Obstacles - Linearized Constraints",
        duration=max_iterations * dt,
        timestep=dt,
        show_predicted_trajectory=True,
        fallback_control_enabled=False,
        max_consecutive_failures=50,
        goal_sequence=goal_sequence,  # Multiple goals requiring turns
        obstacle_configs=obstacle_configs  # Use explicit obstacle configurations
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
    print(f"Number of obstacles: {len(obstacle_positions)}")
    for i, obs_pos in enumerate(obstacle_positions):
        if i == 0:
            print(f"  Obstacle {i+1}: position=({obs_pos[0]:.2f}, {obs_pos[1]:.2f}) - between start and Goal 1")
        else:
            print(f"  Obstacle {i+1}: position=({obs_pos[0]:.2f}, {obs_pos[1]:.2f}) - between Goal {i} and Goal {i+1}")
    print(f"Constraints: Linearized (for obstacle avoidance)")
    print(f"Objective: Goal only")
    print()
    
    result = framework.run_test(config)
    
    # Verify that the vehicle reached all goals
    if result.success and hasattr(result, 'vehicle_states') and result.vehicle_states:
        print(f"\n=== Goal Sequence Verification ===")
        goal_tolerance = 1.0
        
        # Check each goal in sequence
        # CRITICAL: Goals must be reached in order - a goal is only considered reached
        # if it was reached AFTER the previous goal (or after start for the first goal)
        goals_reached = []
        last_reached_step = -1  # Track when the previous goal was reached
        
        for i, goal in enumerate(goal_sequence):
            # Find the closest point to this goal in the trajectory
            # BUT only consider steps AFTER the previous goal was reached
            min_dist = float('inf')
            closest_step = -1
            
            # For the first goal, check all steps
            # For subsequent goals, only check steps after the previous goal was reached
            start_check_step = 0 if i == 0 else max(0, last_reached_step + 1)
            
            for step in range(start_check_step, len(result.vehicle_states)):
                state = result.vehicle_states[step]
                if len(state) >= 2:
                    dist = np.linalg.norm([state[0] - goal[0], state[1] - goal[1]])
                    if dist < min_dist:
                        min_dist = dist
                        closest_step = step
            
            reached = min_dist <= goal_tolerance and closest_step >= start_check_step
            goals_reached.append(reached)
            
            # If this goal was reached, update last_reached_step
            if reached:
                last_reached_step = closest_step
            
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
            print(f"Final goal tolerance: {goal_tolerance:.1f}m")
            print(f"Final goal reached: {'✓ YES' if dist_to_final <= goal_tolerance else '✗ NO (too far)'}")
            
            # CRITICAL: The final position must also be within tolerance
            # This ensures the vehicle actually reached the final goal, not just got close to it
            final_goal_reached = dist_to_final <= goal_tolerance
            
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
                
                if i == 0:
                    location = "between start and Goal 1"
                else:
                    location = f"between Goal {i} and Goal {i+1}"
                
                print(f"Obstacle {i+1} at ({obs_pos[0]:.2f}, {obs_pos[1]:.2f}) [{location}]: "
                      f"min_distance={min_obs_dist:.3f}m, "
                      f"safe_distance={safe_distance:.3f}m, "
                      f"avoided={'✓ YES' if avoided else '✗ NO (too close)'} "
                      f"(closest at step {min_obs_step})")
            
            all_obstacles_avoided = all(d >= 0.75 for d in min_dist_to_obstacles)  # 0.5 + 0.25 = 0.75 minimum
            print(f"\nAll obstacles avoided: {'✓ YES' if all_obstacles_avoided else '✗ NO'}")
            
            all_reached = all(goals_reached)
            print(f"All goals reached in sequence: {'✓ YES' if all_reached else '✗ NO'}")
            print(f"  Goals reached: {sum(goals_reached)}/{len(goal_sequence)}")
            
            # CRITICAL: Final goal must be reached (final position within tolerance)
            # This ensures the vehicle actually completed the full sequence
            print(f"\n=== Final Verification ===")
            print(f"All goals reached in sequence: {'✓ YES' if all_reached else '✗ NO'}")
            print(f"Final goal reached (final position within tolerance): {'✓ YES' if final_goal_reached else '✗ NO'}")
            print(f"All obstacles avoided: {'✓ YES' if all_obstacles_avoided else '✗ NO'}")
            
            # Test passes only if all goals were reached in sequence AND final position is within tolerance
            return all_reached and final_goal_reached and all_obstacles_avoided
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
    # Obstacles placed between each pair of goals
    result = run(start=(0.0, 0.0), max_iterations=400)
    
    # Verify all goals were reached and obstacles were avoided
    assert result, "Vehicle should reach all goals successfully while avoiding obstacles"
    return result


if __name__ == "__main__":
    test()

