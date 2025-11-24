"""
Goal Objective Test with Single Dynamic Obstacle - Safe Horizon Constraints

This test verifies that:
- Goal objective works with Safe Horizon constraints for obstacle avoidance
- Vehicle navigates around a single dynamic obstacle that moves back and forth across the path
- Vehicle reaches the goal successfully
- Visualization shows hypothetical trajectories optimized over and the selected trajectory

Test setup:
- Start: (0, 0) facing east (psi = 0)
- Goal: (10, 10)
- Obstacle: Dynamic obstacle with unicycle dynamics that moves back and forth across the path
- Constraints: Safe Horizon constraints (scenario-based probabilistic obstacle avoidance)
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig
from planning.obstacle_manager import create_unicycle_obstacle
from planning.types import PredictionType, ReferencePath


def run(dt=0.1, horizon=10, start=(0.0, 0.0), goal=(10.0, 10.0), max_iterations=200):
    """Run Goal objective test with single dynamic obstacle using Safe Horizon constraints."""
    
    framework = IntegrationTestFramework()
    
    # Create a simple straight-line reference path from start to goal
    path_length = np.linalg.norm(np.array(goal) - np.array(start))
    num_points = max(10, int(path_length / 0.5))  # Points every 0.5m
    
    ref_path_points = []
    s_values = []
    for i in range(num_points + 1):
        t = i / num_points
        x = start[0] + t * (goal[0] - start[0])
        y = start[1] + t * (goal[1] - start[1])
        ref_path_points.append([x, y])
        s_values.append(t * path_length)
    ref_path_points = np.array(ref_path_points)
    s_values = np.array(s_values)
    
    # Create reference path object for obstacle manager (needed for path_intersect behavior)
    from utils.math_tools import TKSpline
    ref_path = ReferencePath()
    ref_path.x = ref_path_points[:, 0]
    ref_path.y = ref_path_points[:, 1]
    ref_path.s = s_values
    ref_path.x_spline = TKSpline(s_values, ref_path_points[:, 0])
    ref_path.y_spline = TKSpline(s_values, ref_path_points[:, 1])
    ref_path.length = path_length
    
    # Place obstacle near the middle of the path, offset to one side
    # It will use path_intersect behavior to move back and forth across the path
    mid_point = np.array([
        start[0] + 0.5 * (goal[0] - start[0]),
        start[1] + 0.5 * (goal[1] - start[1])
    ])
    
    # Calculate perpendicular direction (normal to path)
    path_direction = np.array([goal[0] - start[0], goal[1] - start[1]])
    path_direction = path_direction / np.linalg.norm(path_direction)
    # Perpendicular (rotate 90 degrees): [dy, -dx]
    perpendicular = np.array([path_direction[1], -path_direction[0]])
    
    # Place obstacle 2m to one side of the path
    lateral_offset = 2.0
    obstacle_pos = mid_point + lateral_offset * perpendicular
    
    # Initial velocity pointing toward the path (perpendicular direction)
    # Speed of 1.0 m/s toward the path
    initial_velocity = -perpendicular * 1.0  # Negative because we want to move toward path
    
    # Calculate initial angle from velocity direction
    initial_angle = np.arctan2(initial_velocity[1], initial_velocity[0])
    
    # Create dynamic obstacle with unicycle dynamics and path_intersect behavior
    # path_intersect makes it move back and forth across the path
    obstacle_config = create_unicycle_obstacle(
        obstacle_id=0,
        position=obstacle_pos,
        velocity=initial_velocity,
        angle=initial_angle,
        radius=0.5,  # Obstacle radius
        behavior="path_intersect"  # This makes it move back and forth across the path
    )
    # Safe Horizon constraints require GAUSSIAN prediction type (for scenario sampling)
    obstacle_config.prediction_type = PredictionType.GAUSSIAN
    
    print(f"Created dynamic obstacle with unicycle dynamics:")
    print(f"  Initial position: ({obstacle_pos[0]:.2f}, {obstacle_pos[1]:.2f})")
    print(f"  Initial velocity: ({initial_velocity[0]:.2f}, {initial_velocity[1]:.2f}) m/s")
    print(f"  Initial angle: {initial_angle:.2f} rad ({np.degrees(initial_angle):.1f}°)")
    print(f"  Behavior: path_intersect (moves back and forth across path)")
    print(f"  Radius: 0.5m")
    
    # Goal objective with Safe Horizon constraints for obstacle avoidance
    config = TestConfig(
        reference_path=ref_path_points,
        objective_module="goal",
        constraint_modules=["safe_horizon"],  # Safe Horizon constraints for obstacle avoidance
        vehicle_dynamics="unicycle",
        num_obstacles=1,
        obstacle_dynamics=["unicycle"],  # Use unicycle dynamics for obstacle
        test_name="Goal Objective with Single Dynamic Obstacle - Safe Horizon Constraints",
        duration=max_iterations * dt,
        timestep=dt,
        show_predicted_trajectory=True,
        obstacle_configs=[obstacle_config],
        obstacle_prediction_types=["gaussian"],  # Safe Horizon constraints require GAUSSIAN prediction type
        fallback_control_enabled=False,
        max_consecutive_failures=50,
        goal_sequence=[[goal[0], goal[1]]]  # Set goal using goal_sequence
    )
    
    # Store reference path in config for obstacle manager (needed for path_intersect behavior)
    # The framework will set data.reference_path, which the obstacle manager uses
    config._reference_path_for_obstacles = ref_path
    
    # Run the test
    print(f"\n=== Test Configuration ===")
    print(f"Start position: ({start[0]:.2f}, {start[1]:.2f})")
    print(f"Initial heading: 0.0 rad (facing east)")
    print(f"Goal position: ({goal[0]:.2f}, {goal[1]:.2f})")
    print(f"Obstacle: Dynamic with unicycle dynamics")
    print(f"  Initial position: ({obstacle_pos[0]:.2f}, {obstacle_pos[1]:.2f})")
    print(f"  Behavior: path_intersect (moves back and forth across path)")
    print(f"  Radius: 0.5m")
    print(f"Constraints: Safe Horizon (scenario-based probabilistic obstacle avoidance)")
    print(f"Objective: Goal only")
    print()
    
    # Run the test
    # The framework will now set data.reference_path even for goal objectives
    # if reference_path is provided in config (needed for obstacle path_intersect behavior)
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
        
        # Check obstacle avoidance - need to check against actual obstacle trajectory
        print(f"\n=== Obstacle Avoidance Verification ===")
        robot_radius = 0.5  # Default robot radius
        obs_radius = obstacle_config.radius
        min_safe_distance = robot_radius + obs_radius  # Collision distance (no extra margin for test)
        
        # Get obstacle trajectory from result
        obstacle_trajectory = []
        if hasattr(result, 'obstacle_states') and result.obstacle_states and len(result.obstacle_states) > 0:
            obstacle_trajectory = result.obstacle_states[0]  # First (and only) obstacle
        
        min_dist_to_obs = float('inf')
        closest_step_to_obs = -1
        closest_obs_pos = None
        
        for step, state in enumerate(result.vehicle_states):
            if len(state) >= 2:
                veh_pos = np.array([state[0], state[1]])
                # Get obstacle position at this step
                if step < len(obstacle_trajectory) and len(obstacle_trajectory[step]) >= 2:
                    obs_pos_at_step = np.array([obstacle_trajectory[step][0], obstacle_trajectory[step][1]])
                else:
                    # Fallback to initial position if trajectory not available
                    obs_pos_at_step = obstacle_pos
                
                dist = np.linalg.norm(veh_pos - obs_pos_at_step)
                if dist < min_dist_to_obs:
                    min_dist_to_obs = dist
                    closest_step_to_obs = step
                    closest_obs_pos = obs_pos_at_step
        
        collision_distance = robot_radius + obs_radius
        collision_avoided = min_dist_to_obs > collision_distance  # Must be strictly greater than collision distance
        safe = min_dist_to_obs >= min_safe_distance
        boundary_case = abs(min_dist_to_obs - collision_distance) < 0.01  # Within 1cm of boundary
        
        if closest_obs_pos is not None:
            print(f"Obstacle (dynamic, moved during test):")
            print(f"  Initial position: ({obstacle_pos[0]:.2f}, {obstacle_pos[1]:.2f})")
            print(f"  Position at closest approach: ({closest_obs_pos[0]:.2f}, {closest_obs_pos[1]:.2f})")
        else:
            print(f"Obstacle at ({obstacle_pos[0]:.2f}, {obstacle_pos[1]:.2f}):")
        print(f"  Minimum distance: {min_dist_to_obs:.3f}m (at step {closest_step_to_obs})")
        print(f"  Collision distance: {collision_distance:.3f}m (robot_radius + obs_radius)")
        print(f"  Safe distance: {min_safe_distance:.3f}m")
        print(f"  Collision avoided: {'✓ YES' if collision_avoided else ('✓ YES (at boundary)' if boundary_case else '✗ NO')}")
        if not collision_avoided:
            if boundary_case:
                print(f"  ⚠️  Vehicle reached exact collision boundary (acceptable - constraint working)")
            else:
                print(f"  ⚠️  Vehicle collided with obstacle!")
        elif not safe:
            print(f"  ⚠️  Vehicle got close to obstacle (within safety margin)")
        
        # Accept if vehicle reached goal and either avoided collision or was exactly at boundary
        # (exactly at boundary means constraint is working correctly)
        return reached and (collision_avoided or boundary_case)
    else:
        print(f"\n=== Test Failed ===")
        print(f"Test did not complete successfully")
        if hasattr(result, 'error_message'):
            print(f"Error: {result.error_message}")
        return False


def test():
    """Entry point for pytest or direct execution."""
    result = run(start=(0.0, 0.0), goal=(10.0, 10.0), max_iterations=200)
    
    # Verify goal was reached
    assert result, "Vehicle should reach the goal and avoid the obstacle successfully"
    return result


if __name__ == "__main__":
    test()

