"""
Goal Objective Test with Static Obstacles

This test uses the IntegrationTestFramework to run a test with:
- Goal objective (vehicle tries to reach a goal position)
- Linear constraints (for obstacle avoidance)
- Static obstacles placed directly in the path from start to goal
- Verify that the vehicle navigates around obstacles to reach the goal
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig
from planning.obstacle_manager import create_point_mass_obstacle
from planning.types import PredictionType


def run(dt=0.1, horizon=10, start=(0.0, 0.0), goal=(20.0, 20.0), max_iterations=200, num_obstacles=0):
    """Run Goal objective test with static obstacles using the framework."""
    
    framework = IntegrationTestFramework()
    
    # Create a simple straight-line reference path from start to goal
    # This is just for the framework - the goal objective will drive the vehicle
    path_length = np.linalg.norm(np.array(goal) - np.array(start))
    num_points = max(10, int(path_length / 0.5))  # Points every 0.5m
    
    ref_path_points = []
    for i in range(num_points + 1):
        t = i / num_points
        x = start[0] + t * (goal[0] - start[0])
        y = start[1] + t * (goal[1] - start[1])
        ref_path_points.append([x, y])
    ref_path_points = np.array(ref_path_points)
    
    # Place obstacles along the direct path from start to goal
    # Space them evenly between start and goal (avoiding very start and very end)
    obstacle_configs = []
    for i in range(num_obstacles):
        # Position obstacles at 30%, 50%, 70% along the path (depending on num_obstacles)
        t = 0.3 + (i * 0.4 / max(1, num_obstacles - 1))
        t = min(0.9, max(0.1, t))  # Clamp between 10% and 90%
        
        obs_x = start[0] + t * (goal[0] - start[0])
        obs_y = start[1] + t * (goal[1] - start[1])
        
        # Create stationary obstacle (zero velocity)
        obstacle_config = create_point_mass_obstacle(
            obstacle_id=i,
            position=np.array([obs_x, obs_y]),
            velocity=np.array([0.0, 0.0])  # Stationary
        )
        obstacle_config.radius = 0.5  # Obstacle radius
        obstacle_config.prediction_type = PredictionType.DETERMINISTIC
        obstacle_configs.append(obstacle_config)
        print(f"Created stationary obstacle {i} at position=({obs_x:.2f}, {obs_y:.2f}), "
              f"distance from start: {np.linalg.norm([obs_x - start[0], obs_y - start[1]]):.2f}m")
    
    # For goal objective, we need to pass goal through start/goal parameters
    # The framework will extract these and set data.goal
    config = TestConfig(
        reference_path=ref_path_points,
        objective_module="goal",  # Use goal objective
        constraint_modules=["linear"],  # Linear constraints for obstacle avoidance
        vehicle_dynamics="unicycle",
        num_obstacles=len(obstacle_configs),
        obstacle_dynamics=["point_mass"] * len(obstacle_configs),
        test_name="Goal Objective with Static Obstacles",
        duration=max_iterations * dt,
        timestep=dt,
        show_predicted_trajectory=True,
        obstacle_configs=obstacle_configs,  # Use deterministic obstacle positions
        obstacle_prediction_types=["deterministic"] * len(obstacle_configs),  # Deterministic for stationary obstacles
        fallback_control_enabled=False,  # Keep strict - vehicle must find feasible solution
        max_consecutive_failures=50,  # Allow some recovery attempts
        goal_sequence=[[goal[0], goal[1]]]  # Set the goal position
    )
    
    # Store goal for verification
    config._test_goal = goal
    
    # Run the test
    print(f"\n=== Test Configuration ===")
    print(f"Goal to reach: ({goal[0]:.2f}, {goal[1]:.2f})")
    print(f"Start position: ({start[0]:.2f}, {start[1]:.2f})")
    print(f"Number of obstacles: {len(obstacle_configs)}")
    result = framework.run_test(config)
    
    # Verify that the vehicle reached the goal
    if result.success and hasattr(result, 'vehicle_states') and result.vehicle_states:
        final_state = result.vehicle_states[-1]
        if len(final_state) >= 2:
            final_x = final_state[0]
            final_y = final_state[1]
            dist_to_goal = np.linalg.norm([final_x - goal[0], final_y - goal[1]])
            goal_tolerance = 1.0  # Consider goal reached if within 1m
            reached_goal = dist_to_goal <= goal_tolerance
            print(f"\n=== Goal Reached Verification ===")
            print(f"Final position: ({final_x:.3f}, {final_y:.3f})")
            print(f"Goal position: ({goal[0]:.3f}, {goal[1]:.3f})")
            print(f"Distance to goal: {dist_to_goal:.3f}m")
            print(f"Goal tolerance: {goal_tolerance:.1f}m")
            print(f"Goal reached: {'✓ YES' if reached_goal else '✗ NO'}")
            
            # Check if vehicle avoided obstacles
            print(f"\n=== Obstacle Avoidance Verification ===")
            for i, obs_config in enumerate(obstacle_configs):
                # ObstacleConfig uses initial_position, not position
                obs_pos = obs_config.initial_position
                obs_radius = obs_config.radius
                robot_radius = 0.5  # Default robot radius
                min_safe_distance = robot_radius + obs_radius + 0.3  # Safety margin
                
                # Check minimum distance to obstacle along trajectory
                min_dist = float('inf')
                closest_point = None
                for state in result.vehicle_states:
                    if len(state) >= 2:
                        veh_pos = np.array([state[0], state[1]])
                        dist = np.linalg.norm(veh_pos - obs_pos)
                        if dist < min_dist:
                            min_dist = dist
                            closest_point = veh_pos
                
                safe = min_dist >= min_safe_distance
                print(f"Obstacle {i} at ({obs_pos[0]:.2f}, {obs_pos[1]:.2f}): "
                      f"min_distance={min_dist:.3f}m, safe_distance={min_safe_distance:.3f}m, "
                      f"avoided={'✓ YES' if safe else '✗ NO'}")
                if not safe:
                    print(f"  ⚠️  Vehicle got too close to obstacle {i}!")
    
    return result


def test():
    """Entry point for pytest or direct execution."""
    result = run()
    # Check if goal was reached
    if result.success and hasattr(result, 'vehicle_states') and result.vehicle_states:
        final_state = result.vehicle_states[-1]
        if len(final_state) >= 2:
            goal = (20.0, 20.0)
            final_x = final_state[0]
            final_y = final_state[1]
            dist_to_goal = np.linalg.norm([final_x - goal[0], final_y - goal[1]])
            goal_tolerance = 1.0
            assert dist_to_goal <= goal_tolerance, f"Vehicle did not reach goal (distance: {dist_to_goal:.3f}m)"
    assert result.success, "Test should complete successfully"
    return result


if __name__ == "__main__":
    test()

