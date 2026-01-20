"""
Safe Horizon Constraint Integration Test with Multiple Dynamic Obstacles

This test runs MPCC with Safe Horizon constraints for obstacle avoidance
with multiple dynamic obstacles. The Safe Horizon constraint is computationally
expensive (~2.7s per MPC step), so we use an extended timeout.

Test scenario:
- Vehicle follows a curved reference path
- Multiple dynamic obstacles with unicycle dynamics move in the environment
- Safe Horizon constraints ensure probabilistic collision avoidance
- Gaussian uncertainty predictions for obstacle motion
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig, create_reference_path
from planning.obstacle_manager import create_unicycle_obstacle
from planning.types import PredictionType
import numpy as np


def run(dt=0.1, duration=25.0, timeout_minutes=15.0, num_obstacles=3):
    """Run Safe Horizon constraints test with multiple dynamic obstacles.
    
    Args:
        dt: Timestep in seconds (default 0.1s)
        duration: Simulation duration in seconds (default 25s)
        timeout_minutes: Test timeout in minutes (default 15 minutes)
        num_obstacles: Number of dynamic obstacles (default 3)
    """
    
    framework = IntegrationTestFramework()
    
    # Create curved reference path (longer path for multi-obstacle scenario)
    ref_path = create_reference_path("curve", length=25.0)
    
    # Get path bounds for obstacle placement
    path_x = np.array(ref_path.x)
    path_y = np.array(ref_path.y)
    path_length = ref_path.length
    
    # Create obstacle configurations placed along the path
    obstacle_configs = []
    
    for i in range(num_obstacles):
        # Place obstacles at different positions along the path
        # Spread them evenly between 20% and 70% of path length
        path_fraction = 0.2 + (i * 0.5 / max(1, num_obstacles - 1))
        s_position = path_fraction * path_length
        
        # Get path point at this arc length
        path_x_at_s = float(ref_path.x_spline(s_position))
        path_y_at_s = float(ref_path.y_spline(s_position))
        
        # Get path tangent to compute normal (for lateral offset)
        dx_ds = float(ref_path.x_spline.derivative()(s_position))
        dy_ds = float(ref_path.y_spline.derivative()(s_position))
        tangent_norm = np.sqrt(dx_ds**2 + dy_ds**2)
        
        if tangent_norm > 1e-6:
            # Normal vector (perpendicular to path, pointing left)
            normal_x = -dy_ds / tangent_norm
            normal_y = dx_ds / tangent_norm
        else:
            normal_x, normal_y = 0.0, 1.0
        
        # Alternate obstacles on left and right side of path
        lateral_offset = 2.0 if i % 2 == 0 else -2.0
        
        # Position obstacle offset from centerline
        obstacle_x = path_x_at_s + lateral_offset * normal_x
        obstacle_y = path_y_at_s + lateral_offset * normal_y
        
        # Velocity toward the path (will cross vehicle's path)
        speed = 0.5 + 0.3 * i  # Varying speeds
        velocity_x = -lateral_offset * normal_x * speed * 0.5
        velocity_y = -lateral_offset * normal_y * speed * 0.5
        
        initial_angle = np.arctan2(velocity_y, velocity_x)
        
        # Create obstacle configuration
        obstacle_config = create_unicycle_obstacle(
            obstacle_id=i,
            position=np.array([obstacle_x, obstacle_y]),
            velocity=np.array([velocity_x, velocity_y]),
            angle=initial_angle,
            radius=0.35,
            behavior="path_intersect"  # Moves back and forth across path
        )
        obstacle_config.prediction_type = PredictionType.GAUSSIAN
        obstacle_config.uncertainty_params = {
            'position_std': 0.15,
            'uncertainty_growth': 0.08
        }
        
        obstacle_configs.append(obstacle_config)
        print(f"Obstacle {i}: position=({obstacle_x:.2f}, {obstacle_y:.2f}), "
              f"velocity=({velocity_x:.2f}, {velocity_y:.2f}), "
              f"at {path_fraction*100:.0f}% of path")
    
    config = TestConfig(
        reference_path=ref_path,
        objective_module="contouring",
        constraint_modules=["safe_horizon", "contouring"],
        vehicle_dynamics="unicycle",
        num_obstacles=num_obstacles,
        obstacle_dynamics=["unicycle"] * num_obstacles,
        obstacle_prediction_types=["gaussian"] * num_obstacles,
        obstacle_configs=obstacle_configs,
        test_name=f"Safe Horizon {num_obstacles} Obstacles Test",
        duration=duration,
        timestep=dt,
        show_predicted_trajectory=True,
        # Extended timeout for Safe Horizon with multiple obstacles
        # More obstacles = more scenarios = more compute time
        timeout_seconds=timeout_minutes * 60.0,
        max_consecutive_failures=10,  # Allow more failures with complex scenarios
    )
    
    # Run the test
    print(f"\n=== Test Configuration ===")
    print(f"Objective: contouring")
    print(f"Constraints: safe_horizon, contouring")
    print(f"Number of obstacles: {num_obstacles}")
    print(f"Duration: {duration}s")
    print(f"Timestep: {dt}s")
    print(f"Max steps: {int(duration/dt)}")
    print(f"Timeout: {timeout_minutes} minutes ({timeout_minutes*60}s)")
    print(f"Expected compute time: ~{int(duration/dt) * 3.0 / 60:.1f} minutes (estimate)")
    print()
    
    result = framework.run_test(config)
    
    # Report results
    print(f"\n=== Results ===")
    print(f"Vehicle states: {len(result.vehicle_states)}")
    print(f"Success: {result.success}")
    print(f"Constraint violations: {sum(result.constraint_violations)}")
    if hasattr(result, 'computation_times') and result.computation_times:
        avg_time = sum(result.computation_times) / len(result.computation_times)
        print(f"Avg computation time: {avg_time:.2f}s per step")
    if hasattr(result, 'output_folder'):
        print(f"Output folder: {result.output_folder}")
    
    return result


def test():
    """Entry point for pytest or direct execution."""
    result = run()
    assert result.success, "Test should complete successfully"
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Safe Horizon Multi-Obstacle Test")
    parser.add_argument("--obstacles", "-n", type=int, default=3, 
                        help="Number of obstacles (default: 3)")
    parser.add_argument("--duration", "-d", type=float, default=25.0,
                        help="Simulation duration in seconds (default: 25)")
    parser.add_argument("--timeout", "-t", type=float, default=15.0,
                        help="Test timeout in minutes (default: 15)")
    args = parser.parse_args()
    
    run(num_obstacles=args.obstacles, duration=args.duration, timeout_minutes=args.timeout)
