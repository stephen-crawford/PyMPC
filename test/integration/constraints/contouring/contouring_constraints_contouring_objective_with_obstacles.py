"""
Contouring Constraints Test with Stationary Obstacles

This test uses the new IntegrationTestFramework to run a test with:
- Contouring objective
- Contouring constraints
- Linear constraints (for obstacle avoidance)
- Stationary obstacles placed in the middle of the reference path
"""
import sys
import os
import numpy as np
from utils.math_tools import TKSpline

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig, create_reference_path
from planning.obstacle_manager import create_point_mass_obstacle
from planning.types import ReferencePath, PredictionType


def run(dt=0.1, horizon=10, start=(0.0, 0.0), goal=(20.0, 20.0), max_iterations=200, num_obstacles=3):
    """Run Contouring constraints test with stationary obstacles using the framework."""
    
    framework = IntegrationTestFramework()
    
    # Create curved reference path
    ref_path_points = create_reference_path("curve", length=25.0)
    
    # Convert reference path points to ReferencePath object to access splines
    # This allows us to place obstacles at specific arc lengths
    ref_path = ReferencePath()
    x_arr = np.array([p[0] for p in ref_path_points])
    y_arr = np.array([p[1] for p in ref_path_points])
    z_arr = np.zeros_like(x_arr)
    
    # Compute arc length parameterization
    dx = np.diff(x_arr)
    dy = np.diff(y_arr)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    
    ref_path.x = x_arr.tolist()
    ref_path.y = y_arr.tolist()
    ref_path.z = z_arr.tolist()
    ref_path.s = s.tolist()
    ref_path.x_spline = TKSpline(s, x_arr)
    ref_path.y_spline = TKSpline(s, y_arr)
    ref_path.z_spline = TKSpline(s, z_arr)
    ref_path.length = float(s[-1])
    
    # Sample arc length values along the path (avoiding start and end)
    s_min = float(s[0])
    s_max = float(s[-1])
    s_range = s_max - s_min
    # Place obstacles at evenly spaced positions along the middle portion of the path
    # Start at 50% to give vehicle plenty of time to get moving and establish trajectory
    # End at 80% to avoid path end
    # Space them evenly across this range
    if num_obstacles == 1:
        obstacle_s_positions = [s_min + s_range * 0.6]  # Single obstacle at 60% along path
    else:
        obstacle_s_positions = [s_min + s_range * (0.5 + i * 0.3 / max(1, num_obstacles - 1)) for i in range(num_obstacles)]
    
    # Create obstacles at these positions
    obstacle_configs = []
    for i, s_pos in enumerate(obstacle_s_positions):
        # Get path point at this s value
        try:
            x_pos = float(ref_path.x_spline(s_pos))
            y_pos = float(ref_path.y_spline(s_pos))
            
            # Place obstacle exactly on centerline (no offset)
            x_obs = x_pos
            y_obs = y_pos
            offset = 0.0  # On centerline
            
            # Create stationary obstacle (zero velocity)
            obstacle_config = create_point_mass_obstacle(
                obstacle_id=i,
                position=np.array([x_obs, y_obs]),
                velocity=np.array([0.0, 0.0])  # Stationary
            )
            obstacle_config.radius = 0.4  # Obstacle radius (smaller to allow more maneuvering room)
            obstacle_config.prediction_type = PredictionType.DETERMINISTIC
            obstacle_configs.append(obstacle_config)
            print(f"Created stationary obstacle {i} at s={s_pos:.2f}, position=({x_obs:.2f}, {y_obs:.2f}), offset={offset:.2f}m from centerline")
        except Exception as e:
            print(f"Warning: Could not create obstacle {i} at s={s_pos}: {e}")
    
    config = TestConfig(
        reference_path=ref_path_points,
        objective_module="contouring",
        constraint_modules=["contouring", "linear"],  # Contouring constraints for road boundaries + linear constraints for obstacle avoidance
        vehicle_dynamics="unicycle",
        num_obstacles=len(obstacle_configs),
        obstacle_dynamics=["point_mass"] * len(obstacle_configs),
        test_name="Contouring Objective with Contouring and Linear Obstacle Constraints",
        duration=max_iterations * dt,
        timestep=dt,
        show_predicted_trajectory=True,
        obstacle_configs=obstacle_configs,  # Use deterministic obstacle positions
        obstacle_prediction_types=["deterministic"] * len(obstacle_configs),  # Deterministic for stationary obstacles
        fallback_control_enabled=False,  # Keep strict - vehicle must find feasible solution
        max_consecutive_failures=50  # Increase limit to allow more recovery attempts
    )
    
    # Run the test
    result = framework.run_test(config)
    
    return result


def test():
    """Entry point for pytest or direct execution."""
    result = run()
    assert result.success, "Test should complete successfully"
    return result


if __name__ == "__main__":
    test()

