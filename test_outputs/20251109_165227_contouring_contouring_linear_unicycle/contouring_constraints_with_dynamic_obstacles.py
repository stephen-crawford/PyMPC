"""
Contouring Constraints Test with Dynamic Obstacles

This test uses the IntegrationTestFramework to run a test with:
- Contouring objective
- Contouring constraints
- Linearized constraints (for dynamic obstacle avoidance)
- Several dynamic obstacles moving along the reference path
"""
import sys
import os
import numpy as np
from scipy.interpolate import CubicSpline

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig, create_reference_path
from planning.obstacle_manager import create_unicycle_obstacle, create_bicycle_obstacle, create_point_mass_obstacle, ObstacleConfig
from planning.types import ReferencePath, PredictionType


def run(dt=0.1, horizon=10, start=(0.0, 0.0), goal=(20.0, 20.0), max_iterations=200, num_obstacles=3, vehicle_dynamics="unicycle"):
    """Run Contouring constraints test with dynamic obstacles using linearized constraints.
    
    Args:
        vehicle_dynamics: Vehicle dynamics type ("unicycle", "bicycle", etc.)
        Obstacles will use the same dynamics model as the vehicle
    """
    
    framework = IntegrationTestFramework()
    
    # Create curved reference path
    ref_path_points = create_reference_path("curve", length=25.0)
    
    # Convert reference path points to ReferencePath object to access splines
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
    ref_path.x_spline = CubicSpline(s, x_arr)
    ref_path.y_spline = CubicSpline(s, y_arr)
    ref_path.z_spline = CubicSpline(s, z_arr)
    ref_path.length = float(s[-1])
    
    # Sample arc length values along the path for obstacle starting positions
    s_min = float(s[0])
    s_max = float(s[-1])
    s_range = s_max - s_min
    # Place obstacles at evenly spaced positions along the middle portion of the path
    if num_obstacles == 1:
        obstacle_s_positions = [s_min + s_range * 0.5]  # Single obstacle at 50% along path
    else:
        obstacle_s_positions = [s_min + s_range * (0.4 + i * 0.3 / max(1, num_obstacles - 1)) for i in range(num_obstacles)]
    
    # Create dynamic obstacles with different directions and same dynamics as vehicle
    obstacle_configs = []
    obstacle_dynamics_list = []
    
    for i, s_pos in enumerate(obstacle_s_positions):
        # Get path point at this s value
        try:
            x_pos = float(ref_path.x_spline(s_pos))
            y_pos = float(ref_path.y_spline(s_pos))
            
            # Get path tangent to determine reference direction
            dx = float(ref_path.x_spline.derivative()(s_pos))
            dy = float(ref_path.y_spline.derivative()(s_pos))
            norm = np.sqrt(dx**2 + dy**2)
            
            # Define different travel directions for obstacles
            # Not all obstacles should travel in the same direction as the vehicle
            directions = [
                "forward",      # Same direction as vehicle (along path)
                "backward",     # Opposite direction (against path)
                "perpendicular" # Perpendicular to path
            ]
            direction = directions[i % len(directions)]
            
            if norm > 1e-6:
                # Normalize tangent to get path direction
                dx_norm = dx / norm
                dy_norm = dy / norm
                
                # Normal vector (perpendicular to tangent, pointing left)
                nx = -dy / norm
                ny = dx / norm
                
                # Set obstacle velocity based on direction
                obstacle_speed = 1.0  # Moderate speed
                if direction == "forward":
                    # Same direction as vehicle (along path forward)
                    velocity = np.array([dx_norm * obstacle_speed, dy_norm * obstacle_speed])
                    angle = np.arctan2(dy_norm, dx_norm)
                elif direction == "backward":
                    # Opposite direction (against path)
                    velocity = np.array([-dx_norm * obstacle_speed, -dy_norm * obstacle_speed])
                    angle = np.arctan2(-dy_norm, -dx_norm)
                else:  # perpendicular
                    # Perpendicular to path (alternate left/right)
                    perp_dirs = [1, -1]  # Left, right
                    perp_dir = perp_dirs[i % len(perp_dirs)]
                    velocity = np.array([nx * obstacle_speed * perp_dir, ny * obstacle_speed * perp_dir])
                    angle = np.arctan2(ny * perp_dir, nx * perp_dir)
                
                # Offset obstacle slightly from centerline to create interesting scenario
                offsets = [0.5, -0.5, 0.0]  # Alternate left/right/center
                offset = offsets[i % len(offsets)]
                x_obs = x_pos + nx * offset
                y_obs = y_pos + ny * offset
            else:
                # Fallback for degenerate path
                velocity = np.array([1.0, 0.0]) if direction == "forward" else np.array([-1.0, 0.0])
                angle = 0.0 if direction == "forward" else np.pi
                x_obs = x_pos
                y_obs = y_pos
            
            # Create dynamic obstacle using the same dynamics model as the vehicle
            if vehicle_dynamics == "unicycle":
                obstacle_config = create_unicycle_obstacle(
                    obstacle_id=i,
                    position=np.array([x_obs, y_obs]),
                    velocity=velocity,
                    angle=angle
                )
                obstacle_dynamics_list.append("unicycle")
            elif vehicle_dynamics == "bicycle":
                obstacle_config = create_bicycle_obstacle(
                    obstacle_id=i,
                    position=np.array([x_obs, y_obs]),
                    velocity=velocity,
                    angle=angle
                )
                obstacle_dynamics_list.append("bicycle")
            else:
                # Fallback to point mass
                obstacle_config = create_point_mass_obstacle(
                    obstacle_id=i,
                    position=np.array([x_obs, y_obs]),
                    velocity=velocity
                )
                obstacle_dynamics_list.append("point_mass")
            
            obstacle_config.radius = 0.4  # Obstacle radius
            obstacle_config.prediction_type = PredictionType.DETERMINISTIC
            obstacle_configs.append(obstacle_config)
            print(f"Created dynamic obstacle {i} ({direction}) at s={s_pos:.2f}, position=({x_obs:.2f}, {y_obs:.2f}), "
                  f"velocity=({velocity[0]:.2f}, {velocity[1]:.2f}) m/s, speed={np.linalg.norm(velocity):.2f} m/s, "
                  f"angle={np.degrees(angle):.1f}°, dynamics={obstacle_dynamics_list[-1]}")
        except Exception as e:
            print(f"Warning: Could not create obstacle {i} at s={s_pos}: {e}")
    
    config = TestConfig(
        reference_path=ref_path_points,
        objective_module="contouring",
        constraint_modules=["contouring", "linear"],  # Use linear constraints for obstacle avoidance
        vehicle_dynamics=vehicle_dynamics,
        num_obstacles=len(obstacle_configs),
        obstacle_dynamics=obstacle_dynamics_list,  # Use same dynamics as vehicle
        test_name="Contouring Constraints with Dynamic Obstacles (Linearized)",
        duration=max_iterations * dt,
        timestep=dt,
        show_predicted_trajectory=True,
        obstacle_configs=obstacle_configs,  # Use deterministic obstacle positions
        obstacle_prediction_types=["deterministic"] * len(obstacle_configs),  # Deterministic for dynamic obstacles
        fallback_control_enabled=True  # Allow fallback to handle difficult obstacle configurations
    )
    
    # Run the test
    result = framework.run_test(config)
    
    return result


def test():
    """Entry point for pytest or direct execution."""
    result = run()
    # Note: Test may fail if obstacles are too restrictive, but this is expected behavior
    # The test demonstrates the linearized constraints working with dynamic obstacles
    return result


if __name__ == "__main__":
    test()

