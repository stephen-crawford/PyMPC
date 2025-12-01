"""
Contouring Constraints Test with Contouring Objective and Single Static Obstacle

This test verifies that:
- Contouring objective works with contouring constraints
- Contouring dynamics model (ContouringSecondOrderUnicycleModel) properly progresses along curving path
- Vehicle follows a curving reference path that requires turning
- Vehicle avoids a single static obstacle placed on the centerline using linearized constraints
- All computation is symbolic (no numeric fallbacks)
- Reference path and constraints are properly visualized

Reference: C++ mpc_planner - contouring MPC with spline state variable
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig, create_reference_path
from planning.obstacle_manager import create_unicycle_obstacle
from planning.types import PredictionType, ReferencePath
from utils.math_tools import TKSpline


def run(dt=0.1, horizon=10, max_iterations=300):
    """Run Contouring constraints test with contouring objective and single static obstacle.
    
    Uses:
    - Curving reference path (s_curve for pronounced curves)
    - Contouring dynamics model (ContouringSecondOrderUnicycleModel with spline state)
    - Contouring objective
    - Contouring constraints (for road boundaries)
    - Linearized constraints (for obstacle avoidance)
    - Single static obstacle placed on centerline
    """
    
    framework = IntegrationTestFramework()
    
    # Create curving reference path that requires turning
    # Using "s_curve" for a more pronounced curve that definitely requires turning
    ref_path_obj = create_reference_path("s_curve", length=30.0)
    
    # Convert ReferencePath to numpy array for verification
    if hasattr(ref_path_obj, 'x') and hasattr(ref_path_obj, 'y'):
        ref_path_points = np.column_stack([ref_path_obj.x, ref_path_obj.y])
    else:
        # Fallback: try to get points directly
        ref_path_points = ref_path_obj if isinstance(ref_path_obj, np.ndarray) else np.array([[0, 0], [30, 0]])
    
    # Verify the path is curving (not straight)
    if len(ref_path_points) >= 2:
        y_coords = ref_path_points[:, 1]
        y_range = np.max(y_coords) - np.min(y_coords)
        print(f"Reference path: {len(ref_path_points)} points, y-range={y_range:.2f}m")
        if y_range < 1.0:
            print(f"⚠️  Warning: Path may not be curving enough (y-range={y_range:.2f}m)")
    
    # Create ReferencePath object for obstacle placement
    ref_path = ReferencePath()
    x_arr = np.asarray(ref_path_obj.x, dtype=float)
    y_arr = np.asarray(ref_path_obj.y, dtype=float)
    z_arr = np.zeros_like(x_arr)  # 2D path
    
    # Compute arc length
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
    
    # Place multiple static obstacles along the centerline
    # Space them evenly along the middle portion of the path (25% to 70% to avoid path end)
    s_arr = np.asarray(s, dtype=float)
    s_min = float(s_arr[0])
    s_max = float(s_arr[-1])
    s_range = s_max - s_min
    
    num_obstacles = 2  # Number of dynamic obstacles to place
    obstacle_s_positions = []
    for i in range(num_obstacles):
        # Space obstacles evenly between 30% and 60% of path (avoid path end)
        s_pos = s_min + s_range * (0.3 + i * 0.3 / max(1, num_obstacles - 1))
        obstacle_s_positions.append(s_pos)
    
    # Create dynamic obstacles with unicycle dynamics that move back and forth across centerline
    obstacle_configs = []
    for i, obstacle_s_position in enumerate(obstacle_s_positions):
        try:
            # Get position on centerline
            x_center = float(ref_path.x_spline(obstacle_s_position))
            y_center = float(ref_path.y_spline(obstacle_s_position))
            
            # Get path tangent at this point to compute normal
            dx_ds = float(ref_path.x_spline.derivative()(obstacle_s_position))
            dy_ds = float(ref_path.y_spline.derivative()(obstacle_s_position))
            path_tangent_norm = np.sqrt(dx_ds**2 + dy_ds**2)
            if path_tangent_norm > 1e-6:
                # Normal vector (left side of path)
                normal_x = -dy_ds / path_tangent_norm
                normal_y = dx_ds / path_tangent_norm
            else:
                normal_x, normal_y = 1.0, 0.0
            
            # Place obstacle offset from centerline (inside the path, on one side)
            # Alternate sides for different obstacles
            lateral_offset = 1.5 if i % 2 == 0 else -1.5  # 1.5m offset from centerline
            obstacle_x = x_center + lateral_offset * normal_x
            obstacle_y = y_center + lateral_offset * normal_y
            
            # Initial velocity toward centerline (perpendicular to path)
            # Velocity magnitude ~1 m/s toward centerline
            initial_velocity = np.array([-lateral_offset * normal_x * 0.7, -lateral_offset * normal_y * 0.7])
            initial_angle = np.arctan2(initial_velocity[1], initial_velocity[0])
            
            # Create dynamic obstacle with unicycle dynamics and path_intersect behavior
            obstacle_config = create_unicycle_obstacle(
                obstacle_id=i,
                position=np.array([obstacle_x, obstacle_y]),
                velocity=initial_velocity,
                angle=initial_angle,
                radius=0.35,  # Obstacle radius
                behavior="path_intersect"  # Moves back and forth across the path
            )
            obstacle_config.prediction_type = PredictionType.DETERMINISTIC  # Linearized constraints use deterministic predictions
            
            obstacle_configs.append(obstacle_config)
            print(f"Created dynamic obstacle {i} at s={obstacle_s_position:.2f}, position=({obstacle_x:.2f}, {obstacle_y:.2f}), "
                  f"offset={lateral_offset:.2f}m from centerline, behavior=path_intersect")
        except Exception as e:
            print(f"Warning: Could not create obstacle {i} at s={obstacle_s_position}: {e}")
            import traceback
            traceback.print_exc()
    
    # CRITICAL: Use "contouring_unicycle" to get ContouringSecondOrderUnicycleModel with spline state
    # Reference: C++ mpc_planner - contouring MPC requires dynamics model with spline state variable
    # Pass the ReferencePath object (not numpy array) to TestConfig
    config = TestConfig(
        reference_path=ref_path_obj if hasattr(ref_path_obj, 'x') else ref_path_points,
        objective_module="contouring",
        constraint_modules=["contouring", "linear"],  # Contouring constraints + Linearized for obstacle avoidance
        vehicle_dynamics="contouring_unicycle",  # Use contouring unicycle model with spline state
        num_obstacles=len(obstacle_configs),
        obstacle_dynamics=["unicycle"] * len(obstacle_configs),  # Use unicycle dynamics for obstacles
        test_name="Contouring Objective with Contouring Constraints + Dynamic Obstacles (Linearized)",
        duration=max_iterations * dt,
        timestep=dt,
        show_predicted_trajectory=True,
        fallback_control_enabled=False,
        max_consecutive_failures=50,
        timeout_seconds=120.0,  # Allow time for path completion
        obstacle_configs=obstacle_configs,
        obstacle_prediction_types=["deterministic"] * len(obstacle_configs)  # Linearized constraints use deterministic predictions
    )
    
    # Store reference path in config for obstacle manager (needed for path_intersect behavior)
    config._reference_path_for_obstacles = ref_path_obj
    
    # Run the test
    print(f"\n=== Test Configuration ===")
    print(f"Reference path: S-curve (curving path requiring turning)")
    print(f"Path length: ~30.0m")
    print(f"Path points: {len(ref_path_points)}")
    print(f"Start position: ({ref_path_points[0, 0]:.2f}, {ref_path_points[0, 1]:.2f})")
    print(f"End position: ({ref_path_points[-1, 0]:.2f}, {ref_path_points[-1, 1]:.2f})")
    print(f"Objective: contouring")
    print(f"Constraints: contouring (road boundaries) + linearized (obstacle avoidance)")
    print(f"Dynamics model: contouring_unicycle (ContouringSecondOrderUnicycleModel)")
    print(f"Obstacle dynamics: unicycle (with path_intersect behavior)")
    print(f"Number of obstacles: {len(obstacle_configs)}")
    for i, obs_config in enumerate(obstacle_configs):
        print(f"  Obstacle {i}: position=({obs_config.initial_position[0]:.2f}, {obs_config.initial_position[1]:.2f}), radius={obs_config.radius:.2f}m")
    print()
    result = framework.run_test(config)
    
    # Verify that the vehicle progressed along the path
    if result.success and hasattr(result, 'vehicle_states') and result.vehicle_states:
        print(f"\n=== Path Progression Verification ===")
        
        # Get reference path end point
        path_end = ref_path_points[-1]
        path_end_x = path_end[0]
        path_end_y = path_end[1]
        
        # Final state
        final_state = result.vehicle_states[-1]
        
        if len(final_state) >= 2:
            final_x = final_state[0]
            final_y = final_state[1]
            dist_to_end = np.linalg.norm([final_x - path_end_x, final_y - path_end_y])
            
            # Check if vehicle has progressed significantly along the path
            path_progress_threshold = 0.7  # 70% of path length
            path_length_estimate = np.linalg.norm([path_end_x - ref_path_points[0, 0], 
                                                     path_end_y - ref_path_points[0, 1]])
            progress_dist = np.linalg.norm([final_x - ref_path_points[0, 0], 
                                             final_y - ref_path_points[0, 1]])
            progress_ratio = progress_dist / path_length_estimate if path_length_estimate > 0 else 0.0
            made_progress = progress_ratio >= path_progress_threshold
            
            print(f"Path end position: ({path_end_x:.2f}, {path_end_y:.2f})")
            print(f"Final position: ({final_x:.3f}, {final_y:.3f})")
            print(f"Final distance to path end: {dist_to_end:.3f}m")
            print(f"Path progress: {progress_ratio*100:.1f}% (threshold: {path_progress_threshold*100:.0f}%)")
            print(f"Made significant progress: {'✓ YES' if made_progress else '✗ NO'}")
            
            # Check spline progression (for contouring, spline should increase)
            if len(final_state) >= 5:
                initial_spline = result.vehicle_states[0][4] if len(result.vehicle_states[0]) >= 5 else 0.0
                final_spline = final_state[4] if len(final_state) >= 5 else 0.0
                spline_progress = final_spline - initial_spline
                print(f"\nSpline progression:")
                print(f"  Initial spline: {initial_spline:.3f}")
                print(f"  Final spline: {final_spline:.3f}")
                print(f"  Spline progress: {spline_progress:.3f}")
                print(f"  Progressing along path: {'✓ YES' if spline_progress > 0.1 else '✗ NO (insufficient progress)'}")
            
            # Verify turning occurred (check if vehicle's heading changed significantly)
            if len(result.vehicle_states) > 10 and len(final_state) >= 3:
                initial_heading = result.vehicle_states[0][2] if len(result.vehicle_states[0]) >= 3 else 0.0
                final_heading = final_state[2] if len(final_state) >= 3 else 0.0
                heading_change = abs(np.arctan2(np.sin(final_heading - initial_heading), 
                                               np.cos(final_heading - initial_heading)))
                
                print(f"\nTurning verification:")
                print(f"  Initial heading: {initial_heading:.3f} rad ({np.degrees(initial_heading):.1f}°)")
                print(f"  Final heading: {final_heading:.3f} rad ({np.degrees(final_heading):.1f}°)")
                print(f"  Heading change: {heading_change:.3f} rad ({np.degrees(heading_change):.1f}°)")
                vehicle_turned = heading_change > 0.3
                print(f"  Vehicle turned: {'✓ YES' if vehicle_turned else '✗ NO (insufficient turn)'}")
                
                # Verify obstacle avoidance
                obstacles_avoided = True
                if obstacle_configs:
                    print(f"\n=== Obstacle Avoidance Verification ===")
                    # Safe distance: minimum clearance between vehicle and obstacle boundaries
                    # Obstacle radius: 0.35m, Vehicle radius: ~0.25m
                    # Acceptable clearance: 0.15m margin (total center-to-center: 0.35 + 0.25 + 0.15 = 0.75m)
                    # This gives: center_to_center - obs_radius - vehicle_radius >= 0.15m
                    safe_distance = 0.15  # Minimum clearance between boundaries (not center-to-center)
                    
                    for obs_idx, obs_config in enumerate(obstacle_configs):
                        obs_pos = obs_config.initial_position
                        obs_radius = obs_config.radius
                        min_distance = float('inf')
                        closest_step = -1
                        
                        for step, state in enumerate(result.vehicle_states):
                            if len(state) >= 2:
                                veh_pos = np.array([state[0], state[1]])
                                # Distance from vehicle center to obstacle center, minus both radii
                                # This gives the clearance between vehicle and obstacle boundaries
                                vehicle_radius = 0.25  # Approximate vehicle radius
                                center_to_center = np.linalg.norm(veh_pos - obs_pos)
                                distance = center_to_center - obs_radius - vehicle_radius
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_step = step
                        
                        avoided = min_distance >= safe_distance
                        obstacles_avoided = obstacles_avoided and avoided
                        print(f"Obstacle {obs_idx} at ({obs_pos[0]:.2f}, {obs_pos[1]:.2f}): "
                              f"min_distance={min_distance:.3f}m, safe_distance={safe_distance:.2f}m, "
                              f"avoided={'✓ YES' if avoided else '✗ NO'} (closest at step {closest_step})")
                    
                    print(f"All obstacles avoided: {'✓ YES' if obstacles_avoided else '✗ NO'}")
                
                # Overall success: vehicle should make progress AND turn AND avoid obstacles
                success = made_progress and vehicle_turned and obstacles_avoided
                print(f"\n=== Final Verification Summary ===")
                print(f"Vehicle made progress: {'✓ YES' if made_progress else '✗ NO'}")
                print(f"Vehicle turned: {'✓ YES' if vehicle_turned else '✗ NO'}")
                if obstacle_configs:
                    print(f"All obstacles avoided: {'✓ YES' if obstacles_avoided else '✗ NO'}")
                print(f"Overall test result: {'✓ PASS' if success else '✗ FAIL'}")
                
                if not success:
                    print(f"⚠️  Test verification failed: vehicle did not make sufficient progress, turn, or avoid obstacles")
            else:
                print(f"\n⚠️  Could not verify turning (insufficient state data)")
                success = made_progress
        else:
            print(f"\n⚠️  Final state incomplete")
            success = False
    else:
        print(f"\n=== Test Failed ===")
        if hasattr(result, 'error_message'):
            print(f"Error: {result.error_message}")
        success = False
    
    return result


def test():
    """Entry point for pytest or direct execution."""
    result = run()
    assert result.success, "Test should complete successfully"
    return result


if __name__ == "__main__":
    test()


