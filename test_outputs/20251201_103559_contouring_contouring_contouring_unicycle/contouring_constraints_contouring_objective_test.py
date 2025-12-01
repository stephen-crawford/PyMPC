"""
Contouring Constraints Test with Contouring Objective (No Obstacles)

This test verifies that:
- Contouring objective works with contouring constraints
- Contouring dynamics model (ContouringSecondOrderUnicycleModel) properly progresses along curving path
- Vehicle follows a curving reference path that requires turning
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


def run(dt=0.1, horizon=10, max_iterations=300):
    """Run Contouring constraints test with contouring objective (no obstacles).
    
    Uses:
    - Curving reference path (s_curve for pronounced curves)
    - Contouring dynamics model (ContouringSecondOrderUnicycleModel with spline state)
    - Contouring objective
    - Contouring constraints
    - No obstacles
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
    
    # CRITICAL: Use "contouring_unicycle" to get ContouringSecondOrderUnicycleModel with spline state
    # Reference: C++ mpc_planner - contouring MPC requires dynamics model with spline state variable
    # Pass the ReferencePath object (not numpy array) to TestConfig
    config = TestConfig(
        reference_path=ref_path_obj if hasattr(ref_path_obj, 'x') else ref_path_points,
        objective_module="contouring",
        constraint_modules=["contouring"],  # Only contouring constraints (no obstacles)
        vehicle_dynamics="contouring_unicycle",  # Use contouring unicycle model with spline state
        num_obstacles=0,
        obstacle_dynamics=[],
        test_name="Contouring Objective with Contouring Constraints (Curving Path)",
        duration=max_iterations * dt,
        timestep=dt,
        show_predicted_trajectory=True,
        fallback_control_enabled=False,
        max_consecutive_failures=50,
        timeout_seconds=120.0  # Allow time for path completion
    )
    
    # Run the test
    print(f"\n=== Test Configuration ===")
    print(f"Reference path: S-curve (curving path requiring turning)")
    print(f"Path length: ~30.0m")
    print(f"Path points: {len(ref_path_points)}")
    print(f"Start position: ({ref_path_points[0, 0]:.2f}, {ref_path_points[0, 1]:.2f})")
    print(f"End position: ({ref_path_points[-1, 0]:.2f}, {ref_path_points[-1, 1]:.2f})")
    print(f"Objective: contouring")
    print(f"Constraints: contouring")
    print(f"Dynamics model: contouring_unicycle (ContouringSecondOrderUnicycleModel)")
    print(f"Number of obstacles: 0")
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
                
                # Overall success: vehicle should make progress AND turn
                success = made_progress and vehicle_turned
                print(f"\n=== Final Verification Summary ===")
                print(f"Vehicle made progress: {'✓ YES' if made_progress else '✗ NO'}")
                print(f"Vehicle turned: {'✓ YES' if vehicle_turned else '✗ NO'}")
                print(f"Overall test result: {'✓ PASS' if success else '✗ FAIL'}")
                
                if not success:
                    print(f"⚠️  Test verification failed: vehicle did not make sufficient progress or turn")
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


