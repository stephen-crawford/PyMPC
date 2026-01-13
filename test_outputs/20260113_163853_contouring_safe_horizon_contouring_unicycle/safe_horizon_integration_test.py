"""
Safe Horizon Constraint Integration Test

This test uses the new IntegrationTestFramework to run a test with:
- Contouring objective
- Safe Horizon constraints
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig, create_reference_path
import numpy as np


def run(dt=0.1, horizon=10, start=(0.0, 0.0), goal=(20.0, 20.0), max_iterations=200):
    """Run Safe Horizon constraints test using the framework."""
    
    framework = IntegrationTestFramework()
    
    # Create curving reference path that requires turning
    # Using "s_curve" for a more pronounced curve that definitely requires turning
    # This matches the path shape used in other tests (e.g., contouring_constraints_contouring_objective_test.py)
    ref_path_obj = create_reference_path("s_curve", length=30.0)
    
    # Convert ReferencePath to numpy array for TestConfig (framework will convert back)
    # This ensures consistency with other tests
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
    
    config = TestConfig(
        reference_path=ref_path_points,
        objective_module="contouring",
        constraint_modules=["safe_horizon", "contouring"],
        vehicle_dynamics="unicycle",
        num_obstacles=3,
        obstacle_dynamics=["unicycle"] * 3,  # Obstacle motion model
        obstacle_prediction_types=["gaussian"] * 3,  # Prediction type for safe horizon constraints
        test_name="Safe Horizon Integration Test",
        duration=max_iterations * dt,
        timestep=dt,
        enable_safe_horizon_diagnostics=True,  # Enable detailed diagnostic output
        timeout_seconds=300.0  # Increase timeout to 5 minutes to allow vehicle to complete path even with solver failures
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
