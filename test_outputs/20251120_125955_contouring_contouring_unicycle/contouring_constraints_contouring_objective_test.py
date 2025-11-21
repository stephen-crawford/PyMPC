"""
Contouring Constraints Test with Contouring Objective (No Obstacles)

This test verifies that:
- Contouring objective works with contouring constraints
- All computation is symbolic (no numeric fallbacks)
- Reference path and constraints are properly visualized
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig, create_reference_path


def run(dt=0.1, horizon=10, max_iterations=200):
    """Run Contouring constraints test with contouring objective (no obstacles)."""
    
    framework = IntegrationTestFramework()
    
    # Create curved reference path
    ref_path_points = create_reference_path("curve", length=25.0)
    
    config = TestConfig(
        reference_path=ref_path_points,
        objective_module="contouring",
        constraint_modules=["contouring"],  # Only contouring constraints (no obstacles)
        vehicle_dynamics="unicycle",
        num_obstacles=0,
        obstacle_dynamics=[],
        test_name="Contouring Objective with Contouring Constraints",
        duration=max_iterations * dt,
        timestep=dt,
        show_predicted_trajectory=True,
        fallback_control_enabled=False,
        max_consecutive_failures=50
    )
    
    # Run the test
    print(f"\n=== Test Configuration ===")
    print(f"Objective: contouring")
    print(f"Constraints: contouring")
    print(f"Number of obstacles: 0")
    print()
    result = framework.run_test(config)
    
    return result


def test():
    """Entry point for pytest or direct execution."""
    result = run()
    assert result.success, "Test should complete successfully"
    return result


if __name__ == "__main__":
    test()


