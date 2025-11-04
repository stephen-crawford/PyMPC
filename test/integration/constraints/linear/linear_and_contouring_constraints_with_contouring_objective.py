"""
Linear and Contouring Constraints Test (with goal)

This test uses the new IntegrationTestFramework to run a test with:
- Contouring objective
- Linear constraints
- Contouring constraints
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig, create_reference_path


def run(dt=0.1, horizon=10, start=(0.0, 0.0), goal=(20.0, 20.0), max_iterations=200):
    """Run Linear + Contouring constraints test using the framework."""
    
    framework = IntegrationTestFramework()
    
    # Create curved reference path
    ref_path_points = create_reference_path("curve", length=25.0)
    
    config = TestConfig(
        reference_path=ref_path_points,
        objective_module="contouring",
        constraint_modules=["linear", "contouring"],
        vehicle_dynamics="unicycle",
        num_obstacles=3,
        obstacle_dynamics=["deterministic"] * 3,
        test_name="Linear and Contouring Constraints Test",
        duration=max_iterations * dt,
        timestep=dt
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
