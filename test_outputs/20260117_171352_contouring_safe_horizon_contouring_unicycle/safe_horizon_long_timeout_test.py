"""
Safe Horizon Constraint Integration Test with Extended Timeout

This test runs MPCC with Safe Horizon constraints for obstacle avoidance.
The Safe Horizon constraint is computationally expensive (~2.7s per MPC step),
so we use an extended timeout of 10 minutes to allow the full test to complete.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig, create_reference_path
import numpy as np


def run(dt=0.1, duration=20.0, timeout_minutes=10.0):
    """Run Safe Horizon constraints test with extended timeout.
    
    Args:
        dt: Timestep in seconds (default 0.1s)
        duration: Simulation duration in seconds (default 20s)
        timeout_minutes: Test timeout in minutes (default 10 minutes)
    """
    
    framework = IntegrationTestFramework()
    
    # Create curved reference path
    ref_path = create_reference_path("curve", length=20.0)
    
    config = TestConfig(
        reference_path=ref_path,
        objective_module="contouring",
        constraint_modules=["safe_horizon", "contouring"],
        vehicle_dynamics="unicycle",
        num_obstacles=1,
        obstacle_dynamics=["unicycle"],
        obstacle_prediction_types=["gaussian"],
        test_name="Safe Horizon Long Timeout Test",
        duration=duration,
        timestep=dt,
        show_predicted_trajectory=True,
        # CRITICAL: Extended timeout for Safe Horizon (2.7s per step)
        # For 200 steps (20s at 0.1s), need ~540s compute time
        timeout_seconds=timeout_minutes * 60.0,
    )
    
    # Run the test
    print(f"\n=== Test Configuration ===")
    print(f"Objective: contouring")
    print(f"Constraints: safe_horizon, contouring")
    print(f"Duration: {duration}s")
    print(f"Timestep: {dt}s")
    print(f"Max steps: {int(duration/dt)}")
    print(f"Timeout: {timeout_minutes} minutes ({timeout_minutes*60}s)")
    print(f"Expected compute time: ~{int(duration/dt) * 2.7 / 60:.1f} minutes")
    print()
    
    result = framework.run_test(config)
    
    # Report results
    print(f"\n=== Results ===")
    print(f"Vehicle states: {len(result.vehicle_states)}")
    print(f"Success: {result.success}")
    if hasattr(result, 'output_folder'):
        print(f"Output folder: {result.output_folder}")
    
    return result


def test():
    """Entry point for pytest or direct execution."""
    result = run()
    assert result.success, "Test should complete successfully"
    return result


if __name__ == "__main__":
    test()
