"""
Test script for moving goal functionality.
The vehicle starts at (0, 0) and moves through a sequence of goals.
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig, create_reference_path

def main():
    """Run moving goal test."""
    framework = IntegrationTestFramework()
    
    # Define a sequence of goals: move in a square pattern
    goal_sequence = [
        [10.0, 0.0],   # First goal: straight ahead
        [10.0, 10.0],  # Second goal: turn right
        [0.0, 10.0],   # Third goal: turn back
        [0.0, 0.0],    # Final goal: return to start
    ]
    
    # Create test configuration with moving goals
    config = TestConfig(
        reference_path=create_reference_path("straight", 30.0),  # Long enough path for all goals
        objective_module="goal",
        constraint_modules=["linear"],
        vehicle_dynamics="unicycle",
        num_obstacles=0,
        obstacle_dynamics=[],
        obstacle_prediction_types=[],
        test_name="Moving Goal Test",
        duration=30.0,  # Longer duration to allow reaching all goals
        timestep=0.1,
        goal_sequence=goal_sequence,
        fallback_control_enabled=False
    )
    
    print("Running Moving Goal Test")
    print(f"Goal sequence: {goal_sequence}")
    print()
    
    result = framework.run_test(config)
    
    if result.success:
        print("✅ Moving goal test completed successfully")
        print(f"📁 Output folder: {result.output_folder}")
        print(f"📊 Vehicle states: {len(result.vehicle_states)}")
        print(f"⏱️  Avg computation time: {sum(result.computation_times)/len(result.computation_times):.3f}s")
        return 0
    else:
        print("❌ Moving goal test failed")
        print(f"📁 Output folder: {result.output_folder}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

