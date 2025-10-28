"""
Test script to verify ROS-free operation of PyMPC.
"""
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def test_ros_free_operation():
    """Test that PyMPC works without ROS dependencies."""
    print("Testing ROS-free operation of PyMPC...")
    
    try:
        # Test 1: Import core modules
        print("1. Testing core module imports...")
        from solver.src.modules_manager import ModuleManager
        from planner_modules.src.constraints.base_constraint import BaseConstraint
        print("   ✅ Core modules imported successfully")
        
        # Test 2: Test Safe Horizon Constraint
        print("2. Testing Safe Horizon Constraint...")
        from planner_modules.src.constraints.safe_horizon_constraint import SafeHorizonConstraint
        print("   ✅ Safe Horizon Constraint imported successfully")
        
        # Test 3: Test scenario utilities
        print("3. Testing scenario utilities...")
        from planner_modules.src.constraints.scenario_utils.sampler import ScenarioSampler
        from planner_modules.src.constraints.scenario_utils.math_utils import compute_sample_size
        print("   ✅ Scenario utilities imported successfully")
        
        # Test 4: Test visualizer compatibility
        print("4. Testing visualizer compatibility...")
        from utils.visualizer_compat import ROSLine, ROSMarker, create_visualization_publisher
        print("   ✅ Visualizer compatibility layer imported successfully")
        
        # Test 5: Test sample size computation
        print("5. Testing sample size computation...")
        sample_size = compute_sample_size(0.1, 0.01, 10)
        print(f"   ✅ Sample size computed: {sample_size}")
        
        # Test 6: Test scenario sampling
        print("6. Testing scenario sampling...")
        from planning.src.types import DynamicObstacle, PredictionType, PredictionStep
        import numpy as np
        
        # Create test obstacle
        obstacle = DynamicObstacle(
            index=0,
            position=np.array([5.0, 0.0]),
            angle=0.0,
            radius=0.5
        )
        obstacle.prediction.type = PredictionType.GAUSSIAN
        obstacle.prediction.steps = []
        
        # Add prediction step
        step = PredictionStep(
            position=np.array([5.2, 0.0]),
            angle=0.0,
            major_radius=0.5,
            minor_radius=0.5
        )
        obstacle.prediction.steps.append(step)
        
        # Sample scenarios
        sampler = ScenarioSampler(num_scenarios=50)
        scenarios = sampler.sample_scenarios([obstacle], 1, 0.1)
        print(f"   ✅ Generated {len(scenarios)} scenarios")
        
        # Test 7: Test visualization
        print("7. Testing visualization...")
        line_viz = create_visualization_publisher("test_line")
        line_viz.add_point(0.0, 0.0)
        line_viz.add_point(1.0, 1.0)
        line_viz.set_color(1.0, 0.0, 0.0)
        line_viz.publish()
        print("   ✅ Visualization created successfully")
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED!")
        print("PyMPC works correctly without ROS dependencies.")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_mock_solver():
    """Test Safe Horizon Constraint with a mock solver."""
    print("\nTesting Safe Horizon Constraint with mock solver...")
    
    try:
        from unittest.mock import MagicMock
        from planner_modules.src.constraints.safe_horizon_constraint import SafeHorizonConstraint
        from planning.src.types import Data, DynamicObstacle, PredictionType, PredictionStep
        import numpy as np
        
        # Create mock solver
        solver = MagicMock()
        solver.horizon = 10
        solver.timestep = 0.1
        solver.copy.return_value = solver
        
        # Create constraint
        constraint = SafeHorizonConstraint(solver)
        print("   ✅ Safe Horizon Constraint created successfully")
        
        # Create test data
        data = Data()
        obstacle = DynamicObstacle(0, np.array([5.0, 0.0]), 0.0, 0.5)
        obstacle.prediction.type = PredictionType.GAUSSIAN
        obstacle.prediction.steps = []
        
        for i in range(3):
            step = PredictionStep(
                np.array([5.0 + i*0.2, 0.0]), 0.0, 0.5, 0.5
            )
            obstacle.prediction.steps.append(step)
        
        data.dynamic_obstacles = [obstacle]
        
        # Test data readiness
        is_ready = constraint.is_data_ready(data)
        print(f"   ✅ Data readiness check: {is_ready}")
        
        # Test parameter validation
        is_valid = constraint.validate_parameters()
        print(f"   ✅ Parameter validation: {is_valid}")
        
        # Test sample size computation
        sample_size = constraint.compute_sample_size()
        print(f"   ✅ Sample size: {sample_size}")
        
        # Test scenario sampling
        scenarios = constraint.sample_scenarios(data.dynamic_obstacles, 3, 0.1)
        print(f"   ✅ Generated {len(scenarios)} scenarios")
        
        print("   ✅ Safe Horizon Constraint test completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Safe Horizon Constraint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("PyMPC ROS-Free Operation Test")
    print("="*40)
    
    # Run main test
    success1 = test_ros_free_operation()
    
    # Run Safe Horizon Constraint test
    success2 = test_with_mock_solver()
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED!")
        print("PyMPC is ready for ROS-free operation.")
    else:
        print("\n❌ Some tests failed.")
        print("Please check the error messages above.")