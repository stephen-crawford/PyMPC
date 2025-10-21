"""
Example Usage of Standardized MPCC Testing Framework

This script demonstrates how to use the standardized MPCC testing framework
with various configurations and scenarios.
"""

import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pympc.testing.mpcc_test_framework import (
    TestConfig, RoadConfig, VehicleConfig, MPCConfig, ObstacleConfig, 
    PerceptionConfig, PerceptionShape, create_standard_mpcc_test,
    create_curved_road_test, create_perception_test
)
from pympc.testing.enhanced_mpcc_framework import (
    create_enhanced_mpcc_test, create_curved_road_enhanced_test
)
from pympc.testing.comprehensive_mpcc_runner import ComprehensiveMPCCRunner


def example_basic_mpcc():
    """Example: Basic MPCC test with curved road."""
    print("="*60)
    print("EXAMPLE: Basic MPCC Test")
    print("="*60)
    
    # Create a basic MPCC test
    test = create_standard_mpcc_test(
        test_name="example_basic_mpcc",
        road=RoadConfig(
            road_type="curved",
            length=100.0,
            width=6.0,
            curvature_intensity=1.0
        ),
        obstacles=ObstacleConfig(
            num_obstacles=3,
            radius_range=(0.8, 1.5),
            velocity_range=(2.0, 6.0),
            intersection_probability=0.7
        ),
        perception=PerceptionConfig(enabled=False),  # No perception limitations
        output_dir="example_outputs"
    )
    
    # Run the test
    result = test.run_test()
    
    # Print results
    print(f"Test completed: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Duration: {result['duration']:.2f}s")
    print(f"Steps completed: {result['steps_completed']}")
    print(f"Constraint violations: {len(result['constraint_violations'])}")
    
    return result


def example_perception_mpcc():
    """Example: MPCC with perception limitations."""
    print("\n" + "="*60)
    print("EXAMPLE: MPCC with Perception Limitations")
    print("="*60)
    
    # Create MPCC test with perception limitations
    test = create_perception_test(
        test_name="example_perception_mpcc",
        road=RoadConfig(
            road_type="s_curve",
            length=120.0,
            width=6.0,
            curvature_intensity=1.2
        ),
        obstacles=ObstacleConfig(
            num_obstacles=4,
            radius_range=(1.0, 2.0),
            velocity_range=(1.5, 8.0),
            trajectory_type="random_walk",
            intersection_probability=0.8
        ),
        perception=PerceptionConfig(
            shape=PerceptionShape.CONE,
            distance=20.0,
            angle=np.pi/3,  # 60 degrees
            enabled=True
        ),
        output_dir="example_outputs"
    )
    
    # Run the test
    result = test.run_test()
    
    # Print results
    print(f"Test completed: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Duration: {result['duration']:.2f}s")
    print(f"Steps completed: {result['steps_completed']}")
    print(f"Constraint violations: {len(result['constraint_violations'])}")
    
    # Print perception statistics
    perception_stats = result['perception_history']
    if perception_stats:
        avg_visible = np.mean([p['visible_obstacles'] for p in perception_stats])
        print(f"Average visible obstacles: {avg_visible:.1f}")
    
    return result


def example_enhanced_mpcc():
    """Example: Enhanced MPCC with C++ reference implementation."""
    print("\n" + "="*60)
    print("EXAMPLE: Enhanced MPCC (C++ Reference)")
    print("="*60)
    
    # Create enhanced MPCC test
    test = create_enhanced_mpcc_test(
        test_name="example_enhanced_mpcc",
        road=RoadConfig(
            road_type="curved",
            length=100.0,
            width=6.0,
            curvature_intensity=1.0
        ),
        obstacles=ObstacleConfig(
            num_obstacles=3,
            radius_range=(0.8, 1.5),
            velocity_range=(2.0, 6.0),
            intersection_probability=0.7
        ),
        perception=PerceptionConfig(enabled=False),
        mpc=MPCConfig(
            contouring_weight=2.0,
            lag_weight=1.0,
            velocity_weight=0.1,
            progress_weight=1.5
        ),
        output_dir="example_outputs"
    )
    
    # Run the test
    result = test.run_test()
    
    # Print results
    print(f"Test completed: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Duration: {result['duration']:.2f}s")
    print(f"Steps completed: {result['steps_completed']}")
    print(f"Constraint violations: {len(result['constraint_violations'])}")
    
    return result


def example_custom_configuration():
    """Example: Custom test configuration."""
    print("\n" + "="*60)
    print("EXAMPLE: Custom Test Configuration")
    print("="*60)
    
    # Create custom test configuration
    custom_config = TestConfig(
        test_name="example_custom_mpcc",
        road=RoadConfig(
            road_type="curved",
            length=150.0,
            width=8.0,
            curvature_intensity=1.5
        ),
        vehicle=VehicleConfig(
            length=5.0,
            width=2.0,
            max_velocity=20.0,
            max_acceleration=4.0,
            max_steering_angle=0.6
        ),
        mpc=MPCConfig(
            horizon=20,
            timestep=0.05,
            max_steps=200,
            contouring_weight=3.0,
            lag_weight=2.0,
            velocity_weight=0.2,
            progress_weight=2.0
        ),
        obstacles=ObstacleConfig(
            num_obstacles=5,
            radius_range=(1.0, 2.5),
            velocity_range=(3.0, 10.0),
            trajectory_type="circular",
            intersection_probability=0.9
        ),
        perception=PerceptionConfig(
            shape=PerceptionShape.RECTANGLE,
            distance=30.0,
            width=20.0,
            height=25.0,
            enabled=True
        ),
        output_dir="example_outputs"
    )
    
    # Create and run test
    from pympc.testing.mpcc_test_framework import MPCCTestFramework
    test = MPCCTestFramework(custom_config)
    result = test.run_test()
    
    # Print results
    print(f"Custom test completed: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Duration: {result['duration']:.2f}s")
    print(f"Steps completed: {result['steps_completed']}")
    print(f"Constraint violations: {len(result['constraint_violations'])}")
    
    return result


def example_comprehensive_suite():
    """Example: Run comprehensive test suite."""
    print("\n" + "="*60)
    print("EXAMPLE: Comprehensive Test Suite")
    print("="*60)
    
    # Create comprehensive test runner
    runner = ComprehensiveMPCCRunner(output_dir="example_outputs")
    
    # Run a subset of the comprehensive suite
    print("Running basic MPCC tests...")
    basic_results = runner.run_basic_mpcc_tests()
    
    print("Running perception tests...")
    perception_results = runner.run_perception_tests()
    
    print("Running enhanced MPCC tests...")
    enhanced_results = runner.run_enhanced_mpcc_tests()
    
    # Print summary
    print(f"\nBasic tests: {len(basic_results)} tests")
    print(f"Perception tests: {len(perception_results)} tests")
    print(f"Enhanced tests: {len(enhanced_results)} tests")
    
    # Count successes
    all_results = {**basic_results, **perception_results, **enhanced_results}
    successful = sum(1 for r in all_results.values() if r['success'])
    total = len(all_results)
    
    print(f"Overall success rate: {successful/total*100:.1f}%")
    
    return all_results


def example_perception_shapes():
    """Example: Test different perception shapes."""
    print("\n" + "="*60)
    print("EXAMPLE: Different Perception Shapes")
    print("="*60)
    
    shapes = [
        (PerceptionShape.CIRCLE, "Circle"),
        (PerceptionShape.CONE, "Cone"),
        (PerceptionShape.RECTANGLE, "Rectangle")
    ]
    
    results = {}
    
    for shape, name in shapes:
        print(f"\nTesting {name} perception...")
        
        test = create_perception_test(
            test_name=f"example_{name.lower()}_perception",
            road=RoadConfig(
                road_type="curved",
                length=100.0,
                width=6.0,
                curvature_intensity=1.0
            ),
            obstacles=ObstacleConfig(
                num_obstacles=3,
                radius_range=(0.8, 1.5),
                velocity_range=(2.0, 6.0),
                intersection_probability=0.7
            ),
            perception=PerceptionConfig(
                shape=shape,
                distance=18.0,
                angle=np.pi/4,
                width=12.0,
                height=18.0,
                enabled=True
            ),
            output_dir="example_outputs"
        )
        
        result = test.run_test()
        results[name] = result
        
        print(f"  {name}: {'SUCCESS' if result['success'] else 'FAILED'} "
              f"({result['duration']:.2f}s, {result['steps_completed']} steps)")
    
    return results


def main():
    """Main example function."""
    print("MPCC Testing Framework - Example Usage")
    print("="*80)
    print("This script demonstrates various ways to use the standardized")
    print("MPCC testing framework with different configurations.")
    print()
    
    # Create output directory
    import os
    os.makedirs("example_outputs", exist_ok=True)
    
    # Run examples
    print("Running example tests...")
    
    # Example 1: Basic MPCC
    basic_result = example_basic_mpcc()
    
    # Example 2: Perception MPCC
    perception_result = example_perception_mpcc()
    
    # Example 3: Enhanced MPCC
    enhanced_result = example_enhanced_mpcc()
    
    # Example 4: Custom configuration
    custom_result = example_custom_configuration()
    
    # Example 5: Different perception shapes
    shape_results = example_perception_shapes()
    
    # Example 6: Comprehensive suite
    comprehensive_results = example_comprehensive_suite()
    
    # Summary
    print("\n" + "="*80)
    print("EXAMPLE SUMMARY")
    print("="*80)
    
    all_tests = [
        ("Basic MPCC", basic_result['success']),
        ("Perception MPCC", perception_result['success']),
        ("Enhanced MPCC", enhanced_result['success']),
        ("Custom Configuration", custom_result['success']),
    ]
    
    for name, success in all_tests:
        print(f"{name:20} | {'SUCCESS' if success else 'FAILED'}")
    
    print(f"\nPerception shape tests:")
    for shape, result in shape_results.items():
        print(f"  {shape:15} | {'SUCCESS' if result['success'] else 'FAILED'}")
    
    print(f"\nComprehensive suite: {len(comprehensive_results)} tests")
    successful_comprehensive = sum(1 for r in comprehensive_results.values() if r['success'])
    print(f"  Success rate: {successful_comprehensive/len(comprehensive_results)*100:.1f}%")
    
    print(f"\nAll example results saved to: example_outputs/")
    print("Check the generated GIF files for visualizations!")


if __name__ == "__main__":
    main()
