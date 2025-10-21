"""
MPCC Demo Script

This script demonstrates the standardized MPCC testing framework with
various configurations and scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pympc.testing.mpcc_test_framework import (
    MPCCTestFramework, TestConfig, RoadConfig, VehicleConfig, 
    MPCConfig, ObstacleConfig, PerceptionConfig, PerceptionShape,
    create_standard_mpcc_test, create_curved_road_test, create_perception_test
)
from pympc.testing.standardized_mpcc_runner import StandardizedMPCCRunner


def demo_basic_mpcc():
    """Demo: Basic MPCC with curved road."""
    print("="*60)
    print("DEMO: Basic MPCC with Curved Road")
    print("="*60)
    
    # Create basic MPCC test
    test = create_standard_mpcc_test(
        test_name="demo_basic_mpcc",
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
        output_dir="demo_outputs"
    )
    
    # Run test
    result = test.run_test()
    
    # Print results
    print(f"Test completed: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Duration: {result['duration']:.2f}s")
    print(f"Steps completed: {result['steps_completed']}")
    print(f"Constraint violations: {len(result['constraint_violations'])}")
    
    return result


def demo_perception_mpcc():
    """Demo: MPCC with perception limitations."""
    print("\n" + "="*60)
    print("DEMO: MPCC with Perception Limitations")
    print("="*60)
    
    # Create perception-limited MPCC test
    test = create_perception_test(
        test_name="demo_perception_mpcc",
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
        output_dir="demo_outputs"
    )
    
    # Run test
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


def demo_multiple_perception_shapes():
    """Demo: MPCC with different perception shapes."""
    print("\n" + "="*60)
    print("DEMO: MPCC with Different Perception Shapes")
    print("="*60)
    
    shapes = [
        (PerceptionShape.CIRCLE, "Circle"),
        (PerceptionShape.CONE, "Cone"),
        (PerceptionShape.RECTANGLE, "Rectangle")
    ]
    
    results = {}
    
    for shape, name in shapes:
        print(f"\nTesting {name} perception...")
        
        test = MPCCTestFramework(TestConfig(
            test_name=f"demo_{name.lower()}_perception",
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
            output_dir="demo_outputs"
        ))
        
        result = test.run_test()
        results[name] = result
        
        print(f"  {name}: {'SUCCESS' if result['success'] else 'FAILED'} "
              f"({result['duration']:.2f}s, {result['steps_completed']} steps)")
    
    return results


def demo_obstacle_density():
    """Demo: MPCC with different obstacle densities."""
    print("\n" + "="*60)
    print("DEMO: MPCC with Different Obstacle Densities")
    print("="*60)
    
    densities = [2, 4, 6, 8]
    results = {}
    
    for num_obstacles in densities:
        print(f"\nTesting with {num_obstacles} obstacles...")
        
        test = MPCCTestFramework(TestConfig(
            test_name=f"demo_{num_obstacles}_obstacles",
            road=RoadConfig(
                road_type="curved",
                length=100.0,
                width=6.0,
                curvature_intensity=1.0
            ),
            obstacles=ObstacleConfig(
                num_obstacles=num_obstacles,
                radius_range=(0.8, 1.5),
                velocity_range=(2.0, 6.0),
                trajectory_type="random_walk",
                intersection_probability=0.8
            ),
            perception=PerceptionConfig(
                shape=PerceptionShape.CONE,
                distance=20.0,
                angle=np.pi/3,
                enabled=True
            ),
            output_dir="demo_outputs"
        ))
        
        result = test.run_test()
        results[num_obstacles] = result
        
        print(f"  {num_obstacles} obstacles: {'SUCCESS' if result['success'] else 'FAILED'} "
              f"({result['duration']:.2f}s, {len(result['constraint_violations'])} violations)")
    
    return results


def demo_comprehensive_suite():
    """Demo: Run comprehensive test suite."""
    print("\n" + "="*60)
    print("DEMO: Comprehensive Test Suite")
    print("="*60)
    
    # Create test runner
    runner = StandardizedMPCCRunner(output_dir="demo_outputs")
    
    # Run a subset of the comprehensive suite
    print("Running standard test suite...")
    standard_results = runner.run_standard_test_suite()
    
    print("Running perception analysis...")
    perception_results = runner.run_perception_analysis_suite()
    
    print("Running obstacle density tests...")
    density_results = runner.run_obstacle_density_suite()
    
    # Combine results
    all_results = {
        'standard': standard_results,
        'perception': perception_results,
        'density': density_results
    }
    
    # Generate report
    runner._generate_comprehensive_report(all_results, 0.0)  # Duration not tracked in demo
    
    return all_results


def main():
    """Main demo function."""
    print("MPCC Testing Framework Demo")
    print("="*80)
    print("This demo showcases the standardized MPCC testing framework")
    print("with various configurations and scenarios.")
    print()
    
    # Create output directory
    import os
    os.makedirs("demo_outputs", exist_ok=True)
    
    # Run individual demos
    print("Running individual demos...")
    
    # Demo 1: Basic MPCC
    basic_result = demo_basic_mpcc()
    
    # Demo 2: Perception MPCC
    perception_result = demo_perception_mpcc()
    
    # Demo 3: Multiple perception shapes
    shape_results = demo_multiple_perception_shapes()
    
    # Demo 4: Obstacle density
    density_results = demo_obstacle_density()
    
    # Demo 5: Comprehensive suite
    comprehensive_results = demo_comprehensive_suite()
    
    # Summary
    print("\n" + "="*80)
    print("DEMO SUMMARY")
    print("="*80)
    
    all_tests = [
        ("Basic MPCC", basic_result['success']),
        ("Perception MPCC", perception_result['success']),
    ]
    
    for name, success in all_tests:
        print(f"{name:20} | {'SUCCESS' if success else 'FAILED'}")
    
    print(f"\nShape perception tests:")
    for shape, result in shape_results.items():
        print(f"  {shape:15} | {'SUCCESS' if result['success'] else 'FAILED'}")
    
    print(f"\nObstacle density tests:")
    for num_obs, result in density_results.items():
        print(f"  {num_obs:2d} obstacles     | {'SUCCESS' if result['success'] else 'FAILED'}")
    
    print(f"\nAll demo results saved to: demo_outputs/")
    print("Check the generated GIF files for visualizations!")


if __name__ == "__main__":
    main()
