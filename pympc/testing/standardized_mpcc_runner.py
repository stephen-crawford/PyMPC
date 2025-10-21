"""
Standardized MPCC Test Runner

This module provides a comprehensive test runner for MPCC tests with:
- Configurable test scenarios
- Multiple constraint types
- Perception area testing
- Automatic visualization generation
- Performance analysis

Based on the C++ mpc_planner implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import os
import time
from pathlib import Path
import json
from dataclasses import dataclass, asdict

from .mpcc_test_framework import (
    MPCCTestFramework, TestConfig, RoadConfig, VehicleConfig, 
    MPCConfig, ObstacleConfig, PerceptionConfig, PerceptionShape,
    create_standard_mpcc_test, create_curved_road_test, create_perception_test
)


@dataclass
class TestSuite:
    """Configuration for a test suite."""
    name: str
    tests: List[Dict[str, Any]]
    output_dir: str = "mpcc_test_suite_outputs"
    generate_comparison: bool = True


class StandardizedMPCCRunner:
    """Standardized MPCC test runner with comprehensive testing capabilities."""
    
    def __init__(self, output_dir: str = "mpcc_test_outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Test results storage
        self.results = {}
        self.comparison_data = {}
        
        # Performance metrics
        self.performance_metrics = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'total_duration': 0.0,
            'average_duration': 0.0
        }
    
    def run_standard_test_suite(self) -> Dict[str, Any]:
        """Run the standard MPCC test suite."""
        print("="*80)
        print("STANDARDIZED MPCC TEST SUITE")
        print("="*80)
        
        suite_results = {}
        
        # Test 1: Basic MPCC with curved road
        print("\n1. Basic MPCC Test (Curved Road)")
        test1 = create_standard_mpcc_test(
            test_name="basic_mpcc",
            road=RoadConfig(road_type="curved", curvature_intensity=1.0),
            obstacles=ObstacleConfig(num_obstacles=3, intersection_probability=0.6),
            perception=PerceptionConfig(enabled=False)
        )
        result1 = test1.run_test()
        suite_results['basic_mpcc'] = result1
        
        # Test 2: MPCC with perception limitations
        print("\n2. MPCC with Perception Limitations")
        test2 = create_perception_test(
            test_name="perception_mpcc",
            perception=PerceptionConfig(
                shape=PerceptionShape.CONE,
                distance=20.0,
                angle=np.pi/3,
                enabled=True
            ),
            obstacles=ObstacleConfig(num_obstacles=4, intersection_probability=0.8)
        )
        result2 = test2.run_test()
        suite_results['perception_mpcc'] = result2
        
        # Test 3: MPCC with multiple obstacle types
        print("\n3. MPCC with Multiple Obstacle Types")
        test3 = create_curved_road_test(
            test_name="multi_obstacle_mpcc",
            obstacles=ObstacleConfig(
                num_obstacles=5,
                radius_range=(0.5, 2.0),
                velocity_range=(1.0, 10.0),
                trajectory_type="random_walk",
                intersection_probability=0.9
            ),
            perception=PerceptionConfig(
                shape=PerceptionShape.RECTANGLE,
                distance=25.0,
                width=15.0,
                height=20.0,
                enabled=True
            )
        )
        result3 = test3.run_test()
        suite_results['multi_obstacle_mpcc'] = result3
        
        # Test 4: MPCC with different perception shapes
        print("\n4. MPCC with Different Perception Shapes")
        perception_shapes = [
            (PerceptionShape.CIRCLE, "circle_mpcc"),
            (PerceptionShape.CONE, "cone_mpcc"),
            (PerceptionShape.RECTANGLE, "rectangle_mpcc")
        ]
        
        for shape, test_name in perception_shapes:
            test = MPCCTestFramework(TestConfig(
                test_name=test_name,
                road=RoadConfig(road_type="s_curve", curvature_intensity=1.2),
                obstacles=ObstacleConfig(num_obstacles=3, intersection_probability=0.7),
                perception=PerceptionConfig(
                    shape=shape,
                    distance=18.0,
                    angle=np.pi/4,
                    enabled=True
                )
            ))
            result = test.run_test()
            suite_results[test_name] = result
        
        # Generate comparison analysis
        self._generate_comparison_analysis(suite_results)
        
        return suite_results
    
    def run_constraint_comparison_suite(self) -> Dict[str, Any]:
        """Run tests comparing different constraint types."""
        print("\n" + "="*80)
        print("CONSTRAINT COMPARISON TEST SUITE")
        print("="*80)
        
        constraint_results = {}
        
        # Test different constraint configurations
        constraint_configs = [
            {
                'name': 'ellipsoid_constraints',
                'constraint_type': 'ellipsoid',
                'description': 'Ellipsoid constraint formulation'
            },
            {
                'name': 'gaussian_constraints', 
                'constraint_type': 'gaussian',
                'description': 'Gaussian constraint formulation'
            },
            {
                'name': 'linearized_constraints',
                'constraint_type': 'linearized', 
                'description': 'Linearized constraint formulation'
            },
            {
                'name': 'scenario_constraints',
                'constraint_type': 'scenario',
                'description': 'Scenario-based constraint formulation'
            }
        ]
        
        for config in constraint_configs:
            print(f"\nTesting {config['description']}")
            
            # Create test with specific constraint type
            test = MPCCTestFramework(TestConfig(
                test_name=config['name'],
                road=RoadConfig(road_type="curved", curvature_intensity=1.0),
                obstacles=ObstacleConfig(num_obstacles=3, intersection_probability=0.7),
                perception=PerceptionConfig(
                    shape=PerceptionShape.CONE,
                    distance=20.0,
                    angle=np.pi/3,
                    enabled=True
                )
            ))
            
            # Add constraint type to test
            test.constraint_type = config['constraint_type']
            
            result = test.run_test()
            constraint_results[config['name']] = result
        
        return constraint_results
    
    def run_perception_analysis_suite(self) -> Dict[str, Any]:
        """Run comprehensive perception analysis tests."""
        print("\n" + "="*80)
        print("PERCEPTION ANALYSIS TEST SUITE")
        print("="*80)
        
        perception_results = {}
        
        # Test different perception shapes
        perception_configs = [
            {
                'shape': PerceptionShape.CIRCLE,
                'distance': 15.0,
                'name': 'circle_15m'
            },
            {
                'shape': PerceptionShape.CIRCLE,
                'distance': 25.0,
                'name': 'circle_25m'
            },
            {
                'shape': PerceptionShape.CONE,
                'distance': 20.0,
                'angle': np.pi/6,
                'name': 'cone_30deg'
            },
            {
                'shape': PerceptionShape.CONE,
                'distance': 20.0,
                'angle': np.pi/3,
                'name': 'cone_60deg'
            },
            {
                'shape': PerceptionShape.RECTANGLE,
                'distance': 20.0,
                'width': 10.0,
                'height': 20.0,
                'name': 'rectangle_10x20'
            },
            {
                'shape': PerceptionShape.RECTANGLE,
                'distance': 20.0,
                'width': 20.0,
                'height': 20.0,
                'name': 'rectangle_20x20'
            }
        ]
        
        for config in perception_configs:
            print(f"\nTesting perception: {config['name']}")
            
            # Create perception config
            perception_config = PerceptionConfig(
                shape=config['shape'],
                distance=config['distance'],
                enabled=True
            )
            
            # Add shape-specific parameters
            if 'angle' in config:
                perception_config.angle = config['angle']
            if 'width' in config:
                perception_config.width = config['width']
            if 'height' in config:
                perception_config.height = config['height']
            
            test = MPCCTestFramework(TestConfig(
                test_name=f"perception_{config['name']}",
                road=RoadConfig(road_type="curved", curvature_intensity=1.0),
                obstacles=ObstacleConfig(num_obstacles=4, intersection_probability=0.8),
                perception=perception_config
            ))
            
            result = test.run_test()
            perception_results[config['name']] = result
        
        return perception_results
    
    def run_obstacle_density_suite(self) -> Dict[str, Any]:
        """Run tests with different obstacle densities."""
        print("\n" + "="*80)
        print("OBSTACLE DENSITY TEST SUITE")
        print("="*80)
        
        density_results = {}
        
        # Test different obstacle densities
        obstacle_configs = [
            {'num_obstacles': 2, 'name': 'low_density'},
            {'num_obstacles': 4, 'name': 'medium_density'},
            {'num_obstacles': 6, 'name': 'high_density'},
            {'num_obstacles': 8, 'name': 'very_high_density'}
        ]
        
        for config in obstacle_configs:
            print(f"\nTesting obstacle density: {config['name']} ({config['num_obstacles']} obstacles)")
            
            test = MPCCTestFramework(TestConfig(
                test_name=f"density_{config['name']}",
                road=RoadConfig(road_type="curved", curvature_intensity=1.0),
                obstacles=ObstacleConfig(
                    num_obstacles=config['num_obstacles'],
                    intersection_probability=0.8,
                    trajectory_type="random_walk"
                ),
                perception=PerceptionConfig(
                    shape=PerceptionShape.CONE,
                    distance=20.0,
                    angle=np.pi/3,
                    enabled=True
                )
            ))
            
            result = test.run_test()
            density_results[config['name']] = result
        
        return density_results
    
    def run_comprehensive_suite(self) -> Dict[str, Any]:
        """Run the complete comprehensive test suite."""
        print("="*80)
        print("COMPREHENSIVE MPCC TEST SUITE")
        print("="*80)
        print("Running all test categories...")
        
        start_time = time.time()
        
        # Run all test suites
        all_results = {}
        
        # Standard tests
        print("\n" + "="*50)
        print("STANDARD TESTS")
        print("="*50)
        all_results['standard'] = self.run_standard_test_suite()
        
        # Constraint comparison
        print("\n" + "="*50)
        print("CONSTRAINT COMPARISON")
        print("="*50)
        all_results['constraints'] = self.run_constraint_comparison_suite()
        
        # Perception analysis
        print("\n" + "="*50)
        print("PERCEPTION ANALYSIS")
        print("="*50)
        all_results['perception'] = self.run_perception_analysis_suite()
        
        # Obstacle density
        print("\n" + "="*50)
        print("OBSTACLE DENSITY")
        print("="*50)
        all_results['density'] = self.run_obstacle_density_suite()
        
        # Generate comprehensive report
        total_duration = time.time() - start_time
        self._generate_comprehensive_report(all_results, total_duration)
        
        return all_results
    
    def _generate_comparison_analysis(self, results: Dict[str, Any]):
        """Generate comparison analysis between different tests."""
        print("\n" + "="*50)
        print("COMPARISON ANALYSIS")
        print("="*50)
        
        # Performance comparison
        performance_data = []
        for test_name, result in results.items():
            performance_data.append({
                'test_name': test_name,
                'success': result['success'],
                'duration': result['duration'],
                'steps': result['steps_completed'],
                'violations': len(result['constraint_violations'])
            })
        
        # Print performance summary
        print("\nPerformance Summary:")
        print("-" * 40)
        for data in performance_data:
            status = "SUCCESS" if data['success'] else "FAILED"
            print(f"{data['test_name']:20} | {status:7} | {data['duration']:6.2f}s | {data['steps']:3d} steps | {data['violations']:2d} violations")
        
        # Save comparison data
        comparison_file = os.path.join(self.output_dir, "comparison_analysis.json")
        with open(comparison_file, 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        print(f"\nComparison analysis saved to: {comparison_file}")
    
    def _generate_comprehensive_report(self, all_results: Dict[str, Any], total_duration: float):
        """Generate comprehensive test report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        # Calculate overall statistics
        total_tests = 0
        successful_tests = 0
        total_violations = 0
        
        for category, results in all_results.items():
            for test_name, result in results.items():
                total_tests += 1
                if result['success']:
                    successful_tests += 1
                total_violations += len(result['constraint_violations'])
        
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Print summary
        print(f"\nTest Summary:")
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Average Duration: {total_duration/total_tests:.2f}s per test")
        print(f"Total Constraint Violations: {total_violations}")
        
        # Category breakdown
        print(f"\nCategory Breakdown:")
        for category, results in all_results.items():
            category_tests = len(results)
            category_successful = sum(1 for r in results.values() if r['success'])
            category_rate = (category_successful / category_tests) * 100 if category_tests > 0 else 0
            print(f"{category:15} | {category_tests:3d} tests | {category_successful:3d} successful | {category_rate:5.1f}%")
        
        # Save comprehensive report
        report = {
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': total_tests - successful_tests,
                'success_rate': success_rate,
                'total_duration': total_duration,
                'average_duration': total_duration / total_tests,
                'total_violations': total_violations
            },
            'categories': {
                category: {
                    'total_tests': len(results),
                    'successful_tests': sum(1 for r in results.values() if r['success']),
                    'success_rate': (sum(1 for r in results.values() if r['success']) / len(results)) * 100 if len(results) > 0 else 0
                }
                for category, results in all_results.items()
            },
            'detailed_results': all_results
        }
        
        report_file = os.path.join(self.output_dir, "comprehensive_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nComprehensive report saved to: {report_file}")
    
    def run_single_test(self, test_config: TestConfig) -> Dict[str, Any]:
        """Run a single MPCC test with custom configuration."""
        print(f"Running single test: {test_config.test_name}")
        
        test = MPCCTestFramework(test_config)
        result = test.run_test()
        
        # Store result
        self.results[test_config.test_name] = result
        
        return result
    
    def create_custom_test(self, test_name: str, **kwargs) -> TestConfig:
        """Create a custom test configuration."""
        return TestConfig(test_name=test_name, **kwargs)


def main():
    """Main function to run the standardized MPCC test suite."""
    runner = StandardizedMPCCRunner()
    
    # Run comprehensive test suite
    results = runner.run_comprehensive_suite()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)
    print(f"Results saved to: {runner.output_dir}")


if __name__ == "__main__":
    main()
