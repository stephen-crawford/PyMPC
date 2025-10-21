"""
Comprehensive MPCC Test Runner

This module provides a comprehensive test runner that integrates all MPCC testing
capabilities with the standardized framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import os
import time
from pathlib import Path
import json
from dataclasses import asdict

from .mpcc_test_framework import (
    MPCCTestFramework, TestConfig, RoadConfig, VehicleConfig, 
    MPCConfig, ObstacleConfig, PerceptionConfig, PerceptionShape,
    create_standard_mpcc_test, create_curved_road_test, create_perception_test
)
from .enhanced_mpcc_framework import (
    EnhancedMPCCTestFramework, create_enhanced_mpcc_test, create_curved_road_enhanced_test
)
from .standardized_mpcc_runner import StandardizedMPCCRunner


class ComprehensiveMPCCRunner:
    """Comprehensive MPCC test runner with all testing capabilities."""
    
    def __init__(self, output_dir: str = "comprehensive_mpcc_outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize sub-runners
        self.standard_runner = StandardizedMPCCRunner(output_dir)
        
        # Results storage
        self.all_results = {}
        self.performance_metrics = {}
        
        # Test categories
        self.test_categories = {
            'basic': 'Basic MPCC functionality',
            'perception': 'Perception area testing',
            'constraints': 'Constraint type comparison',
            'obstacles': 'Obstacle density testing',
            'enhanced': 'Enhanced C++ reference implementation',
            'comprehensive': 'Full comprehensive suite'
        }
    
    def run_basic_mpcc_tests(self) -> Dict[str, Any]:
        """Run basic MPCC tests."""
        print("="*80)
        print("BASIC MPCC TESTS")
        print("="*80)
        
        results = {}
        
        # Test 1: Standard MPCC
        print("\n1. Standard MPCC Test")
        test1 = create_standard_mpcc_test(
            test_name="basic_standard_mpcc",
            road=RoadConfig(road_type="curved", curvature_intensity=1.0),
            obstacles=ObstacleConfig(num_obstacles=3, intersection_probability=0.6),
            perception=PerceptionConfig(enabled=False),
            output_dir=self.output_dir
        )
        result1 = test1.run_test()
        results['standard_mpcc'] = result1
        
        # Test 2: Enhanced MPCC
        print("\n2. Enhanced MPCC Test (C++ Reference)")
        test2 = create_enhanced_mpcc_test(
            test_name="basic_enhanced_mpcc",
            road=RoadConfig(road_type="curved", curvature_intensity=1.0),
            obstacles=ObstacleConfig(num_obstacles=3, intersection_probability=0.6),
            perception=PerceptionConfig(enabled=False),
            output_dir=self.output_dir
        )
        result2 = test2.run_test()
        results['enhanced_mpcc'] = result2
        
        # Test 3: Curved Road MPCC
        print("\n3. Curved Road MPCC Test")
        test3 = create_curved_road_test(
            test_name="basic_curved_road_mpcc",
            road=RoadConfig(road_type="s_curve", curvature_intensity=1.2),
            obstacles=ObstacleConfig(num_obstacles=4, intersection_probability=0.7),
            perception=PerceptionConfig(enabled=False),
            output_dir=self.output_dir
        )
        result3 = test3.run_test()
        results['curved_road_mpcc'] = result3
        
        return results
    
    def run_perception_tests(self) -> Dict[str, Any]:
        """Run perception area tests."""
        print("\n" + "="*80)
        print("PERCEPTION AREA TESTS")
        print("="*80)
        
        results = {}
        
        # Test different perception shapes
        perception_configs = [
            {
                'shape': PerceptionShape.CIRCLE,
                'distance': 20.0,
                'name': 'circle_20m'
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
                'width': 15.0,
                'height': 20.0,
                'name': 'rectangle_15x20'
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
            
            # Standard MPCC test
            test_standard = MPCCTestFramework(TestConfig(
                test_name=f"perception_standard_{config['name']}",
                road=RoadConfig(road_type="curved", curvature_intensity=1.0),
                obstacles=ObstacleConfig(num_obstacles=4, intersection_probability=0.8),
                perception=perception_config,
                output_dir=self.output_dir
            ))
            result_standard = test_standard.run_test()
            results[f"standard_{config['name']}"] = result_standard
            
            # Enhanced MPCC test
            test_enhanced = EnhancedMPCCTestFramework(TestConfig(
                test_name=f"perception_enhanced_{config['name']}",
                road=RoadConfig(road_type="curved", curvature_intensity=1.0),
                obstacles=ObstacleConfig(num_obstacles=4, intersection_probability=0.8),
                perception=perception_config,
                output_dir=self.output_dir
            ))
            result_enhanced = test_enhanced.run_test()
            results[f"enhanced_{config['name']}"] = result_enhanced
        
        return results
    
    def run_constraint_comparison_tests(self) -> Dict[str, Any]:
        """Run constraint comparison tests."""
        print("\n" + "="*80)
        print("CONSTRAINT COMPARISON TESTS")
        print("="*80)
        
        results = {}
        
        # Test different constraint types
        constraint_configs = [
            {
                'name': 'ellipsoid_constraints',
                'description': 'Ellipsoid constraint formulation'
            },
            {
                'name': 'gaussian_constraints',
                'description': 'Gaussian constraint formulation'
            },
            {
                'name': 'linearized_constraints',
                'description': 'Linearized constraint formulation'
            },
            {
                'name': 'scenario_constraints',
                'description': 'Scenario-based constraint formulation'
            }
        ]
        
        for config in constraint_configs:
            print(f"\nTesting {config['description']}")
            
            # Standard MPCC with specific constraint type
            test = MPCCTestFramework(TestConfig(
                test_name=f"constraint_{config['name']}",
                road=RoadConfig(road_type="curved", curvature_intensity=1.0),
                obstacles=ObstacleConfig(num_obstacles=3, intersection_probability=0.7),
                perception=PerceptionConfig(
                    shape=PerceptionShape.CONE,
                    distance=20.0,
                    angle=np.pi/3,
                    enabled=True
                ),
                output_dir=self.output_dir
            ))
            
            # Add constraint type to test
            test.constraint_type = config['name']
            
            result = test.run_test()
            results[config['name']] = result
        
        return results
    
    def run_obstacle_density_tests(self) -> Dict[str, Any]:
        """Run obstacle density tests."""
        print("\n" + "="*80)
        print("OBSTACLE DENSITY TESTS")
        print("="*80)
        
        results = {}
        
        # Test different obstacle densities
        obstacle_configs = [
            {'num_obstacles': 2, 'name': 'low_density'},
            {'num_obstacles': 4, 'name': 'medium_density'},
            {'num_obstacles': 6, 'name': 'high_density'},
            {'num_obstacles': 8, 'name': 'very_high_density'}
        ]
        
        for config in obstacle_configs:
            print(f"\nTesting obstacle density: {config['name']} ({config['num_obstacles']} obstacles)")
            
            # Standard MPCC test
            test_standard = MPCCTestFramework(TestConfig(
                test_name=f"density_standard_{config['name']}",
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
                ),
                output_dir=self.output_dir
            ))
            result_standard = test_standard.run_test()
            results[f"standard_{config['name']}"] = result_standard
            
            # Enhanced MPCC test
            test_enhanced = EnhancedMPCCTestFramework(TestConfig(
                test_name=f"density_enhanced_{config['name']}",
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
                ),
                output_dir=self.output_dir
            ))
            result_enhanced = test_enhanced.run_test()
            results[f"enhanced_{config['name']}"] = result_enhanced
        
        return results
    
    def run_enhanced_mpcc_tests(self) -> Dict[str, Any]:
        """Run enhanced MPCC tests with C++ reference implementation."""
        print("\n" + "="*80)
        print("ENHANCED MPCC TESTS (C++ REFERENCE)")
        print("="*80)
        
        results = {}
        
        # Test 1: Basic Enhanced MPCC
        print("\n1. Basic Enhanced MPCC")
        test1 = create_enhanced_mpcc_test(
            test_name="enhanced_basic_mpcc",
            road=RoadConfig(road_type="curved", curvature_intensity=1.0),
            obstacles=ObstacleConfig(num_obstacles=3, intersection_probability=0.6),
            perception=PerceptionConfig(enabled=False),
            output_dir=self.output_dir
        )
        result1 = test1.run_test()
        results['enhanced_basic'] = result1
        
        # Test 2: Enhanced MPCC with Perception
        print("\n2. Enhanced MPCC with Perception")
        test2 = EnhancedMPCCTestFramework(TestConfig(
            test_name="enhanced_perception_mpcc",
            road=RoadConfig(road_type="curved", curvature_intensity=1.0),
            obstacles=ObstacleConfig(num_obstacles=4, intersection_probability=0.8),
            perception=PerceptionConfig(
                shape=PerceptionShape.CONE,
                distance=20.0,
                angle=np.pi/3,
                enabled=True
            ),
            output_dir=self.output_dir
        ))
        result2 = test2.run_test()
        results['enhanced_perception'] = result2
        
        # Test 3: Enhanced MPCC with High Obstacle Density
        print("\n3. Enhanced MPCC with High Obstacle Density")
        test3 = EnhancedMPCCTestFramework(TestConfig(
            test_name="enhanced_high_density_mpcc",
            road=RoadConfig(road_type="s_curve", curvature_intensity=1.2),
            obstacles=ObstacleConfig(
                num_obstacles=6,
                intersection_probability=0.9,
                trajectory_type="random_walk"
            ),
            perception=PerceptionConfig(
                shape=PerceptionShape.RECTANGLE,
                distance=25.0,
                width=15.0,
                height=20.0,
                enabled=True
            ),
            output_dir=self.output_dir
        ))
        result3 = test3.run_test()
        results['enhanced_high_density'] = result3
        
        return results
    
    def run_comprehensive_suite(self) -> Dict[str, Any]:
        """Run the complete comprehensive test suite."""
        print("="*80)
        print("COMPREHENSIVE MPCC TEST SUITE")
        print("="*80)
        print("Running all test categories...")
        
        start_time = time.time()
        
        # Run all test categories
        all_results = {}
        
        # Basic tests
        print("\n" + "="*50)
        print("BASIC MPCC TESTS")
        print("="*50)
        all_results['basic'] = self.run_basic_mpcc_tests()
        
        # Perception tests
        print("\n" + "="*50)
        print("PERCEPTION AREA TESTS")
        print("="*50)
        all_results['perception'] = self.run_perception_tests()
        
        # Constraint comparison
        print("\n" + "="*50)
        print("CONSTRAINT COMPARISON TESTS")
        print("="*50)
        all_results['constraints'] = self.run_constraint_comparison_tests()
        
        # Obstacle density
        print("\n" + "="*50)
        print("OBSTACLE DENSITY TESTS")
        print("="*50)
        all_results['obstacles'] = self.run_obstacle_density_tests()
        
        # Enhanced MPCC
        print("\n" + "="*50)
        print("ENHANCED MPCC TESTS")
        print("="*50)
        all_results['enhanced'] = self.run_enhanced_mpcc_tests()
        
        # Generate comprehensive report
        total_duration = time.time() - start_time
        self._generate_comprehensive_report(all_results, total_duration)
        
        return all_results
    
    def _generate_comprehensive_report(self, all_results: Dict[str, Any], total_duration: float):
        """Generate comprehensive test report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        # Calculate overall statistics
        total_tests = 0
        successful_tests = 0
        total_violations = 0
        total_duration_all = 0.0
        
        category_stats = {}
        
        for category, results in all_results.items():
            category_tests = len(results)
            category_successful = sum(1 for r in results.values() if r['success'])
            category_violations = sum(len(r['constraint_violations']) for r in results.values())
            category_duration = sum(r['duration'] for r in results.values())
            
            category_stats[category] = {
                'total_tests': category_tests,
                'successful_tests': category_successful,
                'failed_tests': category_tests - category_successful,
                'success_rate': (category_successful / category_tests) * 100 if category_tests > 0 else 0,
                'total_violations': category_violations,
                'total_duration': category_duration
            }
            
            total_tests += category_tests
            successful_tests += category_successful
            total_violations += category_violations
            total_duration_all += category_duration
        
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Print summary
        print(f"\nOverall Test Summary:")
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Average Duration: {total_duration/total_tests:.2f}s per test")
        print(f"Total Constraint Violations: {total_violations}")
        
        # Category breakdown
        print(f"\nCategory Breakdown:")
        print("-" * 80)
        print(f"{'Category':<15} | {'Tests':<5} | {'Success':<7} | {'Rate':<6} | {'Violations':<10} | {'Duration':<8}")
        print("-" * 80)
        
        for category, stats in category_stats.items():
            print(f"{category:<15} | {stats['total_tests']:<5} | {stats['successful_tests']:<7} | "
                  f"{stats['success_rate']:<5.1f}% | {stats['total_violations']:<10} | {stats['total_duration']:<7.1f}s")
        
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
            'categories': category_stats,
            'detailed_results': all_results
        }
        
        report_file = os.path.join(self.output_dir, "comprehensive_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nComprehensive report saved to: {report_file}")
        
        # Generate performance comparison
        self._generate_performance_comparison(all_results)
    
    def _generate_performance_comparison(self, all_results: Dict[str, Any]):
        """Generate performance comparison between different approaches."""
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)
        
        # Compare standard vs enhanced MPCC
        standard_results = []
        enhanced_results = []
        
        for category, results in all_results.items():
            for test_name, result in results.items():
                if 'enhanced' in test_name:
                    enhanced_results.append(result)
                else:
                    standard_results.append(result)
        
        if standard_results and enhanced_results:
            print(f"\nStandard MPCC vs Enhanced MPCC Comparison:")
            print(f"Standard MPCC - Tests: {len(standard_results)}, "
                  f"Success Rate: {sum(1 for r in standard_results if r['success'])/len(standard_results)*100:.1f}%, "
                  f"Avg Duration: {np.mean([r['duration'] for r in standard_results]):.2f}s")
            print(f"Enhanced MPCC - Tests: {len(enhanced_results)}, "
                  f"Success Rate: {sum(1 for r in enhanced_results if r['success'])/len(enhanced_results)*100:.1f}%, "
                  f"Avg Duration: {np.mean([r['duration'] for r in enhanced_results]):.2f}s")
        
        # Perception analysis
        perception_results = []
        no_perception_results = []
        
        for category, results in all_results.items():
            for test_name, result in results.items():
                if 'perception' in test_name:
                    perception_results.append(result)
                else:
                    no_perception_results.append(result)
        
        if perception_results and no_perception_results:
            print(f"\nPerception vs No Perception Comparison:")
            print(f"No Perception - Tests: {len(no_perception_results)}, "
                  f"Success Rate: {sum(1 for r in no_perception_results if r['success'])/len(no_perception_results)*100:.1f}%")
            print(f"With Perception - Tests: {len(perception_results)}, "
                  f"Success Rate: {sum(1 for r in perception_results if r['success'])/len(perception_results)*100:.1f}%")
    
    def run_single_test(self, test_config: TestConfig, enhanced: bool = False) -> Dict[str, Any]:
        """Run a single MPCC test."""
        if enhanced:
            test = EnhancedMPCCTestFramework(test_config)
        else:
            test = MPCCTestFramework(test_config)
        
        result = test.run_test()
        return result


def main():
    """Main function to run the comprehensive MPCC test suite."""
    runner = ComprehensiveMPCCRunner()
    
    # Run comprehensive test suite
    results = runner.run_comprehensive_suite()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)
    print(f"Results saved to: {runner.output_dir}")
    print("Check the generated GIF files for visualizations!")


if __name__ == "__main__":
    main()
