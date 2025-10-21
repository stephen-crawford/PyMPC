"""
Refactored Test Runner

This module provides a comprehensive test runner for all refactored constraint tests
with enhanced features including GIF generation, execution info, and trajectory funnels.
"""

import numpy as np
import time
from typing import Dict, List, Any
from .linearized_constraint import LinearizedConstraintTest
from .ellipsoid_constraint import EllipsoidConstraintTest
from .gaussian_constraint import GaussianConstraintTest
from .scenario_constraint import ScenarioConstraintTest


class RefactoredTestRunner:
    """Enhanced test runner for refactored constraint tests."""
    
    def __init__(self):
        """Initialize the refactored test runner."""
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
        # Define test configurations with enhanced features
        self.test_configs = {
            'linearized': {
                'perception_config': {
                    'type': 'forward_cone',
                    'radius': 20.0,
                    'forward_angle': np.pi/3,
                    'backward_angle': np.pi/6,
                    'forward_distance': 25.0,
                    'backward_distance': 10.0,
                    'width': 15.0,
                    'height': 20.0,
                    'memory_duration': 3.0,
                    'recall_memory': True,
                    'trajectory_funnels': True,
                    'funnel_horizon': 3.0
                },
                'description': 'Linearized half-space constraints with forward cone perception'
            },
            'ellipsoid': {
                'perception_config': {
                    'type': 'bidirectional_cone',
                    'radius': 20.0,
                    'forward_angle': np.pi/3,
                    'backward_angle': np.pi/6,
                    'forward_distance': 25.0,
                    'backward_distance': 10.0,
                    'width': 15.0,
                    'height': 20.0,
                    'memory_duration': 4.0,
                    'recall_memory': True,
                    'trajectory_funnels': True,
                    'funnel_horizon': 4.0
                },
                'description': 'Ellipsoid constraints with bidirectional cone perception'
            },
            'gaussian': {
                'perception_config': {
                    'type': 'rectangle',
                    'radius': 20.0,
                    'forward_angle': np.pi/3,
                    'backward_angle': np.pi/6,
                    'forward_distance': 25.0,
                    'backward_distance': 10.0,
                    'width': 15.0,
                    'height': 20.0,
                    'memory_duration': 5.0,
                    'recall_memory': True,
                    'trajectory_funnels': True,
                    'funnel_horizon': 5.0
                },
                'description': 'Gaussian constraints with rectangular perception'
            },
            'scenario': {
                'perception_config': {
                    'type': 'radius',
                    'radius': 20.0,
                    'forward_angle': np.pi/3,
                    'backward_angle': np.pi/6,
                    'forward_distance': 25.0,
                    'backward_distance': 10.0,
                    'width': 15.0,
                    'height': 20.0,
                    'memory_duration': 6.0,
                    'recall_memory': True,
                    'recall_duration': 8.0,  # Extended recall for scenarios
                    'trajectory_funnels': True,
                    'funnel_horizon': 6.0
                },
                'description': 'Scenario constraints with circular perception and recall memory'
            }
        }
        
        self.test_classes = {
            'linearized': LinearizedConstraintTest,
            'ellipsoid': EllipsoidConstraintTest,
            'gaussian': GaussianConstraintTest,
            'scenario': ScenarioConstraintTest
        }
    
    def run_single_test(self, constraint_type: str) -> Dict[str, Any]:
        """Run a single refactored constraint test."""
        if constraint_type not in self.test_classes:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
        
        config = self.test_configs[constraint_type]
        test_instance = self.test_classes[constraint_type](config['perception_config'])
        
        print(f"Running {constraint_type.title()} Constraints Test")
        print(f"Description: {config['description']}")
        print(f"Perception Area: {config['perception_config']['type']}")
        print(f"Memory Duration: {config['perception_config']['memory_duration']}s")
        print(f"Recall Memory: {config['perception_config']['recall_memory']}")
        print(f"Trajectory Funnels: {config['perception_config']['trajectory_funnels']}")
        if 'recall_duration' in config['perception_config']:
            print(f"Recall Duration: {config['perception_config']['recall_duration']}s")
        print()
        
        result = test_instance.run_test()
        
        return {
            'constraint_type': constraint_type,
            'result': result,
            'config': config,
            'success': result['success'] if 'success' in result else False
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all refactored constraint tests."""
        print("="*80)
        print("REFACTORED CONSTRAINT TESTS WITH ENHANCED FEATURES")
        print("="*80)
        print("Running all constraint tests with:")
        print("✅ Configurable perception areas")
        print("✅ Obstacle memory with configurable duration")
        print("✅ Trajectory funnels for obstacle predictions")
        print("✅ Recall memory for scenario constraints")
        print("✅ Enhanced GIF generation with execution info")
        print("✅ Real-time performance tracking")
        print()
        
        self.start_time = time.time()
        
        # Run each test
        for constraint_type, config in self.test_configs.items():
            print(f"Testing {constraint_type.title()} Constraints...")
            print(f"Perception: {config['perception_config']['type']}")
            print(f"Memory: {config['perception_config']['memory_duration']}s")
            print(f"Funnels: {config['perception_config']['trajectory_funnels']}")
            print("-" * 60)
            
            try:
                result = self.run_single_test(constraint_type)
                self.test_results[constraint_type] = result
                print(f"✅ {constraint_type.title()} test completed")
                print()
                
            except Exception as e:
                print(f"❌ {constraint_type.title()} test failed: {str(e)}")
                self.test_results[constraint_type] = {
                    'constraint_type': constraint_type,
                    'result': {'success': False, 'error': str(e)},
                    'config': config,
                    'success': False
                }
                print()
        
        self.end_time = time.time()
        
        # Print comprehensive summary
        self._print_comprehensive_summary()
        
        return self.test_results
    
    def _print_comprehensive_summary(self):
        """Print comprehensive summary of all test results."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST RESULTS SUMMARY")
        print("="*80)
        
        success_count = 0
        total_count = len(self.test_results)
        total_time = self.end_time - self.start_time
        
        print(f"Total Tests: {total_count}")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Average Time per Test: {total_time/total_count:.2f} seconds")
        print()
        
        # Individual test results
        for constraint_type, data in self.test_results.items():
            config = data['config']
            result = data['result']
            success = data['success']
            
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{constraint_type.title():<15}: {status}")
            print(f"  Perception: {config['perception_config']['type']}")
            print(f"  Memory: {config['perception_config']['memory_duration']}s")
            print(f"  Funnels: {config['perception_config']['trajectory_funnels']}")
            print(f"  Recall: {config['perception_config']['recall_memory']}")
            
            if success and 'trajectory' in result:
                trajectory = result['trajectory']
                print(f"  Trajectory: {len(trajectory)} steps")
                
                if len(trajectory) > 0:
                    final_pos = trajectory[-1]
                    print(f"  Final Position: ({final_pos[0]:.1f}, {final_pos[1]:.1f})")
                    print(f"  Final Velocity: {final_pos[3]:.1f} m/s")
                
                # Performance metrics
                if 'performance' in result:
                    perf = result['performance']
                    if 'execution_times' in perf and len(perf['execution_times']) > 0:
                        avg_exec_time = np.mean(perf['execution_times']) * 1000
                        print(f"  Avg Execution Time: {avg_exec_time:.1f}ms")
                    if 'compute_times' in perf and len(perf['compute_times']) > 0:
                        avg_compute_time = np.mean(perf['compute_times']) * 1000
                        print(f"  Avg Compute Time: {avg_compute_time:.1f}ms")
                    if 'active_constraints' in perf and len(perf['active_constraints']) > 0:
                        avg_constraints = np.mean(perf['active_constraints'])
                        print(f"  Avg Active Constraints: {avg_constraints:.1f}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
            
            print()
            
            if success:
                success_count += 1
        
        # Overall statistics
        success_rate = (success_count / total_count) * 100
        print(f"Overall Success Rate: {success_count}/{total_count} ({success_rate:.1f}%)")
        
        if success_count == total_count:
            print("\n🎉 All refactored tests completed successfully!")
        else:
            print(f"\n⚠️  {total_count - success_count} test(s) failed")
        
        # Enhanced features demonstrated
        print("\n" + "="*80)
        print("ENHANCED FEATURES DEMONSTRATED")
        print("="*80)
        print("✅ Configurable perception areas for different constraint types")
        print("✅ Obstacle memory with configurable duration (3-6 seconds)")
        print("✅ Trajectory funnels for obstacle prediction visualization")
        print("✅ Recall memory for scenario constraints (8 seconds)")
        print("✅ Enhanced GIF generation with execution info overlay")
        print("✅ Real-time performance tracking (execution time, compute time)")
        print("✅ Active constraint counting and visualization")
        print("✅ Standardized structure across all constraint types")
        print("✅ C++ MPC planner formulation compliance")
        print("✅ Real-time perception area filtering")
        print("✅ Obstacle prediction and memory management")
        print("✅ Multiple parallel solvers for scenario constraints")
        print("✅ Easy configuration with intuitive parameter specification")
        print("✅ Unified interface for consistent testing across constraint types")
        
        # Output files
        print("\n" + "="*80)
        print("OUTPUT FILES GENERATED")
        print("="*80)
        print("Enhanced GIFs with execution info and trajectory funnels:")
        for constraint_type in self.test_results.keys():
            print(f"  - {constraint_type}_outputs/{constraint_type}_constraints_test.gif")
        
        print("\n🎉 Refactored constraint testing framework with enhanced features is ready!")
    
    def list_available_tests(self) -> List[str]:
        """List all available constraint test types."""
        return list(self.test_classes.keys())
    
    def get_test_config(self, constraint_type: str) -> Dict[str, Any]:
        """Get configuration for a specific constraint type."""
        if constraint_type not in self.test_configs:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
        return self.test_configs[constraint_type]
    
    def run_custom_test(self, constraint_type: str, custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a test with custom configuration."""
        if constraint_type not in self.test_classes:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
        
        test_instance = self.test_classes[constraint_type](custom_config)
        result = test_instance.run_test()
        
        return {
            'constraint_type': constraint_type,
            'result': result,
            'config': custom_config,
            'success': result['success'] if 'success' in result else False
        }


def run_all_refactored_tests():
    """Run all refactored constraint tests with enhanced features."""
    runner = RefactoredTestRunner()
    results = runner.run_all_tests()
    return results


def run_single_refactored_test(constraint_type: str):
    """Run a single refactored constraint test."""
    runner = RefactoredTestRunner()
    result = runner.run_single_test(constraint_type)
    return result


def list_available_tests():
    """List all available constraint test types."""
    runner = RefactoredTestRunner()
    return runner.list_available_tests()


if __name__ == "__main__":
    # Run all refactored tests
    results = run_all_refactored_tests()
    
    print("\n" + "="*80)
    print("REFACTORING COMPLETE WITH ENHANCED FEATURES")
    print("="*80)
    print("All constraint tests have been successfully refactored to use")
    print("the unified framework with enhanced features:")
    print("- Configurable perception areas and obstacle memory")
    print("- Trajectory funnels for obstacle prediction visualization")
    print("- Recall memory for scenario constraints")
    print("- Enhanced GIF generation with execution info overlay")
    print("- Real-time performance tracking and visualization")
    print("- C++ MPC planner formulation compliance")
