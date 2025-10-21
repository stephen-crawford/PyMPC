"""
Unified Test Runner for All Constraint Types

This module provides a unified interface for running all constraint tests
with configurable perception areas and obstacle memory.
"""

import numpy as np
from typing import Dict, List, Any
from .linearized_constraint_test import LinearizedConstraintTest
from .ellipsoid_constraint_test import EllipsoidConstraintTest
from .gaussian_constraint_test import GaussianConstraintTest
from .scenario_constraint_test import ScenarioConstraintTest


class UnifiedTestRunner:
    """Unified test runner for all constraint types."""
    
    def __init__(self):
        """Initialize the unified test runner."""
        self.test_results = {}
    
    def run_all_tests(self, perception_configs: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run all constraint tests with specified perception configurations.
        
        Args:
            perception_configs: Dictionary mapping constraint types to perception configs
            
        Returns:
            Dictionary containing results for all tests
        """
        if perception_configs is None:
            perception_configs = self._get_default_perception_configs()
        
        print("="*80)
        print("UNIFIED CONSTRAINT TESTING FRAMEWORK")
        print("="*80)
        print("Running all constraint tests with configurable perception areas...")
        print()
        
        # Run each constraint test
        constraint_types = ['linearized', 'ellipsoid', 'gaussian', 'scenario']
        
        for constraint_type in constraint_types:
            print(f"Running {constraint_type.title()} Constraints Test...")
            print("-" * 50)
            
            try:
                if constraint_type == 'linearized':
                    test = LinearizedConstraintTest(perception_configs[constraint_type])
                elif constraint_type == 'ellipsoid':
                    test = EllipsoidConstraintTest(perception_configs[constraint_type])
                elif constraint_type == 'gaussian':
                    test = GaussianConstraintTest(perception_configs[constraint_type])
                elif constraint_type == 'scenario':
                    test = ScenarioConstraintTest(perception_configs[constraint_type])
                
                result = test.run_test()
                self.test_results[constraint_type] = result
                
                print(f"✅ {constraint_type.title()} test completed")
                print()
                
            except Exception as e:
                print(f"❌ {constraint_type.title()} test failed: {str(e)}")
                self.test_results[constraint_type] = {
                    'success': False,
                    'error': str(e)
                }
                print()
        
        # Print summary
        self._print_summary()
        
        return self.test_results
    
    def run_single_test(self, constraint_type: str, perception_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single constraint test with specified perception configuration.
        
        Args:
            constraint_type: Type of constraint to test
            perception_config: Perception area configuration
            
        Returns:
            Test result dictionary
        """
        print(f"Running {constraint_type.title()} Constraints Test...")
        print(f"Perception Area: {perception_config['type']}")
        print(f"Memory Duration: {perception_config['memory_duration']}s")
        print()
        
        try:
            if constraint_type == 'linearized':
                test = LinearizedConstraintTest(perception_config)
            elif constraint_type == 'ellipsoid':
                test = EllipsoidConstraintTest(perception_config)
            elif constraint_type == 'gaussian':
                test = GaussianConstraintTest(perception_config)
            elif constraint_type == 'scenario':
                test = ScenarioConstraintTest(perception_config)
            else:
                raise ValueError(f"Unknown constraint type: {constraint_type}")
            
            result = test.run_test()
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_perception_area_comparison(self, constraint_type: str) -> Dict[str, Any]:
        """
        Run a constraint test with different perception area types for comparison.
        
        Args:
            constraint_type: Type of constraint to test
            
        Returns:
            Dictionary containing results for different perception areas
        """
        perception_types = ['radius', 'forward_cone', 'bidirectional_cone', 'rectangle']
        results = {}
        
        print(f"Running {constraint_type.title()} Constraints Test with Different Perception Areas...")
        print("="*70)
        
        for perception_type in perception_types:
            print(f"\nTesting with {perception_type} perception area...")
            print("-" * 40)
            
            # Configure perception area
            perception_config = self._get_perception_config(perception_type)
            
            # Run test
            result = self.run_single_test(constraint_type, perception_config)
            results[perception_type] = result
            
            if result['success']:
                print(f"✅ {perception_type} perception area: SUCCESS")
            else:
                print(f"❌ {perception_type} perception area: FAILED - {result.get('error', 'Unknown error')}")
        
        return results
    
    def _get_default_perception_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default perception configurations for all constraint types."""
        return {
            'linearized': {
                'type': 'forward_cone',
                'radius': 20.0,
                'forward_angle': np.pi/3,
                'backward_angle': np.pi/6,
                'forward_distance': 25.0,
                'backward_distance': 10.0,
                'width': 15.0,
                'height': 20.0,
                'memory_duration': 3.0
            },
            'ellipsoid': {
                'type': 'bidirectional_cone',
                'radius': 20.0,
                'forward_angle': np.pi/3,
                'backward_angle': np.pi/6,
                'forward_distance': 25.0,
                'backward_distance': 10.0,
                'width': 15.0,
                'height': 20.0,
                'memory_duration': 4.0
            },
            'gaussian': {
                'type': 'rectangle',
                'radius': 20.0,
                'forward_angle': np.pi/3,
                'backward_angle': np.pi/6,
                'forward_distance': 25.0,
                'backward_distance': 10.0,
                'width': 15.0,
                'height': 20.0,
                'memory_duration': 5.0
            },
            'scenario': {
                'type': 'radius',
                'radius': 20.0,
                'forward_angle': np.pi/3,
                'backward_angle': np.pi/6,
                'forward_distance': 25.0,
                'backward_distance': 10.0,
                'width': 15.0,
                'height': 20.0,
                'memory_duration': 6.0
            }
        }
    
    def _get_perception_config(self, perception_type: str) -> Dict[str, Any]:
        """Get perception configuration for a specific perception type."""
        base_config = {
            'radius': 20.0,
            'forward_angle': np.pi/3,
            'backward_angle': np.pi/6,
            'forward_distance': 25.0,
            'backward_distance': 10.0,
            'width': 15.0,
            'height': 20.0,
            'memory_duration': 4.0
        }
        
        base_config['type'] = perception_type
        return base_config
    
    def _print_summary(self):
        """Print summary of all test results."""
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)
        
        success_count = 0
        total_count = len(self.test_results)
        
        for constraint_type, result in self.test_results.items():
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            print(f"{constraint_type.title():<12}: {status}")
            
            if result['success']:
                success_count += 1
                if 'trajectory' in result:
                    print(f"  Trajectory Length: {len(result['trajectory'])} steps")
                if 'road_data' in result:
                    print(f"  Road Length: 120.0m")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
        
        print(f"\nOverall Success Rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")
        
        if success_count == total_count:
            print("\n🎉 All tests completed successfully!")
        else:
            print(f"\n⚠️  {total_count - success_count} test(s) failed")
        
        print("\nCheck the individual output directories for animations:")
        for constraint_type in self.test_results.keys():
            print(f"  - {constraint_type}_outputs/{constraint_type}_constraints_test.gif")


def run_unified_tests():
    """Run all unified constraint tests."""
    runner = UnifiedTestRunner()
    results = runner.run_all_tests()
    return results


def run_perception_comparison(constraint_type: str = 'scenario'):
    """Run perception area comparison for a specific constraint type."""
    runner = UnifiedTestRunner()
    results = runner.run_perception_area_comparison(constraint_type)
    return results


if __name__ == "__main__":
    # Run all tests with default configurations
    run_unified_tests()