#!/usr/bin/env python3
"""
End-to-End Test Runner for MPC Framework

This script runs comprehensive end-to-end tests that demonstrate
a car traversing a road while avoiding obstacles using various constraint types.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .end_to_end_tests import EndToEndTestFramework
from .simple_road_test import SimpleRoadTest
from .constraint_demo_tests import ConstraintDemoTests
from pympc.utils.logger import LOG_INFO, LOG_WARN, LOG_DEBUG


class ComprehensiveTestRunner:
    """Comprehensive test runner for MPC framework."""
    
    def __init__(self, output_dir: str = "test_outputs"):
        """Initialize the test runner."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Test results storage
        self.all_results = {}
        
    def run_simple_tests(self) -> Dict[str, Any]:
        """Run simple road tests."""
        LOG_INFO("="*60)
        LOG_INFO("RUNNING SIMPLE ROAD TESTS")
        LOG_INFO("="*60)
        
        try:
            simple_test = SimpleRoadTest()
            result = simple_test.run_and_visualize()
            
            self.all_results['simple_road'] = {
                'success': result['success'],
                'error': result.get('error', None),
                'trajectory_length': len(result.get('trajectory', [])),
                'animation_path': result.get('animation_path', None)
            }
            
            return result
            
        except Exception as e:
            LOG_WARN(f"Simple road test failed: {e}")
            self.all_results['simple_road'] = {
                'success': False,
                'error': str(e),
                'trajectory_length': 0,
                'animation_path': None
            }
            return {'success': False, 'error': str(e)}
    
    def run_constraint_demo_tests(self) -> Dict[str, Any]:
        """Run constraint demonstration tests."""
        LOG_INFO("="*60)
        LOG_INFO("RUNNING CONSTRAINT DEMONSTRATION TESTS")
        LOG_INFO("="*60)
        
        try:
            constraint_tests = ConstraintDemoTests()
            results = constraint_tests.run_all_constraint_tests()
            
            # Process results
            demo_results = {}
            for test_name, result in results.items():
                demo_results[test_name] = {
                    'success': result.success,
                    'error': result.error_message,
                    'trajectory_length': len(result.trajectory),
                    'constraint_type': result.constraint_type
                }
            
            self.all_results['constraint_demos'] = demo_results
            
            return demo_results
            
        except Exception as e:
            LOG_WARN(f"Constraint demo tests failed: {e}")
            self.all_results['constraint_demos'] = {'error': str(e)}
            return {'error': str(e)}
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive end-to-end tests."""
        LOG_INFO("="*60)
        LOG_INFO("RUNNING COMPREHENSIVE END-TO-END TESTS")
        LOG_INFO("="*60)
        
        try:
            comprehensive_tests = EndToEndTestFramework("comprehensive_mpc_tests")
            results = comprehensive_tests.run_all_tests()
            
            # Process results
            comp_results = {}
            for test_name, result in results.items():
                comp_results[test_name] = {
                    'success': result.success,
                    'error': result.error_message,
                    'trajectory_length': len(result.trajectory),
                    'solve_times': result.solve_times,
                    'constraint_violations': result.constraint_violations
                }
            
            self.all_results['comprehensive'] = comp_results
            
            return comp_results
            
        except Exception as e:
            LOG_WARN(f"Comprehensive tests failed: {e}")
            self.all_results['comprehensive'] = {'error': str(e)}
            return {'error': str(e)}
    
    def run_all_tests(self, test_types: list = None) -> Dict[str, Any]:
        """Run all tests."""
        if test_types is None:
            test_types = ['simple', 'constraint_demos', 'comprehensive']
        
        LOG_INFO("Starting Comprehensive MPC Test Suite")
        LOG_INFO(f"Test types to run: {test_types}")
        
        start_time = time.time()
        
        # Run tests based on selection
        if 'simple' in test_types:
            self.run_simple_tests()
        
        if 'constraint_demos' in test_types:
            self.run_constraint_demo_tests()
        
        if 'comprehensive' in test_types:
            self.run_comprehensive_tests()
        
        total_time = time.time() - start_time
        
        # Generate final report
        self.generate_final_report(total_time)
        
        return self.all_results
    
    def generate_final_report(self, total_time: float) -> str:
        """Generate final comprehensive test report."""
        report_path = os.path.join(self.output_dir, "FINAL_TEST_REPORT.md")
        
        with open(report_path, 'w') as f:
            f.write("# Comprehensive MPC Test Report\n\n")
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Test Time: {total_time:.2f} seconds\n\n")
            
            # Overall summary
            total_tests = 0
            passed_tests = 0
            
            for category, results in self.all_results.items():
                if isinstance(results, dict) and 'error' not in results:
                    if category == 'constraint_demos':
                        for test_name, result in results.items():
                            total_tests += 1
                            if result['success']:
                                passed_tests += 1
                    else:
                        total_tests += 1
                        if results['success']:
                            passed_tests += 1
            
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            f.write("## Overall Test Summary\n\n")
            f.write(f"- Total Tests: {total_tests}\n")
            f.write(f"- Passed: {passed_tests}\n")
            f.write(f"- Failed: {total_tests - passed_tests}\n")
            f.write(f"- Success Rate: {success_rate:.1f}%\n")
            f.write(f"- Total Time: {total_time:.2f}s\n\n")
            
            # Individual test results
            f.write("## Individual Test Results\n\n")
            
            for category, results in self.all_results.items():
                f.write(f"### {category.replace('_', ' ').title()} Tests\n\n")
                
                if isinstance(results, dict) and 'error' not in results:
                    if category == 'constraint_demos':
                        for test_name, result in results.items():
                            status = "✅ PASSED" if result['success'] else "❌ FAILED"
                            f.write(f"- **{test_name}**: {status}\n")
                            if not result['success'] and result['error']:
                                f.write(f"  - Error: {result['error']}\n")
                    else:
                        status = "✅ PASSED" if results['success'] else "❌ FAILED"
                        f.write(f"- **{category}**: {status}\n")
                        if not results['success'] and results['error']:
                            f.write(f"  - Error: {results['error']}\n")
                else:
                    f.write(f"- **{category}**: ❌ ERROR\n")
                    if 'error' in results:
                        f.write(f"  - Error: {results['error']}\n")
                
                f.write("\n")
            
            # Test descriptions
            f.write("## Test Descriptions\n\n")
            f.write("### Simple Road Test\n")
            f.write("- Tests basic MPC functionality with contouring constraints\n")
            f.write("- Demonstrates vehicle following a curved road\n")
            f.write("- Creates animated visualization of the test\n\n")
            
            f.write("### Constraint Demonstration Tests\n")
            f.write("- Tests individual constraint types:\n")
            f.write("  - Contouring constraints only\n")
            f.write("  - Scenario constraints for obstacle avoidance\n")
            f.write("  - Linearized constraints\n")
            f.write("  - Ellipsoid constraints\n")
            f.write("  - Gaussian constraints\n")
            f.write("  - Decomposition constraints\n")
            f.write("  - All constraints combined\n")
            f.write("- Each test creates an animated visualization\n\n")
            
            f.write("### Comprehensive End-to-End Tests\n")
            f.write("- Tests complete MPC system with multiple constraint types\n")
            f.write("- Demonstrates real-world scenarios with dynamic obstacles\n")
            f.write("- Includes performance metrics and constraint violation analysis\n")
            f.write("- Creates comprehensive visualizations\n\n")
            
            # Output files
            f.write("## Output Files\n\n")
            f.write(f"- Test Report: {report_path}\n")
            f.write(f"- Output Directory: {self.output_dir}\n")
            f.write("- Animations: `*.gif` files in output directory\n")
            f.write("- Logs: Check console output for detailed logs\n\n")
        
        LOG_INFO(f"Final test report saved to {report_path}")
        return report_path
    
    def print_summary(self):
        """Print test summary to console."""
        print("\n" + "="*80)
        print("COMPREHENSIVE MPC TEST SUITE SUMMARY")
        print("="*80)
        
        total_tests = 0
        passed_tests = 0
        
        for category, results in self.all_results.items():
            print(f"\n{category.upper().replace('_', ' ')} TESTS:")
            print("-" * 40)
            
            if isinstance(results, dict) and 'error' not in results:
                if category == 'constraint_demos':
                    for test_name, result in results.items():
                        total_tests += 1
                        status = "✅ PASSED" if result['success'] else "❌ FAILED"
                        print(f"  {test_name:<20} {status}")
                        if not result['success'] and result['error']:
                            print(f"    Error: {result['error']}")
                else:
                    total_tests += 1
                    status = "✅ PASSED" if results['success'] else "❌ FAILED"
                    print(f"  {category:<20} {status}")
                    if not results['success'] and results['error']:
                        print(f"    Error: {results['error']}")
            else:
                print(f"  {category:<20} ❌ ERROR")
                if 'error' in results:
                    print(f"    Error: {results['error']}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "="*80)
        print(f"TOTAL TESTS: {total_tests}")
        print(f"PASSED: {passed_tests}")
        print(f"FAILED: {total_tests - passed_tests}")
        print(f"SUCCESS RATE: {success_rate:.1f}%")
        print("="*80)


def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description='Run MPC End-to-End Tests')
    parser.add_argument('--test-types', nargs='+', 
                       choices=['simple', 'constraint_demos', 'comprehensive'],
                       default=['simple', 'constraint_demos', 'comprehensive'],
                       help='Types of tests to run')
    parser.add_argument('--output-dir', default='test_outputs',
                       help='Output directory for test results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        LOG_DEBUG("Verbose logging enabled")
    
    # Create test runner
    test_runner = ComprehensiveTestRunner(args.output_dir)
    
    # Run tests
    results = test_runner.run_all_tests(args.test_types)
    
    # Print summary
    test_runner.print_summary()
    
    # Return success/failure
    total_tests = 0
    passed_tests = 0
    
    for category, results in test_runner.all_results.items():
        if isinstance(results, dict) and 'error' not in results:
            if category == 'constraint_demos':
                for test_name, result in results.items():
                    total_tests += 1
                    if result['success']:
                        passed_tests += 1
            else:
                total_tests += 1
                if results['success']:
                    passed_tests += 1
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    if success_rate >= 80:
        print(f"\n🎉 Test suite completed successfully! ({success_rate:.1f}% pass rate)")
        return 0
    else:
        print(f"\n⚠️  Test suite completed with issues ({success_rate:.1f}% pass rate)")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
