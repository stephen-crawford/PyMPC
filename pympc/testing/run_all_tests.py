"""
Comprehensive test runner for PyMPC.

This script runs all tests in the organized framework:
- Unit tests for individual components
- Integration tests for complete systems
- Constraint tests for all constraint types
- Objective tests for all objective types
- End-to-end MPCC scenario tests
"""

import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pympc.testing.constraint_tests import create_constraint_test_suite
from pympc.testing.mpcc_scenario_test import create_mpcc_scenario_test


class PyMPCTestRunner:
    """
    Comprehensive test runner for PyMPC framework.
    
    This runner organizes and executes all tests in the framework,
    providing detailed reporting and analysis.
    """

    def __init__(self):
        """Initialize the test runner."""
        self.test_suites = []
        self.results = []
        self.start_time = None

    def add_test_suite(self, suite_name: str, suite):
        """Add a test suite to the runner."""
        self.test_suites.append({
            'name': suite_name,
            'suite': suite
        })

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and return comprehensive results."""
        print("PyMPC Comprehensive Test Runner")
        print("=" * 50)
        
        self.start_time = time.time()
        
        # Run each test suite
        for suite_info in self.test_suites:
            print(f"\nRunning {suite_info['name']}...")
            print("-" * 30)
            
            suite_start = time.time()
            suite_results = self._run_suite(suite_info['suite'])
            suite_duration = time.time() - suite_start
            
            self.results.append({
                'suite_name': suite_info['name'],
                'results': suite_results,
                'duration': suite_duration
            })
            
            # Print suite summary
            self._print_suite_summary(suite_info['name'], suite_results, suite_duration)
        
        # Generate comprehensive report
        total_duration = time.time() - self.start_time
        report = self._generate_comprehensive_report(total_duration)
        
        return report

    def _run_suite(self, suite):
        """Run a single test suite."""
        if hasattr(suite, 'run_all_tests'):
            return suite.run_all_tests()
        elif hasattr(suite, 'run_test'):
            return [suite.run_test()]
        else:
            print(f"Warning: Unknown suite type {type(suite)}")
            return []

    def _print_suite_summary(self, suite_name: str, results: List, duration: float):
        """Print summary for a test suite."""
        if not results:
            print(f"{suite_name}: No tests run")
            return
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if hasattr(r, 'success') and r.success)
        failed_tests = total_tests - passed_tests
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"{suite_name} Summary:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {failed_tests}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Duration: {duration:.2f}s")

    def _generate_comprehensive_report(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        # Aggregate results
        total_tests = 0
        total_passed = 0
        total_failed = 0
        
        for suite_result in self.results:
            results = suite_result['results']
            if results:
                total_tests += len(results)
                total_passed += sum(1 for r in results if hasattr(r, 'success') and r.success)
        
        total_failed = total_tests - total_passed
        overall_success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
        
        # Generate report
        report = {
            'overall': {
                'total_tests': total_tests,
                'passed': total_passed,
                'failed': total_failed,
                'success_rate': overall_success_rate,
                'total_duration': total_duration
            },
            'suites': self.results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Print overall summary
        print("\n" + "=" * 50)
        print("OVERALL TEST SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        print(f"Success Rate: {overall_success_rate:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        
        if total_failed == 0:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print(f"\n⚠️  {total_failed} TESTS FAILED")
        
        return report


def create_comprehensive_test_runner() -> PyMPCTestRunner:
    """Create comprehensive test runner with all test suites."""
    runner = PyMPCTestRunner()
    
    # Add constraint test suite
    constraint_suite = create_constraint_test_suite()
    runner.add_test_suite("Constraint Tests", constraint_suite)
    
    # Add MPCC scenario test
    mpcc_test = create_mpcc_scenario_test()
    runner.add_test_suite("MPCC Scenario Test", mpcc_test)
    
    return runner


def main():
    """Main function to run all tests."""
    # Create test runner
    runner = create_comprehensive_test_runner()
    
    # Run all tests
    report = runner.run_all_tests()
    
    # Save report
    report_file = "test_report.json"
    import json
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nTest report saved to: {report_file}")
    
    # Return success/failure
    return report['overall']['failed'] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
