"""
Run Standardized Tests

This script runs all standardized tests with proper visualization and reporting.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from test.framework.standardized_test import TestSuite
from test.integration.standardized_mpcc_test import StandardizedMPCCTest
from utils.standardized_logging import get_test_logger


def run_all_standardized_tests():
    """Run all standardized tests."""
    print("Running All Standardized Tests")
    print("=" * 60)
    
    # Create test suite
    suite = TestSuite("Comprehensive MPCC Test Suite")
    
    # Add all tests
    suite.add_test(StandardizedMPCCTest())
    
    # Run tests
    start_time = time.time()
    results = suite.run_all_tests()
    total_time = time.time() - start_time
    
    # Print detailed results
    print("\n" + "=" * 60)
    print("DETAILED TEST RESULTS")
    print("=" * 60)
    
    for i, result in enumerate(results):
        print(f"\nTest {i+1}: {result.test_name}")
        print(f"  Status: {'PASSED' if result.success else 'FAILED'}")
        print(f"  Duration: {result.duration:.2f}s")
        print(f"  Iterations: {result.iterations}")
        print(f"  Final Error: {result.final_error:.4f}")
        print(f"  Convergence Rate: {result.convergence_rate:.2%}")
        
        if result.error_log:
            print(f"  Errors: {len(result.error_log)}")
            for error in result.error_log[:3]:  # Show first 3 errors
                print(f"    - {error}")
        
        if result.performance_metrics:
            print(f"  Performance Metrics:")
            for key, value in result.performance_metrics.items():
                print(f"    {key}: {value:.4f}")
    
    # Print suite summary
    print("\n" + "=" * 60)
    print("SUITE SUMMARY")
    print("=" * 60)
    print(suite.get_suite_summary())
    
    # Print timing information
    print(f"\nTotal execution time: {total_time:.2f}s")
    
    return results


def run_single_test(test_name: str = "standardized_mpcc_test"):
    """Run a single test by name."""
    print(f"Running Single Test: {test_name}")
    print("=" * 40)
    
    if test_name == "standardized_mpcc_test":
        test = StandardizedMPCCTest()
        result = test.run_test()
        
        print("\nTest Results:")
        print(test.get_test_summary())
        
        return result
    else:
        print(f"Unknown test: {test_name}")
        return None


def run_tests_with_visualization():
    """Run tests with enhanced visualization."""
    print("Running Tests with Enhanced Visualization")
    print("=" * 50)
    
    # This would run tests with specific visualization settings
    # For now, just run the standard tests
    return run_all_standardized_tests()


def run_tests_with_debugging():
    """Run tests with debugging tools enabled."""
    print("Running Tests with Debugging Tools")
    print("=" * 40)
    
    # This would run tests with debugging tools enabled
    # For now, just run the standard tests
    return run_all_standardized_tests()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run standardized MPC tests")
    parser.add_argument("--test", type=str, help="Run specific test")
    parser.add_argument("--visualization", action="store_true", help="Enable enhanced visualization")
    parser.add_argument("--debug", action="store_true", help="Enable debugging tools")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.test:
        run_single_test(args.test)
    elif args.visualization:
        run_tests_with_visualization()
    elif args.debug:
        run_tests_with_debugging()
    elif args.all:
        run_all_standardized_tests()
    else:
        # Default: run all tests
        run_all_standardized_tests()
