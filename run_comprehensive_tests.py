#!/usr/bin/env python3
"""
Run comprehensive test suite for PyMPC framework.

This script executes all the test scenarios as requested by the user.
"""

import sys
import os

# Add pympc to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pympc'))

from pympc.testing.comprehensive_test_suite import ComprehensiveTestSuite

def main():
    """Run comprehensive test suite."""
    print("PyMPC Comprehensive Test Suite")
    print("=" * 50)
    
    # Create test suite
    test_suite = ComprehensiveTestSuite(output_dir="test_results")
    
    # Run all tests
    results = test_suite.run_all_tests()
    
    # Check if any tests failed
    failed_tests = [test_id for test_id, result in results.items() if not result['success']]
    
    if failed_tests:
        print(f"\n⚠️  {len(failed_tests)} tests failed:")
        for test_id in failed_tests:
            print(f"  - {test_id}: {results[test_id].get('error', 'Unknown error')}")
    else:
        print("\n✅ All tests passed!")
    
    return 0 if not failed_tests else 1

if __name__ == "__main__":
    sys.exit(main())
