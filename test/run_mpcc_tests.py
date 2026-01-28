#!/usr/bin/env python3
"""
MPCC Test Suite Runner

Main entry point for running all MPCC library tests.
Provides options for running different test categories.

Reference: https://github.com/tud-amr/mpc_planner

Usage:
    python run_mpcc_tests.py                    # Run all tests
    python run_mpcc_tests.py --unit             # Run unit tests only
    python run_mpcc_tests.py --integration      # Run integration tests
    python run_mpcc_tests.py --quick            # Run quick smoke tests
    python run_mpcc_tests.py --performance      # Run performance benchmarks
    python run_mpcc_tests.py --report           # Generate detailed report
"""

import sys
import os
import argparse
import subprocess
import time
from datetime import datetime
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_pytest(test_paths: List[str], args: Optional[List[str]] = None) -> int:
    """Run pytest on specified test paths."""
    import pytest

    cmd = test_paths + (args or [])
    return pytest.main(cmd)


def run_unit_tests(verbose: bool = False) -> int:
    """Run unit tests."""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)

    test_dir = os.path.join(os.path.dirname(__file__), 'unit')
    args = ['-v'] if verbose else []

    return run_pytest([test_dir], args)


def run_integration_tests(verbose: bool = False) -> int:
    """Run integration tests."""
    print("\n" + "="*60)
    print("RUNNING INTEGRATION TESTS")
    print("="*60)

    test_dir = os.path.join(os.path.dirname(__file__), 'integration')
    args = ['-v'] if verbose else []

    return run_pytest([test_dir], args)


def run_comprehensive_suite(tags: Optional[List[str]] = None) -> int:
    """Run the comprehensive MPCC test suite."""
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE MPCC TEST SUITE")
    print("="*60)

    from test.mpcc_comprehensive_test_suite import MPCCTestSuite

    suite = MPCCTestSuite()
    result = suite.run_suite(tags=tags)

    return 0 if result.failed == 0 else 1


def run_quick_tests() -> int:
    """Run quick smoke tests."""
    print("\n" + "="*60)
    print("RUNNING QUICK SMOKE TESTS")
    print("="*60)

    from test.mpcc_comprehensive_test_suite import run_quick_tests as quick
    result = quick()

    return 0 if result.failed == 0 else 1


def run_performance_tests() -> int:
    """Run performance benchmark tests."""
    print("\n" + "="*60)
    print("RUNNING PERFORMANCE BENCHMARKS")
    print("="*60)

    from test.mpcc_comprehensive_test_suite import MPCCTestSuite

    suite = MPCCTestSuite()
    result = suite.run_suite(tags=["performance", "benchmark"])

    return 0 if result.failed == 0 else 1


def generate_report(output_dir: str = "test_results") -> None:
    """Generate comprehensive test report."""
    print("\n" + "="*60)
    print("GENERATING TEST REPORT")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    report_file = os.path.join(
        output_dir,
        f"mpcc_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )

    with open(report_file, 'w') as f:
        f.write("# MPCC Library Test Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        f.write("## Reference Implementation\n\n")
        f.write("- C++ mpc_planner: https://github.com/tud-amr/mpc_planner\n\n")

        f.write("## Test Categories\n\n")
        f.write("### 1. Core MPCC Functionality\n")
        f.write("- Path contouring (minimize lateral/lag error)\n")
        f.write("- Reference velocity tracking\n")
        f.write("- Terminal constraints\n\n")

        f.write("### 2. Constraint Types\n")
        f.write("- Safe Horizon (SH-MPC) - scenario-based\n")
        f.write("- Gaussian (CC-MPC) - chance constraints\n")
        f.write("- Linearized - halfspace constraints\n")
        f.write("- Ellipsoid - smooth convex constraints\n")
        f.write("- Contouring - road boundary constraints\n\n")

        f.write("### 3. Vehicle Dynamics\n")
        f.write("- SecondOrderUnicycleModel\n")
        f.write("- ContouringSecondOrderUnicycleModel\n")
        f.write("- SecondOrderBicycleModel\n")
        f.write("- PointMassModel\n\n")

        f.write("### 4. Performance Targets\n")
        f.write("- Control frequency: 20-30 Hz\n")
        f.write("- Simple solve: <50ms\n")
        f.write("- With obstacles: <100ms\n\n")

        f.write("## Running Tests\n\n")
        f.write("```bash\n")
        f.write("# Run all tests\n")
        f.write("python test/run_mpcc_tests.py --all\n\n")
        f.write("# Run quick smoke tests\n")
        f.write("python test/run_mpcc_tests.py --quick\n\n")
        f.write("# Run specific category\n")
        f.write("python test/run_mpcc_tests.py --unit\n")
        f.write("python test/run_mpcc_tests.py --integration\n")
        f.write("```\n")

    print(f"\nReport generated: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="MPCC Library Test Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_mpcc_tests.py --all              Run all tests
    python run_mpcc_tests.py --unit             Run unit tests only
    python run_mpcc_tests.py --integration      Run integration tests
    python run_mpcc_tests.py --quick            Run quick smoke tests
    python run_mpcc_tests.py --performance      Run performance benchmarks
    python run_mpcc_tests.py --suite            Run comprehensive MPCC suite
    python run_mpcc_tests.py --report           Generate test report

Reference: https://github.com/tud-amr/mpc_planner
        """
    )

    parser.add_argument('--all', action='store_true',
                        help='Run all tests')
    parser.add_argument('--unit', action='store_true',
                        help='Run unit tests only')
    parser.add_argument('--integration', action='store_true',
                        help='Run integration tests')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick smoke tests')
    parser.add_argument('--performance', action='store_true',
                        help='Run performance benchmarks')
    parser.add_argument('--suite', action='store_true',
                        help='Run comprehensive MPCC test suite')
    parser.add_argument('--report', action='store_true',
                        help='Generate test report')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--tags', nargs='+',
                        help='Run tests with specific tags')

    args = parser.parse_args()

    # Default to --all if nothing specified
    if not any([args.all, args.unit, args.integration, args.quick,
                args.performance, args.suite, args.report]):
        args.all = True

    start_time = time.time()
    exit_code = 0

    print("\n" + "="*70)
    print("MPCC LIBRARY TEST SUITE")
    print("Reference: https://github.com/tud-amr/mpc_planner")
    print("="*70)

    try:
        if args.report:
            generate_report()

        if args.quick:
            exit_code = max(exit_code, run_quick_tests())

        if args.unit or args.all:
            exit_code = max(exit_code, run_unit_tests(args.verbose))

        if args.integration or args.all:
            exit_code = max(exit_code, run_integration_tests(args.verbose))

        if args.performance:
            exit_code = max(exit_code, run_performance_tests())

        if args.suite or args.all:
            exit_code = max(exit_code, run_comprehensive_suite(args.tags))

    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user")
        exit_code = 130

    except Exception as e:
        print(f"\n\nError during test run: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Exit code: {exit_code}")
    print(f"{'='*70}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
