#!/usr/bin/env python3
"""
Comprehensive Test Suite for Model Predictive Contouring Control (MPCC) Library

This test suite verifies the functionality of the PyMPC library against the reference
C++ implementation at https://github.com/tud-amr/mpc_planner

Test Categories:
1. Core MPCC Functionality
   - Path contouring (minimize lateral/lag error)
   - Reference velocity tracking
   - Terminal constraints

2. Constraint Types (matching C++ mpc_planner modules)
   - Safe Horizon (SH-MPC) - scenario-based collision avoidance
   - Gaussian (CC-MPC) - chance-constrained collision avoidance
   - Linearized - halfspace constraints
   - Ellipsoid - ellipsoidal obstacle constraints
   - Contouring - road boundary constraints

3. Vehicle Dynamics Models
   - SecondOrderUnicycleModel
   - ContouringSecondOrderUnicycleModel (with spline tracking)
   - SecondOrderBicycleModel
   - PointMassModel

4. Solver Performance & Efficiency
   - Solve time benchmarks
   - Convergence analysis
   - Warmstart effectiveness

5. Path Types
   - Straight paths
   - Curved paths
   - S-curves
   - Complex multi-segment paths

Reference: https://github.com/tud-amr/mpc_planner
"""

import sys
import os
import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import traceback

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import test framework
from test.integration.integration_test_framework import (
    IntegrationTestFramework, TestConfig, TestResult, create_reference_path
)
from planning.obstacle_manager import ObstacleConfig, create_unicycle_obstacle


@dataclass
class TestCase:
    """Individual test case configuration."""
    name: str
    description: str
    objective: str
    constraints: List[str]
    vehicle_model: str
    path_type: str
    path_length: float
    num_obstacles: int
    obstacle_dynamics: List[str]
    prediction_types: Optional[List[str]] = None
    duration: float = 10.0
    timestep: float = 0.1
    timeout: float = 120.0
    expected_success: bool = True
    performance_threshold_ms: Optional[float] = None  # Max avg solve time
    tags: List[str] = field(default_factory=list)


@dataclass
class TestSuiteResult:
    """Results from running the test suite."""
    total_tests: int
    passed: int
    failed: int
    skipped: int
    test_results: Dict[str, Dict]
    total_time: float
    timestamp: str


class MPCCTestSuite:
    """
    Comprehensive test suite for MPCC library functionality.

    Based on the C++ mpc_planner reference implementation:
    - https://github.com/tud-amr/mpc_planner

    Tests modular components:
    - MPCBaseModule equivalent (core optimization)
    - ContouringModule equivalent (path following)
    - Constraint modules (Safe Horizon, Gaussian, Ellipsoid, etc.)
    - Vehicle dynamics models
    """

    def __init__(self, output_dir: str = "test_results/mpcc_suite"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.framework = IntegrationTestFramework()
        self.test_cases = self._define_test_cases()

    def _define_test_cases(self) -> List[TestCase]:
        """Define all test cases for the suite."""

        cases = []

        # ================================================================
        # 1. CORE MPCC FUNCTIONALITY TESTS
        # Reference: C++ ContouringModule - path contouring control
        # ================================================================

        # 1.1 Basic contouring on straight path
        cases.append(TestCase(
            name="mpcc_straight_path_basic",
            description="Basic MPCC on straight path - verify contouring minimizes lateral error",
            objective="contouring",
            constraints=["contouring"],
            vehicle_model="contouring_unicycle",
            path_type="straight",
            path_length=20.0,
            num_obstacles=0,
            obstacle_dynamics=[],
            duration=15.0,
            timeout=60.0,
            performance_threshold_ms=100.0,
            tags=["core", "contouring", "no_obstacles"]
        ))

        # 1.2 Contouring on curved path
        cases.append(TestCase(
            name="mpcc_curved_path_basic",
            description="MPCC on curved path - verify tracking through turns",
            objective="contouring",
            constraints=["contouring"],
            vehicle_model="contouring_unicycle",
            path_type="curve",
            path_length=20.0,
            num_obstacles=0,
            obstacle_dynamics=[],
            duration=20.0,
            timeout=90.0,
            tags=["core", "contouring", "curved"]
        ))

        # 1.3 Contouring on S-curve path
        cases.append(TestCase(
            name="mpcc_s_curve_basic",
            description="MPCC on S-curve - verify bidirectional turning capability",
            objective="contouring",
            constraints=["contouring"],
            vehicle_model="contouring_unicycle",
            path_type="s_curve",
            path_length=18.0,
            num_obstacles=0,
            obstacle_dynamics=[],
            duration=25.0,
            timeout=120.0,
            tags=["core", "contouring", "s_curve"]
        ))

        # 1.4 Goal reaching objective
        cases.append(TestCase(
            name="goal_reaching_basic",
            description="Goal-based MPC - direct position tracking",
            objective="goal",
            constraints=["linear"],
            vehicle_model="unicycle",
            path_type="straight",
            path_length=15.0,
            num_obstacles=0,
            obstacle_dynamics=[],
            duration=12.0,
            timeout=60.0,
            tags=["core", "goal"]
        ))

        # ================================================================
        # 2. SAFE HORIZON CONSTRAINT TESTS (SH-MPC)
        # Reference: C++ scenario_module - scenario-based robust MPC
        # ================================================================

        # 2.1 Safe Horizon with single obstacle
        cases.append(TestCase(
            name="safe_horizon_single_obstacle",
            description="SH-MPC with single Gaussian-predicted obstacle",
            objective="contouring",
            constraints=["safe_horizon", "contouring"],
            vehicle_model="contouring_unicycle",
            path_type="curve",
            path_length=15.0,
            num_obstacles=1,
            obstacle_dynamics=["unicycle"],
            prediction_types=["gaussian"],
            duration=20.0,
            timeout=180.0,
            tags=["safe_horizon", "obstacle_avoidance"]
        ))

        # 2.2 Safe Horizon with multiple obstacles
        cases.append(TestCase(
            name="safe_horizon_multi_obstacle",
            description="SH-MPC with multiple obstacles - scenario sampling test",
            objective="contouring",
            constraints=["safe_horizon", "contouring"],
            vehicle_model="contouring_unicycle",
            path_type="s_curve",
            path_length=15.0,
            num_obstacles=2,
            obstacle_dynamics=["unicycle", "unicycle"],
            prediction_types=["gaussian", "gaussian"],
            duration=25.0,
            timeout=300.0,
            tags=["safe_horizon", "multi_obstacle"]
        ))

        # ================================================================
        # 3. GAUSSIAN CONSTRAINT TESTS (CC-MPC)
        # Reference: C++ GaussianConstraintModule - chance constraints
        # ================================================================

        # 3.1 Gaussian constraints single obstacle
        cases.append(TestCase(
            name="gaussian_single_obstacle",
            description="CC-MPC with Gaussian uncertainty bounds",
            objective="contouring",
            constraints=["gaussian", "contouring"],
            vehicle_model="contouring_unicycle",
            path_type="curve",
            path_length=18.0,
            num_obstacles=1,
            obstacle_dynamics=["unicycle"],
            prediction_types=["gaussian"],
            duration=20.0,
            timeout=120.0,
            tags=["gaussian", "chance_constraint"]
        ))

        # 3.2 Gaussian constraints multiple obstacles
        cases.append(TestCase(
            name="gaussian_multi_obstacle",
            description="CC-MPC with multiple Gaussian obstacles",
            objective="contouring",
            constraints=["gaussian", "contouring"],
            vehicle_model="contouring_unicycle",
            path_type="s_curve",
            path_length=18.0,
            num_obstacles=2,
            obstacle_dynamics=["unicycle", "unicycle"],
            prediction_types=["gaussian", "gaussian"],
            duration=25.0,
            timeout=180.0,
            tags=["gaussian", "multi_obstacle"]
        ))

        # ================================================================
        # 4. LINEARIZED CONSTRAINT TESTS
        # Reference: C++ LinearizedConstraints - halfspace obstacles
        # ================================================================

        # 4.1 Linearized constraints single obstacle
        cases.append(TestCase(
            name="linearized_single_obstacle",
            description="Linearized halfspace constraints for obstacle avoidance",
            objective="goal",
            constraints=["linear"],
            vehicle_model="unicycle",
            path_type="straight",
            path_length=20.0,
            num_obstacles=1,
            obstacle_dynamics=["unicycle"],
            prediction_types=["deterministic"],
            duration=15.0,
            timeout=90.0,
            performance_threshold_ms=50.0,  # Should be fast
            tags=["linear", "obstacle_avoidance"]
        ))

        # 4.2 Linearized with multiple obstacles
        cases.append(TestCase(
            name="linearized_multi_obstacle",
            description="Linearized constraints with multiple obstacles",
            objective="goal",
            constraints=["linear"],
            vehicle_model="unicycle",
            path_type="curve",
            path_length=20.0,
            num_obstacles=3,
            obstacle_dynamics=["unicycle", "unicycle", "unicycle"],
            prediction_types=["deterministic", "deterministic", "deterministic"],
            duration=20.0,
            timeout=120.0,
            tags=["linear", "multi_obstacle"]
        ))

        # ================================================================
        # 5. ELLIPSOID CONSTRAINT TESTS
        # Reference: C++ EllipsoidConstraints - smooth convex constraints
        # ================================================================

        # 5.1 Ellipsoid constraints
        cases.append(TestCase(
            name="ellipsoid_single_obstacle",
            description="Ellipsoid collision avoidance constraints",
            objective="contouring",
            constraints=["ellipsoid", "contouring"],
            vehicle_model="contouring_unicycle",
            path_type="curve",
            path_length=18.0,
            num_obstacles=1,
            obstacle_dynamics=["unicycle"],
            prediction_types=["gaussian"],
            duration=20.0,
            timeout=120.0,
            tags=["ellipsoid", "smooth_constraints"]
        ))

        # ================================================================
        # 6. ROAD BOUNDARY CONSTRAINT TESTS
        # Reference: C++ ContouringModule - road boundary handling
        # ================================================================

        # 6.1 Contouring constraints on narrow road
        cases.append(TestCase(
            name="contouring_narrow_road",
            description="Road boundary constraints on narrow path",
            objective="contouring",
            constraints=["contouring"],
            vehicle_model="contouring_unicycle",
            path_type="s_curve",
            path_length=15.0,
            num_obstacles=0,
            obstacle_dynamics=[],
            duration=25.0,
            timeout=120.0,
            tags=["contouring", "road_boundary"]
        ))

        # ================================================================
        # 7. VEHICLE DYNAMICS MODEL TESTS
        # Reference: C++ dynamic models - unicycle, bicycle
        # ================================================================

        # 7.1 Standard unicycle model
        cases.append(TestCase(
            name="unicycle_model_basic",
            description="Standard second-order unicycle dynamics",
            objective="goal",
            constraints=["linear"],
            vehicle_model="unicycle",
            path_type="straight",
            path_length=15.0,
            num_obstacles=0,
            obstacle_dynamics=[],
            duration=10.0,
            timeout=60.0,
            tags=["dynamics", "unicycle"]
        ))

        # 7.2 Contouring unicycle model (with spline tracking)
        cases.append(TestCase(
            name="contouring_unicycle_spline_tracking",
            description="Contouring unicycle with spline parameter tracking",
            objective="contouring",
            constraints=["contouring"],
            vehicle_model="contouring_unicycle",
            path_type="curve",
            path_length=20.0,
            num_obstacles=0,
            obstacle_dynamics=[],
            duration=20.0,
            timeout=90.0,
            tags=["dynamics", "contouring_unicycle"]
        ))

        # 7.3 Point mass model
        cases.append(TestCase(
            name="point_mass_model_basic",
            description="Simple point mass double-integrator dynamics",
            objective="goal",
            constraints=["linear"],
            vehicle_model="point_mass",
            path_type="straight",
            path_length=15.0,
            num_obstacles=0,
            obstacle_dynamics=[],
            duration=10.0,
            timeout=60.0,
            tags=["dynamics", "point_mass"]
        ))

        # ================================================================
        # 8. COMBINED CONSTRAINT TESTS
        # Reference: C++ modular stacking of constraints
        # ================================================================

        # 8.1 Safe Horizon + Contouring (full MPCC)
        cases.append(TestCase(
            name="full_mpcc_safe_horizon",
            description="Full MPCC: Safe Horizon + Contouring constraints",
            objective="contouring",
            constraints=["safe_horizon", "contouring"],
            vehicle_model="contouring_unicycle",
            path_type="curve",
            path_length=18.0,
            num_obstacles=2,
            obstacle_dynamics=["unicycle", "unicycle"],
            prediction_types=["gaussian", "gaussian"],
            duration=30.0,
            timeout=300.0,
            tags=["combined", "full_mpcc"]
        ))

        # 8.2 Gaussian + Contouring
        cases.append(TestCase(
            name="combined_gaussian_contouring",
            description="Combined Gaussian obstacle + road boundary constraints",
            objective="contouring",
            constraints=["gaussian", "contouring"],
            vehicle_model="contouring_unicycle",
            path_type="s_curve",
            path_length=18.0,
            num_obstacles=1,
            obstacle_dynamics=["unicycle"],
            prediction_types=["gaussian"],
            duration=25.0,
            timeout=180.0,
            tags=["combined", "gaussian_contouring"]
        ))

        # ================================================================
        # 9. PERFORMANCE & STRESS TESTS
        # Reference: C++ real-time performance (20-30 Hz)
        # ================================================================

        # 9.1 Rapid solve (performance benchmark)
        cases.append(TestCase(
            name="performance_rapid_solve",
            description="Performance test - measure solve time for simple scenario",
            objective="contouring",
            constraints=["contouring"],
            vehicle_model="contouring_unicycle",
            path_type="straight",
            path_length=15.0,
            num_obstacles=0,
            obstacle_dynamics=[],
            duration=8.0,
            timestep=0.1,
            timeout=30.0,
            performance_threshold_ms=100.0,  # Should solve under 100ms avg
            tags=["performance", "benchmark"]
        ))

        # 9.2 Complex scenario stress test
        cases.append(TestCase(
            name="stress_test_complex",
            description="Stress test with multiple obstacles and constraints",
            objective="contouring",
            constraints=["linear", "contouring"],
            vehicle_model="contouring_unicycle",
            path_type="s_curve",
            path_length=20.0,
            num_obstacles=3,
            obstacle_dynamics=["unicycle", "unicycle", "unicycle"],
            prediction_types=["deterministic", "deterministic", "deterministic"],
            duration=30.0,
            timeout=300.0,
            tags=["stress", "complex"]
        ))

        return cases

    def run_test_case(self, test_case: TestCase) -> Dict:
        """Run a single test case and return results."""

        print(f"\n{'='*60}")
        print(f"Running: {test_case.name}")
        print(f"Description: {test_case.description}")
        print(f"{'='*60}")

        result = {
            "name": test_case.name,
            "description": test_case.description,
            "status": "unknown",
            "success": False,
            "error": None,
            "solve_times": [],
            "avg_solve_time_ms": None,
            "constraint_violations": 0,
            "final_position": None,
            "path_error": None,
            "tags": test_case.tags
        }

        start_time = time.time()

        try:
            # Create reference path
            ref_path_obj = create_reference_path(
                test_case.path_type,
                length=test_case.path_length
            )

            # Convert to numpy array for TestConfig
            if hasattr(ref_path_obj, 'x') and hasattr(ref_path_obj, 'y'):
                ref_path_points = np.column_stack([ref_path_obj.x, ref_path_obj.y])
            else:
                ref_path_points = np.array([[0, 0], [test_case.path_length, 0]])

            # Build TestConfig
            config = TestConfig(
                reference_path=ref_path_points,
                objective_module=test_case.objective,
                constraint_modules=test_case.constraints,
                vehicle_dynamics=test_case.vehicle_model,
                num_obstacles=test_case.num_obstacles,
                obstacle_dynamics=test_case.obstacle_dynamics,
                obstacle_prediction_types=test_case.prediction_types,
                test_name=test_case.name,
                duration=test_case.duration,
                timestep=test_case.timestep,
                timeout_seconds=test_case.timeout,
                enable_safe_horizon_diagnostics=False,
                show_predicted_trajectory=False,
                fallback_control_enabled=True
            )

            # Run the test
            test_result = self.framework.run_test(config)

            # Extract results
            result["success"] = test_result.success
            result["status"] = "passed" if test_result.success else "failed"
            result["solve_times"] = test_result.computation_times
            result["constraint_violations"] = sum(test_result.constraint_violations)

            if test_result.computation_times:
                result["avg_solve_time_ms"] = np.mean(test_result.computation_times) * 1000

            if test_result.vehicle_states:
                final_state = test_result.vehicle_states[-1]
                result["final_position"] = final_state[:2].tolist() if len(final_state) >= 2 else None

            # Check performance threshold
            if test_case.performance_threshold_ms is not None:
                if result["avg_solve_time_ms"] is not None:
                    if result["avg_solve_time_ms"] > test_case.performance_threshold_ms:
                        result["status"] = "performance_warning"
                        result["error"] = f"Avg solve time {result['avg_solve_time_ms']:.1f}ms exceeds threshold {test_case.performance_threshold_ms}ms"

            print(f"\nResult: {result['status'].upper()}")
            if result["avg_solve_time_ms"]:
                print(f"  Avg solve time: {result['avg_solve_time_ms']:.2f}ms")
            print(f"  Constraint violations: {result['constraint_violations']}")

        except Exception as e:
            result["status"] = "error"
            result["success"] = False
            result["error"] = str(e)
            print(f"\nERROR: {e}")
            traceback.print_exc()

        result["elapsed_time"] = time.time() - start_time
        return result

    def run_suite(self,
                  tags: Optional[List[str]] = None,
                  exclude_tags: Optional[List[str]] = None,
                  test_names: Optional[List[str]] = None) -> TestSuiteResult:
        """
        Run the test suite.

        Args:
            tags: Only run tests with these tags
            exclude_tags: Skip tests with these tags
            test_names: Only run tests with these names

        Returns:
            TestSuiteResult with all test outcomes
        """

        print("\n" + "="*70)
        print("MPCC COMPREHENSIVE TEST SUITE")
        print("Reference: https://github.com/tud-amr/mpc_planner")
        print("="*70)

        # Filter test cases
        cases_to_run = []
        for case in self.test_cases:
            # Check name filter
            if test_names and case.name not in test_names:
                continue

            # Check tag filters
            if tags:
                if not any(tag in case.tags for tag in tags):
                    continue

            if exclude_tags:
                if any(tag in case.tags for tag in exclude_tags):
                    continue

            cases_to_run.append(case)

        print(f"\nRunning {len(cases_to_run)} test(s)")

        # Run tests
        start_time = time.time()
        results = {}
        passed = 0
        failed = 0
        skipped = 0

        for i, case in enumerate(cases_to_run, 1):
            print(f"\n[{i}/{len(cases_to_run)}] ", end="")

            result = self.run_test_case(case)
            results[case.name] = result

            if result["status"] in ["passed", "performance_warning"]:
                passed += 1
            elif result["status"] == "skipped":
                skipped += 1
            else:
                failed += 1

        total_time = time.time() - start_time

        # Create summary
        suite_result = TestSuiteResult(
            total_tests=len(cases_to_run),
            passed=passed,
            failed=failed,
            skipped=skipped,
            test_results=results,
            total_time=total_time,
            timestamp=datetime.now().isoformat()
        )

        # Print summary
        self._print_summary(suite_result)

        # Save results
        self._save_results(suite_result)

        return suite_result

    def _print_summary(self, result: TestSuiteResult):
        """Print test suite summary."""

        print("\n" + "="*70)
        print("TEST SUITE SUMMARY")
        print("="*70)

        print(f"\nTotal tests: {result.total_tests}")
        print(f"  Passed:  {result.passed}")
        print(f"  Failed:  {result.failed}")
        print(f"  Skipped: {result.skipped}")
        print(f"\nTotal time: {result.total_time:.1f}s")

        # Group by category
        print("\n--- Results by Category ---")

        categories = {}
        for name, res in result.test_results.items():
            for tag in res.get("tags", []):
                if tag not in categories:
                    categories[tag] = {"passed": 0, "failed": 0}
                if res["status"] in ["passed", "performance_warning"]:
                    categories[tag]["passed"] += 1
                else:
                    categories[tag]["failed"] += 1

        for cat, counts in sorted(categories.items()):
            total = counts["passed"] + counts["failed"]
            print(f"  {cat}: {counts['passed']}/{total} passed")

        # List failures
        failures = [
            (name, res) for name, res in result.test_results.items()
            if res["status"] not in ["passed", "performance_warning", "skipped"]
        ]

        if failures:
            print("\n--- Failed Tests ---")
            for name, res in failures:
                print(f"  {name}: {res.get('error', 'Unknown error')}")

        # Performance warnings
        warnings = [
            (name, res) for name, res in result.test_results.items()
            if res["status"] == "performance_warning"
        ]

        if warnings:
            print("\n--- Performance Warnings ---")
            for name, res in warnings:
                print(f"  {name}: {res.get('error', '')}")

        # Pass rate
        pass_rate = (result.passed / result.total_tests * 100) if result.total_tests > 0 else 0
        print(f"\n{'='*70}")
        print(f"PASS RATE: {pass_rate:.1f}%")
        print(f"{'='*70}")

    def _save_results(self, result: TestSuiteResult):
        """Save test results to JSON file."""

        filename = os.path.join(
            self.output_dir,
            f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # Convert to serializable format
        data = {
            "total_tests": result.total_tests,
            "passed": result.passed,
            "failed": result.failed,
            "skipped": result.skipped,
            "total_time": result.total_time,
            "timestamp": result.timestamp,
            "test_results": result.test_results
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"\nResults saved to: {filename}")


# ============================================================================
# INDIVIDUAL TEST FUNCTIONS (for pytest compatibility)
# ============================================================================

def test_mpcc_straight_path():
    """Test basic MPCC on straight path."""
    suite = MPCCTestSuite()
    result = suite.run_suite(test_names=["mpcc_straight_path_basic"])
    assert result.failed == 0, "MPCC straight path test failed"


def test_mpcc_curved_path():
    """Test MPCC on curved path."""
    suite = MPCCTestSuite()
    result = suite.run_suite(test_names=["mpcc_curved_path_basic"])
    assert result.failed == 0, "MPCC curved path test failed"


def test_safe_horizon_constraint():
    """Test Safe Horizon (SH-MPC) constraint."""
    suite = MPCCTestSuite()
    result = suite.run_suite(test_names=["safe_horizon_single_obstacle"])
    assert result.failed == 0, "Safe Horizon constraint test failed"


def test_gaussian_constraint():
    """Test Gaussian (CC-MPC) constraint."""
    suite = MPCCTestSuite()
    result = suite.run_suite(test_names=["gaussian_single_obstacle"])
    assert result.failed == 0, "Gaussian constraint test failed"


def test_linearized_constraint():
    """Test linearized halfspace constraint."""
    suite = MPCCTestSuite()
    result = suite.run_suite(test_names=["linearized_single_obstacle"])
    assert result.failed == 0, "Linearized constraint test failed"


def test_contouring_unicycle():
    """Test contouring unicycle model with spline tracking."""
    suite = MPCCTestSuite()
    result = suite.run_suite(test_names=["contouring_unicycle_spline_tracking"])
    assert result.failed == 0, "Contouring unicycle test failed"


def test_performance_benchmark():
    """Test solver performance benchmark."""
    suite = MPCCTestSuite()
    result = suite.run_suite(test_names=["performance_rapid_solve"])
    # Check for performance warnings
    test_result = result.test_results.get("performance_rapid_solve", {})
    assert test_result.get("status") in ["passed", "performance_warning"], \
        "Performance benchmark failed"


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_all_tests():
    """Run all tests in the suite."""
    suite = MPCCTestSuite()
    return suite.run_suite()


def run_core_tests():
    """Run only core functionality tests."""
    suite = MPCCTestSuite()
    return suite.run_suite(tags=["core"])


def run_constraint_tests():
    """Run constraint-related tests."""
    suite = MPCCTestSuite()
    return suite.run_suite(tags=["safe_horizon", "gaussian", "linear", "ellipsoid"])


def run_quick_tests():
    """Run quick smoke tests."""
    suite = MPCCTestSuite()
    return suite.run_suite(test_names=[
        "mpcc_straight_path_basic",
        "linearized_single_obstacle",
        "performance_rapid_solve"
    ])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MPCC Comprehensive Test Suite")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--core", action="store_true", help="Run core tests only")
    parser.add_argument("--constraints", action="store_true", help="Run constraint tests")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke tests")
    parser.add_argument("--tags", nargs="+", help="Run tests with these tags")
    parser.add_argument("--exclude", nargs="+", help="Exclude tests with these tags")
    parser.add_argument("--test", nargs="+", help="Run specific test(s) by name")

    args = parser.parse_args()

    suite = MPCCTestSuite()

    if args.test:
        result = suite.run_suite(test_names=args.test)
    elif args.tags:
        result = suite.run_suite(tags=args.tags, exclude_tags=args.exclude)
    elif args.core:
        result = run_core_tests()
    elif args.constraints:
        result = run_constraint_tests()
    elif args.quick:
        result = run_quick_tests()
    else:
        # Default: run all tests
        result = run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if result.failed == 0 else 1)
