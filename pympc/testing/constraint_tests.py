"""
Comprehensive constraint tests for PyMPC.

This module provides tests for all constraint types to ensure
they work correctly and integrate properly with the MPC framework.
"""

import numpy as np
import casadi as ca
from typing import Dict, Any, List, Tuple
import time

from ..core.planner import MPCCPlanner
from ..core.dynamics import BicycleModel
from ..constraints import (
    ContouringConstraints,
    ScenarioConstraints,
    LinearizedConstraints,
    EllipsoidConstraints,
    GaussianConstraints,
    DecompositionConstraints,
    GuidanceConstraints
)
from ..objectives import ContouringObjective, GoalObjective
from .base_test import BaseMPCTest, TestConfig, TestResult


class ConstraintTestSuite:
    """
    Test suite for all constraint types.
    
    This suite tests each constraint type individually to ensure
    they work correctly and can be integrated into MPC systems.
    """

    def __init__(self):
        self.tests = []
        self.results = []

    def add_constraint_test(self, constraint_type: str, test_config: TestConfig):
        """Add a constraint test to the suite."""
        test = ConstraintTest(constraint_type, test_config)
        self.tests.append(test)

    def run_all_tests(self) -> List[TestResult]:
        """Run all constraint tests."""
        print("Running Constraint Test Suite")
        print("=" * 50)
        
        for test in self.tests:
            print(f"\nRunning test: {test.constraint_type}")
            result = test.run_test()
            self.results.append(result)
            
            # Print summary
            status = "PASSED" if result.success else "FAILED"
            print(f"Result: {status}")
            print(f"Duration: {result.duration:.2f}s")
            print(f"Final Error: {result.final_error:.4f}")
        
        return self.results

    def get_summary(self) -> str:
        """Get summary of all test results."""
        if not self.results:
            return "No tests run"
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(r.duration for r in self.results)
        avg_duration = total_duration / total_tests if total_tests > 0 else 0
        
        summary = f"""
Constraint Test Suite Summary:
Total Tests: {total_tests}
Passed: {passed_tests}
Failed: {failed_tests}
Success Rate: {passed_tests/total_tests:.1%}
Total Duration: {total_duration:.2f}s
Average Duration: {avg_duration:.2f}s
"""
        
        return summary


class ConstraintTest(BaseMPCTest):
    """
    Individual constraint test.
    
    Tests a specific constraint type to ensure it works correctly.
    """

    def __init__(self, constraint_type: str, config: TestConfig):
        super().__init__(config)
        self.constraint_type = constraint_type
        self.planner = None
        self.dynamics = None
        self.constraint = None

    def setup_test_environment(self):
        """Setup test environment for constraint testing."""
        # Create dynamics model
        self.dynamics = BicycleModel(dt=0.1, wheelbase=2.5)
        
        # Create planner
        self.planner = MPCCPlanner(
            dynamics=self.dynamics,
            horizon_length=10,
            dt=0.1
        )
        
        # Add basic objectives
        goal_obj = GoalObjective(
            goal_position=[10.0, 0.0],
            weight=1.0
        )
        self.planner.add_objective(goal_obj)
        
        # Create constraint based on type
        self.constraint = self._create_constraint()
        if self.constraint:
            self.planner.add_constraint(self.constraint)

    def create_mpc_system(self):
        """Create MPC system for testing."""
        return self.planner

    def run_mpc_iteration(self, iteration: int) -> Tuple[bool, float, Dict[str, Any]]:
        """Run a single MPC iteration."""
        try:
            # Initial state
            initial_state = np.array([0.0, 0.0, 0.0, 5.0])
            
            # Solve MPC problem
            solution = self.planner.solve(initial_state=initial_state)
            
            if not solution or solution.get('status') != 'optimal':
                return False, float('inf'), {'error': 'MPC solve failed'}
            
            # Calculate error
            error = self._calculate_error(solution)
            
            # Create state information
            state = {
                'x': 0.0,
                'y': 0.0,
                'psi': 0.0,
                'v': 5.0,
                'constraint_type': self.constraint_type,
                'error': error
            }
            
            return True, error, state
            
        except Exception as e:
            return False, float('inf'), {'error': str(e)}

    def _create_constraint(self):
        """Create constraint based on type."""
        if self.constraint_type == "contouring":
            return ContouringConstraints(
                reference_path=self._create_reference_path(),
                road_width=4.0
            )
        elif self.constraint_type == "scenario":
            return ScenarioConstraints(
                obstacles=self._create_obstacles(),
                safety_margin=1.0
            )
        elif self.constraint_type == "linearized":
            return LinearizedConstraints(
                obstacles=self._create_obstacles(),
                safety_margin=1.0
            )
        elif self.constraint_type == "ellipsoid":
            return EllipsoidConstraints(
                obstacles=self._create_obstacles(),
                safety_margin=1.0
            )
        elif self.constraint_type == "gaussian":
            return GaussianConstraints(
                obstacles=self._create_obstacles(),
                safety_margin=1.0
            )
        elif self.constraint_type == "decomposition":
            return DecompositionConstraints(
                road_boundaries=self._create_road_boundaries(),
                safety_margin=1.0
            )
        elif self.constraint_type == "guidance":
            return GuidanceConstraints(
                guidance_paths=self._create_guidance_paths(),
                safety_margin=1.0
            )
        else:
            return None

    def _create_reference_path(self) -> np.ndarray:
        """Create reference path for contouring constraints."""
        s = np.linspace(0, 1, 50)
        x = s * 20.0
        y = 2.0 * np.sin(0.2 * x)
        return np.column_stack([x, y])

    def _create_obstacles(self) -> List[Dict[str, Any]]:
        """Create obstacles for testing."""
        obstacles = []
        for i in range(3):
            obstacle = {
                'position': [5.0 + i * 3.0, 0.0],
                'velocity': [0.0, 0.0],
                'radius': 1.0,
                'uncertainty': [0.1, 0.1]
            }
            obstacles.append(obstacle)
        return obstacles

    def _create_road_boundaries(self) -> Dict[str, np.ndarray]:
        """Create road boundaries for decomposition constraints."""
        x = np.linspace(0, 20, 50)
        y_left = -2.0 * np.ones_like(x)
        y_right = 2.0 * np.ones_like(x)
        
        return {
            'left': np.column_stack([x, y_left]),
            'right': np.column_stack([x, y_right])
        }

    def _create_guidance_paths(self) -> List[np.ndarray]:
        """Create guidance paths for guidance constraints."""
        paths = []
        for i in range(3):
            x = np.linspace(0, 20, 50)
            y = (i - 1) * 2.0 * np.ones_like(x)
            paths.append(np.column_stack([x, y]))
        return paths

    def _calculate_error(self, solution: Dict[str, Any]) -> float:
        """Calculate error metric for the test."""
        if 'states' not in solution or solution['states'] is None:
            return float('inf')
        
        states = solution['states']
        if states.size == 0:
            return float('inf')
        
        # Calculate distance to goal
        goal = np.array([10.0, 0.0])
        final_state = states[:, -1]
        error = np.linalg.norm(final_state[:2] - goal)
        
        return error

    def _check_convergence(self, error: float) -> bool:
        """Check if the test has converged."""
        return error < 1.0

    def _check_timeout(self) -> bool:
        """Check if the test has timed out."""
        elapsed = time.time() - self.start_time
        return elapsed > 10.0


def create_constraint_test_suite() -> ConstraintTestSuite:
    """Create comprehensive constraint test suite."""
    suite = ConstraintTestSuite()
    
    # Test all constraint types
    constraint_types = [
        "contouring",
        "scenario", 
        "linearized",
        "ellipsoid",
        "gaussian",
        "decomposition",
        "guidance"
    ]
    
    for constraint_type in constraint_types:
        config = TestConfig(
            test_name=f"constraint_{constraint_type}_test",
            description=f"Test for {constraint_type} constraints",
            timeout=10.0,
            max_iterations=50,
            goal_tolerance=1.0,
            enable_visualization=False,
            save_outputs=False
        )
        suite.add_constraint_test(constraint_type, config)
    
    return suite


if __name__ == "__main__":
    # Run constraint test suite
    suite = create_constraint_test_suite()
    results = suite.run_all_tests()
    
    # Print summary
    print(suite.get_summary())
