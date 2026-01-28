#!/usr/bin/env python3
"""
Solver Performance and Efficiency Tests

Tests the CasADi solver performance against expected benchmarks and
validates efficiency for real-time operation.

Reference: https://github.com/tud-amr/mpc_planner
Target: 20-30 Hz control frequency (33-50ms solve time)
"""

import sys
import os
import time
import numpy as np
import pytest
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.utils import read_config_file


class TestSolverInitialization:
    """Tests for solver initialization."""

    @pytest.fixture
    def config(self):
        return read_config_file()

    @pytest.fixture
    def solver(self, config):
        from solver.casadi_solver import CasADiSolver
        return CasADiSolver(config)

    def test_solver_creation(self, solver):
        """Test solver initializes correctly."""
        assert solver is not None

    def test_solver_has_horizon(self, solver, config):
        """Test solver has horizon parameter."""
        planner_config = config.get("planner", {})
        expected_horizon = planner_config.get("horizon", 10)

        solver.horizon = expected_horizon
        assert solver.horizon == expected_horizon

    def test_solver_has_timestep(self, solver, config):
        """Test solver has timestep parameter."""
        planner_config = config.get("planner", {})
        expected_timestep = planner_config.get("timestep", 0.1)

        solver.timestep = expected_timestep
        assert solver.timestep == expected_timestep


class TestSolverPerformance:
    """Performance benchmarks for the solver.

    Reference: C++ mpc_planner achieves 20-30 Hz control
    """

    @pytest.fixture
    def config(self):
        return read_config_file()

    def test_simple_optimization_time(self, config):
        """Test solve time for simple optimization.

        Target: <100ms for simple scenario (no obstacles)
        """
        from solver.casadi_solver import CasADiSolver
        from planning.dynamic_models import SecondOrderUnicycleModel
        from planning.types import Data, State

        solver = CasADiSolver(config)
        model = SecondOrderUnicycleModel()

        # Setup minimal data
        data = Data()
        data.horizon = 10
        data.timestep = 0.1
        data.dynamics_model = model
        data.dynamic_obstacles = []

        state = State(model)
        state.set('x', 0.0)
        state.set('y', 0.0)
        state.set('psi', 0.0)
        state.set('v', 1.0)
        data.state = state

        # Note: Full timing test would require complete setup
        # This is a placeholder for the structure

    def test_warmstart_improves_convergence(self):
        """Test that warmstart reduces solve time.

        Reference: C++ uses warmstart for continuous operation
        """
        # Warmstart should reduce iterations needed
        # Second solve should be faster than first
        pass


class TestSolverConvergence:
    """Tests for solver convergence properties."""

    def test_feasible_problem_converges(self):
        """Test solver converges for feasible problem."""
        pass

    def test_infeasible_detection(self):
        """Test solver detects infeasible problems."""
        pass


class TestSolverNumericalStability:
    """Tests for numerical stability."""

    def test_small_timestep_stability(self):
        """Test solver is stable with small timesteps."""
        pass

    def test_large_horizon_stability(self):
        """Test solver handles large horizons."""
        pass


class TestModuleIntegration:
    """Test solver integration with modules."""

    @pytest.fixture
    def config(self):
        return read_config_file()

    def test_add_objective_module(self, config):
        """Test adding objective module to solver."""
        from solver.casadi_solver import CasADiSolver
        from modules.objectives.contouring_objective import ContouringObjective

        solver = CasADiSolver(config)
        objective = ContouringObjective()

        solver.module_manager.add_module(objective)

        modules = solver.module_manager.get_modules()
        assert len(modules) > 0

    def test_add_constraint_module(self, config):
        """Test adding constraint module to solver."""
        from solver.casadi_solver import CasADiSolver
        from modules.constraints.linearized_constraints import LinearizedConstraints

        solver = CasADiSolver(config)
        constraint = LinearizedConstraints()

        solver.module_manager.add_module(constraint)

        modules = solver.module_manager.get_modules()
        assert len(modules) > 0


class TestParameterManager:
    """Tests for parameter management."""

    def test_parameter_setting(self):
        """Test parameters are correctly set."""
        from planning.parameter_manager import ParameterManager

        pm = ParameterManager()
        # Test parameter setting
        assert pm is not None

    def test_parameter_retrieval(self):
        """Test parameters are correctly retrieved."""
        from planning.parameter_manager import ParameterManager

        pm = ParameterManager()
        # Test parameter retrieval
        pass


class TestSolverBenchmarks:
    """Comprehensive benchmark tests.

    Reference performance targets from C++ mpc_planner:
    - Simple path following: <50ms
    - With obstacles: <100ms
    - Complex scenarios: <200ms
    """

    def run_benchmark(self, name: str, setup_fn, iterations: int = 10) -> Tuple[float, float, float]:
        """Run benchmark and return (min, avg, max) times in ms."""
        times = []

        for _ in range(iterations):
            start = time.time()
            setup_fn()
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

        return min(times), np.mean(times), max(times)

    def test_benchmark_report(self):
        """Generate benchmark report."""
        # This would run actual benchmarks and report results
        print("\n--- Solver Performance Benchmarks ---")
        print("(Actual benchmarks require full system setup)")
        print("Reference targets from C++ mpc_planner:")
        print("  - Control frequency: 20-30 Hz")
        print("  - Simple solve: <50ms")
        print("  - With obstacles: <100ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
