"""
Tests for the Performance Optimization Module.

Tests the configurable performance optimization pipeline for the MPCC solver.
"""

import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solver.performance_optimizations import (
    PerformanceConfig,
    PerformanceOptimizer,
    PerformanceLevel,
    BoundsCache,
    SplineCache,
    create_optimized_solver_options,
    enable_fast_mode,
    enable_safe_mode,
    get_global_optimizer,
    set_global_optimizer,
    timed,
)


class TestPerformanceConfig:
    """Tests for PerformanceConfig class."""

    def test_fast_preset(self):
        """Test fast preset configuration."""
        config = PerformanceConfig.fast()
        assert config.level == PerformanceLevel.FAST
        assert config.enable_jit is True
        assert config.enable_caching is True
        assert config.log_level == "ERROR"
        assert config.constraint_pruning is True
        assert config.ipopt_settings['ipopt.max_iter'] == 200
        assert config.ipopt_settings['ipopt.tol'] == 5e-2

    def test_balanced_preset(self):
        """Test balanced preset configuration."""
        config = PerformanceConfig.balanced()
        assert config.level == PerformanceLevel.BALANCED
        assert config.enable_jit is True
        assert config.enable_caching is True
        assert config.log_level == "WARNING"
        assert config.constraint_pruning is False
        assert config.ipopt_settings['ipopt.max_iter'] == 500

    def test_safe_preset(self):
        """Test safe preset configuration."""
        config = PerformanceConfig.safe()
        assert config.level == PerformanceLevel.SAFE
        assert config.enable_jit is False
        assert config.enable_caching is True
        assert config.log_level == "INFO"
        assert config.ipopt_settings['ipopt.max_iter'] == 2000
        assert config.ipopt_settings['ipopt.tol'] == 1e-4

    def test_from_config(self):
        """Test creating config from dictionary."""
        config_dict = {
            "performance": {
                "level": "fast",
                "enable_jit": False,  # Override fast preset
            }
        }
        config = PerformanceConfig.from_config(config_dict)
        assert config.level == PerformanceLevel.FAST
        assert config.enable_jit is False  # Overridden


class TestBoundsCache:
    """Tests for BoundsCache class."""

    def test_cache_hit_miss(self):
        """Test cache hit and miss tracking."""
        cache = BoundsCache(max_size=100)

        # Create mock dynamics model
        class MockDynamics:
            def get_bounds(self, var_name):
                return (-1.0, 1.0, None)

        model = MockDynamics()

        # First call - miss
        result = cache.get(model, "x")
        assert result == (-1.0, 1.0, None)
        assert cache.stats["misses"] == 1
        assert cache.stats["hits"] == 0

        # Second call - hit
        result = cache.get(model, "x")
        assert result == (-1.0, 1.0, None)
        assert cache.stats["misses"] == 1
        assert cache.stats["hits"] == 1

        # Different variable - miss
        result = cache.get(model, "y")
        assert cache.stats["misses"] == 2

    def test_cache_eviction(self):
        """Test cache eviction when full."""
        cache = BoundsCache(max_size=5)

        class MockDynamics:
            def get_bounds(self, var_name):
                return (-1.0, 1.0, None)

        model = MockDynamics()

        # Fill cache
        for i in range(10):
            cache.get(model, f"var_{i}")

        # Cache should have been evicted
        assert cache.stats["size"] <= 5

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = BoundsCache()

        class MockDynamics:
            def get_bounds(self, var_name):
                return (-1.0, 1.0, None)

        model = MockDynamics()
        cache.get(model, "x")
        assert cache.stats["size"] > 0

        cache.clear()
        assert cache.stats["size"] == 0
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0


class TestSplineCache:
    """Tests for SplineCache class."""

    def test_spline_cache_computation(self):
        """Test spline caching with computation."""
        cache = SplineCache(max_size=100)

        # Mock spline function
        def mock_spline(s):
            return s * 2

        s_values = np.array([0.0, 0.5, 1.0])

        # First call - computed
        result = cache.get_or_compute(mock_spline, s_values, "test_key")
        np.testing.assert_array_equal(result, np.array([0.0, 1.0, 2.0]))

        # Second call with same key - cached
        result2 = cache.get_or_compute(mock_spline, s_values, "test_key")
        np.testing.assert_array_equal(result, result2)


class TestPerformanceOptimizer:
    """Tests for PerformanceOptimizer class."""

    def test_optimizer_creation(self):
        """Test optimizer creation with different configs."""
        fast_opt = PerformanceOptimizer(PerformanceConfig.fast())
        assert fast_opt.config.level == PerformanceLevel.FAST

        safe_opt = PerformanceOptimizer(PerformanceConfig.safe())
        assert safe_opt.config.level == PerformanceLevel.SAFE

    def test_ipopt_options(self):
        """Test IPOPT options generation."""
        fast_opt = PerformanceOptimizer(PerformanceConfig.fast())
        opts = fast_opt.get_ipopt_options()

        assert 'ipopt.max_iter' in opts
        assert opts['ipopt.max_iter'] == 200
        assert opts['ipopt.print_level'] == 0

    def test_timing(self):
        """Test timing functionality."""
        optimizer = PerformanceOptimizer()

        start = optimizer.start_timing("test")
        time.sleep(0.01)  # 10ms
        elapsed = optimizer.end_timing("test", start)

        assert elapsed > 0.005  # At least 5ms
        assert "test" in optimizer.timing_stats

    def test_context_manager(self):
        """Test context manager usage."""
        optimizer = PerformanceOptimizer(PerformanceConfig.fast())

        with optimizer as opt:
            assert opt is optimizer

        # Logging should be restored after context exit

    def test_cache_stats(self):
        """Test cache statistics."""
        optimizer = PerformanceOptimizer()
        stats = optimizer.cache_stats

        assert "bounds_cache" in stats
        assert "spline_cache" in stats


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_optimized_solver_options(self):
        """Test creating optimized solver options."""
        fast_opts = create_optimized_solver_options(level="fast")
        assert fast_opts['ipopt.max_iter'] == 200

        safe_opts = create_optimized_solver_options(level="safe")
        assert safe_opts['ipopt.max_iter'] == 2000

    def test_global_optimizer(self):
        """Test global optimizer functions."""
        enable_fast_mode()
        opt = get_global_optimizer()
        assert opt.config.level == PerformanceLevel.FAST

        enable_safe_mode()
        opt = get_global_optimizer()
        assert opt.config.level == PerformanceLevel.SAFE

    def test_timed_decorator(self):
        """Test timed decorator."""
        @timed("test_func")
        def slow_func():
            time.sleep(0.01)
            return 42

        result = slow_func()
        assert result == 42


class TestSolverIntegration:
    """Tests for solver integration."""

    def test_solver_with_performance_mode(self):
        """Test CasADi solver with performance mode."""
        try:
            from solver import CasADiSolver, PerformanceConfig

            config = {
                "planner": {
                    "horizon": 5,
                    "timestep": 0.1,
                }
            }

            # Test with string preset
            solver_fast = CasADiSolver(config, performance="fast")
            assert solver_fast._perf_config is not None
            assert solver_fast._perf_config.level == PerformanceLevel.FAST

            # Test with balanced (default)
            solver_balanced = CasADiSolver(config)
            assert solver_balanced._perf_config.level == PerformanceLevel.BALANCED

            # Test with custom config
            custom_config = PerformanceConfig.safe()
            custom_config.enable_jit = True
            solver_custom = CasADiSolver(config, performance=custom_config)
            assert solver_custom._perf_config.enable_jit is True

        except ImportError as e:
            pytest.skip(f"Solver import failed: {e}")

    def test_solver_performance_mode_switch(self):
        """Test switching performance mode at runtime."""
        try:
            from solver import CasADiSolver

            config = {
                "planner": {
                    "horizon": 5,
                    "timestep": 0.1,
                }
            }

            solver = CasADiSolver(config, performance="balanced")
            assert solver._perf_config.level == PerformanceLevel.BALANCED

            solver.set_performance_mode("fast")
            assert solver._perf_config.level == PerformanceLevel.FAST

            solver.set_performance_mode("safe")
            assert solver._perf_config.level == PerformanceLevel.SAFE

        except ImportError as e:
            pytest.skip(f"Solver import failed: {e}")

    def test_solver_performance_stats(self):
        """Test getting performance statistics from solver."""
        try:
            from solver import CasADiSolver

            config = {
                "planner": {
                    "horizon": 5,
                    "timestep": 0.1,
                }
            }

            solver = CasADiSolver(config, performance="fast")
            stats = solver.get_performance_stats()

            assert "config" in stats
            assert stats["config"]["level"] == "fast"
            assert "timing" in stats
            assert "cache" in stats

        except ImportError as e:
            pytest.skip(f"Solver import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
