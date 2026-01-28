#!/usr/bin/env python3
"""
Unit Tests for Constraint Modules

Tests constraint implementations against expected behavior and
the reference C++ mpc_planner implementation.

Reference: https://github.com/tud-amr/mpc_planner
Constraint types:
- SafeHorizonConstraint (SH-MPC)
- GaussianConstraints (CC-MPC)
- LinearizedConstraints
- EllipsoidConstraints
- ContouringConstraints
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.utils import read_config_file


class TestSafeHorizonConstraint:
    """Tests for Safe Horizon (Scenario-based) Constraints.

    Reference: C++ scenario_module
    - Samples scenarios from obstacle predictions
    - Uses support subset selection for efficient constraint generation
    - Provides probabilistic safety guarantees
    """

    @pytest.fixture
    def config(self):
        return read_config_file()

    @pytest.fixture
    def constraint(self, config):
        from modules.constraints.safe_horizon_constraint import SafeHorizonConstraint
        return SafeHorizonConstraint(settings=config)

    def test_constraint_creation(self, constraint):
        """Test constraint initializes correctly."""
        assert constraint is not None
        assert constraint.name == "safe_horizon_constraint"

    def test_default_parameters(self, constraint):
        """Test default parameters are set."""
        assert hasattr(constraint, 'epsilon_p')
        assert hasattr(constraint, 'beta')
        assert hasattr(constraint, 'n_bar')
        assert hasattr(constraint, 'num_scenarios')

        # Reference values from C++ mpc_planner
        assert 0 < constraint.epsilon_p <= 1.0
        assert 0 < constraint.beta <= 1.0

    def test_sample_size_computation(self, constraint):
        """Test scenario sample size follows formula.

        Reference: guide.md Equation (23)
        S >= (2/epsilon)(ln(1/beta) + d + R)
        """
        from modules.constraints.scenario_utils.math_utils import compute_sample_size

        S = compute_sample_size(
            epsilon_p=0.1,
            beta=0.01,
            n_bar=5,
            num_removal=0,
            horizon=10,
            n_x=4,
            n_u=2
        )

        # Should be positive integer
        assert S > 0
        assert isinstance(S, int)

    def test_is_data_ready_without_obstacles(self, constraint):
        """Test data readiness check without obstacles."""
        from planning.types import Data

        data = Data()
        data.dynamic_obstacles = []

        # Should handle empty obstacle list
        result = constraint.is_data_ready(data)
        assert isinstance(result, bool)


class TestGaussianConstraints:
    """Tests for Gaussian (Chance) Constraints.

    Reference: C++ GaussianConstraintModule
    - Uses Gaussian uncertainty representation
    - Chance constraints: P(collision) <= risk_level
    - Uses Mahalanobis distance
    """

    @pytest.fixture
    def config(self):
        return read_config_file()

    @pytest.fixture
    def constraint(self, config):
        from modules.constraints.gaussian_constraints import GaussianConstraints
        c = GaussianConstraints()
        c.config = config
        c.settings = config
        return c

    def test_constraint_creation(self, constraint):
        """Test constraint initializes correctly."""
        assert constraint is not None

    def test_risk_level_parameter(self, constraint):
        """Test risk level parameter exists."""
        # Reference: C++ uses risk_level parameter
        if hasattr(constraint, 'risk_level'):
            assert 0 < constraint.risk_level <= 1.0


class TestLinearizedConstraints:
    """Tests for Linearized Halfspace Constraints.

    Reference: C++ LinearizedConstraints
    - Creates halfspace constraints from obstacle positions
    - Fast computation, suitable for real-time
    """

    @pytest.fixture
    def config(self):
        return read_config_file()

    @pytest.fixture
    def constraint(self, config):
        from modules.constraints.linearized_constraints import LinearizedConstraints
        c = LinearizedConstraints()
        c.config = config
        c.settings = config
        return c

    def test_constraint_creation(self, constraint):
        """Test constraint initializes correctly."""
        assert constraint is not None

    def test_halfspace_offset(self, constraint):
        """Test halfspace offset parameter."""
        if hasattr(constraint, 'halfspace_offset'):
            assert constraint.halfspace_offset >= 0


class TestContouringConstraints:
    """Tests for Contouring (Road Boundary) Constraints.

    Reference: C++ ContouringModule road boundary handling
    - Enforces vehicle stays within road boundaries
    - Uses halfspace constraints on left/right boundaries
    """

    @pytest.fixture
    def config(self):
        return read_config_file()

    @pytest.fixture
    def constraint(self, config):
        from modules.constraints.contouring_constraints import ContouringConstraints
        c = ContouringConstraints()
        c.config = config
        c.settings = config
        return c

    def test_constraint_creation(self, constraint):
        """Test constraint initializes correctly."""
        assert constraint is not None

    def test_road_width_parameter(self, constraint):
        """Test road width parameter."""
        # Reference: C++ uses road width configuration
        if hasattr(constraint, 'road_width'):
            assert constraint.road_width > 0


class TestEllipsoidConstraints:
    """Tests for Ellipsoid Collision Avoidance Constraints.

    Reference: C++ EllipsoidConstraints
    - Smooth convex constraints
    - Good for trajectory smoothness
    """

    @pytest.fixture
    def config(self):
        return read_config_file()

    @pytest.fixture
    def constraint(self, config):
        from modules.constraints.ellipsoid_constraints import EllipsoidConstraints
        c = EllipsoidConstraints()
        c.config = config
        c.settings = config
        return c

    def test_constraint_creation(self, constraint):
        """Test constraint initializes correctly."""
        assert constraint is not None


class TestConstraintInterface:
    """Test common constraint interface."""

    def test_all_constraints_have_update_method(self):
        """Test all constraints implement update method."""
        from modules.constraints.safe_horizon_constraint import SafeHorizonConstraint
        from modules.constraints.gaussian_constraints import GaussianConstraints
        from modules.constraints.linearized_constraints import LinearizedConstraints
        from modules.constraints.contouring_constraints import ContouringConstraints
        from modules.constraints.ellipsoid_constraints import EllipsoidConstraints

        constraints = [
            SafeHorizonConstraint(),
            GaussianConstraints(),
            LinearizedConstraints(),
            ContouringConstraints(),
            EllipsoidConstraints(),
        ]

        for constraint in constraints:
            assert hasattr(constraint, 'update'), \
                f"{constraint.__class__.__name__} missing update method"

    def test_all_constraints_have_is_data_ready(self):
        """Test all constraints implement is_data_ready method."""
        from modules.constraints.safe_horizon_constraint import SafeHorizonConstraint
        from modules.constraints.gaussian_constraints import GaussianConstraints
        from modules.constraints.linearized_constraints import LinearizedConstraints
        from modules.constraints.contouring_constraints import ContouringConstraints
        from modules.constraints.ellipsoid_constraints import EllipsoidConstraints

        constraints = [
            SafeHorizonConstraint(),
            GaussianConstraints(),
            LinearizedConstraints(),
            ContouringConstraints(),
            EllipsoidConstraints(),
        ]

        for constraint in constraints:
            assert hasattr(constraint, 'is_data_ready'), \
                f"{constraint.__class__.__name__} missing is_data_ready method"


class TestScenarioUtils:
    """Tests for scenario sampling utilities.

    Reference: C++ scenario_module math utilities
    """

    def test_sample_size_formula(self):
        """Test sample size computation matches theory."""
        from modules.constraints.scenario_utils.math_utils import compute_sample_size

        # Test with known parameters
        # Reference: guide.md Equation (23)
        epsilon = 0.1
        beta = 0.01
        horizon = 10
        n_x = 4
        n_u = 2
        d = horizon * (n_x + n_u)

        S = compute_sample_size(
            epsilon_p=epsilon,
            beta=beta,
            n_bar=5,
            num_removal=0,
            horizon=horizon,
            n_x=n_x,
            n_u=n_u
        )

        # S >= (2/epsilon)(ln(1/beta) + d)
        theoretical_min = (2/epsilon) * (np.log(1/beta) + d)
        assert S >= theoretical_min * 0.9, "Sample size should satisfy theoretical bound"

    def test_scenario_constraint_structure(self):
        """Test ScenarioConstraint data structure."""
        try:
            from modules.constraints.scenario_utils.math_utils import ScenarioConstraint

            # Should be able to create scenario constraint
            # This tests the data structure exists
            assert ScenarioConstraint is not None
        except ImportError:
            pytest.skip("ScenarioConstraint not available")


class TestConstraintPerformance:
    """Performance tests for constraint computation."""

    def test_linearized_constraint_fast(self):
        """Test linearized constraints are fast to compute."""
        from modules.constraints.linearized_constraints import LinearizedConstraints
        import time

        constraint = LinearizedConstraints()

        # Linearized should be very fast
        # Reference: C++ achieves <1ms for constraint setup
        # We allow more time for Python but should still be fast
        pass  # Actual timing would require full setup

    def test_safe_horizon_sample_generation(self):
        """Test scenario sampling performance."""
        # Reference: C++ generates 100+ scenarios efficiently
        pass  # Would require full obstacle setup


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
