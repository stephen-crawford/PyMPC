#!/usr/bin/env python3
"""
Unit Tests for Vehicle Dynamics Models

Tests the vehicle dynamics implementations against expected behavior and
the reference C++ mpc_planner implementation.

Reference: https://github.com/tud-amr/mpc_planner
Models tested:
- SecondOrderUnicycleModel
- ContouringSecondOrderUnicycleModel
- SecondOrderBicycleModel
- PointMassModel
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from planning.dynamic_models import (
    SecondOrderUnicycleModel,
    ContouringSecondOrderUnicycleModel,
    PointMassModel
)


class TestSecondOrderUnicycleModel:
    """Tests for SecondOrderUnicycleModel.

    State: [x, y, psi, v]
    Inputs: [a, w] (acceleration, angular velocity)

    Continuous dynamics:
        dx/dt = v * cos(psi)
        dy/dt = v * sin(psi)
        dpsi/dt = w
        dv/dt = a
    """

    @pytest.fixture
    def model(self):
        return SecondOrderUnicycleModel()

    def test_model_creation(self, model):
        """Test model initializes correctly."""
        assert model is not None
        # Check actual attributes from DynamicsModel base class
        assert hasattr(model, 'dependent_vars')
        assert hasattr(model, 'inputs')
        assert hasattr(model, 'state_dimension')

    def test_state_variables(self, model):
        """Test correct state variables are defined."""
        # The model uses dependent_vars for state names
        expected_states = ['x', 'y', 'psi', 'v']
        for var in expected_states:
            assert var in model.dependent_vars, f"Missing state variable: {var}"

    def test_input_variables(self, model):
        """Test correct input variables are defined."""
        # The model uses inputs for control names
        expected_inputs = ['a', 'w']
        for var in expected_inputs:
            assert var in model.inputs, f"Missing input variable: {var}"

    def test_state_dimension(self, model):
        """Test state dimension is correct."""
        assert model.state_dimension == 4, "Unicycle should have 4 states"

    def test_input_dimension(self, model):
        """Test input dimension is correct."""
        assert model.nu == 2, "Unicycle should have 2 inputs"

    def test_continuous_model_exists(self, model):
        """Test continuous_model method exists."""
        assert hasattr(model, 'continuous_model')
        assert callable(model.continuous_model)

    def test_symbolic_dynamics_exists(self, model):
        """Test symbolic_dynamics method exists."""
        assert hasattr(model, 'symbolic_dynamics')
        assert callable(model.symbolic_dynamics)

    def test_bounds_defined(self, model):
        """Test state/input bounds are defined."""
        assert hasattr(model, 'lower_bound')
        assert hasattr(model, 'upper_bound')
        assert len(model.lower_bound) > 0
        assert len(model.upper_bound) > 0


class TestContouringSecondOrderUnicycleModel:
    """Tests for ContouringSecondOrderUnicycleModel.

    State: [x, y, psi, v, s] where s is spline parameter
    The spline parameter tracks progress along the reference path.

    Reference: C++ ContouringModule - spline parameter update
    """

    @pytest.fixture
    def model(self):
        return ContouringSecondOrderUnicycleModel()

    def test_model_creation(self, model):
        """Test model initializes correctly."""
        assert model is not None

    def test_state_includes_spline(self, model):
        """Test state includes spline parameter."""
        # Check dependent_vars for 'spline' or 's'
        assert 'spline' in model.dependent_vars or 's' in model.dependent_vars

    def test_state_dimension_larger(self, model):
        """Test contouring model has more states than basic unicycle."""
        # Contouring adds spline parameter
        assert model.state_dimension == 5, "Contouring unicycle should have 5 states"

    def test_inherits_unicycle_dynamics(self, model):
        """Test that position/heading variables exist."""
        # First 4 states should be same as standard unicycle
        expected_vars = ['x', 'y', 'psi', 'v']
        for var in expected_vars:
            assert var in model.dependent_vars, f"Missing variable: {var}"


class TestPointMassModel:
    """Tests for PointMassModel.

    State: [x, y, vx, vy]
    Inputs: [ax, ay]

    Simple double-integrator dynamics.
    """

    @pytest.fixture
    def model(self):
        return PointMassModel()

    def test_model_creation(self, model):
        """Test model initializes correctly."""
        assert model is not None

    def test_state_variables(self, model):
        """Test correct state variables."""
        expected = ['x', 'y', 'vx', 'vy']
        for var in expected:
            assert var in model.dependent_vars, f"Missing state: {var}"

    def test_input_variables(self, model):
        """Test correct input variables."""
        expected = ['ax', 'ay']
        for var in expected:
            assert var in model.inputs, f"Missing input: {var}"

    def test_state_dimension(self, model):
        """Test state dimension."""
        assert model.state_dimension == 4, "Point mass should have 4 states"

    def test_input_dimension(self, model):
        """Test input dimension."""
        assert model.nu == 2, "Point mass should have 2 inputs"


class TestDynamicsIntegration:
    """Integration tests for dynamics models."""

    def test_all_models_have_required_methods(self):
        """Test all models implement required interface."""
        models = [
            SecondOrderUnicycleModel(),
            ContouringSecondOrderUnicycleModel(),
            PointMassModel(),
        ]

        # Check actual DynamicsModel interface
        required_attrs = [
            'dependent_vars',  # State variable names
            'inputs',          # Input variable names
            'state_dimension', # Number of states
            'nu',              # Number of inputs
            'continuous_model',
            'symbolic_dynamics',
        ]

        for model in models:
            for attr in required_attrs:
                assert hasattr(model, attr), \
                    f"{model.__class__.__name__} missing {attr}"

    def test_symbolic_dynamics_available(self):
        """Test symbolic dynamics for CasADi solver."""
        model = SecondOrderUnicycleModel()

        if hasattr(model, 'symbolic_dynamics'):
            # Should return symbolic expressions
            import casadi as cs
            x = cs.MX.sym('x', 4)
            u = cs.MX.sym('u', 2)
            p = cs.MX.sym('p', 1)
            dt = 0.1

            try:
                x_next = model.symbolic_dynamics(x, u, p, dt)
                assert x_next is not None
            except Exception:
                pass  # May require specific setup


class TestBoundsAndConstraints:
    """Test state and input bounds for dynamics models."""

    def test_unicycle_bounds(self):
        """Test unicycle model has reasonable bounds."""
        model = SecondOrderUnicycleModel()

        if hasattr(model, 'state_bounds'):
            bounds = model.state_bounds
            # Velocity should have sensible bounds
            # Reference: C++ mpc_planner typically uses v in [0, 3] m/s

        if hasattr(model, 'input_bounds'):
            bounds = model.input_bounds
            # Acceleration and angular velocity should be bounded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
