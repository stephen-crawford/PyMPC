#!/usr/bin/env python3
"""
Pytest Configuration and Shared Fixtures

Provides common fixtures for PyMPC tests.
"""

import sys
import os
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return os.path.dirname(os.path.dirname(__file__))


@pytest.fixture(scope="session")
def config():
    """Load the default configuration."""
    from utils.utils import read_config_file
    return read_config_file()


@pytest.fixture
def simple_reference_path():
    """Create a simple straight reference path."""
    from planning.types import generate_reference_path

    start = [0.0, 0.0, 0.0]
    goal = [10.0, 0.0, 0.0]
    return generate_reference_path(start, goal, "straight", num_points=50)


@pytest.fixture
def curved_reference_path():
    """Create a curved reference path."""
    from planning.types import generate_reference_path

    start = [0.0, 0.0, 0.0]
    goal = [15.0, 10.0, 0.0]
    return generate_reference_path(start, goal, "curved", num_points=50)


@pytest.fixture
def unicycle_model():
    """Create a SecondOrderUnicycleModel."""
    from planning.dynamic_models import SecondOrderUnicycleModel
    return SecondOrderUnicycleModel()


@pytest.fixture
def contouring_unicycle_model():
    """Create a ContouringSecondOrderUnicycleModel."""
    from planning.dynamic_models import ContouringSecondOrderUnicycleModel
    return ContouringSecondOrderUnicycleModel()


@pytest.fixture
def point_mass_model():
    """Create a PointMassModel."""
    from planning.dynamic_models import PointMassModel
    return PointMassModel()


@pytest.fixture
def casadi_solver(config):
    """Create a CasADiSolver."""
    from solver.casadi_solver import CasADiSolver
    return CasADiSolver(config)


@pytest.fixture
def sample_state(unicycle_model):
    """Create a sample state."""
    from planning.types import State

    state = State(unicycle_model)
    state.set('x', 0.0)
    state.set('y', 0.0)
    state.set('psi', 0.0)
    state.set('v', 1.0)
    return state


@pytest.fixture
def sample_data(unicycle_model, simple_reference_path, config):
    """Create sample Data object."""
    from planning.types import Data

    data = Data()
    data.dynamics_model = unicycle_model
    data.reference_path = simple_reference_path
    data.dynamic_obstacles = []
    data.horizon = config.get("planner", {}).get("horizon", 10)
    data.timestep = config.get("planner", {}).get("timestep", 0.1)
    return data


@pytest.fixture
def sample_obstacle():
    """Create a sample dynamic obstacle."""
    from planning.types import DynamicObstacle

    obstacle = DynamicObstacle()
    obstacle.position = np.array([5.0, 2.0])
    obstacle.velocity = np.array([0.5, 0.0])
    obstacle.radius = 0.5
    return obstacle


# Test markers
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )


# Timeout for tests
def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on naming conventions."""
    for item in items:
        # Add slow marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)

        # Add performance marker
        if "performance" in item.name.lower() or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.performance)
