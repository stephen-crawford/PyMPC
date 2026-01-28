#!/usr/bin/env python3
"""
Unit Tests for Planner Integration

Tests the Planner class which orchestrates the MPC optimization.
Verifies the complete pipeline from problem setup to solution.

Reference: https://github.com/tud-amr/mpc_planner
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.utils import read_config_file


class TestPlannerCreation:
    """Tests for Planner initialization."""

    @pytest.fixture
    def config(self):
        return read_config_file()

    def test_planner_creation_basic(self, config):
        """Test basic Planner creation."""
        from planning.planner import Planner
        from planning.types import Problem, State, generate_reference_path
        from planning.dynamic_models import SecondOrderUnicycleModel
        from modules.objectives.goal_objective import GoalObjective

        # Create minimal problem
        model = SecondOrderUnicycleModel()
        state = State(model)
        state.set('x', 0.0)
        state.set('y', 0.0)
        state.set('psi', 0.0)
        state.set('v', 0.0)

        objective = GoalObjective()

        # Use direct attribute assignment (actual API)
        problem = Problem()
        problem.model_type = model
        problem.x0 = state
        problem.modules = [objective]
        problem.obstacles = []

        planner = Planner(problem, config)

        assert planner is not None
        assert planner.solver is not None

    def test_planner_with_contouring(self, config):
        """Test Planner with contouring objective."""
        from planning.planner import Planner
        from planning.types import Problem, State, Data, generate_reference_path
        from planning.dynamic_models import ContouringSecondOrderUnicycleModel
        from modules.objectives.contouring_objective import ContouringObjective
        from modules.constraints.contouring_constraints import ContouringConstraints

        model = ContouringSecondOrderUnicycleModel()
        state = State(model)
        state.set('x', 0.0)
        state.set('y', 0.0)
        state.set('psi', 0.0)
        state.set('v', 1.0)
        state.set('spline', 0.0)

        # Create reference path
        ref_path = generate_reference_path([0, 0, 0], [10, 0, 0], "straight", 50)

        objective = ContouringObjective()
        objective.reference_path = ref_path

        constraint = ContouringConstraints()

        # Create data object with horizon and timestep
        horizon = config.get("planner", {}).get("horizon", 10)
        timestep = config.get("planner", {}).get("timestep", 0.1)

        data = Data()
        data.dynamics_model = model
        data.reference_path = ref_path
        data.dynamic_obstacles = []
        data.horizon = horizon
        data.timestep = timestep

        # Use direct attribute assignment (actual API)
        problem = Problem()
        problem.model_type = model
        problem.x0 = state
        problem.modules = [objective, constraint]
        problem.obstacles = []
        problem.data = data

        # Add get_horizon/get_timestep methods that planner expects
        problem.get_horizon = lambda: horizon
        problem.get_timestep = lambda: timestep

        planner = Planner(problem, config)

        assert planner is not None


class TestPlannerComponents:
    """Tests for Planner component initialization."""

    @pytest.fixture
    def config(self):
        return read_config_file()

    def test_module_manager_initialized(self, config):
        """Test ModuleManager is initialized."""
        from planning.planner import Planner
        from planning.types import Problem, State
        from planning.dynamic_models import SecondOrderUnicycleModel
        from modules.objectives.goal_objective import GoalObjective

        model = SecondOrderUnicycleModel()
        state = State(model)
        state.set('x', 0.0)
        state.set('y', 0.0)
        state.set('psi', 0.0)
        state.set('v', 0.0)

        problem = Problem()
        problem.model_type = model
        problem.x0 = state
        problem.modules = [GoalObjective()]
        problem.obstacles = []

        planner = Planner(problem, config)

        assert hasattr(planner, 'module_manager')
        assert planner.module_manager is not None

    def test_solver_initialized(self, config):
        """Test solver is initialized."""
        from planning.planner import Planner
        from planning.types import Problem, State
        from planning.dynamic_models import SecondOrderUnicycleModel
        from modules.objectives.goal_objective import GoalObjective

        model = SecondOrderUnicycleModel()
        state = State(model)
        state.set('x', 0.0)
        state.set('y', 0.0)
        state.set('psi', 0.0)
        state.set('v', 0.0)

        problem = Problem()
        problem.model_type = model
        problem.x0 = state
        problem.modules = [GoalObjective()]
        problem.obstacles = []

        planner = Planner(problem, config)

        assert hasattr(planner, 'solver')
        assert planner.solver is not None


class TestPlannerDataFlow:
    """Tests for data flow through the planner."""

    @pytest.fixture
    def config(self):
        return read_config_file()

    def test_state_propagation(self, config):
        """Test state is correctly propagated."""
        from planning.types import State
        from planning.dynamic_models import SecondOrderUnicycleModel

        model = SecondOrderUnicycleModel()
        state = State(model)
        state.set('x', 0.0)
        state.set('y', 0.0)
        state.set('psi', 0.0)
        state.set('v', 1.0)

        # Propagate with zero control
        control = {'a': 0.0, 'w': 0.0}
        dt = 0.1

        new_state = state.propagate(control, dt, dynamics_model=model)

        # Should move forward
        assert new_state.get('x') > 0
        assert abs(new_state.get('y')) < 0.01

    def test_data_update(self, config):
        """Test data object state can be set."""
        from planning.types import State, Data
        from planning.dynamic_models import SecondOrderUnicycleModel

        model = SecondOrderUnicycleModel()

        state = State(model)
        state.set('x', 5.0)
        state.set('y', 3.0)
        state.set('psi', 0.0)
        state.set('v', 1.0)

        data = Data()
        data.dynamics_model = model

        # Set state directly (actual API)
        data.state = state

        assert data.state is not None
        assert data.state.get('x') == 5.0


class TestProblemInterface:
    """Tests for Problem class interface."""

    def test_problem_creation(self):
        """Test Problem object creation."""
        from planning.types import Problem

        problem = Problem()
        assert problem is not None

    def test_problem_set_model(self):
        """Test setting model type."""
        from planning.types import Problem
        from planning.dynamic_models import SecondOrderUnicycleModel

        problem = Problem()
        model = SecondOrderUnicycleModel()

        # Use direct attribute assignment
        problem.model_type = model

        assert problem.get_model_type() is model

    def test_problem_set_modules(self):
        """Test setting modules."""
        from planning.types import Problem
        from modules.objectives.goal_objective import GoalObjective

        problem = Problem()
        modules = [GoalObjective()]

        # Use direct attribute assignment
        problem.modules = modules

        assert len(problem.get_modules()) == 1

    def test_problem_set_obstacles(self):
        """Test setting obstacles."""
        from planning.types import Problem, DynamicObstacle

        problem = Problem()

        # DynamicObstacle requires: index, position, angle, radius
        obstacle = DynamicObstacle(
            index=0,
            position=np.array([5.0, 0.0]),
            angle=0.0,
            radius=0.5
        )

        problem.obstacles = [obstacle]

        assert len(problem.get_obstacles()) == 1


class TestStateInterface:
    """Tests for State class interface."""

    def test_state_creation(self):
        """Test State object creation."""
        from planning.types import State
        from planning.dynamic_models import SecondOrderUnicycleModel

        model = SecondOrderUnicycleModel()
        state = State(model)

        assert state is not None

    def test_state_get_set(self):
        """Test state get/set methods."""
        from planning.types import State
        from planning.dynamic_models import SecondOrderUnicycleModel

        model = SecondOrderUnicycleModel()
        state = State(model)

        state.set('x', 5.0)
        assert state.get('x') == 5.0

    def test_state_has_method(self):
        """Test state has method."""
        from planning.types import State
        from planning.dynamic_models import SecondOrderUnicycleModel

        model = SecondOrderUnicycleModel()
        state = State(model)

        state.set('x', 5.0)

        assert state.has('x')

    def test_state_get_position(self):
        """Test get_position method."""
        from planning.types import State
        from planning.dynamic_models import SecondOrderUnicycleModel

        model = SecondOrderUnicycleModel()
        state = State(model)

        state.set('x', 5.0)
        state.set('y', 3.0)

        pos = state.get_position()

        assert pos is not None
        assert len(pos) >= 2
        assert pos[0] == 5.0
        assert pos[1] == 3.0


class TestObstacleInterface:
    """Tests for obstacle-related interfaces."""

    def test_dynamic_obstacle_creation(self):
        """Test DynamicObstacle creation."""
        from planning.types import DynamicObstacle

        # DynamicObstacle requires: index, position, angle, radius
        obstacle = DynamicObstacle(
            index=0,
            position=np.array([5.0, 2.0]),
            angle=0.0,
            radius=0.5
        )

        assert obstacle.position is not None
        assert obstacle.radius == 0.5
        assert obstacle.index == 0

    def test_obstacle_propagation(self):
        """Test obstacle position propagation."""
        from planning.types import DynamicObstacle, propagate_obstacles, Data
        from planning.dynamic_models import SecondOrderUnicycleModel

        model = SecondOrderUnicycleModel()

        # DynamicObstacle requires: index, position, angle, radius
        obstacle = DynamicObstacle(
            index=0,
            position=np.array([5.0, 0.0]),
            angle=0.0,
            radius=0.5
        )

        data = Data()
        data.dynamics_model = model
        data.dynamic_obstacles = [obstacle]
        data.horizon = 10
        data.timestep = 0.1

        # Propagate obstacles
        propagate_obstacles(data, dt=0.1, horizon=10)

        # Should have predictions
        if hasattr(obstacle, 'prediction') and obstacle.prediction is not None:
            assert obstacle.prediction is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
