"""
Unit tests for scenario_mpc core functionality.

Tests the mathematical formulation from guide.md.
"""

import numpy as np
import pytest
from typing import Dict

# Import modules to test
import sys
sys.path.insert(0, '/home/stephen/PyMPC')

from scenario_mpc.types import (
    EgoState,
    EgoInput,
    ObstacleState,
    ModeModel,
    ModeHistory,
    PredictionStep,
    ObstacleTrajectory,
    Scenario,
    TrajectoryMoments,
    CollisionConstraint,
    WeightType,
)
from scenario_mpc.config import ScenarioMPCConfig
from scenario_mpc.dynamics import EgoDynamics, create_obstacle_mode_models
from scenario_mpc.mode_weights import (
    compute_mode_weights,
    sample_mode_from_weights,
)
from scenario_mpc.trajectory_moments import (
    compute_trajectory_moments,
    compute_single_mode_trajectory,
)
from scenario_mpc.scenario_sampler import sample_scenarios
from scenario_mpc.collision_constraints import (
    compute_linearized_constraints,
    compute_ego_disc_positions,
)
from scenario_mpc.scenario_pruning import (
    prune_dominated_scenarios,
    remove_inactive_scenarios,
)


class TestTypes:
    """Test data structure types."""

    def test_ego_state_to_array(self):
        """Test EgoState to/from array conversion."""
        state = EgoState(x=1.0, y=2.0, theta=0.5, v=1.5)
        arr = state.to_array()

        assert arr.shape == (4,)
        assert np.allclose(arr, [1.0, 2.0, 0.5, 1.5])

        # Test round-trip
        state2 = EgoState.from_array(arr)
        assert state2.x == state.x
        assert state2.y == state.y
        assert state2.theta == state.theta
        assert state2.v == state.v

    def test_obstacle_state(self):
        """Test ObstacleState."""
        obs = ObstacleState(x=5.0, y=3.0, vx=1.0, vy=0.5)

        assert np.allclose(obs.position(), [5.0, 3.0])
        assert np.allclose(obs.velocity(), [1.0, 0.5])

    def test_mode_model_propagate(self):
        """Test ModeModel propagation."""
        dt = 0.1
        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        b = np.zeros(4)
        G = np.eye(4) * 0.1

        mode = ModeModel(mode_id="cv", A=A, b=b, G=G)
        state = ObstacleState(x=0, y=0, vx=1, vy=0)

        # Propagate without noise
        next_state = mode.propagate(state)

        assert np.isclose(next_state.x, dt)
        assert np.isclose(next_state.y, 0)
        assert np.isclose(next_state.vx, 1)

    def test_collision_constraint_evaluate(self):
        """Test collision constraint evaluation."""
        constraint = CollisionConstraint(
            k=0,
            obstacle_id=0,
            scenario_id=0,
            a=np.array([1.0, 0.0]),
            b=2.0
        )

        # Position satisfies constraint: 1*3 + 0*0 >= 2
        pos1 = np.array([3.0, 0.0])
        assert constraint.evaluate(pos1) > 0

        # Position violates constraint: 1*1 + 0*0 < 2
        pos2 = np.array([1.0, 0.0])
        assert constraint.evaluate(pos2) < 0


class TestConfig:
    """Test configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ScenarioMPCConfig()

        assert config.horizon == 20
        assert config.dt == 0.1
        assert config.confidence_level == 0.95
        assert config.epsilon == 0.05

    def test_config_validation(self):
        """Test configuration validation."""
        config = ScenarioMPCConfig()
        config.validate()  # Should not raise

        # Invalid horizon
        config.horizon = -1
        with pytest.raises(AssertionError):
            config.validate()

    def test_required_scenarios_computation(self):
        """Test Theorem 1 sample size computation."""
        config = ScenarioMPCConfig(confidence_level=0.95, beta=0.01)

        # For n_x = 10 decision variables
        S = config.compute_required_scenarios(10)

        # S >= 2/epsilon * (ln(1/beta) + n_x)
        # S >= 2/0.05 * (ln(100) + 10) = 40 * (4.6 + 10) = 584
        assert S >= 500


class TestDynamics:
    """Test dynamics models."""

    def test_ego_dynamics_propagation(self):
        """Test ego vehicle dynamics."""
        dynamics = EgoDynamics(dt=0.1)
        state = EgoState(x=0, y=0, theta=0, v=1)
        input = EgoInput(a=0, delta=0)

        next_state = dynamics.propagate(state, input)

        # Moving forward at v=1, theta=0, should advance in x
        assert next_state.x > state.x
        assert np.isclose(next_state.y, 0, atol=1e-6)
        assert np.isclose(next_state.theta, 0, atol=1e-6)
        assert np.isclose(next_state.v, 1, atol=1e-6)

    def test_ego_dynamics_turning(self):
        """Test ego turning dynamics."""
        dynamics = EgoDynamics(dt=0.1)
        state = EgoState(x=0, y=0, theta=0, v=1)
        input = EgoInput(a=0, delta=0.5)  # Turn left

        next_state = dynamics.propagate(state, input)

        # Should have turned
        assert next_state.theta > 0

    def test_obstacle_modes_creation(self):
        """Test standard mode model creation."""
        modes = create_obstacle_mode_models(dt=0.1)

        assert "constant_velocity" in modes
        assert "turn_left" in modes
        assert "turn_right" in modes

        # Check dimensions
        cv_mode = modes["constant_velocity"]
        assert cv_mode.A.shape == (4, 4)
        assert cv_mode.b.shape == (4,)


class TestModeWeights:
    """Test mode weight computation."""

    def test_uniform_weights(self):
        """Test uniform weight computation (Eq. 4)."""
        modes = create_obstacle_mode_models()
        history = ModeHistory(
            obstacle_id=0,
            available_modes=modes,
            observed_modes=[(0, "constant_velocity"), (1, "turn_left")]
        )

        weights = compute_mode_weights(history, weight_type=WeightType.UNIFORM)

        # All modes should have equal weight
        assert len(weights) == len(modes)
        expected_weight = 1.0 / len(modes)
        for w in weights.values():
            assert np.isclose(w, expected_weight, atol=1e-6)

    def test_frequency_weights(self):
        """Test frequency weight computation (Eq. 6)."""
        modes = create_obstacle_mode_models()
        history = ModeHistory(
            obstacle_id=0,
            available_modes=modes,
            observed_modes=[
                (0, "constant_velocity"),
                (1, "constant_velocity"),
                (2, "constant_velocity"),
                (3, "turn_left"),
            ]
        )

        weights = compute_mode_weights(history, weight_type=WeightType.FREQUENCY)

        # constant_velocity should have higher weight (3/4)
        assert weights["constant_velocity"] > weights["turn_left"]
        assert np.isclose(weights["constant_velocity"], 0.75, atol=1e-6)
        assert np.isclose(weights["turn_left"], 0.25, atol=1e-6)

    def test_recency_weights(self):
        """Test recency weight computation (Eq. 5)."""
        modes = create_obstacle_mode_models()
        history = ModeHistory(
            obstacle_id=0,
            available_modes=modes,
            observed_modes=[
                (0, "constant_velocity"),  # Old
                (8, "turn_left"),          # Recent
                (9, "turn_left"),          # Most recent
            ]
        )

        weights = compute_mode_weights(
            history,
            weight_type=WeightType.RECENCY,
            recency_decay=0.9,
            current_timestep=10
        )

        # Recent observations should have more weight
        assert weights["turn_left"] > weights["constant_velocity"]


class TestTrajectoryMoments:
    """Test trajectory moment computation (Proposition 1)."""

    def test_single_mode_trajectory(self):
        """Test single mode trajectory prediction."""
        modes = create_obstacle_mode_models()
        mode = modes["constant_velocity"]
        state = ObstacleState(x=0, y=0, vx=1, vy=0)

        traj = compute_single_mode_trajectory(state, mode, horizon=10)

        assert traj.horizon == 10
        assert len(traj.steps) == 11  # Including initial

        # Position should advance
        assert traj.steps[10].mean[0] > traj.steps[0].mean[0]

    def test_trajectory_moments_computation(self):
        """Test multi-modal moment computation."""
        modes = create_obstacle_mode_models()
        state = ObstacleState(x=0, y=0, vx=1, vy=0)
        weights = {"constant_velocity": 0.5, "turn_left": 0.5}

        moments = compute_trajectory_moments(
            state, weights,
            {k: modes[k] for k in weights},
            horizon=10
        )

        assert moments.horizon == 10
        assert moments.means.shape == (11, 2)
        assert moments.covariances.shape == (11, 2, 2)

        # Covariance should grow over time (uncertainty increases)
        cov_0 = np.trace(moments.covariances[0])
        cov_10 = np.trace(moments.covariances[10])
        assert cov_10 > cov_0


class TestScenarioSampling:
    """Test scenario sampling (Algorithm 1)."""

    def test_sample_scenarios(self):
        """Test scenario generation."""
        modes = create_obstacle_mode_models()
        obstacles = {
            0: ObstacleState(x=5, y=0, vx=-1, vy=0),
            1: ObstacleState(x=0, y=5, vx=0, vy=-1),
        }
        histories = {
            0: ModeHistory(0, modes, [(0, "constant_velocity")]),
            1: ModeHistory(1, modes, [(0, "turn_left")]),
        }

        scenarios = sample_scenarios(
            obstacles, histories,
            horizon=10,
            num_scenarios=5
        )

        assert len(scenarios) == 5
        for scenario in scenarios:
            assert len(scenario.trajectories) == 2
            assert 0 in scenario.trajectories
            assert 1 in scenario.trajectories


class TestCollisionConstraints:
    """Test collision constraint computation."""

    def test_ego_disc_positions(self):
        """Test ego disc position computation (Eq. 16)."""
        state = EgoState(x=0, y=0, theta=0, v=1)

        # Single disc
        positions = compute_ego_disc_positions(state, num_discs=1)
        assert len(positions) == 1
        assert np.allclose(positions[0], [0, 0])

        # Multiple discs
        positions = compute_ego_disc_positions(state, num_discs=3, vehicle_length=4.0)
        assert len(positions) == 3
        # Should be along x-axis (theta=0)
        assert positions[0][0] < positions[1][0] < positions[2][0]

    def test_linearized_constraints(self):
        """Test linearized constraint computation (Eq. 17-18)."""
        # Simple scenario with one obstacle
        ego_traj = [
            EgoState(x=0, y=0, theta=0, v=1),
            EgoState(x=1, y=0, theta=0, v=1),
        ]

        obs_steps = [
            PredictionStep(0, mean=np.array([2, 0]), covariance=np.eye(2)),
            PredictionStep(1, mean=np.array([1.5, 0]), covariance=np.eye(2)),
        ]
        obs_traj = ObstacleTrajectory(0, "cv", obs_steps)
        scenario = Scenario(0, {0: obs_traj})

        constraints = compute_linearized_constraints(
            ego_traj, [scenario],
            ego_radius=0.5, obstacle_radius=0.5
        )

        assert len(constraints) > 0

        # Constraints should be roughly pointing from obstacle to ego
        for c in constraints:
            # Normal should point away from obstacle
            assert c.a is not None
            assert c.b is not None


class TestScenarioPruning:
    """Test scenario pruning algorithms."""

    def test_prune_dominated(self):
        """Test geometric dominance pruning (Algorithm 3)."""
        # Create two scenarios - one dominates the other
        ego_traj = [EgoState(x=0, y=0, theta=0, v=1)]

        # Scenario 1: obstacle close
        obs1 = ObstacleTrajectory(
            0, "cv",
            [PredictionStep(0, np.array([2, 0]), np.eye(2))]
        )
        scenario1 = Scenario(0, {0: obs1})

        # Scenario 2: obstacle far (dominated)
        obs2 = ObstacleTrajectory(
            0, "cv",
            [PredictionStep(0, np.array([5, 0]), np.eye(2))]
        )
        scenario2 = Scenario(1, {0: obs2})

        pruned = prune_dominated_scenarios(
            [scenario1, scenario2],
            ego_traj,
            ego_radius=0.5,
            obstacle_radius=0.5
        )

        # Scenario 2 should be pruned (obstacle is farther)
        assert len(pruned) == 1
        assert pruned[0].scenario_id == 0


class TestIntegration:
    """Integration tests for full MPC pipeline."""

    def test_full_pipeline(self):
        """Test complete scenario MPC pipeline."""
        from scenario_mpc.mpc_controller import AdaptiveScenarioMPC

        config = ScenarioMPCConfig(
            horizon=10,
            num_scenarios=5,
            dt=0.1
        )

        controller = AdaptiveScenarioMPC(config)

        # Setup
        ego_state = EgoState(x=0, y=0, theta=0, v=0)
        obstacles = {0: ObstacleState(x=5, y=0, vx=-1, vy=0)}
        goal = np.array([10, 0])

        # Initialize obstacle
        controller.initialize_obstacle(0)
        controller.update_mode_observation(0, "constant_velocity")

        # Solve
        result = controller.solve(ego_state, obstacles, goal)

        assert result.ego_trajectory is not None
        assert len(result.ego_trajectory) == config.horizon + 1
        assert len(result.control_inputs) == config.horizon

        # First control should be valid
        first_input = result.first_input
        assert first_input is not None
        assert config.min_acceleration <= first_input.a <= config.max_acceleration


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
