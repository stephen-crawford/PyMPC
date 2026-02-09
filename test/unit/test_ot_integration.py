"""
Tests for OT Learning Integration + Epsilon Guarantee Fix.

Covers:
  1. compute_effective_epsilon formula verification
  2. SafeHorizonModule._verify_scenario_sufficiency
  3. enforce_all_scenarios flag in _process_scenarios_for_step
  4. WeightType.WASSERSTEIN existence and AdaptiveModeSampler creation with OT predictor
  5. WASSERSTEIN weight delegation to ot_predictor.compute_mode_weights
  6. OptimalTransportPredictor.estimate_mode_dynamics recovery
  7. AdaptiveModeSampler.update_mode_dynamics_from_ot updates ModeModel.b and ModeModel.G
"""

import os
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from types import ModuleType

# Mock casadi before any project imports (casadi is not installed in test env)
if 'casadi' not in sys.modules:
    _casadi_mock = ModuleType('casadi')
    _casadi_mock.MX = type('MX', (), {})
    _casadi_mock.SX = type('SX', (), {})
    _casadi_mock.DM = lambda x: float(x)
    sys.modules['casadi'] = _casadi_mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.constraints.scenario_utils.math_utils import compute_effective_epsilon, compute_sample_size
from modules.constraints.scenario_utils.sampler import WeightType, AdaptiveModeSampler, ModeModel
from modules.constraints.scenario_utils.optimal_transport_predictor import (
    OptimalTransportPredictor, TrajectoryObservation, TrajectoryBuffer,
    create_ot_predictor_with_standard_modes, OTWeightType
)


# =========================================================================
# Part 1: compute_effective_epsilon
# =========================================================================

class TestComputeEffectiveEpsilon:
    """Test the formula eps = 2*(ln(1/beta) + d + R) / S."""

    def test_basic_formula(self):
        S = 200
        beta = 0.01
        d = 60  # e.g., horizon=10, n_x=4, n_u=2 -> 10*4 + 10*2 = 60
        R = 0
        eps = compute_effective_epsilon(S, beta, d, R)
        expected = 2.0 * (np.log(1.0 / beta) + d + R) / S
        assert abs(eps - expected) < 1e-12

    def test_with_removal(self):
        S = 300
        beta = 0.05
        d = 40
        R = 5
        eps = compute_effective_epsilon(S, beta, d, R)
        expected = 2.0 * (np.log(1.0 / beta) + d + R) / S
        assert abs(eps - expected) < 1e-12

    def test_zero_samples_returns_one(self):
        eps = compute_effective_epsilon(0, 0.01, 60, 0)
        assert eps == 1.0

    def test_negative_samples_returns_one(self):
        eps = compute_effective_epsilon(-5, 0.01, 60, 0)
        assert eps == 1.0

    def test_invalid_beta_clamps(self):
        # beta=0 or beta=1 should be clamped to 0.01 internally
        eps = compute_effective_epsilon(200, 0.0, 60, 0)
        expected = 2.0 * (np.log(1.0 / 0.01) + 60 + 0) / 200
        assert abs(eps - expected) < 1e-12

    def test_larger_S_gives_smaller_epsilon(self):
        eps_small = compute_effective_epsilon(100, 0.01, 60, 0)
        eps_large = compute_effective_epsilon(500, 0.01, 60, 0)
        assert eps_large < eps_small


# =========================================================================
# Part 2: _verify_scenario_sufficiency
# =========================================================================

class TestEpsilonVerification:
    """Test SafeHorizonModule._verify_scenario_sufficiency."""

    def _make_module(self, epsilon_p=0.1, beta=0.01, n_bar=10,
                     num_removal=0, horizon_length=10, num_scenarios=100):
        """Create a SafeHorizonModule with a minimal mock config."""
        from modules.constraints.scenario_utils.scenario_module import SafeHorizonModule

        solver = MagicMock()
        config = {
            "epsilon_p": epsilon_p,
            "beta": beta,
            "n_bar": n_bar,
            "num_removal": num_removal,
            "horizon_length": horizon_length,
            "num_scenarios": num_scenarios,
            "robot_radius": 0.5,
            "max_constraints_per_disc": 5,
            "num_discs": 1,
        }
        module = SafeHorizonModule(solver, config)
        return module

    def test_sufficient_scenarios(self):
        module = self._make_module(epsilon_p=0.1, beta=0.01, horizon_length=10)
        # S_required for these params is ~1293, so use 1500 to be safe
        from planning.types import Scenario
        module.scenarios = [Scenario(i, 0) for i in range(1500)]

        is_sufficient, S_actual, S_required, eps_eff = module._verify_scenario_sufficiency()
        assert is_sufficient is True
        assert S_actual == 1500
        assert S_actual >= S_required
        assert eps_eff <= module.epsilon_p

    def test_insufficient_scenarios(self):
        module = self._make_module(epsilon_p=0.01, beta=0.01, horizon_length=10)
        # Give it very few scenarios
        from planning.types import Scenario
        module.scenarios = [Scenario(i, 0) for i in range(5)]

        is_sufficient, S_actual, S_required, eps_eff = module._verify_scenario_sufficiency()
        assert is_sufficient is False
        assert S_actual == 5
        assert S_actual < S_required
        assert eps_eff > module.epsilon_p


# =========================================================================
# Part 3: enforce_all_scenarios flag
# =========================================================================

class TestEnforceAllScenariosFlag:
    """Test that enforce_all_scenarios=True uses all step_scenarios."""

    def test_enforce_all_uses_all_scenarios(self):
        """When enforce_all_scenarios=True, _process_scenarios_for_step should
        pass all step_scenarios as support_scenarios (not just n_bar)."""
        from modules.constraints.scenario_utils.scenario_module import SafeHorizonModule
        from planning.types import Scenario

        solver = MagicMock()
        solver.warmstart_intiailized = False
        config = {
            "epsilon_p": 0.1,
            "beta": 0.01,
            "n_bar": 3,
            "num_removal": 0,
            "horizon_length": 5,
            "num_scenarios": 20,
            "robot_radius": 0.3,
            "max_constraints_per_disc": 50,
            "num_discs": 1,
            "enforce_all_scenarios": True,
        }
        module = SafeHorizonModule(solver, config)

        # Create 20 scenarios with trajectories
        scenarios = []
        for i in range(20):
            s = Scenario(i, 0)
            s.position = np.array([2.0 + 0.1 * i, 0.0])
            s.radius = 0.3
            s.trajectory = [np.array([2.0 + 0.1 * i, 0.0])] * 5
            scenarios.append(s)
        module.scenarios = scenarios

        ref_pos = np.array([0.0, 0.0])
        module._process_scenarios_for_step(0, 0, MagicMock(), ref_pos)

        # All 20 scenarios should produce constraints in the polytope
        polytope = module.disc_manager[0].polytopes[0]
        assert len(polytope.halfspaces) == 20

    def test_enforce_false_uses_support_set(self):
        """When enforce_all_scenarios=False, support set is limited to n_bar."""
        from modules.constraints.scenario_utils.scenario_module import SafeHorizonModule
        from planning.types import Scenario

        solver = MagicMock()
        solver.warmstart_intiailized = False
        config = {
            "epsilon_p": 0.1,
            "beta": 0.01,
            "n_bar": 3,
            "num_removal": 0,
            "horizon_length": 5,
            "num_scenarios": 20,
            "robot_radius": 0.3,
            "max_constraints_per_disc": 50,
            "num_discs": 1,
            "enforce_all_scenarios": False,
        }
        module = SafeHorizonModule(solver, config)

        scenarios = []
        for i in range(20):
            s = Scenario(i, 0)
            s.position = np.array([2.0 + 0.1 * i, 0.0])
            s.radius = 0.3
            s.trajectory = [np.array([2.0 + 0.1 * i, 0.0])] * 5
            scenarios.append(s)
        module.scenarios = scenarios

        ref_pos = np.array([0.0, 0.0])
        module._process_scenarios_for_step(0, 0, MagicMock(), ref_pos)

        polytope = module.disc_manager[0].polytopes[0]
        # Should be limited to n_bar = 3
        assert len(polytope.halfspaces) == 3


# =========================================================================
# Part 4: WeightType.WASSERSTEIN existence and sampler creation
# =========================================================================

class TestWassersteinWeightType:
    """Test that WeightType.WASSERSTEIN exists and works with AdaptiveModeSampler."""

    def test_wasserstein_enum_value(self):
        assert WeightType.WASSERSTEIN == WeightType("wasserstein")

    def test_create_sampler_with_ot_predictor(self):
        ot_pred = OptimalTransportPredictor(dt=0.1)
        sampler = AdaptiveModeSampler(
            num_scenarios=50,
            weight_type=WeightType.WASSERSTEIN,
            ot_predictor=ot_pred,
        )
        assert sampler.weight_type == WeightType.WASSERSTEIN
        assert sampler.ot_predictor is ot_pred


# =========================================================================
# Part 5: OT weight integration (WASSERSTEIN delegates to ot_predictor)
# =========================================================================

class TestOTWeightIntegration:
    """Test that WASSERSTEIN weight type delegates to ot_predictor.compute_mode_weights."""

    def test_delegates_to_ot_predictor(self):
        ot_pred = MagicMock()
        ot_pred.compute_mode_weights.return_value = {
            "constant_velocity": 0.7,
            "decelerating": 0.3,
        }

        sampler = AdaptiveModeSampler(
            num_scenarios=10,
            weight_type=WeightType.WASSERSTEIN,
            ot_predictor=ot_pred,
        )

        # Initialize mode history for obstacle 0
        sampler.update_mode_observation(0, "constant_velocity")
        sampler.update_mode_observation(0, "decelerating")

        weights = sampler.get_mode_weights_for_obstacle(0)

        # Should have called ot_predictor.compute_mode_weights
        ot_pred.compute_mode_weights.assert_called_once()
        assert weights == {"constant_velocity": 0.7, "decelerating": 0.3}

    def test_without_ot_predictor_falls_back(self):
        """WASSERSTEIN without ot_predictor should fall back to uniform."""
        sampler = AdaptiveModeSampler(
            num_scenarios=10,
            weight_type=WeightType.WASSERSTEIN,
            ot_predictor=None,
        )
        sampler.update_mode_observation(0, "constant_velocity")
        sampler.update_mode_observation(0, "decelerating")

        weights = sampler.get_mode_weights_for_obstacle(0)
        # Should fall back to compute_mode_weights with WASSERSTEIN branch (uniform fallback)
        assert len(weights) > 0
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9


# =========================================================================
# Part 6: estimate_mode_dynamics recovery
# =========================================================================

class TestDynamicsEstimation:
    """Test OptimalTransportPredictor.estimate_mode_dynamics with known data."""

    def test_recover_bias_from_known_dynamics(self):
        """Generate data from A=I, b=[0.1, 0, 0, 0], G=0.01*I,
        then check that estimate_mode_dynamics recovers b within tolerance."""
        dt = 0.1
        A = np.eye(4)
        b_true = np.array([0.1, 0.0, 0.0, 0.0])
        G_true = 0.01 * np.eye(4)

        predictor = OptimalTransportPredictor(dt=dt, buffer_size=500)

        rng = np.random.default_rng(42)
        state = np.array([0.0, 0.0, 1.0, 0.0])  # initial state
        mode_id = "test_mode"

        # Build trajectory buffer directly with known dynamics states
        buf = TrajectoryBuffer(obstacle_id=0, max_length=500)
        n_obs = 60
        for k in range(n_obs):
            noise = G_true @ rng.standard_normal(4)
            next_state = A @ state + b_true + noise

            # Create observation with the exact dynamics state [x, y, vx, vy]
            obs = TrajectoryObservation(
                timestep=k,
                position=state[:2].copy(),
                velocity=state[2:].copy(),
                acceleration=np.zeros(2),
                mode_id=mode_id,
            )
            buf.add_observation(obs)
            state = next_state

        # Add final observation
        buf.add_observation(TrajectoryObservation(
            timestep=n_obs,
            position=state[:2].copy(),
            velocity=state[2:].copy(),
            acceleration=np.zeros(2),
            mode_id=mode_id,
        ))

        predictor.trajectory_buffers[0] = buf

        result = predictor.estimate_mode_dynamics(0, mode_id, A, dt)
        assert result is not None, "estimate_mode_dynamics returned None (insufficient data?)"

        b_learned, G_learned = result
        # The learned b should be close to b_true within tolerance
        np.testing.assert_allclose(b_learned, b_true, atol=0.1)

    def test_returns_none_with_insufficient_data(self):
        predictor = OptimalTransportPredictor(dt=0.1)
        # Only 2 observations, need at least 5 residuals
        predictor.observe(0, np.array([0.0, 0.0]), mode_id="m")
        predictor.advance_timestep()
        predictor.observe(0, np.array([0.1, 0.0]), mode_id="m")
        predictor.advance_timestep()

        result = predictor.estimate_mode_dynamics(0, "m", np.eye(4), 0.1)
        assert result is None


# =========================================================================
# Part 7: update_mode_dynamics_from_ot updates ModeModel.b and ModeModel.G
# =========================================================================

class TestUpdateModeDynamicsFromOT:
    """Test AdaptiveModeSampler.update_mode_dynamics_from_ot."""

    def test_updates_mode_model_b_and_G(self):
        b_new = np.array([0.05, -0.02, 0.01, 0.0])
        G_new = 0.02 * np.eye(4)

        ot_pred = MagicMock()
        ot_pred.estimate_mode_dynamics.return_value = (b_new, G_new)

        sampler = AdaptiveModeSampler(
            num_scenarios=10,
            weight_type=WeightType.FREQUENCY,
            ot_predictor=ot_pred,
        )

        # Record mode so history exists
        sampler.update_mode_observation(0, "constant_velocity")

        # Record original values
        mode_model = sampler.mode_histories[0].available_modes["constant_velocity"]
        b_original = mode_model.b.copy()
        G_original = mode_model.G.copy()

        # Update dynamics from OT
        sampler.update_mode_dynamics_from_ot(0)

        # b and G should now be updated
        assert np.allclose(mode_model.b, b_new)
        assert np.allclose(mode_model.G, G_new)
        assert not np.allclose(mode_model.b, b_original)

    def test_no_update_without_ot_predictor(self):
        sampler = AdaptiveModeSampler(
            num_scenarios=10,
            weight_type=WeightType.FREQUENCY,
            ot_predictor=None,
        )
        sampler.update_mode_observation(0, "constant_velocity")
        mode_model = sampler.mode_histories[0].available_modes["constant_velocity"]
        b_before = mode_model.b.copy()

        sampler.update_mode_dynamics_from_ot(0)
        # Should be unchanged
        assert np.allclose(mode_model.b, b_before)

    def test_no_update_when_estimate_returns_none(self):
        ot_pred = MagicMock()
        ot_pred.estimate_mode_dynamics.return_value = None

        sampler = AdaptiveModeSampler(
            num_scenarios=10,
            weight_type=WeightType.FREQUENCY,
            ot_predictor=ot_pred,
        )
        sampler.update_mode_observation(0, "constant_velocity")
        mode_model = sampler.mode_histories[0].available_modes["constant_velocity"]
        b_before = mode_model.b.copy()

        sampler.update_mode_dynamics_from_ot(0)
        assert np.allclose(mode_model.b, b_before)
