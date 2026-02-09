#!/usr/bin/env python3
"""
Unit Tests for Optimal Transport Predictor.

Tests the optimal transport-based statistical learning of obstacle dynamics:
- Sinkhorn algorithm for Wasserstein distance computation
- Empirical distribution handling
- Wasserstein barycenter computation
- Mode weight computation
- Trajectory prediction
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.constraints.scenario_utils.optimal_transport_predictor import (
    OptimalTransportPredictor,
    OTWeightType,
    EmpiricalDistribution,
    TrajectoryBuffer,
    TrajectoryObservation,
    ModeDistribution,
    compute_cost_matrix,
    sinkhorn_algorithm,
    wasserstein_distance,
    wasserstein_barycenter,
    create_ot_predictor_with_standard_modes,
)


class TestEmpiricalDistribution:
    """Tests for EmpiricalDistribution class."""

    def test_creation_from_samples(self):
        """Test creating distribution from samples."""
        samples = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        dist = EmpiricalDistribution.from_samples(samples)

        assert dist.n_samples == 4
        assert dist.dim == 2
        assert not dist.is_empty()
        np.testing.assert_array_almost_equal(dist.weights, [0.25, 0.25, 0.25, 0.25])

    def test_weighted_samples(self):
        """Test creating distribution with custom weights."""
        samples = np.array([[0, 0], [1, 0], [0, 1]])
        weights = np.array([0.5, 0.25, 0.25])
        dist = EmpiricalDistribution.from_samples(samples, weights)

        np.testing.assert_array_almost_equal(dist.weights, [0.5, 0.25, 0.25])

    def test_mean_computation(self):
        """Test weighted mean computation."""
        samples = np.array([[0, 0], [2, 0], [0, 2], [2, 2]])
        dist = EmpiricalDistribution.from_samples(samples)

        expected_mean = np.array([1.0, 1.0])
        np.testing.assert_array_almost_equal(dist.mean, expected_mean)

    def test_weighted_mean(self):
        """Test weighted mean with non-uniform weights."""
        samples = np.array([[0, 0], [4, 0]])
        weights = np.array([0.75, 0.25])
        dist = EmpiricalDistribution.from_samples(samples, weights)

        expected_mean = np.array([1.0, 0.0])
        np.testing.assert_array_almost_equal(dist.mean, expected_mean)

    def test_empty_distribution(self):
        """Test empty distribution handling."""
        samples = np.array([]).reshape(0, 2)
        dist = EmpiricalDistribution.from_samples(samples)

        assert dist.is_empty()
        assert dist.n_samples == 0

    def test_covariance_computation(self):
        """Test covariance matrix computation."""
        # Create samples with known covariance
        np.random.seed(42)
        mean = [0, 0]
        cov = [[1, 0.5], [0.5, 1]]
        samples = np.random.multivariate_normal(mean, cov, size=1000)

        dist = EmpiricalDistribution.from_samples(samples)

        # Computed covariance should be close to true covariance
        np.testing.assert_array_almost_equal(dist.covariance, cov, decimal=1)


class TestCostMatrix:
    """Tests for cost matrix computation."""

    def test_cost_matrix_squared_euclidean(self):
        """Test squared Euclidean cost matrix."""
        source = np.array([[0, 0], [1, 0]])
        target = np.array([[0, 0], [0, 1], [1, 1]])

        cost = compute_cost_matrix(source, target, p=2)

        # Expected costs:
        # source[0] to target: [0, 1, 2]
        # source[1] to target: [1, 2, 1]
        expected = np.array([[0, 1, 2], [1, 2, 1]])
        np.testing.assert_array_almost_equal(cost, expected)

    def test_cost_matrix_symmetry(self):
        """Test that cost matrix is symmetric for same source and target."""
        points = np.array([[0, 0], [1, 1], [2, 0]])
        cost = compute_cost_matrix(points, points, p=2)

        np.testing.assert_array_almost_equal(cost, cost.T)

    def test_cost_matrix_zero_diagonal(self):
        """Test that diagonal is zero when source equals target."""
        points = np.array([[0, 0], [1, 1], [2, 0]])
        cost = compute_cost_matrix(points, points, p=2)

        np.testing.assert_array_almost_equal(np.diag(cost), np.zeros(3))


class TestSinkhornAlgorithm:
    """Tests for Sinkhorn algorithm."""

    def test_sinkhorn_basic(self):
        """Test basic Sinkhorn computation."""
        # Simple 2x2 case
        a = np.array([0.5, 0.5])
        b = np.array([0.5, 0.5])
        cost = np.array([[0, 1], [1, 0]])

        plan, distance = sinkhorn_algorithm(a, b, cost, epsilon=0.1)

        # Transport plan should be doubly stochastic
        np.testing.assert_array_almost_equal(plan.sum(axis=0), b, decimal=3)
        np.testing.assert_array_almost_equal(plan.sum(axis=1), a, decimal=3)

    def test_sinkhorn_convergence(self):
        """Test that Sinkhorn converges for random inputs."""
        np.random.seed(42)
        n, m = 10, 15

        a = np.random.dirichlet(np.ones(n))
        b = np.random.dirichlet(np.ones(m))
        cost = np.random.rand(n, m)

        plan, distance = sinkhorn_algorithm(a, b, cost, epsilon=0.1, max_iterations=200)

        # Check marginal constraints
        np.testing.assert_array_almost_equal(plan.sum(axis=0), b, decimal=3)
        np.testing.assert_array_almost_equal(plan.sum(axis=1), a, decimal=3)

        # Distance should be non-negative
        assert distance >= 0

    def test_sinkhorn_identical_distributions(self):
        """Test Sinkhorn for identical distributions gives near-zero distance."""
        samples = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        cost = compute_cost_matrix(samples, samples, p=2)

        plan, distance = sinkhorn_algorithm(weights, weights, cost, epsilon=0.01)

        # Distance should be very small for identical distributions
        assert distance < 0.1

    def test_sinkhorn_empty_inputs(self):
        """Test Sinkhorn handles empty inputs."""
        plan, distance = sinkhorn_algorithm(
            np.array([]), np.array([]),
            np.array([]).reshape(0, 0),
            epsilon=0.1
        )

        assert plan.shape == (0, 0)
        assert distance == 0.0


class TestWassersteinDistance:
    """Tests for Wasserstein distance computation."""

    def test_wasserstein_identical_distributions(self):
        """Test W distance is zero for identical distributions."""
        samples = np.array([[0, 0], [1, 1], [2, 2]])
        dist = EmpiricalDistribution.from_samples(samples)

        w_dist = wasserstein_distance(dist, dist)

        assert w_dist < 0.1  # Should be near zero

    def test_wasserstein_different_distributions(self):
        """Test W distance is positive for different distributions."""
        samples1 = np.array([[0, 0], [1, 0]])
        samples2 = np.array([[10, 10], [11, 10]])

        dist1 = EmpiricalDistribution.from_samples(samples1)
        dist2 = EmpiricalDistribution.from_samples(samples2)

        w_dist = wasserstein_distance(dist1, dist2)

        # Distance should be significant
        assert w_dist > 5.0

    def test_wasserstein_empty_distribution(self):
        """Test W distance returns 0 for empty distributions."""
        dist1 = EmpiricalDistribution.from_samples(np.array([[0, 0]]))
        dist2 = EmpiricalDistribution.from_samples(np.array([]).reshape(0, 2))

        w_dist = wasserstein_distance(dist1, dist2)
        assert w_dist == 0.0

    def test_wasserstein_triangle_inequality(self):
        """Test triangle inequality: W(a,c) <= W(a,b) + W(b,c)."""
        np.random.seed(42)

        dist_a = EmpiricalDistribution.from_samples(
            np.random.randn(50, 2)
        )
        dist_b = EmpiricalDistribution.from_samples(
            np.random.randn(50, 2) + 2
        )
        dist_c = EmpiricalDistribution.from_samples(
            np.random.randn(50, 2) + 4
        )

        w_ab = wasserstein_distance(dist_a, dist_b)
        w_bc = wasserstein_distance(dist_b, dist_c)
        w_ac = wasserstein_distance(dist_a, dist_c)

        # Triangle inequality (with some tolerance for numerical errors)
        assert w_ac <= w_ab + w_bc + 0.5


class TestWassersteinBarycenter:
    """Tests for Wasserstein barycenter computation."""

    def test_barycenter_single_distribution(self):
        """Test barycenter of single distribution equals that distribution."""
        samples = np.array([[0, 0], [1, 0], [0, 1]])
        dist = EmpiricalDistribution.from_samples(samples)

        barycenter = wasserstein_barycenter([dist], [1.0], n_support=3)

        # Mean should be similar
        np.testing.assert_array_almost_equal(
            barycenter.mean, dist.mean, decimal=1
        )

    def test_barycenter_two_distributions(self):
        """Test barycenter of two distributions is between them."""
        samples1 = np.zeros((20, 2))
        samples2 = np.ones((20, 2)) * 4

        dist1 = EmpiricalDistribution.from_samples(samples1)
        dist2 = EmpiricalDistribution.from_samples(samples2)

        barycenter = wasserstein_barycenter(
            [dist1, dist2], [0.5, 0.5], n_support=20
        )

        # Mean should be at midpoint (approximately)
        expected_mean = np.array([2.0, 2.0])
        np.testing.assert_array_almost_equal(
            barycenter.mean, expected_mean, decimal=0
        )

    def test_barycenter_weighted(self):
        """Test weighted barycenter is closer to higher-weighted distribution."""
        samples1 = np.zeros((20, 2))
        samples2 = np.ones((20, 2)) * 4

        dist1 = EmpiricalDistribution.from_samples(samples1)
        dist2 = EmpiricalDistribution.from_samples(samples2)

        # Weight more heavily toward dist1
        barycenter = wasserstein_barycenter(
            [dist1, dist2], [0.8, 0.2], n_support=20
        )

        # Mean should be closer to [0, 0]
        assert barycenter.mean[0] < 2.0
        assert barycenter.mean[1] < 2.0

    def test_barycenter_empty_input(self):
        """Test barycenter handles empty input."""
        barycenter = wasserstein_barycenter([], [], n_support=10)

        assert barycenter.is_empty()


class TestTrajectoryBuffer:
    """Tests for TrajectoryBuffer class."""

    def test_buffer_creation(self):
        """Test buffer initialization."""
        buffer = TrajectoryBuffer(obstacle_id=0, max_length=100)

        assert buffer.obstacle_id == 0
        assert len(buffer) == 0

    def test_buffer_add_observation(self):
        """Test adding observations to buffer."""
        buffer = TrajectoryBuffer(obstacle_id=0, max_length=100)

        obs = TrajectoryObservation(
            timestep=0,
            position=np.array([1.0, 2.0]),
            velocity=np.array([0.1, 0.2]),
            acceleration=np.array([0.01, 0.02])
        )

        buffer.add_observation(obs)

        assert len(buffer) == 1

    def test_buffer_max_length(self):
        """Test buffer respects max length."""
        buffer = TrajectoryBuffer(obstacle_id=0, max_length=5)

        for i in range(10):
            obs = TrajectoryObservation(
                timestep=i,
                position=np.array([float(i), 0.0]),
                velocity=np.array([1.0, 0.0]),
                acceleration=np.array([0.0, 0.0])
            )
            buffer.add_observation(obs)

        assert len(buffer) == 5
        # Should have most recent observations
        recent = buffer.get_recent(5)
        assert recent[0].timestep == 5

    def test_buffer_get_velocity_samples(self):
        """Test extracting velocity samples."""
        buffer = TrajectoryBuffer(obstacle_id=0, max_length=100)

        for i in range(5):
            obs = TrajectoryObservation(
                timestep=i,
                position=np.array([float(i), 0.0]),
                velocity=np.array([float(i), float(i * 2)]),
                acceleration=np.array([0.0, 0.0])
            )
            buffer.add_observation(obs)

        velocities = buffer.get_velocity_samples()

        assert velocities.shape == (5, 2)
        np.testing.assert_array_equal(velocities[0], [0.0, 0.0])
        np.testing.assert_array_equal(velocities[4], [4.0, 8.0])


class TestOptimalTransportPredictor:
    """Tests for OptimalTransportPredictor class."""

    @pytest.fixture
    def predictor(self):
        """Create a basic predictor for testing."""
        return OptimalTransportPredictor(
            dt=0.1,
            buffer_size=100,
            sinkhorn_epsilon=0.1,
            min_samples_for_ot=5
        )

    @pytest.fixture
    def predictor_with_modes(self):
        """Create predictor with standard mode references."""
        return create_ot_predictor_with_standard_modes(dt=0.1, base_speed=1.0)

    def test_predictor_creation(self, predictor):
        """Test predictor initializes correctly."""
        assert predictor is not None
        assert predictor.dt == 0.1
        assert predictor.buffer_size == 100

    def test_observe_creates_buffer(self, predictor):
        """Test observation creates trajectory buffer."""
        predictor.observe(obstacle_id=0, position=np.array([1.0, 2.0]))

        assert 0 in predictor.trajectory_buffers
        assert len(predictor.trajectory_buffers[0]) == 1

    def test_observe_computes_velocity(self, predictor):
        """Test observation computes velocity from positions."""
        predictor.observe(0, np.array([0.0, 0.0]))
        predictor.advance_timestep()
        predictor.observe(0, np.array([0.1, 0.0]))  # Moved 0.1 in 0.1s = 1.0 m/s

        buffer = predictor.trajectory_buffers[0]
        velocities = buffer.get_velocity_samples()

        # Second observation should have velocity ~[1.0, 0.0]
        np.testing.assert_array_almost_equal(velocities[1], [1.0, 0.0], decimal=1)

    def test_observe_with_mode(self, predictor):
        """Test observation with mode label creates mode distribution."""
        predictor.observe(0, np.array([0.0, 0.0]), mode_id="constant_velocity")
        predictor.advance_timestep()
        predictor.observe(0, np.array([0.1, 0.0]), mode_id="constant_velocity")

        assert 0 in predictor.mode_distributions
        assert "constant_velocity" in predictor.mode_distributions[0]
        assert predictor.mode_distributions[0]["constant_velocity"].observation_count == 2

    def test_mode_weights_uniform(self, predictor):
        """Test uniform mode weights."""
        predictor.weight_type = OTWeightType.UNIFORM
        modes = ["mode_a", "mode_b", "mode_c"]

        weights = predictor.compute_mode_weights(0, modes)

        assert len(weights) == 3
        for w in weights.values():
            assert abs(w - 1/3) < 0.01

    def test_mode_weights_wasserstein(self, predictor_with_modes):
        """Test Wasserstein-based mode weights."""
        predictor = predictor_with_modes

        # Add observations consistent with constant velocity mode
        for i in range(20):
            predictor.observe(0, np.array([float(i) * 0.1, 0.0]), mode_id="constant_velocity")
            predictor.advance_timestep()

        modes = ["constant_velocity", "turn_left", "turn_right"]
        weights = predictor.compute_mode_weights(0, modes)

        # Constant velocity should have highest weight
        assert weights["constant_velocity"] > weights["turn_left"]
        assert weights["constant_velocity"] > weights["turn_right"]

    def test_predict_trajectory_constant_velocity(self, predictor):
        """Test trajectory prediction with constant velocity fallback."""
        position = np.array([0.0, 0.0])
        velocity = np.array([1.0, 0.0])
        horizon = 10

        predictions = predictor.predict_trajectory(
            obstacle_id=0,
            current_position=position,
            current_velocity=velocity,
            horizon=horizon
        )

        assert len(predictions) == horizon + 1
        assert predictions[0].position[0] == 0.0

        # After 10 steps at dt=0.1, v=1.0, should move ~1.0
        assert abs(predictions[-1].position[0] - 1.0) < 0.2

    def test_predict_trajectory_uncertainty_grows(self, predictor):
        """Test that prediction uncertainty grows over horizon."""
        position = np.array([0.0, 0.0])
        velocity = np.array([1.0, 0.0])
        horizon = 10

        predictions = predictor.predict_trajectory(
            obstacle_id=0,
            current_position=position,
            current_velocity=velocity,
            horizon=horizon
        )

        # Uncertainty should increase
        assert predictions[-1].major_radius > predictions[0].major_radius

    def test_prediction_error_computation(self, predictor):
        """Test prediction error (Wasserstein distance) computation."""
        predicted = [np.array([0, 0]), np.array([1, 0]), np.array([2, 0])]
        actual = [np.array([0, 0]), np.array([1, 0]), np.array([2, 0])]

        error = predictor.compute_prediction_error(0, predicted, actual)

        # Identical trajectories should have near-zero error
        assert error < 0.1

    def test_prediction_error_nonzero(self, predictor):
        """Test prediction error is nonzero for different trajectories."""
        predicted = [np.array([0, 0]), np.array([1, 0]), np.array([2, 0])]
        actual = [np.array([0, 1]), np.array([1, 1]), np.array([2, 1])]

        error = predictor.compute_prediction_error(0, predicted, actual)

        # Different trajectories should have positive error
        assert error > 0.5

    def test_adapt_uncertainty(self, predictor):
        """Test uncertainty adaptation based on prediction error."""
        # Low error -> low multiplier
        mult_low = predictor.adapt_uncertainty(0, prediction_error=0.1)

        # High error -> high multiplier
        mult_high = predictor.adapt_uncertainty(0, prediction_error=5.0)

        assert mult_high > mult_low

    def test_set_reference_distribution(self, predictor):
        """Test setting reference mode distribution."""
        velocities = np.random.randn(50, 2) + [1.0, 0.0]

        predictor.set_reference_distribution("test_mode", velocities)

        assert "test_mode" in predictor.reference_distributions
        mode_dist = predictor.reference_distributions["test_mode"]
        assert mode_dist.observation_count == 50

    def test_get_learned_modes(self, predictor):
        """Test getting learned modes for an obstacle."""
        # No observations yet
        modes = predictor.get_learned_modes(0)
        assert len(modes) == 0

        # Add observations with modes
        predictor.observe(0, np.array([0, 0]), mode_id="mode_a")
        predictor.advance_timestep()
        predictor.observe(0, np.array([0.1, 0]), mode_id="mode_b")

        modes = predictor.get_learned_modes(0)
        assert "mode_a" in modes
        assert "mode_b" in modes

    def test_reset(self, predictor):
        """Test reset clears learned data but keeps references."""
        # Add reference
        predictor.set_reference_distribution("ref_mode", np.random.randn(10, 2))

        # Add observations
        predictor.observe(0, np.array([0, 0]), mode_id="learned_mode")

        predictor.reset()

        # Learned data should be cleared
        assert len(predictor.trajectory_buffers) == 0
        assert len(predictor.mode_distributions) == 0

        # Reference should remain
        assert "ref_mode" in predictor.reference_distributions

    def test_reset_all(self, predictor):
        """Test full reset clears everything."""
        predictor.set_reference_distribution("ref_mode", np.random.randn(10, 2))
        predictor.observe(0, np.array([0, 0]))

        predictor.reset_all()

        assert len(predictor.trajectory_buffers) == 0
        assert len(predictor.reference_distributions) == 0

    def test_mode_distribution_stats(self, predictor):
        """Test getting mode distribution statistics."""
        # Add several observations for a mode
        for i in range(10):
            predictor.observe(0, np.array([float(i) * 0.1, 0.0]), mode_id="test_mode")
            predictor.advance_timestep()

        stats = predictor.get_mode_distribution_stats(0, "test_mode")

        assert stats is not None
        assert stats['mode_id'] == "test_mode"
        assert stats['observation_count'] == 10
        assert stats['velocity_mean'] is not None


class TestCreateOTPredictorWithStandardModes:
    """Tests for factory function."""

    def test_creates_predictor_with_modes(self):
        """Test factory creates predictor with standard mode references."""
        predictor = create_ot_predictor_with_standard_modes(dt=0.1, base_speed=1.0)

        expected_modes = [
            "constant_velocity", "decelerating", "accelerating",
            "turn_left", "turn_right",
            "lane_change_left", "lane_change_right"
        ]

        for mode in expected_modes:
            assert mode in predictor.reference_distributions

    def test_mode_distributions_have_samples(self):
        """Test mode distributions have proper samples."""
        predictor = create_ot_predictor_with_standard_modes(dt=0.1)

        for mode_id, mode_dist in predictor.reference_distributions.items():
            assert not mode_dist.velocity_dist.is_empty()
            assert mode_dist.observation_count > 0


class TestIntegrationWithMultipleObstacles:
    """Integration tests with multiple obstacles."""

    def test_multiple_obstacles_independent(self):
        """Test predictor handles multiple obstacles independently."""
        predictor = create_ot_predictor_with_standard_modes(dt=0.1)

        # Obstacle 0: constant velocity
        for i in range(20):
            predictor.observe(0, np.array([float(i) * 0.1, 0.0]), mode_id="constant_velocity")
            predictor.observe(1, np.array([0.0, float(i) * 0.1]), mode_id="turn_left")
            predictor.advance_timestep()

        # Each obstacle should have its own buffer
        assert 0 in predictor.trajectory_buffers
        assert 1 in predictor.trajectory_buffers

        # Each should have its own mode distributions
        assert "constant_velocity" in predictor.mode_distributions.get(0, {})
        assert "turn_left" in predictor.mode_distributions.get(1, {})

    def test_predict_different_trajectories(self):
        """Test predictions differ based on learned dynamics."""
        predictor = create_ot_predictor_with_standard_modes(dt=0.1)

        # Train on different modes
        for i in range(30):
            predictor.observe(0, np.array([float(i) * 0.1, 0.0]), mode_id="constant_velocity")
            predictor.observe(1, np.array([0.0, float(i) * 0.1]), mode_id="turn_left")
            predictor.advance_timestep()

        # Predict trajectories
        pred_0 = predictor.predict_trajectory(
            0, np.array([3.0, 0.0]), np.array([1.0, 0.0]), horizon=10
        )
        pred_1 = predictor.predict_trajectory(
            1, np.array([0.0, 3.0]), np.array([0.0, 1.0]), horizon=10
        )

        # Final positions should be different
        final_pos_0 = pred_0[-1].position
        final_pos_1 = pred_1[-1].position

        # They should have moved in different directions
        assert not np.allclose(final_pos_0, final_pos_1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
