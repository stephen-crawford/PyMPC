#!/usr/bin/env python3
"""
Standalone Unit Tests for Optimal Transport Predictor.

This test file tests the OT predictor components without requiring
the full PyMPC infrastructure (CasADi, etc.).

Can be run directly with: python3 test_ot_standalone.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from dataclasses import dataclass
from collections import deque
from typing import List, Dict, Optional, Tuple
from enum import Enum


# =============================================================================
# Mock the external dependencies that would cause import errors
# =============================================================================

class MockPredictionStep:
    def __init__(self, position, angle, major_radius, minor_radius):
        self.position = position
        self.angle = angle
        self.major_radius = major_radius
        self.minor_radius = minor_radius


# Create mock module for planning.types
class MockTypes:
    DynamicObstacle = type('DynamicObstacle', (), {})
    PredictionStep = MockPredictionStep
    Prediction = type('Prediction', (), {})
    PredictionType = type('PredictionType', (), {'GAUSSIAN': 1})


class MockUtils:
    @staticmethod
    def LOG_DEBUG(msg): pass
    @staticmethod
    def LOG_WARN(msg): pass
    @staticmethod
    def LOG_INFO(msg): pass


# Mock the modules
sys.modules['planning.types'] = MockTypes
sys.modules['utils.utils'] = MockUtils


# =============================================================================
# Now import the actual OT predictor code
# =============================================================================

# Copy the core OT components here for standalone testing
# (This avoids import issues while still testing the core algorithms)

@dataclass
class TrajectoryObservation:
    """Single trajectory observation for an obstacle."""
    timestep: int
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    mode_id: Optional[str] = None

    @property
    def state(self) -> np.ndarray:
        return np.concatenate([self.position, self.velocity, self.acceleration])


@dataclass
class EmpiricalDistribution:
    """Empirical probability distribution from samples."""
    samples: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_samples(cls, samples: np.ndarray,
                     weights: Optional[np.ndarray] = None) -> 'EmpiricalDistribution':
        samples = np.atleast_2d(samples)
        n = samples.shape[0]

        if n == 0:
            return cls(samples=np.array([]).reshape(0, samples.shape[1] if samples.ndim > 1 else 1),
                      weights=np.array([]))

        if weights is None:
            weights = np.ones(n) / n
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()

        return cls(samples=samples, weights=weights)

    @property
    def n_samples(self) -> int:
        return self.samples.shape[0]

    @property
    def dim(self) -> int:
        return self.samples.shape[1] if self.samples.ndim > 1 and self.samples.shape[0] > 0 else 0

    @property
    def mean(self) -> np.ndarray:
        if self.n_samples == 0:
            return np.array([])
        return np.average(self.samples, axis=0, weights=self.weights)

    @property
    def covariance(self) -> np.ndarray:
        if self.n_samples < 2:
            return np.eye(self.dim) if self.dim > 0 else np.array([[1.0]])
        centered = self.samples - self.mean
        return np.cov(centered.T, aweights=self.weights)

    def is_empty(self) -> bool:
        return self.n_samples == 0


def compute_cost_matrix(source: np.ndarray, target: np.ndarray, p: int = 2) -> np.ndarray:
    """Compute pairwise cost matrix between source and target samples."""
    diff = source[:, np.newaxis, :] - target[np.newaxis, :, :]
    if p == 2:
        return np.sum(diff ** 2, axis=-1)
    else:
        return np.sum(np.abs(diff) ** p, axis=-1)


def sinkhorn_algorithm(
    source_weights: np.ndarray,
    target_weights: np.ndarray,
    cost_matrix: np.ndarray,
    epsilon: float = 0.1,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-6
) -> Tuple[np.ndarray, float]:
    """Sinkhorn-Knopp algorithm for entropy-regularized optimal transport."""
    n, m = cost_matrix.shape

    if n == 0 or m == 0:
        return np.array([]).reshape(n, m), 0.0

    K = np.exp(-cost_matrix / epsilon)
    K = np.clip(K, 1e-300, None)

    u = np.ones(n)
    v = np.ones(m)

    for iteration in range(max_iterations):
        u_prev = u.copy()

        Kv = K @ v
        u = source_weights / np.clip(Kv, 1e-300, None)

        Ktu = K.T @ u
        v = target_weights / np.clip(Ktu, 1e-300, None)

        if np.max(np.abs(u - u_prev)) < convergence_threshold:
            break

    transport_plan = np.diag(u) @ K @ np.diag(v)
    sinkhorn_distance = np.sum(cost_matrix * transport_plan)

    return transport_plan, sinkhorn_distance


def wasserstein_distance(
    source: EmpiricalDistribution,
    target: EmpiricalDistribution,
    epsilon: float = 0.1,
    p: int = 2
) -> float:
    """Compute (regularized) Wasserstein distance between two distributions."""
    if source.is_empty() or target.is_empty():
        return 0.0

    cost_matrix = compute_cost_matrix(source.samples, target.samples, p=p)
    _, distance = sinkhorn_algorithm(
        source.weights, target.weights, cost_matrix, epsilon
    )

    return distance ** (1.0 / p)


def wasserstein_barycenter(
    distributions: List[EmpiricalDistribution],
    weights: List[float],
    n_support: int = 50,
    epsilon: float = 0.1,
    max_iterations: int = 50,
    convergence_threshold: float = 1e-4
) -> EmpiricalDistribution:
    """Compute Wasserstein barycenter of multiple distributions."""
    if len(distributions) == 0:
        return EmpiricalDistribution(samples=np.array([]).reshape(0, 2),
                                    weights=np.array([]))

    if len(distributions) == 1:
        return distributions[0]

    weights = np.array(weights)
    weights = weights / weights.sum()

    dim = None
    for dist in distributions:
        if not dist.is_empty():
            dim = dist.dim
            break

    if dim is None:
        return EmpiricalDistribution(samples=np.array([]).reshape(0, 2),
                                    weights=np.array([]))

    all_samples = []
    for dist, w in zip(distributions, weights):
        if not dist.is_empty():
            n_from_dist = max(1, int(n_support * w))
            indices = np.random.choice(dist.n_samples, size=n_from_dist,
                                      p=dist.weights, replace=True)
            all_samples.append(dist.samples[indices])

    if len(all_samples) == 0:
        return EmpiricalDistribution(samples=np.array([]).reshape(0, dim),
                                    weights=np.array([]))

    barycenter_samples = np.vstack(all_samples)[:n_support]
    barycenter_weights = np.ones(len(barycenter_samples)) / len(barycenter_samples)

    for iteration in range(max_iterations):
        samples_prev = barycenter_samples.copy()

        transport_updates = np.zeros_like(barycenter_samples)

        for dist, w in zip(distributions, weights):
            if dist.is_empty():
                continue

            cost_matrix = compute_cost_matrix(barycenter_samples, dist.samples, p=2)
            plan, _ = sinkhorn_algorithm(
                barycenter_weights, dist.weights, cost_matrix, epsilon
            )

            row_sums = plan.sum(axis=1, keepdims=True)
            row_sums = np.clip(row_sums, 1e-10, None)
            update = (plan @ dist.samples) / row_sums
            transport_updates += w * update

        barycenter_samples = transport_updates

        change = np.max(np.abs(barycenter_samples - samples_prev))
        if change < convergence_threshold:
            break

    return EmpiricalDistribution(samples=barycenter_samples, weights=barycenter_weights)


# =============================================================================
# Test Functions
# =============================================================================

def test_empirical_distribution_creation():
    """Test creating distribution from samples."""
    print("Test: EmpiricalDistribution creation...", end=" ")
    samples = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    dist = EmpiricalDistribution.from_samples(samples)

    assert dist.n_samples == 4, f"Expected 4 samples, got {dist.n_samples}"
    assert dist.dim == 2, f"Expected 2 dimensions, got {dist.dim}"
    assert not dist.is_empty()
    np.testing.assert_array_almost_equal(dist.weights, [0.25, 0.25, 0.25, 0.25])
    print("PASSED")


def test_weighted_samples():
    """Test creating distribution with custom weights."""
    print("Test: Weighted samples...", end=" ")
    samples = np.array([[0, 0], [1, 0], [0, 1]])
    weights = np.array([0.5, 0.25, 0.25])
    dist = EmpiricalDistribution.from_samples(samples, weights)

    np.testing.assert_array_almost_equal(dist.weights, [0.5, 0.25, 0.25])
    print("PASSED")


def test_mean_computation():
    """Test weighted mean computation."""
    print("Test: Mean computation...", end=" ")
    samples = np.array([[0, 0], [2, 0], [0, 2], [2, 2]])
    dist = EmpiricalDistribution.from_samples(samples)

    expected_mean = np.array([1.0, 1.0])
    np.testing.assert_array_almost_equal(dist.mean, expected_mean)
    print("PASSED")


def test_weighted_mean():
    """Test weighted mean with non-uniform weights."""
    print("Test: Weighted mean...", end=" ")
    samples = np.array([[0, 0], [4, 0]])
    weights = np.array([0.75, 0.25])
    dist = EmpiricalDistribution.from_samples(samples, weights)

    expected_mean = np.array([1.0, 0.0])
    np.testing.assert_array_almost_equal(dist.mean, expected_mean)
    print("PASSED")


def test_empty_distribution():
    """Test empty distribution handling."""
    print("Test: Empty distribution...", end=" ")
    samples = np.array([]).reshape(0, 2)
    dist = EmpiricalDistribution.from_samples(samples)

    assert dist.is_empty()
    assert dist.n_samples == 0
    print("PASSED")


def test_cost_matrix_squared_euclidean():
    """Test squared Euclidean cost matrix."""
    print("Test: Cost matrix...", end=" ")
    source = np.array([[0, 0], [1, 0]])
    target = np.array([[0, 0], [0, 1], [1, 1]])

    cost = compute_cost_matrix(source, target, p=2)

    expected = np.array([[0, 1, 2], [1, 2, 1]])
    np.testing.assert_array_almost_equal(cost, expected)
    print("PASSED")


def test_cost_matrix_symmetry():
    """Test that cost matrix is symmetric for same source and target."""
    print("Test: Cost matrix symmetry...", end=" ")
    points = np.array([[0, 0], [1, 1], [2, 0]])
    cost = compute_cost_matrix(points, points, p=2)

    np.testing.assert_array_almost_equal(cost, cost.T)
    print("PASSED")


def test_sinkhorn_basic():
    """Test basic Sinkhorn computation."""
    print("Test: Sinkhorn basic...", end=" ")
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    cost = np.array([[0, 1], [1, 0]], dtype=float)

    plan, distance = sinkhorn_algorithm(a, b, cost, epsilon=0.1)

    np.testing.assert_array_almost_equal(plan.sum(axis=0), b, decimal=3)
    np.testing.assert_array_almost_equal(plan.sum(axis=1), a, decimal=3)
    print("PASSED")


def test_sinkhorn_convergence():
    """Test that Sinkhorn converges for random inputs."""
    print("Test: Sinkhorn convergence...", end=" ")
    np.random.seed(42)
    n, m = 10, 15

    a = np.random.dirichlet(np.ones(n))
    b = np.random.dirichlet(np.ones(m))
    cost = np.random.rand(n, m)

    plan, distance = sinkhorn_algorithm(a, b, cost, epsilon=0.1, max_iterations=200)

    np.testing.assert_array_almost_equal(plan.sum(axis=0), b, decimal=3)
    np.testing.assert_array_almost_equal(plan.sum(axis=1), a, decimal=3)
    assert distance >= 0
    print("PASSED")


def test_wasserstein_identical_distributions():
    """Test W distance is zero for identical distributions."""
    print("Test: Wasserstein identical...", end=" ")
    samples = np.array([[0, 0], [1, 1], [2, 2]])
    dist = EmpiricalDistribution.from_samples(samples)

    w_dist = wasserstein_distance(dist, dist)

    assert w_dist < 0.1, f"Expected near zero, got {w_dist}"
    print("PASSED")


def test_wasserstein_different_distributions():
    """Test W distance is positive for different distributions."""
    print("Test: Wasserstein different...", end=" ")
    samples1 = np.array([[0, 0], [1, 0]])
    samples2 = np.array([[10, 10], [11, 10]])

    dist1 = EmpiricalDistribution.from_samples(samples1)
    dist2 = EmpiricalDistribution.from_samples(samples2)

    w_dist = wasserstein_distance(dist1, dist2)

    assert w_dist > 5.0, f"Expected > 5, got {w_dist}"
    print("PASSED")


def test_wasserstein_triangle_inequality():
    """Test triangle inequality: W(a,c) <= W(a,b) + W(b,c)."""
    print("Test: Wasserstein triangle inequality...", end=" ")
    np.random.seed(42)

    dist_a = EmpiricalDistribution.from_samples(np.random.randn(50, 2))
    dist_b = EmpiricalDistribution.from_samples(np.random.randn(50, 2) + 2)
    dist_c = EmpiricalDistribution.from_samples(np.random.randn(50, 2) + 4)

    w_ab = wasserstein_distance(dist_a, dist_b)
    w_bc = wasserstein_distance(dist_b, dist_c)
    w_ac = wasserstein_distance(dist_a, dist_c)

    assert w_ac <= w_ab + w_bc + 0.5, f"Triangle inequality violated"
    print("PASSED")


def test_barycenter_single_distribution():
    """Test barycenter of single distribution equals that distribution."""
    print("Test: Barycenter single...", end=" ")
    samples = np.array([[0, 0], [1, 0], [0, 1]])
    dist = EmpiricalDistribution.from_samples(samples)

    barycenter = wasserstein_barycenter([dist], [1.0], n_support=3)

    np.testing.assert_array_almost_equal(barycenter.mean, dist.mean, decimal=1)
    print("PASSED")


def test_barycenter_two_distributions():
    """Test barycenter of two distributions is between them."""
    print("Test: Barycenter two...", end=" ")
    np.random.seed(42)
    samples1 = np.zeros((20, 2))
    samples2 = np.ones((20, 2)) * 4

    dist1 = EmpiricalDistribution.from_samples(samples1)
    dist2 = EmpiricalDistribution.from_samples(samples2)

    barycenter = wasserstein_barycenter([dist1, dist2], [0.5, 0.5], n_support=20)

    # Mean should be approximately at midpoint
    assert 1.0 < barycenter.mean[0] < 3.0, f"Mean x={barycenter.mean[0]} not in expected range"
    assert 1.0 < barycenter.mean[1] < 3.0, f"Mean y={barycenter.mean[1]} not in expected range"
    print("PASSED")


def test_barycenter_weighted():
    """Test weighted barycenter is closer to higher-weighted distribution."""
    print("Test: Barycenter weighted...", end=" ")
    np.random.seed(42)
    samples1 = np.zeros((20, 2))
    samples2 = np.ones((20, 2)) * 4

    dist1 = EmpiricalDistribution.from_samples(samples1)
    dist2 = EmpiricalDistribution.from_samples(samples2)

    # Weight more heavily toward dist1
    barycenter = wasserstein_barycenter([dist1, dist2], [0.8, 0.2], n_support=20)

    # Mean should be closer to [0, 0]
    assert barycenter.mean[0] < 2.0, f"Mean x={barycenter.mean[0]} should be < 2.0"
    assert barycenter.mean[1] < 2.0, f"Mean y={barycenter.mean[1]} should be < 2.0"
    print("PASSED")


def test_trajectory_observation():
    """Test TrajectoryObservation creation and state property."""
    print("Test: TrajectoryObservation...", end=" ")
    obs = TrajectoryObservation(
        timestep=0,
        position=np.array([1.0, 2.0]),
        velocity=np.array([0.1, 0.2]),
        acceleration=np.array([0.01, 0.02]),
        mode_id="test_mode"
    )

    assert obs.timestep == 0
    np.testing.assert_array_equal(obs.position, [1.0, 2.0])
    np.testing.assert_array_equal(obs.state, [1.0, 2.0, 0.1, 0.2, 0.01, 0.02])
    assert obs.mode_id == "test_mode"
    print("PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Optimal Transport Predictor - Standalone Unit Tests")
    print("=" * 60 + "\n")

    tests = [
        test_empirical_distribution_creation,
        test_weighted_samples,
        test_mean_computation,
        test_weighted_mean,
        test_empty_distribution,
        test_cost_matrix_squared_euclidean,
        test_cost_matrix_symmetry,
        test_sinkhorn_basic,
        test_sinkhorn_convergence,
        test_wasserstein_identical_distributions,
        test_wasserstein_different_distributions,
        test_wasserstein_triangle_inequality,
        test_barycenter_single_distribution,
        test_barycenter_two_distributions,
        test_barycenter_weighted,
        test_trajectory_observation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
