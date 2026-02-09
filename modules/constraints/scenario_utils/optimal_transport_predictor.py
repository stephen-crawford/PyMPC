"""
Optimal Transport Predictor for Learning Obstacle Dynamics.

Implements statistical learning of obstacle dynamics using optimal transport theory.
Key features:
- Sinkhorn algorithm for efficient Wasserstein distance computation
- Empirical distribution learning from observed trajectories
- Wasserstein barycenter computation for multi-modal predictions
- Adaptive uncertainty quantification based on distributional distance

Reference: Computational Optimal Transport (Peyré & Cuturi, 2019)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from planning.types import DynamicObstacle, PredictionStep, Prediction, PredictionType
from utils.utils import LOG_DEBUG, LOG_WARN, LOG_INFO


# =============================================================================
# Trajectory Observation and Buffer
# =============================================================================

@dataclass
class TrajectoryObservation:
    """Single trajectory observation for an obstacle."""
    timestep: int
    position: np.ndarray  # [x, y]
    velocity: np.ndarray  # [vx, vy]
    acceleration: np.ndarray  # [ax, ay]
    mode_id: Optional[str] = None

    @property
    def state(self) -> np.ndarray:
        """Full state vector [x, y, vx, vy, ax, ay]."""
        return np.concatenate([self.position, self.velocity, self.acceleration])

    @property
    def dynamics_state(self) -> np.ndarray:
        """Dynamics state vector [x, y, vx, vy] for ModeModel compatibility."""
        return np.concatenate([self.position, self.velocity])


@dataclass
class TrajectoryBuffer:
    """
    Circular buffer storing observed trajectory snippets for an obstacle.

    Used to build empirical distributions of dynamics.
    """
    obstacle_id: int
    max_length: int = 200
    observations: deque = field(default_factory=lambda: deque(maxlen=200))

    def __post_init__(self):
        self.observations = deque(maxlen=self.max_length)

    def add_observation(self, obs: TrajectoryObservation) -> None:
        """Add a new observation to the buffer."""
        self.observations.append(obs)

    def get_recent(self, n: int) -> List[TrajectoryObservation]:
        """Get the n most recent observations."""
        return list(self.observations)[-n:]

    def get_velocity_samples(self) -> np.ndarray:
        """Get all velocity observations as Nx2 array."""
        if len(self.observations) == 0:
            return np.array([]).reshape(0, 2)
        return np.array([obs.velocity for obs in self.observations])

    def get_acceleration_samples(self) -> np.ndarray:
        """Get all acceleration observations as Nx2 array."""
        if len(self.observations) == 0:
            return np.array([]).reshape(0, 2)
        return np.array([obs.acceleration for obs in self.observations])

    def get_state_samples(self) -> np.ndarray:
        """Get all state observations as Nx6 array."""
        if len(self.observations) == 0:
            return np.array([]).reshape(0, 6)
        return np.array([obs.state for obs in self.observations])

    def __len__(self) -> int:
        return len(self.observations)


# =============================================================================
# Empirical Distribution
# =============================================================================

@dataclass
class EmpiricalDistribution:
    """
    Empirical probability distribution from samples.

    Supports discrete distributions with equal or weighted samples.
    Used for optimal transport computations.
    """
    samples: np.ndarray  # N x d array of samples
    weights: np.ndarray  # N array of weights (sum to 1)

    @classmethod
    def from_samples(cls, samples: np.ndarray,
                     weights: Optional[np.ndarray] = None) -> 'EmpiricalDistribution':
        """
        Create empirical distribution from samples.

        Args:
            samples: N x d array of samples
            weights: Optional weights (uniform if None)

        Returns:
            EmpiricalDistribution instance
        """
        samples = np.atleast_2d(samples)
        n = samples.shape[0]

        if n == 0:
            return cls(samples=np.array([]).reshape(0, samples.shape[1] if samples.ndim > 1 else 1),
                      weights=np.array([]))

        if weights is None:
            weights = np.ones(n) / n
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize

        return cls(samples=samples, weights=weights)

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.samples.shape[0]

    @property
    def dim(self) -> int:
        """Dimension of samples."""
        return self.samples.shape[1] if self.samples.ndim > 1 and self.samples.shape[0] > 0 else 0

    @property
    def mean(self) -> np.ndarray:
        """Weighted mean of the distribution."""
        if self.n_samples == 0:
            return np.array([])
        return np.average(self.samples, axis=0, weights=self.weights)

    @property
    def covariance(self) -> np.ndarray:
        """Weighted covariance matrix."""
        if self.n_samples < 2:
            return np.eye(self.dim) if self.dim > 0 else np.array([[1.0]])
        centered = self.samples - self.mean
        return np.cov(centered.T, aweights=self.weights)

    def is_empty(self) -> bool:
        """Check if distribution has no samples."""
        return self.n_samples == 0


# =============================================================================
# Sinkhorn Algorithm for Optimal Transport
# =============================================================================

def compute_cost_matrix(source: np.ndarray, target: np.ndarray,
                        p: int = 2) -> np.ndarray:
    """
    Compute pairwise cost matrix between source and target samples.

    Args:
        source: N x d array of source samples
        target: M x d array of target samples
        p: Power for cost (p=2 for squared Euclidean)

    Returns:
        N x M cost matrix
    """
    # Efficient computation using broadcasting
    # ||s_i - t_j||^p
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
    """
    Sinkhorn-Knopp algorithm for entropy-regularized optimal transport.

    Solves: min_P <C, P> + epsilon * H(P)
    s.t. P @ 1 = a, P.T @ 1 = b, P >= 0

    Reference: Cuturi, M. (2013) "Sinkhorn Distances"

    Args:
        source_weights: Source marginal (a)
        target_weights: Target marginal (b)
        cost_matrix: C[i,j] = cost of transporting source[i] to target[j]
        epsilon: Entropy regularization parameter
        max_iterations: Maximum Sinkhorn iterations
        convergence_threshold: Convergence criterion

    Returns:
        Tuple of (transport_plan P, Sinkhorn distance)
    """
    n, m = cost_matrix.shape

    # Handle edge cases
    if n == 0 or m == 0:
        return np.array([]).reshape(n, m), 0.0

    # Kernel matrix K = exp(-C/epsilon)
    K = np.exp(-cost_matrix / epsilon)

    # Numerical stability: add small constant to avoid division by zero
    K = np.clip(K, 1e-300, None)

    # Initialize scaling vectors
    u = np.ones(n)
    v = np.ones(m)

    # Sinkhorn iterations
    for iteration in range(max_iterations):
        u_prev = u.copy()

        # Update u: u = a / (K @ v)
        Kv = K @ v
        u = source_weights / np.clip(Kv, 1e-300, None)

        # Update v: v = b / (K.T @ u)
        Ktu = K.T @ u
        v = target_weights / np.clip(Ktu, 1e-300, None)

        # Check convergence
        if np.max(np.abs(u - u_prev)) < convergence_threshold:
            LOG_DEBUG(f"Sinkhorn converged after {iteration + 1} iterations")
            break

    # Compute transport plan: P = diag(u) @ K @ diag(v)
    transport_plan = np.diag(u) @ K @ np.diag(v)

    # Sinkhorn distance: <C, P>
    sinkhorn_distance = np.sum(cost_matrix * transport_plan)

    return transport_plan, sinkhorn_distance


def wasserstein_distance(
    source: EmpiricalDistribution,
    target: EmpiricalDistribution,
    epsilon: float = 0.1,
    p: int = 2
) -> float:
    """
    Compute (regularized) Wasserstein distance between two distributions.

    W_p(mu, nu) = (inf_P sum_{i,j} C[i,j] * P[i,j])^(1/p)

    Args:
        source: Source distribution
        target: Target distribution
        epsilon: Sinkhorn regularization
        p: Wasserstein order (typically 2)

    Returns:
        Wasserstein distance (or 0 if either distribution is empty)
    """
    if source.is_empty() or target.is_empty():
        return 0.0

    cost_matrix = compute_cost_matrix(source.samples, target.samples, p=p)
    _, distance = sinkhorn_algorithm(
        source.weights, target.weights, cost_matrix, epsilon
    )

    # Return p-th root for W_p distance
    return distance ** (1.0 / p)


# =============================================================================
# Wasserstein Barycenter for Multi-Modal Predictions
# =============================================================================

def wasserstein_barycenter(
    distributions: List[EmpiricalDistribution],
    weights: List[float],
    n_support: int = 50,
    epsilon: float = 0.1,
    max_iterations: int = 50,
    convergence_threshold: float = 1e-4
) -> EmpiricalDistribution:
    """
    Compute Wasserstein barycenter of multiple distributions.

    Finds: argmin_nu sum_i w_i * W_2(mu_i, nu)^2

    Uses iterative Bregman projection algorithm.

    Reference: Cuturi & Doucet (2014) "Fast Computation of Wasserstein Barycenters"

    Args:
        distributions: List of input distributions
        weights: Barycentric weights (should sum to 1)
        n_support: Number of support points in output
        epsilon: Sinkhorn regularization
        max_iterations: Maximum iterations
        convergence_threshold: Convergence criterion

    Returns:
        Barycenter distribution
    """
    if len(distributions) == 0:
        return EmpiricalDistribution(samples=np.array([]).reshape(0, 2),
                                    weights=np.array([]))

    if len(distributions) == 1:
        return distributions[0]

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Get dimension from first non-empty distribution
    dim = None
    for dist in distributions:
        if not dist.is_empty():
            dim = dist.dim
            break

    if dim is None:
        return EmpiricalDistribution(samples=np.array([]).reshape(0, 2),
                                    weights=np.array([]))

    # Initialize barycenter support from weighted combination
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

    # Iterative refinement using fixed-point iterations
    for iteration in range(max_iterations):
        samples_prev = barycenter_samples.copy()

        # Compute transport plans to each distribution
        transport_updates = np.zeros_like(barycenter_samples)

        for dist, w in zip(distributions, weights):
            if dist.is_empty():
                continue

            cost_matrix = compute_cost_matrix(barycenter_samples, dist.samples, p=2)
            plan, _ = sinkhorn_algorithm(
                barycenter_weights, dist.weights, cost_matrix, epsilon
            )

            # Barycentric update: weighted average of transport targets
            # x_new[i] = sum_j P[i,j] * y[j] / sum_j P[i,j]
            row_sums = plan.sum(axis=1, keepdims=True)
            row_sums = np.clip(row_sums, 1e-10, None)
            update = (plan @ dist.samples) / row_sums
            transport_updates += w * update

        barycenter_samples = transport_updates

        # Check convergence
        change = np.max(np.abs(barycenter_samples - samples_prev))
        if change < convergence_threshold:
            LOG_DEBUG(f"Barycenter converged after {iteration + 1} iterations")
            break

    return EmpiricalDistribution(samples=barycenter_samples, weights=barycenter_weights)


# =============================================================================
# Optimal Transport Predictor
# =============================================================================

class OTWeightType(Enum):
    """Weight computation strategies for OT predictor."""
    WASSERSTEIN = "wasserstein"  # Inverse Wasserstein distance weights
    LIKELIHOOD = "likelihood"    # Likelihood-based weights
    UNIFORM = "uniform"          # Equal weights


@dataclass
class ModeDistribution:
    """Distribution of dynamics for a specific mode."""
    mode_id: str
    velocity_dist: EmpiricalDistribution
    acceleration_dist: EmpiricalDistribution
    observation_count: int = 0
    last_updated: int = 0


class OptimalTransportPredictor:
    """
    Optimal Transport-based predictor for obstacle dynamics.

    Key features:
    1. Learns empirical distributions of obstacle dynamics from observations
    2. Uses Wasserstein distance to compare predicted vs actual trajectories
    3. Computes Wasserstein barycenters for multi-modal predictions
    4. Provides adaptive uncertainty based on distributional distance

    Integration with AdaptiveModeSampler:
    - Can provide OT-based mode weights as alternative to frequency/recency
    - Validates mode predictions against observed dynamics
    - Adjusts uncertainty based on distribution mismatch
    """

    def __init__(
        self,
        dt: float = 0.1,
        buffer_size: int = 200,
        sinkhorn_epsilon: float = 0.1,
        min_samples_for_ot: int = 10,
        uncertainty_scale: float = 1.0,
        weight_type: OTWeightType = OTWeightType.WASSERSTEIN
    ):
        """
        Initialize the Optimal Transport Predictor.

        Args:
            dt: Timestep in seconds
            buffer_size: Maximum observations per obstacle
            sinkhorn_epsilon: Regularization for Sinkhorn algorithm
            min_samples_for_ot: Minimum samples needed for OT computation
            uncertainty_scale: Scaling factor for uncertainty estimates
            weight_type: Strategy for computing mode weights
        """
        self.dt = dt
        self.buffer_size = buffer_size
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.min_samples_for_ot = min_samples_for_ot
        self.uncertainty_scale = uncertainty_scale
        self.weight_type = weight_type

        # Per-obstacle trajectory buffers
        self.trajectory_buffers: Dict[int, TrajectoryBuffer] = {}

        # Per-obstacle, per-mode learned distributions
        self.mode_distributions: Dict[int, Dict[str, ModeDistribution]] = {}

        # Reference mode distributions (prior knowledge)
        self.reference_distributions: Dict[str, ModeDistribution] = {}

        # Current timestep
        self.current_timestep = 0

        # Previous observations for velocity/acceleration computation
        self._prev_observations: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        LOG_INFO(f"OptimalTransportPredictor initialized: epsilon={sinkhorn_epsilon}, "
                f"buffer_size={buffer_size}, weight_type={weight_type.value}")

    def observe(
        self,
        obstacle_id: int,
        position: np.ndarray,
        mode_id: Optional[str] = None
    ) -> None:
        """
        Record an observation of obstacle state.

        Args:
            obstacle_id: Obstacle identifier
            position: Current position [x, y]
            mode_id: Optional mode label for the observation
        """
        position = np.array(position[:2])

        # Initialize buffer if needed
        if obstacle_id not in self.trajectory_buffers:
            self.trajectory_buffers[obstacle_id] = TrajectoryBuffer(
                obstacle_id=obstacle_id,
                max_length=self.buffer_size
            )
            self._prev_observations[obstacle_id] = (position, np.zeros(2))

        # Compute velocity and acceleration from finite differences
        prev_pos, prev_vel = self._prev_observations[obstacle_id]

        velocity = (position - prev_pos) / self.dt
        acceleration = (velocity - prev_vel) / self.dt

        # Create and store observation
        obs = TrajectoryObservation(
            timestep=self.current_timestep,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            mode_id=mode_id
        )

        self.trajectory_buffers[obstacle_id].add_observation(obs)

        # Update previous observation
        self._prev_observations[obstacle_id] = (position, velocity)

        # Update mode distribution if mode is provided
        if mode_id is not None:
            self._update_mode_distribution(obstacle_id, mode_id, velocity, acceleration)

    def _update_mode_distribution(
        self,
        obstacle_id: int,
        mode_id: str,
        velocity: np.ndarray,
        acceleration: np.ndarray
    ) -> None:
        """Update the learned distribution for a specific mode."""
        if obstacle_id not in self.mode_distributions:
            self.mode_distributions[obstacle_id] = {}

        mode_dists = self.mode_distributions[obstacle_id]

        if mode_id not in mode_dists:
            # Initialize new mode distribution
            mode_dists[mode_id] = ModeDistribution(
                mode_id=mode_id,
                velocity_dist=EmpiricalDistribution.from_samples(
                    velocity.reshape(1, 2)
                ),
                acceleration_dist=EmpiricalDistribution.from_samples(
                    acceleration.reshape(1, 2)
                ),
                observation_count=1,
                last_updated=self.current_timestep
            )
        else:
            # Update existing distribution with new sample
            mode_dist = mode_dists[mode_id]

            # Add new samples to existing distributions
            new_vel_samples = np.vstack([
                mode_dist.velocity_dist.samples,
                velocity.reshape(1, 2)
            ])
            new_acc_samples = np.vstack([
                mode_dist.acceleration_dist.samples,
                acceleration.reshape(1, 2)
            ])

            # Keep only recent samples (sliding window)
            max_samples = self.buffer_size // 2
            if len(new_vel_samples) > max_samples:
                new_vel_samples = new_vel_samples[-max_samples:]
                new_acc_samples = new_acc_samples[-max_samples:]

            mode_dist.velocity_dist = EmpiricalDistribution.from_samples(new_vel_samples)
            mode_dist.acceleration_dist = EmpiricalDistribution.from_samples(new_acc_samples)
            mode_dist.observation_count += 1
            mode_dist.last_updated = self.current_timestep

    def advance_timestep(self) -> None:
        """Advance the current timestep."""
        self.current_timestep += 1

    def compute_mode_weights(
        self,
        obstacle_id: int,
        available_modes: List[str],
        reference_velocity: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute mode weights using optimal transport distance.

        Uses inverse Wasserstein distance from observed dynamics to mode
        reference distributions.

        Args:
            obstacle_id: Obstacle identifier
            available_modes: List of available mode IDs
            reference_velocity: Optional recent velocity observation

        Returns:
            Dictionary mapping mode_id to weight (normalized to sum to 1)
        """
        if self.weight_type == OTWeightType.UNIFORM:
            return {m: 1.0 / len(available_modes) for m in available_modes}

        # Get recent velocity samples
        if obstacle_id not in self.trajectory_buffers:
            return {m: 1.0 / len(available_modes) for m in available_modes}

        buffer = self.trajectory_buffers[obstacle_id]
        if len(buffer) < self.min_samples_for_ot:
            # Not enough samples, use uniform
            return {m: 1.0 / len(available_modes) for m in available_modes}

        # Build empirical distribution from recent observations
        recent_obs = buffer.get_recent(self.min_samples_for_ot)
        observed_velocities = np.array([obs.velocity for obs in recent_obs])
        observed_dist = EmpiricalDistribution.from_samples(observed_velocities)

        # Compute Wasserstein distance to each mode's distribution
        weights = {}

        for mode_id in available_modes:
            # Get mode distribution (learned or reference)
            mode_dist = self._get_mode_velocity_distribution(obstacle_id, mode_id)

            if mode_dist is None or mode_dist.is_empty():
                # Use default weight if no distribution available
                weights[mode_id] = 1.0
                continue

            # Compute Wasserstein distance
            w_dist = wasserstein_distance(
                observed_dist, mode_dist,
                epsilon=self.sinkhorn_epsilon
            )

            # Convert distance to weight: higher distance -> lower weight
            # Using exponential kernel: w = exp(-distance / scale)
            weights[mode_id] = np.exp(-w_dist / (self.uncertainty_scale + 1e-6))

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            weights = {m: 1.0 / len(available_modes) for m in available_modes}

        return weights

    def _get_mode_velocity_distribution(
        self,
        obstacle_id: int,
        mode_id: str
    ) -> Optional[EmpiricalDistribution]:
        """Get velocity distribution for a mode (learned or reference)."""
        # Check learned distributions first
        if obstacle_id in self.mode_distributions:
            if mode_id in self.mode_distributions[obstacle_id]:
                return self.mode_distributions[obstacle_id][mode_id].velocity_dist

        # Fall back to reference distributions
        if mode_id in self.reference_distributions:
            return self.reference_distributions[mode_id].velocity_dist

        return None

    def estimate_mode_dynamics(
        self,
        obstacle_id: int,
        mode_id: str,
        A_prior: np.ndarray,
        dt: float
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate mode dynamics parameters (b, G) from observed trajectories.

        For consecutive observations with matching mode_id, compute:
          residual = x_{k+1} - A_prior @ x_k
          b_learned = mean(residuals)
          G_learned = cholesky(cov(residuals) + regularization)

        Args:
            obstacle_id: Obstacle identifier
            mode_id: Mode to estimate dynamics for
            A_prior: Prior state transition matrix (4x4)
            dt: Timestep

        Returns:
            Tuple of (b_learned, G_learned) or None if insufficient data
        """
        if obstacle_id not in self.trajectory_buffers:
            return None

        buffer = self.trajectory_buffers[obstacle_id]
        if len(buffer) < 3:
            return None

        # Filter observations for target mode and compute residuals
        observations = list(buffer.observations)
        residuals = []

        for i in range(len(observations) - 1):
            obs_k = observations[i]
            obs_k1 = observations[i + 1]

            # Only use consecutive pairs with matching mode
            if obs_k.mode_id != mode_id and obs_k1.mode_id != mode_id:
                continue

            # Compute dynamics state [x, y, vx, vy]
            x_k = obs_k.dynamics_state
            x_k1 = obs_k1.dynamics_state

            # residual = x_{k+1} - A @ x_k
            residual = x_k1 - A_prior @ x_k
            residuals.append(residual)

        if len(residuals) < 3:
            return None

        residuals = np.array(residuals)

        # b_learned = mean of residuals
        b_learned = np.mean(residuals, axis=0)

        # G_learned = cholesky(cov + regularization)
        cov = np.cov(residuals.T)
        if cov.ndim < 2:
            cov = np.array([[cov]])

        # Add regularization for numerical stability
        reg = 1e-6 * np.eye(cov.shape[0])
        cov_reg = cov + reg

        try:
            G_learned = np.linalg.cholesky(cov_reg)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use diagonal approximation
            diag_vars = np.maximum(np.diag(cov_reg), 1e-6)
            G_learned = np.diag(np.sqrt(diag_vars))

        return b_learned, G_learned

    def predict_trajectory(
        self,
        obstacle_id: int,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        horizon: int,
        mode_weights: Optional[Dict[str, float]] = None
    ) -> List[PredictionStep]:
        """
        Generate trajectory prediction using Wasserstein barycenter.

        Combines mode-specific predictions using OT barycenter for
        smooth multi-modal prediction.

        Args:
            obstacle_id: Obstacle identifier
            current_position: Current [x, y] position
            current_velocity: Current [vx, vy] velocity
            horizon: Number of prediction steps
            mode_weights: Optional mode weights (computed if None)

        Returns:
            List of PredictionStep objects
        """
        current_position = np.array(current_position[:2])
        current_velocity = np.array(current_velocity[:2])

        # Get available modes
        if obstacle_id in self.mode_distributions:
            available_modes = list(self.mode_distributions[obstacle_id].keys())
        else:
            available_modes = list(self.reference_distributions.keys())

        if len(available_modes) == 0:
            # No modes available, use constant velocity
            return self._constant_velocity_prediction(
                current_position, current_velocity, horizon
            )

        # Compute mode weights if not provided
        if mode_weights is None:
            mode_weights = self.compute_mode_weights(obstacle_id, available_modes)

        # Generate prediction for each mode
        mode_predictions = []
        active_modes = []
        active_weights = []

        for mode_id in available_modes:
            weight = mode_weights.get(mode_id, 0.0)
            if weight < 0.01:  # Skip negligible modes
                continue

            pred = self._predict_with_mode(
                obstacle_id, mode_id,
                current_position, current_velocity, horizon
            )
            mode_predictions.append(pred)
            active_modes.append(mode_id)
            active_weights.append(weight)

        if len(mode_predictions) == 0:
            return self._constant_velocity_prediction(
                current_position, current_velocity, horizon
            )

        if len(mode_predictions) == 1:
            return mode_predictions[0]

        # Combine predictions using Wasserstein barycenter
        return self._combine_predictions_barycenter(
            mode_predictions, active_weights, horizon
        )

    def _predict_with_mode(
        self,
        obstacle_id: int,
        mode_id: str,
        position: np.ndarray,
        velocity: np.ndarray,
        horizon: int
    ) -> List[PredictionStep]:
        """Generate prediction using a specific mode's learned dynamics."""
        predictions = []

        # Get mode distribution
        mode_dist = None
        if obstacle_id in self.mode_distributions:
            if mode_id in self.mode_distributions[obstacle_id]:
                mode_dist = self.mode_distributions[obstacle_id][mode_id]

        if mode_dist is None and mode_id in self.reference_distributions:
            mode_dist = self.reference_distributions[mode_id]

        pos = position.copy()
        vel = velocity.copy()

        for k in range(horizon + 1):
            # Compute uncertainty from distribution spread
            if mode_dist is not None and not mode_dist.velocity_dist.is_empty():
                vel_cov = mode_dist.velocity_dist.covariance
                if vel_cov.ndim == 2 and vel_cov.shape[0] >= 2:
                    # Use eigenvalues for ellipse radii
                    eigvals = np.linalg.eigvalsh(vel_cov[:2, :2])
                    major_r = np.sqrt(max(eigvals)) * (1 + 0.1 * k) * self.uncertainty_scale
                    minor_r = np.sqrt(min(eigvals)) * (1 + 0.1 * k) * self.uncertainty_scale
                else:
                    major_r = 0.3 * (1 + 0.1 * k)
                    minor_r = 0.3 * (1 + 0.1 * k)
            else:
                major_r = 0.3 * (1 + 0.1 * k)
                minor_r = 0.3 * (1 + 0.1 * k)

            # Compute heading from velocity
            speed = np.linalg.norm(vel)
            if speed > 0.01:
                angle = np.arctan2(vel[1], vel[0])
            else:
                angle = 0.0

            predictions.append(PredictionStep(
                position=pos.copy(),
                angle=angle,
                major_radius=major_r,
                minor_radius=minor_r
            ))

            # Propagate position
            pos = pos + vel * self.dt

            # Update velocity based on mode dynamics (simple model)
            if mode_dist is not None and not mode_dist.velocity_dist.is_empty():
                # Drift toward mode's mean velocity
                mean_vel = mode_dist.velocity_dist.mean
                vel = 0.95 * vel + 0.05 * mean_vel

        return predictions

    def _constant_velocity_prediction(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        horizon: int
    ) -> List[PredictionStep]:
        """Simple constant velocity prediction."""
        predictions = []
        pos = position.copy()

        for k in range(horizon + 1):
            speed = np.linalg.norm(velocity)
            angle = np.arctan2(velocity[1], velocity[0]) if speed > 0.01 else 0.0

            major_r = 0.3 * (1 + 0.1 * k)
            minor_r = 0.3 * (1 + 0.1 * k)

            predictions.append(PredictionStep(
                position=pos.copy(),
                angle=angle,
                major_radius=major_r,
                minor_radius=minor_r
            ))

            pos = pos + velocity * self.dt

        return predictions

    def _combine_predictions_barycenter(
        self,
        predictions_list: List[List[PredictionStep]],
        weights: List[float],
        horizon: int
    ) -> List[PredictionStep]:
        """
        Combine multiple predictions using Wasserstein barycenter.

        For each timestep, compute barycenter of predicted positions.
        """
        combined = []
        weights = np.array(weights)
        weights = weights / weights.sum()

        for k in range(horizon + 1):
            # Collect positions and uncertainties at this timestep
            positions = []
            major_radii = []
            minor_radii = []
            angles = []

            for preds, w in zip(predictions_list, weights):
                if k < len(preds):
                    positions.append(preds[k].position)
                    major_radii.append(preds[k].major_radius)
                    minor_radii.append(preds[k].minor_radius)
                    angles.append(preds[k].angle)

            if len(positions) == 0:
                continue

            positions = np.array(positions)

            # Compute barycenter position (weighted average for simplicity)
            # For full OT barycenter, would need more samples per prediction
            barycenter_pos = np.average(positions, axis=0, weights=weights[:len(positions)])

            # Average uncertainty (could also compute OT-based uncertainty)
            avg_major = np.average(major_radii, weights=weights[:len(major_radii)])
            avg_minor = np.average(minor_radii, weights=weights[:len(minor_radii)])

            # Weighted circular mean for angle
            sin_sum = np.sum(weights[:len(angles)] * np.sin(angles))
            cos_sum = np.sum(weights[:len(angles)] * np.cos(angles))
            avg_angle = np.arctan2(sin_sum, cos_sum)

            # Add uncertainty from mode disagreement
            pos_spread = np.std(positions, axis=0)
            spread_factor = 1.0 + np.linalg.norm(pos_spread) * 0.5

            combined.append(PredictionStep(
                position=barycenter_pos,
                angle=avg_angle,
                major_radius=avg_major * spread_factor,
                minor_radius=avg_minor * spread_factor
            ))

        return combined

    def compute_prediction_error(
        self,
        obstacle_id: int,
        predicted_trajectory: List[np.ndarray],
        actual_trajectory: List[np.ndarray]
    ) -> float:
        """
        Compute Wasserstein distance between predicted and actual trajectories.

        Useful for evaluating prediction quality and adapting uncertainty.

        Args:
            obstacle_id: Obstacle identifier
            predicted_trajectory: List of predicted positions
            actual_trajectory: List of actual positions

        Returns:
            Wasserstein distance between trajectories
        """
        if len(predicted_trajectory) == 0 or len(actual_trajectory) == 0:
            return 0.0

        pred_array = np.array(predicted_trajectory)
        actual_array = np.array(actual_trajectory)

        # Truncate to common length
        min_len = min(len(pred_array), len(actual_array))
        pred_array = pred_array[:min_len]
        actual_array = actual_array[:min_len]

        pred_dist = EmpiricalDistribution.from_samples(pred_array)
        actual_dist = EmpiricalDistribution.from_samples(actual_array)

        return wasserstein_distance(pred_dist, actual_dist, epsilon=self.sinkhorn_epsilon)

    def adapt_uncertainty(
        self,
        obstacle_id: int,
        prediction_error: float
    ) -> float:
        """
        Compute adaptive uncertainty scaling based on prediction error.

        Higher prediction error -> larger uncertainty.

        Args:
            obstacle_id: Obstacle identifier
            prediction_error: Recent Wasserstein prediction error

        Returns:
            Uncertainty multiplier
        """
        # Sigmoid-like scaling: error -> multiplier in [1, 3]
        base_scale = 1.0
        error_scale = 2.0 * (1.0 / (1.0 + np.exp(-prediction_error + 1.0)))

        return base_scale + error_scale

    def set_reference_distribution(
        self,
        mode_id: str,
        velocity_samples: np.ndarray,
        acceleration_samples: Optional[np.ndarray] = None
    ) -> None:
        """
        Set reference distribution for a mode (prior knowledge).

        Args:
            mode_id: Mode identifier
            velocity_samples: N x 2 array of velocity samples
            acceleration_samples: Optional N x 2 array of acceleration samples
        """
        vel_dist = EmpiricalDistribution.from_samples(velocity_samples)

        if acceleration_samples is not None:
            acc_dist = EmpiricalDistribution.from_samples(acceleration_samples)
        else:
            acc_dist = EmpiricalDistribution.from_samples(np.zeros((1, 2)))

        self.reference_distributions[mode_id] = ModeDistribution(
            mode_id=mode_id,
            velocity_dist=vel_dist,
            acceleration_dist=acc_dist,
            observation_count=len(velocity_samples),
            last_updated=self.current_timestep
        )

        LOG_INFO(f"Set reference distribution for mode '{mode_id}' with "
                f"{len(velocity_samples)} samples")

    def get_learned_modes(self, obstacle_id: int) -> Set[str]:
        """Get set of modes with learned distributions for an obstacle."""
        if obstacle_id not in self.mode_distributions:
            return set()
        return set(self.mode_distributions[obstacle_id].keys())

    def get_mode_distribution_stats(
        self,
        obstacle_id: int,
        mode_id: str
    ) -> Optional[Dict]:
        """Get statistics for a mode's learned distribution."""
        if obstacle_id not in self.mode_distributions:
            return None
        if mode_id not in self.mode_distributions[obstacle_id]:
            return None

        mode_dist = self.mode_distributions[obstacle_id][mode_id]

        return {
            'mode_id': mode_id,
            'observation_count': mode_dist.observation_count,
            'last_updated': mode_dist.last_updated,
            'velocity_mean': mode_dist.velocity_dist.mean.tolist() if not mode_dist.velocity_dist.is_empty() else None,
            'velocity_cov': mode_dist.velocity_dist.covariance.tolist() if not mode_dist.velocity_dist.is_empty() else None,
        }

    def estimate_mode_dynamics(
        self,
        obstacle_id: int,
        mode_id: str,
        A_prior: np.ndarray,
        dt: float
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate mode dynamics parameters (b, G) from observed trajectory data.

        For consecutive observations with the target mode, compute residuals:
            residual_k = x_{k+1} - A_prior @ x_k
        Then:
            b_learned = mean(residuals)
            G_learned = cholesky(cov(residuals) + regularization)

        Args:
            obstacle_id: Obstacle identifier
            mode_id: Mode to estimate dynamics for
            A_prior: Prior state transition matrix (4x4)
            dt: Timestep

        Returns:
            Tuple (b_learned, G_learned) or None if insufficient data
        """
        if obstacle_id not in self.trajectory_buffers:
            return None

        buffer = self.trajectory_buffers[obstacle_id]
        observations = list(buffer.observations)

        # Filter consecutive pairs where mode matches
        residuals = []
        for i in range(len(observations) - 1):
            obs_k = observations[i]
            obs_k1 = observations[i + 1]

            # Only use observations with matching mode
            if obs_k.mode_id != mode_id and obs_k1.mode_id != mode_id:
                continue

            x_k = obs_k.dynamics_state   # [x, y, vx, vy]
            x_k1 = obs_k1.dynamics_state

            # Compute residual: x_{k+1} - A @ x_k
            predicted = A_prior @ x_k
            residual = x_k1 - predicted
            residuals.append(residual)

        # Need at least 5 samples for meaningful statistics
        if len(residuals) < 5:
            return None

        residuals = np.array(residuals)  # (N, 4)

        # b_learned = mean of residuals
        b_learned = np.mean(residuals, axis=0)

        # G_learned = cholesky(cov(residuals) + regularization)
        cov_residuals = np.cov(residuals.T)
        if cov_residuals.ndim == 0:
            cov_residuals = np.array([[cov_residuals]])

        # Add regularization for numerical stability
        reg = 1e-6 * np.eye(cov_residuals.shape[0])
        cov_reg = cov_residuals + reg

        try:
            G_learned = np.linalg.cholesky(cov_reg)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use diagonal approximation
            diag = np.sqrt(np.maximum(np.diag(cov_reg), 1e-8))
            G_learned = np.diag(diag)

        LOG_DEBUG(f"Estimated dynamics for obstacle {obstacle_id}, mode '{mode_id}': "
                  f"b_norm={np.linalg.norm(b_learned):.4f}, "
                  f"G_trace={np.trace(G_learned):.4f}, "
                  f"n_samples={len(residuals)}")

        return b_learned, G_learned

    def reset(self) -> None:
        """Reset all learned distributions but keep reference distributions."""
        self.trajectory_buffers.clear()
        self.mode_distributions.clear()
        self._prev_observations.clear()
        self.current_timestep = 0
        LOG_INFO("OptimalTransportPredictor reset (reference distributions preserved)")

    def reset_all(self) -> None:
        """Reset all state including reference distributions."""
        self.reset()
        self.reference_distributions.clear()
        LOG_INFO("OptimalTransportPredictor fully reset")


# =============================================================================
# Factory function for creating OT predictor with standard mode references
# =============================================================================

def create_ot_predictor_with_standard_modes(
    dt: float = 0.1,
    base_speed: float = 0.5,
    **kwargs
) -> OptimalTransportPredictor:
    """
    Create OT predictor with reference distributions for standard modes.

    Args:
        dt: Timestep
        base_speed: Base obstacle speed for generating reference samples
        **kwargs: Additional arguments for OptimalTransportPredictor

    Returns:
        Configured OptimalTransportPredictor
    """
    predictor = OptimalTransportPredictor(dt=dt, **kwargs)

    # Generate reference velocity samples for standard modes
    n_samples = 100
    noise_std = 0.1

    # Constant velocity - forward motion
    cv_velocities = np.random.normal(
        loc=[base_speed, 0.0],
        scale=noise_std,
        size=(n_samples, 2)
    )
    predictor.set_reference_distribution("constant_velocity", cv_velocities)

    # Decelerating - reduced forward speed
    dec_velocities = np.random.normal(
        loc=[base_speed * 0.5, 0.0],
        scale=noise_std,
        size=(n_samples, 2)
    )
    predictor.set_reference_distribution("decelerating", dec_velocities)

    # Accelerating - increased forward speed
    acc_velocities = np.random.normal(
        loc=[base_speed * 1.5, 0.0],
        scale=noise_std,
        size=(n_samples, 2)
    )
    predictor.set_reference_distribution("accelerating", acc_velocities)

    # Turn left - forward with positive lateral component
    turn_rate = 0.8
    tl_velocities = np.random.normal(
        loc=[base_speed * np.cos(turn_rate), base_speed * np.sin(turn_rate)],
        scale=noise_std,
        size=(n_samples, 2)
    )
    predictor.set_reference_distribution("turn_left", tl_velocities)

    # Turn right - forward with negative lateral component
    tr_velocities = np.random.normal(
        loc=[base_speed * np.cos(-turn_rate), base_speed * np.sin(-turn_rate)],
        scale=noise_std,
        size=(n_samples, 2)
    )
    predictor.set_reference_distribution("turn_right", tr_velocities)

    # Lane change left
    lcl_velocities = np.random.normal(
        loc=[base_speed, 0.3],
        scale=noise_std,
        size=(n_samples, 2)
    )
    predictor.set_reference_distribution("lane_change_left", lcl_velocities)

    # Lane change right
    lcr_velocities = np.random.normal(
        loc=[base_speed, -0.3],
        scale=noise_std,
        size=(n_samples, 2)
    )
    predictor.set_reference_distribution("lane_change_right", lcr_velocities)

    LOG_INFO(f"Created OT predictor with {len(predictor.reference_distributions)} "
            f"reference mode distributions")

    return predictor
