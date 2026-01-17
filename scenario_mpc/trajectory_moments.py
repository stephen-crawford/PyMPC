"""
Trajectory moment computation for multi-modal obstacle prediction.

Implements Proposition 1 from guide.md (Section 6):
Recursive computation of mean and covariance for trajectory distributions.

Given mode-dependent dynamics x_{k+1} = A_m @ x_k + b_m + G_m @ w_k,
computes the first two moments of the predictive distribution.
"""

import numpy as np
from typing import Dict, List, Optional
from .types import (
    ObstacleState,
    ModeModel,
    ModeHistory,
    PredictionStep,
    ObstacleTrajectory,
    TrajectoryMoments,
)


def compute_trajectory_moments(
    initial_state: ObstacleState,
    mode_weights: Dict[str, float],
    available_modes: Dict[str, ModeModel],
    horizon: int
) -> TrajectoryMoments:
    """
    Compute trajectory moments using Proposition 1.

    For multi-modal predictions, computes the combined mean and covariance
    by marginalizing over modes.

    Proposition 1 (guide.md):
        mu_k = sum_m w_m * (A_m @ mu_{k-1} + b_m)
        Sigma_k = sum_m w_m * (A_m @ Sigma_{k-1} @ A_m^T + G_m @ G_m^T)
                + sum_m w_m * (mu_m_k - mu_k)(mu_m_k - mu_k)^T

    Args:
        initial_state: Initial obstacle state
        mode_weights: Weight for each mode (should sum to 1)
        available_modes: Dict of ModeModel for each mode
        horizon: Prediction horizon N

    Returns:
        TrajectoryMoments with means and covariances for each timestep
    """
    # Initialize storage
    means = np.zeros((horizon + 1, 2))  # Position means
    covariances = np.zeros((horizon + 1, 2, 2))  # Position covariances

    # Initial state (no uncertainty)
    x0 = initial_state.to_array()
    means[0] = x0[:2]  # Position only
    covariances[0] = np.zeros((2, 2))

    # Track full state for each mode
    mode_means = {m: x0.copy() for m in mode_weights}
    mode_covs = {m: np.zeros((4, 4)) for m in mode_weights}

    # Recursive computation for each timestep
    for k in range(1, horizon + 1):
        # Propagate each mode forward
        new_mode_means = {}
        new_mode_covs = {}

        for mode_id, weight in mode_weights.items():
            if weight <= 0:
                continue

            model = available_modes[mode_id]
            mu_prev = mode_means[mode_id]
            Sigma_prev = mode_covs[mode_id]

            # Propagate mean: mu_m_k = A_m @ mu_{k-1} + b_m
            mu_new = model.A @ mu_prev + model.b

            # Propagate covariance: Sigma_m_k = A_m @ Sigma_{k-1} @ A_m^T + G_m @ G_m^T
            Sigma_new = model.A @ Sigma_prev @ model.A.T + model.G @ model.G.T

            new_mode_means[mode_id] = mu_new
            new_mode_covs[mode_id] = Sigma_new

        mode_means = new_mode_means
        mode_covs = new_mode_covs

        # Compute combined moments by marginalizing over modes
        # mu_k = sum_m w_m * mu_m_k
        combined_mean = np.zeros(4)
        for mode_id, weight in mode_weights.items():
            if weight > 0 and mode_id in mode_means:
                combined_mean += weight * mode_means[mode_id]

        # Compute combined covariance
        # Sigma_k = sum_m w_m * Sigma_m_k + sum_m w_m * (mu_m_k - mu_k)(mu_m_k - mu_k)^T
        combined_cov = np.zeros((4, 4))

        for mode_id, weight in mode_weights.items():
            if weight <= 0 or mode_id not in mode_means:
                continue

            # Within-mode covariance
            combined_cov += weight * mode_covs[mode_id]

            # Between-mode covariance (mode mixing term)
            diff = mode_means[mode_id] - combined_mean
            combined_cov += weight * np.outer(diff, diff)

        # Extract position components (first 2 states)
        means[k] = combined_mean[:2]
        covariances[k] = combined_cov[:2, :2]

    return TrajectoryMoments(
        obstacle_id=0,  # Will be set by caller
        means=means,
        covariances=covariances
    )


def compute_single_mode_trajectory(
    initial_state: ObstacleState,
    mode: ModeModel,
    horizon: int,
    noise_samples: Optional[np.ndarray] = None
) -> ObstacleTrajectory:
    """
    Compute trajectory for a single mode with optional noise.

    Args:
        initial_state: Initial obstacle state
        mode: Mode model to use
        horizon: Prediction horizon
        noise_samples: Optional (horizon, noise_dim) array of noise samples

    Returns:
        ObstacleTrajectory with prediction steps
    """
    steps = []
    x = initial_state.to_array()
    cov = np.zeros((4, 4))

    # Initial step
    steps.append(PredictionStep(
        k=0,
        mean=x[:2].copy(),
        covariance=cov[:2, :2].copy()
    ))

    for k in range(horizon):
        # Propagate mean
        if noise_samples is not None:
            noise = noise_samples[k]
            x = mode.A @ x + mode.b + mode.G @ noise
        else:
            x = mode.A @ x + mode.b

        # Propagate covariance
        cov = mode.A @ cov @ mode.A.T + mode.G @ mode.G.T

        steps.append(PredictionStep(
            k=k + 1,
            mean=x[:2].copy(),
            covariance=cov[:2, :2].copy()
        ))

    return ObstacleTrajectory(
        obstacle_id=0,
        mode_id=mode.mode_id,
        steps=steps,
        probability=1.0
    )


def propagate_obstacle_state(
    state: ObstacleState,
    mode: ModeModel,
    noise: Optional[np.ndarray] = None
) -> ObstacleState:
    """
    Propagate obstacle state one step using given mode.

    Args:
        state: Current obstacle state
        mode: Mode model to use
        noise: Optional process noise sample

    Returns:
        Next obstacle state
    """
    return mode.propagate(state, noise)


def compute_multi_mode_prediction(
    initial_state: ObstacleState,
    mode_weights: Dict[str, float],
    available_modes: Dict[str, ModeModel],
    horizon: int,
    obstacle_id: int = 0
) -> List[ObstacleTrajectory]:
    """
    Generate trajectory predictions for each mode.

    Args:
        initial_state: Initial obstacle state
        mode_weights: Weight for each mode
        available_modes: Dict of ModeModel for each mode
        horizon: Prediction horizon
        obstacle_id: Obstacle identifier

    Returns:
        List of ObstacleTrajectory, one per mode with non-zero weight
    """
    trajectories = []

    for mode_id, weight in mode_weights.items():
        if weight <= 0:
            continue

        model = available_modes[mode_id]
        traj = compute_single_mode_trajectory(initial_state, model, horizon)
        traj.obstacle_id = obstacle_id
        traj.probability = weight
        trajectories.append(traj)

    return trajectories
