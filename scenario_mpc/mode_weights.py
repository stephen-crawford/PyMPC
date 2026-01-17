"""
Mode weight computation for adaptive scenario-based MPC.

Implements Section 4 of guide.md: Mode History and Weights

Supports three weight computation strategies:
- Uniform: Equal weights for all modes (Eq. 4)
- Recency: Exponential decay weighting recent observations (Eq. 5)
- Frequency: Weights based on observation frequency (Eq. 6)
"""

import numpy as np
from typing import Dict, List, Tuple
from .types import ModeHistory, WeightType


def compute_mode_weights(
    mode_history: ModeHistory,
    weight_type: WeightType = WeightType.FREQUENCY,
    recency_decay: float = 0.9,
    current_timestep: int = 0
) -> Dict[str, float]:
    """
    Compute mode weights based on observation history.

    Following Section 4 of guide.md.

    Args:
        mode_history: Observed mode history for an obstacle
        weight_type: Weight computation strategy
        recency_decay: Decay factor lambda for recency weighting
        current_timestep: Current timestep for recency computation

    Returns:
        Dictionary mapping mode_id to weight (normalized to sum to 1)
    """
    modes = list(mode_history.available_modes.keys())
    num_modes = len(modes)

    if num_modes == 0:
        return {}

    if weight_type == WeightType.UNIFORM:
        # Eq. 4: w_m = 1/M for all modes
        weights = {mode_id: 1.0 / num_modes for mode_id in modes}

    elif weight_type == WeightType.RECENCY:
        # Eq. 5: w_m = sum_{t: m_t = m} lambda^(T - t)
        weights = _compute_recency_weights(
            mode_history, modes, recency_decay, current_timestep
        )

    elif weight_type == WeightType.FREQUENCY:
        # Eq. 6: w_m = n_m / sum_j n_j
        weights = _compute_frequency_weights(mode_history, modes)

    else:
        raise ValueError(f"Unknown weight type: {weight_type}")

    # Normalize weights to sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    else:
        # Fallback to uniform if no observations
        weights = {mode_id: 1.0 / num_modes for mode_id in modes}

    return weights


def _compute_recency_weights(
    mode_history: ModeHistory,
    modes: List[str],
    decay: float,
    current_timestep: int
) -> Dict[str, float]:
    """
    Compute recency-based weights (Eq. 5).

    w_m = sum_{t: m_t = m} lambda^(T - t)

    Recent observations are weighted more heavily.
    """
    weights = {mode_id: 0.0 for mode_id in modes}

    for timestep, mode_id in mode_history.observed_modes:
        if mode_id in weights:
            # Exponential decay based on how old the observation is
            age = current_timestep - timestep
            weights[mode_id] += decay ** age

    return weights


def _compute_frequency_weights(
    mode_history: ModeHistory,
    modes: List[str]
) -> Dict[str, float]:
    """
    Compute frequency-based weights (Eq. 6).

    w_m = n_m / sum_j n_j

    where n_m is the number of times mode m was observed.
    """
    counts = mode_history.get_mode_counts()
    weights = {mode_id: float(counts.get(mode_id, 0)) for mode_id in modes}
    return weights


def sample_mode_sequence(
    mode_weights: Dict[str, float],
    horizon: int,
    rng: np.random.Generator = None
) -> List[str]:
    """
    Sample a mode sequence for the prediction horizon.

    Assumes modes are i.i.d. across timesteps (can be extended for Markov).

    Args:
        mode_weights: Weights for each mode
        horizon: Number of timesteps to sample
        rng: Random number generator

    Returns:
        List of mode_ids of length horizon
    """
    if rng is None:
        rng = np.random.default_rng()

    modes = list(mode_weights.keys())
    weights = np.array([mode_weights[m] for m in modes])

    # Normalize weights
    weights = weights / weights.sum()

    # Sample mode indices
    indices = rng.choice(len(modes), size=horizon, p=weights)

    return [modes[i] for i in indices]


def sample_mode_from_weights(
    mode_weights: Dict[str, float],
    rng: np.random.Generator = None
) -> str:
    """
    Sample a single mode from the weight distribution.

    Args:
        mode_weights: Weights for each mode
        rng: Random number generator

    Returns:
        Sampled mode_id
    """
    if rng is None:
        rng = np.random.default_rng()

    modes = list(mode_weights.keys())
    weights = np.array([mode_weights[m] for m in modes])
    weights = weights / weights.sum()

    idx = rng.choice(len(modes), p=weights)
    return modes[idx]


def compute_mode_transition_matrix(
    mode_history: ModeHistory,
    modes: List[str]
) -> np.ndarray:
    """
    Estimate mode transition probabilities from history.

    P[i,j] = P(mode_j | mode_i) estimated from observations.

    Args:
        mode_history: Observation history
        modes: List of mode_ids

    Returns:
        Transition matrix (num_modes x num_modes)
    """
    num_modes = len(modes)
    mode_to_idx = {m: i for i, m in enumerate(modes)}

    # Count transitions
    counts = np.zeros((num_modes, num_modes))
    observations = mode_history.observed_modes

    for i in range(len(observations) - 1):
        _, mode_from = observations[i]
        _, mode_to = observations[i + 1]

        if mode_from in mode_to_idx and mode_to in mode_to_idx:
            idx_from = mode_to_idx[mode_from]
            idx_to = mode_to_idx[mode_to]
            counts[idx_from, idx_to] += 1

    # Normalize rows to get probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero

    transition_matrix = counts / row_sums

    # For rows with no observations, use uniform distribution
    zero_rows = (counts.sum(axis=1) == 0)
    transition_matrix[zero_rows, :] = 1.0 / num_modes

    return transition_matrix
