"""
Scenario sampling for adaptive scenario-based MPC.

Implements Algorithm 1 from guide.md: SampleScenarios

Generates scenarios by sampling mode sequences and noise realizations
for each obstacle, then combining into joint scenarios.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .types import (
    ObstacleState,
    ModeModel,
    ModeHistory,
    PredictionStep,
    ObstacleTrajectory,
    Scenario,
)
from .mode_weights import compute_mode_weights, sample_mode_from_weights, WeightType
from .trajectory_moments import compute_single_mode_trajectory


def sample_scenarios(
    obstacles: Dict[int, ObstacleState],
    mode_histories: Dict[int, ModeHistory],
    horizon: int,
    num_scenarios: int,
    weight_type: WeightType = WeightType.FREQUENCY,
    recency_decay: float = 0.9,
    current_timestep: int = 0,
    rng: Optional[np.random.Generator] = None
) -> List[Scenario]:
    """
    Sample scenarios following Algorithm 1 from guide.md.

    Algorithm 1: SampleScenarios
    Input: obstacles, mode_histories, num_scenarios S, horizon N
    Output: List of scenarios

    For each scenario s = 1, ..., S:
        For each obstacle o:
            1. Compute mode weights w_m from history
            2. Sample mode sequence m^(s) ~ Categorical(w)
            3. Sample noise sequence w_k ~ N(0, I)
            4. Propagate trajectory using sampled modes and noise

    Args:
        obstacles: Dict mapping obstacle_id to current ObstacleState
        mode_histories: Dict mapping obstacle_id to ModeHistory
        horizon: Prediction horizon N
        num_scenarios: Number of scenarios to sample S
        weight_type: Strategy for computing mode weights
        recency_decay: Decay factor for recency weighting
        current_timestep: Current timestep for recency computation
        rng: Random number generator

    Returns:
        List of Scenario objects
    """
    if rng is None:
        rng = np.random.default_rng()

    scenarios = []

    for s in range(num_scenarios):
        trajectories = {}

        for obs_id, obs_state in obstacles.items():
            if obs_id not in mode_histories:
                # No mode history - skip this obstacle
                continue

            mode_history = mode_histories[obs_id]

            # Step 1: Compute mode weights
            mode_weights = compute_mode_weights(
                mode_history,
                weight_type=weight_type,
                recency_decay=recency_decay,
                current_timestep=current_timestep
            )

            if not mode_weights:
                continue

            # Step 2 & 3 & 4: Sample trajectory
            trajectory = _sample_obstacle_trajectory(
                obs_id,
                obs_state,
                mode_history.available_modes,
                mode_weights,
                horizon,
                rng
            )

            trajectories[obs_id] = trajectory

        # Compute scenario probability as product of trajectory probabilities
        scenario_prob = 1.0
        for traj in trajectories.values():
            scenario_prob *= traj.probability

        scenarios.append(Scenario(
            scenario_id=s,
            trajectories=trajectories,
            probability=scenario_prob
        ))

    return scenarios


def _sample_obstacle_trajectory(
    obstacle_id: int,
    initial_state: ObstacleState,
    available_modes: Dict[str, ModeModel],
    mode_weights: Dict[str, float],
    horizon: int,
    rng: np.random.Generator
) -> ObstacleTrajectory:
    """
    Sample a single obstacle trajectory.

    Samples a mode and noise sequence, then propagates the trajectory.

    Args:
        obstacle_id: Obstacle identifier
        initial_state: Initial obstacle state
        available_modes: Dict of available mode models
        mode_weights: Weights for each mode
        horizon: Prediction horizon
        rng: Random number generator

    Returns:
        Sampled ObstacleTrajectory
    """
    # Sample mode for this trajectory (constant mode over horizon for simplicity)
    # Can be extended to sample mode sequence for each timestep
    sampled_mode_id = sample_mode_from_weights(mode_weights, rng)
    mode = available_modes[sampled_mode_id]

    # Sample noise sequence
    noise_dim = mode.noise_dim
    noise_samples = rng.standard_normal((horizon, noise_dim))

    # Propagate trajectory
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
        # Propagate with sampled noise
        noise = noise_samples[k]
        x = mode.A @ x + mode.b + mode.G @ noise

        # Update covariance (for uncertainty representation)
        cov = mode.A @ cov @ mode.A.T + mode.G @ mode.G.T

        steps.append(PredictionStep(
            k=k + 1,
            mean=x[:2].copy(),
            covariance=cov[:2, :2].copy()
        ))

    return ObstacleTrajectory(
        obstacle_id=obstacle_id,
        mode_id=sampled_mode_id,
        steps=steps,
        probability=mode_weights[sampled_mode_id]
    )


def sample_scenarios_with_mode_sequences(
    obstacles: Dict[int, ObstacleState],
    mode_histories: Dict[int, ModeHistory],
    horizon: int,
    num_scenarios: int,
    weight_type: WeightType = WeightType.FREQUENCY,
    rng: Optional[np.random.Generator] = None
) -> List[Scenario]:
    """
    Sample scenarios with time-varying mode sequences.

    More sophisticated version that samples a different mode for each timestep,
    allowing for mode switches during the prediction horizon.

    Args:
        obstacles: Dict mapping obstacle_id to current ObstacleState
        mode_histories: Dict mapping obstacle_id to ModeHistory
        horizon: Prediction horizon
        num_scenarios: Number of scenarios to sample
        weight_type: Strategy for computing mode weights
        rng: Random number generator

    Returns:
        List of Scenario objects
    """
    if rng is None:
        rng = np.random.default_rng()

    scenarios = []

    for s in range(num_scenarios):
        trajectories = {}

        for obs_id, obs_state in obstacles.items():
            if obs_id not in mode_histories:
                continue

            mode_history = mode_histories[obs_id]
            mode_weights = compute_mode_weights(mode_history, weight_type=weight_type)

            if not mode_weights:
                continue

            # Sample trajectory with mode switching
            trajectory = _sample_trajectory_with_mode_sequence(
                obs_id,
                obs_state,
                mode_history.available_modes,
                mode_weights,
                horizon,
                rng
            )

            trajectories[obs_id] = trajectory

        scenarios.append(Scenario(
            scenario_id=s,
            trajectories=trajectories,
            probability=1.0 / num_scenarios  # Uniform scenario weight
        ))

    return scenarios


def _sample_trajectory_with_mode_sequence(
    obstacle_id: int,
    initial_state: ObstacleState,
    available_modes: Dict[str, ModeModel],
    mode_weights: Dict[str, float],
    horizon: int,
    rng: np.random.Generator
) -> ObstacleTrajectory:
    """
    Sample trajectory with mode switching at each timestep.
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

    # Track which mode was used most (for trajectory labeling)
    mode_counts = {m: 0 for m in mode_weights}
    trajectory_prob = 1.0

    for k in range(horizon):
        # Sample mode for this timestep
        mode_id = sample_mode_from_weights(mode_weights, rng)
        mode = available_modes[mode_id]
        mode_counts[mode_id] += 1
        trajectory_prob *= mode_weights[mode_id]

        # Sample noise
        noise = rng.standard_normal(mode.noise_dim)

        # Propagate
        x = mode.A @ x + mode.b + mode.G @ noise
        cov = mode.A @ cov @ mode.A.T + mode.G @ mode.G.T

        steps.append(PredictionStep(
            k=k + 1,
            mean=x[:2].copy(),
            covariance=cov[:2, :2].copy()
        ))

    # Label trajectory with most frequent mode
    dominant_mode = max(mode_counts, key=mode_counts.get)

    return ObstacleTrajectory(
        obstacle_id=obstacle_id,
        mode_id=dominant_mode,
        steps=steps,
        probability=trajectory_prob
    )


def compute_required_scenarios(
    epsilon: float,
    beta: float,
    num_decision_vars: int
) -> int:
    """
    Compute required number of scenarios using Theorem 1 from guide.md.

    Theorem 1: For epsilon-chance constraint satisfaction with confidence 1-beta,
    the required number of scenarios is:
        S >= 2/epsilon * (ln(1/beta) + n_x)

    where n_x is the number of decision variables.

    Args:
        epsilon: Violation probability (1 - confidence_level)
        beta: Risk parameter
        num_decision_vars: Number of decision variables n_x

    Returns:
        Minimum number of scenarios required
    """
    return int(np.ceil(
        2.0 / epsilon * (np.log(1.0 / beta) + num_decision_vars)
    ))
