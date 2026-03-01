"""
Ego-conditioned (counterfactual) obstacle prediction and plan selection (Section 6.7).
Placeholder: obstacle mean response = f(ego_control); sample trajectories and pick min-risk plan.
"""

import numpy as np
from typing import List, Tuple, Optional


def obstacle_response_to_ego(
    base_mean: np.ndarray,
    ego_controls: np.ndarray,
    response_gain: float = 0.1,
) -> np.ndarray:
    """
    Placeholder: obstacle mean shifts in response to ego control (e.g. acceleration).
    base_mean: (T+1, 2), ego_controls: (T, 2) e.g. (a, delta).
    """
    T = base_mean.shape[0] - 1
    out = base_mean.copy()
    for t in range(min(T, len(ego_controls))):
        # Lateral response to ego steering
        out[t + 1] = out[t + 1] + response_gain * np.array([ego_controls[t, 1], 0.0])
    return out


def sample_obstacle_trajectories(
    mean_traj: np.ndarray,
    sigma: float,
    S: int,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """Sample S trajectories N(mean_traj, sigma^2 I)."""
    rng = rng or np.random.default_rng()
    return [
        mean_traj + sigma * rng.standard_normal(mean_traj.shape)
        for _ in range(S)
    ]


def violation_rate(
    obs_trajs: List[np.ndarray],
    ego_traj: np.ndarray,
    combined_radius: float,
) -> float:
    """Fraction of trajectories with min distance < combined_radius."""
    count = 0
    for obs in obs_trajs:
        T = min(len(obs), len(ego_traj))
        if T == 0:
            continue
        dists = np.linalg.norm(obs[:T] - ego_traj[:T], axis=1)
        if np.min(dists) < combined_radius:
            count += 1
    return count / len(obs_trajs) if obs_trajs else 0.0


def counterfactual_plan_selection(
    candidate_ego_controls: List[np.ndarray],
    ego_trajectory_from_control: callable,
    base_obstacle_mean: np.ndarray,
    response_gain: float,
    S: int,
    combined_radius: float,
    sigma: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, List[float]]:
    """
    For each candidate control, compute ego traj and conditional obstacle samples;
    estimate violation rate; return j_star = argmin p_j and list of p_hat_j.
    """
    rng = rng or np.random.default_rng()
    p_hats: List[float] = []
    for u in candidate_ego_controls:
        ego_traj = ego_trajectory_from_control(u)
        cond_mean = obstacle_response_to_ego(base_obstacle_mean, u, response_gain)
        obs_samples = sample_obstacle_trajectories(cond_mean, sigma, S, rng)
        p_hats.append(violation_rate(obs_samples, ego_traj, combined_radius))
    j_star = int(np.argmin(p_hats))
    return j_star, p_hats


def run_tests() -> bool:
    """Preliminary tests: j_star index valid; p_hats in [0,1]."""
    rng = np.random.default_rng(42)
    T = 5
    base = np.cumsum(rng.standard_normal((T + 1, 2)) * 0.1, axis=0)
    candidates = [
        np.zeros((T, 2)),
        np.column_stack([np.zeros(T), np.ones(T) * 0.5]),
    ]

    def ego_from_u(u: np.ndarray) -> np.ndarray:
        traj = np.zeros((T + 1, 2))
        for t in range(T):
            traj[t + 1] = traj[t] + (u[t] if t < len(u) else 0) * 0.1
        return traj

    j_star, p_hats = counterfactual_plan_selection(
        candidates, ego_from_u, base, 0.1, S=10, combined_radius=1.0, rng=rng
    )
    assert 0 <= j_star < len(candidates)
    assert all(0 <= p <= 1.01 for p in p_hats)
    print("Counterfactual intent tests passed.")
    return True


if __name__ == "__main__":
    run_tests()
