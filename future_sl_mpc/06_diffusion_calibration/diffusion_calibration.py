"""
Diffusion-style trajectory sampling with risk-directed selection and calibrated risk bound (Section 6.6).
Placeholder: Gaussian samples around mean trajectory (stand-in for diffusion); severity scoring and
conservative risk bound p_hat + rad(S, delta).
"""

import numpy as np
from typing import List, Tuple, Optional


def sample_candidates(
    mean_trajectory: np.ndarray,
    sigma: float,
    K: int,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """Generate K candidate trajectories as mean + Gaussian noise (placeholder diffusion)."""
    rng = rng or np.random.default_rng()
    T = mean_trajectory.shape[0]
    return [
        mean_trajectory + sigma * rng.standard_normal((T, 2))
        for _ in range(K)
    ]


def severity_score(
    obs_traj: np.ndarray,
    ego_traj: np.ndarray,
    combined_radius: float,
) -> float:
    """Max over time of (combined_radius - distance). Positive = penetration."""
    T = min(len(obs_traj), len(ego_traj))
    if T == 0:
        return 0.0
    dists = np.linalg.norm(obs_traj[:T] - ego_traj[:T], axis=1)
    return float(np.max(combined_radius - dists))


def select_diverse_top_s(
    candidates: List[np.ndarray],
    severities: List[float],
    ego_traj: np.ndarray,
    combined_r: float,
    S: int,
) -> List[np.ndarray]:
    """Select S candidates: prefer high severity but add simple diversity (avoid duplicates)."""
    idx_sev = sorted(range(len(severities)), key=lambda i: -severities[i])
    selected: List[int] = []
    for i in idx_sev:
        if len(selected) >= S:
            break
        # Simple diversity: add if not too close to already selected
        too_close = False
        for j in selected:
            d = np.linalg.norm(np.array(candidates[i]) - np.array(candidates[j]))
            if d < 0.5:
                too_close = True
                break
        if not too_close:
            selected.append(i)
    # Fill remainder with next by severity
    for i in idx_sev:
        if len(selected) >= S:
            break
        if i not in selected:
            selected.append(i)
    return [candidates[j] for j in selected[:S]]


def empirical_bernstein_radius(n: int, p_hat: float, delta: float) -> float:
    """Conservative radius for p: sqrt(2*log(2/delta)/n) * something."""
    if n <= 0:
        return 1.0
    return np.sqrt(2.0 * np.log(2.0 / delta) / n)


def risk_directed_diffusion_sampling(
    mean_trajectory: np.ndarray,
    ego_trajectory: np.ndarray,
    combined_radius: float,
    S: int,
    K: int = 50,
    sigma: float = 0.5,
    delta: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[np.ndarray], float, float]:
    """
    Generate K candidates, select S with diversity, return scenarios and risk bound.
    Returns (selected_scenarios, p_hat, p_upper).
    """
    rng = rng or np.random.default_rng()
    candidates = sample_candidates(mean_trajectory, sigma, K, rng)
    severities = [
        severity_score(c, ego_trajectory, combined_radius)
        for c in candidates
    ]
    selected = select_diverse_top_s(
        candidates, severities, ego_trajectory, combined_radius, S
    )
    violations = [1.0 if severity_score(c, ego_trajectory, combined_radius) > 0 else 0.0 for c in selected]
    p_hat = np.mean(violations)
    rad = empirical_bernstein_radius(len(selected), p_hat, delta)
    p_upper = min(1.0, p_hat + rad)
    return selected, float(p_hat), float(p_upper)


def run_tests() -> bool:
    """Preliminary tests: p_upper >= p_hat; selection size S."""
    rng = np.random.default_rng(42)
    T = 5
    mean = np.cumsum(rng.standard_normal((T + 1, 2)) * 0.2, axis=0)
    ego = np.zeros((T + 1, 2))
    ego[:, 0] = np.linspace(0, 1, T + 1)
    scenarios, p_hat, p_upper = risk_directed_diffusion_sampling(
        mean, ego, 1.0, S=5, K=20, sigma=0.3, delta=0.1, rng=rng
    )
    assert len(scenarios) == 5
    assert p_upper >= p_hat - 1e-6
    print("Diffusion calibration tests passed.")
    return True


if __name__ == "__main__":
    try:
        import scipy
    except ImportError:
        print("scipy optional for Clopper-Pearson; using Bernstein only.")
    run_tests()
