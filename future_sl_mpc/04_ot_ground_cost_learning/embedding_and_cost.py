"""
Planner-aligned trajectory embedding and mode-to-mode cost matrix (Section 6.4).
Trajectories as (T+1, 2) positions; safety label = min distance to ego over time.
Contrastive-style: similar margin -> small distance, different margin -> large distance.
Output: cost matrix D[i][j] for modes (CSV or return as list of lists).
"""

import numpy as np
import csv
from pathlib import Path
from typing import List, Tuple, Optional


def trajectory_to_feature(traj: np.ndarray) -> np.ndarray:
    """Flatten trajectory (T+1, 2) to feature vector."""
    return traj.flatten()


def safety_margin(obs_traj: np.ndarray, ego_traj: np.ndarray, combined_r: float) -> float:
    """Minimum distance between obstacle and ego over time minus combined radius."""
    T = min(len(obs_traj), len(ego_traj))
    if T == 0:
        return 0.0
    dists = np.linalg.norm(obs_traj[:T] - ego_traj[:T], axis=1)
    return float(np.min(dists) - combined_r)


def learned_embedding(
    traj: np.ndarray,
    margin: float,
    ref_margins: Optional[np.ndarray] = None,
    dim: int = 8,
) -> np.ndarray:
    """
    Simple planner-aligned embedding: concat flattened positions + margin.
    In a full implementation we would train a small NN with contrastive loss.
    """
    flat = trajectory_to_feature(traj)
    # Pad or truncate to fixed size
    max_len = dim - 1
    if len(flat) > max_len:
        flat = flat[:max_len]
    else:
        flat = np.pad(flat, (0, max_len - len(flat)))
    phi = np.concatenate([flat, [margin]])
    return phi.astype(np.float64)


def compute_cost_matrix_from_trajectories(
    mode_trajectories: List[np.ndarray],
    ego_trajectory: np.ndarray,
    combined_radius: float,
    mode_ids: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute mode-to-mode cost matrix D[i][j] = ||phi(tau_i) - phi(tau_j)||_2.
    phi is the learned (or placeholder) embedding using safety margin.
    """
    margins = [
        safety_margin(traj, ego_trajectory, combined_radius)
        for traj in mode_trajectories
    ]
    embeddings = [
        learned_embedding(traj, m) for traj, m in zip(mode_trajectories, margins)
    ]
    M = len(embeddings)
    D = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            D[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])
    ids = mode_ids or [str(i) for i in range(M)]
    return D, ids


def write_cost_matrix_csv(
    D: np.ndarray,
    mode_ids: List[str],
    path: str,
) -> None:
    """Write cost matrix to CSV (first row = header of mode_ids, then rows of D)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + mode_ids)
        for i, mid in enumerate(mode_ids):
            w.writerow([mid] + D[i].tolist())


def run_tests() -> bool:
    """Preliminary tests: symmetry, non-negativity, larger cost when margins differ."""
    np.random.seed(42)
    T, combined_r = 10, 2.0
    ego = np.cumsum(np.random.randn(T + 1, 2) * 0.1, axis=0)

    traj1 = np.cumsum(np.random.randn(T + 1, 2) * 0.1, axis=0)  # similar to ego -> small margin
    traj2 = traj1 + 5.0  # far -> large margin
    traj3 = traj1 + 0.1  # close -> small margin

    combined_r = 2.0
    D, ids = compute_cost_matrix_from_trajectories(
        [traj1, traj2, traj3], ego, combined_r
    )
    assert np.allclose(D, D.T), "Cost matrix should be symmetric"
    assert np.all(D >= -1e-10), "Cost matrix should be non-negative"
    # traj2 is far from ego so margin large; traj1 and traj3 closer -> D[0,1] and D[1,2] larger
    assert D[0, 1] > D[0, 2] or D[1, 0] > D[2, 0]
    write_cost_matrix_csv(D, ids, Path(__file__).parent / "out_cost_matrix.csv")
    print("OT ground cost learning tests passed.")
    return True


if __name__ == "__main__":
    run_tests()
