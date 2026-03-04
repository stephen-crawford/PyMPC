"""
Seek-avoid game: obstacles actively pursue the vehicle (pursuit/evasion).
Used to model adversarial obstacle behavior and generate scenarios for SHMPC.

Vehicle follows reference path; obstacles move toward (current or predicted) ego
position each step. Multiple scenarios from varying pursuit speed, noise, and lookahead.

Export: CSV scenario_id, obstacle_id, k, dx, dy (same format as GAN/Reservoir).
"""

import csv
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

DT = 0.1
HORIZON_DEFAULT = 20


def nominal_ego_path(
    horizon: int,
    ego_v: float = 1.5,
    x0: float = 0.0,
    y0: float = 0.0,
) -> np.ndarray:
    """Ego path over horizon: (horizon+1, 2). Straight line along x by default."""
    steps = horizon + 1
    path = np.zeros((steps, 2))
    for k in range(steps):
        path[k, 0] = x0 + ego_v * k * DT
        path[k, 1] = y0
    return path


def pursuit_trajectory(
    horizon: int,
    ego_path: np.ndarray,
    obs_start: np.ndarray,
    pursuit_speed: float,
    lookahead_steps: int = 0,
    noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Obstacle pursues ego: at each step k, obstacle moves toward ego_path[k + lookahead_steps].
    Returns (horizon+1, 2) trajectory in same frame as obs_start (dx, dy relative to start).
    """
    rng = rng or np.random.default_rng()
    steps = horizon + 1
    traj = np.zeros((steps, 2))
    traj[0] = obs_start.copy()
    L = min(lookahead_steps, steps - 1)
    for k in range(steps - 1):
        ego_idx = min(k + L, len(ego_path) - 1)
        target = ego_path[ego_idx]
        pos = traj[k]
        diff = target - pos
        dist = np.linalg.norm(diff)
        if dist > 1e-9:
            step = (diff / dist) * pursuit_speed * DT
        else:
            step = np.zeros(2)
        if noise_std > 0:
            step += rng.standard_normal(2) * noise_std
        traj[k + 1] = pos + step
    # Convert to relative (dx, dy) from obstacle start
    origin = traj[0].copy()
    for k in range(steps):
        traj[k] = traj[k] - origin
    return traj


def generate_pursuit_scenarios(
    num_scenarios: int,
    horizon: int = HORIZON_DEFAULT,
    obstacle_id: int = 0,
    ego_v: float = 1.5,
    pursuit_speed_range: Tuple[float, float] = (0.8, 2.5),
    lookahead_range: Tuple[int, int] = (0, 3),
    noise_std: float = 0.05,
    obs_start_range_x: Tuple[float, float] = (2.0, 15.0),
    obs_start_range_y: Tuple[float, float] = (-1.5, 1.5),
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    Generate multiple pursuit scenarios by varying pursuit speed, lookahead, and obstacle start.
    Returns list of (horizon+1, 2) arrays (dx, dy) per scenario.
    """
    rng = rng or np.random.default_rng()
    ego_path = nominal_ego_path(horizon, ego_v=ego_v)
    scenarios = []
    for _ in range(num_scenarios):
        v_pursuit = rng.uniform(*pursuit_speed_range)
        lookahead = int(rng.integers(*lookahead_range))
        ox = rng.uniform(*obs_start_range_x)
        oy = rng.uniform(*obs_start_range_y)
        obs_start = np.array([ox, oy], dtype=float)
        traj = pursuit_trajectory(
            horizon,
            ego_path,
            obs_start,
            pursuit_speed=v_pursuit,
            lookahead_steps=lookahead,
            noise_std=noise_std,
            rng=rng,
        )
        scenarios.append(traj)
    return scenarios


def write_scenarios_csv(
    scenarios: List[np.ndarray],
    out_path: str,
    obstacle_id: int = 0,
    horizon: Optional[int] = None,
) -> None:
    """Write scenarios to CSV for C++ loader. Format: scenario_id, obstacle_id, k, dx, dy."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H = horizon if horizon is not None else (scenarios[0].shape[0] - 1)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id", "obstacle_id", "k", "dx", "dy"])
        for sid, traj in enumerate(scenarios):
            steps = min(traj.shape[0], H + 1)
            for k in range(steps):
                dx, dy = float(traj[k, 0]), float(traj[k, 1])
                w.writerow([sid, obstacle_id, k, dx, dy])


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate seek-avoid (pursuit) scenarios for SHMPC")
    ap.add_argument("--num", type=int, default=30, help="Number of scenarios")
    ap.add_argument("--horizon", type=int, default=HORIZON_DEFAULT)
    ap.add_argument("--obstacle_id", type=int, default=0)
    ap.add_argument("--out", default="seek_avoid_scenarios.csv", help="Output CSV path")
    ap.add_argument("--seed", type=int, default=44)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    scenarios = generate_pursuit_scenarios(
        args.num,
        horizon=args.horizon,
        obstacle_id=args.obstacle_id,
        rng=rng,
    )
    write_scenarios_csv(scenarios, args.out, obstacle_id=args.obstacle_id, horizon=args.horizon)
    print(f"Wrote {len(scenarios)} scenarios to {args.out} (horizon={args.horizon})")


if __name__ == "__main__":
    main()
