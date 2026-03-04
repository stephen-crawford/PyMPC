"""
Reservoir-computing (echo state network) adversarial obstacle trajectory generator for SHMPC.

Same role as the GAN setup: produce trajectory waypoints (dx, dy) that are adversarial
(minimize distance to nominal ego path). Only the readout is trained; the reservoir is fixed.

Export: same CSV format as GAN (scenario_id, obstacle_id, k, dx, dy) for C++ load_scenarios_from_gan_csv.
"""

import csv
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

DT = 0.1
HORIZON_DEFAULT = 20


def _nominal_ego_path(horizon: int, ego_v: float = 1.5) -> np.ndarray:
    """Nominal ego path: straight line along x."""
    steps = horizon + 1
    path = np.zeros((steps, 2))
    for k in range(steps):
        path[k, 0] = ego_v * k * DT
        path[k, 1] = 0.0
    return path


def min_distance_to_path(obs_traj: np.ndarray, ego_path: np.ndarray) -> float:
    """Minimum distance between obstacle trajectory and ego path over time."""
    T = min(len(obs_traj), len(ego_path))
    if T == 0:
        return 1e6
    dists = np.linalg.norm(obs_traj[:T] - ego_path[:T], axis=1)
    return float(np.min(dists))


class ReservoirReadout:
    """
    Echo state reservoir: fixed recurrent dynamics, trainable readout.
    State r(t+1) = (1 - leak) * r(t) + leak * tanh(W_res @ r(t) + W_in @ u(t)).
    Output at step k: y_k = W_out @ r_k  -> (dx_k, dy_k). Only W_out is trained.
    """

    def __init__(
        self,
        reservoir_size: int = 100,
        input_dim: int = 4,
        leak: float = 0.3,
        spectral_radius: float = 0.9,
        rng: Optional[np.random.Generator] = None,
    ):
        self.reservoir_size = reservoir_size
        self.input_dim = input_dim
        self.leak = leak
        self.rng = rng or np.random.default_rng()
        # Fixed reservoir: sparse random matrix, scale to spectral radius
        density = 0.1
        W = self.rng.standard_normal((reservoir_size, reservoir_size))
        W[self.rng.random((reservoir_size, reservoir_size)) > density] = 0
        try:
            radius = np.max(np.abs(np.linalg.eigvals(W)))
            if radius > 1e-10:
                W = W * (spectral_radius / radius)
        except np.linalg.LinAlgError:
            pass
        self.W_res = W
        self.W_in = self.rng.standard_normal((input_dim, reservoir_size)) * 0.5
        # Trainable readout: (reservoir_size,) -> (2,) per step; we use one W_out for all steps
        self.W_out = self.rng.standard_normal((reservoir_size, 2)) * 0.02

    def run_reservoir(
        self,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """inputs: (T, input_dim). Returns reservoir states (T+1, reservoir_size)."""
        T = inputs.shape[0]
        r = np.zeros((T + 1, self.reservoir_size))
        for t in range(T):
            r[t + 1] = (1 - self.leak) * r[t] + self.leak * np.tanh(
                r[t] @ self.W_res.T + inputs[t] @ self.W_in
            )
        return r

    def trajectory_from_states(self, states: np.ndarray) -> np.ndarray:
        """states: (T+1, reservoir_size). Returns (T+1, 2) trajectory (dx, dy)."""
        return (states @ self.W_out).astype(np.float64)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """inputs: (T, input_dim) with T = horizon. Returns trajectory (T+1, 2) = (horizon+1, 2)."""
        r = self.run_reservoir(inputs)
        # Output (horizon+1, 2): use r_0..r_T with T = inputs.shape[0]-1 so we have inputs.shape[0] states = horizon+1
        n_use = inputs.shape[0]  # inputs are (horizon+1, dim), we want (horizon+1) output points
        traj = self.trajectory_from_states(r[:n_use])
        # Scale to plausible range (meters)
        traj = np.tanh(traj / 5.0) * 12.0
        return traj


def train_readout_adversarial(
    reservoir: ReservoirReadout,
    horizon: int,
    num_steps: int = 400,
    lr: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Train W_out so that generated trajectories minimize distance to ego path (adversarial)."""
    rng = rng or np.random.default_rng()
    ego_path = _nominal_ego_path(horizon)
    steps = horizon + 1
    for _ in range(num_steps):
        # Random input drive (time + noise)
        inputs = np.zeros((steps, reservoir.input_dim))
        inputs[:, 0] = np.linspace(0, 1, steps)
        inputs += rng.standard_normal((steps, reservoir.input_dim)) * 0.2
        states = reservoir.run_reservoir(inputs)
        traj_raw = states @ reservoir.W_out
        traj = np.tanh(traj_raw / 5.0) * 12.0
        dist = min_distance_to_path(traj, ego_path)
        loss = dist  # we want to minimize distance
        # Gradient of loss w.r.t. traj: d(dist)/d(traj)
        # Approximate: numerical or use min-over-time index
        T = min(len(traj), len(ego_path))
        diffs = traj[:T] - ego_path[:T]
        dists = np.linalg.norm(diffs, axis=1)
        k_min = int(np.argmin(dists))
        if dists[k_min] < 1e-8:
            continue
        grad_traj = np.zeros_like(traj)
        grad_traj[k_min] = diffs[k_min] / (dists[k_min] + 1e-8)
        # Gradient through tanh scale: d(traj)/d(traj_raw) = (12/5) * (1 - tanh^2(traj_raw/5))
        scale = (12.0 / 5.0) * (1.0 - np.tanh(traj_raw / 5.0) ** 2)
        grad_raw = grad_traj * scale
        # grad_raw = (horizon+1, 2). d(traj_raw)/d(W_out) = states so grad_W_out = states.T @ grad_raw
        grad_W_out = states.T @ grad_raw
        reservoir.W_out -= lr * grad_W_out
        reservoir.W_out = np.clip(reservoir.W_out, -2.0, 2.0)


def generate_reservoir_scenarios(
    num_scenarios: int,
    horizon: int = HORIZON_DEFAULT,
    obstacle_id: int = 0,
    train_steps: int = 400,
    reservoir_size: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """Train reservoir readout and sample num_scenarios adversarial trajectories."""
    rng = rng or np.random.default_rng()
    reservoir = ReservoirReadout(reservoir_size=reservoir_size, rng=rng)
    train_readout_adversarial(reservoir, horizon, num_steps=train_steps, rng=rng)
    steps = horizon + 1
    scenarios = []
    for _ in range(num_scenarios):
        inputs = np.zeros((steps, reservoir.input_dim))
        inputs[:, 0] = np.linspace(0, 1, steps)
        inputs += rng.standard_normal((steps, reservoir.input_dim)) * 0.3
        traj = reservoir.forward(inputs)
        scenarios.append(traj)
    return scenarios


def write_scenarios_csv(
    scenarios: List[np.ndarray],
    out_path: str,
    obstacle_id: int = 0,
    horizon: Optional[int] = None,
) -> None:
    """Same format as GAN: scenario_id, obstacle_id, k, dx, dy."""
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
    ap = argparse.ArgumentParser(description="Generate reservoir adversarial obstacle scenarios for SHMPC")
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=HORIZON_DEFAULT)
    ap.add_argument("--obstacle_id", type=int, default=0)
    ap.add_argument("--train_steps", type=int, default=400)
    ap.add_argument("--reservoir_size", type=int, default=100)
    ap.add_argument("--out", default="reservoir_scenarios.csv")
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    scenarios = generate_reservoir_scenarios(
        args.num,
        horizon=args.horizon,
        obstacle_id=args.obstacle_id,
        train_steps=args.train_steps,
        reservoir_size=args.reservoir_size,
        rng=rng,
    )
    write_scenarios_csv(scenarios, args.out, obstacle_id=args.obstacle_id, horizon=args.horizon)
    print(f"Wrote {len(scenarios)} scenarios to {args.out} (horizon={args.horizon})")


if __name__ == "__main__":
    main()
