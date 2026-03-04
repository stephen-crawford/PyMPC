"""
GAN-based adversarial obstacle trajectory generator for SHMPC scenarios (Section 6.x extension).

Generates trajectory waypoints (dx, dy) relative to obstacle start. Trained so that:
- Outputs look like plausible obstacle motion (discriminator vs synthetic "real" trajectories).
- Optionally adversarial: generator is encouraged to produce trajectories that get close to
  a nominal ego path (threatening scenarios).

Export: CSV with columns scenario_id, obstacle_id, k, dx, dy for loading in C++ SHMPC.
"""

import csv
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

DT = 0.1
HORIZON_DEFAULT = 20


def _synthetic_real_trajectory(
    horizon: int,
    rng: np.random.Generator,
    x_range: Tuple[float, float] = (0.0, 25.0),
    y_range: Tuple[float, float] = (-2.0, 2.0),
) -> np.ndarray:
    """Generate a plausible obstacle trajectory (straight or gentle curve) as 'real' data."""
    # Start at origin (relative), drift in x and possibly y
    steps = horizon + 1
    traj = np.zeros((steps, 2))
    vx = rng.uniform(0.0, 2.0)
    vy = rng.uniform(-0.5, 0.5)
    for k in range(1, steps):
        traj[k, 0] = traj[k - 1, 0] + vx * DT + rng.uniform(-0.05, 0.05)
        traj[k, 1] = traj[k - 1, 1] + vy * DT + rng.uniform(-0.05, 0.05)
        # Slight persistence change
        vy = 0.95 * vy + rng.uniform(-0.1, 0.1)
    return traj


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


class SimpleGenerator:
    """MLP-style generator: noise -> (horizon+1) x 2 trajectory (dx, dy)."""

    def __init__(
        self,
        horizon: int = HORIZON_DEFAULT,
        noise_dim: int = 32,
        hidden: int = 64,
        rng: Optional[np.random.Generator] = None,
    ):
        self.horizon = horizon
        self.noise_dim = noise_dim
        self.out_dim = (horizon + 1) * 2
        self.rng = rng or np.random.default_rng()
        # Weights: input (noise_dim) -> hidden -> hidden -> out_dim
        self.W1 = self.rng.standard_normal((noise_dim, hidden)) * 0.1
        self.b1 = np.zeros(hidden)
        self.W2 = self.rng.standard_normal((hidden, hidden)) * 0.1
        self.b2 = np.zeros(hidden)
        self.W3 = self.rng.standard_normal((hidden, self.out_dim)) * 0.05
        self.b3 = np.zeros(self.out_dim)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def forward(self, z: np.ndarray) -> np.ndarray:
        """z: (noise_dim,) -> traj (horizon+1, 2)."""
        h = self._relu(z @ self.W1 + self.b1)
        h = self._relu(h @ self.W2 + self.b2)
        out = h @ self.W3 + self.b3
        traj = out.reshape(self.horizon + 1, 2)
        # Scale to plausible displacement range (meters over horizon)
        traj = np.tanh(traj) * 15.0  # roughly -15..+15 m in x and y
        return traj

    def generate(self, batch: int = 1) -> np.ndarray:
        """Generate batch trajectories. Returns (batch, horizon+1, 2)."""
        z = self.rng.standard_normal((batch, self.noise_dim))
        return np.array([self.forward(z[i]) for i in range(batch)])


def train_generator_adversarial(
    generator: SimpleGenerator,
    num_steps: int = 500,
    horizon: int = HORIZON_DEFAULT,
    lambda_adv: float = 0.5,
    lr: float = 1e-3,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    Train generator so that (1) outputs look like synthetic real trajectories,
    (2) optionally minimize min distance to nominal ego path (adversarial).
    Uses simple gradient-free optimization: perturb weights to decrease loss.
    """
    rng = rng or np.random.default_rng()
    ego_path = _nominal_ego_path(horizon)
    best_loss = 1e9
    for step in range(num_steps):
        z = rng.standard_normal((1, generator.noise_dim))
        traj = generator.forward(z[0])
        # Loss: (a) match distribution of real (mean squared vs sample real), (b) adversarial
        real_sample = _synthetic_real_trajectory(horizon, rng)
        loss_shape = np.mean((traj - real_sample) ** 2)
        dist = min_distance_to_path(traj, ego_path)
        loss_adv = 1.0 / (dist + 0.5)  # smaller dist -> larger loss -> we want to minimize dist
        loss = loss_shape + lambda_adv * loss_adv
        if loss < best_loss:
            best_loss = loss
        # Simple finite-difference update: perturb generator weights
        if step % 50 == 0 and step > 0:
            eps = lr * (0.95 ** (step // 50))
            for param in [generator.W3, generator.b3]:
                flat = param.flatten()
                for _ in range(min(5, flat.size)):
                    i = rng.integers(0, flat.size)
                    old = flat[i]
                    flat[i] = old + eps
                    new_traj = generator.forward(z[0])
                    new_loss = np.mean((new_traj - real_sample) ** 2) + lambda_adv / (min_distance_to_path(new_traj, ego_path) + 0.5)
                    if new_loss < loss:
                        loss = new_loss
                    else:
                        flat[i] = old
    return


def generate_adversarial_scenarios(
    num_scenarios: int,
    horizon: int = HORIZON_DEFAULT,
    obstacle_id: int = 0,
    train_steps: int = 300,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    Train a small generator and sample num_scenarios adversarial trajectories.
    Returns list of (horizon+1, 2) arrays (dx, dy) per scenario.
    """
    rng = rng or np.random.default_rng()
    gen = SimpleGenerator(horizon=horizon, rng=rng)
    train_generator_adversarial(gen, num_steps=train_steps, horizon=horizon, rng=rng)
    trajs = gen.generate(batch=num_scenarios)
    return [trajs[i] for i in range(num_scenarios)]


def write_gan_scenarios_csv(
    scenarios: List[np.ndarray],
    out_path: str,
    obstacle_id: int = 0,
    horizon: Optional[int] = None,
) -> None:
    """
    Write scenarios to CSV for C++ loader.
    Format: scenario_id, obstacle_id, k, dx, dy (relative waypoints from obstacle start).
    """
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
    return


def main():
    """Generate GAN adversarial scenarios and write CSV for SHMPC."""
    import argparse
    ap = argparse.ArgumentParser(description="Generate GAN adversarial obstacle scenarios for SHMPC")
    ap.add_argument("--num", type=int, default=30, help="Number of scenarios")
    ap.add_argument("--horizon", type=int, default=HORIZON_DEFAULT, help="Prediction horizon")
    ap.add_argument("--obstacle_id", type=int, default=0)
    ap.add_argument("--train_steps", type=int, default=300)
    ap.add_argument("--out", default="gan_scenarios.csv", help="Output CSV path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    scenarios = generate_adversarial_scenarios(
        args.num, horizon=args.horizon, obstacle_id=args.obstacle_id,
        train_steps=args.train_steps, rng=rng,
    )
    write_gan_scenarios_csv(scenarios, args.out, obstacle_id=args.obstacle_id, horizon=args.horizon)
    print(f"Wrote {len(scenarios)} scenarios to {args.out} (horizon={args.horizon})")


if __name__ == "__main__":
    main()
