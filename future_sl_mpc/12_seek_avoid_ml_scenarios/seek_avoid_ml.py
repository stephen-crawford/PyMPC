"""
ML model trained on seek-avoid (pursuit) game data to predict adversarial obstacle trajectories.
Uses seek-avoid to generate (context -> trajectory) data, trains a small MLP, then generates
scenarios by running the model. Export: same CSV format for C++ SHMPC.
"""

import csv
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

# Import seek-avoid data generation (same repo)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "11_seek_avoid_scenarios"))
from seek_avoid import (
    nominal_ego_path,
    pursuit_trajectory,
    generate_pursuit_scenarios,
    DT,
    HORIZON_DEFAULT,
)

HORIZON_DEFAULT_ML = 20


def generate_training_data(
    num_samples: int,
    horizon: int = HORIZON_DEFAULT_ML,
    ego_v: float = 1.5,
    pursuit_speed_range: Tuple[float, float] = (0.5, 2.5),
    lookahead_range: Tuple[int, int] = (0, 4),
    obs_x_range: Tuple[float, float] = (1.0, 18.0),
    obs_y_range: Tuple[float, float] = (-1.5, 1.5),
    noise_std: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate (context, trajectory) pairs from seek-avoid game.
    context: (obs_x, obs_y, pursuit_speed, lookahead) normalized to ~[0,1] scale.
    trajectory: (horizon+1)*2 flattened (dx, dy per step).
    Returns X (num_samples, 4), Y (num_samples, (horizon+1)*2).
    """
    rng = rng or np.random.default_rng()
    ego_path = nominal_ego_path(horizon, ego_v=ego_v)
    X_list, Y_list = [], []
    # Normalization for context (for training stability)
    scale_x, scale_y = 20.0, 3.0
    scale_v, scale_L = 2.5, 4.0
    for _ in range(num_samples):
        ox = rng.uniform(*obs_x_range)
        oy = rng.uniform(*obs_y_range)
        v_pursuit = rng.uniform(*pursuit_speed_range)
        lookahead = int(rng.integers(*lookahead_range))
        obs_start = np.array([ox, oy], dtype=float)
        traj = pursuit_trajectory(
            horizon, ego_path, obs_start, pursuit_speed=v_pursuit,
            lookahead_steps=lookahead, noise_std=noise_std, rng=rng,
        )
        context = np.array([
            ox / scale_x, oy / scale_y,
            v_pursuit / scale_v, lookahead / scale_L,
        ], dtype=np.float64)
        X_list.append(context)
        Y_list.append(traj.flatten())
    return np.array(X_list), np.array(Y_list)


class AdversarialPredictor:
    """MLP: context (4) -> trajectory (horizon+1)*2."""

    def __init__(
        self,
        horizon: int = HORIZON_DEFAULT_ML,
        hidden: int = 64,
        rng: Optional[np.random.Generator] = None,
    ):
        self.horizon = horizon
        self.out_dim = (horizon + 1) * 2
        self.rng = rng or np.random.default_rng()
        self.W1 = self.rng.standard_normal((4, hidden)) * 0.2
        self.b1 = np.zeros(hidden)
        self.W2 = self.rng.standard_normal((hidden, hidden)) * 0.15
        self.b2 = np.zeros(hidden)
        self.W3 = self.rng.standard_normal((hidden, self.out_dim)) * 0.1
        self.b3 = np.zeros(self.out_dim)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (4,) or (batch, 4). Returns (horizon+1, 2) or (batch, horizon+1, 2)."""
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]
        h = self._relu(x @ self.W1 + self.b1)
        h = self._relu(h @ self.W2 + self.b2)
        out = h @ self.W3 + self.b3
        out = out.reshape(-1, self.horizon + 1, 2)
        # Scale to plausible displacement (meters)
        out = np.tanh(out) * 15.0
        if single:
            return out[0]
        return out

    def predict_trajectory(self, obs_x: float, obs_y: float, v_pursuit: float = 1.2, lookahead: int = 1) -> np.ndarray:
        """Convenience: build context and return (horizon+1, 2) trajectory."""
        context = np.array([obs_x / 20.0, obs_y / 3.0, v_pursuit / 2.5, lookahead / 4.0], dtype=np.float64)
        return self.forward(context)


def train_predictor(
    predictor: AdversarialPredictor,
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Train MLP to minimize MSE between predicted and seek-avoid trajectories."""
    rng = rng or np.random.default_rng()
    N = X.shape[0]
    for ep in range(epochs):
        perm = rng.permutation(N)
        loss_sum = 0.0
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            x_b = X[idx]
            y_b = Y[idx]
            pred_flat = predictor.forward(x_b)
            pred_flat = pred_flat.reshape(pred_flat.shape[0], -1)
            loss = np.mean((pred_flat - y_b) ** 2)
            loss_sum += loss * len(idx)
            # Simple gradient: d/dW (pred - y)^2 -> 2(pred - y) * d_pred/d_W
            err = pred_flat - y_b  # (batch, out_dim)
            # Backprop through forward (simplified: one step gradient descent on output layer for speed)
            grad_out = 2.0 * err / (batch_size * err.shape[1])
            grad_out = grad_out.reshape(-1, predictor.horizon + 1, 2)
            h = np.maximum(0, x_b @ predictor.W1 + predictor.b1)
            h = np.maximum(0, h @ predictor.W2 + predictor.b2)
            # W3 update
            dW3 = h.T @ grad_out.reshape(grad_out.shape[0], -1)
            predictor.W3 -= lr * dW3
            predictor.b3 -= lr * grad_out.reshape(grad_out.shape[0], -1).sum(axis=0)
        if (ep + 1) % 50 == 0:
            lr *= 0.95
    return


def generate_ml_scenarios(
    num_scenarios: int,
    horizon: int = HORIZON_DEFAULT_ML,
    obstacle_id: int = 0,
    num_train: int = 400,
    train_epochs: int = 150,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """Generate training data, train predictor, sample num_scenarios trajectories from model."""
    rng = rng or np.random.default_rng()
    X, Y = generate_training_data(num_train, horizon=horizon, rng=rng)
    predictor = AdversarialPredictor(horizon=horizon, rng=rng)
    train_predictor(predictor, X, Y, epochs=train_epochs, rng=rng)
    scenarios = []
    for _ in range(num_scenarios):
        ox = rng.uniform(2.0, 15.0)
        oy = rng.uniform(-1.2, 1.2)
        v_p = rng.uniform(0.8, 2.2)
        L = int(rng.integers(0, 4))
        traj = predictor.predict_trajectory(ox, oy, v_p, L)
        scenarios.append(traj)
    return scenarios


def write_scenarios_csv(
    scenarios: List[np.ndarray],
    out_path: str,
    obstacle_id: int = 0,
    horizon: Optional[int] = None,
) -> None:
    """Same CSV format as GAN/Reservoir/SeekAvoid: scenario_id, obstacle_id, k, dx, dy."""
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
    ap = argparse.ArgumentParser(description="Generate ML (seek-avoid–trained) adversarial scenarios for SHMPC")
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=HORIZON_DEFAULT_ML)
    ap.add_argument("--obstacle_id", type=int, default=0)
    ap.add_argument("--train_samples", type=int, default=400)
    ap.add_argument("--train_epochs", type=int, default=150)
    ap.add_argument("--out", default="seek_avoid_ml_scenarios.csv")
    ap.add_argument("--seed", type=int, default=45)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    scenarios = generate_ml_scenarios(
        args.num,
        horizon=args.horizon,
        obstacle_id=args.obstacle_id,
        num_train=args.train_samples,
        train_epochs=args.train_epochs,
        rng=rng,
    )
    write_scenarios_csv(scenarios, args.out, obstacle_id=args.obstacle_id, horizon=args.horizon)
    print(f"Wrote {len(scenarios)} scenarios to {args.out} (horizon={args.horizon})")


if __name__ == "__main__":
    main()
