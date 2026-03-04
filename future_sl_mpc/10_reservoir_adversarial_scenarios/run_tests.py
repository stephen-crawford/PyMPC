#!/usr/bin/env python3
"""Smoke tests for reservoir adversarial scenario generator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from reservoir_adversarial import (
    ReservoirReadout,
    generate_reservoir_scenarios,
    write_scenarios_csv,
    min_distance_to_path,
    _nominal_ego_path,
    HORIZON_DEFAULT,
)


def test_reservoir_forward():
    rng = np.random.default_rng(42)
    res = ReservoirReadout(reservoir_size=50, input_dim=4, rng=rng)
    steps = 11  # horizon+1 for horizon=10
    inputs = rng.standard_normal((steps, 4)) * 0.5
    traj = res.forward(inputs)
    assert traj.shape == (11, 2), f"expected (11, 2), got {traj.shape}"


def test_generate_and_export():
    scenarios = generate_reservoir_scenarios(5, horizon=10, train_steps=30, reservoir_size=40, rng=np.random.default_rng(123))
    assert len(scenarios) == 5
    assert all(s.shape == (11, 2) for s in scenarios)
    out = Path(__file__).parent / "test_reservoir_out.csv"
    write_scenarios_csv(scenarios, str(out), obstacle_id=0, horizon=10)
    assert out.exists()
    lines = out.read_text().strip().split("\n")
    assert lines[0] == "scenario_id,obstacle_id,k,dx,dy"
    assert len(lines) == 1 + 5 * 11
    out.unlink(missing_ok=True)


def test_min_distance():
    ego = _nominal_ego_path(5)
    obs = np.zeros((6, 2))
    obs[:, 0] = [0, 0.15, 0.3, 0.45, 0.6, 0.75]
    d = min_distance_to_path(obs, ego)
    assert d >= 0


if __name__ == "__main__":
    test_reservoir_forward()
    test_generate_and_export()
    test_min_distance()
    print("All tests passed.")
