#!/usr/bin/env python3
"""Smoke tests for GAN adversarial scenario generator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from gan_adversarial import (
    SimpleGenerator,
    generate_adversarial_scenarios,
    write_gan_scenarios_csv,
    min_distance_to_path,
    _nominal_ego_path,
    HORIZON_DEFAULT,
)


def test_generator_output_shape():
    gen = SimpleGenerator(horizon=10, rng=np.random.default_rng(42))
    traj = gen.forward(np.random.standard_normal(32))
    assert traj.shape == (11, 2), f"expected (11, 2), got {traj.shape}"
    batch = gen.generate(batch=3)
    assert batch.shape == (3, 11, 2)


def test_generate_and_export():
    scenarios = generate_adversarial_scenarios(5, horizon=10, train_steps=50, rng=np.random.default_rng(123))
    assert len(scenarios) == 5
    assert all(s.shape == (11, 2) for s in scenarios)
    out = Path(__file__).parent / "test_gan_out.csv"
    write_gan_scenarios_csv(scenarios, str(out), obstacle_id=0, horizon=10)
    assert out.exists()
    lines = out.read_text().strip().split("\n")
    assert lines[0] == "scenario_id,obstacle_id,k,dx,dy"
    assert len(lines) == 1 + 5 * 11  # header + 5 scenarios * 11 steps
    out.unlink(missing_ok=True)


def test_min_distance():
    ego = _nominal_ego_path(5)
    obs = np.zeros((6, 2))
    obs[:, 0] = [0, 1, 2, 3, 4, 5]  # same path
    d = min_distance_to_path(obs, ego)
    assert d >= 0
    obs_far = np.ones((6, 2)) * 10
    d_far = min_distance_to_path(obs_far, ego)
    assert d_far > d


if __name__ == "__main__":
    test_generator_output_shape()
    test_generate_and_export()
    test_min_distance()
    print("All tests passed.")
