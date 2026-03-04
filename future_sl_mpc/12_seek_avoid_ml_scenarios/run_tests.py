"""Smoke tests for seek-avoid ML scenario generator."""

import numpy as np
from pathlib import Path

# Ensure 11_seek_avoid_scenarios is on path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "11_seek_avoid_scenarios"))

from seek_avoid_ml import (
    generate_training_data,
    AdversarialPredictor,
    train_predictor,
    generate_ml_scenarios,
    write_scenarios_csv,
    HORIZON_DEFAULT_ML,
)


def test_training_data():
    X, Y = generate_training_data(20, horizon=8, rng=np.random.default_rng(42))
    assert X.shape == (20, 4)
    assert Y.shape == (20, (8 + 1) * 2)


def test_predictor_forward():
    rng = np.random.default_rng(43)
    pred = AdversarialPredictor(horizon=10, rng=rng)
    x = np.array([0.2, 0.1, 0.5, 0.25])
    out = pred.forward(x)
    assert out.shape == (11, 2)
    out_batch = pred.forward(np.tile(x, (3, 1)))
    assert out_batch.shape == (3, 11, 2)


def test_train_and_generate():
    scenarios = generate_ml_scenarios(5, horizon=6, num_train=30, train_epochs=20, rng=np.random.default_rng(44))
    assert len(scenarios) == 5
    for s in scenarios:
        assert s.shape == (7, 2)


def test_write_csv():
    base = Path(__file__).parent / "_test_out"
    base.mkdir(parents=True, exist_ok=True)
    scenarios = generate_ml_scenarios(3, horizon=5, num_train=25, train_epochs=15, rng=np.random.default_rng(45))
    out = base / "seek_avoid_ml_test.csv"
    write_scenarios_csv(scenarios, str(out), obstacle_id=0, horizon=5)
    assert out.exists()
    lines = out.read_text().strip().split("\n")
    assert lines[0] == "scenario_id,obstacle_id,k,dx,dy"
    assert len(lines) >= 1 + 3 * 6


if __name__ == "__main__":
    test_training_data()
    test_predictor_forward()
    test_train_and_generate()
    test_write_csv()
    print("All seek-avoid ML tests passed.")
