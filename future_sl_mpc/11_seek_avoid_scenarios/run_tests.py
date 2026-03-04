"""Smoke tests for seek-avoid (pursuit) scenario generator."""

import numpy as np
from seek_avoid import (
    nominal_ego_path,
    pursuit_trajectory,
    generate_pursuit_scenarios,
    write_scenarios_csv,
    DT,
    HORIZON_DEFAULT,
)


def test_ego_path():
    H = 10
    path = nominal_ego_path(H, ego_v=1.5)
    assert path.shape == (H + 1, 2)
    assert np.all(path[:, 1] == 0)
    assert path[0, 0] == 0 and path[-1, 0] == 1.5 * 10 * DT


def test_pursuit_trajectory():
    H = 10
    rng = np.random.default_rng(42)
    ego_path = nominal_ego_path(H, ego_v=1.5)
    obs_start = np.array([3.0, 0.5])
    traj = pursuit_trajectory(
        H, ego_path, obs_start, pursuit_speed=1.2,
        lookahead_steps=0, noise_std=0.0, rng=rng,
    )
    assert traj.shape == (H + 1, 2)
    assert np.allclose(traj[0], 0)  # relative to start
    # Trajectory should be bounded (pursuit toward ego)
    assert np.all(np.isfinite(traj))


def test_generate_pursuit_scenarios():
    scenarios = generate_pursuit_scenarios(5, horizon=10, rng=np.random.default_rng(43))
    assert len(scenarios) == 5
    for s in scenarios:
        assert s.shape == (11, 2)
        assert np.allclose(s[0], 0)


def test_write_csv(tmp_path=None):
    from pathlib import Path
    base = Path(tmp_path) if tmp_path else Path(__file__).parent / "_test_out"
    base.mkdir(parents=True, exist_ok=True)
    scenarios = generate_pursuit_scenarios(3, horizon=5, rng=np.random.default_rng(44))
    out = base / "seek_avoid_test.csv"
    write_scenarios_csv(scenarios, str(out), obstacle_id=0, horizon=5)
    assert out.exists()
    lines = out.read_text().strip().split("\n")
    assert lines[0] == "scenario_id,obstacle_id,k,dx,dy"
    # 3 scenarios * 6 steps = 18 data rows
    assert len(lines) == 1 + 3 * 6


if __name__ == "__main__":
    from pathlib import Path
    test_ego_path()
    test_pursuit_trajectory()
    test_generate_pursuit_scenarios()
    test_write_csv()
    print("All seek-avoid tests passed.")
