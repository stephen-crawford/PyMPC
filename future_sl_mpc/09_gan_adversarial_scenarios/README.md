# GAN Adversarial Obstacle Scenarios for SHMPC

This module trains a small **generator** to produce **adversarial obstacle trajectories** (relative waypoints `dx, dy`) that are both plausible and tend to get close to a nominal ego path. These trajectories are exported to a CSV that the C++ SHMPC pipeline can load and use as scenarios.

## Idea

- **Generator:** Maps noise → trajectory `(horizon+1) x 2` (displacements from obstacle start).
- **Training:** (1) Outputs are encouraged to resemble synthetic “real” obstacle motion; (2) an adversarial term encourages trajectories that minimize distance to the nominal ego path (threatening scenarios).
- **Export:** CSV columns `scenario_id, obstacle_id, k, dx, dy`. The C++ loader adds current obstacle position so that waypoints become absolute positions per step.

## Usage

### 1. Generate scenario CSV (Python)

```bash
cd future_sl_mpc/09_gan_adversarial_scenarios
python3 gan_adversarial.py --num 30 --horizon 20 --out ../experiments/results/gan_scenarios.csv
```

Options: `--num` (number of scenarios), `--horizon` (prediction steps), `--train_steps`, `--obstacle_id`, `--seed`.

### 2. Run SHMPC with GAN scenarios (C++)

Point the experiment config to the CSV (e.g. `gan_scenario_csv_path`) and use method **SHMPC_GAN**. The C++ loader reads the CSV and, at each step, builds scenarios by adding the current obstacle state to `(dx, dy)`, then passes them to the controller via `set_scenarios()`.

### 3. CSV format (for C++)

| Column       | Description                          |
|-------------|--------------------------------------|
| scenario_id | Scenario index (0, 1, …)             |
| obstacle_id | Obstacle index (e.g. 0)              |
| k           | Timestep index (0 … horizon)         |
| dx, dy      | Position offset from obstacle start  |

C++ builds mean position at step `k` as `(obstacle.x + dx, obstacle.y + dy)`.

## Tests

```bash
python3 run_tests.py
```
