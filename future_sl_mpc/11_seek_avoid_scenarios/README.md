# 11. Seek-Avoid (Pursuit) Scenarios

Obstacles are modeled as **actively pursuing** the vehicle: at each step the obstacle moves toward the (current or lookahead) ego position. This encodes a seek-avoid game where the vehicle follows the path and obstacles chase it.

## Model

- **Ego path**: Nominal straight line along x at constant speed (configurable).
- **Obstacle**: Starts at `(ox, oy)`. At step k, moves toward `ego_path[k + lookahead]` at speed `v_pursuit`.
- **Output**: Trajectory `(dx, dy)` relative to obstacle start over horizon; exported in the same CSV format as GAN/Reservoir for the C++ loader.

## Usage

```bash
python3 seek_avoid.py --num 30 --horizon 20 --out seek_avoid_scenarios.csv
```

Scenarios vary in pursuit speed, lookahead, and obstacle start position. Use the output CSV as `seek_avoid_scenario_csv_path` in the experiment config for method **SHMPC_SeekAvoid**.

## Tests

```bash
python3 run_tests.py
```
