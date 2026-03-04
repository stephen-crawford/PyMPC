# 12. Seek-Avoid ML Scenarios

Uses **seek-avoid (pursuit) game** to generate training data, then trains a small MLP to predict adversarial obstacle trajectories from context (obstacle start, pursuit speed, lookahead). The trained model is used to generate scenarios for SHMPC.

## Pipeline

1. **Data**: Run seek-avoid with random (obs_start, pursuit_speed, lookahead) to get many (context → trajectory) pairs.
2. **Train**: MLP maps context (4 dims) → trajectory (horizon+1, 2). MSE loss on trajectory.
3. **Inference**: Sample contexts, run model, export scenarios in same CSV format as GAN/Reservoir/SeekAvoid.

## Usage

```bash
python3 seek_avoid_ml.py --num 30 --horizon 20 --out seek_avoid_ml_scenarios.csv
```

Use the output as `seek_avoid_ml_scenario_csv_path` for method **SHMPC_SeekAvoidML**.

## Tests

```bash
python3 run_tests.py
```
