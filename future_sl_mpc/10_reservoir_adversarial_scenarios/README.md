# Reservoir-Computing Adversarial Scenarios for SHMPC

Same pipeline as the GAN setup: generate **adversarial obstacle trajectories** (relative waypoints `dx, dy`) for use as SHMPC scenarios. Here a **reservoir computer** (echo state network) is used instead of a GAN.

## Idea

- **Reservoir:** Fixed random recurrent layer; only the **readout** is trained.
- **Training:** Readout is trained so that the generated trajectory minimizes distance to the nominal ego path (adversarial / threatening).
- **Export:** Same CSV format as GAN (`scenario_id, obstacle_id, k, dx, dy`) so the C++ loader (`load_scenarios_from_gan_csv`) works unchanged—use `reservoir_scenarios.csv` and method **SHMPC_Reservoir**.

## Usage

### Generate scenario CSV

```bash
python3 future_sl_mpc/10_reservoir_adversarial_scenarios/reservoir_adversarial.py \
  --num 30 --horizon 20 --out future_sl_mpc/experiments/results/reservoir_scenarios.csv
```

### Run SHMPC with reservoir scenarios

Set `config.reservoir_scenario_csv_path` and use method **SHMPC_Reservoir** in the experiment runner (same C++ loader as GAN; CSV format is identical).

## Tests

```bash
python3 future_sl_mpc/10_reservoir_adversarial_scenarios/run_tests.py
```
