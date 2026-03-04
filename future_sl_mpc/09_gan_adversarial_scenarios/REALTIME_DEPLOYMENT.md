# GAN Real-Time Deployment: Recommended Configuration

**Goal:** ≥50% collision improvement vs SHMPC while meeting real-time solve requirements for self-driving (<10 ms per solve; industry target sub-20 ms).

## Sweep Results (35 rollouts per config)

| S (scenarios) | Collision rate | Solve time (ms) | Rel. improve | Meets ≥50% | Meets <10ms |
|---------------|----------------|-----------------|--------------|------------|-------------|
| 8             | 34.3%          | 3.49            | −71%         | No         | Yes         |
| 10            | 17.1%          | 3.82            | +14%         | No         | Yes         |
| **12**        | **2.9%**       | **4.80**        | **+86%**     | **Yes**    | **Yes**     |
| 15            | 5.7%           | 5.67            | +71%         | Yes        | Yes         |
| 18            | 5.7%           | 7.36            | +71%         | Yes        | Yes         |
| 20            | 8.6%           | 8.33            | +57%         | Yes        | Yes         |
| 25            | 2.9%           | 10.88           | +86%         | Yes        | No          |
| 30            | 2.9%           | 12.99           | +86%         | Yes        | No          |

## Recommended Method: **SHMPC_GAN_Reduced** (S=12)

- **Collision:** ~3% (vs 20% SHMPC) → **~86% relative improvement**
- **Solve time:** ~4.8 ms (well under 10 ms)
- **Config:** `gan_reduced_num_scenarios = 12` (default in `ExperimentConfig`)

## Usage

```cpp
// C++ config
config.gan_scenario_csv_path = "path/to/gan_scenarios.csv";
config.gan_reduced_num_scenarios = 12;  // default
// Use method "SHMPC_GAN_Reduced"
```

Generate scenarios:
```bash
python3 future_sl_mpc/09_gan_adversarial_scenarios/gan_adversarial.py --num 30 --horizon 20 --out results/gan_scenarios.csv
```

Run sweep to re-validate on your setup:
```bash
cd cpp_mpc/build && ./run_gan_realtime_sweep ../../future_sl_mpc/experiments/results/ 40
```

## Trade-off

- **S=8, 10:** Too few scenarios → safety drops below 50% improvement.
- **S=12–20:** All meet both constraints; **S=12** is fastest.
- **S=25, 30:** Best safety but solve time exceeds 10 ms (marginal for real-time).

For self-driving deployment, use **SHMPC_GAN_Reduced** with the default 12 scenarios.
