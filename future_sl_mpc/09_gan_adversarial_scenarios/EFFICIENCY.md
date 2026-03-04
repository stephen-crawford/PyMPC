# GAN Approach: Computational Efficiency

The baseline **SHMPC_GAN** uses 30 scenarios from a GAN-generated CSV and has high solve time (~3× baseline) because of per-step file I/O and many constraints. The following changes improve efficiency and are wired into the C++ harness and experiments.

## 1. Cached loader (default on)

- **`GANScenarioCache`** in `cpp_mpc`: load the CSV **once per rollout**, then each step call **`materialize(obstacles, horizon, max_scenarios)`** to build `Scenario` objects from cached relative waypoints + current obstacle state. No file I/O inside the step loop.
- **Config:** `ExperimentConfig::use_gan_cache = true` (default). Set to `false` to fall back to loading from file every step (e.g. for debugging).

## 2. Fewer scenarios per solve

- **`gan_num_scenarios_override`**: If > 0, at most that many GAN scenarios are used per solve (smaller QP).
- **SHMPC_GAN_Reduced**: Uses **12** scenarios per solve (first 12 from the same CSV). In 25-rollout runs this often gave **similar or better** collision rate and **much lower** solve time (~4.8 ms vs ~13 ms for full GAN).

## 3. Quotient-space on GAN (SHMPC_GAN_Quotient)

- Load all GAN scenarios, then **`reduce_scenarios_quotient_space(..., K, horizon)`** to get K representatives; solve with K scenarios. Reduces constraint count; in some runs safety dropped (quotient representatives may not cover worst cases). Tune K or use for speed when some safety loss is acceptable.

## 4. Tests

- **C++:** `test_gan_efficiency` (in `cpp_mpc/tests/test_gan_efficiency.cpp`): writes a minimal CSV, loads via cache, materializes with two different obstacle states, checks `max_scenarios` cap. Run: `cd cpp_mpc/build && ./test_gan_efficiency`.
- **Experiments:** `./future_sl_experiments ... 25` includes SHMPC_GAN, SHMPC_GAN_Reduced, SHMPC_GAN_Quotient; compare collision rate, progress, and avg solve time in `analysis_summary.txt` and strategy plots.

## Summary

| Variant              | Scenarios/solve | Solve time (approx) | Safety (25-rollout run) |
|----------------------|------------------|----------------------|--------------------------|
| SHMPC_GAN            | 30 (cached)      | ~13 ms               | 8% collision             |
| SHMPC_GAN_Reduced    | 12               | ~4.8 ms              | 4% collision             |
| SHMPC_GAN_Quotient   | K (e.g. 12)      | ~5 ms                | run-dependent            |

**Recommendation:** Prefer **SHMPC_GAN_Reduced** when you want GAN-style adversarial scenarios with near-baseline solve time and similar or better safety.

**Real-time deployment:** For ≥50% collision improvement and <10 ms solve (self-driving), use **SHMPC_GAN_Reduced** with default 12 scenarios. Sweep validated in `REALTIME_DEPLOYMENT.md`.
