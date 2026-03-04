# Future SL MPC: Experiments vs SHMPC

Rollouts and analysis comparing **SHMPC** (safe-horizon scenario MPC baseline) to all extensions and Paradigm-Shift variants.

## Methods

**Baseline and DRO/RTA**
- **SHMPC**: Safe horizon enabled, no DRO.
- **SHMPC_DRO**: Safe horizon + fixed Wasserstein DRO.
- **SHMPC_AdaptiveDRO**: DRO with \( \rho(t) \) from `AdaptiveDROShift` (residual drift).
- **SHMPC_RTA**: SHMPC + runtime-assurance wrapper (fallback when distance &lt; threshold).

**Extension allocations (6.1–6.3)**
- **SHMPC_Conformal**: Boundary-aware scenario allocation from conformal boundary scores (rare mode favored).
- **SHMPC_Hazard**: Hazard reweighting + allocation with exploration floor when switch imminent.
- **SHMPC_Bandit**: UCB risk-directed allocation; bandit updated each step with (mode, collision).

**Certificate and compiler (7.1, 7.2)**
- **SHMPC_Certificate**: Tube radii (constant) applied to tighten constraints.
- **SHMPC_Compiler**: Initial 5 scenarios, then add worst-violating mode scenario and re-solve (up to 5 iterations).
- **CertificateFirst**: Same as SHMPC_Certificate (Paradigm-Shift 7.1).
- **ScenarioCompiler**: Same as SHMPC_Compiler (Paradigm-Shift 7.2).

**Adversarial scenario generation (GAN, Reservoir, Seek-Avoid, Seek-Avoid ML)**
- **SHMPC_GAN**: Scenarios from a GAN-generated CSV (cached per rollout for efficiency). Generate with `python3 future_sl_mpc/09_gan_adversarial_scenarios/gan_adversarial.py --num 30 --horizon 20 --out future_sl_mpc/experiments/results/gan_scenarios.csv`.
- **SHMPC_GAN_Reduced**: Same GAN scenarios but only **12** per solve. **Recommended for real-time deployment:** ≥50% collision improvement vs SHMPC, solve time ~5 ms (<10 ms). Config: `gan_reduced_num_scenarios` (default 12). Run `./run_gan_realtime_sweep` to sweep and validate.
- **SHMPC_GAN_Quotient**: GAN scenarios reduced via quotient-space to K representatives before solve (fewer constraints; tune K for safety).
- **SHMPC_Reservoir**: Scenarios from reservoir-computing (echo state); same CSV format. Generate with `python3 future_sl_mpc/10_reservoir_adversarial_scenarios/reservoir_adversarial.py --num 30 --horizon 20 --out future_sl_mpc/experiments/results/reservoir_scenarios.csv`.
- **SHMPC_SeekAvoid**: Pursuit game (obstacles actively chase the vehicle). Generate with `python3 future_sl_mpc/11_seek_avoid_scenarios/seek_avoid.py --num 30 --horizon 20 --out future_sl_mpc/experiments/results/seek_avoid_scenarios.csv`.
- **SHMPC_SeekAvoidML**: ML model trained on seek-avoid data to predict adversarial trajectories. Generate with `python3 future_sl_mpc/12_seek_avoid_ml_scenarios/seek_avoid_ml.py --num 30 --horizon 20 --out future_sl_mpc/experiments/results/seek_avoid_ml_scenarios.csv`.
- **SHMPC_QuotientSpace**: Quotient-space reduction for **computational efficiency**: scenarios are mapped to low-dim features, clustered, and reduced to K representatives (default K = 40% of num_scenarios). Fewer constraints per solve. No CSV required; set `quotient_num_override` in config to fix K.
  If any scenario CSV is missing, that method falls back to normal sampling.

## Build and run

```bash
# From repo root
cd cpp_mpc/build
cmake ..
make future_sl_experiments

# Run 100 rollouts per method, write to repo future_sl_mpc/experiments/results
./future_sl_experiments ../../future_sl_mpc/experiments/results/ 100
```

Output: `future_sl_rollouts.csv` (columns: seed, method, scenario, S, collision, total_progress, missed_mode_steps, avg_solve_ms, …). The `scenario` column is `"baseline"` for this runner.

### Edge-case and tuning runs

```bash
# Build (with future_sl_experiments)
make future_sl_edge_tuning

# Run edge-case scenarios (baseline, high_switch, rare_heavy, low_S, distribution_shift)
# and tuning sweeps (certificate radius, RTA threshold, bandit beta). 25 rollouts per cell by default.
./future_sl_edge_tuning ../../future_sl_mpc/experiments/results/ 25
```

Output: `future_sl_rollouts_edge_tuning.csv` (same schema with `scenario` varying: e.g. `high_switch`, `tune_cert_10`, `tune_rta_15`).

## Analyze

### Main efficacy vs SHMPC

```bash
# From repo root; use path to CSV (may be under cpp_mpc/future_sl_mpc/... if you ran from build without ../../ path)
python3 future_sl_mpc/experiments/analyze_vs_shmpc.py path/to/future_sl_rollouts.csv
```

Produces:

- Printed summary (collision rate, effect vs SHMPC, progress, solve time).
- `future_sl_mpc/experiments/results/analysis_summary.txt`
- `future_sl_mpc/experiments/results/efficacy_vs_shmpc.csv`

### Edge-case and tuning analysis

```bash
python3 future_sl_mpc/experiments/analyze_edge_and_tuning.py path/to/future_sl_rollouts_edge_tuning.csv
```

Produces:

- **Edge cases:** Collision rate and progress by (scenario, method); efficacy vs SHMPC per scenario.
  - `edge_case_summary.csv`, `edge_case_table.txt`
- **Trade-offs:** Safety vs progress vs solve time (baseline scenario); Pareto-efficient methods.
  - `tradeoff_summary.csv`, `tradeoff_table.txt`
- **Tuning sensitivity:** Collision rate vs certificate radius, RTA threshold, bandit beta.
  - `tuning_sensitivity.csv`, `tuning_summary.txt`

Use 25–50+ rollouts per cell for stable edge-case and tuning comparisons.

## Graphics (collision avoidance strategy types)

From the efficacy CSV, generate figures that show strategy taxonomy, efficacy comparison, and conceptual flow:

```bash
python3 future_sl_mpc/experiments/plot_strategy_graphics.py future_sl_mpc/experiments/results/efficacy_vs_shmpc.csv --out future_sl_mpc/experiments/results
```

Outputs in the results directory:

| Figure | Description |
|--------|-------------|
| `strategy_taxonomy.png` | Strategy types and method grouping (Baseline, DRO, Allocation, Certificate, Compiler, RTA). |
| `strategy_collision_bars.png` | Collision rate by method with 95% CI, colored by type. |
| `strategy_delta_vs_shmpc.png` | Effect vs SHMPC: Δ collision rate (horizontal bar; positive = safer). |
| `strategy_tradeoff_scatter.png` | Safety (1 − collision rate) vs progress; bubble size ∝ solve time. |
| `strategy_control_flow.png` | Control loop with intervention points (where each strategy type acts). |
| `strategy_type_diagrams.png` | Conceptual diagrams per type: DRO, allocation, certificate, compiler, RTA. |

## Results summary

See **FINDINGS.md** § "Efficacy Comparison vs SHMPC" and § "Edge cases, trade-offs, and tuning" for interpretation. In a 50-rollout run: RTA gave a solid collision-rate reduction vs SHMPC; Compiler and Bandit also improve safety. Edge-case runs stress-test methods under high switching, rare-mode bias, low scenario budget, and distribution shift; tuning sweeps show sensitivity to certificate radius, RTA threshold, and bandit β.

## Target collision benchmark (≤ 2% with CI)

This benchmark answers: **Which technique reaches a collision rate under a target (default 2%) with minimal compute cost?**

- **Criterion**: declare success only when the **upper bound** of the **95% Wilson CI** is ≤ target (avoids “0/10 = 0%” false confidence).
- **Output**: a sweep CSV plus a “best per method” table and a cost vs safety plot.

Run (from `cpp_mpc/build`):

```bash
make target_collision_benchmark
./target_collision_benchmark ../../future_sl_mpc/experiments/results/ 250 0.02
python3 ../../future_sl_mpc/experiments/analyze_target_collision.py ../../future_sl_mpc/experiments/results/target_collision_benchmark.csv --out ../../future_sl_mpc/experiments/results
```

Produces:

- `target_collision_benchmark.csv`: per-(method, knob) sweep results with CI + solve time.
- `target_collision_best_per_method.csv`: best (lowest solve time) config among those meeting target.
- `target_collision_summary.txt`: readable summary.
- `target_collision_cost_vs_safety.png`: scatter of compute vs certified safety.
