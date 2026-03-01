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

## Results summary

See **FINDINGS.md** § "Efficacy Comparison vs SHMPC" and § "Edge cases, trade-offs, and tuning" for interpretation. In a 50-rollout run: RTA gave a solid collision-rate reduction vs SHMPC; Compiler and Bandit also improve safety. Edge-case runs stress-test methods under high switching, rare-mode bias, low scenario budget, and distribution shift; tuning sweeps show sensitivity to certificate radius, RTA threshold, and bandit β.
