# Double-Dual Scenario Allocation

## Idea

Use the **dual** (binding-constraint) information from the MPC solve to drive scenario allocation at the next step. The MPC solver identifies which scenarios had **active** (binding) collision constraints; those scenarios correspond to obstacle modes that were critical for the solution. The "double dual" approach uses this information twice:

1. **First dual:** Solve scenario MPC → obtain which scenarios are active (constraints binding). After `remove_inactive_scenarios`, the controller keeps only scenarios that were active.
2. **Second dual:** Use the **modes** of those active scenarios to set **mode weights** for the next step’s sampling. Modes that appeared in many active scenarios get higher weight, so the next sample is biased toward “constraint-relevant” modes.

No extra solve is required: we only use the existing active-scenario set and map it to mode counts, then set `custom_mode_weights` for the next iteration.

## Implementation

- **Location:** `cpp_mpc/src/experiment_harness.cpp` (method `SHMPC_DoubleDual` in `run_experiment_rollout_future_sl`).
- **State:** `dual_mode_weights` (per-obstacle map of mode → weight), updated after each successful solve.
- **After solve:** For each scenario in `controller.scenarios()` (the active set after pruning), count how many use each mode for the obstacle. Then:
  - `weight_m = (count_m + floor) / Z` with `floor = 0.1` and `Z` such that weights sum to 1. All modes get at least a small mass so sampling never excludes a mode entirely.
- **Before next step:** If `dual_mode_weights` has an entry for the obstacle, call `controller.set_custom_mode_weights(obs_id, dual_mode_weights[obs_id])` so sampling uses these weights instead of history-based weights.
- **First step:** No previous dual info, so normal (history-based) sampling is used.

## Comparison to other methods

- **Conformal / Hazard / Bandit:** Also set custom mode weights, but from boundary scores, hazard model, or UCB—not from which constraints were binding.
- **Dual risk monitor (6.8):** Uses dual magnitudes and margins as risk features for a trigger; it does not change scenario allocation.
- **Double dual:** Allocates scenario budget toward modes that were **actually binding** in the last solve, adapting online to which obstacle behaviors are currently constraining the plan.

## Usage

- Included in the standard method list in `run_future_sl_experiments.cpp` and `run_future_sl_edge_and_tuning.cpp`.
- No CSV or Python script required; purely C++ in the rollout loop.
- Results appear in `future_sl_rollouts.csv` and in the efficacy comparison plots under the "Double dual" strategy type.

## References

- Section 6.8 (Dual Risk Monitor) uses duals/margins for risk prediction; double dual reuses the same “which constraints matter” idea to drive **allocation** instead of a binary trigger.
