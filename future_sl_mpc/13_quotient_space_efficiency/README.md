# 13. Quotient-Space Scenario Reduction (Computational Efficiency)

Reduces the **dimensionality** of the scenario set for SHMPC by mapping scenarios into a **quotient space**: each scenario is represented by a low-dimensional feature (mean and end positions of obstacle trajectories), then clustered so that only **K representative scenarios** are passed to the MPC instead of the full S. This lowers the number of constraints and decision variables, improving solve time while aiming to preserve safety.

## Idea

- **Full space:** Each scenario is a high-dim object (trajectory waypoints over horizon).
- **Quotient map:** Scenario → low-dim feature (e.g. 4D per obstacle: mean_x, mean_y, end_x, end_y).
- **Reduction:** Cluster scenarios in this low-dim space; keep one representative per cluster (closest to cluster centroid). MPC sees K scenarios instead of S (default K = max(1, 40% of S), or set `quotient_num_override` in `ExperimentConfig`).

## Implementation (C++)

- **`scenario_pruning.hpp/cpp`:**  
  - `scenario_to_quotient_feature(scenario, horizon)` → low-dim feature vector.  
  - `reduce_scenarios_quotient_space(scenarios, num_quotient, horizon)` → K representative scenarios (k-centers + nearest-to-centroid representative).
- **Experiment harness:** Method **SHMPC_QuotientSpace** samples S scenarios, reduces to K via `reduce_scenarios_quotient_space`, then calls `set_scenarios(reduced)` and solves. Config: `quotient_num_override` (if > 0, use as K; else K = 40% of num_scenarios).

## Tests

- **C++:** `test_paper_figures` includes a quotient-space check: 30 scenarios → 10 representatives; `reduce_scenarios_quotient_space(original, 10, horizon)` returns 10 scenarios.
- **Efficacy:** Run `./future_sl_experiments ... 50`; SHMPC_QuotientSpace is included in the method list and in the comparison plots.

## Results (50 rollouts)

- **Collision:** Slight improvement vs SHMPC (+7.1% in one run).  
- **Solve time:** Similar to baseline (~+0.27 ms), with fewer scenarios per solve (reduction overhead vs smaller QP).  
- **Progress:** Slightly better (+0.23).  

Use **SHMPC_QuotientSpace** when you want to trade a small amount of scenario diversity for lower problem size and similar or better solve times.
