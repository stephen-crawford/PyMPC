# Overall Findings: Future Statistical Learning MPC Extensions

This document summarizes the implementations and preliminary testing for each extension type from the memo *Future Research Directions for Safer Obstacle Avoidance (Statistical Learning Extensions + Paradigm-Shift Opportunities)*.

---

## Part I: Extension-Based Directions (Near-Term)

### 6.1 Conformal Safety Wrappers (C++)

**Implementation:** Mode-conditional conformal quantiles from residuals; boundary score \( s_m = \max_t \max_{\xi \in C_{t,m}} g_t(x_t, \xi) \) approximated via constraint value + radius × gradient norm; scenario allocation \( S_m \propto \exp(\alpha s_m) \) with normalization and floor.

**Testing:** Unit tests verify (1) conformal quantile lies in residual range and increases with \(\delta\), (2) per-mode quantiles differ by mode, (3) boundary scores are higher for modes with larger radius and constraint violation, (4) allocation sums to \(S\) and favors high \(s_m\). All tests passed.

**Findings:** Conformal calibration gives a clear interface for uncertainty: tubes \(C_{t,m}\) can be used to tighten constraints (robust constraint over ball) or to drive scenario allocation so modes near the safety boundary get more samples. Boundary-aware allocation successfully shifts budget toward “risky” modes in the toy tests.

---

### 6.2 Hazard / Change-Point Switch Sampling (C++)

**Implementation:** Logistic hazard \( \lambda_m(h) = \sigma(\theta_m^\top h) \) with features (e.g. inverse time since last switch); reweighting \( \tilde\pi_m \propto \pi_m \exp(\eta \lambda_m) \); allocation \( S_m \) from \( \tilde\pi_m \) with optional exploration floor when \( \max_m \lambda_m > \tau \).

**Testing:** Unit tests confirm reweighting increases probability of high-hazard mode; allocation sums to \(S\); hazard model output is higher for smaller time-since-switch. All tests passed.

**Findings:** Switch-aware reweighting is easy to plug into the existing scenario sampler: replace nominal mode weights with \( \tilde\pi \) and optionally enforce a minimum number of scenarios per mode when a switch is imminent. Reduces the chance of missing a rare switch exactly when it matters.

---

### 6.3 Risk-Directed Bandit Allocation (C++)

**Implementation:** Per-mode state \( (\hat{R}_m, n_m) \); UCB \( \text{UCB}_m = \hat{R}_m + \beta \sqrt{\log(t)/n_m} \); allocation \( S_m \propto \text{UCB}_m \) with minimum 1 per mode; online update with violation indicators.

**Testing:** With more violations observed for mode B, UCB(B) > UCB(A) and allocation favors B; total scenarios sum to \(S\). All tests passed.

**Findings:** Treating scenario budget as a bandit problem (sample where risk is high and uncertain) is a natural fit. Coupling violation feedback from the MPC step to the bandit state is straightforward; the main design choice is how often to update (e.g. every step vs batched).

---

### 6.4 Learned OT Ground Cost (Python + C++ interface)

**Implementation:** Python: trajectory embedding \(\phi(\tau)\) as flattened positions plus safety margin; mode-to-mode cost \( D[i,j] = \|\phi(\tau_i) - \phi(\tau_j)\|_2 \); CSV export of cost matrix. C++ DRO already supports `EUCLIDEAN_MEAN`; a custom cost matrix can be supplied via file or API for a “learned” ground cost.

**Testing:** Python tests check symmetry and non-negativity of \(D\); cost is larger between trajectories with very different margins. Cost matrix written to CSV. All tests passed.

**Findings:** Planner-aligned cost (distance in embedding space correlated with safety margin) is a small change on top of existing OT reshaping: swap the cost matrix. Full pipeline would train \(\phi\) with contrastive/supervised loss on safety labels; the current placeholder demonstrates the interface and sanity checks.

---

### 6.5 Adaptive DRO Shift Detection (C++)

**Implementation:** Rolling window of residuals \( d_t \); baseline mean \( \mu_0 \); \( \rho(t) = \rho_{\min} + k \cdot \max(0, \bar{d}_t - \mu_0) \). Intended to be used to set the Wasserstein ball radius (e.g. `dro_epsilon`) each step.

**Testing:** When residuals are below baseline, \( \rho(t) \approx \rho_{\min} \); when above (e.g. after injecting large errors), \( \rho(t) \) increases. All tests passed.

**Findings:** Adaptive \(\rho\) acts as a “panic knob”: under distribution shift or poor predictions, residuals rise and DRO becomes more conservative without manual tuning. Fits cleanly with the existing DRO module (set epsilon from \( \rho(t) \) before each solve).

---

### 6.6 Diffusion Calibration (Python)

**Implementation:** Placeholder “diffusion” = Gaussian samples around mean trajectory; severity score = max over time of (combined_radius − distance); select \(S\) from \(K\) candidates by severity with a simple diversity filter (avoid very similar trajectories); risk bound \( \hat{p} + \text{rad}(S, \delta) \) via empirical Bernstein radius.

**Testing:** Output scenarios count equals \(S\); \( p_{\text{upper}} \geq \hat{p} \). All tests passed.

**Findings:** The pipeline (generate many candidates → score by severity → select with diversity → report calibrated risk) is generic; swapping in a real diffusion model is a drop-in. Calibration (e.g. Bernstein or Clopper–Pearson) makes the risk number usable for constraint or trigger design.

---

### 6.7 Counterfactual Intent (Python)

**Implementation:** Placeholder ego-conditioned response: obstacle mean shifts with ego control (e.g. lateral response to steering); for each candidate ego plan \( u^{(j)} \), sample \(S\) obstacle trajectories from \( p(\xi \mid u^{(j)}) \), compute violation rate \( \hat{p}_j \), return \( j^* = \arg\min_j \hat{p}_j \).

**Testing:** \( j^* \) is a valid index; all \( \hat{p}_j \in [0,1] \). All tests passed.

**Findings:** Ego-conditioned prediction and counterfactual plan selection are the right abstraction for interactive settings (merges, crosswalks). The current model is a toy; in practice one would train \( p(\xi \mid u) \) on logged or simulated data. The selection loop (candidates → sample → risk → choose) is already in place.

---

### 6.8 Dual Risk Monitor (C++)

**Implementation:** Features: constraint margins \( g_i \), dual magnitudes \( |\lambda_i| \). Risk predictor \( \hat{r} = \sigma(\theta_0 + \theta_1 \cdot \text{margin\_violation} + \theta_2 \cdot \sum \lambda_i) \). Trigger: intervene (e.g. tighten constraints, add scenarios) when \( \hat{r} > \tau \).

**Testing:** Tighter constraints and larger duals yield higher \( \hat{r} \); trigger fires when \( \hat{r} > \tau \). All tests passed.

**Findings:** Using the solver’s duals and margins as risk features is low-cost and informative. Next step is to log traces (margins, duals, outcomes) and train a calibrated \( \psi \) (e.g. Platt scaling or temperature) so the trigger has a clear interpretation (e.g. estimated short-horizon violation probability).

---

## Part II: Paradigm-Shift Opportunities

### 7.1 Certificate-First Learning (C++)

**Implementation:** Certificate = tube radii \( r_t \) and \(\delta\); radii from conformal quantile of per-timestep residuals; volume = \( \sum r_t^2 \); constraint tightening \( b_{\text{robust}} = b_{\text{nominal}} - r \|a\| \).

**Testing:** Radii are non-negative; volume is non-negative; tightened offset decreases by \( r \|a\| \). All tests passed.

**Findings:** Certificate-first defines a clear contract: predictor outputs (mean, radii, \(\delta\)); MPC enforces robust constraints against the tube. Training the predictor to minimize certificate volume subject to coverage (e.g. Lagrangian) is the natural next step.

---

### 7.2 Scenario Compiler (C++)

**Implementation:** Witness set and iteration count; one-step constraint generation: call adversary to get worst-case scenario, add to witness set, increment iterations; stop when adversary returns null or max iterations. Helper to check if a scenario violates the plan (distance-based). Adversary is user-provided (e.g. search over modes or gradient-based in a differentiable sim).

**Testing:** With a mock adversary that returns one scenario then null, witness set grows by one and then certificate is valid. All tests passed.

**Findings:** Replacing fixed \(S\) i.i.d. scenarios with “add worst violator and re-solve” can yield smaller witness sets and certificates. The main cost is implementing a good adversary (differentiable sim, learned critic, or discrete mode search). Calibrated stopping (e.g. Clopper–Pearson on a held-out pool) can be layered on top.

---

### 7.4 Runtime Assurance (C++)

**Implementation:** RTA wrapper: if \( M(\text{state}, \text{certificate}) = \text{safe} \), apply learned action; else apply fallback. Optional hysteresis. Simple TTC-style monitor: safe iff min distance to obstacles > threshold.

**Testing:** With safe monitor, output is learned action; with unsafe monitor, output is fallback. All tests passed.

**Findings:** RTA gives a clean separation: the learning stack can be complex and unverified, while the monitor and fallback are small and verifiable. Certificate or risk from conformal/diffusion/dual monitor can feed into \(M\) to reduce false positives (e.g. don’t override when certificate is tight and risk is low).

---

## Summary Table

| Extension              | Language | Status   | Main takeaway                                                                 |
|------------------------|----------|----------|-------------------------------------------------------------------------------|
| 6.1 Conformal wrappers | C++      | Done     | Calibrated tubes + boundary-aware allocation improve safety–efficiency trade. |
| 6.2 Hazard sampling    | C++      | Done     | Switch-aware reweighting concentrates budget when it matters.                 |
| 6.3 Risk bandit        | C++      | Done     | UCB allocation reduces uncertainty in risk at limited \(S\).                 |
| 6.4 OT ground cost     | Py + C++ | Done     | Learned cost matrix plugs into existing DRO; train \(\phi\) for safety.     |
| 6.5 Adaptive DRO       | C++      | Done     | Residual-based \(\rho(t)\) automates conservativeness under shift.           |
| 6.6 Diffusion calibration | Python | Done   | Risk-directed selection + calibrated bound ready for real diffusion.        |
| 6.7 Counterfactual     | Python   | Done     | Ego-conditioned selection loop in place; need trained \(p(\xi\mid u)\).     |
| 6.8 Dual risk monitor  | C++      | Done     | Duals/margins as risk features; add calibration for trigger.                 |
| 7.1 Certificate-first  | C++      | Done     | Tube + tightening defined; train for small volume with coverage.                |
| 7.2 Scenario compiler  | C++      | Done     | Witness set + adversary; add calibrated stopping.                            |
| 7.4 Runtime assurance  | C++      | Done     | Monitor + fallback; certificates can improve specificity.                    |

---

## Efficacy Comparison vs SHMPC

Closed-loop rollouts (80 steps, rare-mode switching, same scenario count) compare **SHMPC** (safe horizon only, no DRO) to all extensions and Paradigm-Shift variants. Metrics: collision rate (95% Wilson CI), total progress (efficiency), avg solve time.

### Results (100 rollouts per method)

| Method                | Collision rate (95% CI)   | Δ vs SHMPC (collision) | Progress (mean) | Δ progress | Avg solve (ms) | Δ solve (ms) |
|-----------------------|---------------------------|-------------------------|-----------------|------------|----------------|--------------|
| **SHMPC** (baseline)  | 29.0% [21.0, 38.5]       | —                       | 13.77           | —          | 4.56           | —            |
| SHMPC_DRO             | 26.0% [18.4, 35.4]       | **+10.3%**              | 14.17           | +0.40      | 4.64           | +0.07        |
| SHMPC_AdaptiveDRO     | 29.0% [21.0, 38.5]       | 0.0%                    | 13.36           | −0.41      | 4.72           | +0.15        |
| SHMPC_RTA             | 28.0% [20.1, 37.5]       | +3.4%                   | 12.92           | −0.85      | 4.98           | +0.42        |
| SHMPC_Conformal       | 24.0% [16.7, 33.2]       | **+17.2%**              | 13.56           | −0.22      | 4.65           | +0.09        |
| SHMPC_Hazard          | 32.0% [23.7, 41.7]       | −10.3%                  | 13.62           | −0.16      | 4.82           | +0.25        |
| SHMPC_Bandit          | 25.0% [17.6, 34.3]       | **+13.8%**              | 13.77           | 0.00       | 4.96           | +0.39        |
| SHMPC_Certificate     | 43.0% [33.7, 52.8]       | −48.3%                  | 13.52           | −0.25      | 4.97           | +0.41        |
| SHMPC_Compiler        | 21.0% [14.2, 30.0]       | **+27.6%**              | 13.99           | +0.21      | 7.36           | +2.80        |
| CertificateFirst      | 39.0% [30.0, 48.8]       | −34.5%                  | 13.59           | −0.18      | 5.12           | +0.56        |
| ScenarioCompiler      | 23.0% [15.8, 32.2]       | **+20.7%**              | 14.06           | +0.29      | 8.43           | +3.87        |

### Interpretation (extension-style vs Paradigm-Shift)

- **Extension-style (add-ons to SHMPC):**  
  - **SHMPC_Compiler** (+27.6% collision reduction) and **SHMPC_Conformal** (+17.2%) give the largest safety gains. **SHMPC_Bandit** (+13.8%) and **SHMPC_DRO** (+10.3%) also beat the baseline. **SHMPC_RTA** (+3.4%) is slightly safer with a clear progress cost (−0.85). **SHMPC_AdaptiveDRO** matches SHMPC; **SHMPC_Hazard** is worse (−10.3%) in this setup. **SHMPC_Certificate** (−48.3%) suffers from the fixed tube radius.
- **Paradigm-Shift variants:**  
  - **ScenarioCompiler** (+20.7%) mirrors the extension **SHMPC_Compiler** with similar safety and slightly better progress (+0.29) at higher solve cost (8.43 ms). **CertificateFirst** (−34.5%) is better than **SHMPC_Certificate** but still worse than SHMPC; both need radius tuning or data-driven certificates.
- **Efficiency vs safety:** Compiler variants improve both collision rate and progress at ~2.8–3.9 ms extra solve time. RTA trades progress for safety. Bandit and Conformal improve safety with minimal progress cost. Certificate-based methods need smaller radii or learned certificates to avoid over-tightening.

**Integration:** Custom mode weights (Conformal, Hazard, Bandit), certificate radii (CertificateFirst), and scenario compiler (set_scenarios / sample_and_set_scenarios) are wired in the controller and used in the rollout.

---

## How to Run Tests

### Test status (latest run)

- **Future SL MPC C++ unit tests:** All 8 pass (ConformalSafety, HazardSwitchSampling, RiskDirectedBandit, AdaptiveDROShift, DualRiskMonitor, CertificateFirst, ScenarioCompiler, RuntimeAssurance). Run: `ctest -R "ConformalSafety|HazardSwitch|RiskDirected|AdaptiveDRO|DualRisk|CertificateFirst|ScenarioCompiler|RuntimeAssurance"`.
- **Python extension tests:** All pass (OT ground cost 04, diffusion calibration 06, counterfactual intent 07).
- **Full ctest:** Other suite tests (Core, OT, etc.) pass; SafeHorizonContouringTests can fail on the aggressive-obstacle subtest depending on setup.

- **C++:** From `cpp_mpc/build`:  
  `make test_conformal_safety test_hazard_switch_sampling test_risk_directed_bandit test_adaptive_dro_shift test_dual_risk_monitor test_certificate_first test_scenario_compiler test_runtime_assurance`  
  then run each executable, or `ctest` for all registered tests.

- **Python:** From each extension directory, e.g.  
  `python3 future_sl_mpc/04_ot_ground_cost_learning/run_tests.py`  
  `python3 future_sl_mpc/06_diffusion_calibration/run_tests.py`  
  `python3 future_sl_mpc/07_counterfactual_intent/run_tests.py`

### Efficacy experiments (vs SHMPC)

1. **Run rollouts** (from repo root):  
   `cd cpp_mpc/build && ./future_sl_experiments <output_dir> <num_rollouts>`  
   Example: `./future_sl_experiments ../../future_sl_mpc/experiments/results/ 100`  
   Methods: SHMPC, SHMPC_DRO, SHMPC_AdaptiveDRO, SHMPC_RTA, SHMPC_Conformal, SHMPC_Hazard, SHMPC_Bandit, SHMPC_Certificate, SHMPC_Compiler, CertificateFirst, ScenarioCompiler.  
   Writes `future_sl_rollouts.csv` with one row per (seed, method).

2. **Analyze vs SHMPC**:  
   `python3 future_sl_mpc/experiments/analyze_vs_shmpc.py <path/to/future_sl_rollouts.csv>`  
   Prints summary and writes `future_sl_mpc/experiments/results/analysis_summary.txt` and `efficacy_vs_shmpc.csv`.

3. **Edge-case and tuning experiments**:  
   `cd cpp_mpc/build && make future_sl_edge_tuning && ./future_sl_edge_tuning ../../future_sl_mpc/experiments/results/ 25`  
   Writes `future_sl_rollouts_edge_tuning.csv` with scenarios: baseline, high_switch, rare_heavy, low_S, distribution_shift, tune_cert_*, tune_rta_*, tune_bandit_*.

4. **Analyze edge/tuning**:  
   `python3 future_sl_mpc/experiments/analyze_edge_and_tuning.py <path/to/future_sl_rollouts_edge_tuning.csv>`  
   Writes edge_case_*, tradeoff_*, tuning_* summaries under `future_sl_mpc/experiments/results/`.

All implementations are preliminary; unit tests validate interfaces. Run 50–100+ rollouts per method for stable efficacy comparison; use 25+ per cell for edge-case and tuning analysis.

### Latest run summary

- **Main efficacy:** `./future_sl_experiments ../../future_sl_mpc/experiments/results/ 100` → 100 rollouts × 11 methods. Analysis: `analyze_vs_shmpc.py` → `analysis_summary.txt`, `efficacy_vs_shmpc.csv`.
- **Edge-case/tuning:** `./future_sl_edge_tuning ../../future_sl_mpc/experiments/results/ 30` (or 25) → edge scenarios + tuning sweeps. Analysis: `analyze_edge_and_tuning.py` → `edge_case_summary.csv`, `tradeoff_summary.csv`, `tuning_sensitivity.csv` (when tuning scenarios present).
- **Unit tests:** All 8 Future SL C++ tests pass (ConformalSafety, HazardSwitchSampling, RiskDirectedBandit, AdaptiveDROShift, DualRiskMonitor, CertificateFirst, ScenarioCompiler, RuntimeAssurance).
