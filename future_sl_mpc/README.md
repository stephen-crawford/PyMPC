# Future Statistical Learning MPC — Implementations

This directory contains implementations of the extensions and paradigm-shift ideas from the memo **"Future Research Directions for Safer Obstacle Avoidance (Statistical Learning Extensions + Paradigm-Shift Opportunities)"** (2026-02-26). Each subdirectory corresponds to one proposal and includes:

- **Explanation** of the method and how it fits the existing scenario MPC / DRO stack
- **Implementation** in C++ (preferred) or Python (for ML-heavy components)
- **Tests** and minimal runnable examples

## Structure

| Directory | Section | Description |
|-----------|---------|-------------|
| `01_conformal_safety_wrappers` | 6.1 | Mode-conditional conformal prediction sets + boundary-aware scenario allocation |
| `02_hazard_switch_sampling` | 6.2 | Hazard/change-point models for switch-aware sampling |
| `03_risk_directed_bandit` | 6.3 | UCB-based scenario allocation over modes |
| `04_ot_ground_cost_learning` | 6.4 | Learned planner-aligned OT ground cost (Python + C++ interface) |
| `05_adaptive_dro_shift` | 6.5 | Distribution shift detection → adaptive Wasserstein radius ρ(t) |
| `06_diffusion_calibration` | 6.6 | Diffusion trajectory generator + risk-directed sampling + calibrated risk (Python) |
| `07_counterfactual_intent` | 6.7 | Ego-conditioned prediction p(ξ\|u) and counterfactual evaluation (Python) |
| `08_dual_risk_monitor` | 6.8 | Risk certificates from MPC dual variables / constraint activity |
| `09_certificate_first` | 7.1 | Certificate-first learning (tubes + calibration as first-class outputs) |
| `10_scenario_compiler` | 7.2 | Minimal witness set + constraint generation + stopping rule |
| `11_runtime_assurance` | 7.4 | RTA wrapper: monitor + fallback controller |

## Building and Testing

- **C++**: Each C++ module can be built and tested via the main `cpp_mpc` CMake (see `cpp_mpc/CMakeLists.txt`). Standalone test executables are under `future_sl_mpc/*/tests/` or linked from `cpp_mpc/tests/`.
- **Python**: Each Python module has a `run_tests.py` or `test_*.py` and optional `requirements.txt`.

## Overall Findings

See **`FINDINGS.md`** for a concise report on each extension type: what was implemented, how it was tested, and the main takeaways.
