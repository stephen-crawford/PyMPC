# 6.8 Learning Risk Certificates from MPC Traces (Dual Variables / Constraint Activity)

## Idea

Use **MPC sensitivities** (constraint margins \( g_i \) and Lagrange multipliers \( \lambda_i \)) to predict risk. Large \( \lambda_i \) means constraint \( i \) is active/sensitive. Train \( \hat{r} = \psi(\{\lambda_i\}, \{g_i\}, \text{context}) \); use as risk monitor to trigger tightened constraints, reduced horizon, or more scenario allocation when \( \hat{r} > \tau \).

## Implementation

- **`dual_risk_monitor.hpp/cpp`**: Log features (constraint margins, dual magnitudes, solver iterations); simple rule-based or linear \( \psi \): e.g. \( \hat{r} = \sigma(\theta_0 + \theta_1 \cdot \max_i(-\min(0, g_i)) + \theta_2 \cdot \sum_i \lambda_i) \). Trigger policy: if \( \hat{r} > \tau \), set flag to increase scenario budget or tighten constraints.

## Testing

- Unit: when constraints are tight, \( \hat{r} \) increases; trigger fires above threshold.
- Integration: closed-loop with adaptive trigger vs fixed; report false positives vs safety gains.
