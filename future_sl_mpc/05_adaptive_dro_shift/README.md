# 6.5 Distribution Shift Detection + Wasserstein DRO MPC Triggered Online

## Idea

Solve a **distributionally robust** constraint:  
\( \sup_{Q : W(Q, \hat{P}) \leq \rho} P_Q(\text{violation}) \leq \varepsilon \).  
When behavior shifts, nominal chance constraints can be over-optimistic. **Adaptive \( \rho(t) \)** provides a principled "panic knob": compute a shift score (residual drift, log-likelihood) and map it to \( \rho(t) = \rho_{\min} + k \cdot \max(0, \bar{d}_t - \mu_0) \).

## Implementation

- **`adaptive_dro_shift.hpp/cpp`**: Rolling window of residuals \( d_t = \rho(\hat\xi_t, \xi_t) \); baseline mean \( \mu_0 \) from calibration; \( \rho(t) = \rho_{\min} + k \cdot \max(0, \bar{d}_t - \mu_0) \); output \( \rho(t) \) for use in DRO (e.g. set epsilon from it). Integrates with existing `WassersteinDRO` by updating its epsilon from \( \rho(t) \) each step.

## Testing

- Unit: when residuals exceed baseline, \( \rho(t) \) increases; when below, stays at \( \rho_{\min} \).
- Integration: under simulated distribution shift, adaptive-DRO reduces collisions vs nominal; compare always-DRO vs adaptive-DRO.
