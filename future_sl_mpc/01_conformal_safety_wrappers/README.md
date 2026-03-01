# 6.1 Conformal Safety Wrappers for Multi-Step Collision Risk

## Idea

Produce **calibrated prediction sets (tubes)** \( C_{1:H} \) such that  
\( P(\xi_{1:H} \in C_{1:H}) \geq 1 - \delta \),  
and enforce **robust constraints**  
\( g_t(x_t, \xi) \leq 0 \; \forall \xi \in C_t \).  
If the future lies in the calibrated tube with probability \( \geq 1-\delta \), the plan is safe.

- **Per-mode sets**: \( C_{t,m} = \{ \xi : \rho(\hat\xi_{t,m}, \xi) \leq q_m \} \) with conformal quantile \( q_m \) per mode.
- **Boundary-aware allocation**: Score modes by \( s_m = \max_{t \leq H_{\text{safe}}} \max_{\xi \in C_{t,m}} g_t(x_t, \xi) \), then allocate scenarios \( S_m \propto \exp(\alpha s_m) \) so that modes near the safety boundary get more samples.

## Implementation

- **`conformal_safety.hpp/cpp`**: Mode-conditional conformal quantiles from residuals \( r_t = \rho(\hat\xi_t, \xi_t) \); construction of \( C_{t,m} \) as balls of radius \( q_m \); boundary score \( s_m \) using worst-case constraint value over the ball; allocation \( S_m \propto \exp(\alpha s_m) \) with normalization.
- **Integration**: Used by scenario sampler to (1) tighten constraint margins by the conformal radius when using tubes, or (2) allocate scenario counts per mode from boundary scores.

## Testing

- **Unit**: Conformal quantile from synthetic residuals; set inclusion; allocation sums to \( S \) and favors high \( s_m \).
- **Integration**: Run scenario MPC with conformal allocation vs uniform allocation; compare collision and missed-mode rates on toy switching scenarios.

## References

- Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* (arXiv:2107.07511).
- Dixit et al., *Adaptive Conformal Prediction for Motion Planning among Dynamic Agents* (arXiv:2212.00278).
