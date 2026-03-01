# 7.1 Certificate-First Learning: Safety Certificates as First-Class Outputs

## Idea

Models output **certificates** (e.g. calibrated tubes/sets and confidence level \( \delta \)) as first-class outputs: \( (\hat\xi_{1:H}, C_{1:H}, \delta) \) with \( P(\xi_{1:H} \in C_{1:H}) \geq 1 - \delta \); MPC enforces robust constraints against \( C \).

## Implementation

- **`certificate_first.hpp/cpp`**: Certificate schema: tube as per-timestep radius \( r_t \) (ball around mean). Calibration module: given calibration residuals, compute \( r_\theta \) as conformal quantile. Planner contract: constraint tightening by radius (robust constraint over ball). Minimal "certificate API": `(mean_trajectory, radii, delta)`.

## Testing

- Unit: certificate volume (sum of r_t^2) decreases when calibration is tighter; coverage check on held-out residuals.
- Integration: MPC with certificate tightening vs nominal; coverage and collision rate.
