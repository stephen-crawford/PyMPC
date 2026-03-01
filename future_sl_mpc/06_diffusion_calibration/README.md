# 6.6 Diffusion / Flow Trajectory Generators with Calibration for Planning

## Idea

Use a **generative model** \( \xi_{1:H} = G_\theta(z, \text{context}) \) to sample diverse futures. Output both scenarios and a **calibrated risk bound**: \( \hat{p} = \frac{1}{S}\sum_s V^{(s)} \), output \( \hat{p} + \text{rad}(S, \delta) \).

## Implementation

- **Python** (`diffusion_calibration.py`): Placeholder diffusion = Gaussian noise around a mean trajectory (stand-in for a full diffusion model). Generate K > S candidates, score by severity \( s^{(k)} = \max_t g_t(x_t, \xi^{(k)}_t) \), select S with diversity (e.g. top per cluster or top-S by severity with diversity filter). Compute \( \hat{p} \) and conservative bound \( \hat{p} + \text{rad}(S, \delta) \) (e.g. Clopper-Pearson or empirical Bernstein).

## Testing

- Unit: risk bound is >= empirical violation rate in synthetic runs; diversity filter retains multiple modes.
- Integration: compare GMM vs diffusion-style sampler at equal latency; safety metrics.
