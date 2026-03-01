# 6.2 Hazard / Change-Point Models for Switch-Aware Sampling

## Idea

Learn a **hazard (switch intensity)** model  
\( \lambda_m(t \mid h_t) = P(\text{switch to mode } m \text{ in } [t, t+\Delta) \mid h_t) / \Delta \),  
or run **online change-point detection** on prediction residuals.  
When \( \lambda_m \) is large, a switch is imminent → diversify scenarios now.

- **Online update**: \( \tilde\pi_m = \pi_m \exp(\eta \lambda_m) / Z \); allocate \( S_m \propto \tilde\pi_m \).
- **Switch spike override**: If \( \max_m \lambda_m > \tau \), enforce a minimum exploration budget over alternative modes.

## Implementation

- **`hazard_switch_sampling.hpp/cpp`**: Logistic hazard \( \lambda_m(h_t) = \sigma(\theta_m^\top h_t) \) with feature vector \( h_t \) (e.g. time-to-conflict, relative velocity, time since last switch). Reweighted mode probs and allocation \( S_m = \max(1, \lfloor S \tilde\pi_m \rfloor) \); when \( \max_m \lambda_m > \tau \), redistribute extra samples to non-dominant modes.

## Testing

- Unit: reweighting increases probability of high-hazard mode; allocation sums to \( S \); spike override increases diversity.
- Integration: compare collision rate at switch points vs static allocation on toy switching trajectories.
