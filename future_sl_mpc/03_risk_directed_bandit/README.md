# 6.3 Risk-Directed Scenario Generation (Bandit Allocation / UCB)

## Idea

Allocate scenario budget to **reduce uncertainty in collision risk**.  
\( V^{(s)} = \mathbb{1}\{\exists t : g_t(x_t, \xi_t^{(s)}) > 0\} \); maintain per-mode empirical violation rate \( \hat{R}_m \) and counts \( n_m \).  
**UCB allocation**: \( \text{UCB}_m = \hat{R}_m + \beta \sqrt{\log t / n_m} \); allocate to high UCB (high risk and uncertain).

## Implementation

- **`risk_directed_bandit.hpp/cpp`**: Per-mode state \( (\hat{R}_m, n_m) \); update with new violation observations; UCB computation; allocation \( S_m \propto \text{UCB}_m \) with minimum 1 per mode.

## Testing

- Unit: UCB increases for under-sampled modes; allocation favors high-risk modes after updates.
- Integration: Compare vs i.i.d. and static reshaping at equal compute; collision and conservativeness.
