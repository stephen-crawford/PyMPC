# 7.2 Scenario Compiler: Minimal Witness Sets + Stopping Rules

## Idea

Instead of drawing S i.i.d. scenarios, **construct the smallest witness set** that certifies safety (or bounds risk), with a **stopping rule**. Constraint generation: \( \xi^{(k+1)} \in \arg\max_{\xi \in D} \max_{t \leq H} g_t(x^\star_t, \xi_t) \); add \( \xi^{(k+1)} \) to the witness set and re-solve.

## Implementation

- **`scenario_compiler.hpp/cpp`**: Maintain witness set; given current plan and state, **adversary/critic** returns a scenario that maximizes violation (e.g. gradient-based in sim or discrete search over modes). Iteratively add worst violator and re-solve until no violation found or max iterations. Stopping: when held-out pool has no violations and a statistical bound (e.g. Clopper-Pearson) gives violation prob below \( \varepsilon \) with confidence \( 1-\delta \).

## Testing

- Unit: adding worst-case scenario increases constraint set; after adding true violator, next solve is feasible only if plan changes.
- Integration: witness set size vs i.i.d. S at same safety level; latency.
