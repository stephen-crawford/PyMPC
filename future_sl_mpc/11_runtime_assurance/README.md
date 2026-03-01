# 7.4 Runtime Assurance + Neuro-Symbolic as Default Integration Stack

## Idea

**Verified monitor + fallback** wraps learned planners:
\( u_t = u^{\text{learned}}_t \) if \( M(\cdot) = \text{safe} \), else \( u_t = u^{\text{fallback}}_t \).  
Certification targets the monitor and fallback, not the entire learning system.

## Implementation

- **`runtime_assurance.hpp/cpp`**: Monitor \( M(x_t, \text{certificates}) \): e.g. TTC > threshold, barrier function, or certificate inclusion. Fallback: constant brake + lane-keep (or zero steering). Output: applied action (learned or fallback). Hysteresis/debounce to avoid oscillation.

## Testing

- Unit: when monitor returns unsafe, output is fallback; when safe, output is learned.
- Integration: autonomy rate vs collision rate under stress.
