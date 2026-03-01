# 6.4 Learned "Geometry of Modes" via Planner-Aligned OT Ground Cost

## Idea

Learn an embedding \( \phi(\tau) \in \mathbb{R}^d \) of trajectories so that distances correlate with **planning sensitivity** (near-miss/collision margin). Use as OT ground cost:
\( c(m, m') = \|\phi(\tau_m) - \phi(\tau_{m'})\|_2 \).

## Implementation

- **Python** (`embedding_and_cost.py`): Trajectory representation as sequence of 2D positions; simple "safety margin" label from min distance to ego; contrastive-style loss (positive pairs = similar margin, negative = different). Output: mode-to-mode cost matrix (e.g. CSV) for use in DRO.
- **C++**: The existing `WassersteinDRO` supports `DROGroundCostType::EUCLIDEAN_MEAN`; for learned cost we provide a **custom cost matrix** via file or API. Optional extension: `set_custom_cost_matrix(D)` so DRO uses it instead of computing D internally.

## Testing

- Python: unit test that embedding distances increase when safety margins differ; cost matrix is symmetric and non-negative.
- Integration: run DRO with learned cost matrix vs W2_BURES on toy modes; compare safety-efficiency Pareto.
