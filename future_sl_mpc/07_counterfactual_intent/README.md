# 6.7 Causal / Counterfactual Intent Modeling for Interactive Obstacle Avoidance

## Idea

Model **ego-conditioned futures**: \( p(\xi_{1:H} \mid \text{history}, \text{map}, u_{0:H-1}) \).  
For each candidate ego plan \( u^{(j)} \), sample obstacle responses and estimate risk; choose feasible plan minimizing (calibrated) risk.

## Implementation

- **Python** (`counterfactual_intent.py`): Placeholder conditional model: obstacle mean trajectory shifts as a function of ego control (e.g. linear response to ego acceleration). Generate candidate ego controls \( \{u^{(j)}\} \), for each sample S obstacle trajectories from \( p(\xi \mid u^{(j)}) \), compute violation rate \( \hat{p}_j \), return \( j^* = \arg\min_j \hat{p}_j \) among feasible.

## Testing

- Unit: when ego is more aggressive, obstacle response shifts; \( j^* \) differs across candidate sets.
- Integration: merge/crosswalk toy scenario; compare ego-conditioned vs exogenous predictor.
