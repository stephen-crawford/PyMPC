# Optimal Transport Extension for Gaussian Dynamics Learning

**Extension to**: *Scenario-based Path Prediction for Targets with Switching Dynamics*

This section describes how optimal transport (OT) theory extends the adaptive scenario MPC framework from Section IV by (i) replacing heuristic mode weights with Wasserstein-distance-based weights, (ii) learning the Gaussian dynamics parameters $(b_m, G_m)$ online from observed trajectories, and (iii) providing a runtime epsilon-guarantee verification mechanism.

---

## A. OT-Based Mode Weight Computation

### Motivation

The base method (Section IV.C) assigns weights $\pi_t^C$ over the mode history $\mathcal{H}_t^C$ using simple heuristics: uniform, frequency-counting, or recency-decay.  These do not leverage the *shape* of the observed dynamics.  We propose replacing these with a Wasserstein-distance-based weighting that compares the empirical velocity distribution of each obstacle against per-mode reference distributions.

### Reference distributions

For each mode $m \in \mathcal{M}_C$, define a reference velocity distribution $\nu_m^{\text{ref}}$ as an empirical distribution over $N_{\text{ref}}$ samples:

$$\nu_m^{\text{ref}} = \frac{1}{N_{\text{ref}}} \sum_{j=1}^{N_{\text{ref}}} \delta_{v_j^m}$$

where $v_j^m \sim \mathcal{N}(\bar{v}_m, \sigma_m^2 I)$ are velocity samples drawn from the mode's characteristic velocity profile.  For example, the *constant velocity* mode has $\bar{v}_m = [v_{\text{base}}, 0]^\top$, while the *decelerating* mode has $\bar{v}_m = [0.5 \cdot v_{\text{base}}, 0]^\top$.

### Observed distribution

At time $t$, let $\{v_k^{v}\}_{k=t-W}^{t}$ denote the $W$ most recent velocity observations for obstacle $v$.  Define the empirical observed distribution:

$$\hat{\mu}_t^v = \frac{1}{W} \sum_{k=t-W}^{t} \delta_{v_k^v}$$

Velocities are computed via finite differences: $v_k = (x_k - x_{k-1})/\Delta t$.

### Sinkhorn-regularized Wasserstein distance

We compute the entropy-regularized optimal transport distance between $\hat{\mu}_t^v$ and each reference $\nu_m^{\text{ref}}$.  Given source samples $\{s_i\}_{i=1}^n$ with weights $a$ and target samples $\{t_j\}_{j=1}^p$ with weights $b$, define the cost matrix:

$$C_{ij} = \|s_i - t_j\|^2$$

The Sinkhorn algorithm solves:

$$\min_{P \in \Pi(a, b)} \langle C, P \rangle + \varepsilon_{\text{sink}} H(P)$$

where $H(P) = -\sum_{ij} P_{ij} \log P_{ij}$ is the entropic regularizer and $\Pi(a, b)$ is the set of couplings with marginals $a, b$.  The solution is computed by iterating scaling vectors:

$$u^{(\ell+1)} = \frac{a}{K v^{(\ell)}}, \qquad v^{(\ell+1)} = \frac{b}{K^\top u^{(\ell+1)}}$$

where $K_{ij} = \exp(-C_{ij}/\varepsilon_{\text{sink}})$.  The transport plan is $P^* = \text{diag}(u) \, K \, \text{diag}(v)$, and the Wasserstein distance is:

$$W_2(\hat{\mu}_t^v, \nu_m^{\text{ref}}) = \left( \sum_{i,j} C_{ij} P^*_{ij} \right)^{1/2}$$

### Wasserstein mode weights

The mode weights for obstacle $v$ at time $t$ are computed via an exponential kernel:

$$\pi_t^C(m) = \frac{\exp\!\big(-W_2(\hat{\mu}_t^v, \nu_m^{\text{ref}}) / \lambda_W\big)}{\sum_{m' \in \mathcal{H}_t^C} \exp\!\big(-W_2(\hat{\mu}_t^v, \nu_{m'}^{\text{ref}}) / \lambda_W\big)}$$

where $\lambda_W > 0$ is a temperature parameter.  Modes whose reference distribution is closest (in Wasserstein distance) to the observed velocities receive the highest sampling probability.

**Key property**: Since the exponential kernel is strictly positive, $\pi_t^C(m) > 0$ for all $m \in \mathcal{H}_t^C$.  This preserves the positive-mass condition required by Corollary 1 of the base paper.

---

## B. Online Gaussian Dynamics Parameter Learning

### Motivation

The base method uses fixed mode dynamics $x_{k+1} = A_m x_k + b_m + G_m w_k$ with hand-tuned parameters $(b_m, G_m)$.  We extend this with an online estimation procedure that learns $(b_m, G_m)$ from observed trajectory residuals.

### Residual computation

Given a trajectory buffer of consecutive observations $\{x_k\}_{k=0}^{T}$ where $x_k = [p_x, p_y, v_x, v_y]_k^\top$ is the full dynamics state, and a prior state transition matrix $A_m$, compute residuals for observations labeled with mode $m$:

$$r_k = x_{k+1} - A_m x_k, \qquad \forall k \text{ where } m_k = m$$

### Bias estimation

The learned bias vector is the sample mean of residuals:

$$\hat{b}_m = \frac{1}{|\mathcal{R}_m|} \sum_{k \in \mathcal{R}_m} r_k$$

where $\mathcal{R}_m$ is the set of timesteps with mode $m$.

### Noise matrix estimation

The learned process noise matrix is derived from the sample covariance of residuals:

$$\hat{\Sigma}_m = \frac{1}{|\mathcal{R}_m| - 1} \sum_{k \in \mathcal{R}_m} (r_k - \hat{b}_m)(r_k - \hat{b}_m)^\top$$

The noise matrix is then $\hat{G}_m = \text{chol}(\hat{\Sigma}_m + \lambda_{\text{reg}} I)$, where $\text{chol}(\cdot)$ denotes the Cholesky decomposition and $\lambda_{\text{reg}} = 10^{-6}$ provides regularization.  If the Cholesky decomposition fails, we fall back to a diagonal approximation: $\hat{G}_m = \text{diag}(\sqrt{\max(\text{diag}(\hat{\Sigma}_m), 10^{-6})})$.

### In-place parameter update

The learned parameters replace the prior:

$$b_m \leftarrow \hat{b}_m, \qquad G_m \leftarrow \hat{G}_m$$

This update is performed periodically (every $T_{\text{learn}}$ timesteps, default 10) and requires a minimum of 5 residual samples for robustness.  After the update, all subsequent scenario sampling uses the learned dynamics, producing trajectories that better match the observed behavior.

### Connection to the Gaussian hypersphere

Under mode $m$, the one-step prediction uncertainty is characterized by $G_m G_m^\top = \hat{\Sigma}_m$, the covariance of the residuals.  The set $\{x : (x - A_m \bar{x} - b_m)^\top \Sigma_m^{-1} (x - A_m \bar{x} - b_m) \leq \chi^2_{n_x, \alpha}\}$ forms an **ellipsoid** in $\mathbb{R}^{n_x}$---the "Gaussian hypersphere" under the Mahalanobis metric.

Initially (before learning), $G_m$ is set to a scaled identity, making this set an **isotropic sphere**.  As OT learning estimates the true covariance structure, the sphere **deforms into an anisotropic ellipsoid** aligned with the principal directions of the observed process noise.  Specifically:

- The eigenvalues of $\hat{\Sigma}_m$ determine the **radii** along each principal axis
- The eigenvectors of $\hat{\Sigma}_m$ determine the **orientation** of the ellipsoid
- Learning reduces the volume of the ellipsoid (tighter uncertainty) in directions where the dynamics are deterministic, while preserving or expanding it in stochastic directions

---

## C. Multi-Modal Prediction via Wasserstein Barycenter

For trajectory prediction, per-mode forecasts are combined using a weighted barycentric approach.  At each prediction step $k$, the combined position is:

$$\bar{x}_k = \sum_{m \in \mathcal{H}_t^C} \pi_t^C(m) \, x_k^{(m)}$$

The combined uncertainty incorporates both the within-mode uncertainty and the between-mode disagreement:

$$\sigma_{k,\text{combined}} = \left(\sum_m \pi_t^C(m) \, \sigma_k^{(m)}\right) \cdot \left(1 + \frac{1}{2}\left\|\text{std}\big(\{x_k^{(m)}\}_m\big)\right\|\right)$$

where the second factor inflates uncertainty when modes disagree on the predicted position.  This ensures conservative coverage during mode transitions.

---

## D. Runtime Epsilon-Guarantee Verification

### Effective epsilon computation

At each MPC step, we verify that the number of scenarios $S$ satisfies Theorem 1.  The effective violation probability is:

$$\hat{\varepsilon} = \frac{2(\ln(1/\beta) + d + R)}{S}$$

where $d = N \cdot n_x + N \cdot n_u$ is the decision variable dimension, $R$ is the number of removed scenarios, and $\beta$ is the confidence parameter.  The guarantee holds when $\hat{\varepsilon} \leq \varepsilon_p$.

### Auto-adjustment

When $\hat{\varepsilon} > \varepsilon_p$ (insufficient scenarios), the system automatically computes and samples the required number:

$$S_{\text{required}} = \left\lceil \frac{2}{\varepsilon_p} \left(\ln\frac{1}{\beta} + d + R\right) \right\rceil$$

### Constraint enforcement

To preserve Theorem 1's guarantee, the `enforce_all_scenarios` flag ensures that **all** $S$ sampled scenarios generate collision constraints, rather than selecting a support subset.  This matches the paper's Eq.~(10c):

$$g\!\left(x_{t+k|t}^{\text{ego}}, \delta_{t+k|t}^{(i)}\right) \leq 0, \qquad \forall\, i = 1, \ldots, S, \quad \forall\, k = 1, \ldots, N$$

---

## E. Preservation of Scenario-Theoretic Guarantees

The OT extensions preserve the feasibility guarantee of Theorem 1 because:

1. **Fresh application**: Theorem 1 is applied independently at each MPC step $t$ with respect to the *current* predictive distribution $\Delta_t^C$.  When OT learning updates mode weights or dynamics parameters, this changes $\Delta_t^C$, but scenarios are sampled **after** the update, so the i.i.d. requirement is satisfied.

2. **Positive mass**: The Wasserstein weight kernel $\exp(-W/\lambda)$ never assigns exactly zero weight to any mode in $\mathcal{H}_t^C$, satisfying Corollary 1's requirement.

3. **Sample sufficiency**: The runtime $\hat{\varepsilon}$ check (Section D) ensures $S \geq S_{\text{required}}$ after any distribution change.

4. **Convexity**: The OT learning modifies only the scenario generation step, not the optimization structure.  The scenario program (10) remains convex in $U_t$.

The only risk is **Remark 2 violation**: if OT learning concentrates the distribution too heavily on observed modes, coverage for unobserved safety-critical modes degrades.  Mitigation: the exponential kernel ensures no observed mode receives exactly zero weight, and conservative mode priors can be added during early interaction.

---

## F. Implementation Summary

| Component | Location | Role |
|-----------|----------|------|
| Sinkhorn distance | `optimal_transport_predictor.py:191` | Entropy-regularized OT solver |
| Wasserstein distance | `optimal_transport_predictor.py:260` | $W_2$ between empirical distributions |
| Wasserstein barycenter | `optimal_transport_predictor.py:296` | Multi-distribution averaging via Bregman projection |
| Mode weight computation | `optimal_transport_predictor.py:581` | $\pi_t^C(m)$ via exponential Wasserstein kernel |
| Dynamics estimation | `optimal_transport_predictor.py:666` | $(\hat{b}_m, \hat{G}_m)$ from trajectory residuals |
| Parameter update | `sampler.py:775` | In-place ModeModel update |
| Scenario sampling | `sampler.py:893` | Wasserstein-weighted mode selection |
| OT observation pipeline | `scenario_module.py:291` | Real-time position/mode forwarding |
| Epsilon verification | `scenario_module.py` | Runtime $\hat{\varepsilon} \leq \varepsilon_p$ check |
