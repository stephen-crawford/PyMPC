# Future Statistical Learning MPC – Mathematical Formulation of Implementations

This document summarizes the mathematical formulation of each implementation used in the **Future Statistical Learning MPC** experiments. It is written to be suitable as an appendix or methods section in a publication.

We assume:

- A single ego vehicle with discrete-time dynamics over horizon \(T\).
- A finite set of obstacles with mode-dependent linear dynamics.
- Scenario-based MPC with linearized collision constraints, solved as a quadratic program (QP).

## 1. Baseline scenario MPC / SHMPC

Let \(x_k \in \mathbb{R}^{n_x}\) be the ego state and \(u_k \in \mathbb{R}^{n_u}\) the control at time \(k\). We consider discrete-time dynamics
\[
x_{k+1} = f(x_k, u_k), \quad k = 0,\dots,T-1,
\]
with a bicycle-like model in implementation (linearized for QP).

Each obstacle \(i\) has state \(o^i_k\), a set of motion modes \(\mathcal{M}^i\), and mode-dependent linear dynamics
\[
o^i_{k+1} = A^{m^i_k} o^i_k + b^{m^i_k}, \quad m^i_k \in \mathcal{M}^i.
\]

A **scenario** \(s\) consists of a sampled mode sequence for each obstacle
\[
\{m^{i,s}_k\}_{k=0}^{T-1},\quad i=1,\dots,N_{\text{obs}},
\]
and the corresponding obstacle trajectories \(\{o^{i,s}_k\}_{k=0}^{T}\).

### 1.1 Collision constraints and linearization

We approximate ego and obstacles by discs of radii \(r_{\text{ego}}, r_{\text{obs}}\), with 2D positions \(p^{\text{ego}}_k(x_k)\) and \(p^{i,s}_k(o^{i,s}_k)\). A hard collision-avoidance constraint is
\[
\|p^{\text{ego}}_k - p^{i,s}_k\|_2 \ge r_{\text{ego}} + r_{\text{obs}}.
\]

Around a reference trajectory \(\{\bar{x}_k\}\), we linearize each constraint to obtain, for each obstacle \(i\), scenario \(s\), and time \(k\),
\[
a^{i,s}_k{}^\top p^{\text{ego}}_k \;\ge\; b^{i,s}_k,
\]
where \(a^{i,s}_k \in \mathbb{R}^2\) is a normal vector, and \(b^{i,s}_k\) encodes the local safety margin at the linearization point.

Let \(z\) denote the stacked decision vector containing all states and controls over the horizon. Stacking all constraints yields
\[
C z \ge d,
\]
where rows of \(C\) and entries of \(d\) correspond to the linearized collision, dynamics, and box constraints.

### 1.2 SHMPC objective and safe horizon

We define a quadratic stage cost that penalizes deviation from a reference path and control effort:
\[
J(z) = \sum_{k=0}^{T-1} \bigl( \|p^{\text{ego}}_k - p^{\text{ref}}_k\|_Q^2 + \|u_k\|_R^2 \bigr).
\]

The **scenario MPC QP** is
\[
\begin{aligned}
\min_{z} \quad & J(z) \\
\text{s.t.} \quad & C z \ge d.
\end{aligned}
\]

The **Safe Horizon MPC (SHMPC)** variant attempts to solve this QP with the full horizon. If infeasible, it reduces the effective horizon \(T'\le T\) (dropping the last steps) until a feasible solution is found, and executes the first control input from that truncated plan.

We record:

- The **active set** of constraints and scenarios at the optimum (those with zero or near-zero slack).
- Solve times and horizon truncation as performance metrics.

## 2. Conformal safety wrappers

Conformal wrappers build **mode-conditional uncertainty tubes** around obstacle predictions and use them both for:

1. Tightening collision constraints, and
2. Allocating scenario budget to modes near the safety boundary.

### 2.1 Mode-conditional residuals and quantiles

For each obstacle \(i\), mode \(m \in \mathcal{M}^i\), and time \(k\), we collect residuals between predicted and realized obstacle positions during calibration:
\[
r^{(j)}_{k,m} = \left\|\hat{p}^{i,(j)}_{k,m} - p^{i,(j)}_{k}\right\|_2, \quad j = 1,\dots,N_{k,m}.
\]

For a target coverage level \(1-\delta\), we define the **conformal quantile**
\[
q_{k,m} = \text{quantile}_{1-\delta}\bigl(\{r^{(j)}_{k,m}\}_{j=1}^{N_{k,m}}\bigr).
\]

This yields a tube
\[
C_{k,m} = \left\{\xi \in \mathbb{R}^2 : \|\xi - \hat{p}^{i}_{k,m}\|_2 \le q_{k,m}\right\}.
\]

### 2.2 Robust constraint tightening

For a collision-avoidance constraint at time \(k\) and mode \(m\)
\[
g_{k,m}(x, \xi) = a_{k,m}^\top p^{\text{ego}}_k - b_{k,m}(\xi) \ge 0,
\]
and tube \(C_{k,m}\), we approximate the worst-case violation over the tube via
\[
\max_{\xi\in C_{k,m}} g_{k,m}(x, \xi)
  \approx g_{k,m}(x, \hat{p}^{i}_{k,m}) - q_{k,m} \|a_{k,m}\|_2.
\]

This motivates a **tightened offset**
\[
b^{\text{robust}}_{k,m} = b_{k,m} + q_{k,m} \|a_{k,m}\|_2,
\]
which is used in the QP to enforce robust collision avoidance against the tube.

### 2.3 Boundary scores and scenario allocation

We compute a **boundary score** for each mode \(m\) as
\[
s_m = \max_{k}
\left(
g_{k,m}^{\text{mean}} + q_{k,m}\,\|\nabla_\xi g_{k,m}\|
\right),
\]
where \(g_{k,m}^{\text{mean}}\) is the constraint value at the tube center and \(\nabla_\xi g_{k,m}\) the sensitivity to obstacle position.

Scenario allocation over modes is then defined by
\[
S_m \propto \exp(\alpha s_m), \quad \sum_m S_m = S,
\]
with a minimum per-mode floor. Modes with larger \(s_m\) receive more scenarios, focusing simulation on configurations that lie near the safety boundary.

## 3. Hazard / change-point switch sampling

The hazard-based sampler biases scenarios toward modes that are likely to switch soon.

Let \(h\) be the **time since last mode switch** for a given obstacle. For each mode \(m\), define a logistic hazard:
\[
\lambda_m(h) = \sigma(\theta_m h + \theta_{0,m}),
\quad
\sigma(x) = \frac{1}{1 + e^{-x}}.
\]

Nominal mode probabilities \(\pi_m\) are reweighted via
\[
\tilde{\pi}_m \propto \pi_m \exp\bigl(\eta \lambda_m(h)\bigr),
\]
with \(\sum_m \tilde{\pi}_m = 1\). Scenario counts are then
\[
S_m = \max\{1, \operatorname{round}(S\,\tilde{\pi}_m)\},\quad
\sum_m S_m \approx S.
\]

This increases the number of scenarios probing modes whose hazard of switching is high, improving coverage of difficult mode transitions.

## 4. Risk-directed bandit allocation

We treat scenario allocation as a multi-armed bandit problem over modes \(m\).

For each mode \(m\) we maintain:

- \(n_m\): the number of times mode \(m\) has been sampled,
- \(\hat{R}_m\): an empirical risk estimate for mode \(m\) (e.g. fraction of rollouts where collisions or severe near-misses involved that mode).

We define a **UCB (Upper Confidence Bound) index**:
\[
\text{UCB}_m(t) = \hat{R}_m + \beta \sqrt{\frac{\log t}{n_m + 1}},
\]
with exploration parameter \(\beta > 0\) and time index \(t\) (e.g. total number of allocations so far).

Scenario counts are then chosen as
\[
S_m \propto \max(\text{UCB}_m, \epsilon),\quad \sum_m S_m = S,
\]
with a small floor \(\epsilon > 0\). This allocates more scenarios to modes that appear risky or under-explored.

## 5. Learned OT ground cost

Wasserstein DRO over mode distributions uses a **ground cost** matrix \(D_{ij}\) between trajectories. Rather than a simple metric (e.g. mean position distance), we define a **learned cost** via an embedding \(\phi(\tau)\).

Let \(\tau_i\) denote a trajectory (e.g. a sequence of obstacle positions and margins). Define an embedding \(\phi: \tau \mapsto \mathbb{R}^d\), for example
\[
\phi(\tau) = \bigl[p_{0},\dots,p_T,\ \text{margin}_0,\dots,\text{margin}_T\bigr] \in \mathbb{R}^d.
\]

The learned ground cost is
\[
D_{ij} = \|\phi(\tau_i) - \phi(\tau_j')\|_2,
\]
where \(\tau_j'\) are trajectories with associated risk scores \(r_j\).

The Wasserstein DRO dual problem for a Wasserstein-1 ball of radius \(\varepsilon\) is (Kantorovich dual):
\[
\min_{\lambda \ge 0} \left\{
  \lambda \varepsilon + \sum_i w_i \max_j \bigl(r_j - \lambda D_{ij}\bigr)
\right\},
\]
where \(w_i\) are nominal weights. We solve this via one-dimensional search over \(\lambda\).

## 6. Adaptive DRO radius (shift detection)

We adapt the DRO radius \(\varepsilon(t)\) based on residuals between predicted and realized behavior.

Let \(d_t\) be a scalar residual at time \(t\) (e.g. prediction error norm). Over a sliding window of size \(W\), define
\[
\bar{d}_t = \frac{1}{W} \sum_{\tau=t-W+1}^t d_\tau,
\]
and a baseline residual level \(\mu_0\). The adaptive factor is
\[
\rho(t) = \rho_{\min} + k \cdot \max(0, \bar{d}_t - \mu_0),
\]
for gain \(k>0\). We then set \(\varepsilon(t)\) as a function of \(\rho(t)\) (e.g. proportional), increasing conservativeness when residuals rise.

## 7. Diffusion-style calibrated scenario selection

We approximate a diffusion trajectory generator by sampling a large **candidate pool** and selecting a subset based on severity and diversity, with a calibrated empirical risk bound.

1. Generate \(K\) candidate trajectories \(\{\tau_j\}_{j=1}^K\) by perturbing a mean obstacle path (e.g. Gaussian noise or a diffusion placeholder).
2. For each \(\tau_j\), define a **severity score**:
   \[
   \text{sev}(\tau_j) =
     \max_k \bigl((r_{\text{ego}}+r_{\text{obs}}) - \|p^{\text{ego}}_k - p^{\text{obs}}_{j,k}\|_2\bigr)_+.
   \]
3. Rank candidates by severity, optionally prune near-duplicates via a distance threshold in trajectory space, then select the top \(S\) for use in MPC.
4. After running \(N\) rollouts, obtain empirical collision rate \(\hat{p}\) and a **Bernstein-style** upper bound
   \[
   p_{\text{upper}} = \hat{p} +
     \sqrt{\frac{2 \hat{p}(1 - \hat{p}) \log(1/\delta)}{N}} +
     \frac{7 \log(1/\delta)}{3 (N-1)}
   \]
   at confidence level \(1-\delta\).

This yields adversarial, diverse scenarios together with a calibrated bound on collision risk.

## 8. Counterfactual intent modeling

We consider a finite set of ego candidate control sequences \(U = \{u^{(j)}_{0:T-1}\}_{j=1}^J\). For each candidate \(j\), we assume a conditional obstacle trajectory distribution
\[
p(\xi \mid u^{(j)}), \quad \xi = \{\xi_k\}_{k=0}^T.
\]

We estimate collision probability for candidate \(j\) via Monte Carlo:
\[
\hat{p}_j = \frac{1}{N} \sum_{\ell=1}^N
\mathbf{1}\{\xi^{(j,\ell)} \text{ collides with } u^{(j)}\},
\quad \xi^{(j,\ell)} \sim p(\xi \mid u^{(j)}).
\]

We then select the **counterfactually safest** ego plan:
\[
j^* = \arg\min_{j} \hat{p}_j.
\]

The current implementation uses a simple surrogate for \(p(\xi \mid u)\); in principle one can learn this distribution from logged driving data.

## 9. Dual risk monitor

Let the QP be written with inequality constraints \(g_i(z) \ge 0\), and let \(\lambda_i \ge 0\) denote the optimal dual variables. Define:

- Constraint margin \(m_i = g_i(z^*)\),
- Margin violation \(\text{margin\_violation} = \sum_i \min(0, m_i)\),
- Dual magnitude sum \(\text{sum\_duals} = \sum_i |\lambda_i|\).

The **dual risk monitor** computes a logistic risk score:
\[
\hat{r} = \sigma\bigl( \theta_0 + \theta_1 \cdot \text{margin\_violation}
                      + \theta_2 \cdot \text{sum\_duals} \bigr),
\]
with parameters \(\theta_0,\theta_1,\theta_2\). If \(\hat{r} > \tau\) for some threshold \(\tau\), we trigger interventions (e.g. add scenarios, tighten constraints, or engage RTA).

## 10. Certificate-first learning

In the certificate-first view, the predictive model outputs not just a mean trajectory but a **safety certificate** in the form of tubes \(\{C_k\}\).

For each time step \(k\):

- The model outputs a center \(\hat{p}_k\) and radius \(r_k\), defining
  \[
  C_k = \{\xi \in \mathbb{R}^2 : \|\xi - \hat{p}_k\|_2 \le r_k\}.
  \]

- The **certificate volume** is
  \[
  V = \sum_{k=0}^{T} c_d r_k^d,
  \]
  where \(d\) is the spatial dimension (here \(d=2\)) and \(c_d\) is the unit-ball volume constant.

Collision constraints are tightened as
\[
a_k^\top p^{\text{ego}}_k \ge b_k^{\text{robust}}
  = b_k - r_k \|a_k\|.
\]

The (conceptual) training objective is to minimize \(V\) subject to coverage constraints
\[
\mathbb{P}\bigl(p^i_k \in C_k \text{ for all }k\bigr) \ge 1 - \delta.
\]

## 11. Scenario compiler

The scenario compiler builds a **minimal witness set** \(W\) that certifies safety under a family of scenarios.

1. Initialize \(W\) with a small set of scenarios drawn from the nominal predictive model.
2. Solve MPC with scenario set \(W\), obtaining plan \(\{x_k^*\}\).
3. Invoke an adversary \(\mathcal{A}\) that searches for a violating scenario:
   \[
   s^\star = \arg\max_{s \in \mathcal{S}} \max_k
   \bigl((r_{\text{ego}}+r_{\text{obs}}) - \|p^{\text{ego},*}_k - p^{\text{obs},s}_k\|_2\bigr).
   \]
   If the maximized violation is \(\le 0\), we stop.
4. Otherwise, add \(s^\star\) to \(W\) and repeat.

This procedure generates a compact set of scenarios that are each necessary (witnesses of potential violations).

## 12. Runtime Assurance (RTA)

RTA wraps the learned or MPC controller in a safety layer with:

- A **monitor** \(M(x, \text{cert}) \in \{\text{safe}, \text{unsafe}\}\) (e.g. based on TTC or distance, optionally conditioned on certificates).
- A **fallback controller** \(u_{\text{safe}}(x)\) (e.g. hard braking).

Given the nominal MPC control \(u_{\text{MPC}}(x)\), RTA outputs
\[
u_{\text{RTA}}(x) =
\begin{cases}
u_{\text{MPC}}(x), & M(x,\text{cert}) = \text{safe},\\
u_{\text{safe}}(x), & M(x,\text{cert}) = \text{unsafe}.
\end{cases}
\]

This architecture keeps the safety-critical logic small and verifiable, while allowing complex learning-based stacks underneath.

## 13. GAN adversarial scenarios (high-level)

GAN-based scenario generation replaces i.i.d. draws from a nominal predictive model with **learned adversarial trajectories**.

Let the generator \(G_\theta\) map latent noise \(z \sim p_Z\) (and optionally context \(c\)) to obstacle-relative trajectories:
\[
G_\theta(z,c) =
  \bigl\{ (\Delta x^{i}_k(z,c), \Delta y^{i}_k(z,c)) \bigr\}_{i,k}.
\]

Given current obstacle positions \(p_0^{i,\text{current}}\), we construct absolute trajectories for scenario \(s\) by
\[
p^{i,s}_k = p_0^{i,\text{current}} + (\Delta x^{i,s}_k, \Delta y^{i,s}_k), \quad k=0,\dots,T.
\]

These trajectories define scenario sets for SHMPC:

- **SHMPC_GAN**: use \(S_{\text{GAN}}\) scenarios from \(G_\theta\).
- **SHMPC_GAN_Reduced**: use a smaller number \(S_{\text{red}}\) (e.g. 12).
- **SHMPC_GAN_Quotient**: start from a larger GAN pool and reduce to \(K\) representatives via quotient-space clustering.

The empirical real-time sweep (separate document) selects \(S_{\text{red}} = 12\) as a configuration that achieves large collision reduction vs SHMPC while meeting strict solve-time constraints.

---

For a more detailed treatment of the GAN-based approaches (training objectives, scenario construction, real-time sweeps, and comparisons), see `09_gan_adversarial_scenarios/GAN_FORMULATION.md`.

