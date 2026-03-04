# GAN-Based Adversarial Scenario Generation for SHMPC

This document provides a detailed mathematical writeup of the **GAN-based adversarial scenario generation** used in the Future Statistical Learning MPC experiments. It is intended as a standalone appendix or methods section for publications focusing on GAN-enhanced scenario MPC.

## 1. Problem setting

We consider a scenario-based MPC controller for an ego vehicle operating among obstacles:

- Ego dynamics (discrete-time, horizon \(T\)):
  \[
  x_{k+1} = f(x_k, u_k), \quad k = 0,\dots,T-1,
  \]
  where \(x_k\) is the ego state and \(u_k\) the control input.

- Obstacles have state \(o^i_k\) and are approximated as discs of radius \(r_{\text{obs}}\); ego is a disc of radius \(r_{\text{ego}}\).

- Collision-avoidance constraint at time \(k\) for obstacle \(i\), scenario \(s\):
  \[
  \|p^{\text{ego}}_k - p^{i,s}_k\|_2 \ge r_{\text{ego}} + r_{\text{obs}},
  \]
  where \(p^{\text{ego}}_k\) and \(p^{i,s}_k\) are ego and obstacle positions.

Instead of sampling obstacle trajectories from a nominal predictive model, we use a **Generative Adversarial Network (GAN)** to produce **adversarial obstacle-relative trajectories** that make the collision-avoidance problem harder, improving safety when the MPC plans against these scenarios.

## 2. GAN trajectory model

### 2.1 Generator and discriminator

Let \(z \sim p_Z\) be latent noise (e.g. \(p_Z = \mathcal{N}(0, I)\)). The **generator** \(G_\theta\) maps this noise (and optionally context) to a multi-step relative trajectory:

\[
G_\theta : \mathbb{R}^{d_z} \to \mathbb{R}^{2 T_{\text{GAN}} N_{\text{obs}}},\quad
G_\theta(z, c) =
  \bigl\{ (\Delta x^{i}_k(z, c), \Delta y^{i}_k(z, c)) \bigr\}_{i=1,\dots,N_{\text{obs}},\ k=0,\dots,T_{\text{GAN}}}.
\]

Here:

- \(c\) is an optional **context vector** (e.g., ego initial state, lane geometry, nominal obstacle behavior),
- Each \((\Delta x^{i}_k, \Delta y^{i}_k)\) is a **relative offset** between the obstacle and ego for obstacle \(i\) at time \(k\).

The **discriminator** \(D_\phi\) takes a full relative trajectory as input and outputs a scalar \(D_\phi(\tau) \in (0, 1)\) representing the probability that \(\tau\) is real (from data) rather than generated:

\[
D_\phi: \mathbb{R}^{2 T_{\text{GAN}} N_{\text{obs}}} \to (0, 1).
\]

### 2.2 Adversarial training objective

We collect a dataset \(\{\tau^{(j)}\}_{j=1}^N\) of **adversarial or near-collision trajectories** for obstacles (relative to the ego), e.g. from stress-test simulations, critical near misses, or curated replay.

The standard non-saturating GAN objective is

\[
\min_{\theta} \max_{\phi}
\;\mathbb{E}_{\tau \sim p_{\text{data}}} \bigl[\log D_\phi(\tau)\bigr]
 + \mathbb{E}_{z\sim p_Z, c \sim p_C} \bigl[\log(1 - D_\phi(G_\theta(z, c)))\bigr],
\]

where:

- \(p_{\text{data}}\) is the empirical distribution over recorded adversarial trajectories,
- \(p_C\) is the distribution over contexts.

In practice we may add regularization (e.g. gradient penalties, spectral normalization) and condition both \(G_\theta\) and \(D_\phi\) on context \(c\) to improve stability and specificity.

The crucial point is that \(p_{\text{data}}\) is **not a generic traffic distribution**; it is **biased toward high-risk behaviors** (near-collisions, tight clearances). Thus, \(G_\theta\) learns to generate trajectories that are inherently challenging for the controller.

## 3. From generator output to SHMPC scenarios

At run time, SHMPC operates in the real world or in simulation with live obstacle states.

### 3.1 Relative-to-absolute reconstruction

Suppose at the start of an MPC step we have current obstacle positions
\[
p^{i,\text{current}}_0 \in \mathbb{R}^2,\quad i=1,\dots,N_{\text{obs}}.
\]

We sample latent variables \(z_s \sim p_Z\) (and context \(c\), if used) to generate \(S_{\text{GAN}}\) scenarios:

\[
(\Delta x^{i,s}_k, \Delta y^{i,s}_k) =
  [G_\theta(z_s, c)]_{(i,k)}.
\]

We then construct **absolute obstacle trajectories** for scenario \(s\) as
\[
p^{i,s}_k = p^{i,\text{current}}_0 + (\Delta x^{i,s}_k, \Delta y^{i,s}_k),
\quad k = 0,\dots,T.
\]

In the implementation, these relative offsets are pre-generated and stored in CSV form as

- Columns: `(scenario_id, obstacle_id, k, dx, dy)`,
- For each scenario we can quickly materialize \(p^{i,s}_k\) at run time.

### 3.2 GANScenarioCache

To avoid per-step file I/O, we introduce a simple cache:

1. **Pre-load** all rows from the GAN CSV into a cache structure:
   \[
   \{\Delta x^{i,s}_k, \Delta y^{i,s}_k\}_{s,i,k}.
   \]
2. At each MPC step, given current obstacle states, **materialize** scenarios in memory:
   \[
   p^{i,s}_k = p^{i,\text{current}}_0 + (\Delta x^{i,s}_k, \Delta y^{i,s}_k).
   \]

Mathematically this is identical to loading from disk each step; it simply reduces latency and CPU overhead.

## 4. QP formulation with GAN scenarios

### 4.1 Collision constraints per GAN scenario

For each GAN scenario \(s \in \{1,\dots, S_{\text{GAN}}\}\), obstacle \(i\), and time \(k\), we enforce the linearized collision constraint

\[
a^{i,s}_k{}^\top p^{\text{ego}}_k \ge b^{i,s}_k,
\]

where \((a^{i,s}_k, b^{i,s}_k)\) are obtained by linearizing the distance constraint
at a reference trajectory \(\bar{x}_k\). Stacking all such constraints yields a QP of the form

\[
\begin{aligned}
\min_{z} \quad & J(z) \\
\text{s.t.} \quad & C_{\text{dyn}} z \ge d_{\text{dyn}} \quad\text{(dynamics, box constraints)}\\
& C_{\text{GAN}} z \ge d_{\text{GAN}} \quad\text{(GAN collision constraints)}.
\end{aligned}
\]

The size of \(C_{\text{GAN}}\) scales as \(\mathcal{O}(S_{\text{GAN}} T N_{\text{obs}})\), so reducing \(S_{\text{GAN}}\) has near-linear effect on QP size and solve time.

### 4.2 Adversarial versus nominal sampling

In classical scenario MPC, scenarios are drawn i.i.d. from a **nominal predictive model**. Theory then often requires \(S \sim \mathcal{O}(d / \varepsilon)\) (with \(d\) the number of decision variables and \(\varepsilon\) the violation probability) to obtain formal chance-constraint guarantees, resulting in hundreds or thousands of scenarios.

Here, by generating from a **GAN trained on adversarial examples**, we replace many “mild” trajectories with a smaller number of **highly informative, adverse trajectories**:

- Each GAN scenario is likely to be close to a near-collision or tight clearance,
- A small number of such scenarios can cover a wide variety of dangerous behaviors,
- The controller thus sees more challenging futures per scenario budget, achieving large empirical safety gains with modest \(S_{\text{GAN}}\) (e.g., 12–30).

## 5. SHMPC_GAN variants

We implement three concrete GAN-based controllers.

### 5.1 SHMPC_GAN (full GAN)

- Choose a scenario budget \(S_{\text{GAN}}\) (e.g., 30).
- For each MPC step:
  1. Sample or select \(S_{\text{GAN}}\) GAN-generated trajectories,
  2. Materialize absolute trajectories via the cache,
  3. Build and solve the QP with all resulting collision constraints.

Empirically, SHMPC_GAN achieves:

- Very large collision-rate reductions versus baseline SHMPC (e.g. from \(\approx 26\%\) to \(\approx 4\%\)),
- Increased solve times (e.g. from \(\approx 5\) ms to \(\approx 13\) ms) due to more constraints.

### 5.2 SHMPC_GAN_Reduced

To balance safety and real-time requirements, we introduce **SHMPC_GAN_Reduced**:

- Fix a smaller number of used GAN scenarios \(S_{\text{red}} \ll S_{\text{GAN}}\).
- In implementation, we pre-generate a pool of adversarial trajectories and take the first \(S_{\text{red}}\) scenarios per solve.

Resulting QP:
\[
\min_{z} J(z) \quad \text{s.t.}\quad C_{\text{dyn}} z \ge d_{\text{dyn}},\; C_{\text{GAN}}^{(\text{red})} z \ge d_{\text{GAN}}^{(\text{red})},
\]
where \(C_{\text{GAN}}^{(\text{red})}\) contains constraints for only \(S_{\text{red}}\) scenarios.

**Empirical findings in our experiments:**

- With \(S_{\text{red}} = 12\), collision rate drops to \(\approx 3\%\) (large reduction vs SHMPC).
- Average QP solve time is \(\approx 4\)–\(5\) ms, comparable to or only slightly larger than baseline SHMPC.

Thus SHMPC_GAN_Reduced solves a practical constrained optimization:

\[
\begin{aligned}
\min_{S_{\text{red}}} \quad & \text{solve\_time}(S_{\text{red}}) \\
\text{s.t.} \quad & \frac{p_{\text{SHMPC}} - p_{\text{GAN}}(S_{\text{red}})}
                         {\max(p_{\text{SHMPC}}, \epsilon)} \;\ge\; \gamma,\\
& \text{solve\_time}(S_{\text{red}}) \le t_{\text{max}},
\end{aligned}
\]

for desired relative improvement \(\gamma\) and maximum allowable solve time \(t_{\text{max}}\). Empirically, \(\gamma\approx 0.8\)–0.9 and \(t_{\text{max}} \approx 10\) ms are simultaneously satisfied by \(S_{\text{red}} = 12\).

### 5.3 SHMPC_GAN_Quotient

SHMPC_GAN_Quotient combines GAN-generated scenarios with **quotient-space reduction**:

1. Start with a larger set of GAN scenarios \(\{s_1,\dots,s_{S_{\text{raw}}}\}\).
2. Compute a low-dimensional feature vector \(\Phi(s)\) per scenario, e.g.:
   \[
   \Phi(s) = \left[\overline{p^{\text{obs},s}},\; p^{\text{obs},s}_T,\; \min_k \|p^{\text{ego,ref}}_k - p^{\text{obs},s}_k\|\right].
   \]
3. Apply k-center selection in feature space to pick \(K\) representative scenarios:
   \[
   c_{m+1} = \arg\max_{s} \min_{c\in\{c_1,\dots,c_m\}} \|\Phi(s)-\Phi(c)\|_2.
   \]
4. Use only these \(K\) representative GAN trajectories in the MPC QP.

This reduces constraint count, but in our experiments the quotient reduction occasionally discards critical adversarial modes, yielding worse safety than SHMPC_GAN_Reduced despite lower solve time.

## 6. Real-time sweep for GAN configurations

To quantify the trade-off between safety and computation, we implemented a dedicated sweep executable.

### 6.1 Sweep setup

Fix:

- Scenario configuration for SHMPC (horizon, number of discs, etc.),
- A set of scenario counts \(\mathcal{S} = \{8, 10, 12, 15, 18, 20, 25, 30\}\),
- Number of rollouts per configuration \(N\),
- Target relative improvement \(\Gamma\) (e.g. \(0.5\) for 50%),
- Maximum acceptable average solve time \(t_{\text{max}}\) (e.g. 10 ms).

Procedure:

1. Run SHMPC with its standard scenario mechanism for \(N\) rollouts; estimate baseline collision rate
   \[
   p_{\text{SHMPC}} = \frac{\#\{\text{collisions under SHMPC}\}}{N}.
   \]
2. For each \(S \in \mathcal{S}\):
   - Configure the GAN method to use \(S\) scenarios per solve.
   - Run \(N\) rollouts; estimate collision rate \(p_{\text{GAN}}(S)\) and mean solve time \(t_{\text{GAN}}(S)\).
   - Compute **relative improvement**:
     \[
     \text{rel\_improve}(S) =
       \frac{p_{\text{SHMPC}} - p_{\text{GAN}}(S)}{\max(p_{\text{SHMPC}}, \epsilon)}.
     \]
   - Check constraints:
     \[
     \text{rel\_improve}(S) \ge \Gamma, \quad t_{\text{GAN}}(S) \le t_{\text{max}}.
     \]
3. Among \(S\) that satisfy both, select the **smallest \(S\)** (fewest scenarios) as preferred configuration.

### 6.2 Empirical outcome

Using this sweep in the implemented environment, we observed:

- \(p_{\text{SHMPC}} \approx 20\%\) collision rate,
- For full GAN with larger \(S\), collision rates dropped to \(\approx 4\%\) but with \(\approx 12\)–13 ms solve times.
- For SHMPC_GAN_Reduced:
  - \(S=12\) yields \(p_{\text{GAN}}(12) \approx 2.9\%\) (≈86% relative improvement),
  - Mean solve time \(t_{\text{GAN}}(12) \approx 4.8\) ms.

Thus, for \(\Gamma = 0.5\) and \(t_{\text{max}} = 10\) ms, \(S=12\) is the smallest scenario budget meeting both safety and real-time constraints.

## 7. Target collision benchmarks

Beyond relative safety improvements, we also evaluate GAN methods under **absolute collision targets**.

Given a target collision rate \(p_{\text{target}}\) (e.g. 2% or 5%), for each (method, hyperparameter) configuration we:

1. Run rollouts sequentially, tracking:
   - \(n\): number of rollouts,
   - \(c\): number of collisions.
2. Compute the **95% Wilson confidence interval** \([\text{ci\_lo}(c,n), \text{ci\_hi}(c,n)]\) for a Bernoulli parameter.
3. Continue until either:
   - (Success) \(\text{ci\_hi}(c,n) \le p_{\text{target}}\) and \(n\) exceeds a minimum threshold,
   - (Failure) \(\text{ci\_lo}(c,n) > p_{\text{target}}\), or
   - (Cap) \(n = N_{\max}\) is reached.

We then record:

- Whether the configuration **meets the target**,
- The number of rollouts used \(n\),
- The empirical rate \(\hat{p} = c/n\),
- The CI upper bound \(\text{ci\_hi}\),
- Mean and quantiles of **solve time**.

Under this framework:

- For a very strict target \(p_{\text{target}} = 2\%\), none of the tested methods (including GANs) reached a certified CI upper bound below 2% within the rollout cap in this environment.
- For a more moderate target \(p_{\text{target}} = 5\%\), GAN-based methods—particularly SHMPC_GAN_Reduced with \(S=12\)—do achieve certified collision rates below 5% while maintaining low solve times, outperforming classical and other learning-based variants.

## 8. Summary

The GAN-based scenario generation framework provides:

- A principled way to generate **adversarial obstacle-relative trajectories** for use in scenario MPC,
- A bridge between high-capacity generative models and classical QP-based MPC,
- Empirical safety gains that are **large in magnitude** and **computationally efficient** when combined with:
  - Scenario caching,
  - Reduced scenario counts \(S_{\text{red}}\),
  - Quotient-space ideas (with some caveats on safety vs. compression).

In the tested environment, SHMPC_GAN_Reduced with \(S=12\) emerges as a particularly attractive configuration:

- It reduces collision rates by ≈80–90% relative to SHMPC,
- It maintains solve times in the 4–5 ms range,
- It satisfies both relative-improvement and absolute-target (e.g. 5%) criteria under reasonable rollout budgets.

These properties make it a promising candidate for real-time deployment in safety-critical autonomous driving MPC stacks, while remaining compatible with the existing SHMPC formulation and infrastructure.

