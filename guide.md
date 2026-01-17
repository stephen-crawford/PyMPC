# Mathematical Implementation Guide
## Adaptive Scenario-Based Trajectory Optimization with Switching Dynamics

This document provides a complete mathematical specification for implementing the adaptive scenario-based MPC method. All equation numbers reference the revised paper.

---

## Table of Contents

1. [Notation and Constants](#1-notation-and-constants)
2. [Data Structures](#2-data-structures)
3. [Core Mathematical Operations](#3-core-mathematical-operations)
4. [Algorithm Implementations](#4-algorithm-implementations)
5. [Numerical Considerations](#5-numerical-considerations)
6. [Testing and Validation](#6-testing-and-validation)

---

## 1. Notation and Constants

### 1.1 Index Sets

| Symbol | Description | Typical Range |
|--------|-------------|---------------|
| $t$ | Discrete time index | $t \in \mathbb{N}$ |
| $k$ | Prediction step within horizon | $k \in \{0, 1, \ldots, N\}$ |
| $v$ | Obstacle index | $v \in \mathcal{V} = \{1, \ldots, V\}$ |
| $d$ | Ego disc index | $d \in \{1, \ldots, D\}$ |
| $m$ | Mode index | $m \in \mathcal{M}_{\mathcal{C}}$ |
| $i$ | Scenario index | $i \in \{1, \ldots, S\}$ |

### 1.2 Dimensions

| Symbol | Description | Typical Value |
|--------|-------------|---------------|
| $n_x$ | Ego state dimension | 4 (x, y, θ, v) |
| $n_u$ | Ego input dimension | 2 (a, δ) |
| $n_v$ | Obstacle state dimension | 4 (x, y, vx, vy) |
| $n_\omega$ | Process noise dimension | 2 or 4 |
| $N$ | Planning horizon | 10-30 |
| $S$ | Number of scenarios | 100-5000 |
| $V$ | Number of obstacles | 1-20 |
| $D$ | Number of ego discs | 1-5 |
| $M_{\mathcal{C}}$ | Modes per class | 2-5 |

### 1.3 Key Parameters

```python
@dataclass
class ScenarioMPCConfig:
    # Horizon and timing
    horizon: int = 20                    # N
    dt: float = 0.1                      # Δt (seconds)
    
    # Scenario parameters
    epsilon: float = 0.05                # ε - violation probability
    beta: float = 0.01                   # β - confidence level
    
    # Computed sample size: S >= (2/ε)(ln(1/β) + d)
    # For d=120, ε=0.05, β=0.01: S >= 4984
    
    # Scenario reduction
    num_removal: int = 0                 # R - scenarios to remove
    pruning_clusters: int = 10           # k for normal clustering
    
    # Mode weight parameters
    recency_decay: float = 0.1           # λ in recency prior
    frequency_smoothing: float = 1.0     # α in frequency prior
    detection_threshold: float = 0.3     # γ for soft mode detection
    
    # Geometry
    ego_disc_radii: List[float]          # r_d^ego
    ego_disc_offsets: List[Tuple[float, float]]  # disc centers relative to ego frame
    
    # Cost weights
    Q: np.ndarray  # State cost (n_x × n_x)
    R: np.ndarray  # Input cost (n_u × n_u)  
    P: np.ndarray  # Terminal cost (n_x × n_x)
```

---

## 2. Data Structures

### 2.1 State Representations

```python
@dataclass
class EgoState:
    """Ego vehicle state x^ego ∈ ℝ^{n_x}"""
    x: float       # Position x (m)
    y: float       # Position y (m)
    theta: float   # Heading (rad)
    v: float       # Speed (m/s)
    
    def to_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.v])
    
    def position(self) -> np.ndarray:
        """Extract position p = [x, y]^T"""
        return np.array([self.x, self.y])


@dataclass
class ObstacleState:
    """Obstacle state x^v ∈ ℝ^{n_v}"""
    x: float       # Position x (m)
    y: float       # Position y (m)
    vx: float      # Velocity x (m/s)
    vy: float      # Velocity y (m/s)
    
    def to_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy])
    
    def position(self) -> np.ndarray:
        """Extract position p^v = [x, y]^T"""
        return np.array([self.x, self.y])


@dataclass 
class EgoInput:
    """Control input u ∈ ℝ^{n_u}"""
    acceleration: float  # a (m/s²)
    steering: float      # δ (rad)
    
    def to_vector(self) -> np.ndarray:
        return np.array([self.acceleration, self.steering])
```

### 2.2 Mode and Dynamics Models

```python
@dataclass
class ModeModel:
    """
    Mode-dependent dynamics: x_{t+1} = A_m x_t + b_m + G_m ω_t
    Reference: Equation (4)
    """
    mode_id: int                    # m ∈ M_C
    A: np.ndarray                   # A_m ∈ ℝ^{n_v × n_v}
    b: np.ndarray                   # b_m ∈ ℝ^{n_v}
    G: np.ndarray                   # G_m ∈ ℝ^{n_v × n_ω}
    
    def propagate(self, x: np.ndarray, omega: np.ndarray = None) -> np.ndarray:
        """
        Propagate state one step: x_{t+1} = A_m x_t + b_m + G_m ω_t
        
        Args:
            x: Current state x_t ∈ ℝ^{n_v}
            omega: Process noise ω_t ~ N(0, I), sampled if None
            
        Returns:
            Next state x_{t+1} ∈ ℝ^{n_v}
        """
        if omega is None:
            omega = np.random.randn(self.G.shape[1])
        return self.A @ x + self.b + self.G @ omega


class ObstacleClass:
    """
    Behavioral class C with mode set M_C
    Reference: Definition 1
    """
    def __init__(self, class_id: str, modes: Dict[int, ModeModel]):
        self.class_id = class_id           # e.g., "pedestrian", "vehicle"
        self.modes = modes                  # {mode_id: ModeModel}
        self.mode_set = set(modes.keys())  # M_C = {1, ..., M_C}
    
    def get_mode(self, m: int) -> ModeModel:
        return self.modes[m]
    
    @property
    def num_modes(self) -> int:
        return len(self.modes)
```

### 2.3 Standard Mode Models

```python
def create_constant_velocity_mode(dt: float, sigma_v: float = 0.5) -> ModeModel:
    """
    Constant velocity model: position integrates velocity, velocity has noise.
    
    State: [x, y, vx, vy]^T
    Dynamics: 
        x_{t+1} = x_t + dt * vx_t
        y_{t+1} = y_t + dt * vy_t
        vx_{t+1} = vx_t + σ_v * ω_x
        vy_{t+1} = vy_t + σ_v * ω_y
    """
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    b = np.zeros(4)
    G = np.array([
        [0, 0],
        [0, 0],
        [sigma_v, 0],
        [0, sigma_v]
    ])
    return ModeModel(mode_id=1, A=A, b=b, G=G)


def create_constant_acceleration_mode(dt: float, ax: float, ay: float, 
                                       sigma_a: float = 0.3) -> ModeModel:
    """
    Constant acceleration model with deterministic bias.
    
    State: [x, y, vx, vy]^T
    b_m encodes the acceleration bias.
    """
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    b = np.array([0.5*dt**2*ax, 0.5*dt**2*ay, dt*ax, dt*ay])
    G = np.array([
        [0.5*dt**2, 0],
        [0, 0.5*dt**2],
        [dt*sigma_a, 0],
        [0, dt*sigma_a]
    ])
    return ModeModel(mode_id=2, A=A, b=b, G=G)


def create_goal_directed_mode(dt: float, goal: np.ndarray, k_goal: float = 0.5,
                               sigma: float = 0.3) -> ModeModel:
    """
    Goal-directed model: velocity biased toward goal.
    
    This is a linearized approximation around the current position.
    For a proper implementation, re-linearize each step.
    """
    # Simplified: constant velocity toward goal
    # More accurate: recompute b_m based on current state at each step
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1-k_goal*dt, 0],
        [0, 0, 0, 1-k_goal*dt]
    ])
    # b depends on goal - this is state-dependent, handle in propagation
    b = np.zeros(4)  # Placeholder
    G = np.array([
        [0, 0],
        [0, 0],
        [sigma, 0],
        [0, sigma]
    ])
    return ModeModel(mode_id=3, A=A, b=b, G=G)
```

### 2.4 Mode History and Weights

```python
@dataclass
class ModeHistory:
    """
    Mode history set H_t^C for a class
    Reference: Definition 3, Equation (7)
    """
    class_id: str
    observed_modes: Set[int]                    # H_t^C ⊆ M_C
    observation_times: Dict[int, List[int]]     # mode -> list of times observed
    observation_counts: Dict[int, int]          # mode -> count
    
    def update(self, mode: int, time: int):
        """
        Update history: H_t^C = H_{t-1}^C ∪ {m_t^v}
        """
        self.observed_modes.add(mode)
        if mode not in self.observation_times:
            self.observation_times[mode] = []
            self.observation_counts[mode] = 0
        self.observation_times[mode].append(time)
        self.observation_counts[mode] += 1
    
    def get_uniform_weights(self) -> Dict[int, float]:
        """
        Uniform prior: π(m) = 1/|H_t^C|
        Reference: Definition 6
        """
        n = len(self.observed_modes)
        if n == 0:
            return {}
        weight = 1.0 / n
        return {m: weight for m in self.observed_modes}
    
    def get_recency_weights(self, current_time: int, 
                            lambda_decay: float = 0.1) -> Dict[int, float]:
        """
        Recency-weighted prior: π(m) ∝ exp(-λ(t - τ_m))
        Reference: Definition 7, Equation (12)
        
        Args:
            current_time: Current time t
            lambda_decay: Decay parameter λ ≥ 0
        """
        if len(self.observed_modes) == 0:
            return {}
        
        weights = {}
        for m in self.observed_modes:
            tau_m = max(self.observation_times[m])  # Most recent observation
            weights[m] = np.exp(-lambda_decay * (current_time - tau_m))
        
        # Normalize
        total = sum(weights.values())
        return {m: w / total for m, w in weights.items()}
    
    def get_frequency_weights(self, alpha: float = 1.0) -> Dict[int, float]:
        """
        Frequency-weighted prior: π(m) = (n_m + α) / Σ(n_{m'} + α)
        Reference: Definition 8, Equation (13)
        
        Args:
            alpha: Dirichlet smoothing parameter α > 0
        """
        if len(self.observed_modes) == 0:
            return {}
        
        weights = {}
        total = 0.0
        for m in self.observed_modes:
            w = self.observation_counts.get(m, 0) + alpha
            weights[m] = w
            total += w
        
        return {m: w / total for m, w in weights.items()}
```

### 2.5 Trajectory and Scenario Structures

```python
@dataclass
class ObstacleTrajectory:
    """
    Trajectory ξ^v = (x_{t+1}^v, ..., x_{t+N}^v) ∈ ℝ^{N·n_v}
    """
    obstacle_id: int
    positions: np.ndarray      # Shape: (N, 2) - just positions for collision checking
    states: np.ndarray         # Shape: (N, n_v) - full states
    mode_sequence: List[int]   # (m_1, ..., m_N)
    
    def position_at_step(self, k: int) -> np.ndarray:
        """Get position p^v_{t+k} ∈ ℝ^2"""
        return self.positions[k-1]  # k=1,...,N maps to index 0,...,N-1


@dataclass
class Scenario:
    """
    A single scenario i containing trajectories for all obstacles.
    ξ^{(i)} = {ξ^{v,(i)}}_{v∈V}
    """
    scenario_id: int
    obstacle_trajectories: Dict[int, ObstacleTrajectory]  # v -> trajectory
    
    def get_obstacle_position(self, v: int, k: int) -> np.ndarray:
        """Get p^{(i)}_{v,k} = p^v(ξ^{v,(i)}_{t+k})"""
        return self.obstacle_trajectories[v].position_at_step(k)


@dataclass
class TrajectoryMoments:
    """
    Gaussian trajectory distribution moments under mode sequence m
    Reference: Proposition 1, Equations (10)-(11)
    """
    mode_sequence: Tuple[int, ...]
    mean: np.ndarray           # μ^m ∈ ℝ^{N·n_v}
    covariance: np.ndarray     # Σ^m ∈ ℝ^{N·n_v × N·n_v}
    
    # Block structure for per-step access
    mean_per_step: np.ndarray      # Shape: (N, n_v)
    cov_blocks: np.ndarray         # Shape: (N, N, n_v, n_v) - Σ_{k,ℓ}
```

---

## 3. Core Mathematical Operations

### 3.1 Trajectory Moment Computation

```python
def compute_trajectory_moments(
    x0: np.ndarray,
    mode_sequence: Tuple[int, ...],
    mode_models: Dict[int, ModeModel],
    horizon: int
) -> TrajectoryMoments:
    """
    Compute closed-form trajectory mean and covariance.
    Reference: Proposition 1, Equations (10)-(11)
    
    μ_k^m(x_0) = Φ_k^m x_0 + Σ_{j=1}^k Φ_{k:j+1}^m b_{m_j}
    Σ_{k,ℓ}^m = Σ_{j=1}^{min(k,ℓ)} Φ_{k:j+1}^m G_{m_j} G_{m_j}^T (Φ_{ℓ:j+1}^m)^T
    
    where:
        Φ_k^m = A_{m_k} A_{m_{k-1}} ... A_{m_1}
        Φ_{k:ℓ}^m = A_{m_k} ... A_{m_ℓ} for k ≥ ℓ
        Φ_{k:k+1}^m = I
    """
    N = horizon
    n_v = x0.shape[0]
    
    # Precompute state transition matrices Φ_k^m
    # Φ[k] = A_{m_k} @ A_{m_{k-1}} @ ... @ A_{m_1}
    Phi = [np.eye(n_v)]  # Φ_0 = I
    for k in range(1, N + 1):
        m_k = mode_sequence[k - 1]  # 0-indexed
        A_mk = mode_models[m_k].A
        Phi.append(A_mk @ Phi[k - 1])
    
    # Precompute Φ_{k:ℓ} matrices
    # Phi_partial[k][ell] = Φ_{k:ℓ}^m for k >= ℓ
    def get_Phi_partial(k: int, ell: int) -> np.ndarray:
        """Compute Φ_{k:ℓ} = A_{m_k} @ ... @ A_{m_ℓ}"""
        if k < ell:
            raise ValueError(f"k={k} must be >= ell={ell}")
        if k == ell:
            return mode_models[mode_sequence[k - 1]].A
        # Φ_{k:ℓ} = Φ_k @ Φ_{ℓ-1}^{-1} but we compute directly
        result = np.eye(n_v)
        for j in range(ell, k + 1):
            m_j = mode_sequence[j - 1]
            result = mode_models[m_j].A @ result
        return result
    
    # Compute means: μ_k = Φ_k x_0 + Σ_{j=1}^k Φ_{k:j+1} b_{m_j}
    mean_per_step = np.zeros((N, n_v))
    for k in range(1, N + 1):
        # First term: Φ_k x_0
        mu_k = Phi[k] @ x0
        
        # Second term: Σ_{j=1}^k Φ_{k:j+1} b_{m_j}
        for j in range(1, k + 1):
            m_j = mode_sequence[j - 1]
            b_mj = mode_models[m_j].b
            if j == k:
                Phi_k_j1 = np.eye(n_v)  # Φ_{k:k+1} = I
            else:
                Phi_k_j1 = get_Phi_partial(k, j + 1)
            mu_k = mu_k + Phi_k_j1 @ b_mj
        
        mean_per_step[k - 1] = mu_k
    
    # Compute covariances: Σ_{k,ℓ} = Σ_{j=1}^{min(k,ℓ)} Φ_{k:j+1} G_{m_j} G_{m_j}^T Φ_{ℓ:j+1}^T
    cov_blocks = np.zeros((N, N, n_v, n_v))
    for k in range(1, N + 1):
        for ell in range(1, N + 1):
            Sigma_kl = np.zeros((n_v, n_v))
            for j in range(1, min(k, ell) + 1):
                m_j = mode_sequence[j - 1]
                G_mj = mode_models[m_j].G
                
                if j == k:
                    Phi_k_j1 = np.eye(n_v)
                else:
                    Phi_k_j1 = get_Phi_partial(k, j + 1)
                
                if j == ell:
                    Phi_ell_j1 = np.eye(n_v)
                else:
                    Phi_ell_j1 = get_Phi_partial(ell, j + 1)
                
                Sigma_kl += Phi_k_j1 @ G_mj @ G_mj.T @ Phi_ell_j1.T
            
            cov_blocks[k - 1, ell - 1] = Sigma_kl
    
    # Assemble full mean and covariance
    mean = mean_per_step.flatten()
    covariance = np.block([
        [cov_blocks[k, ell] for ell in range(N)]
        for k in range(N)
    ])
    
    return TrajectoryMoments(
        mode_sequence=mode_sequence,
        mean=mean,
        covariance=covariance,
        mean_per_step=mean_per_step,
        cov_blocks=cov_blocks
    )
```

### 3.2 Sample Size Computation

```python
def compute_required_samples(
    epsilon: float,
    beta: float,
    n_x: int,
    n_u: int,
    horizon: int,
    num_removal: int = 0
) -> int:
    """
    Compute required sample size S.
    Reference: Theorem 1, Equation (23); Proposition 9, Equation (28)
    
    S >= (2/ε)(ln(1/β) + d + R)
    
    where d = N·n_x + N·n_u is the number of decision variables.
    
    Args:
        epsilon: Violation probability ε ∈ (0, 1)
        beta: Confidence level β ∈ (0, 1)
        n_x: Ego state dimension
        n_u: Ego input dimension
        horizon: Planning horizon N
        num_removal: Number of scenarios to remove R
        
    Returns:
        Required number of scenarios S
    """
    d = horizon * n_x + horizon * n_u
    S = (2.0 / epsilon) * (np.log(1.0 / beta) + d + num_removal)
    return int(np.ceil(S))


# Example:
# epsilon=0.05, beta=0.01, n_x=4, n_u=2, N=20, R=0
# d = 20*4 + 20*2 = 120
# S >= (2/0.05)(ln(100) + 120) = 40 * (4.605 + 120) = 4984.2
# S = 4985
```

### 3.3 Linearized Collision Constraints

```python
def compute_collision_constraint(
    ego_disc_center: np.ndarray,    # c_d(x^ego) ∈ ℝ^2
    ego_disc_radius: float,          # r_d^ego
    obstacle_position: np.ndarray,   # p^v ∈ ℝ^2
    obstacle_radius: float,          # r^v
    reference_ego_center: np.ndarray,  # c̄_d (linearization point)
    reference_obstacle_pos: np.ndarray  # p̄^v (linearization point)
) -> Tuple[np.ndarray, float]:
    """
    Compute linearized collision constraint.
    Reference: Definition 11, Equations (17)-(18)
    
    Constraint: n^T (p^v - c_d) >= r_d^ego + r^v
    
    Where n = (p̄^v - c̄_d) / ||p̄^v - c̄_d||
    
    Returns:
        normal: n ∈ ℝ^2 (unit vector pointing from ego to obstacle)
        rhs: r_d^ego + r^v (minimum required separation)
    """
    diff = reference_obstacle_pos - reference_ego_center
    dist = np.linalg.norm(diff)
    
    if dist < 1e-6:
        # Ego and obstacle at same position - use arbitrary direction
        normal = np.array([1.0, 0.0])
    else:
        normal = diff / dist
    
    rhs = ego_disc_radius + obstacle_radius
    
    return normal, rhs


def evaluate_collision_constraint(
    normal: np.ndarray,
    obstacle_position: np.ndarray,
    ego_disc_center: np.ndarray,
    min_separation: float
) -> float:
    """
    Evaluate constraint value: g̃ = (r_d + r^v) - n^T(p^v - c_d)
    
    Constraint satisfied iff g̃ <= 0
    
    Returns:
        Constraint value (negative = satisfied, positive = violated)
    """
    separation = normal @ (obstacle_position - ego_disc_center)
    return min_separation - separation
```

### 3.4 Ego Disc Positions

```python
def compute_ego_disc_centers(
    ego_state: EgoState,
    disc_offsets: List[Tuple[float, float]]
) -> List[np.ndarray]:
    """
    Compute disc centers in world frame.
    
    c_d(x^ego) = [x, y]^T + R(θ) @ offset_d
    
    where R(θ) is the 2D rotation matrix.
    
    Args:
        ego_state: Current ego state with position (x, y) and heading θ
        disc_offsets: List of (dx, dy) offsets in body frame
        
    Returns:
        List of disc centers in world frame
    """
    x, y, theta = ego_state.x, ego_state.y, ego_state.theta
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    centers = []
    for dx, dy in disc_offsets:
        offset_body = np.array([dx, dy])
        offset_world = R @ offset_body
        center = np.array([x, y]) + offset_world
        centers.append(center)
    
    return centers
```

---

## 4. Algorithm Implementations

### 4.1 Scenario Sampling

```python
def sample_scenario(
    obstacle_initial_states: Dict[int, np.ndarray],  # v -> x_t^v
    obstacle_classes: Dict[int, ObstacleClass],      # v -> class
    mode_histories: Dict[str, ModeHistory],          # class_id -> history
    mode_weights: Dict[str, Dict[int, float]],       # class_id -> {mode: weight}
    horizon: int,
    prior_type: str = "constant"  # "constant" (C1) or "switching" (C2)
) -> Scenario:
    """
    Sample a single scenario from the adaptive predictive distribution.
    Reference: Algorithm 1
    
    Args:
        obstacle_initial_states: Initial state x_t^v for each obstacle
        obstacle_classes: Class assignment C(v) for each obstacle
        mode_histories: Mode history H_t^C for each class
        mode_weights: Mode weights π_t^C for each class
        horizon: Planning horizon N
        prior_type: "constant" (C1) or "switching" (C2)
        
    Returns:
        Scenario containing trajectories for all obstacles
    """
    trajectories = {}
    
    for v, x0 in obstacle_initial_states.items():
        obs_class = obstacle_classes[v]
        class_id = obs_class.class_id
        weights = mode_weights[class_id]
        
        # Sample mode sequence
        modes = list(weights.keys())
        probs = [weights[m] for m in modes]
        
        if prior_type == "constant":
            # (C1) Sample one mode, repeat for horizon
            m = np.random.choice(modes, p=probs)
            mode_sequence = [m] * horizon
        else:
            # (C2) Sample independently at each step
            mode_sequence = list(np.random.choice(modes, size=horizon, p=probs))
        
        # Propagate trajectory
        states = np.zeros((horizon, len(x0)))
        positions = np.zeros((horizon, 2))
        
        x = x0.copy()
        for k in range(horizon):
            m_k = mode_sequence[k]
            mode_model = obs_class.get_mode(m_k)
            x = mode_model.propagate(x)
            states[k] = x
            positions[k] = x[:2]  # Extract position
        
        trajectories[v] = ObstacleTrajectory(
            obstacle_id=v,
            positions=positions,
            states=states,
            mode_sequence=mode_sequence
        )
    
    return Scenario(scenario_id=-1, obstacle_trajectories=trajectories)


def sample_scenarios(
    num_scenarios: int,
    obstacle_initial_states: Dict[int, np.ndarray],
    obstacle_classes: Dict[int, ObstacleClass],
    mode_histories: Dict[str, ModeHistory],
    mode_weights: Dict[str, Dict[int, float]],
    horizon: int,
    prior_type: str = "constant"
) -> List[Scenario]:
    """
    Sample S scenarios i.i.d. from adaptive distribution.
    Reference: Algorithm 1
    """
    scenarios = []
    for i in range(num_scenarios):
        scenario = sample_scenario(
            obstacle_initial_states,
            obstacle_classes,
            mode_histories,
            mode_weights,
            horizon,
            prior_type
        )
        scenario.scenario_id = i
        scenarios.append(scenario)
    return scenarios
```

### 4.2 Constraint Assembly

```python
@dataclass
class LinearConstraint:
    """
    Linear constraint: a^T x <= b
    For collision avoidance: n^T c_d(x^ego) <= n^T p^v - (r_d + r^v)
    """
    a: np.ndarray      # Constraint normal (in decision variable space)
    b: float           # Right-hand side
    scenario_id: int
    obstacle_id: int
    disc_id: int
    step: int


def assemble_collision_constraints(
    scenarios: List[Scenario],
    reference_trajectory: List[EgoState],  # Reference for linearization
    ego_disc_offsets: List[Tuple[float, float]],
    ego_disc_radii: List[float],
    obstacle_radii: Dict[int, float],
    horizon: int
) -> List[LinearConstraint]:
    """
    Assemble all collision avoidance constraints for the QP.
    Reference: Equation (22)
    
    For each scenario i, obstacle v, disc d, step k:
        n_{d,v,k}^{(i)T} (p_{v,k}^{(i)} - c_d(x_k^ego)) >= r_d + r^v
    
    Rearranged for QP:
        -n^T c_d(x_k^ego) <= -n^T p_{v,k}^{(i)} + (r_d + r^v)
    """
    constraints = []
    
    for scenario in scenarios:
        i = scenario.scenario_id
        
        for k in range(1, horizon + 1):
            # Get reference ego state for linearization
            ref_ego = reference_trajectory[k]
            ref_ego_centers = compute_ego_disc_centers(ref_ego, ego_disc_offsets)
            
            for v, traj in scenario.obstacle_trajectories.items():
                obs_pos = traj.position_at_step(k)
                obs_radius = obstacle_radii[v]
                
                for d, (ref_center, ego_radius) in enumerate(
                    zip(ref_ego_centers, ego_disc_radii)
                ):
                    # Compute linearization normal
                    normal, min_sep = compute_collision_constraint(
                        ego_disc_center=ref_center,
                        ego_disc_radius=ego_radius,
                        obstacle_position=obs_pos,
                        obstacle_radius=obs_radius,
                        reference_ego_center=ref_center,
                        reference_obstacle_pos=obs_pos
                    )
                    
                    # Constraint: n^T (p - c_d) >= min_sep
                    # => -n^T c_d <= -n^T p + min_sep
                    # => -n^T c_d <= b where b = -n^T p + min_sep
                    
                    # Note: c_d depends on x^ego, need to express in terms of decision vars
                    # This is handled by the QP formulation
                    
                    constraints.append(LinearConstraint(
                        a=-normal,  # Will be multiplied by c_d
                        b=-normal @ obs_pos + min_sep,
                        scenario_id=i,
                        obstacle_id=v,
                        disc_id=d,
                        step=k
                    ))
    
    return constraints
```

### 4.3 Geometric Dominance Pruning

```python
def prune_dominated_scenarios(
    scenarios: List[Scenario],
    reference_trajectory: List[EgoState],
    ego_disc_offsets: List[Tuple[float, float]],
    num_clusters: int = 10
) -> List[int]:
    """
    Prune geometrically dominated scenarios.
    Reference: Algorithm 3, Definition 14, Proposition 6
    
    For parallel constraints (same normal), keep only the most restrictive.
    
    Returns:
        List of scenario indices to keep
    """
    from sklearn.cluster import KMeans
    
    keep_indices = set()
    
    # Process each (obstacle, step, disc) tuple separately
    if not scenarios:
        return []
    
    V = len(scenarios[0].obstacle_trajectories)
    N = len(list(scenarios[0].obstacle_trajectories.values())[0].positions)
    D = len(ego_disc_offsets)
    
    for v in scenarios[0].obstacle_trajectories.keys():
        for k in range(1, N + 1):
            for d in range(D):
                # Collect normals and constraint values for this (v, k, d)
                normals = []
                b_values = []
                scenario_indices = []
                
                ref_ego = reference_trajectory[k]
                ref_center = compute_ego_disc_centers(ref_ego, ego_disc_offsets)[d]
                
                for scenario in scenarios:
                    obs_pos = scenario.obstacle_trajectories[v].position_at_step(k)
                    diff = obs_pos - ref_center
                    dist = np.linalg.norm(diff)
                    if dist > 1e-6:
                        normal = diff / dist
                    else:
                        normal = np.array([1.0, 0.0])
                    
                    # b = n^T p (larger b = more restrictive)
                    b = normal @ obs_pos
                    
                    normals.append(normal)
                    b_values.append(b)
                    scenario_indices.append(scenario.scenario_id)
                
                normals = np.array(normals)
                b_values = np.array(b_values)
                
                # Cluster normals by direction
                if len(normals) <= num_clusters:
                    # Keep all
                    keep_indices.update(scenario_indices)
                else:
                    # Cluster on unit circle (use angles)
                    angles = np.arctan2(normals[:, 1], normals[:, 0]).reshape(-1, 1)
                    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
                    labels = kmeans.fit_predict(angles)
                    
                    # For each cluster, keep most restrictive (smallest b for <= constraint)
                    for cluster_id in range(num_clusters):
                        mask = labels == cluster_id
                        if not np.any(mask):
                            continue
                        
                        cluster_b = b_values[mask]
                        cluster_indices = np.array(scenario_indices)[mask]
                        
                        # Most restrictive = largest n^T p (for >= constraint)
                        best_idx = cluster_indices[np.argmax(cluster_b)]
                        keep_indices.add(best_idx)
    
    return sorted(keep_indices)
```

### 4.4 Support-Based Scenario Removal

```python
def identify_support_constraints(
    solution_trajectory: List[np.ndarray],  # Optimal ego trajectory
    scenarios: List[Scenario],
    ego_disc_offsets: List[Tuple[float, float]],
    ego_disc_radii: List[float],
    obstacle_radii: Dict[int, float],
    slack_threshold: float = 1e-4
) -> List[Tuple[int, int, int, int, float]]:
    """
    Identify support (active) constraints at the optimal solution.
    Reference: Definition 15, Algorithm 4
    
    A constraint is active if its slack is below threshold.
    
    Returns:
        List of (scenario_id, obstacle_id, disc_id, step, slack) for active constraints
    """
    active_constraints = []
    
    for scenario in scenarios:
        i = scenario.scenario_id
        
        for k, ego_state in enumerate(solution_trajectory[1:], start=1):
            ego_centers = compute_ego_disc_centers(
                EgoState(*ego_state[:4]),  # Assuming state vector format
                ego_disc_offsets
            )
            
            for v, traj in scenario.obstacle_trajectories.items():
                obs_pos = traj.position_at_step(k)
                obs_radius = obstacle_radii[v]
                
                for d, (center, ego_radius) in enumerate(zip(ego_centers, ego_disc_radii)):
                    # Compute constraint slack
                    separation = np.linalg.norm(obs_pos - center)
                    required = ego_radius + obs_radius
                    slack = separation - required
                    
                    if slack < slack_threshold:
                        active_constraints.append((i, v, d, k, slack))
    
    return active_constraints


def remove_scenarios_by_support(
    scenarios: List[Scenario],
    solution_trajectory: List[np.ndarray],
    ego_disc_offsets: List[Tuple[float, float]],
    ego_disc_radii: List[float],
    obstacle_radii: Dict[int, float],
    num_removal: int
) -> Tuple[List[Scenario], List[int]]:
    """
    Remove most constraining scenarios based on support analysis.
    Reference: Algorithm 4, Theorem 2
    
    Returns:
        (remaining_scenarios, removed_scenario_ids)
    """
    if num_removal <= 0:
        return scenarios, []
    
    # Identify active constraints
    active = identify_support_constraints(
        solution_trajectory,
        scenarios,
        ego_disc_offsets,
        ego_disc_radii,
        obstacle_radii
    )
    
    # Count active constraints per scenario
    scenario_activity = {}
    scenario_min_slack = {}
    
    for (i, v, d, k, slack) in active:
        if i not in scenario_activity:
            scenario_activity[i] = 0
            scenario_min_slack[i] = float('inf')
        scenario_activity[i] += 1
        scenario_min_slack[i] = min(scenario_min_slack[i], slack)
    
    # Sort scenarios by: (1) number of active constraints, (2) minimum slack
    # Remove those with most active constraints and smallest slack
    scenario_scores = [
        (i, scenario_activity.get(i, 0), scenario_min_slack.get(i, float('inf')))
        for i, _ in enumerate(scenarios)
    ]
    scenario_scores.sort(key=lambda x: (-x[1], x[2]))  # Most active, smallest slack first
    
    # Remove top num_removal scenarios
    removed_ids = [s[0] for s in scenario_scores[:num_removal]]
    remaining = [s for s in scenarios if s.scenario_id not in removed_ids]
    
    return remaining, removed_ids
```

### 4.5 Main MPC Loop

```python
class AdaptiveScenarioMPC:
    """
    Main adaptive scenario-based MPC controller.
    Reference: Algorithm 2
    """
    
    def __init__(self, config: ScenarioMPCConfig, obstacle_classes: Dict[str, ObstacleClass]):
        self.config = config
        self.obstacle_classes = obstacle_classes
        
        # Initialize mode histories for each class
        self.mode_histories: Dict[str, ModeHistory] = {
            class_id: ModeHistory(class_id=class_id, observed_modes=set(),
                                  observation_times={}, observation_counts={})
            for class_id in obstacle_classes.keys()
        }
        
        # Compute required sample size
        self.num_scenarios = compute_required_samples(
            epsilon=config.epsilon,
            beta=config.beta,
            n_x=4,  # Assuming 4D ego state
            n_u=2,  # Assuming 2D input
            horizon=config.horizon,
            num_removal=config.num_removal
        )
        
        self.current_time = 0
    
    def update_mode_histories(
        self, 
        obstacle_modes: Dict[int, int],  # v -> observed mode
        obstacle_class_assignments: Dict[int, str]  # v -> class_id
    ):
        """
        Update mode histories with current observations.
        Reference: Line 7-8 of Algorithm 2, Equation (7)
        
        H_t^C = H_{t-1}^C ∪ {m_t^v : C(v) = C}
        """
        for v, mode in obstacle_modes.items():
            class_id = obstacle_class_assignments[v]
            self.mode_histories[class_id].update(mode, self.current_time)
    
    def get_mode_weights(self, weight_type: str = "uniform") -> Dict[str, Dict[int, float]]:
        """
        Compute mode weights for each class.
        Reference: Section IV-E
        """
        weights = {}
        for class_id, history in self.mode_histories.items():
            if weight_type == "uniform":
                weights[class_id] = history.get_uniform_weights()
            elif weight_type == "recency":
                weights[class_id] = history.get_recency_weights(
                    self.current_time, 
                    self.config.recency_decay
                )
            elif weight_type == "frequency":
                weights[class_id] = history.get_frequency_weights(
                    self.config.frequency_smoothing
                )
            else:
                raise ValueError(f"Unknown weight type: {weight_type}")
        return weights
    
    def solve(
        self,
        ego_state: EgoState,
        obstacle_states: Dict[int, ObstacleState],
        obstacle_modes: Dict[int, int],
        obstacle_class_assignments: Dict[int, str],
        obstacle_radii: Dict[int, float],
        reference_trajectory: List[EgoState] = None
    ) -> Tuple[List[EgoInput], List[EgoState]]:
        """
        Solve one MPC iteration.
        Reference: Algorithm 2, Lines 3-14
        
        Returns:
            (optimal_inputs, optimal_trajectory)
        """
        # Step 1: Update mode histories
        self.update_mode_histories(obstacle_modes, obstacle_class_assignments)
        
        # Step 2: Compute mode weights
        mode_weights = self.get_mode_weights(weight_type="uniform")
        
        # Step 3: Check if any class has empty history
        for class_id, history in self.mode_histories.items():
            if len(history.observed_modes) == 0:
                raise RuntimeError(f"No modes observed for class {class_id}")
        
        # Step 4: Prepare obstacle initial states
        obstacle_initial_states = {
            v: obs.to_vector() for v, obs in obstacle_states.items()
        }
        
        # Step 5: Prepare obstacle classes lookup
        obstacle_class_lookup = {
            v: self.obstacle_classes[class_id]
            for v, class_id in obstacle_class_assignments.items()
        }
        
        # Step 6: Sample scenarios
        scenarios = sample_scenarios(
            num_scenarios=self.num_scenarios,
            obstacle_initial_states=obstacle_initial_states,
            obstacle_classes=obstacle_class_lookup,
            mode_histories=self.mode_histories,
            mode_weights=mode_weights,
            horizon=self.config.horizon,
            prior_type="constant"
        )
        
        # Step 7: Compute reference trajectory for linearization
        if reference_trajectory is None:
            reference_trajectory = self._compute_nominal_trajectory(ego_state)
        
        # Step 8: Prune dominated scenarios (optional)
        keep_indices = prune_dominated_scenarios(
            scenarios,
            reference_trajectory,
            self.config.ego_disc_offsets,
            num_clusters=self.config.pruning_clusters
        )
        scenarios = [s for s in scenarios if s.scenario_id in keep_indices]
        
        # Step 9: Assemble and solve QP
        optimal_inputs, optimal_trajectory = self._solve_qp(
            ego_state,
            scenarios,
            reference_trajectory,
            obstacle_radii
        )
        
        # Step 10: Update time
        self.current_time += 1
        
        return optimal_inputs, optimal_trajectory
    
    def _compute_nominal_trajectory(self, ego_state: EgoState) -> List[EgoState]:
        """Compute nominal trajectory for linearization (e.g., constant velocity)."""
        trajectory = [ego_state]
        state = ego_state
        for _ in range(self.config.horizon):
            # Simple constant velocity prediction
            next_state = EgoState(
                x=state.x + self.config.dt * state.v * np.cos(state.theta),
                y=state.y + self.config.dt * state.v * np.sin(state.theta),
                theta=state.theta,
                v=state.v
            )
            trajectory.append(next_state)
            state = next_state
        return trajectory
    
    def _solve_qp(
        self,
        ego_state: EgoState,
        scenarios: List[Scenario],
        reference_trajectory: List[EgoState],
        obstacle_radii: Dict[int, float]
    ) -> Tuple[List[EgoInput], List[EgoState]]:
        """
        Formulate and solve the QP.
        Reference: Equation (22)
        
        This is a placeholder - actual implementation depends on QP solver choice
        (e.g., OSQP, qpOASES, CasADi).
        """
        # TODO: Implement QP formulation and solve
        # Return placeholder
        raise NotImplementedError("QP solver not implemented")
```

---

## 5. Numerical Considerations

### 5.1 Covariance Regularization

```python
def regularize_covariance(cov: np.ndarray, min_eigenvalue: float = 1e-6) -> np.ndarray:
    """
    Ensure covariance matrix is positive definite.
    
    Args:
        cov: Covariance matrix
        min_eigenvalue: Minimum allowed eigenvalue
        
    Returns:
        Regularized covariance matrix
    """
    # Symmetrize
    cov = (cov + cov.T) / 2
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Clamp eigenvalues
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    
    # Reconstruct
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
```

### 5.2 Safe Division for Normals

```python
def safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Safely normalize a vector, handling near-zero vectors.
    """
    norm = np.linalg.norm(v)
    if norm < eps:
        # Return arbitrary unit vector
        result = np.zeros_like(v)
        result[0] = 1.0
        return result
    return v / norm
```

### 5.3 Mode Weight Normalization

```python
def normalize_weights(weights: Dict[int, float], eps: float = 1e-10) -> Dict[int, float]:
    """
    Normalize weights to sum to 1, handling numerical issues.
    """
    total = sum(weights.values())
    if total < eps:
        # Uniform fallback
        n = len(weights)
        return {m: 1.0/n for m in weights}
    return {m: w/total for m, w in weights.items()}
```

---

## 6. Testing and Validation

### 6.1 Unit Tests

```python
def test_trajectory_moments():
    """Test closed-form moment computation against Monte Carlo."""
    # Setup
    dt = 0.1
    mode = create_constant_velocity_mode(dt, sigma_v=0.5)
    x0 = np.array([0, 0, 1, 0])  # At origin, moving right
    mode_sequence = (1, 1, 1, 1, 1)  # 5 steps, same mode
    
    # Closed-form
    moments = compute_trajectory_moments(
        x0, mode_sequence, {1: mode}, horizon=5
    )
    
    # Monte Carlo
    num_samples = 10000
    samples = np.zeros((num_samples, 5, 4))
    for i in range(num_samples):
        x = x0.copy()
        for k in range(5):
            x = mode.propagate(x)
            samples[i, k] = x
    
    mc_mean = samples.mean(axis=0)
    mc_cov = np.cov(samples.reshape(num_samples, -1).T)
    
    # Compare
    assert np.allclose(moments.mean_per_step, mc_mean, atol=0.1)
    assert np.allclose(moments.covariance, mc_cov, atol=0.1)


def test_sample_size_formula():
    """Verify sample size computation."""
    # Example from Remark 3
    S = compute_required_samples(
        epsilon=0.05, beta=0.01, n_x=4, n_u=2, horizon=20
    )
    # d = 120, S >= 40 * (4.605 + 120) = 4984.2
    assert S >= 4984


def test_mode_history_update():
    """Test mode history accumulation."""
    history = ModeHistory("test", set(), {}, {})
    
    history.update(mode=1, time=0)
    history.update(mode=2, time=1)
    history.update(mode=1, time=2)
    
    assert history.observed_modes == {1, 2}
    assert history.observation_counts[1] == 2
    assert history.observation_counts[2] == 1
    
    # Recency weights should favor mode 1 (observed at t=2)
    weights = history.get_recency_weights(current_time=2, lambda_decay=0.5)
    assert weights[1] > weights[2]


def test_collision_constraint():
    """Test linearized collision constraint."""
    ego_center = np.array([0, 0])
    obs_pos = np.array([3, 0])
    
    normal, rhs = compute_collision_constraint(
        ego_disc_center=ego_center,
        ego_disc_radius=0.5,
        obstacle_position=obs_pos,
        obstacle_radius=0.5,
        reference_ego_center=ego_center,
        reference_obstacle_pos=obs_pos
    )
    
    # Normal should point from ego to obstacle (positive x direction)
    assert np.allclose(normal, [1, 0])
    assert rhs == 1.0  # 0.5 + 0.5
    
    # Constraint should be satisfied (ego at origin, obstacle at (3,0))
    # n^T (p - c) = 1 * (3 - 0) = 3 >= 1.0 ✓
    value = evaluate_collision_constraint(normal, obs_pos, ego_center, rhs)
    assert value < 0  # Satisfied (negative = feasible)
```

### 6.2 Integration Tests

```python
def test_full_scenario_sampling():
    """Test end-to-end scenario sampling."""
    # Setup obstacle class
    dt = 0.1
    modes = {
        1: create_constant_velocity_mode(dt, sigma_v=0.3),
        2: create_constant_acceleration_mode(dt, ax=0.5, ay=0, sigma_a=0.2)
    }
    obs_class = ObstacleClass("pedestrian", modes)
    
    # Setup mode history (both modes observed)
    history = ModeHistory("pedestrian", {1, 2}, 
                          {1: [0, 2], 2: [1]}, {1: 2, 2: 1})
    
    # Sample scenarios
    scenarios = sample_scenarios(
        num_scenarios=100,
        obstacle_initial_states={0: np.array([5, 0, 0, 1])},
        obstacle_classes={0: obs_class},
        mode_histories={"pedestrian": history},
        mode_weights={"pedestrian": {1: 0.6, 2: 0.4}},
        horizon=10,
        prior_type="constant"
    )
    
    assert len(scenarios) == 100
    
    # Check mode distribution roughly matches weights
    mode_1_count = sum(
        1 for s in scenarios 
        if s.obstacle_trajectories[0].mode_sequence[0] == 1
    )
    assert 40 < mode_1_count < 80  # Should be around 60


def test_constraint_feasibility():
    """Test that initial position satisfies constraints."""
    # Ego at origin, obstacle at (5, 0)
    ego_state = EgoState(x=0, y=0, theta=0, v=1)
    obs_pos = np.array([5, 0])
    
    ego_center = compute_ego_disc_centers(ego_state, [(0, 0)])[0]
    
    normal, rhs = compute_collision_constraint(
        ego_disc_center=ego_center,
        ego_disc_radius=0.5,
        obstacle_position=obs_pos,
        obstacle_radius=0.5,
        reference_ego_center=ego_center,
        reference_obstacle_pos=obs_pos
    )
    
    value = evaluate_collision_constraint(normal, obs_pos, ego_center, rhs)
    assert value < 0, "Initial position should be feasible"
```

---

## Appendix A: Quick Reference

### Key Equations

| Equation | Description | Implementation |
|----------|-------------|----------------|
| (4) | Obstacle dynamics | `ModeModel.propagate()` |
| (7) | Mode history update | `ModeHistory.update()` |
| (10)-(11) | Trajectory moments | `compute_trajectory_moments()` |
| (12) | Recency weights | `ModeHistory.get_recency_weights()` |
| (13) | Frequency weights | `ModeHistory.get_frequency_weights()` |
| (17)-(18) | Linearized constraint | `compute_collision_constraint()` |
| (22) | Scenario QP | `AdaptiveScenarioMPC._solve_qp()` |
| (23) | Sample size bound | `compute_required_samples()` |

### Data Flow

```
Input: ego_state, obstacle_states, obstacle_modes
                    │
                    ▼
        ┌─────────────────────┐
        │ Update Mode History │  ← Eq. (7)
        │     H_t^C ← ...     │
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │ Compute Mode Weights│  ← Eq. (12) or (13)
        │     π_t^C(m)        │
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │  Sample Scenarios   │  ← Algorithm 1
        │   {ξ^{(i)}}_{i=1}^S │
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │ Prune Scenarios     │  ← Algorithm 3
        │ (optional)          │
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │ Assemble Constraints│  ← Eq. (17)-(18)
        │  n^T(p - c) ≥ r     │
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │     Solve QP        │  ← Eq. (22)
        │   min J s.t. ...    │
        └─────────────────────┘
                    │
                    ▼
Output: optimal_inputs, optimal_trajectory
```

### Complexity Summary

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Scenario sampling | O(S·V·N·n_v²) | Matrix-vector products |
| Moment computation | O(N³·n_v³) | Per mode sequence |
| Constraint assembly | O(S·V·D·N) | Linear in all dimensions |
| Dominance pruning | O(S·V·D·N·log(S)) | Clustering + sorting |
| QP solve | O(d³ + d²·n_c) | Interior point method |

Where: S=scenarios, V=obstacles, D=discs, N=horizon, n_v=obstacle state dim, d=decision vars, n_c=constraints.