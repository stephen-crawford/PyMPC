"""
Scenario sampling utilities for safe horizon constraints.

Supports adaptive mode-based sampling following guide.md:
- Mode history tracking for each obstacle
- Mode weight computation (uniform, recency, frequency)
- Mode-dependent dynamics for trajectory prediction
"""
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from planning.types import DynamicObstacle, PredictionType, Scenario
from utils.utils import LOG_DEBUG, LOG_WARN, LOG_INFO


# =============================================================================
# Mode-based sampling types (following guide.md Section 2-4)
# =============================================================================

class WeightType(Enum):
    """Mode weight computation strategies."""
    UNIFORM = "uniform"
    RECENCY = "recency"
    FREQUENCY = "frequency"


@dataclass
class ModeModel:
    """
    Mode-dependent dynamics model for obstacle prediction.

    Dynamics: x_{k+1} = A @ x_k + b + G @ w_k
    where w_k ~ N(0, I) is process noise.

    Reference: guide.md Section 3 - Mode and Dynamics Models
    """
    mode_id: str
    A: np.ndarray  # State transition matrix (4x4)
    b: np.ndarray  # Bias/drift vector (4,)
    G: np.ndarray  # Process noise matrix (4xn_noise)
    description: str = ""

    def propagate(self, state: np.ndarray, noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Propagate state one timestep forward.

        Args:
            state: Current state [x, y, vx, vy]
            noise: Process noise sample (default: zero)

        Returns:
            Next state
        """
        x_next = self.A @ state + self.b
        if noise is not None:
            x_next += self.G @ noise
        return x_next

    @property
    def noise_dim(self) -> int:
        """Dimension of process noise."""
        return self.G.shape[1]


@dataclass
class ModeHistory:
    """
    Track observed modes for an obstacle over time.

    Reference: guide.md Section 4 - Mode History and Weights
    """
    obstacle_id: int
    available_modes: Dict[str, ModeModel]
    observed_modes: List[Tuple[int, str]] = field(default_factory=list)
    max_history: int = 100

    def record_observation(self, timestep: int, mode_id: str) -> None:
        """Record a mode observation at the given timestep."""
        if mode_id not in self.available_modes:
            LOG_WARN(f"Unknown mode: {mode_id}, available modes: {list(self.available_modes.keys())}")
            return

        self.observed_modes.append((timestep, mode_id))

        # Trim history if too long
        if len(self.observed_modes) > self.max_history:
            self.observed_modes = self.observed_modes[-self.max_history:]

    def get_observed_mode_set(self) -> Set[str]:
        """Get set of all observed modes."""
        return set(mode_id for _, mode_id in self.observed_modes)

    def get_mode_counts(self) -> Dict[str, int]:
        """Count occurrences of each mode in history."""
        counts = {mode_id: 0 for mode_id in self.available_modes}
        for _, mode_id in self.observed_modes:
            if mode_id in counts:
                counts[mode_id] += 1
        return counts

    def get_recent_modes(self, n: int) -> List[str]:
        """Get the n most recent observed modes."""
        return [mode_id for _, mode_id in self.observed_modes[-n:]]


def create_obstacle_mode_models(dt: float = 0.1) -> Dict[str, ModeModel]:
    """
    Create standard obstacle mode models.

    Reference: guide.md Section 3 - Standard Mode Models

    Args:
        dt: Timestep in seconds

    Returns:
        Dictionary of mode models
    """
    modes = {}

    # Constant velocity mode
    A_cv = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    b_cv = np.zeros(4)
    G_cv = np.array([
        [0.5 * dt**2, 0],
        [0, 0.5 * dt**2],
        [dt, 0],
        [0, dt]
    ]) * 0.5  # Scale process noise

    modes["constant_velocity"] = ModeModel(
        mode_id="constant_velocity",
        A=A_cv, b=b_cv, G=G_cv,
        description="Constant velocity motion"
    )

    # Decelerating mode
    A_dec = A_cv.copy()
    b_dec = np.array([0, 0, -0.5 * dt, -0.5 * dt])  # Deceleration

    modes["decelerating"] = ModeModel(
        mode_id="decelerating",
        A=A_dec, b=b_dec, G=G_cv,
        description="Decelerating motion"
    )

    # Accelerating mode
    b_acc = np.array([0, 0, 0.5 * dt, 0.5 * dt])  # Acceleration

    modes["accelerating"] = ModeModel(
        mode_id="accelerating",
        A=A_cv.copy(), b=b_acc, G=G_cv,
        description="Accelerating motion"
    )

    # Left turn mode
    omega = 0.3  # Turn rate [rad/s]
    cos_w = np.cos(omega * dt)
    sin_w = np.sin(omega * dt)

    A_left = np.array([
        [1, 0, dt * cos_w, -dt * sin_w],
        [0, 1, dt * sin_w, dt * cos_w],
        [0, 0, cos_w, -sin_w],
        [0, 0, sin_w, cos_w]
    ])

    modes["turn_left"] = ModeModel(
        mode_id="turn_left",
        A=A_left, b=np.zeros(4), G=G_cv,
        description="Left turning motion"
    )

    # Right turn mode
    A_right = np.array([
        [1, 0, dt * cos_w, dt * sin_w],
        [0, 1, -dt * sin_w, dt * cos_w],
        [0, 0, cos_w, sin_w],
        [0, 0, -sin_w, cos_w]
    ])

    modes["turn_right"] = ModeModel(
        mode_id="turn_right",
        A=A_right, b=np.zeros(4), G=G_cv,
        description="Right turning motion"
    )

    # Lane change left
    b_lc_left = np.array([0, 0.3 * dt, 0, 0])  # Lateral drift left

    modes["lane_change_left"] = ModeModel(
        mode_id="lane_change_left",
        A=A_cv.copy(), b=b_lc_left, G=G_cv,
        description="Lane change left"
    )

    # Lane change right
    b_lc_right = np.array([0, -0.3 * dt, 0, 0])  # Lateral drift right

    modes["lane_change_right"] = ModeModel(
        mode_id="lane_change_right",
        A=A_cv.copy(), b=b_lc_right, G=G_cv,
        description="Lane change right"
    )

    return modes


def compute_mode_weights(
    mode_history: ModeHistory,
    weight_type: WeightType = WeightType.FREQUENCY,
    recency_decay: float = 0.9,
    current_timestep: int = 0
) -> Dict[str, float]:
    """
    Compute mode weights based on observation history.

    Reference: guide.md Section 4 - Mode History and Weights

    Args:
        mode_history: Observed mode history for an obstacle
        weight_type: Weight computation strategy
        recency_decay: Decay factor lambda for recency weighting
        current_timestep: Current timestep for recency computation

    Returns:
        Dictionary mapping mode_id to weight (normalized to sum to 1)
    """
    observed_modes = mode_history.get_observed_mode_set()

    # If no modes observed, use all available modes with uniform weights
    if not observed_modes:
        modes = list(mode_history.available_modes.keys())
        return {mode_id: 1.0 / len(modes) for mode_id in modes}

    # Only use observed modes for weight computation
    modes = list(observed_modes)
    num_modes = len(modes)

    if weight_type == WeightType.UNIFORM:
        weights = {mode_id: 1.0 / num_modes for mode_id in modes}

    elif weight_type == WeightType.RECENCY:
        # w_m = sum_{t: m_t = m} lambda^(T - t)
        weights = {mode_id: 0.0 for mode_id in modes}
        for timestep, mode_id in mode_history.observed_modes:
            if mode_id in weights:
                age = current_timestep - timestep
                weights[mode_id] += recency_decay ** age

    elif weight_type == WeightType.FREQUENCY:
        # w_m = n_m / sum_j n_j
        counts = mode_history.get_mode_counts()
        weights = {mode_id: float(counts.get(mode_id, 0)) for mode_id in modes}

    else:
        raise ValueError(f"Unknown weight type: {weight_type}")

    # Normalize weights to sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    else:
        # Fallback to uniform
        weights = {mode_id: 1.0 / num_modes for mode_id in modes}

    return weights


def sample_mode_from_weights(
    mode_weights: Dict[str, float],
    rng: Optional[np.random.Generator] = None
) -> str:
    """
    Sample a single mode from the weight distribution.

    Args:
        mode_weights: Weights for each mode
        rng: Random number generator

    Returns:
        Sampled mode_id
    """
    if rng is None:
        rng = np.random.default_rng()

    modes = list(mode_weights.keys())
    weights = np.array([mode_weights[m] for m in modes])
    weights = weights / weights.sum()

    idx = rng.choice(len(modes), p=weights)
    return modes[idx]


class ScenarioSampler:
    """Handles sampling of scenarios from obstacle predictions."""
    
    def __init__(self, num_scenarios: int = 100, enable_outlier_removal: bool = True):
        self.num_scenarios = num_scenarios
        self.enable_outlier_removal = enable_outlier_removal
        self.scenarios = []
        
    def sample_scenarios(self, obstacles: List[DynamicObstacle], 
                        horizon_length: int, timestep: float) -> List[Scenario]:
        """
        Sample scenarios from obstacle predictions.
        
        Args:
            obstacles: List of dynamic obstacles
            horizon_length: MPC horizon length
            timestep: Time step size
            
        Returns:
            List of sampled scenarios
        """
        scenarios = []
        
        for obstacle_idx, obstacle in enumerate(obstacles):
            if obstacle.prediction.type != PredictionType.GAUSSIAN:
                LOG_WARN(f"Obstacle {obstacle_idx} has non-Gaussian prediction, skipping")
                continue
                
            # Sample scenarios for this obstacle
            obstacle_scenarios = self._sample_obstacle_scenarios(
                obstacle, obstacle_idx, horizon_length, timestep
            )
            scenarios.extend(obstacle_scenarios)
        
        self.scenarios = scenarios
        
        # Optional outlier removal
        if self.enable_outlier_removal:
            scenarios = self._remove_outliers(scenarios)
            
        LOG_DEBUG(f"Sampled {len(scenarios)} scenarios from {len(obstacles)} obstacles")
        return scenarios
    
    def _sample_obstacle_scenarios(self, obstacle: DynamicObstacle, obstacle_idx: int,
                                 horizon_length: int, _timestep: float) -> List[Scenario]:
        """Sample scenarios for a single obstacle."""
        scenarios = []
        
        # For Gaussian predictions, sample from multivariate normal
        if obstacle.prediction.type == PredictionType.GAUSSIAN:
            scenarios = self._sample_gaussian_scenarios(
                obstacle, obstacle_idx, horizon_length, _timestep
            )
        
        return scenarios
    
    def _sample_gaussian_scenarios(self, obstacle: DynamicObstacle, obstacle_idx: int,
                                 horizon_length: int, timestep: float) -> List[Scenario]:
        """
        Sample scenarios from Gaussian prediction.
        
        Reference: scenario_module - scenarios represent full obstacle trajectories
        Each scenario is a complete future trajectory sampled from the prediction distribution.
        """
        scenarios = []
        
        # Get prediction steps
        prediction_steps = obstacle.prediction.steps
        if not prediction_steps:
            LOG_WARN(f"No prediction steps for obstacle {obstacle_idx}")
            return scenarios
        
        # Sample num_scenarios complete trajectories (one scenario = one full trajectory)
        # Each scenario represents a possible future for the obstacle over the entire horizon
        for scenario_idx in range(self.num_scenarios):
            scenario = Scenario(scenario_idx, obstacle_idx)
            scenario.radius = obstacle.radius
            
            # Sample a position for each time step in the horizon
            scenario_positions = []
            for step in range(min(horizon_length, len(prediction_steps))):
                pred_step = prediction_steps[step]
                
                # Create covariance matrix from major/minor radii
                # Use proper ellipsoidal covariance if available
                major_radius = float(getattr(pred_step, 'major_radius', 0.5))
                minor_radius = float(getattr(pred_step, 'minor_radius', 0.5))
                angle = float(getattr(pred_step, 'angle', 0.0))
                
                R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                cov_matrix = R @ np.diag([major_radius**2, minor_radius**2]) @ R.T
        
                
                # Sample position for this time step
                mean_pos = np.array([float(pred_step.position[0]), float(pred_step.position[1])])
                sampled_pos = np.random.multivariate_normal(mean_pos, cov_matrix)
                scenario_positions.append(sampled_pos)
            
            # Store the first position as the scenario position (for constraint formulation)
            # The full trajectory is represented by sampling positions at each time step
            # Reference: scenario_module - scenarios store full trajectories for visualization
            if scenario_positions:
                scenario.position = scenario_positions[0]  # Initial position
                scenario.time_step = 0  # Start at time step 0
                # Store full trajectory (Scenario class allows dynamic attributes)
                scenario.trajectory = scenario_positions
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _remove_outliers(self, scenarios: List[Scenario], 
                        outlier_threshold: float = 2.0) -> List[Scenario]:
        """
        Remove outlier scenarios using statistical methods.
        
        Args:
            scenarios: List of scenarios to filter
            outlier_threshold: Z-score threshold for outlier detection
            
        Returns:
            Filtered list of scenarios
        """
        if len(scenarios) < 10:  # Not enough data for outlier detection
            return scenarios
        
        # Group scenarios by obstacle and time step
        grouped_scenarios = {}
        for scenario in scenarios:
            key = (scenario.obstacle_idx_, scenario.time_step)
            if key not in grouped_scenarios:
                grouped_scenarios[key] = []
            grouped_scenarios[key].append(scenario)
        
        filtered_scenarios = []
        
        for key, group_scenarios in grouped_scenarios.items():
            if len(group_scenarios) < 5:  # Not enough for outlier detection
                filtered_scenarios.extend(group_scenarios)
                continue
            
            # Extract positions
            positions = np.array([s.position for s in group_scenarios])
            
            # Compute mean and standard deviation
            mean_pos = np.mean(positions, axis=0)
            std_pos = np.std(positions, axis=0)
            
            # Filter outliers
            for scenario in group_scenarios:
                z_scores = np.abs((scenario.position - mean_pos) / (std_pos + 1e-6))
                max_z_score = np.max(z_scores)
                
                if max_z_score <= outlier_threshold:
                    filtered_scenarios.append(scenario)
        
        removed_count = len(scenarios) - len(filtered_scenarios)
        if removed_count > 0:
            LOG_DEBUG(f"Removed {removed_count} outlier scenarios")
        
        return filtered_scenarios
    
    def integrate_and_translate_to_mean_and_variance(self, obstacles: List[DynamicObstacle], 
                                                   timestep: float):
        """
        Integrate obstacle dynamics and translate to mean and variance representation.
        
        Args:
            obstacles: List of dynamic obstacles
            timestep: Time step size
        """
        for obstacle in obstacles:
            if obstacle.prediction.type != PredictionType.GAUSSIAN:
                continue
                
            # This method would integrate the obstacle dynamics over time
            # and update the prediction steps with mean and variance
            # For now, we'll implement a simplified version
            
            if not obstacle.prediction.steps:
                # Create initial prediction step
                from planning.types import PredictionStep
                initial_step = PredictionStep(
                    position=obstacle.position,
                    angle=obstacle.angle,
                    major_radius=obstacle.radius,
                    minor_radius=obstacle.radius
                )
                obstacle.prediction.steps = [initial_step]
            
            # Update prediction steps with integrated dynamics
            self._integrate_obstacle_dynamics(obstacle, timestep)
    
    def _integrate_obstacle_dynamics(self, obstacle: DynamicObstacle, _timestep: float):
        """Integrate obstacle dynamics over time."""
        # Simplified integration - in practice, this would use the actual
        # obstacle dynamics model
        
        if not obstacle.prediction.steps:
            return
        
        # For now, just propagate uncertainty
        for i, step in enumerate(obstacle.prediction.steps):
            # Increase uncertainty over time (simple model)
            time_factor = 1.0 + i * 0.1
            step.major_radius *= time_factor
            step.minor_radius *= time_factor
    
    def get_sampler(self):
        """Get the sampler instance (for compatibility with existing code)."""
        return self
    
    def reset(self):
        """Reset sampler state."""
        self.scenarios = []


class MonteCarloValidator:
    """Validates collision probability using Monte Carlo methods."""
    
    def __init__(self, num_samples: int = 10000):
        self.num_samples = num_samples
    
    def validate_collision_probability(self, robot_trajectory: List[np.ndarray],
                                    obstacles: List[DynamicObstacle],
                                    robot_radius: float,
                                    target_probability: float = 0.05) -> Tuple[bool, float]:
        """
        Validate collision probability using Monte Carlo simulation.
        
        Args:
            robot_trajectory: Robot trajectory over horizon
            obstacles: List of dynamic obstacles
            robot_radius: Robot radius
            target_probability: Target collision probability threshold
            
        Returns:
            Tuple of (is_safe, actual_probability)
        """
        collision_count = 0
        
        for _ in range(self.num_samples):
            if self._check_collision_sample(robot_trajectory, obstacles, robot_radius):
                collision_count += 1
        
        actual_probability = collision_count / self.num_samples
        is_safe = actual_probability <= target_probability
        
        LOG_DEBUG(f"Monte Carlo validation: P(collision) = {actual_probability:.4f}, "
                 f"target = {target_probability:.4f}, safe = {is_safe}")
        
        return is_safe, actual_probability
    
    def _check_collision_sample(self, robot_trajectory: List[np.ndarray],
                              obstacles: List[DynamicObstacle],
                              robot_radius: float) -> bool:
        """Check collision for a single Monte Carlo sample."""
        for step, robot_pos in enumerate(robot_trajectory):
            for obstacle in obstacles:
                if step >= len(obstacle.prediction.steps):
                    continue
                
                pred_step = obstacle.prediction.steps[step]
                
                # Sample obstacle position from prediction
                sigma = max(pred_step.major_radius, pred_step.minor_radius)
                obstacle_pos = np.random.multivariate_normal(
                    pred_step.position[:2], 
                    np.array([[sigma**2, 0], [0, sigma**2]])
                )
                
                # Check collision
                distance = np.linalg.norm(robot_pos - obstacle_pos)
                if distance < robot_radius + obstacle.radius:
                    return True

        return False


# =============================================================================
# Adaptive Mode Sampler (following guide.md Algorithm 1)
# =============================================================================

class AdaptiveModeSampler(ScenarioSampler):
    """
    Adaptive scenario sampler that uses mode-based trajectory prediction.

    Following guide.md Algorithm 1: SampleScenarios with adaptive mode weights.

    This sampler:
    1. Tracks observed modes for each obstacle
    2. Computes mode weights based on history (uniform, recency, or frequency)
    3. Samples mode sequences for trajectory prediction
    4. Propagates trajectories using mode-dependent dynamics
    """

    def __init__(
        self,
        num_scenarios: int = 100,
        enable_outlier_removal: bool = True,
        weight_type: WeightType = WeightType.FREQUENCY,
        recency_decay: float = 0.9,
        dt: float = 0.1,
        prior_type: str = "constant"  # "constant" (C1) or "switching" (C2)
    ):
        """
        Initialize adaptive mode sampler.

        Args:
            num_scenarios: Number of scenarios to sample per obstacle
            enable_outlier_removal: Whether to remove outlier scenarios
            weight_type: Mode weight computation strategy
            recency_decay: Decay factor for recency weighting
            dt: Timestep for dynamics propagation
            prior_type: "constant" samples one mode per trajectory,
                       "switching" samples mode at each timestep
        """
        super().__init__(num_scenarios, enable_outlier_removal)
        self.weight_type = weight_type
        self.recency_decay = recency_decay
        self.dt = dt
        self.prior_type = prior_type

        # Mode models and histories
        self.mode_models: Dict[str, ModeModel] = create_obstacle_mode_models(dt)
        self.mode_histories: Dict[int, ModeHistory] = {}
        self.current_timestep = 0

        # Random number generator for reproducibility
        self.rng = np.random.default_rng()

        LOG_INFO(f"AdaptiveModeSampler initialized: weight_type={weight_type.value}, "
                f"prior_type={prior_type}, num_modes={len(self.mode_models)}")

    def update_mode_observation(
        self,
        obstacle_id: int,
        observed_mode: str,
        available_modes: Optional[List[str]] = None,
        initial_mode: Optional[str] = None
    ) -> None:
        """
        Update mode history with a new observation.

        Reference: guide.md Eq. 7 - H_t^C = H_{t-1}^C ∪ {m_t^v}

        Args:
            obstacle_id: Obstacle identifier
            observed_mode: The mode observed for this obstacle
            available_modes: List of available modes for this obstacle (optional)
            initial_mode: The initial mode from configuration (recorded on first call)
        """
        is_new_history = obstacle_id not in self.mode_histories

        if is_new_history:
            # Initialize mode history for new obstacle
            if available_modes:
                modes = {m: self.mode_models[m] for m in available_modes if m in self.mode_models}
            else:
                modes = self.mode_models
            self.mode_histories[obstacle_id] = ModeHistory(
                obstacle_id=obstacle_id,
                available_modes=modes
            )
            LOG_INFO(f"Created mode history for obstacle {obstacle_id} with modes: {list(modes.keys())}")

            # Record initial mode first if provided and different from observed
            if initial_mode and initial_mode in self.mode_models and initial_mode != observed_mode:
                self.mode_histories[obstacle_id].record_observation(0, initial_mode)
                LOG_INFO(f"Recorded INITIAL mode for obstacle {obstacle_id}: {initial_mode}")

        self.mode_histories[obstacle_id].record_observation(
            self.current_timestep, observed_mode
        )
        LOG_INFO(f"Recorded mode observation: obstacle {obstacle_id}, mode={observed_mode}, "
                 f"timestep={self.current_timestep}, history_size={len(self.mode_histories[obstacle_id].observed_modes)}")

    def advance_timestep(self) -> None:
        """Advance the current timestep."""
        self.current_timestep += 1

    def sample_scenarios(
        self,
        obstacles: List[DynamicObstacle],
        horizon_length: int,
        timestep: float
    ) -> List[Scenario]:
        """
        Sample scenarios using adaptive mode-based prediction.

        Reference: guide.md Algorithm 1 - SampleScenarios

        Args:
            obstacles: List of dynamic obstacles
            horizon_length: MPC horizon length
            timestep: Time step size

        Returns:
            List of sampled scenarios
        """
        scenarios = []
        self.dt = timestep

        for obstacle_idx, obstacle in enumerate(obstacles):
            # Get or create mode history for this obstacle
            if obstacle_idx not in self.mode_histories:
                self._initialize_mode_history(obstacle, obstacle_idx)

            # Check if obstacle has a current mode observation
            current_mode = getattr(obstacle, 'current_mode', None)
            if current_mode and current_mode in self.mode_models:
                self.update_mode_observation(obstacle_idx, current_mode)

            # Sample scenarios using adaptive mode weights
            obstacle_scenarios = self._sample_adaptive_scenarios(
                obstacle, obstacle_idx, horizon_length
            )
            scenarios.extend(obstacle_scenarios)

        self.scenarios = scenarios

        # Optional outlier removal
        if self.enable_outlier_removal:
            scenarios = self._remove_outliers(scenarios)

        LOG_DEBUG(f"AdaptiveModeSampler: Sampled {len(scenarios)} scenarios "
                 f"from {len(obstacles)} obstacles using {self.weight_type.value} weights")

        return scenarios

    def _initialize_mode_history(
        self,
        obstacle: DynamicObstacle,
        obstacle_idx: int
    ) -> None:
        """Initialize mode history for a new obstacle."""
        # Check if obstacle has available modes defined
        available_modes = getattr(obstacle, 'available_modes', None)
        if available_modes:
            modes = {m: self.mode_models[m] for m in available_modes if m in self.mode_models}
        else:
            # Use all available modes
            modes = self.mode_models

        self.mode_histories[obstacle_idx] = ModeHistory(
            obstacle_id=obstacle_idx,
            available_modes=modes
        )

        # If obstacle has initial mode, record it
        # IMPORTANT: Use initial_mode (static config) over current_mode (dynamic detection)
        initial_mode = getattr(obstacle, 'initial_mode', None)
        current_mode = getattr(obstacle, 'current_mode', None)
        mode_to_record = initial_mode or current_mode

        LOG_INFO(f"Initializing mode history for obstacle {obstacle_idx}: "
                f"initial_mode={initial_mode}, current_mode={current_mode}, "
                f"recording={mode_to_record}")

        if mode_to_record and mode_to_record in self.mode_models:
            self.mode_histories[obstacle_idx].record_observation(
                self.current_timestep, mode_to_record
            )

    def _sample_adaptive_scenarios(
        self,
        obstacle: DynamicObstacle,
        obstacle_idx: int,
        horizon_length: int
    ) -> List[Scenario]:
        """
        Sample scenarios using adaptive mode weights.

        Reference: guide.md Algorithm 1 - For each scenario:
        1. Compute mode weights from history
        2. Sample mode sequence
        3. Sample noise sequence
        4. Propagate trajectory
        """
        scenarios = []
        mode_history = self.mode_histories[obstacle_idx]

        # Compute mode weights based on history
        mode_weights = compute_mode_weights(
            mode_history,
            weight_type=self.weight_type,
            recency_decay=self.recency_decay,
            current_timestep=self.current_timestep
        )

        if not mode_weights:
            LOG_WARN(f"No mode weights computed for obstacle {obstacle_idx}")
            return scenarios

        # Log mode weights at INFO level to verify adaptive sampling is working
        LOG_INFO(f"Obstacle {obstacle_idx} mode weights: {mode_weights}")

        # Count sampled modes for verification
        mode_counts = {}

        # Get initial state from obstacle
        initial_state = self._get_obstacle_state(obstacle)

        # Sample scenarios
        for scenario_idx in range(self.num_scenarios):
            scenario = self._sample_single_scenario(
                obstacle, obstacle_idx, scenario_idx,
                initial_state, mode_weights, horizon_length
            )
            scenarios.append(scenario)
            # Count modes for verification
            if hasattr(scenario, 'mode_id'):
                mode_counts[scenario.mode_id] = mode_counts.get(scenario.mode_id, 0) + 1

        # Log mode distribution in sampled scenarios
        LOG_INFO(f"Obstacle {obstacle_idx} sampled mode distribution: {mode_counts}")

        return scenarios

    def _get_obstacle_state(self, obstacle: DynamicObstacle) -> np.ndarray:
        """Extract state vector from obstacle: [x, y, vx, vy]."""
        position = np.array(obstacle.position[:2])

        # Try to get velocity from obstacle
        velocity = getattr(obstacle, 'velocity', None)
        if velocity is not None:
            velocity = np.array(velocity[:2])
        else:
            # Estimate velocity from prediction steps
            if obstacle.prediction.steps and len(obstacle.prediction.steps) >= 2:
                p0 = np.array(obstacle.prediction.steps[0].position[:2])
                p1 = np.array(obstacle.prediction.steps[1].position[:2])
                velocity = (p1 - p0) / self.dt
            else:
                velocity = np.array([0.0, 0.0])

        return np.concatenate([position, velocity])

    def _sample_single_scenario(
        self,
        obstacle: DynamicObstacle,
        obstacle_idx: int,
        scenario_idx: int,
        initial_state: np.ndarray,
        mode_weights: Dict[str, float],
        horizon_length: int
    ) -> Scenario:
        """
        Sample a single scenario trajectory.

        Args:
            obstacle: Dynamic obstacle
            obstacle_idx: Obstacle index
            scenario_idx: Scenario index
            initial_state: Initial state [x, y, vx, vy]
            mode_weights: Mode weights for sampling
            horizon_length: Prediction horizon

        Returns:
            Sampled scenario with trajectory
        """
        scenario = Scenario(scenario_idx, obstacle_idx)
        scenario.radius = obstacle.radius

        mode_history = self.mode_histories[obstacle_idx]

        if self.prior_type == "constant":
            # (C1) Sample one mode for entire trajectory
            sampled_mode_id = sample_mode_from_weights(mode_weights, self.rng)
            mode = mode_history.available_modes[sampled_mode_id]
            mode_sequence = [sampled_mode_id] * horizon_length
            scenario.mode_id = sampled_mode_id
        else:
            # (C2) Sample mode at each timestep (switching)
            mode_sequence = []
            mode_counts = {}
            for _ in range(horizon_length):
                mode_id = sample_mode_from_weights(mode_weights, self.rng)
                mode_sequence.append(mode_id)
                mode_counts[mode_id] = mode_counts.get(mode_id, 0) + 1
            # Label with dominant mode
            scenario.mode_id = max(mode_counts, key=mode_counts.get)

        # Propagate trajectory using sampled mode sequence
        # CRITICAL: trajectory[0] should be the CURRENT position, trajectory[k] is position at time k
        trajectory_positions = []
        state = initial_state.copy()

        # Include initial position at index 0 (current obstacle position)
        trajectory_positions.append(state[:2].copy())

        for k in range(horizon_length):
            mode_id = mode_sequence[k]
            mode = mode_history.available_modes[mode_id]

            # Sample process noise
            noise = self.rng.standard_normal(mode.noise_dim)

            # Propagate state
            state = mode.propagate(state, noise)
            trajectory_positions.append(state[:2].copy())

        # Store scenario
        # CRITICAL: position should be the CURRENT obstacle position (initial_state), not propagated
        scenario.position = initial_state[:2].copy()
        scenario.time_step = 0
        scenario.trajectory = trajectory_positions
        scenario.mode_sequence = mode_sequence

        return scenario

    def get_mode_weights_for_obstacle(self, obstacle_idx: int) -> Dict[str, float]:
        """Get current mode weights for an obstacle."""
        if obstacle_idx not in self.mode_histories:
            return {}
        return compute_mode_weights(
            self.mode_histories[obstacle_idx],
            weight_type=self.weight_type,
            recency_decay=self.recency_decay,
            current_timestep=self.current_timestep
        )

    def get_observed_modes(self, obstacle_idx: int) -> Set[str]:
        """Get set of observed modes for an obstacle."""
        if obstacle_idx not in self.mode_histories:
            return set()
        return self.mode_histories[obstacle_idx].get_observed_mode_set()

    def reset(self) -> None:
        """Reset sampler state but keep mode histories."""
        super().reset()
        # Note: We intentionally keep mode_histories to maintain learned behavior

    def reset_all(self) -> None:
        """Reset all state including mode histories."""
        super().reset()
        self.mode_histories.clear()
        self.current_timestep = 0
