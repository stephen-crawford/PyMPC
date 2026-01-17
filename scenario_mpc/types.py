"""
Core data structures for Adaptive Scenario-Based MPC.

Following the mathematical formulation in guide.md:
- Section 2: State Representations
- Section 3: Mode and Dynamics Models
- Section 4: Mode History and Weights
- Section 5: Trajectory and Scenario Structures
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np


# =============================================================================
# Section 2: State Representations
# =============================================================================

@dataclass
class EgoState:
    """
    Ego vehicle state: x_ego = (x, y, theta, v)

    Attributes:
        x: Position x-coordinate [m]
        y: Position y-coordinate [m]
        theta: Heading angle [rad]
        v: Velocity magnitude [m/s]
    """
    x: float
    y: float
    theta: float
    v: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, theta, v]."""
        return np.array([self.x, self.y, self.theta, self.v])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "EgoState":
        """Create from numpy array."""
        return cls(x=arr[0], y=arr[1], theta=arr[2], v=arr[3])

    def position(self) -> np.ndarray:
        """Get position as 2D array [x, y]."""
        return np.array([self.x, self.y])


@dataclass
class EgoInput:
    """
    Ego vehicle control input: u = (a, delta)

    Attributes:
        a: Acceleration [m/s^2]
        delta: Steering angle or angular velocity [rad or rad/s]
    """
    a: float
    delta: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [a, delta]."""
        return np.array([self.a, self.delta])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "EgoInput":
        """Create from numpy array."""
        return cls(a=arr[0], delta=arr[1])


@dataclass
class ObstacleState:
    """
    Obstacle state: x_obs = (x, y, vx, vy)

    Attributes:
        x: Position x-coordinate [m]
        y: Position y-coordinate [m]
        vx: Velocity x-component [m/s]
        vy: Velocity y-component [m/s]
    """
    x: float
    y: float
    vx: float
    vy: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, vx, vy]."""
        return np.array([self.x, self.y, self.vx, self.vy])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ObstacleState":
        """Create from numpy array."""
        return cls(x=arr[0], y=arr[1], vx=arr[2], vy=arr[3])

    def position(self) -> np.ndarray:
        """Get position as 2D array [x, y]."""
        return np.array([self.x, self.y])

    def velocity(self) -> np.ndarray:
        """Get velocity as 2D array [vx, vy]."""
        return np.array([self.vx, self.vy])


# =============================================================================
# Section 3: Mode and Dynamics Models
# =============================================================================

@dataclass
class ModeModel:
    """
    Mode-dependent dynamics model for obstacle prediction.

    Dynamics: x_{k+1} = A @ x_k + b + G @ w_k
    where w_k ~ N(0, I) is process noise.

    Attributes:
        mode_id: Unique identifier for this mode (e.g., "lane_keep", "turn_left")
        A: State transition matrix (4x4)
        b: Bias/drift vector (4,)
        G: Process noise matrix (4x2 or 4x4)
        description: Human-readable description of the mode
    """
    mode_id: str
    A: np.ndarray  # (4, 4)
    b: np.ndarray  # (4,)
    G: np.ndarray  # (4, n_noise)
    description: str = ""

    def __post_init__(self):
        """Validate matrix dimensions."""
        assert self.A.shape == (4, 4), f"A must be 4x4, got {self.A.shape}"
        assert self.b.shape == (4,), f"b must be (4,), got {self.b.shape}"
        assert self.G.shape[0] == 4, f"G must have 4 rows, got {self.G.shape}"

    def propagate(self, state: ObstacleState, noise: Optional[np.ndarray] = None) -> ObstacleState:
        """
        Propagate state one timestep forward.

        Args:
            state: Current obstacle state
            noise: Process noise sample (default: zero)

        Returns:
            Next obstacle state
        """
        x = state.to_array()
        x_next = self.A @ x + self.b
        if noise is not None:
            x_next += self.G @ noise
        return ObstacleState.from_array(x_next)

    @property
    def noise_dim(self) -> int:
        """Dimension of process noise."""
        return self.G.shape[1]


# =============================================================================
# Section 4: Mode History and Weights
# =============================================================================

@dataclass
class ModeHistory:
    """
    Track observed modes for an obstacle over time.

    Attributes:
        obstacle_id: Unique obstacle identifier
        observed_modes: List of (timestep, mode_id) tuples
        available_modes: Dict mapping mode_id to ModeModel
        max_history: Maximum history length to maintain
    """
    obstacle_id: int
    available_modes: Dict[str, ModeModel]
    observed_modes: List[Tuple[int, str]] = field(default_factory=list)
    max_history: int = 100

    def record_observation(self, timestep: int, mode_id: str) -> None:
        """
        Record a mode observation at the given timestep.

        Args:
            timestep: Current timestep
            mode_id: Observed mode identifier
        """
        if mode_id not in self.available_modes:
            raise ValueError(f"Unknown mode: {mode_id}")

        self.observed_modes.append((timestep, mode_id))

        # Trim history if too long
        if len(self.observed_modes) > self.max_history:
            self.observed_modes = self.observed_modes[-self.max_history:]

    def get_mode_counts(self) -> Dict[str, int]:
        """Count occurrences of each mode in history."""
        counts = {mode_id: 0 for mode_id in self.available_modes}
        for _, mode_id in self.observed_modes:
            counts[mode_id] += 1
        return counts

    def get_recent_modes(self, n: int) -> List[str]:
        """Get the n most recent observed modes."""
        return [mode_id for _, mode_id in self.observed_modes[-n:]]


# =============================================================================
# Section 5: Trajectory and Scenario Structures
# =============================================================================

@dataclass
class PredictionStep:
    """
    Single step of an obstacle trajectory prediction.

    Attributes:
        k: Timestep index
        mean: Mean position [x, y]
        covariance: Position covariance (2x2)
    """
    k: int
    mean: np.ndarray  # (2,)
    covariance: np.ndarray  # (2, 2)

    def __post_init__(self):
        """Validate dimensions."""
        self.mean = np.asarray(self.mean)
        self.covariance = np.asarray(self.covariance)
        assert self.mean.shape == (2,), f"mean must be (2,), got {self.mean.shape}"
        assert self.covariance.shape == (2, 2), f"covariance must be (2,2), got {self.covariance.shape}"


@dataclass
class ObstacleTrajectory:
    """
    Predicted trajectory for a single obstacle over the horizon.

    Attributes:
        obstacle_id: Unique obstacle identifier
        mode_id: Mode used for this trajectory
        steps: List of prediction steps over horizon
        probability: Probability/weight of this trajectory
    """
    obstacle_id: int
    mode_id: str
    steps: List[PredictionStep]
    probability: float = 1.0

    @property
    def horizon(self) -> int:
        """Prediction horizon length."""
        return len(self.steps)

    def get_mean_at(self, k: int) -> np.ndarray:
        """Get mean position at timestep k."""
        return self.steps[k].mean

    def get_covariance_at(self, k: int) -> np.ndarray:
        """Get position covariance at timestep k."""
        return self.steps[k].covariance


@dataclass
class Scenario:
    """
    A scenario is a collection of obstacle trajectories.

    Represents one possible future realization of all obstacles.

    Attributes:
        scenario_id: Unique scenario identifier
        trajectories: Dict mapping obstacle_id to ObstacleTrajectory
        probability: Combined probability of this scenario
    """
    scenario_id: int
    trajectories: Dict[int, ObstacleTrajectory]
    probability: float = 1.0

    @property
    def num_obstacles(self) -> int:
        """Number of obstacles in this scenario."""
        return len(self.trajectories)

    def get_obstacle_position_at(self, obstacle_id: int, k: int) -> np.ndarray:
        """Get obstacle mean position at timestep k."""
        return self.trajectories[obstacle_id].get_mean_at(k)


@dataclass
class TrajectoryMoments:
    """
    First and second moments of obstacle trajectory distribution.

    Used for efficient Gaussian approximation of multi-modal predictions.

    Attributes:
        obstacle_id: Unique obstacle identifier
        means: Mean positions at each timestep [(N+1, 2)]
        covariances: Position covariances at each timestep [(N+1, 2, 2)]
    """
    obstacle_id: int
    means: np.ndarray  # (N+1, 2)
    covariances: np.ndarray  # (N+1, 2, 2)

    @property
    def horizon(self) -> int:
        """Prediction horizon length."""
        return len(self.means) - 1

    def get_mean_at(self, k: int) -> np.ndarray:
        """Get mean position at timestep k."""
        return self.means[k]

    def get_covariance_at(self, k: int) -> np.ndarray:
        """Get position covariance at timestep k."""
        return self.covariances[k]


# =============================================================================
# Additional Helper Types
# =============================================================================

class WeightType(Enum):
    """Mode weight computation strategies."""
    UNIFORM = "uniform"
    RECENCY = "recency"
    FREQUENCY = "frequency"


@dataclass
class CollisionConstraint:
    """
    Linearized collision avoidance constraint.

    Form: a^T @ p_ego >= b
    where p_ego is ego position.

    Attributes:
        k: Timestep index
        obstacle_id: Obstacle this constraint is for
        scenario_id: Scenario this constraint belongs to
        a: Constraint normal vector (2,)
        b: Constraint offset (scalar)
    """
    k: int
    obstacle_id: int
    scenario_id: int
    a: np.ndarray  # (2,)
    b: float

    def evaluate(self, ego_position: np.ndarray) -> float:
        """
        Evaluate constraint: positive = satisfied, negative = violated.

        Args:
            ego_position: Ego position [x, y]

        Returns:
            Constraint value (a^T @ p - b)
        """
        return np.dot(self.a, ego_position) - self.b


@dataclass
class MPCResult:
    """
    Result from MPC optimization.

    Attributes:
        success: Whether optimization succeeded
        ego_trajectory: Planned ego states over horizon
        control_inputs: Planned control inputs
        active_scenarios: Scenarios that were binding constraints
        solve_time: Optimization solve time [s]
        cost: Optimal cost value
    """
    success: bool
    ego_trajectory: List[EgoState]
    control_inputs: List[EgoInput]
    active_scenarios: List[int] = field(default_factory=list)
    solve_time: float = 0.0
    cost: float = float('inf')

    @property
    def first_input(self) -> Optional[EgoInput]:
        """Get first control input for execution."""
        return self.control_inputs[0] if self.control_inputs else None
