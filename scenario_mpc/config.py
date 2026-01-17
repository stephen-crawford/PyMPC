"""
Configuration for Adaptive Scenario-Based MPC.

Provides a clean interface for all hyperparameters following guide.md.
"""

from dataclasses import dataclass, field
from typing import Optional
from .types import WeightType


@dataclass
class ScenarioMPCConfig:
    """
    Configuration parameters for Adaptive Scenario-Based MPC.

    Attributes:
        # Horizon and timing
        horizon: Prediction horizon N
        dt: Timestep duration [s]

        # Ego vehicle parameters
        ego_radius: Collision radius for ego vehicle [m]
        max_acceleration: Maximum acceleration [m/s^2]
        min_acceleration: Minimum acceleration (braking) [m/s^2]
        max_steering_rate: Maximum steering rate [rad/s]

        # Obstacle parameters
        obstacle_radius: Default obstacle collision radius [m]

        # Scenario sampling (Section 6)
        num_scenarios: Number of scenarios to sample (S)
        confidence_level: Chance constraint confidence (1 - epsilon)
        beta: Risk parameter for sample size computation

        # Mode weights (Section 4)
        weight_type: Strategy for computing mode weights
        recency_decay: Decay factor for recency weighting (lambda)

        # Constraint parameters (Section 7)
        safety_margin: Additional safety margin [m]

        # Solver parameters
        solver_max_iter: Maximum solver iterations
        solver_tolerance: Convergence tolerance

        # Objective weights
        goal_weight: Weight for goal tracking
        velocity_weight: Weight for velocity tracking
        acceleration_weight: Weight for acceleration penalty
        steering_weight: Weight for steering penalty
    """
    # Horizon and timing
    horizon: int = 20
    dt: float = 0.1

    # Ego vehicle parameters
    ego_radius: float = 1.0
    max_acceleration: float = 3.0
    min_acceleration: float = -5.0
    max_steering_rate: float = 0.8

    # Obstacle parameters
    obstacle_radius: float = 0.5

    # Scenario sampling (Section 6 - Theorem 1)
    num_scenarios: int = 10
    confidence_level: float = 0.95  # 1 - epsilon
    beta: float = 0.01  # Risk parameter

    # Mode weights (Section 4)
    weight_type: WeightType = WeightType.FREQUENCY
    recency_decay: float = 0.9  # lambda in Eq. 5

    # Constraint parameters
    safety_margin: float = 0.1

    # Solver parameters
    solver_max_iter: int = 500
    solver_tolerance: float = 1e-4

    # Objective weights
    goal_weight: float = 10.0
    velocity_weight: float = 1.0
    acceleration_weight: float = 0.1
    steering_weight: float = 0.1

    @property
    def epsilon(self) -> float:
        """Violation probability (1 - confidence_level)."""
        return 1.0 - self.confidence_level

    @property
    def combined_radius(self) -> float:
        """Combined ego + obstacle radius for collision checking."""
        return self.ego_radius + self.obstacle_radius

    def compute_required_scenarios(self, num_constraints: int) -> int:
        """
        Compute required number of scenarios using Theorem 1.

        S >= 2/epsilon * (ln(1/beta) + num_constraints)

        Args:
            num_constraints: Number of decision variables (n_x)

        Returns:
            Minimum number of scenarios required
        """
        import numpy as np
        return int(np.ceil(
            2.0 / self.epsilon * (np.log(1.0 / self.beta) + num_constraints)
        ))

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.horizon > 0, "horizon must be positive"
        assert self.dt > 0, "dt must be positive"
        assert 0 < self.confidence_level < 1, "confidence_level must be in (0, 1)"
        assert 0 < self.beta < 1, "beta must be in (0, 1)"
        assert self.num_scenarios > 0, "num_scenarios must be positive"

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ScenarioMPCConfig":
        """Create config from dictionary."""
        # Handle weight_type conversion
        if "weight_type" in config_dict and isinstance(config_dict["weight_type"], str):
            config_dict["weight_type"] = WeightType(config_dict["weight_type"])
        return cls(**config_dict)
