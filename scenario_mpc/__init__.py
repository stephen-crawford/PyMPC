"""
Adaptive Scenario-Based MPC for Motion Planning

This module implements the mathematical formulation from guide.md:
- Mode-dependent obstacle dynamics
- Adaptive mode weight computation
- Scenario sampling with chance constraints
- Linearized collision avoidance
"""

from .types import (
    EgoState,
    EgoInput,
    ObstacleState,
    ModeModel,
    ModeHistory,
    PredictionStep,
    ObstacleTrajectory,
    Scenario,
    TrajectoryMoments,
    CollisionConstraint,
    MPCResult,
    WeightType,
)
from .config import ScenarioMPCConfig
from .dynamics import EgoDynamics
from .mode_weights import compute_mode_weights
from .trajectory_moments import compute_trajectory_moments
from .scenario_sampler import sample_scenarios
from .collision_constraints import compute_linearized_constraints
from .scenario_pruning import prune_dominated_scenarios, remove_inactive_scenarios
from .mpc_controller import AdaptiveScenarioMPC

__all__ = [
    # Types
    "EgoState",
    "EgoInput",
    "ObstacleState",
    "ModeModel",
    "ModeHistory",
    "PredictionStep",
    "ObstacleTrajectory",
    "Scenario",
    "TrajectoryMoments",
    "CollisionConstraint",
    "MPCResult",
    "WeightType",
    # Config
    "ScenarioMPCConfig",
    # Core functions
    "EgoDynamics",
    "compute_mode_weights",
    "compute_trajectory_moments",
    "sample_scenarios",
    "compute_linearized_constraints",
    "prune_dominated_scenarios",
    "remove_inactive_scenarios",
    # Controller
    "AdaptiveScenarioMPC",
]
