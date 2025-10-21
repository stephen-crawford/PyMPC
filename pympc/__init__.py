"""
PyMPC: Python Model Predictive Control Framework

A modular and extensible Python port of the TUD-AMR MPC Planner,
designed for real-time motion planning under constraints.

This package provides:
- Model Predictive Control (MPC) framework
- Modular constraint and objective system
- Multiple solver backends (CasADi, OSQP)
- Scenario-based robust MPC
- Contouring control for path following
"""

__version__ = "1.0.0"
__author__ = "Stephen Crawford"

# Core imports
from .core.planner import MPCCPlanner
from .core.solver import BaseSolver, CasADiSolver
from .core.dynamics import BaseDynamics, BicycleModel

# Constraint imports
from .constraints import (
    BaseConstraint,
    ContouringConstraints,
    ScenarioConstraints,
    LinearizedConstraints,
    EllipsoidConstraints,
    GaussianConstraints,
    DecompositionConstraints,
    GuidanceConstraints
)

# Objective imports
from .objectives import (
    BaseObjective,
    ContouringObjective,
    GoalObjective,
    PathReferenceVelocityObjective
)

# Utility imports
from .utils.visualization import StandardizedVisualizer
from .utils.logging import get_logger

__all__ = [
    # Core
    'MPCCPlanner',
    'BaseSolver',
    'CasADiSolver',
    'BaseDynamics',
    'BicycleModel',
    
    # Constraints
    'BaseConstraint',
    'ContouringConstraints',
    'ScenarioConstraints',
    'LinearizedConstraints',
    'EllipsoidConstraints',
    'GaussianConstraints',
    'DecompositionConstraints',
    'GuidanceConstraints',
    
    # Objectives
    'BaseObjective',
    'ContouringObjective',
    'GoalObjective',
    'PathReferenceVelocityObjective',
    
    # Utils
    'StandardizedVisualizer',
    'get_logger'
]
