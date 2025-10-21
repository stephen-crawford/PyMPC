"""
Objective modules for MPC.

This module provides various objective functions for MPC:
- Contouring objectives for path following
- Goal objectives for target reaching
- Path reference velocity objectives for speed control
"""

from .base_objective import BaseObjective
from .contouring_objective import ContouringObjective
from .goal_objective import GoalObjective
from .path_reference_velocity_objective import PathReferenceVelocityObjective

__all__ = [
    'BaseObjective',
    'ContouringObjective',
    'GoalObjective',
    'PathReferenceVelocityObjective'
]
