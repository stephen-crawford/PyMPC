"""
Constraint modules for MPC.

This module provides various constraint types for MPC:
- Contouring constraints for path following
- Scenario constraints for robust MPC
- Linearized constraints for obstacle avoidance
- Ellipsoid constraints for smooth obstacle representation
- Gaussian constraints for probabilistic obstacle modeling
- Decomposition constraints for road boundaries
- Guidance constraints for multi-homotopy planning
"""

from .base_constraint import BaseConstraint
from .contouring_constraints import ContouringConstraints
from .scenario_constraints import ScenarioConstraints
from .linearized_constraints import LinearizedConstraints
from .ellipsoid_constraints import EllipsoidConstraints
from .gaussian_constraints import GaussianConstraints
from .decomposition_constraints import DecompositionConstraints
from .guidance_constraints import GuidanceConstraints

__all__ = [
    'BaseConstraint',
    'ContouringConstraints',
    'ScenarioConstraints',
    'LinearizedConstraints',
    'EllipsoidConstraints',
    'GaussianConstraints',
    'DecompositionConstraints',
    'GuidanceConstraints'
]
