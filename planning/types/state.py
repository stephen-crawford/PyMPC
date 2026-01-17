"""
State and Disc classes - re-exported from types_impl for backward compatibility.
"""

from planning.types_impl import State, Disc, define_robot_area

# Alias for backward compatibility
ScenarioDisc = Disc

__all__ = ['State', 'Disc', 'ScenarioDisc', 'define_robot_area']
