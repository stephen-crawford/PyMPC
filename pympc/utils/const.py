"""
Constants for MPC framework.
"""

# Module types
CONSTRAINT = "constraint"
OBJECTIVE = "objective"

# Solver types
CASADI_SOLVER = "casadi"
OSQP_SOLVER = "osqp"

# Dynamics types
BICYCLE_MODEL = "bicycle"
KINEMATIC_MODEL = "kinematic"

# Constraint types
CONTOURING_CONSTRAINT = "contouring"
SCENARIO_CONSTRAINT = "scenario"
LINEARIZED_CONSTRAINT = "linearized"
ELLIPSOID_CONSTRAINT = "ellipsoid"
GAUSSIAN_CONSTRAINT = "gaussian"
DECOMPOSITION_CONSTRAINT = "decomposition"

# Objective types
CONTOURING_OBJECTIVE = "contouring"
GOAL_OBJECTIVE = "goal"
VELOCITY_OBJECTIVE = "velocity"
CONTROL_OBJECTIVE = "control"

# Test types
SIMPLE_DEMO = "simple_demo"
CONSTRAINT_DEMO = "constraint_demo"
END_TO_END = "end_to_end"
CUSTOM = "custom"

# Road types
CURVED_ROAD = "curved"
STRAIGHT_ROAD = "straight"
COMPLEX_ROAD = "complex"
HIGHWAY_ROAD = "highway"
