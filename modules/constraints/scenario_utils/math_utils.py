"""
Mathematical utilities for scenario-based constraints.
"""
import numpy as np
from typing import List
from utils.math_tools import Halfspace
from utils.utils import LOG_DEBUG, LOG_WARN


class Polytope:
    """Represents a polytope defined by halfspace constraints."""
    
    def __init__(self, halfspaces: List[Halfspace]):
        self.halfspaces = halfspaces
        self.polygon_out = []  # Output polygon vertices
        
    def add_halfspace(self, halfspace: Halfspace):
        """Add a halfspace constraint to the polytope."""
        self.halfspaces.append(halfspace)
        
    def is_empty(self) -> bool:
        """Check if the polytope is empty (no feasible region)."""
        return len(self.halfspaces) == 0
        
    def get_constraint_count(self) -> int:
        """Get the number of halfspace constraints."""
        return len(self.halfspaces)


class ScenarioConstraint:
    """Represents a single scenario constraint.
    
    Stores pre-computed constraint parameters (a1, a2, b) from polytope optimization.
    These are applied symbolically in calculate_constraints() using the predicted robot position.
    
    Reference: mpc_planner - constraints are pre-computed from scenarios and applied symbolically.
    """
    
    def __init__(self, a1: float, a2: float, b: float, scenario_idx: int, obstacle_idx: int, time_step: int,
                 obstacle_pos: np.ndarray = None, obstacle_radius: float = None):
        self.a1 = a1
        self.a2 = a2
        self.b = b
        self.scenario_idx = scenario_idx
        self.obstacle_idx = obstacle_idx
        self.time_step = time_step
        # Store obstacle info for visualization (optional)
        self.obstacle_pos = obstacle_pos
        self.obstacle_radius = obstacle_radius
        
    def to_halfspace(self) -> Halfspace:
        """Convert to halfspace representation."""
        A = np.array([[self.a1, self.a2]])
        b = np.array([self.b])
        return Halfspace(A, b)


def compute_sample_size(epsilon_p: float, beta: float, n_bar: int) -> int:
    """
    Compute the required sample size for scenario optimization.
    
    Args:
        epsilon_p: Probability of constraint violation
        beta: Confidence level (1 - beta is the confidence)
        n_bar: Support dimension
        
    Returns:
        Required sample size
    """
    if epsilon_p <= 0 or epsilon_p >= 1:
        LOG_WARN(f"Invalid epsilon_p: {epsilon_p}, using default 0.1")
        epsilon_p = 0.1
        
    if beta <= 0 or beta >= 1:
        LOG_WARN(f"Invalid beta: {beta}, using default 0.01")
        beta = 0.01
        
    # Sample size formula for scenario optimization
    # n >= (2/epsilon_p) * ln(1/beta) + 2*n_bar + (2*n_bar/epsilon_p) * ln(2/epsilon_p)
    n = int(np.ceil(
        (2.0 / epsilon_p) * np.log(1.0 / beta) + 
        2.0 * n_bar + 
        (2.0 * n_bar / epsilon_p) * np.log(2.0 / epsilon_p)
    ))
    
    LOG_DEBUG(f"Computed sample size: {n} (epsilon_p={epsilon_p}, beta={beta}, n_bar={n_bar})")
    return max(n, 10)  # Minimum sample size


def linearize_collision_constraint(robot_pos: np.ndarray, obstacle_pos: np.ndarray, 
                                 robot_radius: float, obstacle_radius: float) -> ScenarioConstraint:
    """
    Linearize collision constraint between robot and obstacle.
    
    Args:
        robot_pos: Robot position [x, y]
        obstacle_pos: Obstacle position [x, y]
        robot_radius: Robot radius
        obstacle_radius: Obstacle radius
        
    Returns:
        Linearized constraint
    """
    # Distance between centers
    dx = obstacle_pos[0] - robot_pos[0]
    dy = obstacle_pos[1] - robot_pos[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    if distance < 1e-6:
        # Avoid division by zero
        dx, dy = 1.0, 0.0
        distance = 1.0
    
    # Normalize direction vector
    nx = dx / distance
    ny = dy / distance
    
    # Safety margin
    safety_margin = robot_radius + obstacle_radius
    
    # Linear constraint: nx * (x - robot_x) + ny * (y - robot_y) >= safety_margin
    # Rearranged: nx * x + ny * y >= safety_margin + nx * robot_x + ny * robot_y
    a1 = nx
    a2 = ny
    b = safety_margin + nx * robot_pos[0] + ny * robot_pos[1]
    
    return ScenarioConstraint(a1, a2, b, 0, 0, 0)


def construct_free_space_polytope(scenarios: List[ScenarioConstraint]) -> Polytope:
    """
    Construct free-space polytope from scenario constraints.
    
    Args:
        scenarios: List of scenario constraints
        
    Returns:
        Polytope representing free space
    """
    halfspaces = []
    
    for scenario in scenarios:
        # Convert scenario constraint to halfspace
        # Original: a1*x + a2*y <= b
        # We want: a1*x + a2*y <= b (keep as is for free space)
        A = np.array([[scenario.a1, scenario.a2]])
        b = np.array([scenario.b])
        halfspace = Halfspace(A, b)
        halfspaces.append(halfspace)
    
    return Polytope(halfspaces)


def check_constraint_violation(position: np.ndarray, constraint: ScenarioConstraint) -> bool:
    """
    Check if a position violates a constraint.
    
    Args:
        position: Position to check [x, y]
        constraint: Constraint to check against
        
    Returns:
        True if constraint is violated
    """
    value = constraint.a1 * position[0] + constraint.a2 * position[1]
    return value > constraint.b


def compute_constraint_gradient(constraint: ScenarioConstraint) -> np.ndarray:
    """
    Compute the gradient of a constraint.
    
    Args:
        constraint: Constraint to compute gradient for
        
    Returns:
        Gradient vector [a1, a2]
    """
    return np.array([constraint.a1, constraint.a2])


def validate_polytope_feasibility(polytope: Polytope, test_points: List[np.ndarray]) -> bool:
    """
    Validate polytope feasibility by testing sample points.
    
    Args:
        polytope: Polytope to validate
        test_points: Points to test
        
    Returns:
        True if polytope appears feasible
    """
    if polytope.is_empty():
        return False
        
    feasible_count = 0
    for point in test_points:
        feasible = True
        for halfspace in polytope.halfspaces:
            if not halfspace.contains(point):
                feasible = False
                break
        if feasible:
            feasible_count += 1
    
    # Consider feasible if at least some points satisfy constraints
    return feasible_count > 0


def compute_polytope_volume_estimate(polytope: Polytope, num_samples: int = 1000) -> float:
    """
    Estimate polytope volume using Monte Carlo sampling.
    
    Args:
        polytope: Polytope to estimate volume for
        num_samples: Number of Monte Carlo samples
        
    Returns:
        Estimated volume
    """
    if polytope.is_empty():
        return 0.0
        
    # Define bounding box for sampling
    # This is a simplified approach - in practice, you'd want a tighter bound
    min_x, max_x = -10.0, 10.0
    min_y, max_y = -10.0, 10.0
    
    feasible_count = 0
    for _ in range(num_samples):
        # Sample random point
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        point = np.array([x, y])
        
        # Check if point is feasible
        feasible = True
        for halfspace in polytope.halfspaces:
            if not halfspace.contains(point):
                feasible = False
                break
                
        if feasible:
            feasible_count += 1
    
    # Estimate volume
    bounding_area = (max_x - min_x) * (max_y - min_y)
    volume_estimate = (feasible_count / num_samples) * bounding_area
    
    return volume_estimate
