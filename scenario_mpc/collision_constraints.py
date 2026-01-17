"""
Linearized collision avoidance constraints.

Implements Section 7 of guide.md: Linearized Collision Constraints

The key equations are:
- Eq. 17: Direction vector computation
- Eq. 18: Linearized constraint formulation

Constraints are linearized around a reference trajectory to enable QP/SQP solving.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from .types import (
    EgoState,
    Scenario,
    CollisionConstraint,
    PredictionStep,
)


def compute_linearized_constraints(
    reference_trajectory: List[EgoState],
    scenarios: List[Scenario],
    ego_radius: float,
    obstacle_radius: float,
    safety_margin: float = 0.0,
    num_discs: int = 1
) -> List[CollisionConstraint]:
    """
    Compute linearized collision constraints for all scenarios.

    Following Section 7 of guide.md:

    For each scenario s and obstacle o at timestep k:
    1. Compute ego disc positions p_ego (Eq. 16)
    2. Compute direction from obstacle to ego (Eq. 17)
    3. Formulate linearized constraint (Eq. 18)

    Args:
        reference_trajectory: Reference ego trajectory for linearization
        scenarios: List of sampled scenarios
        ego_radius: Ego vehicle collision radius
        obstacle_radius: Obstacle collision radius
        safety_margin: Additional safety margin
        num_discs: Number of discs to represent ego vehicle

    Returns:
        List of CollisionConstraint objects
    """
    constraints = []
    combined_radius = ego_radius + obstacle_radius + safety_margin

    for scenario in scenarios:
        scenario_constraints = _compute_scenario_constraints(
            reference_trajectory,
            scenario,
            combined_radius,
            num_discs
        )
        constraints.extend(scenario_constraints)

    return constraints


def _compute_scenario_constraints(
    reference_trajectory: List[EgoState],
    scenario: Scenario,
    combined_radius: float,
    num_discs: int
) -> List[CollisionConstraint]:
    """
    Compute linearized constraints for a single scenario.
    """
    constraints = []
    horizon = len(reference_trajectory) - 1

    for k in range(horizon + 1):
        ref_state = reference_trajectory[k]

        # Compute ego disc positions (Eq. 16)
        disc_positions = compute_ego_disc_positions(ref_state, num_discs)

        for obs_id, trajectory in scenario.trajectories.items():
            if k >= len(trajectory.steps):
                continue

            obs_step = trajectory.steps[k]
            obs_position = obs_step.mean

            # For each disc, compute constraint
            for disc_pos in disc_positions:
                constraint = _compute_single_constraint(
                    k,
                    obs_id,
                    scenario.scenario_id,
                    disc_pos,
                    obs_position,
                    combined_radius
                )
                if constraint is not None:
                    constraints.append(constraint)

    return constraints


def _compute_single_constraint(
    k: int,
    obstacle_id: int,
    scenario_id: int,
    ego_position: np.ndarray,
    obstacle_position: np.ndarray,
    combined_radius: float
) -> Optional[CollisionConstraint]:
    """
    Compute a single linearized collision constraint.

    Following Eq. 17 and Eq. 18 from guide.md:

    Eq. 17: a = (p_ego - p_obs) / ||p_ego - p_obs||
    Eq. 18: a^T @ p_ego >= a^T @ p_obs + r_combined

    Args:
        k: Timestep index
        obstacle_id: Obstacle identifier
        scenario_id: Scenario identifier
        ego_position: Reference ego position [x, y]
        obstacle_position: Obstacle position [x, y]
        combined_radius: Combined collision radius

    Returns:
        CollisionConstraint or None if constraint is degenerate
    """
    # Compute direction vector (Eq. 17)
    diff = ego_position - obstacle_position
    dist = np.linalg.norm(diff)

    # Handle degenerate case
    if dist < 1e-6:
        # Default direction if positions coincide
        a = np.array([1.0, 0.0])
    else:
        a = diff / dist

    # Compute constraint offset (Eq. 18)
    # a^T @ p_ego >= a^T @ p_obs + r_combined
    # Rearranged: a^T @ p_ego >= b
    # where b = a^T @ p_obs + r_combined
    b = np.dot(a, obstacle_position) + combined_radius

    return CollisionConstraint(
        k=k,
        obstacle_id=obstacle_id,
        scenario_id=scenario_id,
        a=a,
        b=b
    )


def compute_ego_disc_positions(
    state: EgoState,
    num_discs: int = 1,
    vehicle_length: float = 4.0
) -> List[np.ndarray]:
    """
    Compute ego disc positions for collision checking.

    Following Eq. 16 from guide.md:
    For multi-disc representation, discs are placed along the vehicle centerline.

    Args:
        state: Ego vehicle state
        num_discs: Number of discs
        vehicle_length: Vehicle length for disc placement

    Returns:
        List of 2D positions for each disc
    """
    if num_discs == 1:
        return [state.position()]

    positions = []
    center = state.position()
    theta = state.theta

    # Direction vector
    direction = np.array([np.cos(theta), np.sin(theta)])

    # Place discs evenly along vehicle
    if num_discs > 1:
        offsets = np.linspace(-vehicle_length / 2, vehicle_length / 2, num_discs)
    else:
        offsets = [0.0]

    for offset in offsets:
        pos = center + offset * direction
        positions.append(pos)

    return positions


def evaluate_constraint_violation(
    constraints: List[CollisionConstraint],
    ego_trajectory: List[EgoState]
) -> Tuple[float, List[CollisionConstraint]]:
    """
    Evaluate constraint violations for a trajectory.

    Args:
        constraints: List of collision constraints
        ego_trajectory: Ego trajectory to evaluate

    Returns:
        (max_violation, violated_constraints)
        where max_violation > 0 means constraint is violated
    """
    max_violation = 0.0
    violated = []

    for constraint in constraints:
        k = constraint.k
        if k >= len(ego_trajectory):
            continue

        ego_pos = ego_trajectory[k].position()
        value = constraint.evaluate(ego_pos)

        if value < 0:
            violation = -value
            if violation > max_violation:
                max_violation = violation
            violated.append(constraint)

    return max_violation, violated


def compute_constraint_jacobian(
    constraint: CollisionConstraint,
    state: EgoState,
    dynamics_jacobian: np.ndarray
) -> np.ndarray:
    """
    Compute Jacobian of constraint w.r.t. decision variables.

    For constraint: a^T @ p_ego >= b
    The Jacobian w.r.t. position is simply a^T.

    For full state, we need to chain with dynamics Jacobian.

    Args:
        constraint: The collision constraint
        state: Current ego state
        dynamics_jacobian: Jacobian of position w.r.t. state

    Returns:
        Constraint Jacobian (1, state_dim)
    """
    # For position-only constraint, Jacobian is simply [a1, a2, 0, 0]
    # assuming state = [x, y, theta, v]
    jac = np.zeros(4)
    jac[0] = constraint.a[0]
    jac[1] = constraint.a[1]
    return jac


def filter_constraints_by_distance(
    constraints: List[CollisionConstraint],
    ego_trajectory: List[EgoState],
    max_distance: float = 50.0
) -> List[CollisionConstraint]:
    """
    Filter out constraints for obstacles that are too far away.

    Args:
        constraints: All collision constraints
        ego_trajectory: Reference ego trajectory
        max_distance: Maximum distance to consider

    Returns:
        Filtered list of constraints
    """
    filtered = []

    for constraint in constraints:
        k = constraint.k
        if k >= len(ego_trajectory):
            continue

        ego_pos = ego_trajectory[k].position()
        # Reconstruct approximate obstacle position from constraint
        # p_obs ≈ (b - r) * a (rough estimate)
        # This is approximate since we don't store obstacle position directly

        # Better: keep constraint if the constraint value is within reasonable range
        value = constraint.evaluate(ego_pos)
        if value < max_distance:
            filtered.append(constraint)

    return filtered


def merge_redundant_constraints(
    constraints: List[CollisionConstraint],
    angle_threshold: float = 0.1,
    offset_threshold: float = 0.5
) -> List[CollisionConstraint]:
    """
    Merge nearly identical constraints to reduce problem size.

    Two constraints are merged if their normal vectors are nearly parallel
    and their offsets are similar.

    Args:
        constraints: List of constraints
        angle_threshold: Maximum angle difference (radians) for merging
        offset_threshold: Maximum offset difference for merging

    Returns:
        Reduced list of constraints
    """
    if not constraints:
        return []

    # Group constraints by timestep
    by_timestep: Dict[int, List[CollisionConstraint]] = {}
    for c in constraints:
        if c.k not in by_timestep:
            by_timestep[c.k] = []
        by_timestep[c.k].append(c)

    merged = []

    for k, k_constraints in by_timestep.items():
        # Simple approach: keep the most conservative constraint for similar directions
        kept = []

        for c in k_constraints:
            is_redundant = False

            for existing in kept:
                # Check if directions are similar
                cos_angle = np.dot(c.a, existing.a)
                if cos_angle > np.cos(angle_threshold):
                    # Similar direction - keep the more conservative one (larger b)
                    if c.b > existing.b:
                        kept.remove(existing)
                        kept.append(c)
                    is_redundant = True
                    break

            if not is_redundant:
                kept.append(c)

        merged.extend(kept)

    return merged
