"""
Scenario pruning for efficient optimization.

Implements Algorithms 3 and 4 from guide.md:
- Algorithm 3: Geometric dominance pruning
- Algorithm 4: Support-based scenario removal

These algorithms reduce the number of scenarios while maintaining
chance constraint guarantees.
"""

import numpy as np
from typing import List, Set, Tuple, Optional
from .types import (
    EgoState,
    Scenario,
    CollisionConstraint,
)


def prune_dominated_scenarios(
    scenarios: List[Scenario],
    reference_trajectory: List[EgoState],
    ego_radius: float,
    obstacle_radius: float
) -> List[Scenario]:
    """
    Prune geometrically dominated scenarios.

    Algorithm 3 from guide.md: GeometricDominancePruning

    A scenario s1 dominates s2 if for all timesteps k and obstacles o,
    the obstacle position in s1 is further from the ego than in s2.

    Args:
        scenarios: List of scenarios to prune
        reference_trajectory: Reference ego trajectory
        ego_radius: Ego collision radius
        obstacle_radius: Obstacle collision radius

    Returns:
        Pruned list of non-dominated scenarios
    """
    if len(scenarios) <= 1:
        return scenarios

    n = len(scenarios)
    dominated = set()

    # Pairwise comparison
    for i in range(n):
        if i in dominated:
            continue

        for j in range(i + 1, n):
            if j in dominated:
                continue

            # Check if i dominates j or vice versa
            dom_i_j = _scenario_dominates(
                scenarios[i], scenarios[j], reference_trajectory
            )
            dom_j_i = _scenario_dominates(
                scenarios[j], scenarios[i], reference_trajectory
            )

            if dom_i_j:
                dominated.add(j)
            elif dom_j_i:
                dominated.add(i)
                break

    # Return non-dominated scenarios
    return [s for i, s in enumerate(scenarios) if i not in dominated]


def _scenario_dominates(
    s1: Scenario,
    s2: Scenario,
    reference_trajectory: List[EgoState]
) -> bool:
    """
    Check if scenario s1 dominates s2.

    s1 dominates s2 if obstacles in s1 are always at least as far
    from the ego as in s2 (s1 is "safer" everywhere).
    """
    horizon = len(reference_trajectory) - 1

    for k in range(horizon + 1):
        ego_pos = reference_trajectory[k].position()

        # Check all obstacles present in both scenarios
        common_obs = set(s1.trajectories.keys()) & set(s2.trajectories.keys())

        for obs_id in common_obs:
            traj1 = s1.trajectories[obs_id]
            traj2 = s2.trajectories[obs_id]

            if k >= len(traj1.steps) or k >= len(traj2.steps):
                continue

            pos1 = traj1.steps[k].mean
            pos2 = traj2.steps[k].mean

            dist1 = np.linalg.norm(ego_pos - pos1)
            dist2 = np.linalg.norm(ego_pos - pos2)

            # If s2 obstacle is further at any point, s1 doesn't dominate
            if dist2 > dist1 + 1e-6:
                return False

    return True


def remove_inactive_scenarios(
    scenarios: List[Scenario],
    constraints: List[CollisionConstraint],
    optimal_trajectory: List[EgoState],
    tolerance: float = 1e-4
) -> Tuple[List[Scenario], Set[int]]:
    """
    Remove scenarios with inactive constraints.

    Algorithm 4 from guide.md: SupportBasedRemoval

    After solving the optimization, identify scenarios whose constraints
    are not active (not binding) and remove them.

    A constraint is active if it is satisfied with equality (within tolerance).

    Args:
        scenarios: Current list of scenarios
        constraints: Collision constraints for all scenarios
        optimal_trajectory: Optimal ego trajectory from solver
        tolerance: Tolerance for checking constraint activity

    Returns:
        (remaining_scenarios, active_scenario_ids)
    """
    # Group constraints by scenario
    constraints_by_scenario = {}
    for c in constraints:
        if c.scenario_id not in constraints_by_scenario:
            constraints_by_scenario[c.scenario_id] = []
        constraints_by_scenario[c.scenario_id].append(c)

    active_scenarios = set()

    for scenario in scenarios:
        sid = scenario.scenario_id

        if sid not in constraints_by_scenario:
            continue

        # Check if any constraint from this scenario is active
        for constraint in constraints_by_scenario[sid]:
            k = constraint.k
            if k >= len(optimal_trajectory):
                continue

            ego_pos = optimal_trajectory[k].position()
            value = constraint.evaluate(ego_pos)

            # Constraint is active if value is close to 0
            if abs(value) < tolerance:
                active_scenarios.add(sid)
                break

    # Keep scenarios that have active constraints
    remaining = [s for s in scenarios if s.scenario_id in active_scenarios]

    # If no scenarios are active, keep all (safety fallback)
    if not remaining:
        return scenarios, set(s.scenario_id for s in scenarios)

    return remaining, active_scenarios


def prune_by_probability(
    scenarios: List[Scenario],
    min_probability: float = 0.01
) -> List[Scenario]:
    """
    Remove scenarios with very low probability.

    Simple pruning that removes scenarios unlikely to occur.

    Args:
        scenarios: List of scenarios
        min_probability: Minimum probability threshold

    Returns:
        Filtered scenarios
    """
    return [s for s in scenarios if s.probability >= min_probability]


def cluster_similar_scenarios(
    scenarios: List[Scenario],
    distance_threshold: float = 1.0,
    horizon: int = None
) -> List[Scenario]:
    """
    Cluster similar scenarios and keep representative ones.

    Scenarios are similar if their obstacle trajectories are close
    across the horizon.

    Args:
        scenarios: List of scenarios
        distance_threshold: Maximum distance for clustering
        horizon: Prediction horizon (inferred if not provided)

    Returns:
        Representative scenarios (cluster centroids)
    """
    if len(scenarios) <= 1:
        return scenarios

    # Compute scenario features (flattened obstacle positions)
    features = []
    for scenario in scenarios:
        feat = _scenario_to_feature(scenario, horizon)
        features.append(feat)

    features = np.array(features)

    # Simple greedy clustering
    kept_indices = []
    for i, feat in enumerate(features):
        is_close = False
        for j in kept_indices:
            if np.linalg.norm(feat - features[j]) < distance_threshold:
                is_close = True
                break
        if not is_close:
            kept_indices.append(i)

    return [scenarios[i] for i in kept_indices]


def _scenario_to_feature(scenario: Scenario, horizon: int = None) -> np.ndarray:
    """
    Convert scenario to feature vector for clustering.
    """
    if horizon is None:
        # Infer horizon from trajectory length
        for traj in scenario.trajectories.values():
            horizon = len(traj.steps) - 1
            break
        if horizon is None:
            horizon = 10

    features = []

    # Sort obstacle IDs for consistent ordering
    for obs_id in sorted(scenario.trajectories.keys()):
        traj = scenario.trajectories[obs_id]
        for step in traj.steps[:horizon + 1]:
            features.extend(step.mean)

    return np.array(features) if features else np.array([0.0])


def select_diverse_scenarios(
    scenarios: List[Scenario],
    num_select: int,
    horizon: int = None
) -> List[Scenario]:
    """
    Select a diverse subset of scenarios.

    Uses farthest point sampling to maximize diversity.

    Args:
        scenarios: All scenarios
        num_select: Number of scenarios to select
        horizon: Prediction horizon

    Returns:
        Diverse subset of scenarios
    """
    if len(scenarios) <= num_select:
        return scenarios

    # Convert to features
    features = np.array([_scenario_to_feature(s, horizon) for s in scenarios])

    # Farthest point sampling
    selected = [0]  # Start with first scenario

    while len(selected) < num_select:
        # Find point farthest from all selected points
        max_min_dist = -1
        best_idx = -1

        for i in range(len(scenarios)):
            if i in selected:
                continue

            # Min distance to any selected point
            min_dist = float('inf')
            for j in selected:
                dist = np.linalg.norm(features[i] - features[j])
                min_dist = min(min_dist, dist)

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = i

        if best_idx >= 0:
            selected.append(best_idx)
        else:
            break

    return [scenarios[i] for i in selected]


def adaptive_scenario_budget(
    current_scenarios: int,
    constraint_violations: float,
    max_scenarios: int = 100,
    min_scenarios: int = 5,
    growth_factor: float = 1.5,
    shrink_factor: float = 0.8
) -> int:
    """
    Adaptively adjust scenario budget based on constraint violations.

    If constraints are frequently violated, increase scenarios.
    If constraints are easily satisfied, decrease scenarios.

    Args:
        current_scenarios: Current number of scenarios
        constraint_violations: Sum of constraint violations
        max_scenarios: Maximum allowed scenarios
        min_scenarios: Minimum scenarios to maintain
        growth_factor: Factor to increase scenarios
        shrink_factor: Factor to decrease scenarios

    Returns:
        New scenario budget
    """
    if constraint_violations > 0.1:
        # Violations present - increase scenarios
        new_budget = int(current_scenarios * growth_factor)
    elif constraint_violations < 0.001:
        # No violations - can decrease scenarios
        new_budget = int(current_scenarios * shrink_factor)
    else:
        new_budget = current_scenarios

    return max(min_scenarios, min(max_scenarios, new_budget))
