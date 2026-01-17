"""
MPCC Integration Test Framework

Model Predictive Contouring Control (MPCC) tests on curving reference paths
with different obstacle avoidance constraint types.

Features:
- Deterministic, reproducible test scenarios
- Multiple curving path types (gentle curve, sharp curve, S-turn, chicane)
- All obstacle avoidance constraint types (scenario, linearized, gaussian, ellipsoid, safe_horizon)
- Comprehensive metrics (goal reached, path deviation, collisions, solve times)
- Comparison mode to evaluate all constraint types on the same scenario

Usage:
    from test.mpcc_test_framework import run_mpcc_test, MPCCScenario, compare_constraint_types

    # Run a single test
    scenario = MPCCScenario.gentle_curve_with_obstacles()
    result = run_mpcc_test(scenario, constraint_type="scenario")

    # Compare all constraint types
    results = compare_constraint_types(scenario)
"""

import numpy as np
import time
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import create_planner, create_default_config, list_constraint_types, run_mpc
from planning.types import (
    Data, DynamicObstacle, ReferencePath, State,
    Prediction, PredictionStep, PredictionType,
    generate_reference_path
)
from planning.dynamic_models import ContouringSecondOrderUnicycleModel
from utils.utils import read_config_file, LOG_INFO, LOG_WARN
from utils.math_tools import TKSpline


# =============================================================================
# Path Types
# =============================================================================

class PathType(Enum):
    """Types of curving reference paths for MPCC testing."""
    STRAIGHT = "straight"
    GENTLE_CURVE = "gentle_curve"
    SHARP_CURVE = "sharp_curve"
    S_TURN = "s_turn"
    CHICANE = "chicane"
    HAIRPIN = "hairpin"


def create_curving_path(
    path_type: PathType,
    length: float = 30.0,
    num_points: int = 100,
    start: np.ndarray = None,
    curvature_factor: float = 1.0
) -> ReferencePath:
    """
    Create a curving reference path for MPCC testing.

    Args:
        path_type: Type of curve (gentle, sharp, s-turn, chicane, hairpin)
        length: Approximate total path length in meters
        num_points: Number of points to generate
        start: Starting position [x, y], defaults to [0, 0]
        curvature_factor: Multiplier for curvature (higher = tighter curves)

    Returns:
        ReferencePath object with splines and arc length
    """
    if start is None:
        start = np.array([0.0, 0.0])

    t = np.linspace(0, 1, num_points)

    if path_type == PathType.STRAIGHT:
        # Simple straight path
        x = start[0] + t * length
        y = np.zeros(num_points) + start[1]

    elif path_type == PathType.GENTLE_CURVE:
        # Gentle curve: arc with large radius
        # Parametric circle arc: ~30 degrees turn
        angle_span = np.pi / 6 * curvature_factor  # 30 degrees
        radius = length / angle_span
        angles = t * angle_span
        x = start[0] + radius * np.sin(angles)
        y = start[1] + radius * (1 - np.cos(angles))

    elif path_type == PathType.SHARP_CURVE:
        # Sharp curve: arc with smaller radius
        # ~90 degrees turn
        angle_span = np.pi / 2 * curvature_factor  # 90 degrees
        radius = length / angle_span
        angles = t * angle_span
        x = start[0] + radius * np.sin(angles)
        y = start[1] + radius * (1 - np.cos(angles))

    elif path_type == PathType.S_TURN:
        # S-turn: sinusoidal path
        amplitude = 3.0 * curvature_factor
        x = start[0] + t * length
        y = start[1] + amplitude * np.sin(2 * np.pi * t)

    elif path_type == PathType.CHICANE:
        # Chicane: double S-curve (like avoiding obstacles)
        amplitude = 4.0 * curvature_factor
        x = start[0] + t * length
        # Two bumps: one positive, one negative
        y = start[1] + amplitude * np.sin(4 * np.pi * t) * np.exp(-((t - 0.5) ** 2) / 0.1)

    elif path_type == PathType.HAIRPIN:
        # Hairpin turn: 180 degree turn
        angle_span = np.pi * curvature_factor
        radius = length / angle_span / 2
        # First half: curve
        t_mid = 0.5
        angles = np.where(t < t_mid, t / t_mid * angle_span, angle_span)
        x = start[0] + radius * np.sin(angles)
        y = start[1] + radius * (1 - np.cos(angles))
        # Second half: straight after turn
        straight_mask = t >= t_mid
        straight_t = (t[straight_mask] - t_mid) / (1 - t_mid)
        x[straight_mask] = x[straight_mask][-1] + straight_t * (length / 2)
        y[straight_mask] = y[straight_mask][-1]
    else:
        raise ValueError(f"Unknown path type: {path_type}")

    # Compute arc length
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate([[0], np.cumsum(ds)])

    # Create splines
    x_spline = TKSpline(s, x)
    y_spline = TKSpline(s, y)

    # Create ReferencePath
    ref_path = ReferencePath()
    ref_path.x = x
    ref_path.y = y
    ref_path.s = s
    ref_path.x_spline = x_spline
    ref_path.y_spline = y_spline

    return ref_path


# =============================================================================
# Obstacle Configurations
# =============================================================================

@dataclass
class ObstacleConfig:
    """Deterministic obstacle configuration."""
    position: np.ndarray  # [x, y]
    velocity: np.ndarray  # [vx, vy]
    radius: float = 0.5
    prediction_type: str = "gaussian"  # gaussian, deterministic

    def to_dynamic_obstacle(self, obstacle_id: int, horizon: int = 20, timestep: float = 0.1) -> DynamicObstacle:
        """Convert to DynamicObstacle with prediction."""
        # Compute initial angle from velocity
        angle = np.arctan2(self.velocity[1], self.velocity[0]) if np.linalg.norm(self.velocity) > 0 else 0.0

        # Create DynamicObstacle with required arguments
        obstacle = DynamicObstacle(
            index=obstacle_id,
            position=self.position.copy(),
            angle=angle,
            radius=self.radius
        )
        obstacle.velocity = self.velocity.copy()

        # Create prediction
        pred = Prediction()
        if self.prediction_type == "gaussian":
            pred.type = PredictionType.GAUSSIAN
        else:
            pred.type = PredictionType.DETERMINISTIC

        pred.steps = []
        for k in range(horizon + 1):
            # Compute position and uncertainty for this step
            step_position = self.position + self.velocity * k * timestep
            step_angle = angle

            # Gaussian uncertainty grows with time
            if self.prediction_type == "gaussian":
                uncertainty = 0.1 + k * 0.05
                major_radius = uncertainty
                minor_radius = uncertainty * 0.5
            else:
                major_radius = 0.01
                minor_radius = 0.01

            # Create PredictionStep with required arguments
            step = PredictionStep(
                position=step_position,
                angle=step_angle,
                major_radius=major_radius,
                minor_radius=minor_radius
            )
            pred.steps.append(step)

        obstacle.prediction = pred
        return obstacle


def create_obstacle_scenario(
    scenario_name: str,
    path: ReferencePath,
    num_obstacles: int = 3
) -> List[ObstacleConfig]:
    """
    Create deterministic obstacle configurations for a scenario.

    Obstacles are placed relative to the reference path for reproducibility.

    Args:
        scenario_name: Name of the obstacle pattern
        path: Reference path (used to place obstacles along/near it)
        num_obstacles: Number of obstacles to create

    Returns:
        List of ObstacleConfig objects
    """
    obstacles = []
    path_length = float(path.s[-1])

    if scenario_name == "blocking":
        # Obstacles directly on the path at intervals
        for i in range(num_obstacles):
            progress = (i + 1) / (num_obstacles + 1)  # Evenly spaced
            s_pos = progress * path_length

            # Get path position
            path_x = float(path.x_spline.at(s_pos))
            path_y = float(path.y_spline.at(s_pos))

            # Small lateral offset (alternating sides)
            lateral_offset = 1.0 * (1 if i % 2 == 0 else -1)

            # Get path tangent for perpendicular offset
            dx = float(path.x_spline.deriv(s_pos))
            dy = float(path.y_spline.deriv(s_pos))
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 0:
                normal_x = -dy / norm
                normal_y = dx / norm
            else:
                normal_x, normal_y = 0, 1

            pos = np.array([path_x + lateral_offset * normal_x,
                           path_y + lateral_offset * normal_y])

            # Moving slowly across path
            vel = np.array([normal_x * 0.3, normal_y * 0.3])

            obstacles.append(ObstacleConfig(
                position=pos,
                velocity=vel,
                radius=0.5,
                prediction_type="gaussian"
            ))

    elif scenario_name == "crossing":
        # Obstacles crossing the path from sides
        for i in range(num_obstacles):
            progress = (i + 1) / (num_obstacles + 1)
            s_pos = progress * path_length

            path_x = float(path.x_spline.at(s_pos))
            path_y = float(path.y_spline.at(s_pos))

            # Start off path
            side = 1 if i % 2 == 0 else -1
            offset = 4.0 * side

            dx = float(path.x_spline.deriv(s_pos))
            dy = float(path.y_spline.deriv(s_pos))
            norm = np.sqrt(dx**2 + dy**2)
            normal_x = -dy / norm if norm > 0 else 0
            normal_y = dx / norm if norm > 0 else 1

            pos = np.array([path_x + offset * normal_x,
                           path_y + offset * normal_y])

            # Moving toward path
            vel = np.array([-side * normal_x * 0.8, -side * normal_y * 0.8])

            obstacles.append(ObstacleConfig(
                position=pos,
                velocity=vel,
                radius=0.5,
                prediction_type="gaussian"
            ))

    elif scenario_name == "oncoming":
        # Obstacles moving toward the vehicle along the path
        for i in range(num_obstacles):
            progress = 0.3 + (i * 0.2)  # Start at 30%, 50%, 70% of path
            s_pos = min(progress * path_length, path_length * 0.9)

            path_x = float(path.x_spline.at(s_pos))
            path_y = float(path.y_spline.at(s_pos))

            # Small lateral offset
            lateral_offset = 1.5 * (1 if i % 2 == 0 else -1)

            dx = float(path.x_spline.deriv(s_pos))
            dy = float(path.y_spline.deriv(s_pos))
            norm = np.sqrt(dx**2 + dy**2)
            normal_x = -dy / norm if norm > 0 else 0
            normal_y = dx / norm if norm > 0 else 1
            tangent_x = dx / norm if norm > 0 else 1
            tangent_y = dy / norm if norm > 0 else 0

            pos = np.array([path_x + lateral_offset * normal_x,
                           path_y + lateral_offset * normal_y])

            # Moving opposite to path direction (oncoming)
            vel = np.array([-tangent_x * 1.0, -tangent_y * 1.0])

            obstacles.append(ObstacleConfig(
                position=pos,
                velocity=vel,
                radius=0.5,
                prediction_type="deterministic"
            ))

    elif scenario_name == "static":
        # Static obstacles near the path
        for i in range(num_obstacles):
            progress = (i + 1) / (num_obstacles + 1)
            s_pos = progress * path_length

            path_x = float(path.x_spline.at(s_pos))
            path_y = float(path.y_spline.at(s_pos))

            lateral_offset = 2.0 * (1 if i % 2 == 0 else -1)

            dx = float(path.x_spline.deriv(s_pos))
            dy = float(path.y_spline.deriv(s_pos))
            norm = np.sqrt(dx**2 + dy**2)
            normal_x = -dy / norm if norm > 0 else 0
            normal_y = dx / norm if norm > 0 else 1

            pos = np.array([path_x + lateral_offset * normal_x,
                           path_y + lateral_offset * normal_y])

            obstacles.append(ObstacleConfig(
                position=pos,
                velocity=np.array([0.0, 0.0]),
                radius=0.6,
                prediction_type="deterministic"
            ))

    else:
        raise ValueError(f"Unknown obstacle scenario: {scenario_name}")

    return obstacles


# =============================================================================
# Test Scenarios
# =============================================================================

@dataclass
class MPCCScenario:
    """Complete MPCC test scenario definition."""
    name: str
    path_type: PathType
    path_length: float
    obstacle_scenario: str  # blocking, crossing, oncoming, static, none
    num_obstacles: int
    duration: float = 15.0
    timestep: float = 0.1
    initial_velocity: float = 1.0
    goal_tolerance: float = 2.0
    curvature_factor: float = 1.0

    # Computed at runtime
    reference_path: ReferencePath = field(default=None, repr=False)
    obstacles: List[ObstacleConfig] = field(default_factory=list, repr=False)
    goal: np.ndarray = field(default=None, repr=False)

    def setup(self):
        """Initialize the scenario (create path and obstacles)."""
        # Create reference path
        self.reference_path = create_curving_path(
            self.path_type,
            length=self.path_length,
            curvature_factor=self.curvature_factor
        )

        # Set goal at end of path
        self.goal = np.array([
            float(self.reference_path.x[-1]),
            float(self.reference_path.y[-1])
        ])

        # Create obstacles
        if self.obstacle_scenario != "none" and self.num_obstacles > 0:
            self.obstacles = create_obstacle_scenario(
                self.obstacle_scenario,
                self.reference_path,
                self.num_obstacles
            )
        else:
            self.obstacles = []

    @classmethod
    def gentle_curve_with_obstacles(cls, num_obstacles: int = 3) -> 'MPCCScenario':
        """Gentle curve with crossing obstacles."""
        scenario = cls(
            name="gentle_curve_crossing",
            path_type=PathType.GENTLE_CURVE,
            path_length=25.0,
            obstacle_scenario="crossing",
            num_obstacles=num_obstacles,
            duration=15.0
        )
        scenario.setup()
        return scenario

    @classmethod
    def sharp_curve_with_obstacles(cls, num_obstacles: int = 3) -> 'MPCCScenario':
        """Sharp 90-degree curve with blocking obstacles."""
        scenario = cls(
            name="sharp_curve_blocking",
            path_type=PathType.SHARP_CURVE,
            path_length=20.0,
            obstacle_scenario="blocking",
            num_obstacles=num_obstacles,
            duration=15.0
        )
        scenario.setup()
        return scenario

    @classmethod
    def s_turn_with_obstacles(cls, num_obstacles: int = 4) -> 'MPCCScenario':
        """S-turn with oncoming obstacles."""
        scenario = cls(
            name="s_turn_oncoming",
            path_type=PathType.S_TURN,
            path_length=30.0,
            obstacle_scenario="oncoming",
            num_obstacles=num_obstacles,
            duration=20.0
        )
        scenario.setup()
        return scenario

    @classmethod
    def chicane_with_obstacles(cls, num_obstacles: int = 4) -> 'MPCCScenario':
        """Chicane with static obstacles."""
        scenario = cls(
            name="chicane_static",
            path_type=PathType.CHICANE,
            path_length=25.0,
            obstacle_scenario="static",
            num_obstacles=num_obstacles,
            duration=15.0
        )
        scenario.setup()
        return scenario

    @classmethod
    def straight_baseline(cls, num_obstacles: int = 3) -> 'MPCCScenario':
        """Straight path baseline for comparison."""
        scenario = cls(
            name="straight_baseline",
            path_type=PathType.STRAIGHT,
            path_length=20.0,
            obstacle_scenario="crossing",
            num_obstacles=num_obstacles,
            duration=12.0
        )
        scenario.setup()
        return scenario


# =============================================================================
# Test Metrics
# =============================================================================

@dataclass
class MPCCTestMetrics:
    """Comprehensive metrics from an MPCC test run."""
    # Basic outcomes
    success: bool = False
    goal_reached: bool = False
    collision_detected: bool = False
    solver_failures: int = 0

    # Distance metrics
    final_distance_to_goal: float = float('inf')
    min_distance_to_obstacle: float = float('inf')
    max_path_deviation: float = 0.0  # Max contour error
    avg_path_deviation: float = 0.0

    # Progress metrics
    total_path_progress: float = 0.0  # Final spline value / path length
    avg_velocity: float = 0.0

    # Performance metrics
    total_steps: int = 0
    avg_solve_time_ms: float = 0.0
    max_solve_time_ms: float = 0.0
    total_time_s: float = 0.0

    # Trajectory data (for analysis)
    positions: List[Tuple[float, float]] = field(default_factory=list)
    velocities: List[float] = field(default_factory=list)
    spline_values: List[float] = field(default_factory=list)
    path_deviations: List[float] = field(default_factory=list)

    def compute_summary(self):
        """Compute summary statistics from trajectory data."""
        if self.path_deviations:
            self.max_path_deviation = max(abs(d) for d in self.path_deviations)
            self.avg_path_deviation = np.mean(np.abs(self.path_deviations))
        if self.velocities:
            self.avg_velocity = np.mean(self.velocities)

    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting."""
        return {
            'success': self.success,
            'goal_reached': self.goal_reached,
            'collision': self.collision_detected,
            'solver_failures': self.solver_failures,
            'final_dist_to_goal': f"{self.final_distance_to_goal:.2f}m",
            'min_dist_to_obs': f"{self.min_distance_to_obstacle:.2f}m",
            'max_path_dev': f"{self.max_path_deviation:.2f}m",
            'avg_path_dev': f"{self.avg_path_deviation:.2f}m",
            'path_progress': f"{self.total_path_progress*100:.1f}%",
            'steps': self.total_steps,
            'avg_solve_ms': f"{self.avg_solve_time_ms:.1f}",
            'total_time': f"{self.total_time_s:.1f}s"
        }


# =============================================================================
# Test Runner
# =============================================================================

def check_collision(
    position: np.ndarray,
    obstacles: List[DynamicObstacle],
    vehicle_radius: float = 0.5
) -> Tuple[bool, float]:
    """
    Check if vehicle position collides with any obstacle.

    Returns:
        Tuple of (collision_detected, min_distance_to_obstacle)
    """
    min_dist = float('inf')
    collision = False

    for obs in obstacles:
        obs_pos = obs.position if hasattr(obs, 'position') else np.array([0, 0])
        obs_radius = getattr(obs, 'radius', 0.5)

        dist = np.linalg.norm(position - obs_pos) - vehicle_radius - obs_radius
        min_dist = min(min_dist, dist)

        if dist < 0:
            collision = True

    return collision, min_dist


def compute_path_deviation(
    position: np.ndarray,
    reference_path: ReferencePath,
    spline_value: float
) -> float:
    """
    Compute the contour error (perpendicular distance from path).

    Args:
        position: Current [x, y] position
        reference_path: Reference path
        spline_value: Current arc length progress

    Returns:
        Signed contour error (positive = left of path, negative = right)
    """
    # Clamp spline value to valid range
    s_max = float(reference_path.s[-1])
    s = max(0, min(spline_value, s_max))

    # Get path position and tangent at this arc length
    path_x = float(reference_path.x_spline.at(s))
    path_y = float(reference_path.y_spline.at(s))

    dx = float(reference_path.x_spline.deriv(s))
    dy = float(reference_path.y_spline.deriv(s))

    norm = np.sqrt(dx**2 + dy**2)
    if norm > 0:
        # Normal vector (perpendicular to tangent)
        normal_x = -dy / norm
        normal_y = dx / norm
    else:
        normal_x, normal_y = 0, 1

    # Contour error = dot product of (position - path_point) with normal
    error = (position[0] - path_x) * normal_x + (position[1] - path_y) * normal_y

    return error


def run_mpcc_test(
    scenario: MPCCScenario,
    constraint_type: str = "scenario",
    verbose: bool = True
) -> MPCCTestMetrics:
    """
    Run an MPCC test with the given scenario and constraint type.

    Args:
        scenario: MPCCScenario with path and obstacles
        constraint_type: Obstacle avoidance constraint type
        verbose: Print progress information

    Returns:
        MPCCTestMetrics with test results
    """
    metrics = MPCCTestMetrics()

    try:
        # Ensure scenario is setup
        if scenario.reference_path is None:
            scenario.setup()

        # Setup initial state
        initial_state = {
            'x': float(scenario.reference_path.x[0]),
            'y': float(scenario.reference_path.y[0]),
            'psi': 0.0,  # Will be computed from path tangent
            'v': scenario.initial_velocity,
            'spline': 0.0  # Start at beginning of path
        }

        # Compute initial heading from path tangent
        dx = float(scenario.reference_path.x_spline.deriv(0))
        dy = float(scenario.reference_path.y_spline.deriv(0))
        initial_state['psi'] = np.arctan2(dy, dx)

        # Convert obstacle configs to DynamicObstacles
        horizon = 20
        dynamic_obstacles = []
        for i, obs_config in enumerate(scenario.obstacles):
            dyn_obs = obs_config.to_dynamic_obstacle(
                obstacle_id=i,
                horizon=horizon,
                timestep=scenario.timestep
            )
            dynamic_obstacles.append(dyn_obs)

        # Create planner with all required arguments
        planner = create_planner(
            initial_state=initial_state,
            reference_path=scenario.reference_path,
            obstacles=dynamic_obstacles,
            goal=scenario.goal,
            constraint_type=constraint_type
        )

        # Run MPC
        num_steps = int(scenario.duration / scenario.timestep)
        start_time = time.time()

        if verbose:
            print(f"\nRunning MPCC test: {scenario.name}")
            print(f"  Constraint type: {constraint_type}")
            print(f"  Path type: {scenario.path_type.value}")
            print(f"  Obstacles: {len(scenario.obstacles)}")
            print(f"  Duration: {scenario.duration}s")

        # Run the MPC loop
        mpc_result = run_mpc(
            planner=planner,
            max_steps=num_steps,
            goal_threshold=scenario.goal_tolerance
        )

        total_time = time.time() - start_time

        # Extract metrics
        metrics.total_time_s = total_time
        metrics.total_steps = mpc_result.get('steps', 0)
        metrics.goal_reached = mpc_result.get('goal_reached', False)

        # Get final state
        final_state = planner.get_state()
        final_x = final_state.get('x') if final_state.get('x') is not None else 0.0
        final_y = final_state.get('y') if final_state.get('y') is not None else 0.0
        final_pos = np.array([final_x, final_y])
        metrics.final_distance_to_goal = np.linalg.norm(final_pos - scenario.goal)

        # Get trajectory
        if mpc_result.get('trajectory'):
            metrics.positions = mpc_result['trajectory']

        # Compute solve time stats
        if metrics.total_steps > 0:
            metrics.avg_solve_time_ms = (total_time / metrics.total_steps) * 1000
            metrics.max_solve_time_ms = metrics.avg_solve_time_ms  # Approximate

        # Check for collisions and compute path deviation
        # (This is a simplified check - actual collision checking should be more thorough)
        min_obs_dist = float('inf')
        for pos in metrics.positions:
            pos_arr = np.array(pos)
            for obs in dynamic_obstacles:
                dist = np.linalg.norm(pos_arr - obs.position)
                min_obs_dist = min(min_obs_dist, dist)

        metrics.min_distance_to_obstacle = min_obs_dist - 0.5 - 0.5  # Vehicle and obstacle radius
        metrics.collision_detected = metrics.min_distance_to_obstacle < 0

        # Compute path progress
        final_spline = final_state.get('spline') if final_state.get('spline') is not None else 0.0
        path_length = float(scenario.reference_path.s[-1])
        metrics.total_path_progress = final_spline / path_length if path_length > 0 else 0

        # Determine success
        metrics.success = metrics.goal_reached and not metrics.collision_detected

        if verbose:
            status = "PASS" if metrics.success else "FAIL"
            print(f"Test: {scenario.name} [{status}]")
            print(f"  Goal reached: {metrics.goal_reached}")
            print(f"  Final distance to goal: {metrics.final_distance_to_goal:.2f}m")
            print(f"  Collision: {metrics.collision_detected}")
            print(f"  Min obstacle distance: {metrics.min_distance_to_obstacle:.2f}m")
            print(f"  Steps: {metrics.total_steps}")
            print(f"  Avg solve time: {metrics.avg_solve_time_ms:.1f}ms")

    except Exception as e:
        metrics.success = False
        if verbose:
            print(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()

    return metrics


def compare_constraint_types(
    scenario: MPCCScenario,
    constraint_types: List[str] = None,
    verbose: bool = True
) -> Dict[str, MPCCTestMetrics]:
    """
    Run the same scenario with different constraint types and compare results.

    Args:
        scenario: MPCC scenario to test
        constraint_types: List of constraint types to compare
        verbose: Print progress and comparison

    Returns:
        Dictionary mapping constraint type to metrics
    """
    if constraint_types is None:
        constraint_types = ["scenario", "linearized", "gaussian", "ellipsoid", "safe_horizon"]

    results = {}

    if verbose:
        print("\n" + "=" * 60)
        print(f"CONSTRAINT TYPE COMPARISON: {scenario.name}")
        print("=" * 60)

    for ctype in constraint_types:
        if verbose:
            print(f"\n--- Testing with {ctype} constraints ---")

        try:
            metrics = run_mpcc_test(scenario, constraint_type=ctype, verbose=verbose)
            results[ctype] = metrics
        except Exception as e:
            if verbose:
                print(f"  Error with {ctype}: {e}")
            results[ctype] = MPCCTestMetrics(success=False)

    # Print comparison summary
    if verbose:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Constraint Type':<15} {'Success':<8} {'Goal':<6} {'Collision':<10} {'Dist':<8} {'Steps':<6} {'Time':<8}")
        print("-" * 60)

        for ctype, m in results.items():
            success = "PASS" if m.success else "FAIL"
            goal = "Yes" if m.goal_reached else "No"
            collision = "Yes" if m.collision_detected else "No"
            dist = f"{m.final_distance_to_goal:.2f}m"
            steps = str(m.total_steps)
            time_s = f"{m.avg_solve_time_ms:.0f}ms"

            print(f"{ctype:<15} {success:<8} {goal:<6} {collision:<10} {dist:<8} {steps:<6} {time_s:<8}")

    return results


# =============================================================================
# Predefined Test Suite
# =============================================================================

def get_all_scenarios() -> List[MPCCScenario]:
    """Get all predefined test scenarios."""
    return [
        MPCCScenario.straight_baseline(),
        MPCCScenario.gentle_curve_with_obstacles(),
        MPCCScenario.sharp_curve_with_obstacles(),
        MPCCScenario.s_turn_with_obstacles(),
        MPCCScenario.chicane_with_obstacles(),
    ]


def run_full_test_suite(
    constraint_types: List[str] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, MPCCTestMetrics]]:
    """
    Run the full test suite with all scenarios and constraint types.

    Returns:
        Nested dict: {scenario_name: {constraint_type: metrics}}
    """
    if constraint_types is None:
        constraint_types = ["scenario", "linearized", "gaussian"]

    all_results = {}
    scenarios = get_all_scenarios()

    if verbose:
        print("\n" + "=" * 70)
        print("MPCC FULL TEST SUITE")
        print("=" * 70)
        print(f"Scenarios: {len(scenarios)}")
        print(f"Constraint types: {constraint_types}")
        print("=" * 70)

    for scenario in scenarios:
        if verbose:
            print(f"\n\n{'#' * 70}")
            print(f"# SCENARIO: {scenario.name}")
            print(f"{'#' * 70}")

        results = compare_constraint_types(
            scenario,
            constraint_types=constraint_types,
            verbose=verbose
        )
        all_results[scenario.name] = results

    # Print final summary
    if verbose:
        print("\n\n" + "=" * 70)
        print("FINAL TEST SUITE SUMMARY")
        print("=" * 70)

        # Count passes/fails
        total_tests = 0
        passed_tests = 0

        for scenario_name, results in all_results.items():
            for ctype, metrics in results.items():
                total_tests += 1
                if metrics.success:
                    passed_tests += 1

        print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

        # Detailed breakdown
        print("\nBreakdown by scenario:")
        for scenario_name, results in all_results.items():
            passes = sum(1 for m in results.values() if m.success)
            total = len(results)
            print(f"  {scenario_name}: {passes}/{total}")

        print("\nBreakdown by constraint type:")
        for ctype in constraint_types:
            passes = sum(1 for results in all_results.values() if results.get(ctype, MPCCTestMetrics()).success)
            total = len(all_results)
            print(f"  {ctype}: {passes}/{total}")

    return all_results


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MPCC Integration Test Framework")
    parser.add_argument("--scenario", "-s", type=str, default=None,
                       help="Specific scenario to run (straight_baseline, gentle_curve_crossing, etc.)")
    parser.add_argument("--constraint-type", "-c", type=str, default=None,
                       help="Specific constraint type (scenario, linearized, gaussian, ellipsoid, safe_horizon)")
    parser.add_argument("--compare", action="store_true",
                       help="Compare all constraint types on the scenario")
    parser.add_argument("--full-suite", action="store_true",
                       help="Run the full test suite")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available scenarios and constraint types")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Reduce output verbosity")

    args = parser.parse_args()

    if args.list:
        print("Available scenarios:")
        for s in get_all_scenarios():
            print(f"  - {s.name}: {s.path_type.value}, {s.num_obstacles} obstacles")
        print("\nAvailable constraint types:")
        for ct in ["scenario", "linearized", "gaussian", "ellipsoid", "safe_horizon"]:
            print(f"  - {ct}")
        sys.exit(0)

    if args.full_suite:
        run_full_test_suite(verbose=not args.quiet)
    elif args.scenario:
        # Find the scenario
        scenarios = {s.name: s for s in get_all_scenarios()}
        if args.scenario not in scenarios:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {list(scenarios.keys())}")
            sys.exit(1)

        scenario = scenarios[args.scenario]

        if args.compare:
            compare_constraint_types(scenario, verbose=not args.quiet)
        elif args.constraint_type:
            run_mpcc_test(scenario, constraint_type=args.constraint_type, verbose=not args.quiet)
        else:
            # Default: run with scenario constraint
            run_mpcc_test(scenario, constraint_type="scenario", verbose=not args.quiet)
    else:
        # Default: run comparison on gentle curve
        scenario = MPCCScenario.gentle_curve_with_obstacles()
        compare_constraint_types(scenario, verbose=not args.quiet)
