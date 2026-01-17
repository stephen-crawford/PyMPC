"""
Simplified Integration Test Framework for PyMPC.

Provides a clean, easy-to-use API for running integration tests that
integrates with the main.py constraint type registry.

Usage:
    from test.simple_framework import run_test, TestScenario

    scenario = TestScenario(
        name="obstacle_avoidance",
        constraint_type="scenario",
        num_obstacles=3,
    )
    result = run_test(scenario)
"""

import numpy as np
import time
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import create_planner, create_default_config, list_constraint_types, run_mpc
from planning.types import (
    Data, DynamicObstacle, ReferencePath,
    generate_reference_path, PredictionType
)
from planning.obstacle_manager import (
    ObstacleManager, ObstacleConfig,
    create_unicycle_obstacle, create_point_mass_obstacle
)
from utils.utils import read_config_file


@dataclass
class TestScenario:
    """Configuration for a test scenario."""
    name: str = "test"

    # Constraint configuration
    constraint_type: str = "linearized"  # scenario, linearized, gaussian, ellipsoid, safe_horizon

    # Path configuration
    path_type: str = "straight"  # straight, curved, s-turn
    path_length: float = 20.0

    # Goal configuration
    goal: Optional[np.ndarray] = None

    # Obstacle configuration
    num_obstacles: int = 0
    obstacle_speed: float = 1.0
    prediction_type: str = "gaussian"  # gaussian, deterministic

    # Simulation parameters
    duration: float = 10.0
    timestep: float = 0.1

    # Vehicle configuration
    vehicle_type: str = "unicycle"  # unicycle, bicycle
    initial_position: Tuple[float, float] = (0.0, 0.0)
    initial_velocity: float = 1.0

    # Visualization
    visualize: bool = False
    save_animation: bool = False
    output_dir: Optional[str] = None


@dataclass
class TestResult:
    """Result of running a test scenario."""
    success: bool
    name: str

    # State history
    positions: List[np.ndarray] = field(default_factory=list)
    velocities: List[float] = field(default_factory=list)

    # Performance metrics
    solve_times: List[float] = field(default_factory=list)
    collision_count: int = 0
    goal_reached: bool = False
    final_distance_to_goal: float = float('inf')

    # Error info
    error_message: str = ""

    @property
    def avg_solve_time(self) -> float:
        return np.mean(self.solve_times) if self.solve_times else 0.0

    @property
    def max_solve_time(self) -> float:
        return max(self.solve_times) if self.solve_times else 0.0

    def summary(self) -> str:
        """Return a summary string."""
        status = "PASS" if self.success else "FAIL"
        lines = [
            f"Test: {self.name} [{status}]",
            f"  Steps: {len(self.positions)}",
            f"  Goal reached: {self.goal_reached}",
            f"  Final distance to goal: {self.final_distance_to_goal:.2f}m",
            f"  Collisions: {self.collision_count}",
            f"  Avg solve time: {self.avg_solve_time*1000:.1f}ms",
            f"  Max solve time: {self.max_solve_time*1000:.1f}ms",
        ]
        if self.error_message:
            lines.append(f"  Error: {self.error_message}")
        return "\n".join(lines)


def create_reference_path(path_type: str, length: float, num_points: int = 50) -> np.ndarray:
    """Create a reference path for testing."""
    if path_type == "straight":
        start = [0.0, 0.0, 0.0]
        goal = [length, 0.0, 0.0]
    elif path_type == "curved":
        start = [0.0, 0.0, 0.0]
        goal = [length * 0.8, length * 0.5, 0.0]
    elif path_type == "s-turn":
        start = [0.0, 0.0, 0.0]
        goal = [length * 0.9, length * 0.2, 0.0]
    else:
        start = [0.0, 0.0, 0.0]
        goal = [length, 0.0, 0.0]

    return generate_reference_path(start, goal, path_type, num_points)


def create_obstacles(
    num_obstacles: int,
    speed: float = 1.0,
    prediction_type: str = "gaussian",
    bounds: Tuple[float, float, float, float] = (2.0, 18.0, -5.0, 5.0)
) -> List[DynamicObstacle]:
    """Create dynamic obstacles for testing."""
    config = read_config_file()
    manager = ObstacleManager(config, plot_bounds=bounds)

    obstacles = []
    for i in range(num_obstacles):
        # Random position in bounds
        x = np.random.uniform(bounds[0] + 2, bounds[1] - 2)
        y = np.random.uniform(bounds[2] + 1, bounds[3] - 1)

        # Random direction
        angle = np.random.uniform(0, 2 * np.pi)
        velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])

        obs_config = create_unicycle_obstacle(i, np.array([x, y]), velocity)

        # Set prediction type
        if prediction_type.lower() == "gaussian":
            obs_config.prediction_type = PredictionType.GAUSSIAN
        else:
            obs_config.prediction_type = PredictionType.DETERMINISTIC

        obstacles.append(obs_config)

    return manager.create_obstacles_from_config(obstacles)


def run_test(scenario: TestScenario, verbose: bool = True) -> TestResult:
    """
    Run an integration test with the given scenario.

    Args:
        scenario: Test scenario configuration
        verbose: Whether to print progress

    Returns:
        TestResult with test outcomes
    """
    result = TestResult(success=False, name=scenario.name)

    try:
        # Create reference path
        ref_path = create_reference_path(
            scenario.path_type,
            scenario.path_length
        )

        # Set goal
        if scenario.goal is not None:
            goal = scenario.goal
        else:
            goal = np.array([scenario.path_length, 0.0])

        # Create obstacles
        obstacles = []
        if scenario.num_obstacles > 0:
            obstacles = create_obstacles(
                scenario.num_obstacles,
                speed=scenario.obstacle_speed,
                prediction_type=scenario.prediction_type
            )

        # Create configuration
        config = create_default_config(constraint_type=scenario.constraint_type)
        config["planner"]["timestep"] = scenario.timestep

        # Initial state
        initial_state = {
            'x': scenario.initial_position[0],
            'y': scenario.initial_position[1],
            'psi': 0.0,
            'v': scenario.initial_velocity,
        }

        # Create planner
        planner = create_planner(
            initial_state=initial_state,
            reference_path=ref_path,
            obstacles=obstacles,
            goal=goal,
            config=config,
            constraint_type=scenario.constraint_type
        )

        if verbose:
            print(f"Running test: {scenario.name}")
            print(f"  Constraint type: {scenario.constraint_type}")
            print(f"  Obstacles: {scenario.num_obstacles}")
            print(f"  Duration: {scenario.duration}s")

        # Run simulation using main.py's run_mpc function
        num_steps = int(scenario.duration / scenario.timestep)

        start_time = time.time()
        try:
            mpc_result = run_mpc(
                planner=planner,
                max_steps=num_steps,
                goal_threshold=1.0
            )
            total_time = time.time() - start_time

            # Extract results
            if mpc_result['trajectory']:
                result.positions = [np.array(p) for p in mpc_result['trajectory']]

            result.goal_reached = mpc_result.get('goal_reached', False)
            result.solve_times = [total_time / max(mpc_result['steps'], 1)] * mpc_result['steps']

            # Final metrics
            if result.positions:
                final_pos = result.positions[-1]
            else:
                final_pos = np.array([0.0, 0.0])
            result.final_distance_to_goal = np.linalg.norm(final_pos - goal[:2])
            result.success = True

        except Exception as e:
            result.error_message = str(e)
            result.success = False
            if verbose:
                print(f"  MPC run failed: {e}")

        if verbose:
            print(result.summary())

    except Exception as e:
        result.error_message = str(e)
        result.success = False
        if verbose:
            print(f"Test failed: {e}")

    return result


def run_comparison_test(
    scenario_base: TestScenario,
    constraint_types: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, TestResult]:
    """
    Run the same scenario with different constraint types for comparison.

    Args:
        scenario_base: Base scenario configuration
        constraint_types: List of constraint types to compare
        verbose: Whether to print progress

    Returns:
        Dictionary mapping constraint type to TestResult
    """
    if constraint_types is None:
        constraint_types = list_constraint_types()

    results = {}

    if verbose:
        print(f"\nComparison test: {scenario_base.name}")
        print(f"Testing constraint types: {constraint_types}")
        print("-" * 50)

    for ctype in constraint_types:
        # Create scenario with this constraint type
        scenario = TestScenario(
            name=f"{scenario_base.name}_{ctype}",
            constraint_type=ctype,
            path_type=scenario_base.path_type,
            path_length=scenario_base.path_length,
            goal=scenario_base.goal,
            num_obstacles=scenario_base.num_obstacles,
            obstacle_speed=scenario_base.obstacle_speed,
            prediction_type=scenario_base.prediction_type,
            duration=scenario_base.duration,
            timestep=scenario_base.timestep,
            vehicle_type=scenario_base.vehicle_type,
            initial_position=scenario_base.initial_position,
            initial_velocity=scenario_base.initial_velocity,
        )

        if verbose:
            print(f"\nConstraint type: {ctype}")

        results[ctype] = run_test(scenario, verbose=verbose)

    # Print comparison summary
    if verbose:
        print("\n" + "=" * 50)
        print("COMPARISON SUMMARY")
        print("=" * 50)
        for ctype, result in results.items():
            status = "PASS" if result.success else "FAIL"
            print(f"  {ctype:15} [{status}] - "
                  f"goal: {result.goal_reached}, "
                  f"collisions: {result.collision_count}, "
                  f"avg_time: {result.avg_solve_time*1000:.1f}ms")

    return results


# Predefined test scenarios
SCENARIOS = {
    "basic_straight": TestScenario(
        name="basic_straight",
        constraint_type="linearized",
        path_type="straight",
        path_length=20.0,
        num_obstacles=0,
        duration=10.0,
    ),

    "obstacle_avoidance": TestScenario(
        name="obstacle_avoidance",
        constraint_type="scenario",
        path_type="straight",
        path_length=20.0,
        num_obstacles=3,
        obstacle_speed=1.0,
        prediction_type="gaussian",
        duration=15.0,
    ),

    "curved_path": TestScenario(
        name="curved_path",
        constraint_type="linearized",
        path_type="curved",
        path_length=25.0,
        num_obstacles=2,
        duration=12.0,
    ),

    "dense_obstacles": TestScenario(
        name="dense_obstacles",
        constraint_type="gaussian",
        path_type="straight",
        path_length=30.0,
        num_obstacles=6,
        obstacle_speed=0.8,
        duration=20.0,
    ),
}


def get_scenario(name: str) -> TestScenario:
    """Get a predefined test scenario by name."""
    if name not in SCENARIOS:
        available = ", ".join(SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{name}'. Available: {available}")
    return SCENARIOS[name]


def list_scenarios() -> List[str]:
    """List available predefined scenarios."""
    return list(SCENARIOS.keys())
