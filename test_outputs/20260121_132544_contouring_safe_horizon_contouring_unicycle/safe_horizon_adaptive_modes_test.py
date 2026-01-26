"""
Safe Horizon Constraint Integration Test with Adaptive Mode Sampling

This test demonstrates the adaptive mode-based sampling for safe horizon constraints,
showing correct handling of obstacles displaying different dynamic modes.

Following guide.md:
- Mode history tracking for each obstacle
- Mode weight computation (uniform, recency, frequency)
- Mode-dependent dynamics for trajectory prediction

Test scenario:
- Vehicle follows a curved reference path
- Multiple dynamic obstacles with DIFFERENT behavior modes:
  * Obstacle 0: Constant velocity (straight line motion)
  * Obstacle 1: Turn left (curved trajectory)
  * Obstacle 2: Decelerating (slowing down)
  * Obstacle 3: Lane change left (lateral drift)
- Safe Horizon constraints with adaptive mode sampling ensure probabilistic collision avoidance
- Mode weights are updated based on observed behavior history
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig, create_reference_path
from planning.obstacle_manager import create_unicycle_obstacle
from planning.types import PredictionType
import numpy as np


# Define mode configurations for each obstacle
OBSTACLE_MODE_CONFIGS = [
    {
        "name": "Constant Velocity Obstacle",
        "initial_mode": "constant_velocity",
        "available_modes": ["constant_velocity", "decelerating", "accelerating"],
        "mode_switch_probability": 0.05,  # Low probability of mode switch
        "color": "blue"
    },
    {
        "name": "Turning Obstacle",
        "initial_mode": "turn_left",
        "available_modes": ["turn_left", "turn_right", "constant_velocity"],
        "mode_switch_probability": 0.1,  # Medium probability of mode switch
        "color": "green"
    },
    {
        "name": "Decelerating Obstacle",
        "initial_mode": "decelerating",
        "available_modes": ["decelerating", "constant_velocity", "accelerating"],
        "mode_switch_probability": 0.15,  # Higher probability of mode switch
        "color": "orange"
    },
    {
        "name": "Lane Changing Obstacle",
        "initial_mode": "lane_change_left",
        "available_modes": ["lane_change_left", "lane_change_right", "constant_velocity"],
        "mode_switch_probability": 0.2,  # High probability of mode switch
        "color": "red"
    },
]


def create_mode_aware_obstacle(
    obstacle_id: int,
    position: np.ndarray,
    velocity: np.ndarray,
    angle: float,
    radius: float,
    mode_config: dict
):
    """
    Create an obstacle with mode-aware configuration for adaptive sampling.

    Args:
        obstacle_id: Obstacle identifier
        position: Initial position [x, y]
        velocity: Initial velocity [vx, vy]
        angle: Initial heading angle
        radius: Obstacle radius
        mode_config: Dictionary with mode configuration

    Returns:
        ObstacleConfig with mode attributes
    """
    # Create base obstacle configuration
    obstacle_config = create_unicycle_obstacle(
        obstacle_id=obstacle_id,
        position=position,
        velocity=velocity,
        angle=angle,
        radius=radius,
        behavior="path_intersect"
    )

    # Set prediction type to Gaussian for safe horizon constraints
    obstacle_config.prediction_type = PredictionType.GAUSSIAN
    obstacle_config.uncertainty_params = {
        'position_std': 0.2,
        'uncertainty_growth': 0.1
    }

    # Add mode-aware attributes for adaptive sampling
    # These will be read by the AdaptiveModeSampler
    obstacle_config.initial_mode = mode_config["initial_mode"]
    obstacle_config.current_mode = mode_config["initial_mode"]  # Start with initial mode
    obstacle_config.available_modes = mode_config["available_modes"]
    obstacle_config.mode_switch_probability = mode_config["mode_switch_probability"]
    obstacle_config.mode_name = mode_config["name"]

    return obstacle_config


def run(
    dt=0.1,
    duration=30.0,
    timeout_minutes=20.0,
    weight_type="frequency",
    prior_type="constant"
):
    """
    Run Safe Horizon constraints test with adaptive mode sampling.

    This test demonstrates how obstacles with different dynamic modes are handled
    by the adaptive mode-based scenario sampling.

    Args:
        dt: Timestep in seconds (default 0.1s)
        duration: Simulation duration in seconds (default 30s)
        timeout_minutes: Test timeout in minutes (default 20 minutes)
        weight_type: Mode weight computation strategy ("uniform", "recency", "frequency")
        prior_type: Mode prior type ("constant" for C1, "switching" for C2)
    """
    num_obstacles = len(OBSTACLE_MODE_CONFIGS)

    print(f"\n{'='*60}")
    print("Safe Horizon Adaptive Mode Sampling Test")
    print(f"{'='*60}")
    print(f"\nThis test demonstrates adaptive mode-based scenario sampling")
    print(f"for obstacles displaying different dynamic behaviors.\n")

    framework = IntegrationTestFramework()

    # Create curved reference path
    ref_path = create_reference_path("s_curve", length=25.0)
    path_length = ref_path.length

    # Create obstacle configurations with different modes
    obstacle_configs = []

    for i, mode_config in enumerate(OBSTACLE_MODE_CONFIGS):
        # Place obstacles at different positions along the path
        # Start further along path (25%) to give vehicle time to react
        path_fraction = 0.25 + (i * 0.18)  # 25%, 43%, 61%, 79% along path
        s_position = min(path_fraction * path_length, path_length - 0.1)

        # Get path point at this arc length
        path_x_at_s = float(ref_path.x_spline(s_position))
        path_y_at_s = float(ref_path.y_spline(s_position))

        # Get path tangent to compute normal
        dx_ds = float(ref_path.x_spline.derivative()(s_position))
        dy_ds = float(ref_path.y_spline.derivative()(s_position))
        tangent_norm = np.sqrt(dx_ds**2 + dy_ds**2)

        if tangent_norm > 1e-6:
            normal_x = -dy_ds / tangent_norm
            normal_y = dx_ds / tangent_norm
        else:
            normal_x, normal_y = 0.0, 1.0

        # Alternate obstacles on left and right side of path
        # Use larger offset to give vehicle more room to maneuver
        lateral_offset = 3.0 if i % 2 == 0 else -3.0

        # Position obstacle offset from centerline
        obstacle_x = path_x_at_s + lateral_offset * normal_x
        obstacle_y = path_y_at_s + lateral_offset * normal_y

        # Base velocity toward the path (reduced for more manageable scenarios)
        base_speed = 0.25 + 0.1 * (i % 3)
        velocity_x = -lateral_offset * normal_x * base_speed * 0.4
        velocity_y = -lateral_offset * normal_y * base_speed * 0.4

        # Adjust velocity based on initial mode
        initial_mode = mode_config["initial_mode"]
        if "turn_left" in initial_mode:
            # Add lateral component for turning
            velocity_x += 0.2 * normal_y
            velocity_y -= 0.2 * normal_x
        elif "turn_right" in initial_mode:
            velocity_x -= 0.2 * normal_y
            velocity_y += 0.2 * normal_x
        elif "lane_change" in initial_mode:
            # Add lateral drift
            velocity_x += 0.3 * normal_x
            velocity_y += 0.3 * normal_y

        initial_angle = np.arctan2(velocity_y, velocity_x)

        # Create mode-aware obstacle
        obstacle_config = create_mode_aware_obstacle(
            obstacle_id=i,
            position=np.array([obstacle_x, obstacle_y]),
            velocity=np.array([velocity_x, velocity_y]),
            angle=initial_angle,
            radius=0.35,
            mode_config=mode_config
        )

        obstacle_configs.append(obstacle_config)

        print(f"Obstacle {i}: {mode_config['name']}")
        print(f"  - Position: ({obstacle_x:.2f}, {obstacle_y:.2f})")
        print(f"  - Velocity: ({velocity_x:.2f}, {velocity_y:.2f})")
        print(f"  - Initial mode: {initial_mode}")
        print(f"  - Available modes: {mode_config['available_modes']}")
        print(f"  - Mode switch prob: {mode_config['mode_switch_probability']:.2f}")
        print()

    # Create test configuration
    config = TestConfig(
        reference_path=ref_path,
        objective_module="contouring",
        constraint_modules=["safe_horizon", "contouring"],
        vehicle_dynamics="unicycle",
        num_obstacles=num_obstacles,
        obstacle_dynamics=["unicycle"] * num_obstacles,
        obstacle_prediction_types=["gaussian"] * num_obstacles,
        obstacle_configs=obstacle_configs,
        test_name=f"Safe Horizon Adaptive Modes ({weight_type}, {prior_type})",
        duration=duration,
        timestep=dt,
        show_predicted_trajectory=True,
        timeout_seconds=timeout_minutes * 60.0,
        max_consecutive_failures=15,
    )

    # Print test configuration
    print(f"\n{'='*60}")
    print("Test Configuration")
    print(f"{'='*60}")
    print(f"Objective: contouring")
    print(f"Constraints: safe_horizon (adaptive), contouring")
    print(f"Number of obstacles: {num_obstacles}")
    print(f"Mode weight type: {weight_type}")
    print(f"Mode prior type: {prior_type} ({'constant mode per trajectory' if prior_type == 'constant' else 'mode switching per timestep'})")
    print(f"Duration: {duration}s")
    print(f"Timestep: {dt}s")
    print(f"Max steps: {int(duration/dt)}")
    print(f"Timeout: {timeout_minutes} minutes")
    print()

    # Set adaptive mode sampling configuration in the solver config
    # This will be picked up by SafeHorizonConstraint
    if hasattr(framework, 'config'):
        if 'safe_horizon_constraints' not in framework.config:
            framework.config['safe_horizon_constraints'] = {}
        framework.config['safe_horizon_constraints']['enable_adaptive_mode_sampling'] = True
        framework.config['safe_horizon_constraints']['mode_weight_type'] = weight_type
        framework.config['safe_horizon_constraints']['mode_prior_type'] = prior_type
        framework.config['safe_horizon_constraints']['mode_recency_decay'] = 0.9
        # Use more relaxed parameters for practical computation speed
        # This reduces sample size from ~1000 to ~200 scenarios
        framework.config['safe_horizon_constraints']['epsilon_p'] = 0.15  # Higher violation tolerance
        framework.config['safe_horizon_constraints']['beta'] = 0.1  # Lower confidence

    # Run the test
    print(f"\n{'='*60}")
    print("Running Test...")
    print(f"{'='*60}\n")

    result = framework.run_test(config)

    # Report results
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Vehicle states: {len(result.vehicle_states)}")
    print(f"Success: {result.success}")
    print(f"Constraint violations: {sum(result.constraint_violations)}")
    if hasattr(result, 'computation_times') and result.computation_times:
        avg_time = sum(result.computation_times) / len(result.computation_times)
        max_time = max(result.computation_times)
        min_time = min(result.computation_times)
        print(f"Computation time: avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s")
    if hasattr(result, 'output_folder'):
        print(f"Output folder: {result.output_folder}")

    # Print mode-related summary
    print(f"\n{'='*60}")
    print("Mode Sampling Summary")
    print(f"{'='*60}")
    for i, mode_config in enumerate(OBSTACLE_MODE_CONFIGS):
        print(f"Obstacle {i} ({mode_config['name']}): Started with {mode_config['initial_mode']}")

    return result


def test_uniform_weights():
    """Test with uniform mode weights."""
    result = run(duration=15.0, timeout_minutes=10.0, weight_type="uniform")
    assert result.success, "Test with uniform weights should complete successfully"
    return result


def test_frequency_weights():
    """Test with frequency-based mode weights."""
    result = run(duration=15.0, timeout_minutes=10.0, weight_type="frequency")
    assert result.success, "Test with frequency weights should complete successfully"
    return result


def test_recency_weights():
    """Test with recency-based mode weights."""
    result = run(duration=15.0, timeout_minutes=10.0, weight_type="recency")
    assert result.success, "Test with recency weights should complete successfully"
    return result


def test_switching_prior():
    """Test with switching mode prior (mode can change each timestep)."""
    result = run(duration=15.0, timeout_minutes=10.0, prior_type="switching")
    assert result.success, "Test with switching prior should complete successfully"
    return result


def test():
    """Default test entry point."""
    return test_frequency_weights()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Safe Horizon Adaptive Mode Sampling Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python safe_horizon_adaptive_modes_test.py                    # Default: frequency weights
  python safe_horizon_adaptive_modes_test.py --weight uniform   # Uniform weights
  python safe_horizon_adaptive_modes_test.py --weight recency   # Recency-based weights
  python safe_horizon_adaptive_modes_test.py --prior switching  # Mode switching per timestep
        """
    )
    parser.add_argument(
        "--duration", "-d", type=float, default=30.0,
        help="Simulation duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--timeout", "-t", type=float, default=20.0,
        help="Test timeout in minutes (default: 20)"
    )
    parser.add_argument(
        "--weight", "-w", type=str, default="frequency",
        choices=["uniform", "recency", "frequency"],
        help="Mode weight computation type (default: frequency)"
    )
    parser.add_argument(
        "--prior", "-p", type=str, default="constant",
        choices=["constant", "switching"],
        help="Mode prior type (default: constant)"
    )

    args = parser.parse_args()

    run(
        duration=args.duration,
        timeout_minutes=args.timeout,
        weight_type=args.weight,
        prior_type=args.prior
    )
