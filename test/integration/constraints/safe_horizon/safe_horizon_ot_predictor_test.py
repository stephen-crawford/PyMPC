"""
Safe Horizon Constraint Integration Test with Optimal Transport Predictor.

This test demonstrates the integration of the Optimal Transport (OT) predictor
for learning obstacle dynamics within the adaptive safe horizon MPCC framework.

Key features tested:
1. Online learning of obstacle dynamics from observations
2. Wasserstein-based mode weight computation
3. OT barycenter-based trajectory prediction
4. Adaptive uncertainty quantification

Test scenario:
- Vehicle follows a reference path
- Multiple dynamic obstacles with different behavior modes
- OT predictor learns obstacle dynamics online
- Safe Horizon constraints use OT-based predictions for collision avoidance
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

from modules.constraints.scenario_utils.optimal_transport_predictor import (
    OptimalTransportPredictor,
    OTWeightType,
    create_ot_predictor_with_standard_modes,
    wasserstein_distance,
    EmpiricalDistribution,
)


def _import_mpcc_deps():
    """Lazy import of MPCC dependencies (requires CasADi)."""
    from test.integration.integration_test_framework import (
        IntegrationTestFramework, TestConfig, create_reference_path
    )
    from planning.obstacle_manager import create_unicycle_obstacle
    from planning.types import (
        PredictionType, DynamicObstacle, Prediction, PredictionStep
    )
    return (IntegrationTestFramework, TestConfig, create_reference_path,
            create_unicycle_obstacle, PredictionType, DynamicObstacle,
            Prediction, PredictionStep)


# =============================================================================
# Test Configuration
# =============================================================================

OBSTACLE_CONFIGS = [
    {
        "name": "Constant Velocity Obstacle",
        "initial_mode": "constant_velocity",
        "available_modes": ["constant_velocity", "decelerating", "accelerating"],
        "true_behavior": "constant_velocity",  # Ground truth for validation
        "color": "blue"
    },
    {
        "name": "Turning Obstacle",
        "initial_mode": "turn_left",
        "available_modes": ["turn_left", "turn_right", "constant_velocity"],
        "true_behavior": "turn_left",
        "color": "green"
    },
    {
        "name": "Decelerating Obstacle",
        "initial_mode": "constant_velocity",  # Start with wrong mode to test learning
        "available_modes": ["decelerating", "constant_velocity", "accelerating"],
        "true_behavior": "decelerating",
        "color": "orange"
    },
    {
        "name": "Lane Changing Obstacle",
        "initial_mode": "constant_velocity",  # Start with wrong mode to test learning
        "available_modes": ["lane_change_left", "lane_change_right", "constant_velocity"],
        "true_behavior": "lane_change_left",
        "color": "red"
    },
]


class OTIntegrationTester:
    """
    Integration tester for Optimal Transport predictor.

    Simulates obstacle motion, feeds observations to OT predictor,
    and validates learning and prediction quality.
    """

    def __init__(self, dt: float = 0.1, horizon: int = 10, sight_radius: float = 5.0):
        self.dt = dt
        self.horizon = horizon
        self.sight_radius = sight_radius

        # Create OT predictor with standard modes
        self.ot_predictor = create_ot_predictor_with_standard_modes(
            dt=dt,
            base_speed=0.5,
            buffer_size=200,
            min_samples_for_ot=10,
            uncertainty_scale=1.0,
            weight_type=OTWeightType.WASSERSTEIN
        )

        # Ground truth obstacle states
        self.obstacle_states: Dict[int, Dict] = {}

        # Metrics tracking
        self.prediction_errors: Dict[int, List[float]] = {}
        self.mode_weight_history: Dict[int, List[Dict[str, float]]] = {}
        self.learned_mode_history: Dict[int, List[str]] = {}

        # Ego position for sight radius filtering
        self.ego_position = np.array([0.0, 0.0])

    def initialize_obstacle(
        self,
        obstacle_id: int,
        position: np.ndarray,
        velocity: np.ndarray,
        true_mode: str
    ) -> None:
        """Initialize ground truth obstacle state."""
        self.obstacle_states[obstacle_id] = {
            "position": np.array(position),
            "velocity": np.array(velocity),
            "true_mode": true_mode
        }
        self.prediction_errors[obstacle_id] = []
        self.mode_weight_history[obstacle_id] = []
        self.learned_mode_history[obstacle_id] = []

    def simulate_obstacle_step(self, obstacle_id: int) -> np.ndarray:
        """
        Simulate one step of obstacle motion based on true mode.

        Returns new position.
        """
        state = self.obstacle_states[obstacle_id]
        pos = state["position"]
        vel = state["velocity"]
        mode = state["true_mode"]

        # Apply mode-specific dynamics
        if mode == "constant_velocity":
            # No change to velocity
            pass
        elif mode == "decelerating":
            # Reduce velocity magnitude
            vel = vel * 0.95
        elif mode == "accelerating":
            # Increase velocity magnitude
            vel = vel * 1.05
        elif mode == "turn_left":
            # Rotate velocity counterclockwise
            omega = 0.3 * self.dt  # Turn rate
            cos_w, sin_w = np.cos(omega), np.sin(omega)
            vel = np.array([vel[0] * cos_w - vel[1] * sin_w,
                          vel[0] * sin_w + vel[1] * cos_w])
        elif mode == "turn_right":
            # Rotate velocity clockwise
            omega = -0.3 * self.dt
            cos_w, sin_w = np.cos(omega), np.sin(omega)
            vel = np.array([vel[0] * cos_w - vel[1] * sin_w,
                          vel[0] * sin_w + vel[1] * cos_w])
        elif mode == "lane_change_left":
            # Add lateral velocity
            heading = np.arctan2(vel[1], vel[0])
            lateral = np.array([-np.sin(heading), np.cos(heading)]) * 0.1 * self.dt
            vel = vel + lateral
        elif mode == "lane_change_right":
            heading = np.arctan2(vel[1], vel[0])
            lateral = np.array([np.sin(heading), -np.cos(heading)]) * 0.1 * self.dt
            vel = vel + lateral

        # Update state
        new_pos = pos + vel * self.dt
        state["position"] = new_pos
        state["velocity"] = vel

        return new_pos

    def observe_and_predict(
        self,
        obstacle_id: int,
        config: Dict
    ) -> Optional[List]:
        """
        Observe obstacle and generate OT-based prediction.

        Skips observation and prediction if the obstacle is outside the
        ego vehicle's sight radius.

        Returns list of prediction steps, or None if out of sight.
        """
        state = self.obstacle_states[obstacle_id]
        pos = state["position"]
        vel = state["velocity"]

        # Check sight radius
        dist_to_ego = np.linalg.norm(self.ego_position - pos)
        if dist_to_ego > self.sight_radius:
            return None

        # Don't label observations until OT can infer the mode.
        # Labeling with a wrong initial_mode pollutes the learned
        # per-obstacle distributions and prevents correct convergence.
        observed_mode = None

        # After enough observations, use OT to infer mode per obstacle
        if len(self.ot_predictor.trajectory_buffers.get(obstacle_id, [])) > 15:
            # Use OT to infer mode
            weights = self.ot_predictor.compute_mode_weights(
                obstacle_id,
                config["available_modes"]
            )
            # Record weights
            self.mode_weight_history[obstacle_id].append(weights.copy())

            # Use highest weight mode as observation
            observed_mode = max(weights, key=weights.get)
            self.learned_mode_history[obstacle_id].append(observed_mode)

        self.ot_predictor.observe(obstacle_id, pos, mode_id=observed_mode)

        # Generate prediction
        predictions = self.ot_predictor.predict_trajectory(
            obstacle_id=obstacle_id,
            current_position=pos,
            current_velocity=vel,
            horizon=self.horizon
        )

        return predictions

    def compute_prediction_error_step(
        self,
        obstacle_id: int,
        predicted: List,
        steps_ahead: int = 5
    ) -> float:
        """
        Compute prediction error by simulating forward and comparing.
        """
        if steps_ahead > len(predicted) - 1:
            steps_ahead = len(predicted) - 1

        # Save current state
        current_state = {
            "position": self.obstacle_states[obstacle_id]["position"].copy(),
            "velocity": self.obstacle_states[obstacle_id]["velocity"].copy(),
            "true_mode": self.obstacle_states[obstacle_id]["true_mode"]
        }

        # Simulate forward
        actual_positions = [current_state["position"].copy()]
        for _ in range(steps_ahead):
            self.simulate_obstacle_step(obstacle_id)
            actual_positions.append(
                self.obstacle_states[obstacle_id]["position"].copy()
            )

        # Restore state
        self.obstacle_states[obstacle_id] = current_state

        # Compare predicted vs actual
        predicted_positions = [p.position[:2] for p in predicted[:steps_ahead + 1]]

        # Compute Wasserstein distance
        pred_dist = EmpiricalDistribution.from_samples(np.array(predicted_positions))
        actual_dist = EmpiricalDistribution.from_samples(np.array(actual_positions))

        error = wasserstein_distance(pred_dist, actual_dist)
        self.prediction_errors[obstacle_id].append(error)

        return error

    def advance_timestep(self) -> None:
        """Advance time for all components."""
        self.ot_predictor.advance_timestep()


def create_ot_aware_obstacles(
    ref_path,
    configs: List[Dict]
) -> List[Dict]:
    """Create obstacle configurations for OT integration test."""
    (_, _, _, create_unicycle_obstacle, PredictionType,
     _, _, _) = _import_mpcc_deps()

    obstacle_configs = []
    path_length = ref_path.length

    for i, config in enumerate(configs):
        # Place obstacles along the path
        path_fraction = 0.2 + (i * 0.2)
        s_position = min(path_fraction * path_length, path_length - 0.1)

        # Get path point
        path_x = float(ref_path.x_spline(s_position))
        path_y = float(ref_path.y_spline(s_position))

        # Get path tangent for normal computation
        dx_ds = float(ref_path.x_spline.derivative()(s_position))
        dy_ds = float(ref_path.y_spline.derivative()(s_position))
        tangent_norm = np.sqrt(dx_ds**2 + dy_ds**2)

        if tangent_norm > 1e-6:
            normal_x = -dy_ds / tangent_norm
            normal_y = dx_ds / tangent_norm
        else:
            normal_x, normal_y = 0.0, 1.0

        # Alternate sides
        lateral_offset = 2.5 if i % 2 == 0 else -2.5

        obstacle_x = path_x + lateral_offset * normal_x
        obstacle_y = path_y + lateral_offset * normal_y

        # Velocity toward path center
        base_speed = 0.3
        velocity_x = -lateral_offset * normal_x * base_speed * 0.5
        velocity_y = -lateral_offset * normal_y * base_speed * 0.5

        # Create obstacle config
        obs_config = create_unicycle_obstacle(
            obstacle_id=i,
            position=np.array([obstacle_x, obstacle_y]),
            velocity=np.array([velocity_x, velocity_y]),
            angle=np.arctan2(velocity_y, velocity_x),
            radius=0.35,
            behavior="path_intersect"
        )

        obs_config.prediction_type = PredictionType.GAUSSIAN
        obs_config.uncertainty_params = {
            'position_std': 0.2,
            'uncertainty_growth': 0.1
        }
        obs_config.initial_mode = config["initial_mode"]
        obs_config.current_mode = config["initial_mode"]
        obs_config.available_modes = config["available_modes"]
        obs_config.mode_name = config["name"]

        obstacle_configs.append({
            "config": obs_config,
            "meta": config,
            "position": np.array([obstacle_x, obstacle_y]),
            "velocity": np.array([velocity_x, velocity_y])
        })

    return obstacle_configs


def visualize_ot_learning(
    tester: OTIntegrationTester,
    configs: List[Dict],
    dt: float,
    output_path: Optional[str] = None
) -> str:
    """
    Create visualization of OT mode weight learning with a sub-figure per obstacle.

    Shows how each obstacle's mode weights evolve over time as the OT predictor
    learns the true dynamics from observations.

    Args:
        tester: OTIntegrationTester with completed simulation data
        configs: OBSTACLE_CONFIGS list
        dt: Timestep used in simulation
        output_path: Where to save the figure (auto-generated if None)

    Returns:
        Path to the saved figure
    """
    num_obstacles = len(configs)
    mode_colors = {
        'constant_velocity': '#1f77b4',
        'decelerating': '#e377c2',
        'accelerating': '#ff7f0e',
        'turn_left': '#2ca02c',
        'turn_right': '#17becf',
        'lane_change_left': '#bcbd22',
        'lane_change_right': '#7f7f7f',
    }
    obstacle_colors = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd']

    fig = plt.figure(figsize=(6 * num_obstacles, 10))
    fig.suptitle(
        'Optimal Transport Mode Weight Learning Per Obstacle',
        fontsize=16, fontweight='bold', y=0.98
    )

    gs = fig.add_gridspec(
        3, num_obstacles, hspace=0.45, wspace=0.3,
        top=0.92, bottom=0.06, left=0.06, right=0.96
    )

    # --- Row 1: Mode weights per obstacle ---
    for obs_id in range(num_obstacles):
        ax = fig.add_subplot(gs[0, obs_id])
        config = configs[obs_id]
        true_mode = config['true_behavior']
        initial_mode = config['initial_mode']

        weight_history = tester.mode_weight_history.get(obs_id, [])

        if weight_history:
            # Each entry is a Dict[str, float] of mode weights
            # The weight history starts after min_samples_for_ot observations
            n_entries = len(weight_history)
            # Timestep indices where weight history starts (after 15 observations)
            time_offset = 16  # observations 0..15 -> weight recording starts at step 16
            time_axis = np.arange(n_entries) * dt + time_offset * dt

            # Collect all modes that appear
            all_modes = set()
            for w in weight_history:
                all_modes.update(w.keys())

            for mode in sorted(all_modes):
                weights = [w.get(mode, 0.0) for w in weight_history]
                color = mode_colors.get(mode, 'gray')
                linewidth = 3.0 if mode == true_mode else 1.5
                alpha = 1.0 if mode == true_mode else 0.6
                linestyle = '-' if mode == true_mode else '--'
                label = mode.replace('_', ' ')
                if mode == true_mode:
                    label += ' (TRUE)'
                ax.plot(time_axis, weights, color=color, linewidth=linewidth,
                        alpha=alpha, linestyle=linestyle, label=label)

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Time [s]', fontsize=9)
        if obs_id == 0:
            ax.set_ylabel('Mode Weight', fontsize=10)
        ax.set_title(
            f'Obs {obs_id}: {config["name"]}\n'
            f'True: {true_mode} | Init: {initial_mode}',
            fontsize=10, fontweight='bold',
            color=obstacle_colors[obs_id % len(obstacle_colors)]
        )
        ax.legend(loc='best', fontsize=7, ncol=1)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)

    # --- Row 2: Prediction error per obstacle ---
    for obs_id in range(num_obstacles):
        ax = fig.add_subplot(gs[1, obs_id])
        config = configs[obs_id]

        errors = tester.prediction_errors.get(obs_id, [])
        if errors:
            # Prediction errors are recorded every 10 steps starting from step 20
            n_errors = len(errors)
            error_times = []
            step_start = 21  # first error at step 21 (step > 20 and step % 10 == 0, so step=30)
            # errors recorded at steps: 30, 40, 50, ... (every 10 steps after step 20)
            for idx in range(n_errors):
                error_times.append((30 + idx * 10) * dt)

            ax.plot(error_times, errors, '-',
                    color=obstacle_colors[obs_id % len(obstacle_colors)],
                    linewidth=2, alpha=0.8)
            ax.fill_between(error_times, 0, errors,
                           color=obstacle_colors[obs_id % len(obstacle_colors)],
                           alpha=0.15)

            # Add trend line
            if len(errors) > 5:
                window = min(10, len(errors) // 3)
                smoothed = np.convolve(errors, np.ones(window)/window, mode='valid')
                smooth_times = error_times[window-1:]
                ax.plot(smooth_times, smoothed, 'k-', linewidth=2, alpha=0.5,
                        label='Trend')

            # Annotate final error
            ax.annotate(
                f'Final: {errors[-1]:.3f}',
                xy=(error_times[-1], errors[-1]),
                fontsize=8, fontweight='bold',
                ha='right', va='bottom',
                color=obstacle_colors[obs_id % len(obstacle_colors)]
            )

        ax.set_xlabel('Time [s]', fontsize=9)
        if obs_id == 0:
            ax.set_ylabel('Wasserstein Error', fontsize=10)
        ax.set_title(f'Obs {obs_id}: Prediction Error', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        if errors:
            ax.legend(fontsize=7)

    # --- Row 3: Learned mode convergence (bar chart of final weights) ---
    for obs_id in range(num_obstacles):
        ax = fig.add_subplot(gs[2, obs_id])
        config = configs[obs_id]
        true_mode = config['true_behavior']

        weight_history = tester.mode_weight_history.get(obs_id, [])

        if weight_history:
            # Use the average of last 20% of weights for final assessment
            n_final = max(1, len(weight_history) // 5)
            final_weights = weight_history[-n_final:]

            # Average the final weights
            avg_weights = {}
            all_modes = set()
            for w in final_weights:
                all_modes.update(w.keys())
            for mode in all_modes:
                avg_weights[mode] = np.mean([w.get(mode, 0.0) for w in final_weights])

            modes_sorted = sorted(avg_weights.keys())
            values = [avg_weights[m] for m in modes_sorted]
            colors = []
            for m in modes_sorted:
                if m == true_mode:
                    colors.append('#2ecc71')  # green for true mode
                else:
                    colors.append(mode_colors.get(m, 'gray'))

            bars = ax.bar(range(len(modes_sorted)), values, color=colors,
                         edgecolor='black', linewidth=0.5, alpha=0.85)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                if val > 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=8,
                           fontweight='bold')

            ax.set_xticks(range(len(modes_sorted)))
            ax.set_xticklabels(
                [m.replace('_', '\n') for m in modes_sorted],
                fontsize=7, rotation=0
            )

            # Mark the true mode
            for i, m in enumerate(modes_sorted):
                if m == true_mode:
                    ax.get_xticklabels()[i].set_fontweight('bold')
                    ax.get_xticklabels()[i].set_color('#2ecc71')

            # Check if learning succeeded
            learned_mode = max(avg_weights, key=avg_weights.get) if avg_weights else None
            success = learned_mode == true_mode
            status = 'LEARNED' if success else 'LEARNING...'
            status_color = '#2ecc71' if success else '#e74c3c'
            ax.set_title(f'Obs {obs_id}: Final Weights [{status}]',
                        fontsize=10, fontweight='bold', color=status_color)
        else:
            ax.text(0.5, 0.5, 'No weight data\n(insufficient observations)',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=10, color='gray')
            ax.set_title(f'Obs {obs_id}: Final Weights', fontsize=10)

        ax.set_ylim(0, 1.15)
        if obs_id == 0:
            ax.set_ylabel('Avg Weight', fontsize=10)
        ax.grid(True, alpha=0.2, axis='y')

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__),
            'ot_learning_per_obstacle.png'
        )

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()

    return output_path


def run_ot_standalone_test(
    duration: float = 10.0,
    dt: float = 0.1,
    verbose: bool = True
) -> Dict:
    """
    Run standalone OT predictor test without full MPCC integration.

    This tests the OT predictor's learning and prediction capabilities
    in isolation.
    """
    print("\n" + "=" * 60)
    print("Optimal Transport Predictor Standalone Test")
    print("=" * 60 + "\n")

    sight_radius = 5.0
    tester = OTIntegrationTester(dt=dt, horizon=10, sight_radius=sight_radius)
    num_steps = int(duration / dt)

    # Initialize obstacles
    for i, config in enumerate(OBSTACLE_CONFIGS):
        tester.initialize_obstacle(
            obstacle_id=i,
            position=np.array([float(i) * 2.0, 0.0]),
            velocity=np.array([0.5, 0.1]),
            true_mode=config["true_behavior"]
        )
        if verbose:
            print(f"Obstacle {i}: {config['name']}")
            print(f"  True mode: {config['true_behavior']}")
            print(f"  Initial mode guess: {config['initial_mode']}")

    if verbose:
        print(f"\n  Sight radius: {sight_radius:.1f}m")

    print("\n" + "-" * 40)
    print("Running simulation...")
    print("-" * 40)

    start_time = time.time()

    # Simulation loop
    for step in range(num_steps):
        # Advance ego position along x-axis
        tester.ego_position = tester.ego_position + np.array([0.5 * dt, 0.0])

        for i, config in enumerate(OBSTACLE_CONFIGS):
            # Simulate obstacle motion
            tester.simulate_obstacle_step(i)

            # Observe and predict (returns None if out of sight)
            predictions = tester.observe_and_predict(i, config)

            # Compute prediction error every 10 steps
            if predictions is not None and step > 20 and step % 10 == 0:
                error = tester.compute_prediction_error_step(i, predictions)

        tester.advance_timestep()

        # Progress update
        if verbose and (step + 1) % 20 == 0:
            print(f"Step {step + 1}/{num_steps}")

    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.2f}s")

    # Compute results
    results = {
        "success": True,
        "duration": elapsed,
        "num_steps": num_steps,
        "obstacles": {}
    }

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    for i, config in enumerate(OBSTACLE_CONFIGS):
        obs_results = {
            "name": config["name"],
            "true_mode": config["true_behavior"],
            "initial_mode": config["initial_mode"],
        }

        # Analyze prediction errors
        errors = tester.prediction_errors[i]
        if errors:
            obs_results["mean_prediction_error"] = np.mean(errors)
            obs_results["max_prediction_error"] = np.max(errors)
            obs_results["final_prediction_error"] = errors[-1]

        # Analyze mode learning
        learned_modes = tester.learned_mode_history[i]
        if learned_modes:
            # Count mode occurrences in second half (after learning)
            second_half = learned_modes[len(learned_modes)//2:]
            mode_counts = {}
            for m in second_half:
                mode_counts[m] = mode_counts.get(m, 0) + 1

            # Dominant learned mode
            if mode_counts:
                dominant_mode = max(mode_counts, key=mode_counts.get)
                obs_results["learned_mode"] = dominant_mode
                obs_results["mode_accuracy"] = mode_counts.get(
                    config["true_behavior"], 0
                ) / len(second_half)

        # Check if learning was successful
        mode_correct = obs_results.get("learned_mode") == config["true_behavior"]
        obs_results["learning_success"] = mode_correct

        results["obstacles"][i] = obs_results

        if verbose:
            print(f"\nObstacle {i}: {config['name']}")
            print(f"  True mode: {config['true_behavior']}")
            print(f"  Learned mode: {obs_results.get('learned_mode', 'N/A')}")
            print(f"  Mode accuracy: {obs_results.get('mode_accuracy', 0):.1%}")
            print(f"  Learning success: {'YES' if mode_correct else 'NO'}")
            if errors:
                print(f"  Mean prediction error: {obs_results['mean_prediction_error']:.3f}")
                print(f"  Final prediction error: {obs_results['final_prediction_error']:.3f}")

    # Overall success check
    learning_successes = sum(
        1 for obs in results["obstacles"].values()
        if obs.get("learning_success", False)
    )
    results["learning_success_rate"] = learning_successes / len(OBSTACLE_CONFIGS)

    print("\n" + "-" * 40)
    print(f"Overall Learning Success Rate: {results['learning_success_rate']:.1%}")
    print("-" * 40)

    # Generate per-obstacle visualization
    try:
        fig_path = visualize_ot_learning(tester, OBSTACLE_CONFIGS, dt)
        results["visualization_path"] = fig_path
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")

    return results


def run_ot_mpcc_integration_test(
    dt: float = 0.1,
    duration: float = 20.0,
    timeout_minutes: float = 15.0,
    verbose: bool = True
) -> Dict:
    """
    Run full MPCC integration test with OT predictor.

    This tests the OT predictor integrated with the safe horizon
    constraint framework.
    """
    print("\n" + "=" * 60)
    print("OT Predictor + Safe Horizon MPCC Integration Test")
    print("=" * 60 + "\n")

    (IntegrationTestFramework, TestConfig, create_reference_path,
     _, _, _, _, _) = _import_mpcc_deps()

    num_obstacles = len(OBSTACLE_CONFIGS)

    framework = IntegrationTestFramework()

    # Create reference path
    ref_path = create_reference_path("s_curve", length=20.0)

    # Create obstacles
    obstacle_configs_full = create_ot_aware_obstacles(ref_path, OBSTACLE_CONFIGS)
    obstacle_configs = [oc["config"] for oc in obstacle_configs_full]

    for i, oc in enumerate(obstacle_configs_full):
        if verbose:
            print(f"Obstacle {i}: {oc['meta']['name']}")
            print(f"  Position: ({oc['position'][0]:.2f}, {oc['position'][1]:.2f})")
            print(f"  True behavior: {oc['meta']['true_behavior']}")

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
        test_name="OT Predictor Integration Test",
        duration=duration,
        timestep=dt,
        show_predicted_trajectory=True,
        timeout_seconds=timeout_minutes * 60.0,
        max_consecutive_failures=10,
    )

    # Configure safe horizon with adaptive mode sampling
    if hasattr(framework, 'config'):
        if 'safe_horizon_constraints' not in framework.config:
            framework.config['safe_horizon_constraints'] = {}
        framework.config['safe_horizon_constraints']['enable_adaptive_mode_sampling'] = True
        framework.config['safe_horizon_constraints']['mode_weight_type'] = "frequency"
        framework.config['safe_horizon_constraints']['epsilon_p'] = 0.15
        framework.config['safe_horizon_constraints']['beta'] = 0.1
        framework.config['safe_horizon_constraints']['num_scenarios'] = 200

    print("\n" + "-" * 40)
    print("Running MPCC with OT-based predictions...")
    print("-" * 40 + "\n")

    # Run test
    result = framework.run_test(config)

    print("\n" + "=" * 60)
    print("MPCC Integration Test Results")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Vehicle states: {len(result.vehicle_states)}")
    print(f"Constraint violations: {sum(result.constraint_violations)}")

    if hasattr(result, 'computation_times') and result.computation_times:
        avg_time = sum(result.computation_times) / len(result.computation_times)
        print(f"Avg computation time: {avg_time:.3f}s")

    return {"success": result.success, "result": result}


def run(
    test_type: str = "standalone",
    dt: float = 0.1,
    duration: float = 10.0,
    timeout_minutes: float = 15.0
):
    """
    Run OT predictor integration test.

    Args:
        test_type: "standalone" for OT-only test, "mpcc" for full integration
        dt: Timestep
        duration: Test duration
        timeout_minutes: Timeout for MPCC test
    """
    if test_type == "standalone":
        return run_ot_standalone_test(duration=duration, dt=dt)
    elif test_type == "mpcc":
        return run_ot_mpcc_integration_test(
            dt=dt, duration=duration, timeout_minutes=timeout_minutes
        )
    else:
        raise ValueError(f"Unknown test type: {test_type}")


def test_standalone():
    """Standalone test entry point."""
    result = run_ot_standalone_test(duration=10.0)
    assert result["learning_success_rate"] >= 0.5, "OT learning should succeed for at least half the obstacles"
    return result


def test_mpcc_integration():
    """MPCC integration test entry point."""
    result = run_ot_mpcc_integration_test(duration=15.0, timeout_minutes=10.0)
    assert result["success"], "MPCC integration test should complete successfully"
    return result


def test():
    """Default test entry point."""
    return test_standalone()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimal Transport Predictor Integration Test"
    )
    parser.add_argument(
        "--type", "-t", type=str, default="standalone",
        choices=["standalone", "mpcc"],
        help="Test type: standalone (OT only) or mpcc (full integration)"
    )
    parser.add_argument(
        "--duration", "-d", type=float, default=10.0,
        help="Test duration in seconds"
    )
    parser.add_argument(
        "--timeout", type=float, default=15.0,
        help="Timeout in minutes (for MPCC test)"
    )

    args = parser.parse_args()

    run(
        test_type=args.type,
        duration=args.duration,
        timeout_minutes=args.timeout
    )
