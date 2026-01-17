"""
Basic example demonstrating Adaptive Scenario-Based MPC.

This example shows how to:
1. Configure the MPC controller
2. Initialize obstacles with mode models
3. Run the MPC loop
4. Visualize results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
sys.path.insert(0, '/home/stephen/PyMPC')

from scenario_mpc import (
    AdaptiveScenarioMPC,
    ScenarioMPCConfig,
    EgoState,
    ObstacleState,
    WeightType,
)


def run_simulation():
    """Run a simple simulation demonstrating the MPC controller."""

    # Configure the MPC
    config = ScenarioMPCConfig(
        horizon=15,
        dt=0.1,
        num_scenarios=10,
        confidence_level=0.95,
        weight_type=WeightType.FREQUENCY,
        ego_radius=0.5,
        obstacle_radius=0.5,
        goal_weight=5.0,
        velocity_weight=0.5,
        acceleration_weight=0.1,
        steering_weight=0.1,
    )

    # Create controller
    controller = AdaptiveScenarioMPC(config)

    # Initial states
    ego_state = EgoState(x=0, y=0, theta=0, v=0.5)
    goal = np.array([15, 0])

    # Obstacles
    obstacles = {
        0: ObstacleState(x=7, y=0.5, vx=-0.5, vy=0),   # Coming towards ego
        1: ObstacleState(x=5, y=-2, vx=0, vy=0.3),     # Moving upward
    }

    # Initialize obstacles with some mode observations
    for obs_id in obstacles:
        controller.initialize_obstacle(obs_id)
        # Simulate some historical observations
        for t in range(5):
            controller.update_mode_observation(obs_id, "constant_velocity", timestep=t)

    # Simulation loop
    sim_steps = 100
    ego_trajectory = [ego_state]
    obstacle_trajectories = {obs_id: [obs] for obs_id, obs in obstacles.items()}

    print("Starting simulation...")
    print(f"Goal: {goal}")
    print(f"Initial ego state: x={ego_state.x:.2f}, y={ego_state.y:.2f}")

    for step in range(sim_steps):
        # Solve MPC
        result = controller.solve(ego_state, obstacles, goal, reference_velocity=1.5)

        if result.success:
            # Apply first control
            first_input = result.first_input
            if first_input is not None:
                # Propagate ego state
                ego_state = controller.ego_dynamics.propagate(ego_state, first_input)
        else:
            print(f"Step {step}: Solver failed, using fallback")
            # Simple braking fallback
            ego_state = EgoState(
                x=ego_state.x + ego_state.v * np.cos(ego_state.theta) * config.dt,
                y=ego_state.y + ego_state.v * np.sin(ego_state.theta) * config.dt,
                theta=ego_state.theta,
                v=max(0, ego_state.v - 1.0 * config.dt)
            )

        ego_trajectory.append(ego_state)

        # Update obstacles (simple constant velocity for demo)
        for obs_id, obs in obstacles.items():
            new_obs = ObstacleState(
                x=obs.x + obs.vx * config.dt,
                y=obs.y + obs.vy * config.dt,
                vx=obs.vx,
                vy=obs.vy
            )
            obstacles[obs_id] = new_obs
            obstacle_trajectories[obs_id].append(new_obs)

            # Update mode observation
            controller.update_mode_observation(obs_id, "constant_velocity", timestep=step + 5)

        # Check if goal reached
        dist_to_goal = np.linalg.norm(ego_state.position() - goal)
        if dist_to_goal < 0.5:
            print(f"Goal reached at step {step}!")
            break

        # Progress update
        if step % 20 == 0:
            print(f"Step {step}: ego at ({ego_state.x:.2f}, {ego_state.y:.2f}), "
                  f"dist to goal: {dist_to_goal:.2f}")

    # Print statistics
    stats = controller.get_statistics()
    print(f"\nSimulation complete!")
    print(f"Total iterations: {stats['iteration_count']}")
    print(f"Avg solve time: {stats['avg_solve_time']*1000:.1f} ms")

    return ego_trajectory, obstacle_trajectories, goal


def visualize_results(ego_trajectory, obstacle_trajectories, goal):
    """Visualize simulation results."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot ego trajectory
    ego_x = [s.x for s in ego_trajectory]
    ego_y = [s.y for s in ego_trajectory]
    ax.plot(ego_x, ego_y, 'b-', linewidth=2, label='Ego trajectory')
    ax.plot(ego_x[0], ego_y[0], 'bo', markersize=10, label='Ego start')
    ax.plot(ego_x[-1], ego_y[-1], 'bs', markersize=10, label='Ego end')

    # Plot obstacle trajectories
    colors = ['r', 'orange', 'purple', 'brown']
    for i, (obs_id, traj) in enumerate(obstacle_trajectories.items()):
        obs_x = [s.x for s in traj]
        obs_y = [s.y for s in traj]
        color = colors[i % len(colors)]
        ax.plot(obs_x, obs_y, f'{color}--', linewidth=1.5, alpha=0.7,
                label=f'Obstacle {obs_id}')
        ax.plot(obs_x[0], obs_y[0], f'{color}o', markersize=8)
        ax.plot(obs_x[-1], obs_y[-1], f'{color}s', markersize=8)

    # Plot goal
    ax.plot(goal[0], goal[1], 'g*', markersize=15, label='Goal')
    goal_circle = Circle(goal, 0.5, fill=False, color='green', linestyle='--')
    ax.add_patch(goal_circle)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Adaptive Scenario-Based MPC Simulation')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim(-2, 18)
    ax.set_ylim(-5, 5)

    plt.tight_layout()
    plt.savefig('/home/stephen/PyMPC/scenario_mpc/examples/simulation_result.png', dpi=150)
    plt.show()
    print("Plot saved to simulation_result.png")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Adaptive Scenario-Based MPC Example")
    print("Following guide.md mathematical formulation")
    print("=" * 60)
    print()

    # Run simulation
    ego_traj, obs_trajs, goal = run_simulation()

    # Visualize
    try:
        visualize_results(ego_traj, obs_trajs, goal)
    except Exception as e:
        print(f"Visualization skipped: {e}")


if __name__ == "__main__":
    main()
