#!/usr/bin/env python3
"""
Visualize MPC trajectory data and create animated GIF.

Reads trajectory_data.csv and creates an animation showing:
- Ego vehicle trajectory
- Obstacle trajectory
- Goal position
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import csv
import os

def load_trajectory_data(filename):
    """Load trajectory data from CSV file."""
    data = {
        'time': [], 'ego_x': [], 'ego_y': [], 'ego_theta': [], 'ego_v': [],
        'obs_x': [], 'obs_y': [], 'goal_x': [], 'goal_y': [],
        'accel': [], 'steer': []
    }

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data.keys():
                data[key].append(float(row[key]))

    for key in data.keys():
        data[key] = np.array(data[key])

    return data

def create_vehicle_patch(x, y, theta, length=1.0, width=0.5, color='blue'):
    """Create a rectangle patch for vehicle visualization."""
    # Vehicle corners in local frame
    corners = np.array([
        [-length/2, -width/2],
        [length/2, -width/2],
        [length/2, width/2],
        [-length/2, width/2]
    ])

    # Rotation matrix
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    # Transform corners
    corners_global = (R @ corners.T).T + np.array([x, y])

    return plt.Polygon(corners_global, closed=True, facecolor=color,
                       edgecolor='black', alpha=0.7)

def create_animation(data, output_filename='mpc_demo.gif'):
    """Create animated GIF of the trajectory."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Setup main trajectory plot
    ax1.set_xlim(-2, 14)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('Adaptive Scenario-Based MPC')
    ax1.grid(True, alpha=0.3)

    # Goal marker
    goal_x, goal_y = data['goal_x'][0], data['goal_y'][0]
    ax1.plot(goal_x, goal_y, 'g*', markersize=20, label='Goal')
    ax1.add_patch(plt.Circle((goal_x, goal_y), 0.5, color='green', alpha=0.2))

    # Initialize elements
    ego_trail, = ax1.plot([], [], 'b-', linewidth=2, alpha=0.5, label='Ego trajectory')
    obs_trail, = ax1.plot([], [], 'r--', linewidth=1, alpha=0.5, label='Obstacle trajectory')

    ego_patch = create_vehicle_patch(0, 0, 0, color='blue')
    obs_patch = plt.Circle((0, 0), 0.5, color='red', alpha=0.7)
    ax1.add_patch(ego_patch)
    ax1.add_patch(obs_patch)

    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                         verticalalignment='top', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.legend(loc='upper right')

    # Setup velocity/control plot
    ax2.set_xlim(0, data['time'][-1])
    ax2.set_ylim(-3, 4)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Value')
    ax2.set_title('Velocity and Control Inputs')
    ax2.grid(True, alpha=0.3)

    vel_line, = ax2.plot([], [], 'b-', linewidth=2, label='Velocity [m/s]')
    accel_line, = ax2.plot([], [], 'g-', linewidth=1.5, label='Acceleration [m/s²]')
    steer_line, = ax2.plot([], [], 'r-', linewidth=1.5, label='Steering [rad/s]')
    time_marker = ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)

    ax2.legend(loc='upper right')

    def init():
        ego_trail.set_data([], [])
        obs_trail.set_data([], [])
        vel_line.set_data([], [])
        accel_line.set_data([], [])
        steer_line.set_data([], [])
        return ego_trail, obs_trail, vel_line, accel_line, steer_line

    # Store patches that need updating
    patches_to_update = {'ego': None}

    def update(frame):
        nonlocal patches_to_update

        # Update trails
        ego_trail.set_data(data['ego_x'][:frame+1], data['ego_y'][:frame+1])
        obs_trail.set_data(data['obs_x'][:frame+1], data['obs_y'][:frame+1])

        # Update ego vehicle - remove old patch if exists
        if patches_to_update['ego'] is not None:
            try:
                patches_to_update['ego'].remove()
            except:
                pass

        new_ego_patch = create_vehicle_patch(
            data['ego_x'][frame], data['ego_y'][frame],
            data['ego_theta'][frame], color='blue'
        )
        ax1.add_patch(new_ego_patch)
        patches_to_update['ego'] = new_ego_patch

        # Update obstacle
        obs_patch.center = (data['obs_x'][frame], data['obs_y'][frame])

        # Update time text
        dist_to_goal = np.sqrt((data['ego_x'][frame] - goal_x)**2 +
                               (data['ego_y'][frame] - goal_y)**2)
        time_text.set_text(f'Time: {data["time"][frame]:.1f}s\n'
                          f'Velocity: {data["ego_v"][frame]:.2f} m/s\n'
                          f'Dist to goal: {dist_to_goal:.2f} m')

        # Update velocity/control plot
        vel_line.set_data(data['time'][:frame+1], data['ego_v'][:frame+1])
        accel_line.set_data(data['time'][:frame+1], data['accel'][:frame+1])
        steer_line.set_data(data['time'][:frame+1], data['steer'][:frame+1])
        time_marker.set_xdata([data['time'][frame]])

        return ego_trail, obs_trail, vel_line, accel_line, steer_line, time_marker

    # Create animation
    num_frames = len(data['time'])
    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                        blit=False, interval=100, repeat=True)

    # Save as GIF
    print(f"Creating animation with {num_frames} frames...")
    writer = PillowWriter(fps=10)
    anim.save(output_filename, writer=writer)
    print(f"Animation saved to: {output_filename}")

    plt.close()

    # Also create a static summary plot
    create_summary_plot(data)

def create_summary_plot(data):
    """Create static summary plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Trajectory plot
    ax1 = axes[0, 0]
    ax1.plot(data['ego_x'], data['ego_y'], 'b-', linewidth=2, label='Ego')
    ax1.plot(data['obs_x'], data['obs_y'], 'r--', linewidth=2, label='Obstacle')
    ax1.plot(data['goal_x'][0], data['goal_y'][0], 'g*', markersize=15, label='Goal')
    ax1.plot(data['ego_x'][0], data['ego_y'][0], 'bo', markersize=10, label='Start')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Velocity plot
    ax2 = axes[0, 1]
    ax2.plot(data['time'], data['ego_v'], 'b-', linewidth=2)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Velocity [m/s]')
    ax2.set_title('Ego Velocity')
    ax2.grid(True, alpha=0.3)

    # Control inputs
    ax3 = axes[1, 0]
    ax3.plot(data['time'], data['accel'], 'g-', linewidth=2, label='Acceleration')
    ax3.plot(data['time'], data['steer'], 'r-', linewidth=2, label='Steering')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Control Input')
    ax3.set_title('Control Inputs')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Distance to goal and obstacle
    ax4 = axes[1, 1]
    dist_to_goal = np.sqrt((data['ego_x'] - data['goal_x'])**2 +
                           (data['ego_y'] - data['goal_y'])**2)
    dist_to_obs = np.sqrt((data['ego_x'] - data['obs_x'])**2 +
                          (data['ego_y'] - data['obs_y'])**2)
    ax4.plot(data['time'], dist_to_goal, 'g-', linewidth=2, label='To Goal')
    ax4.plot(data['time'], dist_to_obs, 'r-', linewidth=2, label='To Obstacle')
    ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Safety radius')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Distance [m]')
    ax4.set_title('Distances')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trajectory_summary.png', dpi=150)
    print("Summary plot saved to: trajectory_summary.png")
    plt.close()

if __name__ == '__main__':
    data_file = 'trajectory_data.csv'

    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        print("Please run the C++ demo first: ./demo_visualization")
        exit(1)

    print("Loading trajectory data...")
    data = load_trajectory_data(data_file)
    print(f"Loaded {len(data['time'])} timesteps")

    create_animation(data, 'mpc_demo.gif')
