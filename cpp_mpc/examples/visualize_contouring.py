#!/usr/bin/env python3
"""
Visualize Contouring MPC trajectory data with multiple obstacles and scenario predictions.

Creates animated GIF showing:
- S-curve reference path
- Ego vehicle following the path
- Multiple dynamic obstacles
- Collision avoidance behavior
- Sampled scenario predictions (adaptive mode-based)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
import csv
import sys
import os

def create_s_curve_path(length=25.0, amplitude=3.0, num_points=200):
    """Generate S-curve reference path."""
    x = np.linspace(0, length, num_points)
    y = amplitude * np.sin(2 * np.pi * x / length)
    return x, y

def load_contouring_data(filename):
    """Load contouring trajectory data from CSV file."""
    data = {
        'step': [], 'time': [],
        'ego_x': [], 'ego_y': [], 'ego_theta': [], 'ego_v': [],
        'progress': [], 'lateral_error': [],
        'obstacles': [],
        'solve_time_ms': [], 'num_scenarios': []
    }

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        # Parse header to find obstacle columns
        num_obstacles = 0
        for col in header:
            if col.startswith('obs') and col.endswith('_x'):
                num_obstacles += 1

        data['num_obstacles'] = num_obstacles
        for i in range(num_obstacles):
            data[f'obs{i}_x'] = []
            data[f'obs{i}_y'] = []
            data[f'obs{i}_dist'] = []

        for row in reader:
            if len(row) < 8:
                continue
            data['step'].append(int(row[0]))
            data['time'].append(float(row[1]))
            data['ego_x'].append(float(row[2]))
            data['ego_y'].append(float(row[3]))
            data['ego_theta'].append(float(row[4]))
            data['ego_v'].append(float(row[5]))
            data['progress'].append(float(row[6]))
            data['lateral_error'].append(float(row[7]))

            col_idx = 8
            for i in range(num_obstacles):
                if col_idx + 2 < len(row):
                    data[f'obs{i}_x'].append(float(row[col_idx]))
                    data[f'obs{i}_y'].append(float(row[col_idx + 1]))
                    data[f'obs{i}_dist'].append(float(row[col_idx + 2]))
                    col_idx += 3

            if col_idx < len(row):
                data['solve_time_ms'].append(float(row[col_idx]))
            if col_idx + 1 < len(row):
                data['num_scenarios'].append(int(row[col_idx + 1]))

    # Convert to numpy arrays
    for key in ['step', 'time', 'ego_x', 'ego_y', 'ego_theta', 'ego_v',
                'progress', 'lateral_error', 'solve_time_ms', 'num_scenarios']:
        data[key] = np.array(data[key])

    for i in range(num_obstacles):
        data[f'obs{i}_x'] = np.array(data[f'obs{i}_x'])
        data[f'obs{i}_y'] = np.array(data[f'obs{i}_y'])
        data[f'obs{i}_dist'] = np.array(data[f'obs{i}_dist'])

    return data

def load_scenarios_data(filename):
    """Load scenario predictions from CSV file."""
    scenarios = {}  # step -> list of (scenario_id, obs_id, mode, timestep, x, y, prob)

    if not os.path.exists(filename):
        return scenarios

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            if len(row) < 8:
                continue
            step = int(row[0])
            scenario_id = int(row[1])
            obs_id = int(row[2])
            mode = row[3]
            timestep = int(row[4])
            pred_x = float(row[5])
            pred_y = float(row[6])
            prob = float(row[7])

            if step not in scenarios:
                scenarios[step] = []
            scenarios[step].append({
                'scenario_id': scenario_id,
                'obs_id': obs_id,
                'mode': mode,
                'timestep': timestep,
                'x': pred_x,
                'y': pred_y,
                'probability': prob
            })

    return scenarios

def create_vehicle_patch(x, y, theta, length=0.8, width=0.4, color='blue'):
    """Create a rectangle patch for vehicle visualization."""
    corners = np.array([
        [-length/2, -width/2],
        [length/2, -width/2],
        [length/2, width/2],
        [-length/2, width/2]
    ])

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    corners_global = (R @ corners.T).T + np.array([x, y])

    return plt.Polygon(corners_global, closed=True, facecolor=color,
                       edgecolor='black', alpha=0.8, linewidth=1.5)

def get_scenario_trajectories(scenarios_data, step):
    """Extract scenario trajectories for a given step."""
    if step not in scenarios_data:
        return {}

    # Group by (scenario_id, obs_id)
    trajectories = {}
    for pred in scenarios_data[step]:
        key = (pred['scenario_id'], pred['obs_id'])
        if key not in trajectories:
            trajectories[key] = {
                'mode': pred['mode'],
                'points': [],
                'probability': pred['probability']
            }
        trajectories[key]['points'].append((pred['timestep'], pred['x'], pred['y']))

    # Sort points by timestep
    for key in trajectories:
        trajectories[key]['points'].sort(key=lambda p: p[0])

    return trajectories

def create_animation(data, scenarios_data, output_filename, path_length=25.0, amplitude=3.0,
                     road_width=10.0, ego_radius=0.5, obs_radius=0.35):
    """Create animated GIF of the contouring trajectory with scenario predictions."""

    # Generate reference path
    path_x, path_y = create_s_curve_path(path_length, amplitude)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Main trajectory plot
    ax1 = axes[0]
    ax1.set_xlim(-2, path_length + 2)
    ax1.set_ylim(-amplitude - road_width/2 - 2, amplitude + road_width/2 + 2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X [m]', fontsize=12)
    ax1.set_ylabel('Y [m]', fontsize=12)
    ax1.set_title('Adaptive Scenario MPC with Mode Predictions', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Draw road
    road_upper_y = amplitude * np.sin(2 * np.pi * path_x / path_length) + road_width/2
    road_lower_y = amplitude * np.sin(2 * np.pi * path_x / path_length) - road_width/2
    ax1.fill_between(path_x, road_lower_y, road_upper_y, color='gray', alpha=0.15)
    ax1.plot(path_x, road_upper_y, 'k--', linewidth=1, alpha=0.5)
    ax1.plot(path_x, road_lower_y, 'k--', linewidth=1, alpha=0.5)
    ax1.plot(path_x, path_y, 'g-', linewidth=2.5, alpha=0.7, label='Reference Path')
    ax1.plot(path_length, 0, 'g*', markersize=20, label='Goal')

    # Ego trail
    ego_trail, = ax1.plot([], [], 'b-', linewidth=2.5, alpha=0.6, label='Ego Trajectory')

    # Obstacle colors and mode colors
    obs_colors = ['red', 'orange', 'purple', 'brown']
    mode_colors = {
        'constant_velocity': 'blue',
        'turn_left': 'green',
        'turn_right': 'cyan',
        'decelerating': 'magenta',
        'lane_change_left': 'yellow',
        'lane_change_right': 'lime',
    }

    num_obs = data['num_obstacles']

    # Create obstacle patches
    obs_patches = []
    for i in range(num_obs):
        color = obs_colors[i % len(obs_colors)]
        patch = plt.Circle((0, 0), obs_radius, color=color, alpha=0.8)
        ax1.add_patch(patch)
        obs_patches.append(patch)

    # Scenario prediction lines (will be updated each frame)
    scenario_lines = []

    # Ego patch placeholder
    ego_patch_holder = {'patch': None}

    # Info text
    info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                         verticalalignment='top', fontsize=9,
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    # Mode legend text
    mode_text = ax1.text(0.98, 0.02, '', transform=ax1.transAxes,
                         verticalalignment='bottom', horizontalalignment='right',
                         fontsize=8, fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

    ax1.legend(loc='upper right', fontsize=9)

    # Right side: metrics plots
    ax2 = axes[1]
    ax2.set_xlim(0, data['time'][-1] if len(data['time']) > 0 else 15)
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_title('Performance Metrics', fontsize=14)
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()

    vel_line, = ax2.plot([], [], 'b-', linewidth=2, label='Velocity [m/s]')
    lat_err_line, = ax2.plot([], [], 'g-', linewidth=1.5, label='Lateral Error [m]')
    min_dist_line, = ax2b.plot([], [], 'r-', linewidth=2, label='Min Obs Dist [m]')
    ax2b.axhline(y=ego_radius + obs_radius, color='r', linestyle='--', alpha=0.5,
                 label=f'Collision ({ego_radius + obs_radius:.2f}m)')

    ax2.set_ylabel('Velocity / Lateral Error', fontsize=11)
    ax2b.set_ylabel('Min Distance [m]', fontsize=11, color='red')
    ax2.set_ylim(-1, 4)
    ax2b.set_ylim(0, 12)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    time_marker = ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)

    def init():
        ego_trail.set_data([], [])
        vel_line.set_data([], [])
        lat_err_line.set_data([], [])
        min_dist_line.set_data([], [])
        return [ego_trail, vel_line, lat_err_line, min_dist_line]

    def update(frame):
        nonlocal scenario_lines

        # Remove old scenario lines
        for line in scenario_lines:
            line.remove()
        scenario_lines = []

        # Update ego trail
        ego_trail.set_data(data['ego_x'][:frame+1], data['ego_y'][:frame+1])

        # Remove old ego patch
        if ego_patch_holder['patch'] is not None:
            try:
                ego_patch_holder['patch'].remove()
            except:
                pass

        # Add new ego patch
        new_ego = create_vehicle_patch(
            data['ego_x'][frame], data['ego_y'][frame],
            data['ego_theta'][frame], color='royalblue'
        )
        ax1.add_patch(new_ego)
        ego_patch_holder['patch'] = new_ego

        # Update obstacles
        for i in range(num_obs):
            if frame < len(data[f'obs{i}_x']):
                obs_patches[i].center = (data[f'obs{i}_x'][frame], data[f'obs{i}_y'][frame])

        # Draw scenario predictions
        step = data['step'][frame]
        trajectories = get_scenario_trajectories(scenarios_data, step)

        modes_shown = set()
        for (scenario_id, obs_id), traj_data in trajectories.items():
            if len(traj_data['points']) < 2:
                continue

            mode = traj_data['mode']
            modes_shown.add(mode)

            # Get color for this mode
            color = mode_colors.get(mode, 'gray')
            alpha = min(0.6, 0.2 + traj_data['probability'])

            # Draw prediction trajectory
            points = traj_data['points']
            xs = [p[1] for p in points]
            ys = [p[2] for p in points]

            line, = ax1.plot(xs, ys, '-', color=color, alpha=alpha, linewidth=1.5)
            scenario_lines.append(line)

            # Draw endpoint marker
            marker, = ax1.plot(xs[-1], ys[-1], 'o', color=color, alpha=alpha, markersize=4)
            scenario_lines.append(marker)

        # Compute min distance
        min_dist = float('inf')
        for i in range(num_obs):
            if frame < len(data[f'obs{i}_dist']):
                min_dist = min(min_dist, data[f'obs{i}_dist'][frame])

        # Update info text
        num_scen = data['num_scenarios'][frame] if frame < len(data['num_scenarios']) else 0
        info_text.set_text(
            f"Time: {data['time'][frame]:.1f}s\n"
            f"Progress: {data['progress'][frame]:.1f}m\n"
            f"Velocity: {data['ego_v'][frame]:.2f} m/s\n"
            f"Lat Error: {data['lateral_error'][frame]:.2f}m\n"
            f"Min Dist: {min_dist:.2f}m\n"
            f"Scenarios: {num_scen}"
        )

        # Update mode legend
        if modes_shown:
            mode_legend = "Predicted Modes:\n"
            for mode in sorted(modes_shown):
                color = mode_colors.get(mode, 'gray')
                mode_legend += f"  {mode}\n"
            mode_text.set_text(mode_legend.strip())
        else:
            mode_text.set_text("")

        # Update metrics plots
        vel_line.set_data(data['time'][:frame+1], data['ego_v'][:frame+1])
        lat_err_line.set_data(data['time'][:frame+1], np.abs(data['lateral_error'][:frame+1]))

        min_dists = []
        for f in range(frame + 1):
            md = float('inf')
            for i in range(num_obs):
                if f < len(data[f'obs{i}_dist']):
                    md = min(md, data[f'obs{i}_dist'][f])
            min_dists.append(md)
        min_dist_line.set_data(data['time'][:frame+1], min_dists)

        time_marker.set_xdata([data['time'][frame]])

        return [ego_trail, vel_line, lat_err_line, min_dist_line]

    # Create animation
    num_frames = len(data['time'])
    frame_step = max(1, num_frames // 150)
    frames = list(range(0, num_frames, frame_step))

    print(f"Creating animation with {len(frames)} frames...")
    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                        blit=False, interval=100, repeat=False)

    writer = PillowWriter(fps=10)
    anim.save(output_filename, writer=writer)
    print(f"Animation saved to: {output_filename}")

    plt.close()

    # Create summary plot
    create_summary_plot(data, scenarios_data, output_filename.replace('.gif', '_summary.png'),
                        path_x, path_y, road_width, ego_radius, obs_radius)

def create_summary_plot(data, scenarios_data, output_filename, path_x, path_y, road_width,
                        ego_radius, obs_radius):
    """Create static summary plot with scenario snapshots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    num_obs = data['num_obstacles']
    obs_colors = ['red', 'orange', 'purple', 'brown']
    mode_colors = {
        'constant_velocity': 'blue',
        'turn_left': 'green',
        'turn_right': 'cyan',
        'decelerating': 'magenta',
        'lane_change_left': 'yellow',
        'lane_change_right': 'lime',
    }

    # Full trajectory plot
    ax1 = axes[0, 0]
    amplitude = np.max(path_y)
    road_upper = path_y + road_width/2
    road_lower = path_y - road_width/2
    ax1.fill_between(path_x, road_lower, road_upper, color='gray', alpha=0.15)
    ax1.plot(path_x, road_upper, 'k--', linewidth=1, alpha=0.5)
    ax1.plot(path_x, road_lower, 'k--', linewidth=1, alpha=0.5)
    ax1.plot(path_x, path_y, 'g-', linewidth=2, alpha=0.7, label='Reference')
    ax1.plot(data['ego_x'], data['ego_y'], 'b-', linewidth=2.5, label='Ego')
    ax1.plot(data['ego_x'][0], data['ego_y'][0], 'bo', markersize=10)
    ax1.plot(data['ego_x'][-1], data['ego_y'][-1], 'bs', markersize=10)

    for i in range(num_obs):
        color = obs_colors[i % len(obs_colors)]
        ax1.plot(data[f'obs{i}_x'], data[f'obs{i}_y'], '--',
                 color=color, linewidth=1.5, alpha=0.7, label=f'Obs {i}')

    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('Complete Trajectories')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Scenario snapshots at different times
    snapshot_times = [0.25, 0.5, 0.75]  # Fractions of total time
    for idx, frac in enumerate(snapshot_times):
        ax = axes[0, idx + 1] if idx < 2 else axes[1, 0]
        frame = int(frac * len(data['time']))
        frame = min(frame, len(data['time']) - 1)

        step = data['step'][frame]

        # Draw road and path
        ax.fill_between(path_x, road_lower, road_upper, color='gray', alpha=0.15)
        ax.plot(path_x, path_y, 'g-', linewidth=1.5, alpha=0.5)

        # Draw ego position
        ax.plot(data['ego_x'][frame], data['ego_y'][frame], 'b^', markersize=12)

        # Draw obstacles
        for i in range(num_obs):
            if frame < len(data[f'obs{i}_x']):
                color = obs_colors[i % len(obs_colors)]
                circle = plt.Circle((data[f'obs{i}_x'][frame], data[f'obs{i}_y'][frame]),
                                     obs_radius, color=color, alpha=0.7)
                ax.add_patch(circle)

        # Draw scenario predictions
        trajectories = get_scenario_trajectories(scenarios_data, step)
        for (scenario_id, obs_id), traj_data in trajectories.items():
            if len(traj_data['points']) < 2:
                continue
            mode = traj_data['mode']
            color = mode_colors.get(mode, 'gray')
            points = traj_data['points']
            xs = [p[1] for p in points]
            ys = [p[2] for p in points]
            ax.plot(xs, ys, '-', color=color, alpha=0.4, linewidth=1)

        ax.set_xlim(data['ego_x'][frame] - 8, data['ego_x'][frame] + 12)
        ax.set_ylim(data['ego_y'][frame] - 6, data['ego_y'][frame] + 6)
        ax.set_title(f'Scenario Predictions at t={data["time"][frame]:.1f}s')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Obstacle distances
    ax3 = axes[1, 1]
    collision_radius = ego_radius + obs_radius
    for i in range(num_obs):
        color = obs_colors[i % len(obs_colors)]
        ax3.plot(data['time'], data[f'obs{i}_dist'], '-',
                 color=color, linewidth=1.5, label=f'Obs {i}')
    ax3.axhline(y=collision_radius, color='red', linestyle='--', linewidth=2,
                label=f'Collision ({collision_radius:.2f}m)')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Distance [m]')
    ax3.set_title('Obstacle Distances')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, None)

    # Velocity and lateral error
    ax4 = axes[1, 2]
    ax4.plot(data['time'], data['ego_v'], 'b-', linewidth=2, label='Velocity')
    ax4b = ax4.twinx()
    ax4b.plot(data['time'], data['lateral_error'], 'r-', linewidth=1.5, label='Lat Error')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Velocity [m/s]', color='blue')
    ax4b.set_ylabel('Lateral Error [m]', color='red')
    ax4.set_title('Velocity & Tracking')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    print(f"Summary plot saved to: {output_filename}")
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_contouring.py <csv_file> [output.gif]")
        print("Example: python visualize_contouring.py test_uniform_weights_trajectory.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        sys.exit(1)

    # Output filename
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = csv_file.replace('.csv', '.gif').replace('_trajectory', '')

    # Look for scenarios file
    scenarios_file = csv_file.replace('_trajectory.csv', '_scenarios.csv')

    print(f"Loading data from: {csv_file}")
    data = load_contouring_data(csv_file)
    print(f"Loaded {len(data['time'])} timesteps with {data['num_obstacles']} obstacles")

    scenarios_data = {}
    if os.path.exists(scenarios_file):
        print(f"Loading scenarios from: {scenarios_file}")
        scenarios_data = load_scenarios_data(scenarios_file)
        print(f"Loaded scenario predictions for {len(scenarios_data)} timesteps")
    else:
        print(f"No scenarios file found at {scenarios_file}")

    create_animation(data, scenarios_data, output_file)

if __name__ == '__main__':
    main()
