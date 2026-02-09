#!/usr/bin/env python3
"""
Visualize OT Predictor integration with Contouring MPC.

Creates animated GIF showing:
- S-curve reference path with ego + obstacles
- OT-learned mode weight evolution per obstacle
- Prediction error and adaptive uncertainty
- Scenario predictions colored by mode
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import csv
import sys
import os


# ============================================================================
# Gaussian evaluation helpers (pure numpy, no project dependencies)
# ============================================================================

# Chi-squared thresholds for 2 DOF
CHI2_1SIGMA = 2.2789   # 68.3% coverage
CHI2_2SIGMA = 5.9915   # 95.0% coverage
CHI2_3SIGMA = 9.2103   # 99.0% coverage


def reconstruct_covariance(angle, major_r, minor_r):
    """PredictionStep ellipse params -> 2x2 covariance matrix."""
    if not np.isfinite(major_r) or major_r <= 0:
        major_r = 0.3
    if not np.isfinite(minor_r) or minor_r <= 0:
        minor_r = 0.3
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    D = np.diag([major_r**2, minor_r**2])
    return R @ D @ R.T


def ellipse_area(major_r, minor_r):
    """Area of the prediction ellipse: pi * a * b."""
    return np.pi * major_r * minor_r


def mahalanobis_distance_sq(point, mean, cov):
    """Squared Mahalanobis distance, regularized."""
    diff = np.array(point[:2]) - np.array(mean[:2])
    reg_cov = cov + 1e-8 * np.eye(2)
    try:
        cov_inv = np.linalg.inv(reg_cov)
        return float(diff @ cov_inv @ diff)
    except np.linalg.LinAlgError:
        return float('inf')


def gaussian_nll(point, mean, cov):
    """Negative log-likelihood under 2D Gaussian."""
    d2 = mahalanobis_distance_sq(point, mean, cov)
    reg_cov = cov + 1e-8 * np.eye(2)
    sign, logdet = np.linalg.slogdet(reg_cov)
    if sign <= 0:
        return float('inf')
    return 0.5 * (d2 + logdet) + np.log(2 * np.pi)


def coverage_check(point, mean, cov):
    """Check if point falls within 1/2/3-sigma ellipsoid."""
    d2 = mahalanobis_distance_sq(point, mean, cov)
    return {
        '1sigma': d2 <= CHI2_1SIGMA,
        '2sigma': d2 <= CHI2_2SIGMA,
        '3sigma': d2 <= CHI2_3SIGMA,
    }


# ============================================================================


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
        'solve_time_ms': [], 'num_scenarios': []
    }

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

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
    scenarios = {}

    if not os.path.exists(filename):
        return scenarios

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 8:
                continue
            step = int(row[0])
            if step not in scenarios:
                scenarios[step] = []
            scenarios[step].append({
                'scenario_id': int(row[1]),
                'obs_id': int(row[2]),
                'mode': row[3],
                'timestep': int(row[4]),
                'x': float(row[5]),
                'y': float(row[6]),
                'probability': float(row[7])
            })

    return scenarios


def load_ot_metrics(filename):
    """Load OT predictor metrics from CSV file.

    Returns (ot_data, sight_radius) where sight_radius is parsed from
    the comment header line '# sight_radius=...' or None if absent.
    """
    ot_data = {}
    sight_radius = None

    if not os.path.exists(filename):
        return ot_data, sight_radius

    with open(filename, 'r') as f:
        # Check for comment header with sight_radius
        first_line = f.readline().strip()
        if first_line.startswith('# sight_radius='):
            sight_radius = float(first_line.split('=')[1])
            header_line = f.readline().strip()
        else:
            header_line = first_line

        header = header_line.split(',')

        # Parse mode weight columns
        mode_cols = [h.replace('w_', '') for h in header if h.startswith('w_')]
        has_in_sight = 'in_sight' in header
        has_pred_ellipse = 'pred_angle' in header
        has_pred5 = 'pred5_x' in header

        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            step = int(row[0])
            time = float(row[1])
            obs_id = int(row[2])

            if step not in ot_data:
                ot_data[step] = {}

            col_idx = 3
            mode_weights = {}
            for mode in mode_cols:
                if col_idx < len(row):
                    mode_weights[mode] = float(row[col_idx])
                    col_idx += 1

            pred_error = float(row[col_idx]) if col_idx < len(row) else 0.0
            col_idx += 1
            uncertainty = float(row[col_idx]) if col_idx < len(row) else 1.0
            col_idx += 1
            num_obs = int(row[col_idx]) if col_idx < len(row) else 0
            col_idx += 1
            in_sight = bool(int(row[col_idx])) if (has_in_sight and col_idx < len(row)) else True

            # Parse optional prediction ellipse columns
            pred_angle = pred_major_r = pred_minor_r = 0.0
            pred5_x = pred5_y = pred5_angle = 0.0
            pred5_major_r = pred5_minor_r = 0.3

            if has_pred_ellipse:
                col_idx += 1  # advance past in_sight
                pred_angle = float(row[col_idx]) if col_idx < len(row) else 0.0
                col_idx += 1
                pred_major_r = float(row[col_idx]) if col_idx < len(row) else 0.3
                col_idx += 1
                pred_minor_r = float(row[col_idx]) if col_idx < len(row) else 0.3

            if has_pred5:
                col_idx += 1
                pred5_x = float(row[col_idx]) if col_idx < len(row) else 0.0
                col_idx += 1
                pred5_y = float(row[col_idx]) if col_idx < len(row) else 0.0
                col_idx += 1
                pred5_major_r = float(row[col_idx]) if col_idx < len(row) else 0.3
                col_idx += 1
                pred5_minor_r = float(row[col_idx]) if col_idx < len(row) else 0.3
                col_idx += 1
                pred5_angle = float(row[col_idx]) if col_idx < len(row) else 0.0

            ot_data[step][obs_id] = {
                'time': time,
                'mode_weights': mode_weights,
                'pred_error': pred_error,
                'uncertainty': uncertainty,
                'num_observations': num_obs,
                'in_sight': in_sight,
                'pred_angle': pred_angle,
                'pred_major_r': pred_major_r,
                'pred_minor_r': pred_minor_r,
                'pred5_x': pred5_x,
                'pred5_y': pred5_y,
                'pred5_major_r': pred5_major_r,
                'pred5_minor_r': pred5_minor_r,
                'pred5_angle': pred5_angle,
            }

    return ot_data, sight_radius


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

    for key in trajectories:
        trajectories[key]['points'].sort(key=lambda p: p[0])

    return trajectories


def compute_ground_truth_metrics(data, ot_data, num_obs, lookahead=5):
    """Compute ground truth metrics from k=5 predictions vs actual positions.

    Returns dict: obs_id -> list of {step, time, mahal_sq, coverage_*, nll, area}
    """
    gt_metrics = {i: [] for i in range(num_obs)}
    steps_sorted = sorted(ot_data.keys())

    for step in steps_sorted:
        future_step = step + lookahead
        if future_step not in ot_data and future_step >= len(data['time']):
            continue

        for obs_id in range(num_obs):
            if obs_id not in ot_data[step]:
                continue
            entry = ot_data[step][obs_id]
            if not entry.get('in_sight', True):
                continue

            # Get predicted position/ellipse at k=5
            pred_x = entry.get('pred5_x', 0.0)
            pred_y = entry.get('pred5_y', 0.0)
            p5_major = entry.get('pred5_major_r', 0.3)
            p5_minor = entry.get('pred5_minor_r', 0.3)
            p5_angle = entry.get('pred5_angle', 0.0)

            if pred_x == 0.0 and pred_y == 0.0 and p5_major == 0.3:
                continue  # default/missing values

            # Get actual position at future_step
            if future_step >= len(data[f'obs{obs_id}_x']):
                continue
            actual = np.array([data[f'obs{obs_id}_x'][future_step],
                               data[f'obs{obs_id}_y'][future_step]])
            pred_mean = np.array([pred_x, pred_y])
            cov = reconstruct_covariance(p5_angle, p5_major, p5_minor)

            d2 = mahalanobis_distance_sq(actual, pred_mean, cov)
            cov_check = coverage_check(actual, pred_mean, cov)
            nll = gaussian_nll(actual, pred_mean, cov)
            area = ellipse_area(entry.get('pred_major_r', 0.3),
                                entry.get('pred_minor_r', 0.3))

            gt_metrics[obs_id].append({
                'step': step,
                'time': entry['time'],
                'mahal_sq': d2,
                'coverage_1sigma': cov_check['1sigma'],
                'coverage_2sigma': cov_check['2sigma'],
                'coverage_3sigma': cov_check['3sigma'],
                'nll': nll,
                'area': area,
            })

    return gt_metrics


def create_animation(data, scenarios_data, ot_data, output_filename,
                     path_length=25.0, amplitude=3.0, road_width=10.0,
                     ego_radius=0.5, obs_radius=0.35, sight_radius=None):
    """Create animated GIF with per-obstacle OT mode weight sub-figures."""

    path_x, path_y = create_s_curve_path(path_length, amplitude)

    num_obs = data['num_obstacles']
    obs_colors = ['#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#17becf', '#bcbd22']
    mode_colors = {
        'constant_velocity': '#1f77b4',
        'turn_left': '#2ca02c',
        'turn_right': '#17becf',
        'decelerating': '#e377c2',
        'accelerating': '#ff7f0e',
        'lane_change_left': '#bcbd22',
        'lane_change_right': '#7f7f7f',
    }

    # Layout: 4 rows x 4 cols
    #   Row 0-2, Col 0-1: Main trajectory (3x2 span)
    #   Row 0, Col 2-3:   Per-obstacle weights (obs 0, obs 1)
    #   Row 1, Col 2-3:   Per-obstacle weights (obs 2, obs 3)
    #   Row 2, Col 2-3:   Per-obstacle weights (obs 4, obs 5)
    #   Row 3, Col 0-3:   Distances, Velocity, Pred Error, Obs Count
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(4, 4, hspace=0.38, wspace=0.3)

    # ---- Main trajectory plot (top-left 3x2) ----
    ax_main = fig.add_subplot(gs[0:3, 0:2])
    ax_main.set_xlim(-2, path_length + 2)
    ax_main.set_ylim(-amplitude - road_width/2 - 2, amplitude + road_width/2 + 2)
    ax_main.set_aspect('equal')
    ax_main.set_xlabel('X [m]', fontsize=11)
    ax_main.set_ylabel('Y [m]', fontsize=11)
    ax_main.set_title('Safe Horizon MPC + OT Dynamics Learning', fontsize=13, fontweight='bold')
    ax_main.grid(True, alpha=0.3)

    road_upper = amplitude * np.sin(2 * np.pi * path_x / path_length) + road_width/2
    road_lower = amplitude * np.sin(2 * np.pi * path_x / path_length) - road_width/2
    ax_main.fill_between(path_x, road_lower, road_upper, color='gray', alpha=0.12)
    ax_main.plot(path_x, road_upper, 'k--', linewidth=1, alpha=0.4)
    ax_main.plot(path_x, road_lower, 'k--', linewidth=1, alpha=0.4)
    ax_main.plot(path_x, path_y, 'g-', linewidth=2.5, alpha=0.7, label='Reference Path')
    ax_main.plot(path_length, 0, 'g*', markersize=20, label='Goal')

    ego_trail, = ax_main.plot([], [], 'b-', linewidth=2.5, alpha=0.6, label='Ego Trajectory')

    # Sight radius circle (centered on ego, updated each frame)
    sight_circle = None
    if sight_radius is not None:
        sight_circle = plt.Circle((0, 0), sight_radius, fill=False,
                                   color='dodgerblue', linestyle='--',
                                   linewidth=1.5, alpha=0.5,
                                   label=f'Sight Radius ({sight_radius:.1f}m)')
        ax_main.add_patch(sight_circle)

    obs_patches = []
    for i in range(num_obs):
        patch = plt.Circle((0, 0), obs_radius, color=obs_colors[i % len(obs_colors)], alpha=0.8)
        ax_main.add_patch(patch)
        obs_patches.append(patch)

    scenario_lines = []
    ego_patch_holder = {'patch': None}

    info_text = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes,
                              verticalalignment='top', fontsize=9,
                              fontfamily='monospace',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    ax_main.legend(loc='upper right', fontsize=8)

    # ---- Per-obstacle mode weight sub-figures (right 3x2 grid) ----
    t_max = data['time'][-1] if len(data['time']) > 0 else 15

    # Map obstacle index to grid position in the right 3x2
    weight_axes = {}
    per_obs_weight_lines = {}
    per_obs_weight_history = {}

    for oid in range(min(num_obs, 6)):
        row = oid // 2    # 0, 1, or 2
        col = 2 + oid % 2  # 2 or 3
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(f'Obs {oid}: Mode Weights', fontsize=9, fontweight='bold',
                     color=obs_colors[oid % len(obs_colors)])
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, t_max)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time [s]', fontsize=8)
        if oid % 2 == 0:
            ax.set_ylabel('Weight', fontsize=8)

        lines = {}
        history = {}
        for mode, color in mode_colors.items():
            line, = ax.plot([], [], '-', color=color, linewidth=1.3,
                            label=mode.replace('_', ' '), alpha=0.8)
            lines[mode] = line
            history[mode] = ([], [])

        ax.legend(loc='upper left', fontsize=5.5, ncol=2, framealpha=0.7)
        weight_axes[oid] = ax
        per_obs_weight_lines[oid] = lines
        per_obs_weight_history[oid] = history

    # ---- Bottom row: distances, velocity, prediction error, obs count ----
    ax_dist = fig.add_subplot(gs[3, 0])
    ax_dist.set_title('Obstacle Distances', fontsize=10, fontweight='bold')
    ax_dist.set_xlabel('Time [s]', fontsize=9)
    ax_dist.set_ylabel('Distance [m]', fontsize=9)
    ax_dist.set_xlim(0, t_max)
    ax_dist.set_ylim(0, 12)
    ax_dist.grid(True, alpha=0.3)
    ax_dist.axhline(y=ego_radius + obs_radius, color='red', linestyle='--', alpha=0.5,
                     label=f'Collision ({ego_radius + obs_radius:.2f}m)')
    dist_lines = {}
    for i in range(num_obs):
        line, = ax_dist.plot([], [], '-', color=obs_colors[i % len(obs_colors)], linewidth=1.5,
                              label=f'Obs {i}', alpha=0.8)
        dist_lines[i] = line
    ax_dist.legend(loc='upper right', fontsize=7)

    ax_vel = fig.add_subplot(gs[3, 1])
    ax_vel.set_title('Velocity & Tracking', fontsize=10, fontweight='bold')
    ax_vel.set_xlabel('Time [s]', fontsize=9)
    ax_vel.set_xlim(0, t_max)
    ax_vel.set_ylim(-1, 4)
    ax_vel.grid(True, alpha=0.3)
    vel_line, = ax_vel.plot([], [], 'b-', linewidth=2, label='Velocity [m/s]')
    lat_line, = ax_vel.plot([], [], 'g-', linewidth=1.5, label='|Lat Error| [m]')
    ax_vel.legend(loc='upper right', fontsize=8)

    # Per-obstacle prediction error (combined)
    ax_error = fig.add_subplot(gs[3, 2])
    ax_error.set_title('Prediction Error', fontsize=10, fontweight='bold')
    ax_error.set_xlabel('Time [s]', fontsize=9)
    ax_error.set_ylabel('Error [m]', fontsize=9)
    ax_error.set_xlim(0, t_max)
    ax_error.set_ylim(0, 3)
    ax_error.grid(True, alpha=0.3)
    error_lines = {}
    error_history_per_obs = {}
    for i in range(num_obs):
        line, = ax_error.plot([], [], '-', color=obs_colors[i % len(obs_colors)], linewidth=1.5,
                              label=f'Obs {i}', alpha=0.8)
        error_lines[i] = line
        error_history_per_obs[i] = ([], [])
    ax_error.legend(loc='upper right', fontsize=7)

    # Observation buffer count
    ax_obs_count = fig.add_subplot(gs[3, 3])
    ax_obs_count.set_title('OT Observation Buffer', fontsize=10, fontweight='bold')
    ax_obs_count.set_xlabel('Time [s]', fontsize=9)
    ax_obs_count.set_ylabel('Observations', fontsize=9)
    ax_obs_count.set_xlim(0, t_max)
    ax_obs_count.grid(True, alpha=0.3)
    obs_count_lines = {}
    obs_count_history = {i: ([], []) for i in range(num_obs)}
    for i in range(num_obs):
        line, = ax_obs_count.plot([], [], '-', color=obs_colors[i % len(obs_colors)], linewidth=1.5,
                                   label=f'Obs {i}', alpha=0.8)
        obs_count_lines[i] = line
    ax_obs_count.legend(loc='upper left', fontsize=7)

    def update(frame):
        nonlocal scenario_lines

        for line in scenario_lines:
            line.remove()
        scenario_lines = []

        ego_trail.set_data(data['ego_x'][:frame+1], data['ego_y'][:frame+1])

        if ego_patch_holder['patch'] is not None:
            try:
                ego_patch_holder['patch'].remove()
            except:
                pass

        ego_x_now = data['ego_x'][frame]
        ego_y_now = data['ego_y'][frame]

        new_ego = create_vehicle_patch(
            ego_x_now, ego_y_now,
            data['ego_theta'][frame], color='royalblue')
        ax_main.add_patch(new_ego)
        ego_patch_holder['patch'] = new_ego

        # Update sight radius circle position
        if sight_circle is not None:
            sight_circle.center = (ego_x_now, ego_y_now)

        step = data['step'][frame]

        for i in range(num_obs):
            if frame < len(data[f'obs{i}_x']):
                obs_patches[i].center = (data[f'obs{i}_x'][frame], data[f'obs{i}_y'][frame])
                # Dim obstacles outside sight radius
                if step in ot_data and i in ot_data[step]:
                    in_sight = ot_data[step][i].get('in_sight', True)
                    obs_patches[i].set_alpha(0.8 if in_sight else 0.2)
                    obs_patches[i].set_linestyle('-' if in_sight else ':')
        trajectories = get_scenario_trajectories(scenarios_data, step)

        for (scenario_id, obs_id), traj_data in trajectories.items():
            if len(traj_data['points']) < 2:
                continue
            mode = traj_data['mode']
            color = mode_colors.get(mode, 'gray')
            alpha = min(0.6, 0.2 + traj_data['probability'])
            points = traj_data['points']
            xs = [p[1] for p in points]
            ys = [p[2] for p in points]
            line, = ax_main.plot(xs, ys, '-', color=color, alpha=alpha, linewidth=1.5)
            scenario_lines.append(line)
            marker, = ax_main.plot(xs[-1], ys[-1], 'o', color=color, alpha=alpha, markersize=4)
            scenario_lines.append(marker)

        # Draw 2-sigma prediction ellipses for in-sight obstacles
        if step in ot_data:
            from matplotlib.patches import Ellipse as EllipsePatch
            for i in range(num_obs):
                if i in ot_data[step] and ot_data[step][i].get('in_sight', False):
                    entry = ot_data[step][i]
                    p_angle = entry.get('pred_angle', 0.0)
                    p_major = entry.get('pred_major_r', 0.0)
                    p_minor = entry.get('pred_minor_r', 0.0)
                    if p_major > 0.01 and p_minor > 0.01:
                        cov = reconstruct_covariance(p_angle, p_major, p_minor)
                        eigvals, eigvecs = np.linalg.eigh(cov)
                        angle_deg = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
                        scale = np.sqrt(CHI2_2SIGMA)
                        w = 2 * scale * np.sqrt(max(eigvals[1], 1e-10))
                        h = 2 * scale * np.sqrt(max(eigvals[0], 1e-10))
                        ox = data[f'obs{i}_x'][frame] if frame < len(data[f'obs{i}_x']) else 0
                        oy = data[f'obs{i}_y'][frame] if frame < len(data[f'obs{i}_y']) else 0
                        ell = EllipsePatch(xy=(ox, oy), width=w, height=h,
                                           angle=angle_deg,
                                           facecolor=obs_colors[i % len(obs_colors)],
                                           edgecolor=obs_colors[i % len(obs_colors)],
                                           alpha=0.15, linewidth=0.8, linestyle='--')
                        ax_main.add_patch(ell)
                        scenario_lines.append(ell)

        min_dist = float('inf')
        for i in range(num_obs):
            if frame < len(data[f'obs{i}_dist']):
                min_dist = min(min_dist, data[f'obs{i}_dist'][frame])

        num_scen = data['num_scenarios'][frame] if frame < len(data['num_scenarios']) else 0
        info_text.set_text(
            f"Time: {data['time'][frame]:.1f}s\n"
            f"Progress: {data['progress'][frame]:.1f}m\n"
            f"Velocity: {data['ego_v'][frame]:.2f} m/s\n"
            f"Min Dist: {min_dist:.2f}m\n"
            f"Scenarios: {num_scen}"
        )

        # ---- Update per-obstacle OT mode weights ----
        time_val = data['time'][frame]
        if step in ot_data:
            for oid in range(min(num_obs, 6)):
                if oid in ot_data[step]:
                    obs_ot = ot_data[step][oid]
                    for mode in mode_colors:
                        w = obs_ot['mode_weights'].get(mode, 0.0)
                        ts, ws = per_obs_weight_history[oid][mode]
                        ts.append(time_val)
                        ws.append(w)
                        per_obs_weight_lines[oid][mode].set_data(ts, ws)

                    # Prediction error per obstacle
                    ts_e, vs_e = error_history_per_obs[oid]
                    ts_e.append(time_val)
                    vs_e.append(obs_ot['pred_error'])
                    error_lines[oid].set_data(ts_e, vs_e)

                    # Obs count
                    ts_c, cs = obs_count_history[oid]
                    ts_c.append(time_val)
                    cs.append(obs_ot['num_observations'])
                    obs_count_lines[oid].set_data(ts_c, cs)

            # Auto-scale obs count
            max_count = max((cs[-1] if cs else 0) for _, (_, cs) in obs_count_history.items() if cs)
            if max_count > 0:
                ax_obs_count.set_ylim(0, max_count * 1.2)

        for i in range(num_obs):
            dist_lines[i].set_data(data['time'][:frame+1], data[f'obs{i}_dist'][:frame+1])

        vel_line.set_data(data['time'][:frame+1], data['ego_v'][:frame+1])
        lat_line.set_data(data['time'][:frame+1], np.abs(data['lateral_error'][:frame+1]))

        return []

    num_frames = len(data['time'])
    frame_step = max(1, num_frames // 150)
    frames = list(range(0, num_frames, frame_step))

    print(f"Creating OT predictor animation with {len(frames)} frames...")
    anim = FuncAnimation(fig, update, frames=frames, blit=False, interval=100, repeat=False)

    writer = PillowWriter(fps=10)
    anim.save(output_filename, writer=writer)
    print(f"OT predictor animation saved to: {output_filename}")
    plt.close()

    # Create summary plot
    create_summary_plot(data, ot_data, output_filename.replace('.gif', '_summary.png'),
                        path_x, path_y, road_width, ego_radius, obs_radius, mode_colors,
                        obs_colors, num_obs, sight_radius=sight_radius)


def create_summary_plot(data, ot_data, output_filename, path_x, path_y, road_width,
                        ego_radius, obs_radius, mode_colors, obs_colors, num_obs,
                        sight_radius=None):
    """Create static summary plot with per-obstacle mode weight sub-figures.

    Layout: 5 rows
      Row 0: trajectory (span cols) + distances + velocity
      Row 1: per-obstacle mode weights
      Row 2: per-obstacle prediction error & uncertainty
      Row 3: per-obstacle ellipse area narrowing
      Row 4: per-obstacle ground truth coverage probability
    """
    n_cols = max(num_obs, 3)
    fig = plt.figure(figsize=(5.5 * n_cols, 22))
    fig.suptitle('OT Predictor Integration - Per-Obstacle Learning Summary',
                 fontsize=15, fontweight='bold', y=0.99)

    gs = fig.add_gridspec(5, n_cols, hspace=0.50, wspace=0.35,
                          top=0.94, bottom=0.04, left=0.05, right=0.97,
                          height_ratios=[1.2, 1, 1, 0.8, 1])

    amplitude = np.max(path_y)
    steps_sorted = sorted(ot_data.keys()) if ot_data else []

    # Compute ground truth metrics
    gt_metrics = compute_ground_truth_metrics(data, ot_data, num_obs)

    # ---- Row 0: Trajectory (spanning left columns) + distances + velocity ----
    traj_span = max(1, n_cols - 2)
    ax_traj = fig.add_subplot(gs[0, :traj_span])
    road_upper = path_y + road_width/2
    road_lower = path_y - road_width/2
    ax_traj.fill_between(path_x, road_lower, road_upper, color='gray', alpha=0.15)
    ax_traj.plot(path_x, path_y, 'g-', linewidth=2, alpha=0.7, label='Reference')
    ax_traj.plot(data['ego_x'], data['ego_y'], 'b-', linewidth=2.5, label='Ego')
    for i in range(num_obs):
        ax_traj.plot(data[f'obs{i}_x'], data[f'obs{i}_y'], '--',
                     color=obs_colors[i % len(obs_colors)], linewidth=1.5,
                     alpha=0.7, label=f'Obs {i}')
    if sight_radius is not None and len(data['ego_x']) > 0:
        mid = len(data['ego_x']) // 2
        sr_circle = plt.Circle((data['ego_x'][mid], data['ego_y'][mid]),
                                sight_radius, fill=False, color='dodgerblue',
                                linestyle='--', linewidth=1.5, alpha=0.4,
                                label=f'Sight ({sight_radius:.0f}m)')
        ax_traj.add_patch(sr_circle)
    ax_traj.set_xlabel('X [m]')
    ax_traj.set_ylabel('Y [m]')
    ax_traj.set_title('Complete Trajectories')
    ax_traj.legend(loc='upper left', fontsize=7)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_aspect('equal')

    # Obstacle distances
    ax_dist = fig.add_subplot(gs[0, traj_span])
    for i in range(num_obs):
        ax_dist.plot(data['time'], data[f'obs{i}_dist'], '-',
                     color=obs_colors[i % len(obs_colors)], linewidth=1.5,
                     label=f'Obs {i}')
    ax_dist.axhline(y=ego_radius + obs_radius, color='red', linestyle='--',
                     linewidth=2, label=f'Collision ({ego_radius+obs_radius:.2f}m)')
    ax_dist.set_xlabel('Time [s]')
    ax_dist.set_ylabel('Distance [m]')
    ax_dist.set_title('Obstacle Distances')
    ax_dist.legend(fontsize=7)
    ax_dist.grid(True, alpha=0.3)
    ax_dist.set_ylim(0, None)

    # Velocity + lateral error
    if n_cols > traj_span + 1:
        ax_vel = fig.add_subplot(gs[0, traj_span + 1])
        ax_vel.plot(data['time'], data['ego_v'], 'b-', linewidth=2, label='Velocity')
        ax_vel.plot(data['time'], np.abs(data['lateral_error']), 'g-',
                    linewidth=1.5, label='|Lat Error|')
        ax_vel.set_xlabel('Time [s]')
        ax_vel.set_ylabel('[m/s] or [m]')
        ax_vel.set_title('Velocity & Tracking')
        ax_vel.legend(fontsize=8)
        ax_vel.grid(True, alpha=0.3)

    # ---- Row 1: Per-obstacle mode weight evolution ----
    for obs_id in range(num_obs):
        col = obs_id if obs_id < n_cols else obs_id % n_cols
        ax = fig.add_subplot(gs[1, col])

        obs_weights = {mode: ([], []) for mode in mode_colors}
        for step in steps_sorted:
            if obs_id in ot_data[step]:
                t = ot_data[step][obs_id]['time']
                for mode in mode_colors:
                    w = ot_data[step][obs_id]['mode_weights'].get(mode, 0.0)
                    obs_weights[mode][0].append(t)
                    obs_weights[mode][1].append(w)

        has_data = False
        for mode, color in mode_colors.items():
            ts, ws = obs_weights[mode]
            if ts:
                has_data = True
                ax.plot(ts, ws, '-', color=color, linewidth=1.5,
                        label=mode.replace('_', ' '), alpha=0.8)

        ax.set_xlabel('Time [s]', fontsize=8)
        if obs_id == 0:
            ax.set_ylabel('Mode Weight', fontsize=10)
        ax.set_title(f'Obs {obs_id}: OT Mode Weights', fontsize=10, fontweight='bold',
                     color=obs_colors[obs_id % len(obs_colors)])
        ax.legend(loc='best', fontsize=6, ncol=2, framealpha=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', ls=':', alpha=0.25)

    # ---- Row 2: Per-obstacle prediction error & uncertainty ----
    for obs_id in range(num_obs):
        col = obs_id if obs_id < n_cols else obs_id % n_cols
        ax = fig.add_subplot(gs[2, col])

        err_times = []
        err_vals = []
        unc_vals = []
        for step in steps_sorted:
            if obs_id in ot_data[step]:
                err_times.append(ot_data[step][obs_id]['time'])
                err_vals.append(ot_data[step][obs_id]['pred_error'])
                unc_vals.append(ot_data[step][obs_id]['uncertainty'])

        if err_times:
            ax.plot(err_times, err_vals, '-',
                    color=obs_colors[obs_id % len(obs_colors)],
                    linewidth=1.5, alpha=0.85, label='Pred Error')
            ax.fill_between(err_times, 0, err_vals,
                            color=obs_colors[obs_id % len(obs_colors)], alpha=0.1)
            ax2 = ax.twinx()
            ax2.plot(err_times, unc_vals, 'k--', linewidth=1, alpha=0.5,
                     label='Uncertainty')
            ax2.set_ylabel('Scale', fontsize=7, color='gray')

        ax.set_xlabel('Time [s]', fontsize=8)
        if obs_id == 0:
            ax.set_ylabel('Prediction Error [m]', fontsize=9)
        ax.set_title(f'Obs {obs_id}: Prediction Error', fontsize=10,
                     color=obs_colors[obs_id % len(obs_colors)])
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        if err_times:
            ax.legend(fontsize=7, loc='upper right')

    # ---- Row 3: Per-obstacle ellipse area narrowing ----
    for obs_id in range(num_obs):
        col = obs_id if obs_id < n_cols else obs_id % n_cols
        ax = fig.add_subplot(gs[3, col])
        oc = obs_colors[obs_id % len(obs_colors)]

        area_times = []
        area_vals = []
        for step in steps_sorted:
            if obs_id in ot_data[step] and ot_data[step][obs_id].get('in_sight', False):
                mr = ot_data[step][obs_id].get('pred_major_r', 0.3)
                mnr = ot_data[step][obs_id].get('pred_minor_r', 0.3)
                area_times.append(ot_data[step][obs_id]['time'])
                area_vals.append(ellipse_area(mr, mnr))

        if area_times:
            ax.plot(area_times, area_vals, '-', color=oc, linewidth=1.5, alpha=0.85)
            ax.fill_between(area_times, 0, area_vals, color=oc, alpha=0.10)
            win = max(3, len(area_vals) // 5)
            if len(area_vals) > win:
                sm = np.convolve(area_vals, np.ones(win)/win, mode='valid')
                ax.plot(area_times[win-1:], sm, 'k-', lw=1.5, alpha=0.45, label='Trend')
            ax.annotate(f'{area_vals[-1]:.4f}', xy=(area_times[-1], area_vals[-1]),
                        fontsize=7, fontweight='bold', ha='right', va='bottom', color=oc)

        ax.set_xlabel('Time [s]', fontsize=8)
        if obs_id == 0:
            ax.set_ylabel('Ellipse Area [m²]', fontsize=9)
        ax.set_title(f'Obs {obs_id}: Uncertainty Narrowing', fontsize=9, color=oc)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)
        if area_times and len(area_times) > 5:
            ax.legend(fontsize=7)

    # ---- Row 4: Per-obstacle ground truth coverage ----
    for obs_id in range(num_obs):
        col = obs_id if obs_id < n_cols else obs_id % n_cols
        ax = fig.add_subplot(gs[4, col])
        oc = obs_colors[obs_id % len(obs_colors)]

        metrics = gt_metrics.get(obs_id, [])
        if len(metrics) > 3:
            t_gt = [m['time'] for m in metrics]
            c1 = [float(m['coverage_1sigma']) for m in metrics]
            c2 = [float(m['coverage_2sigma']) for m in metrics]
            c3 = [float(m['coverage_3sigma']) for m in metrics]
            mahal_vals = [np.sqrt(m['mahal_sq']) for m in metrics]

            roll_win = min(10, max(1, len(metrics) // 3))
            kernel = np.ones(roll_win) / roll_win

            def rolling(arr):
                if len(arr) >= roll_win:
                    return np.convolve(arr, kernel, mode='valid')
                return np.array(arr)

            c1_r = rolling(c1)
            c2_r = rolling(c2)
            c3_r = rolling(c3)
            t_r = t_gt[roll_win-1:] if len(t_gt) >= roll_win else t_gt

            ax.plot(t_r, np.array(c1_r) * 100, '-', color='#e74c3c', linewidth=1.5,
                    alpha=0.85, label=f'1σ ({100*np.mean(c1):.0f}%)')
            ax.plot(t_r, np.array(c2_r) * 100, '-', color='#f39c12', linewidth=1.5,
                    alpha=0.85, label=f'2σ ({100*np.mean(c2):.0f}%)')
            ax.plot(t_r, np.array(c3_r) * 100, '-', color='#27ae60', linewidth=1.5,
                    alpha=0.85, label=f'3σ ({100*np.mean(c3):.0f}%)')

            ax.axhline(68.3, color='#e74c3c', ls=':', alpha=0.3, lw=0.8)
            ax.axhline(95.0, color='#f39c12', ls=':', alpha=0.3, lw=0.8)
            ax.axhline(99.0, color='#27ae60', ls=':', alpha=0.3, lw=0.8)

            ax.set_ylim(0, 105)
            ax.set_ylabel('Coverage %', fontsize=8)
            ax.legend(loc='lower right', fontsize=6, framealpha=0.8)

            ax2 = ax.twinx()
            mahal_r = rolling(mahal_vals)
            ax2.plot(t_r, mahal_r, 'k--', linewidth=1.0, alpha=0.4)
            ax2.set_ylabel('√Mahal', fontsize=7, color='gray')
            ax2.tick_params(axis='y', labelsize=6, colors='gray')
        else:
            ax.text(0.5, 0.5, 'Insufficient\ndata',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color='gray')

        ax.set_xlabel('Time [s]', fontsize=8)
        ax.set_title(f'Obs {obs_id}: GT Coverage (k=5)', fontsize=9, color=oc)
        ax.grid(True, alpha=0.25)

    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"OT summary plot saved to: {output_filename}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_ot_predictor.py <trajectory_csv> [output.gif]")
        sys.exit(1)

    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = csv_file.replace('.csv', '_ot.gif').replace('_trajectory', '')

    # Derive other filenames
    scenarios_file = csv_file.replace('_trajectory.csv', '_scenarios.csv')
    ot_metrics_file = csv_file.replace('_trajectory.csv', '_ot_metrics.csv')

    print(f"Loading trajectory data from: {csv_file}")
    data = load_contouring_data(csv_file)
    print(f"Loaded {len(data['time'])} timesteps with {data['num_obstacles']} obstacles")

    scenarios_data = {}
    if os.path.exists(scenarios_file):
        print(f"Loading scenarios from: {scenarios_file}")
        scenarios_data = load_scenarios_data(scenarios_file)
        print(f"Loaded scenario predictions for {len(scenarios_data)} timesteps")

    ot_data = {}
    sight_radius = None
    if os.path.exists(ot_metrics_file):
        print(f"Loading OT metrics from: {ot_metrics_file}")
        ot_data, sight_radius = load_ot_metrics(ot_metrics_file)
        print(f"Loaded OT metrics for {len(ot_data)} timesteps")
        if sight_radius is not None:
            print(f"Sight radius: {sight_radius:.1f}m")
    else:
        print(f"Warning: No OT metrics file found at {ot_metrics_file}")

    create_animation(data, scenarios_data, ot_data, output_file,
                     sight_radius=sight_radius)


if __name__ == '__main__':
    main()
