#!/usr/bin/env python3
"""
Visualize paper figure data from C++ test_paper_figures.

Reads paper_fig_*.csv files and produces publication-quality PNGs and GIFs.

Usage:
    python3 visualize_paper_figures.py [figure_name | all]

Examples:
    python3 visualize_paper_figures.py all
    python3 visualize_paper_figures.py scenario_fan
    python3 visualize_paper_figures.py mpc_loop
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
import csv
import sys
import os
from collections import defaultdict

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

MODE_COLORS = {
    'constant_velocity': '#1f77b4',
    'turn_left': '#2ca02c',
    'turn_right': '#17becf',
    'decelerating': '#d62728',
    'lane_change_left': '#ff7f0e',
    'lane_change_right': '#9467bd',
}

def get_mode_color(mode):
    return MODE_COLORS.get(mode, '#7f7f7f')


# ============================================================================
# Fig 1: Scenario Fan — Mode Weight Comparison
# ============================================================================

def generate_scenario_fan(csv_file='paper_fig_scenario_fan.csv'):
    if not os.path.exists(csv_file):
        print(f"  Skipping scenario_fan: {csv_file} not found")
        return

    data = defaultdict(lambda: defaultdict(list))
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            wt = row['weight_type']
            sid = int(row['scenario_id'])
            key = (wt, sid, row['mode'])
            data[wt][(sid, row['mode'])].append(
                (int(row['timestep']), float(row['x']), float(row['y']))
            )

    weight_types = ['UNIFORM', 'FREQUENCY', 'RECENCY']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, wt in enumerate(weight_types):
        ax = axes[idx]
        ax.set_title(f'{wt} Weights')
        ax.set_xlabel('X [m]')
        if idx == 0:
            ax.set_ylabel('Y [m]')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Draw obstacle origin
        ax.plot(5.0, 0.0, 'rs', markersize=10, label='Obstacle', zorder=5)

        # Count modes for legend
        mode_counts = defaultdict(int)
        for (sid, mode), points in data[wt].items():
            mode_counts[mode] += 1
            points.sort(key=lambda p: p[0])
            xs = [p[1] for p in points]
            ys = [p[2] for p in points]
            color = get_mode_color(mode)
            ax.plot(xs, ys, '-', color=color, alpha=0.4, linewidth=1.2)
            ax.plot(xs[-1], ys[-1], 'o', color=color, alpha=0.5, markersize=3)

        # Mode probability annotations
        total = sum(mode_counts.values())
        mode_text = []
        for mode in sorted(mode_counts.keys()):
            frac = mode_counts[mode] / max(1, total)
            color = get_mode_color(mode)
            ax.plot([], [], '-', color=color, linewidth=2.5, label=f'{mode} ({frac:.0%})')

        ax.legend(loc='upper left', fontsize=8)

    fig.suptitle('Fig 1: Scenario Fan — Mode Weight Comparison (Algorithm 1 + Section 4)', fontsize=14)
    plt.tight_layout()
    plt.savefig('paper_fig_scenario_fan.png')
    print("  Saved paper_fig_scenario_fan.png")
    plt.close()


# ============================================================================
# Fig 2: Trajectory Moments with Uncertainty Ellipses
# ============================================================================

def draw_covariance_ellipse(ax, mean_x, mean_y, cov_xx, cov_xy, cov_yy,
                             n_std=2.0, **kwargs):
    """Draw a covariance ellipse at the given position."""
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2 * n_std * np.sqrt(eigvals[0])
    height = 2 * n_std * np.sqrt(eigvals[1])

    if width < 1e-6 and height < 1e-6:
        return

    ellipse = Ellipse(xy=(mean_x, mean_y), width=width, height=height,
                      angle=angle, **kwargs)
    ax.add_patch(ellipse)


def generate_moments(csv_file='paper_fig_moments.csv'):
    if not os.path.exists(csv_file):
        print(f"  Skipping moments: {csv_file} not found")
        return

    data = defaultdict(list)
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['mode']].append({
                'k': int(row['timestep']),
                'mx': float(row['mean_x']),
                'my': float(row['mean_y']),
                'cxx': float(row['cov_xx']),
                'cxy': float(row['cov_xy']),
                'cyy': float(row['cov_yy']),
            })

    for mode in data:
        data[mode].sort(key=lambda r: r['k'])

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Fig 2: Trajectory Moments with Uncertainty Ellipses (Proposition 1)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Draw origin
    ax.plot(0, 0, 'ks', markersize=10, label='Start', zorder=5)

    # Per-mode trajectories
    mode_styles = {
        'constant_velocity': ('#1f77b4', '-'),
        'turn_left': ('#2ca02c', '-'),
        'combined': ('#d62728', '--'),
    }

    ellipse_steps = {0, 5, 10, 15, 20}

    for mode, rows in data.items():
        color, ls = mode_styles.get(mode, ('#7f7f7f', '-'))
        xs = [r['mx'] for r in rows]
        ys = [r['my'] for r in rows]

        lw = 2.5 if mode == 'combined' else 2.0
        ax.plot(xs, ys, ls, color=color, linewidth=lw, label=mode, zorder=3)

        # Draw ellipses at selected steps
        for r in rows:
            if r['k'] in ellipse_steps and (r['cxx'] > 1e-9 or r['cyy'] > 1e-9):
                alpha = 0.3 if mode == 'combined' else 0.2
                draw_covariance_ellipse(
                    ax, r['mx'], r['my'], r['cxx'], r['cxy'], r['cyy'],
                    n_std=2.0, facecolor=color, edgecolor=color,
                    alpha=alpha, linewidth=1, zorder=2
                )
                # Draw step label
                ax.annotate(f'k={r["k"]}', (r['mx'], r['my']),
                           fontsize=7, ha='center', va='bottom',
                           color=color, alpha=0.8)

    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig('paper_fig_moments.png')
    print("  Saved paper_fig_moments.png")
    plt.close()


# ============================================================================
# Fig 3: Linearized Collision Constraints
# ============================================================================

def generate_constraints(csv_file='paper_fig_constraints.csv'):
    if not os.path.exists(csv_file):
        print(f"  Skipping constraints: {csv_file} not found")
        return

    rows = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'cid': int(row['constraint_id']),
                'k': int(row['timestep']),
                'ax': float(row['a_x']),
                'ay': float(row['a_y']),
                'b': float(row['b']),
                'ego_x': float(row['ego_x']),
                'ego_y': float(row['ego_y']),
                'obs_x': float(row['obs_x']),
                'obs_y': float(row['obs_y']),
                'sid': int(row['scenario_id']),
            })

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Fig 3: Linearized Collision Constraints (Section 7, Eq. 17-18)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Extract unique ego and obstacle positions across timesteps
    ego_xs = sorted(set(r['ego_x'] for r in rows))
    ego_ys = [next(r['ego_y'] for r in rows if r['ego_x'] == ex) for ex in ego_xs]
    ax.plot(ego_xs, ego_ys, 'b-o', linewidth=2.5, markersize=5, label='Ego trajectory', zorder=4)

    # Obstacle trajectories per scenario
    scenario_ids = sorted(set(r['sid'] for r in rows))
    cmap = plt.cm.Set2
    for i, sid in enumerate(scenario_ids):
        s_rows = [r for r in rows if r['sid'] == sid]
        # Unique timestep-obs positions
        seen = set()
        obs_xs, obs_ys = [], []
        for r in sorted(s_rows, key=lambda x: x['k']):
            key = (r['k'], r['obs_x'], r['obs_y'])
            if key not in seen:
                seen.add(key)
                obs_xs.append(r['obs_x'])
                obs_ys.append(r['obs_y'])
        color = cmap(i / max(1, len(scenario_ids) - 1))
        ax.plot(obs_xs, obs_ys, '--o', color=color, linewidth=1.5, markersize=4,
                alpha=0.7, label=f'Obstacle (Scen {sid})')

        # Draw constraint halfplanes as arrows
        for r in s_rows:
            if r['k'] % 2 == 0:  # every other timestep
                arrow_scale = 0.5
                ax.annotate('', xy=(r['ego_x'] + r['ax'] * arrow_scale,
                                    r['ego_y'] + r['ay'] * arrow_scale),
                           xytext=(r['ego_x'], r['ego_y']),
                           arrowprops=dict(arrowstyle='->', color=color, alpha=0.5, lw=1.2))

    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig('paper_fig_constraints.png')
    print("  Saved paper_fig_constraints.png")
    plt.close()


# ============================================================================
# Fig 4: Scenario Pruning Before/After
# ============================================================================

def generate_pruning(csv_file='paper_fig_pruning.csv'):
    if not os.path.exists(csv_file):
        print(f"  Skipping pruning: {csv_file} not found")
        return

    data = defaultdict(lambda: defaultdict(list))
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            stage = row['stage']
            key = (int(row['scenario_id']), int(row['obs_id']))
            data[stage][key].append(
                (int(row['timestep']), float(row['x']), float(row['y']))
            )

    stages = ['original', 'after_dominance', 'after_inactive']
    titles = ['Original (30 scenarios)', 'After Dominance Pruning', 'After Inactive Removal']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, stage in enumerate(stages):
        ax = axes[idx]
        stage_data = data[stage]
        num_scen = len(set(k[0] for k in stage_data.keys()))
        ax.set_title(f'{titles[idx]}\n({num_scen} scenarios)')
        ax.set_xlabel('X [m]')
        if idx == 0:
            ax.set_ylabel('Y [m]')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Ego reference (straight line from origin)
        ax.plot([0, 2], [0, 0], 'b-', linewidth=3, alpha=0.5, label='Ego ref')

        obs_colors_map = {0: '#d62728', 1: '#ff7f0e'}
        for (sid, oid), points in stage_data.items():
            points.sort(key=lambda p: p[0])
            xs = [p[1] for p in points]
            ys = [p[2] for p in points]
            color = obs_colors_map.get(oid, '#7f7f7f')
            ax.plot(xs, ys, '-', color=color, alpha=0.3, linewidth=1)

        # Legend entries for obstacles
        for oid, color in obs_colors_map.items():
            ax.plot([], [], '-', color=color, linewidth=2, label=f'Obstacle {oid}')

        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)

    fig.suptitle('Fig 4: Scenario Pruning — Algorithm 3 (Dominance) + Algorithm 4 (Inactive Removal)', fontsize=14)
    plt.tight_layout()
    plt.savefig('paper_fig_pruning.png')
    print("  Saved paper_fig_pruning.png")
    plt.close()


# ============================================================================
# Fig 5: Mode Weight Adaptation Over Time
# ============================================================================

def generate_weight_evolution(csv_file='paper_fig_weight_evolution.csv'):
    if not os.path.exists(csv_file):
        print(f"  Skipping weight_evolution: {csv_file} not found")
        return

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    true_modes = {}
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row['step'])
            wt = row['weight_type']
            mode = row['mode']
            w = float(row['weight'])
            data[wt][step][mode] = w
            true_modes[step] = row['true_mode']

    weight_types = ['UNIFORM', 'FREQUENCY', 'RECENCY']
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Collect all mode names
    all_modes = sorted(set(
        mode for wt in data for step in data[wt] for mode in data[wt][step]
    ))

    for idx, wt in enumerate(weight_types):
        ax = axes[idx]
        ax.set_title(f'{wt} Weights')
        if idx == 2:
            ax.set_xlabel('Timestep')
        ax.set_ylabel('Weight')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        steps_sorted = sorted(data[wt].keys())
        if not steps_sorted:
            continue

        # Build weight arrays for stacked area
        mode_weights = {}
        for mode in all_modes:
            mode_weights[mode] = [data[wt][s].get(mode, 0.0) for s in steps_sorted]

        # Stacked area chart
        bottom = np.zeros(len(steps_sorted))
        for mode in all_modes:
            vals = np.array(mode_weights[mode])
            color = get_mode_color(mode)
            ax.fill_between(steps_sorted, bottom, bottom + vals,
                           alpha=0.6, color=color, label=mode)
            bottom += vals

        # Mode switch lines
        ax.axvline(x=20, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(x=40, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

        # True mode annotations
        ax.text(10, 0.95, 'CV', ha='center', va='top', fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax.text(30, 0.95, 'Turn Left', ha='center', va='top', fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax.text(50, 0.95, 'Decel', ha='center', va='top', fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        if idx == 0:
            ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8)

    fig.suptitle('Fig 5: Mode Weight Adaptation Over Time (Section 4)', fontsize=14)
    plt.tight_layout()
    plt.savefig('paper_fig_weight_evolution.png')
    print("  Saved paper_fig_weight_evolution.png")
    plt.close()


# ============================================================================
# Fig 6: Full MPC Avoidance Loop (Animated)
# ============================================================================

def generate_mpc_loop(csv_main='paper_fig_mpc_loop.csv',
                      csv_scen='paper_fig_mpc_scenarios.csv'):
    if not os.path.exists(csv_main):
        print(f"  Skipping mpc_loop: {csv_main} not found")
        return

    # Load main data - group by step
    step_data = defaultdict(lambda: {
        'ego_x': 0, 'ego_y': 0, 'ego_theta': 0, 'ego_v': 0,
        'obs_x': 0, 'obs_y': 0, 'goal_x': 15.0, 'goal_y': 0.0,
        'plan': [], 'num_scenarios': 0, 'num_constraints': 0
    })

    with open(csv_main) as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row['step'])
            sd = step_data[step]
            sd['ego_x'] = float(row['ego_x'])
            sd['ego_y'] = float(row['ego_y'])
            sd['ego_theta'] = float(row['ego_theta'])
            sd['ego_v'] = float(row['ego_v'])
            sd['obs_x'] = float(row['obs_x'])
            sd['obs_y'] = float(row['obs_y'])
            sd['goal_x'] = float(row['goal_x'])
            sd['goal_y'] = float(row['goal_y'])
            sd['num_scenarios'] = int(row['num_scenarios'])
            sd['num_constraints'] = int(row['num_constraints'])
            pk = int(row['plan_k'])
            sd['plan'].append((pk, float(row['plan_x']), float(row['plan_y'])))

    # Load scenario data
    scen_data = defaultdict(lambda: defaultdict(list))
    if os.path.exists(csv_scen):
        with open(csv_scen) as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(row['step'])
                key = (int(row['scenario_id']), int(row['obs_id']), row['mode'])
                scen_data[step][key].append(
                    (int(row['timestep']), float(row['pred_x']), float(row['pred_y']))
                )

    steps = sorted(step_data.keys())
    if not steps:
        print("  No data in mpc_loop CSV")
        return

    # --- Create animated GIF ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    def update(frame_idx):
        ax.clear()
        step = steps[frame_idx]
        sd = step_data[step]

        # Determine view window around ego
        cx, cy = sd['ego_x'], sd['ego_y']
        ax.set_xlim(cx - 5, cx + 10)
        ax.set_ylim(cy - 5, cy + 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'MPC Avoidance Loop (Step {step})')

        # Goal
        ax.plot(sd['goal_x'], sd['goal_y'], 'g*', markersize=20, zorder=5, label='Goal')

        # Ego trail
        trail_steps = [s for s in steps if s <= step]
        trail_x = [step_data[s]['ego_x'] for s in trail_steps]
        trail_y = [step_data[s]['ego_y'] for s in trail_steps]
        ax.plot(trail_x, trail_y, 'b-', linewidth=2, alpha=0.4)

        # Ego vehicle rectangle
        ex, ey, eth = sd['ego_x'], sd['ego_y'], sd['ego_theta']
        L, W = 0.8, 0.4
        corners = np.array([[-L/2, -W/2], [L/2, -W/2], [L/2, W/2], [-L/2, W/2]])
        c, s_val = np.cos(eth), np.sin(eth)
        R = np.array([[c, -s_val], [s_val, c]])
        corners_g = (R @ corners.T).T + np.array([ex, ey])
        ego_patch = plt.Polygon(corners_g, closed=True, facecolor='royalblue',
                                edgecolor='black', alpha=0.9, linewidth=1.5, zorder=5)
        ax.add_patch(ego_patch)

        # Obstacle
        obs_circle = plt.Circle((sd['obs_x'], sd['obs_y']), 0.5,
                                 color='red', alpha=0.8, zorder=5)
        ax.add_patch(obs_circle)
        ax.plot(sd['obs_x'], sd['obs_y'], 'rx', markersize=8, zorder=6)

        # Planned trajectory
        plan = sorted(sd['plan'], key=lambda p: p[0])
        if len(plan) > 1:
            px = [p[1] for p in plan]
            py = [p[2] for p in plan]
            ax.plot(px, py, 'b--', linewidth=2, alpha=0.7, label='Planned traj')

        # Scenario fans
        modes_shown = set()
        for (sid, oid, mode), pts in scen_data[step].items():
            pts.sort(key=lambda p: p[0])
            if len(pts) > 1:
                xs = [p[1] for p in pts]
                ys = [p[2] for p in pts]
                color = get_mode_color(mode)
                ax.plot(xs, ys, '-', color=color, alpha=0.4, linewidth=1)
                modes_shown.add(mode)

        # Legend for modes
        for mode in sorted(modes_shown):
            ax.plot([], [], '-', color=get_mode_color(mode), linewidth=2, label=mode)

        # Info box
        dist = np.sqrt((sd['ego_x'] - sd['obs_x'])**2 + (sd['ego_y'] - sd['obs_y'])**2)
        info = (f"v={sd['ego_v']:.2f} m/s\n"
                f"dist={dist:.2f} m\n"
                f"scen={sd['num_scenarios']}")
        ax.text(0.02, 0.98, info, transform=ax.transAxes,
                verticalalignment='top', fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        ax.legend(loc='upper right', fontsize=8)
        return []

    # Skip frames for speed
    frame_step = max(1, len(steps) // 80)
    frames = list(range(0, len(steps), frame_step))

    print(f"  Creating MPC loop animation with {len(frames)} frames...")
    anim = FuncAnimation(fig, update, frames=frames, blit=False, interval=150, repeat=False)
    writer = PillowWriter(fps=8)
    anim.save('paper_fig_mpc_loop.gif', writer=writer)
    print("  Saved paper_fig_mpc_loop.gif")
    plt.close()

    # --- Create summary PNG with snapshots ---
    snapshot_fracs = [0.0, 0.375, 0.75]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, frac in enumerate(snapshot_fracs):
        step_idx = int(frac * (len(steps) - 1))
        step = steps[step_idx]
        sd = step_data[step]
        ax = axes[idx]

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X [m]')
        if idx == 0:
            ax.set_ylabel('Y [m]')
        ax.set_title(f'Step {step}')

        # Full ego trail up to this step
        trail_steps = [s for s in steps if s <= step]
        trail_x = [step_data[s]['ego_x'] for s in trail_steps]
        trail_y = [step_data[s]['ego_y'] for s in trail_steps]
        ax.plot(trail_x, trail_y, 'b-', linewidth=2, alpha=0.5)

        # Full obstacle trail
        obs_trail_x = [step_data[s]['obs_x'] for s in trail_steps]
        obs_trail_y = [step_data[s]['obs_y'] for s in trail_steps]
        ax.plot(obs_trail_x, obs_trail_y, 'r--', linewidth=1.5, alpha=0.5)

        # Current positions
        ax.plot(sd['ego_x'], sd['ego_y'], 'b^', markersize=12, zorder=5)
        obs_circle = plt.Circle((sd['obs_x'], sd['obs_y']), 0.5,
                                 color='red', alpha=0.6, zorder=5)
        ax.add_patch(obs_circle)
        ax.plot(sd['goal_x'], sd['goal_y'], 'g*', markersize=15, zorder=5)

        # Scenario fans
        for (sid, oid, mode), pts in scen_data[step].items():
            pts.sort(key=lambda p: p[0])
            if len(pts) > 1:
                xs = [p[1] for p in pts]
                ys = [p[2] for p in pts]
                ax.plot(xs, ys, '-', color=get_mode_color(mode), alpha=0.4, linewidth=1)

        # Auto-range
        all_x = trail_x + obs_trail_x + [sd['goal_x']]
        all_y = trail_y + obs_trail_y + [sd['goal_y']]
        if all_x and all_y:
            margin = 3
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    fig.suptitle('Fig 6: Full MPC Avoidance Loop — Snapshots (Algorithm 2)', fontsize=14)
    plt.tight_layout()
    plt.savefig('paper_fig_mpc_loop.png')
    print("  Saved paper_fig_mpc_loop.png")
    plt.close()


# ============================================================================
# Fig 7: Epsilon Guarantee Curve
# ============================================================================

def generate_epsilon(csv_file='paper_fig_epsilon.csv'):
    if not os.path.exists(csv_file):
        print(f"  Skipping epsilon: {csv_file} not found")
        return

    rows = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'S': int(row['num_scenarios']),
                'd': int(row['dimension']),
                'N': int(row['horizon']),
                'eps': float(row['epsilon_effective']),
                'eps_req': float(row['epsilon_required']),
                'beta': float(row['beta']),
            })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: epsilon vs S for different d (beta=0.01)
    ax = axes[0]
    ax.set_xlabel('Number of Scenarios (S)')
    ax.set_ylabel('Effective Epsilon')
    ax.set_title('Effective Epsilon vs Scenario Count')
    ax.grid(True, alpha=0.3)

    dim_colors = {30: '#1f77b4', 60: '#ff7f0e', 120: '#2ca02c'}
    for d in [30, 60, 120]:
        subset = [r for r in rows if r['d'] == d and r['beta'] == 0.01]
        if not subset:
            continue
        Ss = [r['S'] for r in subset]
        epss = [r['eps'] for r in subset]
        N_val = subset[0]['N']
        ax.plot(Ss, epss, '-', color=dim_colors[d], linewidth=2,
                label=f'd={d} (N={N_val})')

        # Find intersection with eps=0.05
        for i in range(len(epss) - 1):
            if epss[i] >= 0.05 >= epss[i+1]:
                S_intersect = Ss[i] + (0.05 - epss[i]) / (epss[i+1] - epss[i]) * (Ss[i+1] - Ss[i])
                ax.plot(S_intersect, 0.05, 'ko', markersize=6, zorder=5)
                ax.annotate(f'S={S_intersect:.0f}', (S_intersect, 0.05),
                           textcoords='offset points', xytext=(5, 10), fontsize=8)
                break

    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Target eps=0.05')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right')

    # Right: epsilon vs S for different beta (d=60)
    ax = axes[1]
    ax.set_xlabel('Number of Scenarios (S)')
    ax.set_ylabel('Effective Epsilon')
    ax.set_title('Effect of Confidence Parameter Beta (d=60)')
    ax.grid(True, alpha=0.3)

    beta_colors = {0.01: '#1f77b4', 0.05: '#ff7f0e', 0.1: '#2ca02c'}
    # Filter for beta sweep (d=60, not beta=0.01 from first sweep unless also in second)
    beta_data = [r for r in rows if r['d'] == 60]

    for beta in [0.01, 0.05, 0.1]:
        subset = [r for r in beta_data if abs(r['beta'] - beta) < 1e-6]
        if not subset:
            continue
        # Deduplicate by S (take last occurrence)
        by_s = {}
        for r in subset:
            by_s[r['S']] = r
        subset = sorted(by_s.values(), key=lambda r: r['S'])

        Ss = [r['S'] for r in subset]
        epss = [r['eps'] for r in subset]
        ax.plot(Ss, epss, '-', color=beta_colors[beta], linewidth=2,
                label=f'beta={beta}')

    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Target eps=0.05')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right')

    fig.suptitle('Fig 7: Epsilon Guarantee Curve (Theorem 1)', fontsize=14)
    plt.tight_layout()
    plt.savefig('paper_fig_epsilon.png')
    print("  Saved paper_fig_epsilon.png")
    plt.close()


# ============================================================================
# Fig 8: OT Weight Comparison
# ============================================================================

def generate_ot_weights(csv_file='paper_fig_ot_weights.csv'):
    if not os.path.exists(csv_file):
        print(f"  Skipping ot_weights: {csv_file} not found")
        return

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row['step'])
            method = row['method']
            mode = row['mode']
            w = float(row['weight'])
            data[method][step][mode] = w

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    # Top: Bar chart comparing weights at step 29 (last step)
    ax = axes[0]
    ax.set_title('Mode Weight Comparison at Step 29')
    ax.set_ylabel('Weight')

    methods = ['frequency', 'recency', 'wasserstein']
    all_modes = sorted(set(
        mode for method in data for step in data[method]
        for mode in data[method][step]
    ))

    x = np.arange(len(all_modes))
    width = 0.25
    method_colors = {'frequency': '#1f77b4', 'recency': '#ff7f0e', 'wasserstein': '#2ca02c'}

    for i, method in enumerate(methods):
        step_29 = data[method].get(29, {})
        if not step_29:
            # Try the last available step
            if data[method]:
                last_step = max(data[method].keys())
                step_29 = data[method][last_step]
        vals = [step_29.get(mode, 0.0) for mode in all_modes]
        ax.bar(x + i * width, vals, width, label=method.capitalize(),
               color=method_colors[method], alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(all_modes, rotation=30, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom: Time-series of OT (Wasserstein) weight evolution
    ax = axes[1]
    ax.set_title('Wasserstein Weight Evolution Over Time')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Weight')
    ax.grid(True, alpha=0.3)

    wass_data = data.get('wasserstein', {})
    if wass_data:
        steps_sorted = sorted(wass_data.keys())
        for mode in all_modes:
            weights_over_time = [wass_data[s].get(mode, 0.0) for s in steps_sorted]
            color = get_mode_color(mode)
            ax.plot(steps_sorted, weights_over_time, '-o', color=color,
                    linewidth=2, markersize=3, label=mode)
        ax.legend(loc='best', fontsize=8)
        ax.set_ylim(-0.05, 1.05)
    else:
        # Show frequency as fallback if no wasserstein data
        freq_data = data.get('frequency', {})
        steps_sorted = sorted(freq_data.keys())
        for mode in all_modes:
            weights_over_time = [freq_data[s].get(mode, 0.0) for s in steps_sorted]
            color = get_mode_color(mode)
            ax.plot(steps_sorted, weights_over_time, '-o', color=color,
                    linewidth=2, markersize=3, label=mode)
        ax.legend(loc='best', fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.text(0.5, 0.5, '(Wasserstein data unavailable,\nshowing frequency)',
                transform=ax.transAxes, ha='center', va='center', fontsize=10,
                style='italic', alpha=0.5)

    fig.suptitle('Fig 8: OT Weight Comparison (Extension Section A)', fontsize=14)
    plt.tight_layout()
    plt.savefig('paper_fig_ot_weights.png')
    print("  Saved paper_fig_ot_weights.png")
    plt.close()


# ============================================================================
# Dispatch table
# ============================================================================

FIGURE_GENERATORS = {
    'scenario_fan': generate_scenario_fan,
    'moments': generate_moments,
    'constraints': generate_constraints,
    'pruning': generate_pruning,
    'weight_evolution': generate_weight_evolution,
    'mpc_loop': generate_mpc_loop,
    'epsilon': generate_epsilon,
    'ot_weights': generate_ot_weights,
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_paper_figures.py [figure_name | all]")
        print(f"Available figures: {', '.join(FIGURE_GENERATORS.keys())}")
        sys.exit(1)

    target = sys.argv[1]

    if target == 'all':
        print("Generating all paper figures...")
        for name, gen_func in FIGURE_GENERATORS.items():
            print(f"\n--- {name} ---")
            try:
                gen_func()
            except Exception as e:
                print(f"  ERROR generating {name}: {e}")
    elif target in FIGURE_GENERATORS:
        print(f"Generating {target}...")
        FIGURE_GENERATORS[target]()
    else:
        print(f"Unknown figure: {target}")
        print(f"Available: {', '.join(FIGURE_GENERATORS.keys())}")
        sys.exit(1)

    print("\nDone!")


if __name__ == '__main__':
    main()
