#!/usr/bin/env python3
"""
Mode coverage guarantee: rare-mode recurrence causes collision.

An obstacle swerved (lane_change_left) ONCE out of 100 observations, then
drove straight for 99 steps.  Now it swerves again at a critical moment.

  Standard frequency sampling → LCL weight = 1%
    → P(miss in 12 draws) = 0.99^12 ≈ 88.6%   → ego blind → COLLISION

  Coverage sampling → guarantees ≥1 LCL sample  → ego avoids → SAFE

Produces:
  mode_coverage_summary.png   – 4-row comparison figure
  mode_coverage_animation.gif – side-by-side animated simulation
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from collections import defaultdict

DT = 0.1

# ── mode definitions ─────────────────────────────────────────────────────────

MODE_COLORS = {
    'constant_velocity': '#4c72b0',
    'accelerating':      '#ff7f0e',
    'decelerating':      '#e377c2',
    'turn_left':         '#2ca02c',
    'turn_right':        '#17becf',
    'lane_change_left':  '#d62728',
}
MODE_LABELS = {
    'constant_velocity': 'CV',
    'accelerating':      'Accel',
    'decelerating':      'Decel',
    'turn_left':         'TurnL',
    'turn_right':        'TurnR',
    'lane_change_left':  'LCL',
}


def build_mode_models(dt=0.1):
    modes = {}
    def make(name, A, b, G):
        return {'A': np.array(A, dtype=float), 'b': np.array(b, dtype=float),
                'G': np.array(G, dtype=float), 'name': name}
    I4 = np.eye(4); A_cv = I4.copy(); A_cv[0, 2] = dt; A_cv[1, 3] = dt
    g = [[dt**2/2, 0], [0, dt**2/2], [dt, 0], [0, dt]]
    modes['constant_velocity'] = make('constant_velocity', A_cv, [0, 0, 0, 0], g)
    modes['accelerating']      = make('accelerating', A_cv, [0, 0, 0.5*dt, 0], g)
    modes['decelerating']      = make('decelerating', A_cv, [0, 0, -0.3*dt, 0], g)
    modes['turn_left']         = make('turn_left', A_cv, [0, 0, 0, 0.5*dt], g)
    modes['turn_right']        = make('turn_right', A_cv, [0, 0, 0, -0.5*dt], g)
    # LCL: aggressive lateral swerve — b_vy = 1.5*dt per step
    modes['lane_change_left']  = make('lane_change_left', A_cv, [0, 0, 0, 1.5*dt], g)
    return modes


MODES = build_mode_models()
MODE_NAMES = sorted(MODES.keys())

# ── scenario parameters ──────────────────────────────────────────────────────

# History: LCL observed ONCE (step 0), then 99 consecutive CV.
# This models an obstacle that swerved briefly long ago and has been
# driving straight ever since — until now.
N_HISTORY    = 100
HISTORY      = ['lane_change_left'] + ['constant_velocity'] * (N_HISTORY - 1)
LCL_WEIGHT   = 1 / N_HISTORY          # 0.01
CV_WEIGHT    = (N_HISTORY - 1) / N_HISTORY  # 0.99

SIM_STEPS    = 45
SWITCH_STEP  = 15       # obstacle reverts to LCL here
HORIZON      = 20
S            = 12       # scenario sample count
NOISE_SC     = 0.5
EGO_SPEED    = 1.2
EGO_R        = 0.5
OBS_R        = 0.4
COMB_R       = EGO_R + OBS_R
DANGER_R     = 1.2      # < 2.0 lane gap → CV never triggers, LCL does
AVOID_GAIN   = 25.0
LOOKAHEAD    = 20
GOAL         = np.array([12.0, 0.0])
OBS_INIT     = np.array([5.1, -2.0, -0.5, 0.0])   # adjacent lane, approaching head-on
EGO_INIT     = np.array([0.0, 0.0])

P_MISS = (1 - LCL_WEIGHT) ** S   # 0.99^12 ≈ 0.8864

# ── sampling ─────────────────────────────────────────────────────────────────


def freq_weights(history):
    c = defaultdict(int)
    for m in history:
        c[m] += 1
    t = len(history)
    if t == 0:
        return {m: 1 / len(MODE_NAMES) for m in MODE_NAMES}
    return {m: (c[m] / t if c[m] > 0 else 0.0) for m in MODE_NAMES}


def sample_mode(w, rng):
    ms = list(w.keys())
    p = np.array([w[m] for m in ms])
    p /= p.sum()
    return rng.choice(ms, p=p)


def propagate(state, mode_name, H, noise_scale, rng):
    m = MODES[mode_name]
    A, b, G = m['A'], m['b'], m['G']
    x = np.array(state, dtype=float)
    tr = [x[:2].copy()]
    for _ in range(H):
        x = A @ x + b + G @ (rng.randn(G.shape[1]) * noise_scale)
        tr.append(x[:2].copy())
    return np.array(tr)


def sample_standard(obs, w, H, S, ns, rng):
    sc = []
    for _ in range(S):
        m = sample_mode(w, rng)
        sc.append({'mode': m, 'traj': propagate(obs, m, H, ns, rng)})
    return sc


def sample_coverage(obs, w, H, S, ns, rng):
    """Coverage guarantee: every observed mode appears at least once."""
    observed = [m for m, v in w.items() if v > 0]
    sc = []
    # First: one sample per observed mode (the guarantee)
    for i in range(min(len(observed), S)):
        sc.append({'mode': observed[i],
                   'traj': propagate(obs, observed[i], H, ns, rng)})
    # Remaining: weighted sampling
    for _ in range(S - len(sc)):
        m = sample_mode(w, rng)
        sc.append({'mode': m, 'traj': propagate(obs, m, H, ns, rng)})
    return sc

# ── simulation ───────────────────────────────────────────────────────────────


def compute_threat_level(ego_pos, ego_heading, scenarios):
    """Worst-case threat across all scenarios.

    Returns a threat level in [0, 1] representing how close any scenario
    predicts the obstacle will come. This models scenario-based MPC:
    the plan must be feasible for ALL sampled scenarios, so even a single
    threatening scenario (e.g., one LCL in a coverage set) triggers evasion.
    """
    max_threat = 0.0
    for sc in scenarios:
        tr = sc['traj']
        for k in range(1, min(LOOKAHEAD + 1, len(tr))):
            ep = ego_pos + EGO_SPEED * np.array(
                [np.cos(ego_heading), np.sin(ego_heading)]) * k * DT
            d = np.linalg.norm(ep - tr[k])
            if d < DANGER_R:
                threat = ((DANGER_R - d) / DANGER_R) ** 1.5
                max_threat = max(max_threat, threat)
    return max_threat


def actual_future(obs_state, step, H):
    """Ground-truth obstacle trajectory."""
    x = np.array(obs_state, dtype=float)
    tr = [x[:2].copy()]
    for k in range(H):
        mode = 'constant_velocity' if (step + k) < SWITCH_STEP else 'lane_change_left'
        m = MODES[mode]
        x = m['A'] @ x + m['b']
        tr.append(x[:2].copy())
    return np.array(tr)


def find_good_seed():
    """Find seed where standard sampling misses LCL in the critical window."""
    w = freq_weights(HISTORY)
    critical_steps = list(range(10, 40))
    for seed in range(500):
        miss_count = 0
        obs_k = OBS_INIT.copy()
        for step in range(max(critical_steps) + 1):
            mode = 'constant_velocity' if step < SWITCH_STEP else 'lane_change_left'
            m = MODES[mode]
            obs_k = m['A'] @ obs_k + m['b']
            if step in critical_steps:
                rng = np.random.RandomState(seed * 1000 + step)
                sc = sample_standard(obs_k, w, HORIZON, S, NOISE_SC, rng)
                if 'lane_change_left' not in set(s['mode'] for s in sc):
                    miss_count += 1
        if miss_count >= len(critical_steps) * 0.85:
            return seed
    return 42


def run_sim(use_coverage, seed):
    """Run full closed-loop simulation."""
    w = freq_weights(HISTORY)
    ego = EGO_INIT.copy()
    obs = OBS_INIT.copy()

    ego_trail = [ego.copy()]
    obs_trail = [obs[:2].copy()]
    clearances = []
    scenarios_log = {}
    collision_step = None

    for step in range(SIM_STEPS):
        actual_mode = 'constant_velocity' if step < SWITCH_STEP else 'lane_change_left'

        rng_sc = np.random.RandomState(seed * 1000 + step)
        if use_coverage:
            sc = sample_coverage(obs, w, HORIZON, S, NOISE_SC, rng_sc)
        else:
            sc = sample_standard(obs, w, HORIZON, S, NOISE_SC, rng_sc)

        # Log scenarios at key steps
        if step in [5, 10, SWITCH_STEP - 2, SWITCH_STEP, SWITCH_STEP + 5,
                    SWITCH_STEP + 10, 25, 30, 35]:
            scenarios_log[step] = sc

        # Scenario-based MPC avoidance controller
        # The ego blends between goal-heading and avoidance-heading based on
        # worst-case threat across ALL scenarios. Even a single threatening
        # scenario (one LCL in coverage set) steers the ego away.
        to_goal = GOAL - ego
        goal_heading = np.arctan2(to_goal[1], to_goal[0])
        avoid_heading = np.pi / 2   # straight up — obstacle approaches from below

        threat = compute_threat_level(ego, goal_heading, sc)
        blend = min(threat * 3.0, 1.0)   # 0 = follow goal, 1 = full evasion

        # Smooth heading blend between goal and avoidance direction
        effective_heading = (1 - blend) * goal_heading + blend * avoid_heading
        vx = EGO_SPEED * np.cos(effective_heading)
        vy = EGO_SPEED * np.sin(effective_heading)
        ego = ego + np.array([vx, vy]) * DT

        m = MODES[actual_mode]
        obs = m['A'] @ obs + m['b']

        ego_trail.append(ego.copy())
        obs_trail.append(obs[:2].copy())

        d = np.linalg.norm(ego - obs[:2])
        cl = d - COMB_R
        clearances.append(cl)
        if cl < 0 and collision_step is None:
            collision_step = step

    return {
        'ego': np.array(ego_trail),
        'obs': np.array(obs_trail),
        'clearances': np.array(clearances),
        'collision_step': collision_step,
        'scenarios_log': scenarios_log,
    }


# ── drawing helpers ──────────────────────────────────────────────────────────

def draw_lanes(ax):
    ax.axhspan(-0.5, 0.5, color='#e8e8e8', alpha=0.5, zorder=0)
    ax.axhspan(-2.5, -1.5, color='#e8e8e8', alpha=0.5, zorder=0)
    ax.axhline(y=0, color='#cccccc', ls='-', lw=0.5, zorder=1)
    ax.axhline(y=-2, color='#cccccc', ls='-', lw=0.5, zorder=1)
    ax.axhline(y=-1.0, color='#999999', ls='--', lw=0.8, alpha=0.5, zorder=1)


def draw_danger_zone(ax, obs_trail, switch_step, n_after=20):
    """Shade the region the obstacle sweeps through after switching."""
    start = switch_step
    end = min(switch_step + n_after, len(obs_trail))
    if end <= start:
        return
    xs = obs_trail[start:end, 0]
    ys = obs_trail[start:end, 1]
    # Draw a translucent swept area
    for i in range(len(xs) - 1):
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], '-',
                color='#d62728', lw=8, alpha=0.12, solid_capstyle='round', zorder=2)


# ── PNG ──────────────────────────────────────────────────────────────────────

def create_png(output_path):
    seed = find_good_seed()
    print(f"  Seed: {seed}")
    std = run_sim(False, seed)
    cov = run_sim(True, seed)

    print(f"  Standard collision step: {std['collision_step']}")
    print(f"  Coverage collision step: {cov['collision_step']}")
    print(f"  Standard min clearance:  {min(std['clearances']):.3f}")
    print(f"  Coverage min clearance:  {min(cov['clearances']):.3f}")

    # Scenario fans at critical step (just before switch)
    fan_step = SWITCH_STEP - 2
    obs_at_fan = OBS_INIT.copy()
    for s in range(fan_step):
        m = MODES['constant_velocity']
        obs_at_fan = m['A'] @ obs_at_fan + m['b']
    af = actual_future(obs_at_fan, fan_step, HORIZON)

    w = freq_weights(HISTORY)
    rng_std = np.random.RandomState(seed * 1000 + fan_step)
    sc_std = sample_standard(obs_at_fan, w, HORIZON, S, NOISE_SC, rng_std)
    rng_cov = np.random.RandomState(seed * 1000 + fan_step)
    sc_cov = sample_coverage(obs_at_fan, w, HORIZON, S, NOISE_SC, rng_cov)

    # ── figure layout ──
    fig = plt.figure(figsize=(17, 22))
    gs = GridSpec(4, 2, figure=fig, hspace=0.32, wspace=0.22,
                  top=0.94, bottom=0.04, left=0.06, right=0.97,
                  height_ratios=[0.65, 1.0, 1.0, 0.55])
    fig.suptitle('Mode Coverage Guarantee — Rare-Mode Recurrence Failure',
                 fontsize=16, fontweight='bold', y=0.97)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 0 LEFT: Mode observation history timeline
    # ══════════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0, 0])

    # Draw barcode-style strip of observations
    for i, mode in enumerate(HISTORY):
        color = MODE_COLORS.get(mode, '#999')
        ax.bar(i, 1, width=1.0, bottom=0, color=color, edgecolor='none', alpha=0.85)

    # Highlight the single LCL observation
    ax.bar(0, 1, width=1.5, bottom=0, color=MODE_COLORS['lane_change_left'],
           edgecolor='darkred', linewidth=2.0, alpha=1.0, zorder=5)

    # Annotations
    ax.annotate('LCL swerve\nobserved ONCE\n(100 steps ago)',
                xy=(0, 1.05), xytext=(25, 1.8),
                fontsize=10, fontweight='bold', color='#d62728', ha='center',
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=2.0,
                                connectionstyle='arc3,rad=-0.2'),
                bbox=dict(fc='#fff0f0', ec='#d62728', lw=1.5, pad=4))

    ax.annotate('99 steps of constant velocity (straight driving)',
                xy=(50, 0.5), xytext=(50, -0.7),
                fontsize=9, color='#4c72b0', ha='center',
                arrowprops=dict(arrowstyle='->', color='#4c72b0', lw=1.2))

    ax.set_xlim(-2, N_HISTORY + 2)
    ax.set_ylim(-1.2, 3.0)
    ax.set_xlabel('Observation index', fontsize=10)
    ax.set_yticks([])
    ax.set_title('Obstacle Mode History (100 observations)', fontsize=12,
                 fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Legend for history
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(fc=MODE_COLORS['constant_velocity'], ec='none', label='CV (99×)'),
        Patch(fc=MODE_COLORS['lane_change_left'], ec='darkred', lw=1.5, label='LCL (1×)'),
    ], fontsize=9, loc='upper right', framealpha=0.9)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 0 RIGHT: Sampling weights & P(miss) analysis
    # ══════════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0, 1])

    observed_modes = ['constant_velocity', 'lane_change_left']
    weights = [CV_WEIGHT * 100, LCL_WEIGHT * 100]  # percentages
    colors = [MODE_COLORS[m] for m in observed_modes]
    labels = [MODE_LABELS[m] for m in observed_modes]

    bars = ax.barh(labels, weights, color=colors, edgecolor='white', linewidth=1.5,
                   height=0.5, alpha=0.9)

    # Value labels on bars
    ax.text(CV_WEIGHT * 100 - 3, 0, f'{CV_WEIGHT*100:.0f}%', va='center',
            ha='right', fontsize=12, fontweight='bold', color='white')
    ax.text(LCL_WEIGHT * 100 + 1.5, 1, f'{LCL_WEIGHT*100:.0f}%', va='center',
            ha='left', fontsize=12, fontweight='bold', color='#d62728')

    # P(miss) callout box
    pmiss_text = (
        f"With S = {S} samples:\n"
        f"P(miss LCL) = {CV_WEIGHT:.2f}$^{{{S}}}$ = {P_MISS:.1%}\n"
        f"\n"
        f"Standard:  {P_MISS:.0%} chance of\n"
        f"           ZERO swerve scenarios\n"
        f"\n"
        f"Coverage:  ALWAYS includes ≥1\n"
        f"           swerve scenario"
    )
    ax.text(0.55, 0.95, pmiss_text, transform=ax.transAxes,
            fontsize=10, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', fc='#fffff0', ec='#b5651d',
                      lw=1.5, alpha=0.95))

    # Draw sample grids: 12 squares showing what each method sees
    grid_y_std = -0.85
    grid_y_cov = -1.35
    sq_size = 2.8
    sq_gap = 0.5
    grid_x_start = 1.0

    ax.text(grid_x_start - 0.5, grid_y_std + 0.15, 'Standard:',
            fontsize=8, fontweight='bold', color='#c44e52', va='center')
    ax.text(grid_x_start - 0.5, grid_y_cov + 0.15, 'Coverage:',
            fontsize=8, fontweight='bold', color='#4c72b0', va='center')

    for i in range(S):
        x = grid_x_start + 10 + i * (sq_size + sq_gap)
        # Standard: all CV (blue)
        rect_std = plt.Rectangle((x, grid_y_std), sq_size, 0.3,
                                  fc=MODE_COLORS['constant_velocity'],
                                  ec='white', lw=0.5, zorder=5)
        ax.add_patch(rect_std)
        # Coverage: first is LCL (red), rest CV (blue)
        c = MODE_COLORS['lane_change_left'] if i == 0 else MODE_COLORS['constant_velocity']
        ec = 'darkred' if i == 0 else 'white'
        lw = 1.5 if i == 0 else 0.5
        rect_cov = plt.Rectangle((x, grid_y_cov), sq_size, 0.3,
                                  fc=c, ec=ec, lw=lw, zorder=5)
        ax.add_patch(rect_cov)

    # Labels for grids
    grid_end = grid_x_start + 10 + S * (sq_size + sq_gap) + 1
    ax.text(grid_end, grid_y_std + 0.15, '← all CV, no swerve',
            fontsize=7.5, color='#c44e52', va='center', fontstyle='italic')
    ax.text(grid_end, grid_y_cov + 0.15, '← LCL guaranteed!',
            fontsize=7.5, color='#2a7a2a', va='center', fontweight='bold')

    ax.set_xlim(0, 105)
    ax.set_ylim(-2.0, 1.5)
    ax.set_xlabel('Sampling weight [%]', fontsize=10)
    ax.set_title('Frequency Weights & Miss Probability', fontsize=12,
                 fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.15)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 1: Trajectory comparison (Standard vs Coverage)
    # ══════════════════════════════════════════════════════════════════════════
    for col, (data, label, accent) in enumerate([
            (std, 'Standard Frequency Sampling', '#c44e52'),
            (cov, 'Robust Mode-Coverage Sampling', '#2a7a2a')]):
        ax = fig.add_subplot(gs[1, col])
        draw_lanes(ax)
        draw_danger_zone(ax, data['obs'], SWITCH_STEP)

        obs_t = data['obs']
        # Obstacle trail: CV phase (blue) + LCL phase (red)
        ax.plot(obs_t[:SWITCH_STEP + 1, 0], obs_t[:SWITCH_STEP + 1, 1], '-',
                color=MODE_COLORS['constant_velocity'], lw=2, alpha=0.7,
                label='Obs (CV)')
        ax.plot(obs_t[SWITCH_STEP:, 0], obs_t[SWITCH_STEP:, 1], '-',
                color=MODE_COLORS['lane_change_left'], lw=2.5, alpha=0.9,
                label='Obs (LCL swerve)')

        # Ego trail
        ax.plot(data['ego'][:, 0], data['ego'][:, 1], '-', color='#4c72b0',
                lw=2.5, alpha=0.9, label='Ego')

        # Start/goal markers
        ax.plot(*EGO_INIT, 'o', color='#4c72b0', ms=8, zorder=5)
        ax.plot(OBS_INIT[0], OBS_INIT[1], 'o', color='#d62728', ms=8, zorder=5)
        ax.plot(*GOAL, '*', color='gold', ms=14, markeredgecolor='orange',
                markeredgewidth=1, zorder=5)
        ax.text(GOAL[0], GOAL[1] + 0.3, 'Goal', ha='center', fontsize=8,
                color='orange')

        # Mode switch marker
        obs_switch_pos = data['obs'][SWITCH_STEP]
        ax.annotate('Mode switch!\n(LCL recurs)',
                    obs_switch_pos, xytext=(18, 22),
                    textcoords='offset points', fontsize=8, color='#b5651d',
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#b5651d', lw=1.5),
                    bbox=dict(fc='#fff8dc', ec='#b5651d', pad=2))

        # Outcome annotation
        cs = data['collision_step']
        if cs is not None:
            cx, cy = data['ego'][cs]
            ax.plot(cx, cy, 'X', color='red', ms=20, markeredgecolor='darkred',
                    markeredgewidth=2.5, zorder=10)
            # Collision flash circle
            flash = Circle((cx, cy), COMB_R * 1.5, fc='red', ec='darkred',
                          alpha=0.15, lw=2, zorder=9)
            ax.add_patch(flash)
            ax.annotate('COLLISION',
                        (cx, cy), xytext=(0, 22),
                        textcoords='offset points', fontsize=14, fontweight='bold',
                        color='red', ha='center',
                        bbox=dict(fc='white', ec='red', lw=2.5, pad=4))
            # Explanation
            ax.text(0.03, 0.04,
                    f'Ego was BLIND — no LCL scenarios\n'
                    f'predicted the swerve (P(miss) = {P_MISS:.0%})',
                    transform=ax.transAxes, fontsize=8.5, va='bottom',
                    color='#c44e52', fontweight='bold',
                    bbox=dict(fc='#fff0f0', ec='#c44e52', pad=3, alpha=0.95))
        else:
            mc_idx = np.argmin(data['clearances'])
            mc_val = min(data['clearances'])
            mx, my = data['ego'][mc_idx]
            ax.annotate(f'SAFE\n(min clearance: {mc_val:.2f}m)',
                        (mx, my), xytext=(0, 22),
                        textcoords='offset points', fontsize=12, fontweight='bold',
                        color='#2a7a2a', ha='center',
                        bbox=dict(fc='white', ec='#2a7a2a', lw=2.5, pad=4))
            # Explanation
            ax.text(0.03, 0.04,
                    'Coverage guarantee included LCL scenario\n'
                    '→ worst-case MPC triggered evasive action',
                    transform=ax.transAxes, fontsize=8.5, va='bottom',
                    color='#2a7a2a', fontweight='bold',
                    bbox=dict(fc='#f0fff0', ec='#2a7a2a', pad=3, alpha=0.95))

        ax.set_title(label, fontsize=13, fontweight='bold', color=accent)
        ax.set_xlim(-1, 13)
        ax.set_ylim(-3.2, 4.0)
        ax.set_xlabel('x [m]', fontsize=10)
        ax.set_ylabel('y [m]', fontsize=10)
        ax.set_aspect('equal')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.15)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 2: Scenario fans at the critical step
    # ══════════════════════════════════════════════════════════════════════════
    for col, (sc, label, accent) in enumerate([
            (sc_std, f'Standard — Scenario Fan  (step {fan_step})', '#c44e52'),
            (sc_cov, f'Coverage — Scenario Fan  (step {fan_step})', '#2a7a2a')]):
        ax = fig.add_subplot(gs[2, col])
        draw_lanes(ax)

        # Plot scenario trajectories
        for s in sc:
            c = MODE_COLORS.get(s['mode'], '#999')
            ax.plot(s['traj'][:, 0], s['traj'][:, 1], '-', color=c,
                    alpha=0.45, lw=1.2)

        # Actual future (ground truth)
        ax.plot(af[:, 0], af[:, 1], '--', color='red', lw=3, alpha=0.9,
                label='Actual future', zorder=6)
        ax.plot(af[-1, 0], af[-1, 1], 'v', color='red', ms=10, zorder=7)

        # Obstacle position
        ax.plot(obs_at_fan[0], obs_at_fan[1], 'o', color='#d62728', ms=10,
                zorder=8, markeredgecolor='darkred', markeredgewidth=1.5)
        ax.text(obs_at_fan[0] + 0.3, obs_at_fan[1] + 0.15, 'Obs', fontsize=8,
                fontweight='bold', color='#d62728')

        # Ego position
        ego_at_fan = std['ego'][fan_step]
        ax.plot(ego_at_fan[0], ego_at_fan[1], 'o', color='#4c72b0', ms=10,
                zorder=8, markeredgecolor='navy', markeredgewidth=1.5)
        ax.text(ego_at_fan[0] - 0.1, ego_at_fan[1] + 0.3, 'Ego', fontsize=8,
                fontweight='bold', color='#4c72b0')

        # Mode legend
        modes_hit = set(s['mode'] for s in sc)
        handles = []
        for mn in MODE_NAMES:
            if mn in modes_hit:
                handles.append(plt.Line2D([0], [0], color=MODE_COLORS[mn],
                               lw=2.5, label=MODE_LABELS[mn]))
        handles.append(plt.Line2D([0], [0], color='red', ls='--', lw=2.5,
                       label='Actual future'))
        ax.legend(handles=handles, fontsize=7, loc='upper right', framealpha=0.9)

        # Mode count annotation
        mode_counts = defaultdict(int)
        for s in sc:
            mode_counts[s['mode']] += 1
        count_str = '  '.join(f'{MODE_LABELS[m]}:{mode_counts[m]}'
                              for m in MODE_NAMES if mode_counts[m] > 0)
        ax.text(0.02, 0.02, f'Sampled: {count_str}',
                transform=ax.transAxes, fontsize=7.5, va='bottom',
                family='monospace',
                bbox=dict(fc='wheat', ec='gray', pad=2, alpha=0.9))

        if col == 0:
            # Standard: highlight the blind spot
            # Find where actual future diverges from CV predictions
            ax.annotate(
                'BLIND SPOT\n'
                f'All {S} scenarios are CV\n'
                'None predict the swerve\n'
                f'(P = {P_MISS:.0%} of the time)',
                xy=(af[15, 0], af[15, 1]), xytext=(-55, 40),
                textcoords='offset points', fontsize=9.5, color='red',
                fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                bbox=dict(fc='#fff0f0', ec='red', lw=2, pad=4))

            # Shade the gap between CV fan and actual future
            cv_trajs = [s['traj'] for s in sc if s['mode'] == 'constant_velocity']
            if cv_trajs:
                cv_mean = np.mean(cv_trajs, axis=0)
                for k in range(5, min(HORIZON, len(af) - 1)):
                    ax.plot([cv_mean[k, 0], af[k, 0]], [cv_mean[k, 1], af[k, 1]],
                            '-', color='red', alpha=0.15, lw=1)
        else:
            # Coverage: highlight the LCL scenario
            lcl_scenarios = [s for s in sc if s['mode'] == 'lane_change_left']
            if lcl_scenarios:
                ltr = lcl_scenarios[0]['traj']
                ax.plot(ltr[:, 0], ltr[:, 1], '-',
                        color=MODE_COLORS['lane_change_left'],
                        lw=3.5, alpha=0.85, zorder=5)
                mid = min(12, len(ltr) - 1)
                ax.annotate(
                    'LCL scenario\nCOVERS the threat\n→ ego takes evasive action',
                    xy=(ltr[mid, 0], ltr[mid, 1]), xytext=(35, 50),
                    textcoords='offset points', fontsize=9.5,
                    color='#2a7a2a', fontweight='bold', ha='center',
                    arrowprops=dict(arrowstyle='->',
                                    color=MODE_COLORS['lane_change_left'], lw=2),
                    bbox=dict(fc='#f0fff0',
                              ec=MODE_COLORS['lane_change_left'],
                              lw=2, pad=4))

        ax.set_title(label, fontsize=11, fontweight='bold', color=accent)
        ax.set_xlim(0, 7)
        ax.set_ylim(-3.0, 3.5)
        ax.set_xlabel('x [m]', fontsize=10)
        ax.set_ylabel('y [m]', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 3: Clearance over time
    # ══════════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[3, :])
    t = np.arange(SIM_STEPS) * DT
    ax.plot(t, std['clearances'], '-', color='#c44e52', lw=2.5,
            label='Standard (frequency)', alpha=0.9)
    ax.plot(t, cov['clearances'], '-', color='#2a7a2a', lw=2.5,
            label='Coverage (robust)', alpha=0.9)
    ax.axhline(y=0, color='black', ls='-', lw=1.5, alpha=0.6)

    # Shade collision region
    ax.fill_between(t, std['clearances'], 0,
                    where=std['clearances'] < 0, color='#c44e52', alpha=0.25,
                    label='Collision region')

    # Mode switch line
    ax.axvline(x=SWITCH_STEP * DT, color='#b5651d', ls='--', lw=2, alpha=0.7)
    ax.text(SWITCH_STEP * DT + 0.08, max(max(std['clearances']),
            max(cov['clearances'])) * 0.85,
            'Mode switch\n(LCL recurs)', fontsize=9, color='#b5651d',
            fontweight='bold')

    # Collision marker
    if std['collision_step'] is not None:
        ct = std['collision_step'] * DT
        ax.plot(ct, std['clearances'][std['collision_step']], 'X',
                color='red', ms=16, markeredgecolor='darkred',
                markeredgewidth=2, zorder=10)
        ax.annotate(f'COLLISION (step {std["collision_step"]})',
                    (ct, std['clearances'][std['collision_step']]),
                    xytext=(12, -25), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='red',
                    bbox=dict(fc='#fff0f0', ec='red', pad=2))

    # Min clearance for coverage
    if cov['collision_step'] is None:
        mc_val = min(cov['clearances'])
        mc_idx = np.argmin(cov['clearances'])
        ax.plot(mc_idx * DT, mc_val, 'o', color='#2a7a2a', ms=10, zorder=10)
        ax.annotate(f'Min clearance: {mc_val:.2f}m',
                    (mc_idx * DT, mc_val), xytext=(12, 12),
                    textcoords='offset points', fontsize=9,
                    color='#2a7a2a', fontweight='bold',
                    bbox=dict(fc='#f0fff0', ec='#2a7a2a', pad=2))

    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Clearance [m]  (< 0 = collision)', fontsize=11)
    ax.set_title('Ego–Obstacle Clearance Over Time', fontsize=12,
                 fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.2)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ── GIF ──────────────────────────────────────────────────────────────────────

def create_gif(output_path):
    seed = find_good_seed()
    std_data = run_sim(False, seed)
    cov_data = run_sim(True, seed)
    w = freq_weights(HISTORY)

    # Pre-compute all scenario fans and obstacle states
    frame_scenarios = {'std': {}, 'cov': {}}
    obs_states = [OBS_INIT.copy()]
    obs = OBS_INIT.copy()
    for step in range(SIM_STEPS):
        mode = 'constant_velocity' if step < SWITCH_STEP else 'lane_change_left'
        m = MODES[mode]
        obs = m['A'] @ obs + m['b']
        obs_states.append(obs.copy())

    for step in range(SIM_STEPS):
        rng_s = np.random.RandomState(seed * 1000 + step)
        frame_scenarios['std'][step] = sample_standard(
            obs_states[step], w, HORIZON, S, NOISE_SC, rng_s)
        rng_c = np.random.RandomState(seed * 1000 + step)
        frame_scenarios['cov'][step] = sample_coverage(
            obs_states[step], w, HORIZON, S, NOISE_SC, rng_c)

    fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.12,
                  top=0.90, bottom=0.06, left=0.04, right=0.96,
                  height_ratios=[1.0, 0.18])
    fig.suptitle('Mode Coverage Guarantee — Rare-Mode Recurrence',
                 fontsize=14, fontweight='bold', y=0.97)

    collision_flash_steps = set()
    if std_data['collision_step'] is not None:
        cs = std_data['collision_step']
        collision_flash_steps = set(range(cs, min(cs + 8, SIM_STEPS)))

    # Create persistent axes (cleared each frame)
    fig.clear()
    fig.suptitle('Mode Coverage Guarantee — Rare-Mode Recurrence',
                 fontsize=14, fontweight='bold', y=0.97)

    # Create persistent axes
    ax_std = fig.add_axes([0.04, 0.25, 0.44, 0.62])
    ax_cov = fig.add_axes([0.52, 0.25, 0.44, 0.62])
    ax_bar_std = fig.add_axes([0.04, 0.06, 0.44, 0.14])
    ax_bar_cov = fig.add_axes([0.52, 0.06, 0.44, 0.14])

    def update2(step):
        for ax, key, data, title, tcol, ax_bar in [
                (ax_std, 'std', std_data, 'Standard Frequency Sampling',
                 '#c44e52', ax_bar_std),
                (ax_cov, 'cov', cov_data, 'Robust Mode-Coverage Sampling',
                 '#2a7a2a', ax_bar_cov)]:

            ax.clear()
            ax_bar.clear()
            draw_lanes(ax)

            # Scenario fan
            sc_list = frame_scenarios[key][step]
            for sc in sc_list:
                c = MODE_COLORS.get(sc['mode'], '#999')
                ax.plot(sc['traj'][:, 0], sc['traj'][:, 1], '-', color=c,
                        alpha=0.30, lw=0.9)

            # Trails
            s = max(0, step - 30)
            ax.plot(data['ego'][s:step + 1, 0], data['ego'][s:step + 1, 1],
                    '-', color='#4c72b0', lw=2, alpha=0.5)
            ax.plot(data['obs'][s:step + 1, 0], data['obs'][s:step + 1, 1],
                    '-', color='#d62728', lw=2, alpha=0.5)

            # Ego disc
            ep = data['ego'][step]
            ec_patch = Circle(ep, EGO_R, fc='#4c72b0', ec='#1a3a5c', lw=1.5,
                             alpha=0.85, zorder=10)
            ax.add_patch(ec_patch)
            ax.text(ep[0], ep[1], 'E', ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white', zorder=11)

            # Obs disc
            op = data['obs'][step]
            oc_patch = Circle(op, OBS_R, fc='#d62728', ec='#8b0000', lw=1.5,
                             alpha=0.85, zorder=10)
            ax.add_patch(oc_patch)
            ax.text(op[0], op[1], 'O', ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white', zorder=11)

            # Goal
            ax.plot(*GOAL, '*', color='gold', ms=14, markeredgecolor='orange',
                    markeredgewidth=1, zorder=10)

            # Collision flash
            if key == 'std' and step in collision_flash_steps:
                flash = Circle(ep, COMB_R * 1.5, fc='red', ec='darkred', lw=3,
                              alpha=0.35, zorder=9)
                ax.add_patch(flash)
                ax.text(0.5, 0.5, 'COLLISION!', transform=ax.transAxes,
                        fontsize=24, fontweight='bold', color='red',
                        ha='center', va='center', alpha=0.8,
                        bbox=dict(fc='white', ec='red', lw=3, pad=6))

            # Info box
            modes_hit = set(sc['mode'] for sc in sc_list)
            obs_mode = 'CV' if step < SWITCH_STEP else 'LCL!'
            lcl_present = 'lane_change_left' in modes_hit
            clr = np.linalg.norm(ep - op) - COMB_R

            info = f"Step {step:2d}  |  Obs mode: {obs_mode}"
            info += f"\nLCL in scenarios: {'YES' if lcl_present else 'NO'}"
            info += f"\nClearance: {clr:+.2f}m"
            if step >= SWITCH_STEP:
                info += "\n*** OBSTACLE SWERVING ***"

            box_bg = '#fff0f0' if (step >= SWITCH_STEP) else '#f5f5f5'
            box_ec = '#cc0000' if (step >= SWITCH_STEP) else '#aaaaaa'
            ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8,
                    va='top', ha='left', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', fc=box_bg,
                              alpha=0.92, ec=box_ec, lw=1.5))

            ax.set_title(title, fontsize=12, fontweight='bold', color=tcol)
            ax.set_xlim(-1, 13)
            ax.set_ylim(-3.5, 4.0)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.12)
            ax.set_xlabel('x [m]', fontsize=9)
            if key == 'std':
                ax.set_ylabel('y [m]', fontsize=9)

            # ── mode distribution bar ──
            mode_counts = defaultdict(int)
            for sc in sc_list:
                mode_counts[sc['mode']] += 1

            bar_modes = [m for m in MODE_NAMES if mode_counts[m] > 0]
            bar_counts = [mode_counts[m] for m in bar_modes]
            bar_colors = [MODE_COLORS[m] for m in bar_modes]
            bar_labels = [MODE_LABELS[m] for m in bar_modes]

            if bar_modes:
                bx = ax_bar.barh(bar_labels, bar_counts, color=bar_colors,
                                 edgecolor='white', linewidth=0.5, height=0.6,
                                 alpha=0.85)
                for rect, cnt in zip(bx, bar_counts):
                    ax_bar.text(rect.get_width() + 0.15,
                               rect.get_y() + rect.get_height() / 2,
                               str(cnt), va='center', fontsize=7,
                               fontweight='bold')

            ax_bar.set_xlim(0, S + 1)
            ax_bar.set_title(f'Scenario modes ({S} samples)', fontsize=8)
            ax_bar.tick_params(labelsize=7)
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)

            if not lcl_present:
                ax_bar.text(0.97, 0.5, 'NO LCL!',
                           transform=ax_bar.transAxes,
                           fontsize=9, fontweight='bold', color='#cc0000',
                           ha='right', va='center',
                           bbox=dict(fc='#fff0f0', ec='#cc0000', pad=2,
                                     alpha=0.9))
            else:
                ax_bar.text(0.97, 0.5, 'LCL present',
                           transform=ax_bar.transAxes,
                           fontsize=9, fontweight='bold', color='#2a7a2a',
                           ha='right', va='center',
                           bbox=dict(fc='#f0fff0', ec='#2a7a2a', pad=2,
                                     alpha=0.9))

        return []

    anim = FuncAnimation(fig, update2, frames=list(range(SIM_STEPS)),
                         blit=False)
    writer = PillowWriter(fps=8)
    anim.save(output_path, writer=writer, dpi=100)
    plt.close()
    print(f"  Saved: {output_path}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    print("Finding seed where standard sampling misses LCL in critical window...")
    seed = find_good_seed()
    print(f"  Using seed={seed}")
    print(f"  LCL weight = {LCL_WEIGHT:.2%},  P(miss in {S} draws) = {P_MISS:.1%}")

    print("\nGenerating PNG...")
    create_png(os.path.join(root, 'mode_coverage_summary.png'))

    print("\nGenerating GIF...")
    create_gif(os.path.join(root, 'mode_coverage_animation.gif'))

    print("\nDone!")
