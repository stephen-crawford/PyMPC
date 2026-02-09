#!/usr/bin/env python3
"""
Visualize how OT learning deforms initially isotropic Gaussian "hyperspheres"
into anisotropic ellipsoids as the dynamics parameters are learned.

Produces an animated GIF showing:
  - Per-obstacle uncertainty ellipses evolving over time
  - Initial isotropic (circular) uncertainty -> learned anisotropic (elliptical)
  - Side-by-side: prior G (sphere) vs learned G (ellipsoid) at each frame
  - Color-coded by obstacle / mode

Usage:
    python3 test/unit/visualize_ot_deformation.py
    python3 test/unit/visualize_ot_deformation.py --duration 8 --output my_deformation.gif
"""

import sys
import os
import types

# ============================================================================
# Mock external dependencies (same pattern as visualize_ot_learning.py)
# ============================================================================

class MockPredictionStep:
    def __init__(self, position, angle, major_radius, minor_radius):
        self.position = position
        self.angle = angle
        self.major_radius = major_radius
        self.minor_radius = minor_radius

_mock_planning_types = types.ModuleType('planning.types')
_mock_planning_types.DynamicObstacle = type('DynamicObstacle', (), {})
_mock_planning_types.PredictionStep = MockPredictionStep
_mock_planning_types.Prediction = type('Prediction', (), {})
_mock_planning_types.PredictionType = type('PredictionType', (), {'GAUSSIAN': 1})

_mock_utils = types.ModuleType('utils.utils')
_mock_utils.LOG_DEBUG = lambda msg: None
_mock_utils.LOG_WARN = lambda msg: None
_mock_utils.LOG_INFO = lambda msg: None

sys.modules['planning'] = types.ModuleType('planning')
sys.modules['planning.types'] = _mock_planning_types
sys.modules['utils'] = types.ModuleType('utils')
sys.modules['utils.utils'] = _mock_utils
sys.modules['utils.math_tools'] = types.ModuleType('utils.math_tools')
sys.modules['utils.math_tools_impl'] = types.ModuleType('utils.math_tools_impl')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

_su_path = os.path.join(PROJECT_ROOT, 'modules', 'constraints', 'scenario_utils')
_mock_modules = types.ModuleType('modules')
_mock_modules.__path__ = [os.path.join(PROJECT_ROOT, 'modules')]
sys.modules['modules'] = _mock_modules
_mock_constraints = types.ModuleType('modules.constraints')
_mock_constraints.__path__ = [os.path.join(PROJECT_ROOT, 'modules', 'constraints')]
sys.modules['modules.constraints'] = _mock_constraints
_mock_su = types.ModuleType('modules.constraints.scenario_utils')
_mock_su.__path__ = [_su_path]
sys.modules['modules.constraints.scenario_utils'] = _mock_su
sys.modules['modules.constraints.scenario_utils.math_utils'] = types.ModuleType('modules.constraints.scenario_utils.math_utils')
sys.modules['modules.constraints.scenario_utils.sampler'] = types.ModuleType('modules.constraints.scenario_utils.sampler')
sys.modules['modules.constraints.scenario_utils.scenario_module'] = types.ModuleType('modules.constraints.scenario_utils.scenario_module')

import importlib
_ot_mod = importlib.import_module('modules.constraints.scenario_utils.optimal_transport_predictor')

OptimalTransportPredictor = _ot_mod.OptimalTransportPredictor
TrajectoryObservation = _ot_mod.TrajectoryObservation
TrajectoryBuffer = _ot_mod.TrajectoryBuffer
OTWeightType = _ot_mod.OTWeightType
create_ot_predictor_with_standard_modes = _ot_mod.create_ot_predictor_with_standard_modes

# ============================================================================
# Standard imports
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Dict, Tuple

# ============================================================================
# Configuration
# ============================================================================

OBSTACLE_CONFIGS = [
    {
        "name": "Constant Velocity",
        "true_mode": "constant_velocity",
        "A": np.eye(4),  # Identity (state = [x,y,vx,vy])
        "b_true": np.array([0.0, 0.0, 0.0, 0.0]),
        "G_true_diag": np.array([0.01, 0.01, 0.05, 0.01]),  # Anisotropic: more noise in vx
        "initial_state": np.array([0.0, 0.0, 0.5, 0.0]),
        "color": "#1f77b4",
    },
    {
        "name": "Decelerating",
        "true_mode": "decelerating",
        "A": np.eye(4),
        "b_true": np.array([0.0, 0.0, -0.015, 0.0]),  # Deceleration bias
        "G_true_diag": np.array([0.01, 0.02, 0.03, 0.02]),  # More lateral noise
        "initial_state": np.array([0.0, 2.0, 0.6, 0.0]),
        "color": "#e377c2",
    },
    {
        "name": "Turning Left",
        "true_mode": "turn_left",
        "A": np.eye(4),
        "b_true": np.array([0.0, 0.0, -0.01, 0.02]),  # Lateral acceleration
        "G_true_diag": np.array([0.02, 0.02, 0.02, 0.04]),  # More vy noise
        "initial_state": np.array([0.0, -2.0, 0.4, 0.1]),
        "color": "#2ca02c",
    },
]

# Chi-squared 2-sigma threshold for 2 DOF
CHI2_2SIGMA = 5.9915

# ============================================================================
# Simulation engine
# ============================================================================

class DeformationSimulator:
    """
    Simulates obstacles with known anisotropic dynamics, then runs the OT
    dynamics learner to show how the initially isotropic G matrix deforms
    to match the true anisotropic covariance structure.
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.rng = np.random.default_rng(42)

        # Frames: list of dicts, each containing per-obstacle ellipse data
        self.frames: List[Dict] = []

    def run(self, duration: float = 8.0):
        num_steps = int(duration / self.dt)

        # Create one OT predictor per obstacle so we can track independently
        predictors = []
        states = []
        for cfg in OBSTACLE_CONFIGS:
            pred = create_ot_predictor_with_standard_modes(
                dt=self.dt, base_speed=0.5, buffer_size=500,
                min_samples_for_ot=5, uncertainty_scale=1.0,
                weight_type=OTWeightType.WASSERSTEIN,
            )
            predictors.append(pred)
            states.append(cfg["initial_state"].copy())

        # Initial G is isotropic (from create_obstacle_mode_models)
        # We'll track the learned G over time
        initial_G = 0.15 * np.eye(4)  # Approximate initial G from sampler.py

        # Track the most recently learned G for persistence across frames
        last_learned = {i: None for i in range(len(OBSTACLE_CONFIGS))}

        for step in range(num_steps):
            frame_data = {
                "step": step,
                "time": step * self.dt,
                "obstacles": [],
            }

            for obs_idx, cfg in enumerate(OBSTACLE_CONFIGS):
                state = states[obs_idx]
                pred = predictors[obs_idx]
                A = cfg["A"]
                b_true = cfg["b_true"]
                G_true = np.diag(cfg["G_true_diag"])

                # Add observation with TRUE velocity to the buffer
                # (observe() uses finite-diff which is noisy; we need
                # clean dynamics_state for accurate residual computation)
                if obs_idx not in pred.trajectory_buffers:
                    pred.trajectory_buffers[obs_idx] = TrajectoryBuffer(
                        obstacle_id=obs_idx, max_length=500)
                    pred._prev_observations[obs_idx] = (state[:2].copy(), np.zeros(2))

                obs = TrajectoryObservation(
                    timestep=step,
                    position=state[:2].copy(),
                    velocity=state[2:].copy(),
                    acceleration=np.zeros(2),
                    mode_id=cfg["true_mode"],
                )
                pred.trajectory_buffers[obs_idx].add_observation(obs)

                # Propagate true dynamics
                noise = G_true @ self.rng.standard_normal(4)
                new_state = A @ state + b_true + noise
                states[obs_idx] = new_state

                # Attempt dynamics learning every 5 steps after warmup
                learned_G = None
                learned_b = None
                if step >= 10 and step % 5 == 0:
                    result = pred.estimate_mode_dynamics(
                        obs_idx, cfg["true_mode"], A, self.dt)
                    if result is not None:
                        learned_b, learned_G = result
                        last_learned[obs_idx] = (learned_b, learned_G)

                # Use most recent learned values if available
                if last_learned[obs_idx] is not None:
                    learned_b, learned_G = last_learned[obs_idx]

                # Compute ellipse params for prior (isotropic) and learned
                # Project 4D covariance to 2D position space
                prior_cov_4d = initial_G @ initial_G.T
                prior_cov_2d = prior_cov_4d[:2, :2]

                if learned_G is not None:
                    learned_cov_4d = learned_G @ learned_G.T
                    learned_cov_2d = learned_cov_4d[:2, :2]
                else:
                    learned_cov_2d = prior_cov_2d.copy()

                # True covariance for reference
                true_cov_4d = G_true @ G_true.T
                true_cov_2d = true_cov_4d[:2, :2]

                frame_data["obstacles"].append({
                    "name": cfg["name"],
                    "position": state[:2].copy(),
                    "color": cfg["color"],
                    "prior_cov": prior_cov_2d,
                    "learned_cov": learned_cov_2d,
                    "true_cov": true_cov_2d,
                    "learned_b": learned_b,
                    "b_true": b_true,
                    "has_learned": learned_G is not None,
                })

                pred.advance_timestep()

            self.frames.append(frame_data)

        return self.frames


def cov_to_ellipse_params(cov_2d, scale=1.0):
    """Convert 2x2 covariance to (width, height, angle_deg) for Ellipse patch."""
    eigvals, eigvecs = np.linalg.eigh(cov_2d)
    # Clamp for numerical safety
    eigvals = np.maximum(eigvals, 1e-10)
    angle_deg = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    chi2_scale = np.sqrt(CHI2_2SIGMA)
    w = 2 * chi2_scale * np.sqrt(eigvals[1]) * scale
    h = 2 * chi2_scale * np.sqrt(eigvals[0]) * scale
    return w, h, angle_deg


def cov_volume(cov_2d):
    """Ellipse area from covariance: pi * sqrt(det(Sigma)) * chi2_scale^2."""
    det = max(np.linalg.det(cov_2d), 1e-20)
    return np.pi * CHI2_2SIGMA * np.sqrt(det)


# ============================================================================
# Animated GIF
# ============================================================================

def create_animation(frames, output_path, dt, fps=10):
    """Create an animated GIF showing ellipse deformation over time."""
    n_obs = len(OBSTACLE_CONFIGS)

    fig, axes = plt.subplots(2, n_obs, figsize=(6 * n_obs, 10))
    if n_obs == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        'OT Dynamics Learning: Gaussian Hypersphere Deformation',
        fontsize=14, fontweight='bold', y=0.98)

    # Ellipse display scale (amplify for visibility)
    display_scale = 8.0

    # Collect all positions for axis limits
    all_x = []
    all_y = []
    for f in frames:
        for obs in f["obstacles"]:
            all_x.append(obs["position"][0])
            all_y.append(obs["position"][1])

    x_min, x_max = min(all_x) - 1.0, max(all_x) + 1.0
    y_min, y_max = min(all_y) - 1.0, max(all_y) + 1.0

    # Track volume history for bottom row
    volume_history = {i: {"prior": [], "learned": [], "true": [], "time": []}
                      for i in range(n_obs)}

    # Pre-compute volume histories
    for f in frames:
        for obs_idx, obs in enumerate(f["obstacles"]):
            t = f["time"]
            volume_history[obs_idx]["time"].append(t)
            volume_history[obs_idx]["prior"].append(cov_volume(obs["prior_cov"]))
            volume_history[obs_idx]["learned"].append(cov_volume(obs["learned_cov"]))
            volume_history[obs_idx]["true"].append(cov_volume(obs["true_cov"]))

    def init():
        for row in axes:
            for ax in row:
                ax.clear()
        return []

    def update(frame_idx):
        frame = frames[frame_idx]
        artists = []

        for obs_idx in range(n_obs):
            obs = frame["obstacles"][obs_idx]
            cfg = OBSTACLE_CONFIGS[obs_idx]

            # --- Top row: Ellipse comparison ---
            ax = axes[0][obs_idx]
            ax.clear()

            pos = obs["position"]

            # Draw trajectory trace
            trace_x = []
            trace_y = []
            for f in frames[:frame_idx + 1]:
                p = f["obstacles"][obs_idx]["position"]
                trace_x.append(p[0])
                trace_y.append(p[1])
            ax.plot(trace_x, trace_y, '-', color=cfg["color"],
                    linewidth=1.0, alpha=0.4)
            ax.plot(pos[0], pos[1], 'o', color=cfg["color"],
                    markersize=6, zorder=10)

            # Prior ellipse (isotropic = circle) - dashed gray
            pw, ph, pa = cov_to_ellipse_params(obs["prior_cov"], display_scale)
            ell_prior = Ellipse(
                xy=pos, width=pw, height=ph, angle=pa,
                facecolor='none', edgecolor='gray',
                linestyle='--', linewidth=2.0, alpha=0.7,
                label='Prior (isotropic)')
            ax.add_patch(ell_prior)

            # True ellipse - dotted black
            tw, th, ta = cov_to_ellipse_params(obs["true_cov"], display_scale)
            ell_true = Ellipse(
                xy=pos, width=tw, height=th, angle=ta,
                facecolor='none', edgecolor='black',
                linestyle=':', linewidth=1.5, alpha=0.5,
                label='True covariance')
            ax.add_patch(ell_true)

            # Learned ellipse - solid colored
            lw_e, lh, la = cov_to_ellipse_params(obs["learned_cov"], display_scale)
            ell_learned = Ellipse(
                xy=pos, width=lw_e, height=lh, angle=la,
                facecolor=cfg["color"], edgecolor=cfg["color"],
                linewidth=2.5, alpha=0.25,
                label='Learned (OT)')
            ax.add_patch(ell_learned)

            # Axis styling
            margin = max(pw, ph, tw, th, lw_e, lh) / 2 + 0.3
            ax.set_xlim(pos[0] - margin, pos[0] + margin)
            ax.set_ylim(pos[1] - margin, pos[1] + margin)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)

            status = "Learned" if obs["has_learned"] else "Prior only"
            ax.set_title(
                f'{cfg["name"]} [{status}]\nt = {frame["time"]:.1f}s',
                fontsize=10, fontweight='bold', color=cfg["color"])

            if frame_idx == 0:
                ax.legend(loc='upper right', fontsize=7, framealpha=0.8)

            # --- Bottom row: Volume convergence ---
            ax2 = axes[1][obs_idx]
            ax2.clear()

            vh = volume_history[obs_idx]
            t_so_far = vh["time"][:frame_idx + 1]
            ax2.plot(t_so_far, vh["prior"][:frame_idx + 1],
                     '--', color='gray', linewidth=2.0, alpha=0.7,
                     label='Prior (isotropic)')
            ax2.plot(t_so_far, vh["learned"][:frame_idx + 1],
                     '-', color=cfg["color"], linewidth=2.5, alpha=0.9,
                     label='Learned (OT)')
            ax2.axhline(vh["true"][0], color='black', linestyle=':',
                        linewidth=1.5, alpha=0.5, label='True volume')

            ax2.set_xlabel('Time [s]', fontsize=9)
            ax2.set_ylabel('Ellipse Area [m$^2$]', fontsize=9)
            ax2.set_title(f'{cfg["name"]}: Volume Convergence', fontsize=10,
                          color=cfg["color"])
            ax2.legend(loc='upper right', fontsize=7, framealpha=0.8)
            ax2.grid(True, alpha=0.2)
            ax2.set_xlim(0, frames[-1]["time"])
            all_vols = vh["prior"] + vh["learned"] + vh["true"]
            ax2.set_ylim(0, max(all_vols) * 1.15)

        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        return artists

    # Sample frames for the GIF (every nth frame)
    total_frames = len(frames)
    frame_step = max(1, total_frames // 80)  # ~80 frames in GIF
    frame_indices = list(range(0, total_frames, frame_step))
    if frame_indices[-1] != total_frames - 1:
        frame_indices.append(total_frames - 1)

    print(f"  Creating animation with {len(frame_indices)} frames...")
    anim = FuncAnimation(fig, update, frames=frame_indices,
                         init_func=init, blit=False)

    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=100)
    plt.close()
    print(f"  Saved: {output_path}")

    return output_path


# ============================================================================
# Static summary figure
# ============================================================================

def create_summary_figure(frames, output_path, dt):
    """Create a static summary figure with key snapshots."""
    n_obs = len(OBSTACLE_CONFIGS)
    n_snapshots = 5  # Number of time snapshots

    fig, axes = plt.subplots(n_obs, n_snapshots + 1, figsize=(4 * (n_snapshots + 1), 4 * n_obs))
    if n_obs == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        'Gaussian Hypersphere Deformation via OT Dynamics Learning\n'
        '(2$\\sigma$ ellipses, amplified 8x for visibility)',
        fontsize=14, fontweight='bold', y=0.99)

    display_scale = 8.0
    snapshot_indices = np.linspace(0, len(frames) - 1, n_snapshots, dtype=int)

    for obs_idx in range(n_obs):
        cfg = OBSTACLE_CONFIGS[obs_idx]

        # Snapshot columns
        for col, fidx in enumerate(snapshot_indices):
            ax = axes[obs_idx][col]
            frame = frames[fidx]
            obs = frame["obstacles"][obs_idx]
            pos = obs["position"]

            # Trajectory trace up to this point
            trace_x = [frames[f]["obstacles"][obs_idx]["position"][0]
                       for f in range(fidx + 1)]
            trace_y = [frames[f]["obstacles"][obs_idx]["position"][1]
                       for f in range(fidx + 1)]
            ax.plot(trace_x, trace_y, '-', color=cfg["color"],
                    linewidth=0.8, alpha=0.3)
            ax.plot(pos[0], pos[1], 'o', color=cfg["color"],
                    markersize=5, zorder=10)

            # Prior (gray dashed circle)
            pw, ph, pa = cov_to_ellipse_params(obs["prior_cov"], display_scale)
            ell_prior = Ellipse(
                xy=pos, width=pw, height=ph, angle=pa,
                facecolor='none', edgecolor='gray',
                linestyle='--', linewidth=1.5, alpha=0.6)
            ax.add_patch(ell_prior)

            # True (black dotted)
            tw, th, ta = cov_to_ellipse_params(obs["true_cov"], display_scale)
            ell_true = Ellipse(
                xy=pos, width=tw, height=th, angle=ta,
                facecolor='none', edgecolor='black',
                linestyle=':', linewidth=1.2, alpha=0.4)
            ax.add_patch(ell_true)

            # Learned (solid fill)
            lw_e, lh, la = cov_to_ellipse_params(obs["learned_cov"], display_scale)
            ell_learned = Ellipse(
                xy=pos, width=lw_e, height=lh, angle=la,
                facecolor=cfg["color"], edgecolor=cfg["color"],
                linewidth=2.0, alpha=0.3)
            ax.add_patch(ell_learned)

            margin = max(pw, ph, tw, th, lw_e, lh) / 2 + 0.2
            ax.set_xlim(pos[0] - margin, pos[0] + margin)
            ax.set_ylim(pos[1] - margin, pos[1] + margin)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)

            t_label = f't = {frame["time"]:.1f}s'
            if col == 0:
                ax.set_ylabel(cfg["name"], fontsize=11, fontweight='bold',
                              color=cfg["color"])
            ax.set_title(t_label, fontsize=9)

        # Final column: volume convergence
        ax_vol = axes[obs_idx][n_snapshots]
        times = [f["time"] for f in frames]
        prior_vols = [cov_volume(f["obstacles"][obs_idx]["prior_cov"]) for f in frames]
        learned_vols = [cov_volume(f["obstacles"][obs_idx]["learned_cov"]) for f in frames]
        true_vol = cov_volume(frames[0]["obstacles"][obs_idx]["true_cov"])

        ax_vol.plot(times, prior_vols, '--', color='gray', linewidth=2.0,
                    alpha=0.7, label='Prior (sphere)')
        ax_vol.plot(times, learned_vols, '-', color=cfg["color"], linewidth=2.5,
                    alpha=0.9, label='Learned (OT)')
        ax_vol.axhline(true_vol, color='black', linestyle=':',
                       linewidth=1.5, alpha=0.5, label='True')
        ax_vol.set_xlabel('Time [s]', fontsize=9)
        ax_vol.set_ylabel('Ellipse Area', fontsize=9)
        ax_vol.set_title('Volume Convergence', fontsize=9)
        ax_vol.legend(fontsize=7, framealpha=0.8)
        ax_vol.grid(True, alpha=0.2)

    # Add legend for first subplot
    axes[0][0].plot([], [], '--', color='gray', linewidth=1.5, label='Prior (isotropic)')
    axes[0][0].plot([], [], ':', color='black', linewidth=1.2, label='True covariance')
    axes[0][0].plot([], [], '-', color=OBSTACLE_CONFIGS[0]["color"],
                    linewidth=2.0, alpha=0.5, label='Learned (OT)')
    axes[0][0].legend(loc='upper left', fontsize=7, framealpha=0.8)

    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    summary_path = output_path.replace('.gif', '_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved summary: {summary_path}")
    return summary_path


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Visualize OT Gaussian hypersphere deformation')
    parser.add_argument('--duration', '-d', type=float, default=8.0,
                        help='Simulation duration in seconds (default 8)')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Timestep (default 0.1)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output GIF path')
    parser.add_argument('--fps', type=int, default=10,
                        help='GIF frames per second (default 10)')
    args = parser.parse_args()

    output = args.output or os.path.join(
        os.path.dirname(__file__), '..', '..',
        'ot_gaussian_deformation.gif')
    output = os.path.abspath(output)

    print("=" * 60)
    print("OT Gaussian Hypersphere Deformation Visualization")
    print("=" * 60)
    print(f"  Duration: {args.duration}s, dt: {args.dt}s")
    print(f"  Obstacles: {len(OBSTACLE_CONFIGS)}")

    sim = DeformationSimulator(dt=args.dt)
    print("\nRunning simulation...")
    frames = sim.run(duration=args.duration)

    # Print final state
    final = frames[-1]
    print(f"\n--- Final state (t = {final['time']:.1f}s) ---")
    for obs_idx, obs in enumerate(final["obstacles"]):
        cfg = OBSTACLE_CONFIGS[obs_idx]
        prior_vol = cov_volume(obs["prior_cov"])
        learned_vol = cov_volume(obs["learned_cov"])
        true_vol = cov_volume(obs["true_cov"])
        ratio = learned_vol / true_vol if true_vol > 0 else float('inf')
        print(f"  {cfg['name']:20s}: prior_vol={prior_vol:.4f}  "
              f"learned_vol={learned_vol:.4f}  true_vol={true_vol:.4f}  "
              f"ratio={ratio:.2f}  {'converged' if obs['has_learned'] else 'no data'}")
        if obs["learned_b"] is not None:
            b_err = np.linalg.norm(obs["learned_b"] - obs["b_true"])
            print(f"{'':24s}b_err={b_err:.4f}  "
                  f"b_learned={obs['learned_b'][:2].round(4)}  "
                  f"b_true={obs['b_true'][:2].round(4)}")

    print("\nCreating animated GIF...")
    create_animation(frames, output, args.dt, fps=args.fps)

    print("\nCreating summary figure...")
    create_summary_figure(frames, output, args.dt)

    print("\nDone!")


if __name__ == '__main__':
    main()
