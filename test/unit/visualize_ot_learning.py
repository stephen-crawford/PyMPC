#!/usr/bin/env python3
"""
Visualize Optimal Transport Mode Weight Learning Per Obstacle.

Creates a multi-panel figure showing how the OT predictor independently
learns the true dynamics mode for each obstacle from observations.

Layout (4 rows x N_obstacles columns):
  Row 0: Trajectory overview with sight radius and full obstacle paths
  Row 1: Mode weight evolution over time (one subplot per obstacle)
  Row 2: Wasserstein prediction error over time
  Row 3: Final converged weight distribution (bar chart)

Can run standalone without CasADi.

Usage:
    python3 test/unit/visualize_ot_learning.py
    python3 test/unit/visualize_ot_learning.py --duration 15 --output my_plot.png
"""

import sys
import os
import types

# ============================================================================
# Mock external dependencies to avoid CasADi requirement
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

# Pre-register mocks before any project imports
sys.modules['planning'] = types.ModuleType('planning')
sys.modules['planning.types'] = _mock_planning_types
sys.modules['utils'] = types.ModuleType('utils')
sys.modules['utils.utils'] = _mock_utils
sys.modules['utils.math_tools'] = types.ModuleType('utils.math_tools')
sys.modules['utils.math_tools_impl'] = types.ModuleType('utils.math_tools_impl')

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

# Mock the scenario_utils package so __init__.py doesn't pull in CasADi deps
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

# Now safe to import the OT predictor directly
import importlib
_ot_mod = importlib.import_module('modules.constraints.scenario_utils.optimal_transport_predictor')

OptimalTransportPredictor = _ot_mod.OptimalTransportPredictor
OTWeightType = _ot_mod.OTWeightType
create_ot_predictor_with_standard_modes = _ot_mod.create_ot_predictor_with_standard_modes
wasserstein_distance = _ot_mod.wasserstein_distance
EmpiricalDistribution = _ot_mod.EmpiricalDistribution

# ============================================================================
# Standard imports (no CasADi needed)
# ============================================================================

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


# ============================================================================
# Obstacle Configurations
# ============================================================================

OBSTACLE_CONFIGS = [
    {
        "name": "Constant Velocity",
        "initial_mode": "constant_velocity",
        "available_modes": ["constant_velocity", "decelerating", "accelerating"],
        "true_behavior": "constant_velocity",
        "initial_velocity": [0.5, 0.0],
        "initial_position": [0, 0],
    },
    {
        "name": "Turning Left",
        "initial_mode": "constant_velocity",  # Wrong initial guess
        "available_modes": ["turn_left", "turn_right", "constant_velocity"],
        "true_behavior": "turn_left",
        "initial_velocity": [0.4, 0.1],
        "initial_position": [2, 1],
    },
    {
        "name": "Decelerating",
        "initial_mode": "constant_velocity",  # Wrong initial guess
        "available_modes": ["decelerating", "constant_velocity", "accelerating"],
        "true_behavior": "decelerating",
        "initial_velocity": [0.6, 0.0],
        "initial_position": [4, -1],
    },
    {
        "name": "Accelerating",
        "initial_mode": "constant_velocity",  # Wrong initial guess
        "available_modes": ["accelerating", "constant_velocity", "decelerating"],
        "true_behavior": "accelerating",
        "initial_velocity": [0.3, 0.0],
        "initial_position": [6, 0.5],
    },
    {
        "name": "Lane Change Right",
        "initial_mode": "constant_velocity",  # Wrong initial guess
        "available_modes": ["lane_change_right", "lane_change_left", "constant_velocity"],
        "true_behavior": "lane_change_right",
        "initial_velocity": [0.5, -0.1],
        "initial_position": [8, -0.5],
    },
    {
        "name": "Turning Right",
        "initial_mode": "constant_velocity",  # Wrong initial guess
        "available_modes": ["turn_right", "turn_left", "constant_velocity"],
        "true_behavior": "turn_right",
        "initial_velocity": [0.45, -0.05],
        "initial_position": [10, 0],
    },
]


# ============================================================================
# Simulation
# ============================================================================

class OTLearningSimulator:
    """
    Simulates obstacle motion and feeds observations to the OT predictor.
    Tracks per-obstacle mode weight history and prediction errors.
    """

    def __init__(self, dt: float = 0.1, horizon: int = 10, sight_radius: float = 5.0):
        self.dt = dt
        self.horizon = horizon
        self.sight_radius = sight_radius

        self.ot_predictor = create_ot_predictor_with_standard_modes(
            dt=dt, base_speed=0.5,
            buffer_size=200, min_samples_for_ot=10,
            uncertainty_scale=0.3,  # sharper discrimination between modes
            weight_type=OTWeightType.WASSERSTEIN,
        )

        self.obstacle_states: Dict[int, Dict] = {}
        self.prediction_errors: Dict[int, List[float]] = {}
        self.mode_weight_history: Dict[int, List[Optional[Dict[str, float]]]] = {}
        self.learned_mode_history: Dict[int, List[str]] = {}
        self.in_sight_history: Dict[int, List[bool]] = {}
        self.position_history: Dict[int, List[np.ndarray]] = {}

        # Gaussian prediction tracking
        self.prediction_snapshots: Dict[int, List] = {}   # per obs: [(capture_idx, [(mean,cov),...]), ...]
        self.ellipse_area_history: Dict[int, List] = {}    # per obs: [(capture_idx, area), ...]
        self.ground_truth_metrics: Dict[int, List] = {}    # per obs: [{step,k,mahal_sq,coverage_*,nll}, ...]

        # Ego position (constant velocity along x-axis, faster than obstacles
        # so it catches up and passes through the obstacle field)
        self.ego_position = np.array([0.0, 0.0])
        self.ego_velocity = 1.5  # m/s along x-axis
        self.ego_positions: List[np.ndarray] = []

    def initialize_obstacle(self, obs_id: int, position, velocity, true_mode: str):
        self.obstacle_states[obs_id] = {
            "position": np.array(position, dtype=float),
            "velocity": np.array(velocity, dtype=float),
            "true_mode": true_mode,
        }
        self.prediction_errors[obs_id] = []
        self.mode_weight_history[obs_id] = []
        self.learned_mode_history[obs_id] = []
        self.in_sight_history[obs_id] = []
        self.position_history[obs_id] = [np.array(position, dtype=float).copy()]
        self.prediction_snapshots[obs_id] = []
        self.ellipse_area_history[obs_id] = []
        self.ground_truth_metrics[obs_id] = []

    # Steady-state target velocities matching reference distributions
    # (from create_ot_predictor_with_standard_modes with base_speed=0.5)
    _MODE_TARGET_VEL = {
        'constant_velocity': np.array([0.5, 0.0]),
        'decelerating':      np.array([0.25, 0.0]),
        'accelerating':      np.array([0.75, 0.0]),
        'turn_left':         np.array([0.5 * np.cos(0.8), 0.5 * np.sin(0.8)]),
        'turn_right':        np.array([0.5 * np.cos(-0.8), 0.5 * np.sin(-0.8)]),
        'lane_change_left':  np.array([0.5, 0.3]),
        'lane_change_right': np.array([0.5, -0.3]),
    }

    def _simulate_step(self, obs_id: int) -> np.ndarray:
        """
        Advance obstacle one timestep using its true dynamics mode.

        Uses exponential smoothing toward each mode's characteristic velocity
        so that observed velocities converge to the reference distributions
        used by the OT predictor.  A small noise term keeps the distribution
        realistic (not a single point).
        """
        s = self.obstacle_states[obs_id]
        pos, vel, mode = s["position"], s["velocity"], s["true_mode"]

        target = self._MODE_TARGET_VEL.get(mode, vel)
        alpha = 0.25          # smoothing rate toward target
        noise_std = 0.02      # process noise

        vel = (1 - alpha) * vel + alpha * target
        vel = vel + np.random.randn(2) * noise_std

        new_pos = pos + vel * self.dt
        s["position"] = new_pos
        s["velocity"] = vel
        self.position_history[obs_id].append(new_pos.copy())
        return new_pos

    def _observe_and_predict(self, obs_id: int, config: Dict, sim_step: int = 0):
        """Feed observation to OT predictor, compute mode weights, predict.

        Skips observation and prediction if the obstacle is outside the
        ego vehicle's sight radius.  Tracks Gaussian prediction snapshots
        for ground truth evaluation.
        """
        s = self.obstacle_states[obs_id]
        pos, vel = s["position"], s["velocity"]

        # Check sight radius
        dist_to_ego = np.linalg.norm(self.ego_position - pos)
        in_sight = dist_to_ego <= self.sight_radius
        self.in_sight_history[obs_id].append(in_sight)

        if not in_sight:
            self.mode_weight_history[obs_id].append(None)
            return None

        observed_mode = None  # don't label until OT can infer

        buf = self.ot_predictor.trajectory_buffers.get(obs_id, [])
        if len(buf) > 15:
            weights = self.ot_predictor.compute_mode_weights(
                obs_id, config["available_modes"]
            )
            self.mode_weight_history[obs_id].append(weights.copy())
            observed_mode = max(weights, key=weights.get)
            self.learned_mode_history[obs_id].append(observed_mode)
        else:
            self.mode_weight_history[obs_id].append(None)

        self.ot_predictor.observe(obs_id, pos, mode_id=observed_mode)

        predictions = self.ot_predictor.predict_trajectory(
            obstacle_id=obs_id,
            current_position=pos,
            current_velocity=vel,
            horizon=self.horizon,
        )

        # Track Gaussian prediction snapshot
        if predictions:
            capture_idx = len(self.position_history[obs_id]) - 1
            snapshot = []
            for pred in predictions:
                mean = pred.position[:2].copy()
                cov = reconstruct_covariance(pred.angle, pred.major_radius, pred.minor_radius)
                snapshot.append((mean, cov))
            self.prediction_snapshots[obs_id].append((capture_idx, snapshot))
            # Record k=0 ellipse area for narrowing plot
            area = ellipse_area(predictions[0].major_radius, predictions[0].minor_radius)
            self.ellipse_area_history[obs_id].append((capture_idx, area))

        return predictions

    def _prediction_error(self, obs_id: int, predicted, steps_ahead: int = 5) -> float:
        """Wasserstein distance between predicted and simulated trajectories."""
        if steps_ahead > len(predicted) - 1:
            steps_ahead = len(predicted) - 1

        saved = {k: v.copy() if hasattr(v, 'copy') else v
                 for k, v in self.obstacle_states[obs_id].items()}
        saved_pos_len = len(self.position_history[obs_id])

        actuals = [saved["position"].copy()]
        for _ in range(steps_ahead):
            self._simulate_step(obs_id)
            actuals.append(self.obstacle_states[obs_id]["position"].copy())

        self.obstacle_states[obs_id] = saved
        del self.position_history[obs_id][saved_pos_len:]

        preds = [p.position[:2] for p in predicted[:steps_ahead + 1]]
        p_dist = EmpiricalDistribution.from_samples(np.array(preds))
        a_dist = EmpiricalDistribution.from_samples(np.array(actuals))
        err = wasserstein_distance(p_dist, a_dist)
        self.prediction_errors[obs_id].append(err)
        return err

    def evaluate_ground_truth(self, eval_horizons=(1, 3, 5, 8, 10)):
        """Compare prediction snapshots against actual future positions.

        For each obstacle and each stored prediction snapshot, look up the
        ground truth position at capture_idx + k and compute Mahalanobis
        distance, coverage checks, and negative log-likelihood.
        """
        for obs_id, snapshots in self.prediction_snapshots.items():
            pos_hist = self.position_history[obs_id]
            max_idx = len(pos_hist) - 1
            for capture_idx, snapshot in snapshots:
                for k in eval_horizons:
                    if k >= len(snapshot):
                        continue
                    future_idx = capture_idx + k
                    if future_idx > max_idx:
                        continue
                    gt_pos = pos_hist[future_idx]
                    mean, cov = snapshot[k]
                    d2 = mahalanobis_distance_sq(gt_pos, mean, cov)
                    cov_check = coverage_check(gt_pos, mean, cov)
                    nll = gaussian_nll(gt_pos, mean, cov)
                    self.ground_truth_metrics[obs_id].append({
                        'capture_idx': capture_idx,
                        'k': k,
                        'mahal_sq': d2,
                        'coverage_1sigma': cov_check['1sigma'],
                        'coverage_2sigma': cov_check['2sigma'],
                        'coverage_3sigma': cov_check['3sigma'],
                        'nll': nll,
                    })

    def run(self, duration: float = 10.0, verbose: bool = True) -> Dict:
        np.random.seed(42)
        num_steps = int(duration / self.dt)

        for i, cfg in enumerate(OBSTACLE_CONFIGS):
            self.initialize_obstacle(
                obs_id=i,
                position=cfg.get("initial_position", [float(i) * 3.0, 0.0]),
                velocity=cfg.get("initial_velocity", [0.5, 0.0]),
                true_mode=cfg["true_behavior"],
            )

        # Reset ego position
        self.ego_position = np.array([0.0, 0.0])
        self.ego_positions = []

        if verbose:
            print(f"  Sight radius: {self.sight_radius:.1f}m")
            for i, cfg in enumerate(OBSTACLE_CONFIGS):
                print(f"  Obs {i}: {cfg['name']}  "
                      f"true={cfg['true_behavior']}  init={cfg['initial_mode']}")

        t0 = time.time()
        for step in range(num_steps):
            # Advance ego along x-axis
            self.ego_position = self.ego_position + np.array([self.ego_velocity * self.dt, 0.0])
            self.ego_positions.append(self.ego_position.copy())

            for i, cfg in enumerate(OBSTACLE_CONFIGS):
                self._simulate_step(i)
                preds = self._observe_and_predict(i, cfg, sim_step=step)
                if preds is not None and step > 20 and step % 10 == 0:
                    self._prediction_error(i, preds)
            self.ot_predictor.advance_timestep()
            if verbose and (step + 1) % 25 == 0:
                print(f"    step {step+1}/{num_steps}")

        # Evaluate predictions against ground truth positions
        self.evaluate_ground_truth()

        elapsed = time.time() - t0

        results = {"duration": elapsed, "obstacles": {}}
        for i, cfg in enumerate(OBSTACLE_CONFIGS):
            errs = self.prediction_errors[i]
            learned = self.learned_mode_history[i]
            second_half = learned[len(learned)//2:] if learned else []
            mode_counts = {}
            for m in second_half:
                mode_counts[m] = mode_counts.get(m, 0) + 1
            dom = max(mode_counts, key=mode_counts.get) if mode_counts else None
            acc = (mode_counts.get(cfg["true_behavior"], 0) / len(second_half)
                   if second_half else 0)
            results["obstacles"][i] = {
                "name": cfg["name"],
                "true_mode": cfg["true_behavior"],
                "learned_mode": dom,
                "accuracy": acc,
                "success": dom == cfg["true_behavior"],
                "mean_error": float(np.mean(errs)) if errs else None,
                "final_error": errs[-1] if errs else None,
            }

        return results


# ============================================================================
# Visualization
# ============================================================================

MODE_COLORS = {
    'constant_velocity': '#1f77b4',
    'decelerating':      '#e377c2',
    'accelerating':      '#ff7f0e',
    'turn_left':         '#2ca02c',
    'turn_right':        '#17becf',
    'lane_change_left':  '#bcbd22',
    'lane_change_right': '#7f7f7f',
}
OBS_COLORS = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#17becf', '#8c564b']

# Chi-squared thresholds for 2 DOF (position is 2D)
CHI2_1SIGMA = 2.2789   # 68.3% coverage
CHI2_2SIGMA = 5.9915   # 95.0% coverage
CHI2_3SIGMA = 9.2103   # 99.0% coverage


def reconstruct_covariance(angle: float, major_r: float, minor_r: float) -> np.ndarray:
    """Convert PredictionStep ellipse params to 2x2 covariance matrix.

    Cov = R(angle) @ diag(major_r^2, minor_r^2) @ R(angle)^T
    NaN radii are replaced with a default of 0.3.
    """
    if not np.isfinite(major_r) or major_r <= 0:
        major_r = 0.3
    if not np.isfinite(minor_r) or minor_r <= 0:
        minor_r = 0.3
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    D = np.diag([major_r**2, minor_r**2])
    return R @ D @ R.T


def ellipse_area(major_r: float, minor_r: float) -> float:
    """Area of the prediction ellipse: pi * a * b."""
    return np.pi * major_r * minor_r


def mahalanobis_distance_sq(point: np.ndarray, mean: np.ndarray,
                            cov: np.ndarray) -> float:
    """Squared Mahalanobis distance: (x-mu)^T Sigma^-1 (x-mu), regularized."""
    diff = point[:2] - mean[:2]
    reg_cov = cov + 1e-8 * np.eye(2)
    try:
        cov_inv = np.linalg.inv(reg_cov)
        return float(diff @ cov_inv @ diff)
    except np.linalg.LinAlgError:
        return float('inf')


def gaussian_nll(point: np.ndarray, mean: np.ndarray,
                 cov: np.ndarray) -> float:
    """Negative log-likelihood under 2D Gaussian: 0.5*(d^2 + ln(det Sigma)) + ln(2pi)."""
    d2 = mahalanobis_distance_sq(point, mean, cov)
    reg_cov = cov + 1e-8 * np.eye(2)
    sign, logdet = np.linalg.slogdet(reg_cov)
    if sign <= 0:
        return float('inf')
    return 0.5 * (d2 + logdet) + np.log(2 * np.pi)


def coverage_check(point: np.ndarray, mean: np.ndarray,
                   cov: np.ndarray) -> Dict[str, bool]:
    """Check if point falls within 1/2/3-sigma ellipsoid."""
    d2 = mahalanobis_distance_sq(point, mean, cov)
    return {
        '1sigma': d2 <= CHI2_1SIGMA,
        '2sigma': d2 <= CHI2_2SIGMA,
        '3sigma': d2 <= CHI2_3SIGMA,
    }


def visualize(sim: OTLearningSimulator, dt: float, output_path: str):
    """
    Create 6-row x N_obs figure of OT learning per obstacle.

    Row 0: Trajectory with sight radius + Gaussian ellipses at key snapshots
    Row 1: Mode weight evolution over time (gaps when out of sight)
    Row 2: Gaussian uncertainty narrowing (ellipse area vs time)
    Row 3: Ground truth coverage (rolling 1/2/3-sigma + Mahalanobis)
    Row 4: Wasserstein prediction error over time
    Row 5: Final converged weight distribution (bar chart)
    """
    from matplotlib.patches import Ellipse

    n_obs = len(OBSTACLE_CONFIGS)

    fig = plt.figure(figsize=(5.5 * n_obs, 24))
    fig.suptitle(
        f'Optimal Transport: Per-Obstacle Statistical Learning  '
        f'(sight radius = {sim.sight_radius:.1f}m)',
        fontsize=15, fontweight='bold', y=0.99,
    )

    gs = fig.add_gridspec(6, n_obs, hspace=0.50, wspace=0.35,
                          top=0.93, bottom=0.04, left=0.06, right=0.97,
                          height_ratios=[1.2, 1, 0.8, 1, 1, 1])

    # ---- Row 0: Trajectory with sight radius + Gaussian ellipses ----
    ax_traj = fig.add_subplot(gs[0, :])
    if sim.ego_positions:
        ego_xs = [p[0] for p in sim.ego_positions]
        ego_ys = [p[1] for p in sim.ego_positions]
        ax_traj.plot(ego_xs, ego_ys, 'b-', linewidth=2.5, alpha=0.7, label='Ego')
        n_circles = 5
        step_interval = max(1, len(sim.ego_positions) // n_circles)
        for ci in range(0, len(sim.ego_positions), step_interval):
            ep = sim.ego_positions[ci]
            alpha_val = 0.15 + 0.15 * (ci / len(sim.ego_positions))
            circle = plt.Circle(ep, sim.sight_radius, fill=False,
                                color='dodgerblue', linestyle='--',
                                linewidth=1.0, alpha=alpha_val)
            ax_traj.add_patch(circle)
        ep_final = sim.ego_positions[-1]
        final_circle = plt.Circle(ep_final, sim.sight_radius, fill=False,
                                   color='dodgerblue', linestyle='--',
                                   linewidth=2.0, alpha=0.6,
                                   label=f'Sight ({sim.sight_radius:.0f}m)')
        ax_traj.add_patch(final_circle)

    for oid in range(n_obs):
        pos_hist = sim.position_history.get(oid, [])
        color = OBS_COLORS[oid % len(OBS_COLORS)]
        if pos_hist:
            obs_xs = [p[0] for p in pos_hist]
            obs_ys = [p[1] for p in pos_hist]
            ax_traj.plot(obs_xs, obs_ys, '-', color=color, linewidth=1.5,
                         alpha=0.7, label=f'Obs {oid}: {OBSTACLE_CONFIGS[oid]["name"]}')
            ax_traj.plot(obs_xs[0], obs_ys[0], 'o', markersize=10,
                         color=color, zorder=5)
            ax_traj.plot(obs_xs[-1], obs_ys[-1], 's', markersize=8,
                         color=color, alpha=0.6, zorder=5)

        # Draw 2-sigma Gaussian ellipses at 4 evenly-spaced snapshots (k=5)
        snapshots = sim.prediction_snapshots.get(oid, [])
        if len(snapshots) >= 4:
            indices = np.linspace(0, len(snapshots) - 1, 4, dtype=int)
            for si, snap_idx in enumerate(indices):
                capture_idx, snapshot = snapshots[snap_idx]
                k_plot = min(5, len(snapshot) - 1)
                mean, cov = snapshot[k_plot]
                eigvals, eigvecs = np.linalg.eigh(cov)
                angle_deg = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
                scale = np.sqrt(CHI2_2SIGMA)  # 2-sigma
                w = 2 * scale * np.sqrt(max(eigvals[1], 1e-10))
                h = 2 * scale * np.sqrt(max(eigvals[0], 1e-10))
                alpha_e = 0.15 + 0.20 * (si / 3)  # lighter early, darker late
                ell = Ellipse(xy=mean, width=w, height=h, angle=angle_deg,
                              facecolor=color, edgecolor=color,
                              alpha=alpha_e, linewidth=0.8, linestyle='--',
                              zorder=3)
                ax_traj.add_patch(ell)

    ax_traj.set_xlabel('X [m]')
    ax_traj.set_ylabel('Y [m]')
    ax_traj.set_title('Trajectory Overview with Gaussian Prediction Ellipses (2σ, k=5)',
                       fontsize=11, fontweight='bold')
    ax_traj.legend(loc='upper left', fontsize=7, ncol=3)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_aspect('equal')

    # ---- Row 1: mode weight time-series ----
    for oid in range(n_obs):
        ax = fig.add_subplot(gs[1, oid])
        cfg = OBSTACLE_CONFIGS[oid]
        true_mode = cfg['true_behavior']
        init_mode = cfg['initial_mode']
        wh_raw = sim.mode_weight_history.get(oid, [])

        wh = [(i, w) for i, w in enumerate(wh_raw) if w is not None]

        if wh:
            t_axis = [i * dt for i, _ in wh]
            all_modes = sorted({m for _, w in wh for m in w})
            for mode in all_modes:
                vals = [w.get(mode, 0.0) for _, w in wh]
                is_true = (mode == true_mode)
                ax.plot(
                    t_axis, vals,
                    color=MODE_COLORS.get(mode, 'gray'),
                    linewidth=2.8 if is_true else 1.3,
                    alpha=1.0 if is_true else 0.55,
                    linestyle='-' if is_true else '--',
                    label=(mode.replace('_', ' ')
                           + (' (TRUE)' if is_true else '')),
                )

            in_sight = sim.in_sight_history.get(oid, [])
            if in_sight:
                t_full = np.arange(len(in_sight)) * dt
                for k in range(len(in_sight)):
                    if not in_sight[k]:
                        ax.axvspan(t_full[k] - dt/2, t_full[k] + dt/2,
                                   color='gray', alpha=0.08)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel('Time [s]', fontsize=8)
        if oid == 0:
            ax.set_ylabel('Mode Weight', fontsize=10)
        ax.set_title(
            f'Obs {oid}: {cfg["name"]}\n'
            f'true={true_mode}  init={init_mode}',
            fontsize=9, fontweight='bold',
            color=OBS_COLORS[oid % len(OBS_COLORS)],
        )
        ax.legend(loc='best', fontsize=6.5, ncol=1, framealpha=0.8)
        ax.grid(True, alpha=0.25)
        ax.axhline(0.5, color='gray', ls=':', alpha=0.25)

    # ---- Row 2: Gaussian uncertainty narrowing (ellipse area) ----
    for oid in range(n_obs):
        ax = fig.add_subplot(gs[2, oid])
        area_hist = sim.ellipse_area_history.get(oid, [])
        color = OBS_COLORS[oid % len(OBS_COLORS)]

        if area_hist:
            t_area = [idx * dt for idx, _ in area_hist]
            areas = [a for _, a in area_hist]
            ax.plot(t_area, areas, '-', color=color, linewidth=1.5, alpha=0.85)
            ax.fill_between(t_area, 0, areas, color=color, alpha=0.10)
            # Smoothed trend
            win = max(3, len(areas) // 5)
            if len(areas) > win:
                sm = np.convolve(areas, np.ones(win)/win, mode='valid')
                ax.plot(t_area[win-1:], sm, 'k-', lw=1.8, alpha=0.45, label='Trend')
            ax.annotate(f'{areas[-1]:.4f}', xy=(t_area[-1], areas[-1]),
                        fontsize=7, fontweight='bold', ha='right', va='bottom',
                        color=color)

        ax.set_xlabel('Time [s]', fontsize=8)
        if oid == 0:
            ax.set_ylabel('Ellipse Area [m²]', fontsize=9)
        ax.set_title(f'Obs {oid}: Uncertainty Narrowing', fontsize=9,
                     color=color)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)
        if area_hist and len(area_hist) > 5:
            ax.legend(fontsize=7)

    # ---- Row 3: Ground truth coverage (k=5) ----
    for oid in range(n_obs):
        ax = fig.add_subplot(gs[3, oid])
        color = OBS_COLORS[oid % len(OBS_COLORS)]
        gt_metrics = sim.ground_truth_metrics.get(oid, [])

        # Filter to k=5 only
        k5 = [m for m in gt_metrics if m['k'] == 5]

        if len(k5) > 3:
            t_gt = [m['capture_idx'] * dt for m in k5]
            mahal_vals = [np.sqrt(m['mahal_sq']) for m in k5]
            c1 = [float(m['coverage_1sigma']) for m in k5]
            c2 = [float(m['coverage_2sigma']) for m in k5]
            c3 = [float(m['coverage_3sigma']) for m in k5]

            # Rolling average (window=10 or smaller)
            roll_win = min(10, max(1, len(k5) // 3))
            kernel = np.ones(roll_win) / roll_win

            def rolling(arr):
                if len(arr) >= roll_win:
                    return np.convolve(arr, kernel, mode='valid')
                return np.array(arr)

            c1_r = rolling(c1)
            c2_r = rolling(c2)
            c3_r = rolling(c3)
            t_r = t_gt[roll_win-1:] if len(t_gt) >= roll_win else t_gt

            ax.plot(t_r, c1_r * 100, '-', color='#e74c3c', linewidth=1.5,
                    alpha=0.85, label=f'1σ ({100*np.mean(c1):.0f}%)')
            ax.plot(t_r, c2_r * 100, '-', color='#f39c12', linewidth=1.5,
                    alpha=0.85, label=f'2σ ({100*np.mean(c2):.0f}%)')
            ax.plot(t_r, c3_r * 100, '-', color='#27ae60', linewidth=1.5,
                    alpha=0.85, label=f'3σ ({100*np.mean(c3):.0f}%)')

            # Reference lines
            ax.axhline(68.3, color='#e74c3c', ls=':', alpha=0.3, lw=0.8)
            ax.axhline(95.0, color='#f39c12', ls=':', alpha=0.3, lw=0.8)
            ax.axhline(99.0, color='#27ae60', ls=':', alpha=0.3, lw=0.8)

            ax.set_ylim(0, 105)
            ax.set_ylabel('Coverage %', fontsize=8)
            ax.legend(loc='lower right', fontsize=6, framealpha=0.8)

            # Twin axis for Mahalanobis distance
            ax2 = ax.twinx()
            mahal_r = rolling(mahal_vals)
            ax2.plot(t_r, mahal_r, 'k--', linewidth=1.0, alpha=0.4, label='Mahal')
            ax2.set_ylabel('√Mahal', fontsize=7, color='gray')
            ax2.tick_params(axis='y', labelsize=6, colors='gray')
        else:
            ax.text(0.5, 0.5, 'Insufficient\ndata',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color='gray')

        ax.set_xlabel('Time [s]', fontsize=8)
        ax.set_title(f'Obs {oid}: GT Coverage (k=5)', fontsize=9,
                     color=color)
        ax.grid(True, alpha=0.25)

    # ---- Row 4: prediction error ----
    for oid in range(n_obs):
        ax = fig.add_subplot(gs[4, oid])
        errs = sim.prediction_errors.get(oid, [])

        if errs:
            t_err = [(30 + k * 10) * dt for k in range(len(errs))]
            ax.plot(t_err, errs, '-',
                    color=OBS_COLORS[oid % len(OBS_COLORS)],
                    linewidth=1.8, alpha=0.85)
            ax.fill_between(t_err, 0, errs,
                            color=OBS_COLORS[oid % len(OBS_COLORS)], alpha=0.12)
            win = max(3, len(errs) // 5)
            if len(errs) > win:
                sm = np.convolve(errs, np.ones(win)/win, mode='valid')
                ax.plot(t_err[win-1:], sm, 'k-', lw=1.8, alpha=0.45, label='Trend')
            ax.annotate(f'{errs[-1]:.3f}', xy=(t_err[-1], errs[-1]),
                        fontsize=7.5, fontweight='bold', ha='right', va='bottom',
                        color=OBS_COLORS[oid % len(OBS_COLORS)])

        ax.set_xlabel('Time [s]', fontsize=8)
        if oid == 0:
            ax.set_ylabel('Wasserstein Prediction Error', fontsize=9)
        ax.set_title(f'Obs {oid}: Prediction Error', fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)
        if errs and len(errs) > 5:
            ax.legend(fontsize=7)

    # ---- Row 5: final converged weights (bar chart) ----
    for oid in range(n_obs):
        ax = fig.add_subplot(gs[5, oid])
        cfg = OBSTACLE_CONFIGS[oid]
        true_mode = cfg['true_behavior']
        wh_raw = sim.mode_weight_history.get(oid, [])
        wh = [w for w in wh_raw if w is not None]

        if wh:
            n_tail = max(1, len(wh) // 5)
            tail = wh[-n_tail:]
            all_modes = sorted({m for w in tail for m in w})
            avg = {m: float(np.mean([w.get(m, 0.0) for w in tail]))
                   for m in all_modes}

            modes = sorted(avg.keys())
            vals = [avg[m] for m in modes]
            colors = ['#2ecc71' if m == true_mode
                      else MODE_COLORS.get(m, 'gray') for m in modes]

            bars = ax.bar(range(len(modes)), vals, color=colors,
                          edgecolor='black', linewidth=0.5, alpha=0.85)
            for b, v in zip(bars, vals):
                if v > 0.03:
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                            f'{v:.2f}', ha='center', va='bottom',
                            fontsize=7.5, fontweight='bold')

            ax.set_xticks(range(len(modes)))
            ax.set_xticklabels([m.replace('_', '\n') for m in modes],
                               fontsize=6.5)
            for i, m in enumerate(modes):
                if m == true_mode:
                    ax.get_xticklabels()[i].set_fontweight('bold')
                    ax.get_xticklabels()[i].set_color('#27ae60')

            best = max(avg, key=avg.get)
            ok = best == true_mode
            ax.set_title(
                f'Obs {oid}: Final Weights  [{"LEARNED" if ok else "LEARNING..."}]',
                fontsize=9, fontweight='bold',
                color='#27ae60' if ok else '#c0392b',
            )
        else:
            ax.text(0.5, 0.5, 'Insufficient\nobservations',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color='gray')
            ax.set_title(f'Obs {oid}: Final Weights', fontsize=9)

        ax.set_ylim(0, 1.15)
        if oid == 0:
            ax.set_ylabel('Avg Weight', fontsize=9)
        ax.grid(True, alpha=0.18, axis='y')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Visualize OT mode weight learning per obstacle')
    parser.add_argument('--duration', '-d', type=float, default=10.0,
                        help='Simulation duration in seconds (default 10)')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Timestep (default 0.1)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output PNG path (default: ot_learning_per_obstacle.png)')
    parser.add_argument('--sight-radius', '-s', type=float, default=5.0,
                        help='Ego sight radius in meters (default 5.0)')
    args = parser.parse_args()

    output = args.output or os.path.join(
        os.path.dirname(__file__), '..', '..',
        'ot_learning_per_obstacle.png')
    output = os.path.abspath(output)

    print("=" * 60)
    print("OT Per-Obstacle Learning Visualization")
    print("=" * 60)

    sim = OTLearningSimulator(dt=args.dt, sight_radius=args.sight_radius)
    results = sim.run(duration=args.duration, verbose=True)

    print("\n--- Per-obstacle results ---")
    for oid, r in sorted(results["obstacles"].items()):
        tag = 'OK' if r['success'] else 'MISS'
        print(f"  Obs {oid} ({r['name']}): "
              f"true={r['true_mode']}  learned={r['learned_mode']}  "
              f"acc={r['accuracy']:.0%}  [{tag}]")

    # Ground truth verification summary
    print("\n--- Ground Truth Verification (k=5 step-ahead) ---")
    for oid in sorted(sim.ground_truth_metrics.keys()):
        k5 = [m for m in sim.ground_truth_metrics[oid] if m['k'] == 5]
        if k5:
            c1 = np.mean([m['coverage_1sigma'] for m in k5])
            c2 = np.mean([m['coverage_2sigma'] for m in k5])
            c3 = np.mean([m['coverage_3sigma'] for m in k5])
            finite_mahal = [np.sqrt(m['mahal_sq']) for m in k5
                           if np.isfinite(m['mahal_sq'])]
            avg_mahal = np.mean(finite_mahal) if finite_mahal else float('nan')
            avg_nll = np.mean([m['nll'] for m in k5 if np.isfinite(m['nll'])])
            print(f"  Obs {oid}: "
                  f"1σ={c1:.0%} 2σ={c2:.0%} 3σ={c3:.0%}  "
                  f"Mahal={avg_mahal:.2f}  NLL={avg_nll:.2f}")
        else:
            print(f"  Obs {oid}: insufficient data")

    print(f"\nGenerating figure...")
    path = visualize(sim, args.dt, output)
    print(f"Saved to: {path}")


if __name__ == '__main__':
    main()
