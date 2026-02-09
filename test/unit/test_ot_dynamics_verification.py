#!/usr/bin/env python3
"""
Verification: Gaussian Hypersphere Optimal Transport Dynamics Learning.

For each of the 7 standard mode models, simulates an obstacle following the
true underlying dynamics and verifies that the OT predictor:

  1. Mode identification  - correct mode gets highest weight
  2. Velocity convergence - learned mean velocity matches true mode target
  3. Covariance tightness - learned covariance is small (concentrated)
  4. Cross-mode separation - Wasserstein distance is small for correct mode,
                             large for incorrect modes
  5. Trajectory tracking  - predicted trajectory stays close to ground truth
  6. Per-obstacle independence - multiple simultaneous obstacles learn
                                 independently without cross-contamination

Usage:
    python3 test/unit/test_ot_dynamics_verification.py
"""

import sys
import os
import types

# ============================================================================
# Mock external dependencies (no CasADi)
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
sys.modules['modules.constraints.scenario_utils.math_utils'] = types.ModuleType(
    'modules.constraints.scenario_utils.math_utils')
sys.modules['modules.constraints.scenario_utils.sampler'] = types.ModuleType(
    'modules.constraints.scenario_utils.sampler')
sys.modules['modules.constraints.scenario_utils.scenario_module'] = types.ModuleType(
    'modules.constraints.scenario_utils.scenario_module')

import importlib
_ot_mod = importlib.import_module(
    'modules.constraints.scenario_utils.optimal_transport_predictor')

OptimalTransportPredictor = _ot_mod.OptimalTransportPredictor
OTWeightType = _ot_mod.OTWeightType
EmpiricalDistribution = _ot_mod.EmpiricalDistribution
wasserstein_distance = _ot_mod.wasserstein_distance
create_ot_predictor_with_standard_modes = _ot_mod.create_ot_predictor_with_standard_modes

import numpy as np

# ============================================================================
# Ground truth mode dynamics
# ============================================================================

# Must match create_ot_predictor_with_standard_modes(base_speed=0.5)
BASE_SPEED = 0.5
TURN_RATE = 0.8

MODE_TARGET_VELOCITIES = {
    'constant_velocity': np.array([BASE_SPEED, 0.0]),
    'decelerating':      np.array([BASE_SPEED * 0.5, 0.0]),
    'accelerating':      np.array([BASE_SPEED * 1.5, 0.0]),
    'turn_left':         np.array([BASE_SPEED * np.cos(TURN_RATE),
                                   BASE_SPEED * np.sin(TURN_RATE)]),
    'turn_right':        np.array([BASE_SPEED * np.cos(-TURN_RATE),
                                   BASE_SPEED * np.sin(-TURN_RATE)]),
    'lane_change_left':  np.array([BASE_SPEED, 0.3]),
    'lane_change_right': np.array([BASE_SPEED, -0.3]),
}

ALL_MODES = list(MODE_TARGET_VELOCITIES.keys())


def simulate_obstacle(true_mode, n_steps=150, dt=0.1, alpha=0.25,
                      noise_std=0.02, seed=None):
    """Simulate obstacle with exponential smoothing toward mode target velocity.

    Returns positions and velocities arrays (n_steps+1 x 2 each).
    """
    rng = np.random.RandomState(seed)
    target_vel = MODE_TARGET_VELOCITIES[true_mode]

    positions = np.zeros((n_steps + 1, 2))
    velocities = np.zeros((n_steps + 1, 2))

    # Start with perturbed velocity (not exactly on target)
    velocities[0] = target_vel * 0.5 + rng.randn(2) * 0.1

    for k in range(n_steps):
        velocities[k + 1] = ((1 - alpha) * velocities[k]
                             + alpha * target_vel
                             + rng.randn(2) * noise_std)
        positions[k + 1] = positions[k] + velocities[k + 1] * dt

    return positions, velocities


# ============================================================================
# Test helpers
# ============================================================================

def make_predictor(dt=0.1):
    """Create OT predictor with standard reference modes."""
    np.random.seed(0)
    return create_ot_predictor_with_standard_modes(
        dt=dt, base_speed=BASE_SPEED,
        buffer_size=200, min_samples_for_ot=10,
        uncertainty_scale=0.3,
        weight_type=OTWeightType.WASSERSTEIN,
    )


def feed_observations(predictor, obs_id, positions, dt=0.1,
                      label_after=15, available_modes=None):
    """Feed position observations to predictor, labelling with inferred mode
    once enough samples are collected.

    Returns list of inferred mode weights dicts (one per step after label_after).
    """
    if available_modes is None:
        available_modes = ALL_MODES
    weight_history = []

    for k in range(len(positions)):
        mode_id = None
        if k > label_after:
            weights = predictor.compute_mode_weights(obs_id, available_modes)
            weight_history.append(weights)
            mode_id = max(weights, key=weights.get)

        predictor.observe(obs_id, positions[k], mode_id=mode_id)
        predictor.advance_timestep()

    return weight_history


# ============================================================================
# Tests
# ============================================================================

passed = 0
failed = 0
errors = []


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f"  -- {detail}"
        print(msg)
        errors.append(msg)


# ---------- Test 1: mode identification for every mode --------------------

def test_mode_identification():
    """Each mode's true dynamics must be identified as highest-weight mode."""
    print("\n=== Test 1: Mode Identification (all 7 modes) ===")

    for true_mode in ALL_MODES:
        predictor = make_predictor()
        positions, velocities = simulate_obstacle(true_mode, n_steps=120,
                                                  seed=42)
        wh = feed_observations(predictor, obs_id=0, positions=positions)

        # Use second half of weight history (after learning stabilizes)
        second_half = wh[len(wh) // 2:]
        if not second_half:
            check(f"mode_id/{true_mode}", False, "no weight history")
            continue

        # Tally which mode had highest weight most often
        mode_wins = {}
        for w in second_half:
            best = max(w, key=w.get)
            mode_wins[best] = mode_wins.get(best, 0) + 1

        dominant = max(mode_wins, key=mode_wins.get)
        accuracy = mode_wins.get(true_mode, 0) / len(second_half)

        check(f"mode_id/{true_mode}",
              dominant == true_mode,
              f"dominant={dominant} accuracy={accuracy:.0%}")


# ---------- Test 2: learned velocity mean convergence --------------------

def test_velocity_convergence():
    """Learned velocity distribution mean must converge to true target."""
    print("\n=== Test 2: Velocity Mean Convergence ===")

    for true_mode in ALL_MODES:
        predictor = make_predictor()
        positions, _ = simulate_obstacle(true_mode, n_steps=150, seed=7)
        feed_observations(predictor, obs_id=0, positions=positions)

        target = MODE_TARGET_VELOCITIES[true_mode]

        # Check if mode was learned
        learned = predictor.get_learned_modes(0)
        if true_mode not in learned:
            check(f"vel_conv/{true_mode}", False,
                  f"mode not learned; learned={learned}")
            continue

        stats = predictor.get_mode_distribution_stats(0, true_mode)
        if stats is None or stats['velocity_mean'] is None:
            check(f"vel_conv/{true_mode}", False, "no velocity stats")
            continue

        learned_mean = np.array(stats['velocity_mean'])
        error = np.linalg.norm(learned_mean - target)

        # Tolerance: 0.15 m/s (accounts for process noise and finite samples)
        check(f"vel_conv/{true_mode}",
              error < 0.15,
              f"target={target} learned={learned_mean} err={error:.4f}")


# ---------- Test 3: covariance tightness ---------------------------------

def test_covariance_tightness():
    """Learned velocity covariance eigenvalues should be small."""
    print("\n=== Test 3: Covariance Tightness ===")

    for true_mode in ALL_MODES:
        predictor = make_predictor()
        positions, _ = simulate_obstacle(true_mode, n_steps=150, seed=13)
        feed_observations(predictor, obs_id=0, positions=positions)

        learned = predictor.get_learned_modes(0)
        if true_mode not in learned:
            check(f"cov_tight/{true_mode}", False, "mode not learned")
            continue

        stats = predictor.get_mode_distribution_stats(0, true_mode)
        cov = np.array(stats['velocity_cov'])
        eigvals = np.linalg.eigvalsh(cov)
        max_eigval = max(eigvals)

        # With noise_std=0.02 and alpha=0.25, steady-state variance is small.
        # Allow up to 0.05 (generous bound).
        check(f"cov_tight/{true_mode}",
              max_eigval < 0.05,
              f"max_eigval={max_eigval:.5f}")


# ---------- Test 4: Wasserstein cross-mode separation --------------------

def test_cross_mode_separation():
    """Wasserstein distance must be smallest for the correct mode."""
    print("\n=== Test 4: Cross-Mode Wasserstein Separation ===")

    for true_mode in ALL_MODES:
        predictor = make_predictor()
        positions, _ = simulate_obstacle(true_mode, n_steps=120, seed=21)
        feed_observations(predictor, obs_id=0, positions=positions)

        # Build observed velocity distribution from buffer
        buf = predictor.trajectory_buffers.get(0)
        if buf is None or len(buf) < 10:
            check(f"separation/{true_mode}", False, "insufficient buffer")
            continue

        recent = buf.get_recent(30)
        obs_vels = np.array([o.velocity for o in recent])
        obs_dist = EmpiricalDistribution.from_samples(obs_vels)

        # Compute Wasserstein to every reference mode
        distances = {}
        for mode_id in ALL_MODES:
            ref = predictor.reference_distributions.get(mode_id)
            if ref is None:
                continue
            d = wasserstein_distance(obs_dist, ref.velocity_dist,
                                     epsilon=0.1)
            distances[mode_id] = d

        if not distances:
            check(f"separation/{true_mode}", False, "no references")
            continue

        closest = min(distances, key=distances.get)
        d_true = distances.get(true_mode, float('inf'))
        d_closest = distances[closest]

        check(f"separation/{true_mode}",
              closest == true_mode,
              f"closest={closest} d_true={d_true:.4f} d_closest={d_closest:.4f}")


# ---------- Test 5: trajectory prediction tracking -----------------------

def test_trajectory_tracking():
    """Predicted trajectory must stay close to ground truth over horizon."""
    print("\n=== Test 5: Trajectory Tracking ===")
    horizon = 10
    dt = 0.1

    for true_mode in ALL_MODES:
        predictor = make_predictor(dt=dt)
        positions, velocities = simulate_obstacle(true_mode, n_steps=120,
                                                  seed=35)
        feed_observations(predictor, obs_id=0, positions=positions)

        # Predict from the last observed state
        last_idx = len(positions) - 1
        pred = predictor.predict_trajectory(
            obstacle_id=0,
            current_position=positions[last_idx],
            current_velocity=velocities[last_idx],
            horizon=horizon,
        )

        # Simulate ground truth forward
        rng = np.random.RandomState(99)
        gt_pos = [positions[last_idx].copy()]
        vel = velocities[last_idx].copy()
        target = MODE_TARGET_VELOCITIES[true_mode]
        for _ in range(horizon):
            vel = 0.75 * vel + 0.25 * target + rng.randn(2) * 0.02
            gt_pos.append(gt_pos[-1] + vel * dt)

        # Compute mean position error over horizon
        errs = []
        for k in range(min(len(pred), len(gt_pos))):
            e = np.linalg.norm(pred[k].position[:2] - gt_pos[k])
            errs.append(e)

        mean_err = np.mean(errs)
        max_err = np.max(errs)

        # Generous tolerance: mean < 0.5m, max < 1.0m over 1s horizon
        check(f"tracking/{true_mode}",
              mean_err < 0.5 and max_err < 1.0,
              f"mean_err={mean_err:.3f}m  max_err={max_err:.3f}m")


# ---------- Test 6: per-obstacle independence ----------------------------

def test_per_obstacle_independence():
    """Multiple obstacles learning simultaneously must not cross-contaminate."""
    print("\n=== Test 6: Per-Obstacle Independence ===")
    predictor = make_predictor()
    n_steps = 120

    # Assign distinctly different modes to obstacles
    test_pairs = [
        (0, 'constant_velocity'),
        (1, 'turn_left'),
        (2, 'accelerating'),
        (3, 'lane_change_right'),
    ]

    # Simulate and feed all obstacles simultaneously
    obs_data = {}
    for obs_id, mode in test_pairs:
        positions, velocities = simulate_obstacle(mode, n_steps=n_steps,
                                                  seed=obs_id * 10 + 5)
        obs_data[obs_id] = (positions, velocities, mode)

    # Feed observations interleaved (like real-time)
    for k in range(n_steps + 1):
        for obs_id, mode in test_pairs:
            positions = obs_data[obs_id][0]
            mode_id = None

            if k > 15:
                weights = predictor.compute_mode_weights(obs_id, ALL_MODES)
                mode_id = max(weights, key=weights.get)

            predictor.observe(obs_id, positions[k], mode_id=mode_id)

        predictor.advance_timestep()

    # Verify each obstacle learned its own mode
    for obs_id, true_mode in test_pairs:
        weights = predictor.compute_mode_weights(obs_id, ALL_MODES)
        best = max(weights, key=weights.get)
        w_true = weights.get(true_mode, 0.0)

        # With 7 modes, uniform = 1/7 ≈ 0.14; correct mode should be
        # highest AND well above uniform
        check(f"independence/obs{obs_id}_{true_mode}",
              best == true_mode and w_true > 0.20,
              f"best={best} w_true={w_true:.3f}")

    # Check that obstacle 0 (constant_velocity) doesn't have turn_left learned
    learned_0 = predictor.get_learned_modes(0)
    learned_1 = predictor.get_learned_modes(1)
    # Obstacle 0 should predominantly label as constant_velocity, not turn_left
    check("independence/no_cross_mode",
          'turn_left' not in learned_0 or 'constant_velocity' not in learned_1,
          f"obs0_modes={learned_0} obs1_modes={learned_1}")


# ---------- Test 7: weight convergence dynamics --------------------------

def test_weight_convergence_dynamics():
    """Mode weights must converge monotonically (in a smoothed sense)."""
    print("\n=== Test 7: Weight Convergence Dynamics ===")

    for true_mode in ['turn_left', 'decelerating', 'accelerating']:
        predictor = make_predictor()
        positions, _ = simulate_obstacle(true_mode, n_steps=150, seed=50)
        wh = feed_observations(predictor, obs_id=0, positions=positions)

        if len(wh) < 20:
            check(f"convergence/{true_mode}", False, "insufficient history")
            continue

        # Extract weight for the true mode over time
        true_weights = [w.get(true_mode, 0.0) for w in wh]

        # Compare average of first quarter vs last quarter
        q1 = np.mean(true_weights[:len(true_weights) // 4])
        q4 = np.mean(true_weights[-len(true_weights) // 4:])

        # With 7 modes (uniform=0.14), correct mode should dominate (>0.25)
        # and stay at least as high or higher than its initial weight
        check(f"convergence/{true_mode}",
              q4 >= q1 * 0.9 and q4 > 0.25,
              f"q1_avg={q1:.3f} q4_avg={q4:.3f}")


# ---------- Test 8: learned vs reference distributional match ------------

def test_learned_matches_reference():
    """After learning, the per-obstacle distribution should be close to
    the reference distribution for the true mode (small Wasserstein distance),
    and far from wrong modes."""
    print("\n=== Test 8: Learned Distribution Matches Reference ===")

    for true_mode in ALL_MODES:
        predictor = make_predictor()
        positions, _ = simulate_obstacle(true_mode, n_steps=150, seed=77)
        feed_observations(predictor, obs_id=0, positions=positions)

        learned = predictor.get_learned_modes(0)
        if true_mode not in learned:
            check(f"dist_match/{true_mode}", False, "mode not learned")
            continue

        learned_dist = predictor.mode_distributions[0][true_mode].velocity_dist
        ref_dist = predictor.reference_distributions[true_mode].velocity_dist

        d_match = wasserstein_distance(learned_dist, ref_dist, epsilon=0.1)

        # Pick a distant wrong mode to compare
        wrong_modes = [m for m in ALL_MODES if m != true_mode]
        d_wrong = []
        for wm in wrong_modes:
            ref_w = predictor.reference_distributions[wm].velocity_dist
            d_wrong.append(wasserstein_distance(learned_dist, ref_w, epsilon=0.1))

        avg_wrong = np.mean(d_wrong)
        min_wrong = np.min(d_wrong)

        # The distance to the correct reference should be smaller
        # than the distance to any wrong reference
        ratio = d_match / (min_wrong + 1e-8)

        check(f"dist_match/{true_mode}",
              d_match < 0.3 and ratio < 0.8,
              f"d_correct={d_match:.4f} d_wrong_min={min_wrong:.4f} "
              f"d_wrong_avg={avg_wrong:.4f} ratio={ratio:.3f}")


# ---------- Test 9: prediction uncertainty calibration -------------------

def test_uncertainty_calibration():
    """Ground truth positions should fall within predicted uncertainty
    ellipses at roughly the expected rate (2-sigma ~ 95%)."""
    print("\n=== Test 9: Uncertainty Calibration ===")
    dt = 0.1
    horizon = 5
    n_trials = 30

    for true_mode in ['constant_velocity', 'turn_left', 'accelerating']:
        predictor = make_predictor(dt=dt)
        positions, velocities = simulate_obstacle(true_mode, n_steps=200,
                                                  seed=88)
        feed_observations(predictor, obs_id=0, positions=positions[:120])

        # Run multiple prediction trials from different points
        within_2sigma = 0
        total_checks = 0

        for trial in range(n_trials):
            start_idx = 120 + trial
            if start_idx + horizon >= len(positions):
                break

            pred = predictor.predict_trajectory(
                obstacle_id=0,
                current_position=positions[start_idx],
                current_velocity=velocities[start_idx],
                horizon=horizon,
            )

            # Feed the observation for next trial
            predictor.observe(0, positions[start_idx])
            predictor.advance_timestep()

            # Check k=horizon step
            if len(pred) > horizon:
                p = pred[horizon]
                gt = positions[start_idx + horizon]

                # Reconstruct covariance from ellipse parameters
                c, s = np.cos(p.angle), np.sin(p.angle)
                R = np.array([[c, -s], [s, c]])
                D = np.diag([max(p.major_radius, 1e-6) ** 2,
                             max(p.minor_radius, 1e-6) ** 2])
                cov = R @ D @ R.T + 1e-8 * np.eye(2)

                diff = gt[:2] - p.position[:2]
                try:
                    d2 = diff @ np.linalg.inv(cov) @ diff
                    # 2-sigma for 2D: chi2(2) = 5.99
                    if d2 <= 5.99:
                        within_2sigma += 1
                except np.linalg.LinAlgError:
                    pass
                total_checks += 1

        if total_checks < 5:
            check(f"calibration/{true_mode}", False, "insufficient trials")
            continue

        coverage = within_2sigma / total_checks
        # We expect ~95% within 2-sigma; allow 60% as lower bound
        # (the predictor drifts velocity toward the mode mean which
        # may not perfectly match the stochastic simulation)
        check(f"calibration/{true_mode}",
              coverage >= 0.60,
              f"2sigma_coverage={coverage:.0%} ({within_2sigma}/{total_checks})")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 65)
    print("  Gaussian Hypersphere OT Dynamics Learning Verification")
    print("=" * 65)

    test_mode_identification()
    test_velocity_convergence()
    test_covariance_tightness()
    test_cross_mode_separation()
    test_trajectory_tracking()
    test_per_obstacle_independence()
    test_weight_convergence_dynamics()
    test_learned_matches_reference()
    test_uncertainty_calibration()

    print("\n" + "=" * 65)
    print(f"  Results: {passed} passed, {failed} failed "
          f"(total {passed + failed})")
    print("=" * 65)

    if errors:
        print("\nFailures:")
        for e in errors:
            print(f"  {e}")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
