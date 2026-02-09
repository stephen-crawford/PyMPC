/**
 * @file test_mode_coverage.cpp
 * @brief Comparison experiments for mode coverage guarantee in scenario sampling.
 *
 * Three experiments comparing coverage ON vs OFF:
 *   1. Rare mode switch safety — collision rate when obstacle switches to rare mode
 *   2. Coverage statistics — how often all modes are represented
 *   3. Epsilon guarantee with coverage — clearance under mode switching + noise
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <vector>
#include <map>
#include <string>
#include <set>

#include "types.hpp"
#include "dynamics.hpp"
#include "mode_weights.hpp"
#include "scenario_sampler.hpp"
#include "collision_constraints.hpp"
#include "scenario_pruning.hpp"
#include "mpc_controller.hpp"
#include "config.hpp"

using namespace scenario_mpc;

// ============================================================================
// Helpers
// ============================================================================

static std::map<std::string, ModeModel> create_modes_with_noise_scale(
    double dt, double noise_scale)
{
    auto modes = create_obstacle_mode_models(dt);
    for (auto& [id, mode] : modes) {
        mode.G *= noise_scale;
    }
    return modes;
}

// ============================================================================
// Experiment 1: Rare Mode Switch Safety
//
// Obstacle starts in constant_velocity, switches to turn_left at step 30.
// Mode history: ~95% CV, ~5% turn_left before the switch.
// Measure collision rate with coverage ON vs OFF.
// ============================================================================

struct Exp1Result {
    int num_scenarios;
    bool coverage_on;
    int num_trials;
    int collisions;
    double collision_rate;
    double avg_min_clearance;
};

static Exp1Result run_experiment1_config(
    int num_scenarios, bool coverage_on, int num_trials, int seed_base)
{
    const double dt = 0.1;
    const int switch_step = 30;
    const int sim_steps = 60;
    const double obs_x = 7.0;
    const double obs_y = 0.5;  // Close pass — vulnerable to wrong mode
    const double obs_vx = -0.8;

    Exp1Result result;
    result.num_scenarios = num_scenarios;
    result.coverage_on = coverage_on;
    result.num_trials = num_trials;
    result.collisions = 0;
    result.avg_min_clearance = 0.0;

    for (int trial = 0; trial < num_trials; ++trial) {
        ScenarioMPCConfig config;
        config.horizon = 15;
        config.dt = dt;
        config.num_scenarios = num_scenarios;
        config.weight_type = WeightType::FREQUENCY;
        config.recency_decay = 0.9;
        config.ego_radius = 0.8;
        config.obstacle_radius = 0.5;
        config.safety_margin = 0.15;
        config.ensure_mode_coverage = coverage_on;

        double combined_radius = config.ego_radius + config.obstacle_radius;

        AdaptiveScenarioMPC controller(config);
        auto modes = create_obstacle_mode_models(dt);
        controller.initialize_obstacle(0, modes);

        // Build a mode history: mostly CV, a few turn_left observations
        // to ensure the mode exists in frequency weights
        for (int t = 0; t < 20; ++t) {
            controller.update_mode_observation(0, "constant_velocity", t);
        }
        // Sprinkle in a single turn_left observation so the mode has nonzero weight
        controller.update_mode_observation(0, "turn_left", 20);

        EgoState ego(0.0, 0.0, 0.0, 1.0);
        Eigen::Vector2d goal(15.0, 0.0);
        EgoDynamics ego_dyn(dt);

        const auto& cv_mode = modes.at("constant_velocity");
        const auto& tl_mode = modes.at("turn_left");

        Eigen::Vector4d obs_state;
        obs_state << obs_x, obs_y, obs_vx, 0.0;

        std::mt19937 rng(seed_base + trial);
        std::normal_distribution<double> noise_dist(0.0, 1.0);

        double min_clearance = 1e9;
        bool had_collision = false;

        for (int step = 21; step < sim_steps; ++step) {
            ObstacleState obs(obs_state(0), obs_state(1), obs_state(2), obs_state(3));

            // Record observed mode
            std::string current_mode = (step < switch_step) ? "constant_velocity" : "turn_left";
            controller.update_mode_observation(0, current_mode, step);

            std::map<int, ObstacleState> obs_map = {{0, obs}};
            auto mpc_result = controller.solve(ego, obs_map, goal, 1.5);

            double dist = (ego.position() - obs.position()).norm();
            double clearance = dist - combined_radius;
            min_clearance = std::min(min_clearance, clearance);
            if (clearance < 0) had_collision = true;

            // Propagate ego
            if (mpc_result.success && mpc_result.first_input().has_value()) {
                ego = ego_dyn.propagate(ego, mpc_result.first_input().value());
            } else {
                ego = ego_dyn.propagate(ego, EgoInput(-0.5, 0.0));
            }

            // Propagate obstacle with correct mode
            const ModeModel& active_mode = (step < switch_step) ? cv_mode : tl_mode;
            Eigen::Vector2d w(noise_dist(rng), noise_dist(rng));
            obs_state = active_mode.A * obs_state + active_mode.b + 0.5 * active_mode.G * w;
        }

        if (had_collision) result.collisions++;
        result.avg_min_clearance += min_clearance;
    }

    result.collision_rate = static_cast<double>(result.collisions) / num_trials;
    result.avg_min_clearance /= num_trials;
    return result;
}

static void experiment1_rare_mode_switch() {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "  Experiment 1: Rare Mode Switch Safety" << std::endl;
    std::cout << "  Obstacle switches from CV to turn_left at step 30." << std::endl;
    std::cout << "  Mode history: ~95% CV, ~5% turn_left." << std::endl;
    std::cout << "============================================================" << std::endl;

    std::vector<int> scenario_counts = {5, 10, 15, 30, 50};
    const int num_trials = 20;

    std::cout << std::left
              << std::setw(8)  << "S"
              << std::setw(12) << "Coverage"
              << std::setw(14) << "CollRate"
              << std::setw(14) << "AvgClearance"
              << std::endl;
    std::cout << std::string(48, '-') << std::endl;

    for (int S : scenario_counts) {
        auto off = run_experiment1_config(S, false, num_trials, 1000);
        auto on  = run_experiment1_config(S, true,  num_trials, 1000);

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(8) << S
                  << std::setw(12) << "OFF"
                  << std::setw(14) << off.collision_rate
                  << std::setw(14) << off.avg_min_clearance
                  << std::endl;
        std::cout << std::setw(8) << S
                  << std::setw(12) << "ON"
                  << std::setw(14) << on.collision_rate
                  << std::setw(14) << on.avg_min_clearance
                  << std::endl;
    }
}

// ============================================================================
// Experiment 2: Coverage Statistics
//
// 1 obstacle with 6 modes, dominant mode at 80%, 5 rare modes at 4% each.
// For each S, count how many trials have all 6 modes represented.
// ============================================================================

static void experiment2_coverage_statistics() {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "  Experiment 2: Coverage Statistics" << std::endl;
    std::cout << "  1 obstacle, 6 modes (80% + 5x4%). Check all-mode coverage." << std::endl;
    std::cout << "============================================================" << std::endl;

    const double dt = 0.1;
    const int horizon = 15;
    const int num_trials = 100;
    std::vector<int> scenario_counts = {5, 10, 15, 20};

    // Build mode models — use standard 6 modes
    auto modes = create_obstacle_mode_models(dt);

    // Build a mode history: dominant mode gets ~80%, rest ~4% each
    // We'll record 100 observations: 80 constant_velocity, 4 each of the rest
    ModeHistory mode_history(0, modes);
    std::vector<std::string> mode_names;
    for (const auto& [name, _] : modes) {
        mode_names.push_back(name);
    }

    // Find constant_velocity as dominant, rest are rare
    std::string dominant = "constant_velocity";
    int t = 0;
    for (int i = 0; i < 80; ++i, ++t) {
        mode_history.record_observation(t, dominant);
    }
    for (const auto& name : mode_names) {
        if (name == dominant) continue;
        for (int i = 0; i < 4; ++i, ++t) {
            mode_history.record_observation(t, name);
        }
    }

    std::map<int, ModeHistory> hist_map = {{0, mode_history}};
    ObstacleState obs(5.0, 1.0, -0.5, 0.0);
    std::map<int, ObstacleState> obs_map = {{0, obs}};

    int num_modes = static_cast<int>(mode_names.size());

    std::cout << std::left
              << std::setw(8)  << "S"
              << std::setw(12) << "Coverage"
              << std::setw(20) << "AllModesRepresented"
              << std::setw(16) << "AvgModesCovered"
              << std::endl;
    std::cout << std::string(56, '-') << std::endl;

    for (int S : scenario_counts) {
        for (int coverage_on = 0; coverage_on <= 1; ++coverage_on) {
            int all_covered_count = 0;
            double total_modes_covered = 0;

            for (int trial = 0; trial < num_trials; ++trial) {
                std::mt19937 rng(42 + trial);

                std::vector<Scenario> scenarios;
                if (coverage_on) {
                    scenarios = sample_scenarios_with_mode_coverage(
                        obs_map, hist_map, horizon, S,
                        WeightType::FREQUENCY, 0.9, t, &rng
                    );
                } else {
                    scenarios = sample_scenarios(
                        obs_map, hist_map, horizon, S,
                        WeightType::FREQUENCY, 0.9, t, &rng
                    );
                }

                // Count which modes are represented
                std::set<std::string> represented;
                for (const auto& sc : scenarios) {
                    for (const auto& [obs_id, traj] : sc.trajectories) {
                        represented.insert(traj.mode_id);
                    }
                }

                int modes_covered = static_cast<int>(represented.size());
                total_modes_covered += modes_covered;
                if (modes_covered == num_modes) all_covered_count++;
            }

            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(8)  << S
                      << std::setw(12) << (coverage_on ? "ON" : "OFF")
                      << std::setw(20) << (static_cast<double>(all_covered_count) / num_trials * 100.0)
                      << std::setw(16) << (total_modes_covered / num_trials)
                      << std::endl;
        }
    }
}

// ============================================================================
// Experiment 3: Epsilon Guarantee with Coverage
//
// Head-on scenario: ego→goal, obstacle at (7,Y) moving left.
// Obstacle occasionally switches modes (5% probability per step).
// Compare collision rate and clearance: coverage ON vs OFF.
// ============================================================================

struct Exp3Result {
    double noise_scale;
    bool coverage_on;
    int collisions;
    int num_trials;
    double collision_rate;
    double avg_min_clearance;
};

static Exp3Result run_experiment3_config(
    double noise_scale, bool coverage_on, int num_trials, int seed_base)
{
    const double dt = 0.1;
    const int sim_steps = 80;
    const double obs_x = 7.0;
    const double obs_y = 0.8;
    const double obs_vx = -0.8;
    const double mode_switch_prob = 0.05;

    Exp3Result result;
    result.noise_scale = noise_scale;
    result.coverage_on = coverage_on;
    result.num_trials = num_trials;
    result.collisions = 0;
    result.avg_min_clearance = 0.0;

    auto modes = create_obstacle_mode_models(dt);
    std::vector<std::string> mode_names;
    for (const auto& [name, _] : modes) {
        mode_names.push_back(name);
    }

    for (int trial = 0; trial < num_trials; ++trial) {
        auto pred_modes = create_modes_with_noise_scale(dt, noise_scale);

        ScenarioMPCConfig config;
        config.horizon = 15;
        config.dt = dt;
        config.num_scenarios = 15;
        config.weight_type = WeightType::FREQUENCY;
        config.recency_decay = 0.9;
        config.ego_radius = 0.8;
        config.obstacle_radius = 0.5;
        config.safety_margin = 0.2;
        config.ensure_mode_coverage = coverage_on;

        double combined_radius = config.ego_radius + config.obstacle_radius;

        AdaptiveScenarioMPC controller(config);
        controller.initialize_obstacle(0, pred_modes);

        EgoState ego(0.0, 0.0, 0.0, 1.0);
        Eigen::Vector2d goal(15.0, 0.0);
        EgoDynamics ego_dyn(dt);

        Eigen::Vector4d obs_state;
        obs_state << obs_x, obs_y, obs_vx, 0.0;

        std::mt19937 rng(seed_base + trial);
        std::normal_distribution<double> noise_dist(0.0, 1.0);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        std::string current_mode = "constant_velocity";
        double min_clearance = 1e9;
        bool had_collision = false;

        for (int step = 0; step < sim_steps; ++step) {
            // Random mode switch
            if (uniform(rng) < mode_switch_prob) {
                int idx = static_cast<int>(uniform(rng) * mode_names.size());
                idx = std::min(idx, static_cast<int>(mode_names.size()) - 1);
                current_mode = mode_names[idx];
            }

            ObstacleState obs(obs_state(0), obs_state(1), obs_state(2), obs_state(3));
            controller.update_mode_observation(0, current_mode, step);

            std::map<int, ObstacleState> obs_map = {{0, obs}};
            auto mpc_result = controller.solve(ego, obs_map, goal, 1.5);

            double dist = (ego.position() - obs.position()).norm();
            double clearance = dist - combined_radius;
            min_clearance = std::min(min_clearance, clearance);
            if (clearance < 0) had_collision = true;

            if (mpc_result.success && mpc_result.first_input().has_value()) {
                ego = ego_dyn.propagate(ego, mpc_result.first_input().value());
            } else {
                ego = ego_dyn.propagate(ego, EgoInput(-0.5, 0.0));
            }

            // Propagate obstacle
            const ModeModel& active_mode = modes.at(current_mode);
            Eigen::Vector2d w(noise_dist(rng), noise_dist(rng));
            obs_state = active_mode.A * obs_state + active_mode.b
                      + noise_scale * active_mode.G * w;
        }

        if (had_collision) result.collisions++;
        result.avg_min_clearance += min_clearance;
    }

    result.collision_rate = static_cast<double>(result.collisions) / num_trials;
    result.avg_min_clearance /= num_trials;
    return result;
}

static void experiment3_epsilon_with_coverage() {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "  Experiment 3: Epsilon Guarantee with Mode Coverage" << std::endl;
    std::cout << "  Head-on scenario, 5% mode switch probability per step." << std::endl;
    std::cout << "============================================================" << std::endl;

    std::vector<double> noise_scales = {0.0, 2.0, 5.0, 10.0};
    const int num_trials = 20;

    std::cout << std::left
              << std::setw(10) << "Noise"
              << std::setw(12) << "Coverage"
              << std::setw(14) << "CollRate"
              << std::setw(14) << "AvgClearance"
              << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    for (double ns : noise_scales) {
        auto off = run_experiment3_config(ns, false, num_trials, 5000);
        auto on  = run_experiment3_config(ns, true,  num_trials, 5000);

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << ns
                  << std::setw(12) << "OFF"
                  << std::setw(14) << off.collision_rate
                  << std::setw(14) << off.avg_min_clearance
                  << std::endl;
        std::cout << std::setw(10) << ns
                  << std::setw(12) << "ON"
                  << std::setw(14) << on.collision_rate
                  << std::setw(14) << on.avg_min_clearance
                  << std::endl;
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "  Mode Coverage Guarantee — Comparison Experiments" << std::endl;
    std::cout << "================================================================" << std::endl;

    experiment1_rare_mode_switch();
    experiment2_coverage_statistics();
    experiment3_epsilon_with_coverage();

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  All experiments complete." << std::endl;
    std::cout << "================================================================" << std::endl;

    return 0;
}
