/**
 * @file test_uncertainty_limits.cpp
 * @brief Determine the limits of obstacle dynamics uncertainty the MPC can handle.
 *
 * Key insight: noise must be applied to the ACTUAL obstacle trajectory, not just
 * the MPC's prediction model. The obstacle propagates stochastically via:
 *   x_{k+1} = A * x_k + b + noise_scale * G * w_k,   w_k ~ N(0, I)
 *
 * Experiments:
 *   1. Baseline sweep: deterministic obstacle, find safe offsets
 *   2. Matched model: actual + prediction noise matched, sweep noise scale
 *   3. Phase diagram: (offset x noise_scale) with actual stochastic obstacle
 *   4. Model mismatch: prediction noise != actual noise
 *   5. Scenario count recovery: can more scenarios compensate?
 *
 * Scenario: ego (0,0)->goal(15,0), obstacle at (7,Y) moving left.
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <numeric>
#include <vector>
#include <map>
#include <string>

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
// Result structure
// ============================================================================

struct TrialResult {
    double actual_noise;         // noise applied to real obstacle
    double prediction_noise;     // noise in MPC's model
    int num_scenarios;
    double obs_y_offset;
    int trial_seed;
    int total_steps;
    int collision_steps;
    double min_clearance;        // min(distance - combined_radius)
    double final_goal_distance;
    bool goal_reached;
    double avg_min_distance;
    int scenarios_after_pruning;
    double max_obs_deviation;    // max deviation from nominal trajectory
};

// ============================================================================
// Create mode models with scaled noise
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
// Run a single trial with STOCHASTIC obstacle
// ============================================================================

static TrialResult run_trial(
    double actual_noise,        // noise scale for actual obstacle trajectory
    double prediction_noise,    // noise scale for MPC prediction model
    int num_scenarios,
    double obs_y_offset,
    double obs_x_start,
    double obs_vx,
    int trial_seed,
    int sim_steps = 80,
    double dt = 0.1)
{
    TrialResult result;
    result.actual_noise = actual_noise;
    result.prediction_noise = prediction_noise;
    result.num_scenarios = num_scenarios;
    result.obs_y_offset = obs_y_offset;
    result.trial_seed = trial_seed;
    result.total_steps = sim_steps;
    result.collision_steps = 0;
    result.min_clearance = 1e9;
    result.max_obs_deviation = 0.0;
    result.scenarios_after_pruning = 0;

    // MPC uses prediction_noise for its model
    auto pred_modes = create_modes_with_noise_scale(dt, prediction_noise);

    // Get the actual dynamics model (constant_velocity) for propagating the real obstacle
    auto actual_modes = create_obstacle_mode_models(dt);
    const auto& cv_mode = actual_modes.at("constant_velocity");

    ScenarioMPCConfig config;
    config.horizon = 15;
    config.dt = dt;
    config.num_scenarios = num_scenarios;
    config.weight_type = WeightType::FREQUENCY;
    config.recency_decay = 0.9;
    config.ego_radius = 0.8;
    config.obstacle_radius = 0.5;
    config.safety_margin = 0.2;
    config.goal_weight = 10.0;
    config.velocity_weight = 2.0;

    double combined_radius = config.ego_radius + config.obstacle_radius;

    AdaptiveScenarioMPC controller(config);
    controller.initialize_obstacle(0, pred_modes);

    EgoState ego(0.0, 0.0, 0.0, 1.0);
    Eigen::Vector2d goal(15.0, 0.0);

    // Obstacle state as 4D vector for dynamics propagation
    Eigen::Vector4d obs_state;
    obs_state << obs_x_start, obs_y_offset, obs_vx, 0.0;

    // Nominal trajectory for deviation tracking
    Eigen::Vector4d obs_nominal = obs_state;

    EgoDynamics ego_dyn(dt);

    // Random number generator for actual obstacle noise
    std::mt19937 rng(trial_seed);
    std::normal_distribution<double> noise_dist(0.0, 1.0);

    double total_min_dist = 0.0;
    int pruning_sum = 0;

    for (int step = 0; step < sim_steps; ++step) {
        // Create ObstacleState from current 4D state
        ObstacleState obs(obs_state(0), obs_state(1), obs_state(2), obs_state(3));

        controller.update_mode_observation(0, "constant_velocity", step);
        std::map<int, ObstacleState> obs_map = {{0, obs}};
        auto mpc_result = controller.solve(ego, obs_map, goal, 1.5);

        double dist = (ego.position() - obs.position()).norm();
        double clearance = dist - combined_radius;
        result.min_clearance = std::min(result.min_clearance, clearance);
        total_min_dist += dist;
        if (clearance < 0) result.collision_steps++;

        pruning_sum += static_cast<int>(controller.scenarios().size());

        // Track deviation from nominal (noise-free) trajectory
        Eigen::Vector2d deviation = obs_state.head<2>() - obs_nominal.head<2>();
        result.max_obs_deviation = std::max(result.max_obs_deviation, deviation.norm());

        // Propagate ego
        if (mpc_result.success && mpc_result.first_input().has_value()) {
            ego = ego_dyn.propagate(ego, mpc_result.first_input().value());
        } else {
            ego = ego_dyn.propagate(ego, EgoInput(-0.5, 0.0));
        }

        // Propagate actual obstacle with STOCHASTIC dynamics
        // x_{k+1} = A * x_k + b + actual_noise * G * w_k
        Eigen::Vector2d w(noise_dist(rng), noise_dist(rng));
        obs_state = cv_mode.A * obs_state + cv_mode.b
                  + actual_noise * cv_mode.G * w;

        // Propagate nominal (deterministic) obstacle for deviation tracking
        obs_nominal = cv_mode.A * obs_nominal + cv_mode.b;
    }

    result.avg_min_distance = total_min_dist / sim_steps;
    result.final_goal_distance = (ego.position() - goal).norm();
    result.goal_reached = result.final_goal_distance < 3.0;
    result.scenarios_after_pruning = pruning_sum / std::max(1, sim_steps);

    return result;
}

// ============================================================================
// Experiment 1: Baseline — deterministic obstacle, find safe offsets
// ============================================================================

void experiment_baseline_offsets() {
    std::cout << "\n============================================" << std::endl;
    std::cout << "  Experiment 1: Baseline Capability" << std::endl;
    std::cout << "  Deterministic obstacle (noise=0)" << std::endl;
    std::cout << "============================================" << std::endl;

    std::vector<double> y_offsets = {
        0.5, 1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0
    };

    std::ofstream csv("uncertainty_baseline.csv");
    csv << "obs_y_offset,trial,collision_steps,min_clearance,final_goal_dist,goal_reached" << std::endl;

    std::cout << std::left
              << std::setw(12) << "Y_offset"
              << std::setw(14) << "CollRate"
              << std::setw(14) << "Clearance"
              << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    for (double yoff : y_offsets) {
        // deterministic: actual_noise=0, prediction_noise=1.0
        auto r = run_trial(0.0, 1.0, 15, yoff, 7.0, -0.8, 42);
        csv << std::fixed << std::setprecision(4)
            << yoff << ",0,"
            << r.collision_steps << "," << r.min_clearance << ","
            << r.final_goal_distance << "," << (r.goal_reached?1:0) << std::endl;

        std::string status = (r.collision_steps > 0) ? "COLLISION" : "SAFE";
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(12) << yoff
                  << std::setw(14) << std::setprecision(3) << r.min_clearance
                  << status << std::endl;
    }
    csv.close();
}

// ============================================================================
// Experiment 2: Matched model noise sweep
// ============================================================================

void experiment_matched_noise() {
    std::cout << "\n============================================" << std::endl;
    std::cout << "  Experiment 2: Matched Model Noise Sweep" << std::endl;
    std::cout << "  actual_noise = prediction_noise, sweep scale" << std::endl;
    std::cout << "============================================" << std::endl;

    // Use offsets that are near the safe boundary
    std::vector<double> y_offsets = {1.5, 2.0, 2.5, 3.0, 4.0};
    std::vector<double> noise_scales = {
        0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0
    };
    int num_trials = 10;
    int num_scenarios = 15;

    std::ofstream csv("uncertainty_matched_noise.csv");
    csv << "obs_y_offset,noise_scale,trial,collision_steps,min_clearance,"
        << "final_goal_dist,goal_reached,max_obs_deviation,scenarios_after_pruning" << std::endl;

    for (double yoff : y_offsets) {
        std::cout << "\n  y_offset = " << yoff << std::endl;

        for (double ns : noise_scales) {
            int colls = 0;
            double sum_clear = 0;
            double sum_dev = 0;

            for (int trial = 0; trial < num_trials; ++trial) {
                auto r = run_trial(ns, ns, num_scenarios, yoff, 7.0, -0.8, 42+trial);
                csv << std::fixed << std::setprecision(4)
                    << yoff << "," << ns << "," << trial << ","
                    << r.collision_steps << "," << r.min_clearance << ","
                    << r.final_goal_distance << "," << (r.goal_reached?1:0) << ","
                    << r.max_obs_deviation << "," << r.scenarios_after_pruning << std::endl;

                if (r.collision_steps > 0) colls++;
                sum_clear += r.min_clearance;
                sum_dev += r.max_obs_deviation;
            }

            double coll_rate = 100.0 * colls / num_trials;
            std::cout << "    noise=" << std::setw(6) << std::fixed << std::setprecision(0) << ns
                      << " | coll=" << std::setw(4) << coll_rate << "%"
                      << " | clear=" << std::setprecision(3) << std::setw(7) << (sum_clear/num_trials)
                      << " | dev=" << std::setprecision(2) << (sum_dev/num_trials) << "m"
                      << std::endl;
        }
    }
    csv.close();
}

// ============================================================================
// Experiment 3: Phase diagram — finer grid near critical boundary
// ============================================================================

void experiment_phase_diagram() {
    std::cout << "\n============================================" << std::endl;
    std::cout << "  Experiment 3: Phase Diagram" << std::endl;
    std::cout << "  (offset x noise_scale) -> collision rate" << std::endl;
    std::cout << "============================================" << std::endl;

    std::vector<double> y_offsets = {1.5, 1.8, 2.0, 2.5, 3.0, 4.0};
    std::vector<double> noise_scales = {
        0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0
    };
    int num_trials = 10;
    int num_scenarios = 15;

    std::ofstream csv("uncertainty_phase_diagram.csv");
    csv << "obs_y_offset,noise_scale,trial,collision_steps,min_clearance,"
        << "final_goal_dist,goal_reached,max_obs_deviation" << std::endl;

    for (double yoff : y_offsets) {
        std::cout << "\n  y_offset = " << yoff << std::endl;

        for (double ns : noise_scales) {
            int colls = 0;
            double sum_clear = 0;

            for (int trial = 0; trial < num_trials; ++trial) {
                auto r = run_trial(ns, ns, num_scenarios, yoff, 7.0, -0.8, 42+trial);
                csv << std::fixed << std::setprecision(4)
                    << yoff << "," << ns << "," << trial << ","
                    << r.collision_steps << "," << r.min_clearance << ","
                    << r.final_goal_distance << "," << (r.goal_reached?1:0) << ","
                    << r.max_obs_deviation << std::endl;

                if (r.collision_steps > 0) colls++;
                sum_clear += r.min_clearance;
            }

            double coll_rate = 100.0 * colls / num_trials;
            std::cout << "    noise=" << std::setw(6) << std::fixed << std::setprecision(0) << ns
                      << " | coll=" << std::setw(4) << coll_rate << "%"
                      << " | clear=" << std::setprecision(3) << (sum_clear/num_trials)
                      << std::endl;
        }
    }
    csv.close();
}

// ============================================================================
// Experiment 4: Model mismatch — prediction under/over estimates actual noise
// ============================================================================

void experiment_model_mismatch() {
    std::cout << "\n============================================" << std::endl;
    std::cout << "  Experiment 4: Model Mismatch" << std::endl;
    std::cout << "  actual_noise in critical regime, vary pred" << std::endl;
    std::cout << "============================================" << std::endl;

    double y_offset = 2.0;   // Near the safe boundary
    int num_scenarios = 15;
    int num_trials = 20;

    // Focus on the critical noise regime where collisions actually happen
    std::vector<double> actual_noises = {2.0, 5.0, 10.0};
    std::vector<double> pred_ratios = {0.0, 0.25, 0.5, 1.0, 2.0, 4.0};

    std::ofstream csv("uncertainty_mismatch.csv");
    csv << "actual_noise,prediction_noise,pred_ratio,trial,collision_steps,min_clearance,"
        << "final_goal_dist,goal_reached,max_obs_deviation" << std::endl;

    for (double an : actual_noises) {
        std::cout << "\n  actual_noise = " << an << std::endl;

        for (double ratio : pred_ratios) {
            double pn = an * ratio;
            int colls = 0;
            double sum_clear = 0;

            for (int trial = 0; trial < num_trials; ++trial) {
                auto r = run_trial(an, pn, num_scenarios, y_offset, 7.0, -0.8, 42+trial);
                csv << std::fixed << std::setprecision(4)
                    << an << "," << pn << "," << ratio << "," << trial << ","
                    << r.collision_steps << "," << r.min_clearance << ","
                    << r.final_goal_distance << "," << (r.goal_reached?1:0) << ","
                    << r.max_obs_deviation << std::endl;

                if (r.collision_steps > 0) colls++;
                sum_clear += r.min_clearance;
            }

            double coll_rate = 100.0 * colls / num_trials;
            std::cout << "    pred_ratio=" << std::setw(5) << std::fixed << std::setprecision(2) << ratio
                      << " (pred=" << std::setprecision(0) << std::setw(5) << pn << ")"
                      << " | coll=" << std::setw(4) << coll_rate << "%"
                      << " | clear=" << std::setprecision(3) << (sum_clear/num_trials)
                      << std::endl;
        }
    }
    csv.close();
}

// ============================================================================
// Experiment 5: Scenario count recovery at critical noise levels
// ============================================================================

void experiment_scenario_recovery() {
    std::cout << "\n============================================" << std::endl;
    std::cout << "  Experiment 5: Scenario Count Recovery" << std::endl;
    std::cout << "  Can more scenarios compensate for noise?" << std::endl;
    std::cout << "============================================" << std::endl;

    double y_offset = 2.0;
    // Focus on critical noise regime
    std::vector<double> noise_scales = {2.0, 5.0, 10.0};
    std::vector<int> scenario_counts = {3, 5, 10, 15, 20, 30, 50, 75, 100};
    int num_trials = 20;

    std::ofstream csv("uncertainty_scenario_recovery.csv");
    csv << "noise_scale,num_scenarios,trial,collision_steps,min_clearance,"
        << "final_goal_dist,goal_reached,max_obs_deviation" << std::endl;

    for (double ns : noise_scales) {
        std::cout << "\n  noise_scale = " << ns << std::endl;

        for (int S : scenario_counts) {
            int colls = 0;
            double sum_clear = 0;

            for (int trial = 0; trial < num_trials; ++trial) {
                auto r = run_trial(ns, ns, S, y_offset, 7.0, -0.8, 42+trial);
                csv << std::fixed << std::setprecision(4)
                    << ns << "," << S << "," << trial << ","
                    << r.collision_steps << "," << r.min_clearance << ","
                    << r.final_goal_distance << "," << (r.goal_reached?1:0) << ","
                    << r.max_obs_deviation << std::endl;

                if (r.collision_steps > 0) colls++;
                sum_clear += r.min_clearance;
            }

            double coll_rate = 100.0 * colls / num_trials;
            std::cout << "    S=" << std::setw(3) << S
                      << " | coll=" << std::setw(4) << std::fixed << std::setprecision(0) << coll_rate << "%"
                      << " | clear=" << std::setprecision(3) << (sum_clear/num_trials)
                      << std::endl;
        }
    }
    csv.close();
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "====================================================================" << std::endl;
    std::cout << "  Obstacle Dynamics Uncertainty Limits Experiment" << std::endl;
    std::cout << "  Scenario: Ego (0,0)->goal(15,0), Obstacle at (7,Y) moving left" << std::endl;
    std::cout << "  Combined collision radius: 1.3m + 0.2m safety = 1.5m" << std::endl;
    std::cout << "  Noise applied to ACTUAL obstacle: x_{k+1}=A*x+b+noise*G*w" << std::endl;
    std::cout << "====================================================================" << std::endl;

    experiment_baseline_offsets();
    experiment_matched_noise();
    experiment_phase_diagram();
    experiment_model_mismatch();
    experiment_scenario_recovery();

    // Call visualization
    const std::string VISUALIZE_SCRIPT = "../examples/visualize_uncertainty_limits.py";
    std::string cmd = "python3 " + VISUALIZE_SCRIPT + " 2>&1";
    std::cout << "\nGenerating uncertainty limit visualizations..." << std::endl;
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "Warning: Visualization script failed (return code " << ret << ")" << std::endl;
    }

    return 0;
}
