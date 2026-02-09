/**
 * @file test_sqp_comparison.cpp
 * @brief Side-by-side comparison of SQP vs heuristic solver on uncertainty experiments.
 *
 * Runs the same stochastic obstacle scenarios with both solvers and prints
 * collision rates and clearance metrics for direct comparison.
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <random>
#include <vector>
#include <map>
#include <string>

#include "types.hpp"
#include "dynamics.hpp"
#include "mpc_controller.hpp"
#include "config.hpp"

using namespace scenario_mpc;

struct TrialResult {
    int collision_steps;
    double min_clearance;
    double final_goal_distance;
    bool goal_reached;
    double max_obs_deviation;
};

static std::map<std::string, ModeModel> create_modes_with_noise_scale(
    double dt, double noise_scale)
{
    auto modes = create_obstacle_mode_models(dt);
    for (auto& [id, mode] : modes) {
        mode.G *= noise_scale;
    }
    return modes;
}

static TrialResult run_trial(
    bool use_sqp,
    double actual_noise,
    double prediction_noise,
    int num_scenarios,
    double obs_y_offset,
    double obs_x_start,
    double obs_vx,
    int trial_seed,
    int sim_steps = 80,
    double dt = 0.1)
{
    TrialResult result;
    result.collision_steps = 0;
    result.min_clearance = 1e9;
    result.max_obs_deviation = 0.0;

    auto pred_modes = create_modes_with_noise_scale(dt, prediction_noise);
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
    config.use_sqp_solver = use_sqp;

    double combined_radius = config.ego_radius + config.obstacle_radius;

    AdaptiveScenarioMPC controller(config);
    controller.initialize_obstacle(0, pred_modes);

    EgoState ego(0.0, 0.0, 0.0, 1.0);
    Eigen::Vector2d goal(15.0, 0.0);

    Eigen::Vector4d obs_state;
    obs_state << obs_x_start, obs_y_offset, obs_vx, 0.0;
    Eigen::Vector4d obs_nominal = obs_state;

    EgoDynamics ego_dyn(dt);
    std::mt19937 rng(trial_seed);
    std::normal_distribution<double> noise_dist(0.0, 1.0);

    for (int step = 0; step < sim_steps; ++step) {
        ObstacleState obs(obs_state(0), obs_state(1), obs_state(2), obs_state(3));
        controller.update_mode_observation(0, "constant_velocity", step);
        std::map<int, ObstacleState> obs_map = {{0, obs}};
        auto mpc_result = controller.solve(ego, obs_map, goal, 1.5);

        double dist = (ego.position() - obs.position()).norm();
        double clearance = dist - combined_radius;
        result.min_clearance = std::min(result.min_clearance, clearance);
        if (clearance < 0) result.collision_steps++;

        Eigen::Vector2d deviation = obs_state.head<2>() - obs_nominal.head<2>();
        result.max_obs_deviation = std::max(result.max_obs_deviation, deviation.norm());

        if (mpc_result.success && mpc_result.first_input().has_value()) {
            ego = ego_dyn.propagate(ego, mpc_result.first_input().value());
        } else {
            ego = ego_dyn.propagate(ego, EgoInput(-0.5, 0.0));
        }

        Eigen::Vector2d w(noise_dist(rng), noise_dist(rng));
        obs_state = cv_mode.A * obs_state + cv_mode.b + actual_noise * cv_mode.G * w;
        obs_nominal = cv_mode.A * obs_nominal + cv_mode.b;
    }

    result.final_goal_distance = (ego.position() - goal).norm();
    result.goal_reached = result.final_goal_distance < 3.0;
    return result;
}

// ============================================================================
// Comparison: Baseline offsets (deterministic obstacle)
// ============================================================================

void compare_baseline() {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Comparison 1: Baseline (deterministic obstacle, noise=0)" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::left
              << std::setw(10) << "Offset"
              << " | " << std::setw(18) << "Heuristic Clear."
              << " | " << std::setw(18) << "SQP Clearance"
              << " | " << "Improvement" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    std::vector<double> offsets = {0.5, 1.0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0};

    for (double yoff : offsets) {
        auto r_heur = run_trial(false, 0.0, 1.0, 15, yoff, 7.0, -0.8, 42);
        auto r_sqp  = run_trial(true,  0.0, 1.0, 15, yoff, 7.0, -0.8, 42);

        std::string h_status = (r_heur.collision_steps > 0) ? " COLLISION" : "";
        std::string s_status = (r_sqp.collision_steps > 0) ? " COLLISION" : "";

        double improvement = r_sqp.min_clearance - r_heur.min_clearance;
        std::string imp_str = (improvement > 0 ? "+" : "") +
            std::to_string(improvement).substr(0, 6) + "m";

        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(10) << yoff
                  << " | " << std::setprecision(3) << std::setw(8) << r_heur.min_clearance
                  << "m" << std::setw(9) << h_status
                  << " | " << std::setw(8) << r_sqp.min_clearance
                  << "m" << std::setw(9) << s_status
                  << " | " << imp_str << std::endl;
    }
}

// ============================================================================
// Comparison: Matched noise sweep at critical offset
// ============================================================================

void compare_matched_noise() {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Comparison 2: Matched Noise (offset=2.0m, 10 trials each)" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::left
              << std::setw(8) << "Noise"
              << " | " << std::setw(10) << "Heur Coll%"
              << " | " << std::setw(10) << "Heur Clear"
              << " | " << std::setw(10) << "SQP Coll%"
              << " | " << std::setw(10) << "SQP Clear"
              << " | " << "Winner" << std::endl;
    std::cout << std::string(78, '-') << std::endl;

    double yoff = 2.0;
    int num_trials = 10;
    std::vector<double> noise_scales = {0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0};

    for (double ns : noise_scales) {
        int h_colls = 0, s_colls = 0;
        double h_clear = 0, s_clear = 0;

        for (int trial = 0; trial < num_trials; ++trial) {
            auto rh = run_trial(false, ns, ns, 15, yoff, 7.0, -0.8, 42 + trial);
            auto rs = run_trial(true,  ns, ns, 15, yoff, 7.0, -0.8, 42 + trial);

            if (rh.collision_steps > 0) h_colls++;
            if (rs.collision_steps > 0) s_colls++;
            h_clear += rh.min_clearance;
            s_clear += rs.min_clearance;
        }

        double h_rate = 100.0 * h_colls / num_trials;
        double s_rate = 100.0 * s_colls / num_trials;
        double h_avg = h_clear / num_trials;
        double s_avg = s_clear / num_trials;

        std::string winner = (s_rate < h_rate) ? "SQP" :
                             (s_rate > h_rate) ? "Heuristic" :
                             (s_avg > h_avg) ? "SQP" : "Tie";

        std::cout << std::fixed
                  << std::setw(8) << std::setprecision(0) << ns
                  << " | " << std::setw(10) << std::setprecision(0) << h_rate << "%"
                  << " | " << std::setw(10) << std::setprecision(3) << h_avg
                  << " | " << std::setw(10) << std::setprecision(0) << s_rate << "%"
                  << " | " << std::setw(10) << std::setprecision(3) << s_avg
                  << " | " << winner << std::endl;
    }
}

// ============================================================================
// Comparison: Scenario count recovery at noise=10
// ============================================================================

void compare_scenario_recovery() {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Comparison 3: Scenario Recovery (offset=2.0, noise=10)" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::left
              << std::setw(6) << "S"
              << " | " << std::setw(10) << "Heur Coll%"
              << " | " << std::setw(10) << "Heur Clear"
              << " | " << std::setw(10) << "SQP Coll%"
              << " | " << std::setw(10) << "SQP Clear"
              << " | " << "Winner" << std::endl;
    std::cout << std::string(76, '-') << std::endl;

    double yoff = 2.0;
    double ns = 10.0;
    int num_trials = 20;
    std::vector<int> scenario_counts = {3, 5, 10, 15, 30, 50, 100};

    for (int S : scenario_counts) {
        int h_colls = 0, s_colls = 0;
        double h_clear = 0, s_clear = 0;

        for (int trial = 0; trial < num_trials; ++trial) {
            auto rh = run_trial(false, ns, ns, S, yoff, 7.0, -0.8, 42 + trial);
            auto rs = run_trial(true,  ns, ns, S, yoff, 7.0, -0.8, 42 + trial);

            if (rh.collision_steps > 0) h_colls++;
            if (rs.collision_steps > 0) s_colls++;
            h_clear += rh.min_clearance;
            s_clear += rs.min_clearance;
        }

        double h_rate = 100.0 * h_colls / num_trials;
        double s_rate = 100.0 * s_colls / num_trials;
        double h_avg = h_clear / num_trials;
        double s_avg = s_clear / num_trials;

        std::string winner = (s_rate < h_rate) ? "SQP" :
                             (s_rate > h_rate) ? "Heuristic" :
                             (s_avg > h_avg) ? "SQP" : "Tie";

        std::cout << std::fixed
                  << std::setw(6) << S
                  << " | " << std::setw(10) << std::setprecision(0) << h_rate << "%"
                  << " | " << std::setw(10) << std::setprecision(3) << h_avg
                  << " | " << std::setw(10) << std::setprecision(0) << s_rate << "%"
                  << " | " << std::setw(10) << std::setprecision(3) << s_avg
                  << " | " << winner << std::endl;
    }
}

// ============================================================================
// Comparison: Model mismatch
// ============================================================================

void compare_mismatch() {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Comparison 4: Model Mismatch (offset=2.0, 20 trials)" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::left
              << std::setw(8) << "ActNoise"
              << std::setw(8) << "Ratio"
              << " | " << std::setw(10) << "Heur Coll%"
              << " | " << std::setw(10) << "SQP Coll%"
              << " | " << std::setw(10) << "Heur Clear"
              << " | " << std::setw(10) << "SQP Clear"
              << std::endl;
    std::cout << std::string(76, '-') << std::endl;

    double yoff = 2.0;
    int num_trials = 20;
    std::vector<double> actual_noises = {2.0, 5.0, 10.0};
    std::vector<double> pred_ratios = {0.0, 0.5, 1.0, 2.0, 4.0};

    for (double an : actual_noises) {
        for (double ratio : pred_ratios) {
            double pn = an * ratio;
            int h_colls = 0, s_colls = 0;
            double h_clear = 0, s_clear = 0;

            for (int trial = 0; trial < num_trials; ++trial) {
                auto rh = run_trial(false, an, pn, 15, yoff, 7.0, -0.8, 42 + trial);
                auto rs = run_trial(true,  an, pn, 15, yoff, 7.0, -0.8, 42 + trial);

                if (rh.collision_steps > 0) h_colls++;
                if (rs.collision_steps > 0) s_colls++;
                h_clear += rh.min_clearance;
                s_clear += rs.min_clearance;
            }

            std::cout << std::fixed
                      << std::setw(8) << std::setprecision(0) << an
                      << std::setw(8) << std::setprecision(2) << ratio
                      << " | " << std::setw(10) << std::setprecision(0) << (100.0*h_colls/num_trials) << "%"
                      << " | " << std::setw(10) << std::setprecision(0) << (100.0*s_colls/num_trials) << "%"
                      << " | " << std::setw(10) << std::setprecision(3) << (h_clear/num_trials)
                      << " | " << std::setw(10) << std::setprecision(3) << (s_clear/num_trials)
                      << std::endl;
        }
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "====================================================================" << std::endl;
    std::cout << "  SQP vs Heuristic Solver: Side-by-Side Comparison" << std::endl;
    std::cout << "  Same random seeds, same scenarios, same obstacle dynamics" << std::endl;
    std::cout << "====================================================================" << std::endl;

    compare_baseline();
    compare_matched_noise();
    compare_scenario_recovery();
    compare_mismatch();

    std::cout << "\n====================================================================" << std::endl;
    std::cout << "  Comparison complete." << std::endl;
    std::cout << "====================================================================" << std::endl;

    return 0;
}
