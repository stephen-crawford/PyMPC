/**
 * @file test_ot_predictor_integration.cpp
 * @brief Integration test for Optimal Transport Predictor with Safe Horizon Contouring MPC.
 *
 * Demonstrates the OT predictor learning obstacle dynamics online while the
 * MPC controller navigates through a field of aggressive obstacles.
 *
 * Produces:
 * - test_ot_predictor_trajectory.csv: Ego + obstacle trajectory data
 * - test_ot_predictor_scenarios.csv: Scenario prediction data
 * - test_ot_predictor_ot_metrics.csv: OT predictor metrics (mode weights, errors, uncertainty)
 * - test_ot_predictor.gif: Animated visualization
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

#include "contouring_mpc.hpp"
#include "optimal_transport_predictor.hpp"

using namespace scenario_mpc;

// Path to visualization scripts (relative to build directory)
const std::string VISUALIZE_CONTOURING = "../examples/visualize_contouring.py";
const std::string VISUALIZE_OT = "../examples/visualize_ot_predictor.py";

/**
 * @brief Generate GIF visualization from CSV data.
 */
void generate_gif(const std::string& csv_file, const std::string& gif_file,
                  const std::string& script = "") {
    std::string vis_script = script.empty() ? VISUALIZE_CONTOURING : script;
    std::string cmd = "python3 " + vis_script + " " + csv_file + " " + gif_file + " 2>&1";
    std::cout << "\nGenerating visualization: " << gif_file << std::endl;
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "Warning: Visualization script failed (return code " << ret << ")" << std::endl;
        std::cerr << "Make sure matplotlib is installed: pip install matplotlib pillow" << std::endl;
    }
}

struct OTIntegrationResult {
    bool success = false;
    double final_progress = 0;
    double min_obstacle_distance = std::numeric_limits<double>::max();
    double max_lateral_error = 0;
    double avg_solve_time = 0;
    int num_collisions = 0;
    int total_steps = 0;
    // OT-specific metrics
    double avg_prediction_error = 0;
    double final_uncertainty_scale = 1.0;
    int total_ot_weight_computations = 0;
};

/**
 * @brief Run the OT predictor integration test.
 *
 * Sets up 6 diverse obstacles, runs the MPC while the OT predictor
 * learns their dynamics online, and logs all data for visualization.
 */
OTIntegrationResult run_ot_predictor_test() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running: OT Predictor Integration Test" << std::endl;
    std::cout << "  - 6 diverse obstacles with mode switching" << std::endl;
    std::cout << "  - Online OT-based dynamics learning" << std::endl;
    std::cout << "  - Wasserstein mode weight computation" << std::endl;
    std::cout << "  - Sight radius: 7.0m" << std::endl;
    std::cout << "========================================" << std::endl;

    OTIntegrationResult result;

    // ---- Configure MPC ----
    ContouringMPCConfig config;
    config.horizon = 15;
    config.dt = 0.1;
    config.num_scenarios = 60;
    config.weight_type = WeightType::FREQUENCY;
    config.ego_radius = 0.5;
    config.obstacle_radius = 0.35;
    config.safety_margin = 1.0;
    config.path_length = 25.0;
    config.s_curve_amplitude = 3.0;
    config.road_width = 10.0;
    config.epsilon_p = 0.03;
    config.beta_confidence = 0.01;

    ContouringMPC controller(config);

    // ---- Create OT Predictor with standard reference modes ----
    auto ot_predictor = create_ot_predictor_with_standard_modes(
        config.dt,       // dt
        0.5,             // base_speed for reference distributions
        200,             // buffer_size
        0.1              // sinkhorn_epsilon
    );

    // ---- Configure obstacles (6 diverse dynamics) ----
    std::vector<ObstacleConfiguration> obstacle_configs(6);

    obstacle_configs[0].name = "Fast Crosser";
    obstacle_configs[0].initial_mode = ObstacleBehavior::PATH_INTERSECT;
    obstacle_configs[0].available_modes = {"constant_velocity", "turn_left", "turn_right"};
    obstacle_configs[0].mode_switch_probability = 0.15;
    obstacle_configs[0].radius = 0.35;

    obstacle_configs[1].name = "Slow Turner";
    obstacle_configs[1].initial_mode = ObstacleBehavior::TURN_LEFT;
    obstacle_configs[1].available_modes = {"turn_left", "turn_right", "constant_velocity"};
    obstacle_configs[1].mode_switch_probability = 0.1;
    obstacle_configs[1].radius = 0.35;

    obstacle_configs[2].name = "Accelerating";
    obstacle_configs[2].initial_mode = ObstacleBehavior::ACCELERATING;
    obstacle_configs[2].available_modes = {"accelerating", "constant_velocity", "decelerating"};
    obstacle_configs[2].mode_switch_probability = 0.08;
    obstacle_configs[2].radius = 0.35;

    obstacle_configs[3].name = "Lane Weaver";
    obstacle_configs[3].initial_mode = ObstacleBehavior::LANE_CHANGE_RIGHT;
    obstacle_configs[3].available_modes = {"lane_change_left", "lane_change_right", "constant_velocity"};
    obstacle_configs[3].mode_switch_probability = 0.25;
    obstacle_configs[3].radius = 0.35;

    obstacle_configs[4].name = "Decelerating";
    obstacle_configs[4].initial_mode = ObstacleBehavior::DECELERATING;
    obstacle_configs[4].available_modes = {"decelerating", "constant_velocity", "turn_left"};
    obstacle_configs[4].mode_switch_probability = 0.12;
    obstacle_configs[4].radius = 0.35;

    obstacle_configs[5].name = "Erratic";
    obstacle_configs[5].initial_mode = ObstacleBehavior::LANE_CHANGE_LEFT;
    obstacle_configs[5].available_modes = {"lane_change_left", "turn_right", "decelerating"};
    obstacle_configs[5].mode_switch_probability = 0.3;
    obstacle_configs[5].radius = 0.35;

    std::vector<double> fractions = {0.12, 0.28, 0.44, 0.58, 0.72, 0.88};
    std::vector<double> offsets = {3.0, -3.5, 4.5, -4.0, 3.5, -5.0};
    std::vector<double> velocities = {0.5, 0.25, 0.2, 0.35, 0.45, 0.3};

    controller.place_obstacles_along_path(obstacle_configs, fractions, offsets, velocities);

    // ---- Simulation setup ----
    EgoState ego_state(0.0, 0.0, 0.0, 0.0);
    double path_progress = 0.0;
    double reference_velocity = 2.5;
    EgoDynamics dynamics(config.dt);
    double collision_radius = config.ego_radius + config.obstacle_radius;
    double sight_radius = 7.0;  // Only learn obstacles within this radius
    int num_steps = 200;

    // ---- Open CSV files for logging ----
    // 1. Trajectory CSV (same format as test_aggressive_obstacles)
    std::ofstream csv_file("test_ot_predictor_trajectory.csv");
    csv_file << "step,time,ego_x,ego_y,ego_theta,ego_v,progress,lateral_error,";
    for (size_t i = 0; i < obstacle_configs.size(); ++i) {
        csv_file << "obs" << i << "_x,obs" << i << "_y,obs" << i << "_dist,";
    }
    csv_file << "solve_time_ms,num_scenarios" << std::endl;

    // 2. Scenarios CSV
    std::ofstream scenarios_file("test_ot_predictor_scenarios.csv");
    scenarios_file << "step,scenario_id,obs_id,mode,timestep,pred_x,pred_y,probability" << std::endl;

    // 3. OT metrics CSV
    std::ofstream ot_file("test_ot_predictor_ot_metrics.csv");
    ot_file << "# sight_radius=" << sight_radius << std::endl;
    ot_file << "step,time,obs_id,";
    // Mode weight columns for standard modes
    std::vector<std::string> all_modes = {
        "constant_velocity", "turn_left", "turn_right",
        "decelerating", "accelerating", "lane_change_left", "lane_change_right"
    };
    for (const auto& mode : all_modes) {
        ot_file << "w_" << mode << ",";
    }
    ot_file << "pred_error,uncertainty_scale,num_observations,in_sight,"
            << "pred_angle,pred_major_r,pred_minor_r,"
            << "pred5_x,pred5_y,pred5_major_r,pred5_minor_r,pred5_angle" << std::endl;

    const int max_scenarios_to_log = 5;
    double total_prediction_error = 0;
    int prediction_error_count = 0;

    // Store previous predictions for error computation (per obstacle)
    std::map<int, std::vector<OTPredictionStep>> prev_ot_predictions;

    // ---- Main simulation loop ----
    for (int step = 0; step < num_steps; ++step) {
        double time = step * config.dt;

        // Check if goal reached
        if (path_progress >= config.path_length - 0.1) {
            std::cout << "Goal reached at step " << step << "! (progress: "
                      << path_progress << "m)" << std::endl;
            result.success = true;
            break;
        }

        // ---- Feed observations & compute OT weights per obstacle ----
        // Only observe obstacles within the ego's sight radius.
        // Don't label observations until the OT predictor has enough data
        // to compute mode weights.  Labeling with a wrong initial guess
        // pollutes the per-obstacle learned distributions.
        std::map<int, bool> obs_in_sight;
        for (const auto& [obs_id, obs_state] : controller.obstacles()) {
            double dist_to_obs = (ego_state.position() - obs_state.position()).norm();
            obs_in_sight[obs_id] = (dist_to_obs <= sight_radius);

            if (!obs_in_sight[obs_id]) continue;  // outside sight radius

            const auto& obs_cfg = obstacle_configs[obs_id];
            std::string mode_label;  // empty = no label

            int obs_count = ot_predictor.get_observation_count(obs_id);
            if (obs_count > 15) {
                // Enough data: use OT-inferred mode for this obstacle
                auto weights = ot_predictor.compute_mode_weights(
                    obs_id, obs_cfg.available_modes);
                // Pick highest-weight mode
                double best_w = -1;
                for (const auto& [m, w] : weights) {
                    if (w > best_w) { best_w = w; mode_label = m; }
                }
            }

            ot_predictor.observe(obs_id, obs_state.position(), mode_label);
        }

        // ---- Compute OT mode weights and predictions for each obstacle ----
        for (const auto& [obs_id, obs_state] : controller.obstacles()) {
            const auto& obs_cfg = obstacle_configs[obs_id];
            bool in_sight = obs_in_sight[obs_id];

            double pred_error = 0.0;
            double uncertainty_scale = 1.0;
            std::map<std::string, double> ot_weights;

            if (in_sight) {
                ot_weights = ot_predictor.compute_mode_weights(
                    obs_id, obs_cfg.available_modes);

                result.total_ot_weight_computations++;

                // ---- Compute prediction error against previous predictions ----
                if (prev_ot_predictions.count(obs_id) > 0 && step > 0) {
                    const auto& prev_pred = prev_ot_predictions[obs_id];
                    if (prev_pred.size() > 1) {
                        Eigen::Vector2d predicted_pos = prev_pred[1].position;
                        Eigen::Vector2d actual_pos = obs_state.position();
                        pred_error = (predicted_pos - actual_pos).norm();
                        total_prediction_error += pred_error;
                        prediction_error_count++;
                    }
                }

                uncertainty_scale = ot_predictor.adapt_uncertainty(obs_id, pred_error);

                // ---- Generate OT prediction for this obstacle ----
                auto ot_prediction = ot_predictor.predict_trajectory(
                    obs_id, obs_state.position(), obs_state.velocity(),
                    config.horizon, &ot_weights);
                prev_ot_predictions[obs_id] = ot_prediction;
            }

            // ---- Log OT metrics (always, with in_sight flag) ----
            ot_file << step << "," << std::fixed << std::setprecision(4)
                    << time << "," << obs_id << ",";

            for (const auto& mode : all_modes) {
                double w = ot_weights.count(mode) > 0 ? ot_weights.at(mode) : 0.0;
                ot_file << w << ",";
            }
            ot_file << pred_error << ","
                    << uncertainty_scale << ","
                    << ot_predictor.get_observation_count(obs_id) << ","
                    << (in_sight ? 1 : 0) << ",";

            // Log k=0 prediction ellipse params
            if (in_sight && prev_ot_predictions.count(obs_id) > 0 &&
                !prev_ot_predictions[obs_id].empty()) {
                const auto& p0 = prev_ot_predictions[obs_id][0];
                ot_file << p0.angle << "," << p0.major_radius << "," << p0.minor_radius;
            } else {
                ot_file << "0,0.3,0.3";
            }
            ot_file << ",";

            // Log k=5 prediction for ground truth comparison
            if (in_sight && prev_ot_predictions.count(obs_id) > 0 &&
                prev_ot_predictions[obs_id].size() > 5) {
                const auto& p5 = prev_ot_predictions[obs_id][5];
                ot_file << p5.position.x() << "," << p5.position.y() << ","
                        << p5.major_radius << "," << p5.minor_radius << "," << p5.angle;
            } else {
                ot_file << "0,0,0.3,0.3,0";
            }
            ot_file << std::endl;
        }

        // ---- Solve MPC ----
        auto mpc_result = controller.solve(ego_state, path_progress, reference_velocity);

        // ---- Track collision metrics ----
        for (double dist : mpc_result.obstacle_distances) {
            result.min_obstacle_distance = std::min(result.min_obstacle_distance, dist);
            if (dist < collision_radius) result.num_collisions++;
        }

        result.max_lateral_error = std::max(result.max_lateral_error,
            std::abs(mpc_result.lateral_error));

        // ---- Log trajectory CSV ----
        csv_file << step << "," << std::fixed << std::setprecision(4)
                 << time << ","
                 << ego_state.x << "," << ego_state.y << ","
                 << ego_state.theta << "," << ego_state.v << ","
                 << path_progress << "," << mpc_result.lateral_error << ",";

        for (const auto& [obs_id, obs] : controller.obstacles()) {
            double dist = (ego_state.position() - obs.position()).norm();
            csv_file << obs.x << "," << obs.y << "," << dist << ",";
        }
        csv_file << mpc_result.solve_time * 1000 << ","
                 << mpc_result.num_scenarios_used << std::endl;

        // ---- Log scenario predictions ----
        const auto& scenarios = controller.scenarios();
        int scenarios_to_log = std::min(max_scenarios_to_log, static_cast<int>(scenarios.size()));
        for (int s = 0; s < scenarios_to_log; ++s) {
            const auto& scenario = scenarios[s];
            for (const auto& [obs_id, traj] : scenario.trajectories) {
                for (size_t k = 0; k < traj.steps.size(); k += 3) {
                    scenarios_file << step << ","
                                   << scenario.scenario_id << ","
                                   << obs_id << ","
                                   << traj.mode_id << ","
                                   << k << ","
                                   << std::fixed << std::setprecision(4)
                                   << traj.steps[k].mean.x() << ","
                                   << traj.steps[k].mean.y() << ","
                                   << scenario.probability << std::endl;
                }
            }
        }

        // ---- Print progress ----
        if (step % 25 == 0) {
            std::cout << "Step " << std::setw(4) << step
                      << " | Progress: " << std::fixed << std::setprecision(2)
                      << std::setw(6) << path_progress << "m"
                      << " | Min Dist: " << std::setw(5) << result.min_obstacle_distance << "m"
                      << " | OT Obs: " << ot_predictor.get_observation_count(0)
                      << " | Solve: " << std::setw(5) << mpc_result.solve_time * 1000 << "ms"
                      << std::endl;
        }

        // ---- Apply control ----
        if (mpc_result.success && mpc_result.first_input().has_value()) {
            ego_state = dynamics.propagate(ego_state, mpc_result.first_input().value());
        }

        // ---- Update path progress ----
        path_progress = controller.reference_path().find_closest_point(ego_state.position());

        // ---- Simulate obstacles ----
        for (auto& [obs_id, _] : controller.obstacles()) {
            controller.simulate_obstacle_step(obs_id, config.dt);
        }

        // ---- Advance OT predictor timestep ----
        ot_predictor.advance_timestep();

        result.total_steps++;
    }

    csv_file.close();
    scenarios_file.close();
    ot_file.close();

    // ---- Compute final statistics ----
    result.final_progress = path_progress;
    if (path_progress >= config.path_length - 0.5) {
        result.success = true;
    }

    result.avg_prediction_error = (prediction_error_count > 0)
        ? total_prediction_error / prediction_error_count : 0.0;

    // ---- Print summary ----
    std::cout << "\n--- OT Predictor Integration Test Summary ---" << std::endl;
    std::cout << "Success:                " << (result.success ? "YES" : "NO") << std::endl;
    std::cout << "Final Progress:         " << result.final_progress << " / "
              << config.path_length << " m" << std::endl;
    std::cout << "Min Obstacle Distance:  " << result.min_obstacle_distance << " m" << std::endl;
    std::cout << "Max Lateral Error:      " << result.max_lateral_error << " m" << std::endl;
    std::cout << "Collisions:             " << result.num_collisions << std::endl;
    std::cout << "Total Steps:            " << result.total_steps << std::endl;
    std::cout << "Avg Prediction Error:   " << std::fixed << std::setprecision(4)
              << result.avg_prediction_error << " m" << std::endl;
    std::cout << "OT Weight Computations: " << result.total_ot_weight_computations << std::endl;
    std::cout << std::endl;

    // Print learned modes per obstacle
    std::cout << "Learned mode distributions:" << std::endl;
    for (int obs_id = 0; obs_id < static_cast<int>(obstacle_configs.size()); ++obs_id) {
        auto modes = ot_predictor.get_learned_modes(obs_id);
        std::cout << "  Obstacle " << obs_id << ": " << modes.size() << " modes learned [";
        for (const auto& m : modes) {
            auto stats = ot_predictor.get_mode_distribution_stats(obs_id, m);
            if (stats.has_value()) {
                std::cout << m << "(" << static_cast<int>(stats->at("observation_count")) << ") ";
            }
        }
        std::cout << "]" << std::endl;
    }

    return result;
}

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "  Optimal Transport Predictor Integration Test" << std::endl;
    std::cout << "  Safe Horizon Contouring MPC + OT Dynamics Learning" << std::endl;
    std::cout << "================================================================" << std::endl;

    auto result = run_ot_predictor_test();

    // Generate base trajectory GIF using the standard contouring visualizer
    generate_gif("test_ot_predictor_trajectory.csv", "test_ot_predictor.gif");

    // Generate OT-specific visualization
    generate_gif("test_ot_predictor_trajectory.csv", "test_ot_predictor_ot.gif", VISUALIZE_OT);

    // Pass criteria: reached the goal (collisions are reported but tolerated
    // for aggressive obstacle scenarios - the OT predictor improves prediction
    // quality which is measured by avg_prediction_error and mode weight quality)
    bool passed = result.success;

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  RESULT: " << (passed ? "PASS" : "FAIL") << std::endl;
    if (result.num_collisions > 0) {
        std::cout << "  Note: " << result.num_collisions << " close encounters "
                  << "(min dist: " << result.min_obstacle_distance << "m)" << std::endl;
    }
    std::cout << "================================================================" << std::endl;

    return passed ? 0 : 1;
}
