/**
 * @file test_safe_horizon_contouring.cpp
 * @brief Test Safe Horizon Adaptive MPC with Contouring and Multiple Dynamic Obstacles.
 *
 * Replicates the Python test setup from:
 * - test/integration/constraints/safe_horizon/safe_horizon_adaptive_modes_test.py
 * - test/integration/constraints/contouring/contouring_constraints_with_dynamic_obstacles.py
 *
 * Tests:
 * 1. Uniform mode weights
 * 2. Frequency-based mode weights
 * 3. Recency-based mode weights
 * 4. Multiple obstacles with different behavior modes
 * 5. Path following with contouring objective
 * 6. Collision avoidance with scenario-based constraints
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstdlib>

#include "contouring_mpc.hpp"

using namespace scenario_mpc;

// Path to visualization script (relative to build directory)
const std::string VISUALIZE_SCRIPT = "../examples/visualize_contouring.py";

/**
 * @brief Generate GIF visualization from CSV data.
 */
void generate_gif(const std::string& csv_file, const std::string& gif_file) {
    std::string cmd = "python3 " + VISUALIZE_SCRIPT + " " + csv_file + " " + gif_file + " 2>&1";
    std::cout << "\nGenerating visualization: " << gif_file << std::endl;
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "Warning: Visualization script failed (return code " << ret << ")" << std::endl;
        std::cerr << "Make sure matplotlib is installed: pip install matplotlib pillow" << std::endl;
    }
}

// Test configuration
struct TestResult {
    bool success = false;
    double final_progress = 0;
    double min_obstacle_distance = std::numeric_limits<double>::max();
    double max_lateral_error = 0;
    double avg_solve_time = 0;
    int num_collisions = 0;
    int total_steps = 0;
    std::string test_name;
};

/**
 * @brief Run a single test scenario.
 */
TestResult run_test(
    const std::string& test_name,
    WeightType weight_type,
    double duration_seconds = 20.0,  // Longer simulation for longer path
    bool verbose = true
) {
    TestResult result;
    result.test_name = test_name;

    if (verbose) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Running: " << test_name << std::endl;
        std::cout << "========================================" << std::endl;
    }

    // Configure MPC
    ContouringMPCConfig config;
    config.horizon = 15;
    config.dt = 0.1;
    config.num_scenarios = 60;  // Many scenarios for robust coverage
    config.weight_type = weight_type;
    config.recency_decay = 0.1;
    config.ego_radius = 0.5;
    config.obstacle_radius = 0.35;
    config.safety_margin = 1.0;  // Large safety margin
    config.path_length = 30.0;
    config.s_curve_amplitude = 3.5;
    config.road_width = 10.0;
    config.epsilon_p = 0.03;
    config.beta_confidence = 0.01;

    // Create controller
    ContouringMPC controller(config);

    // Create dynamic obstacles with high mode switching
    std::vector<ObstacleConfiguration> obstacle_configs(4);

    // Obstacle 0: Fast path crosser
    obstacle_configs[0].name = "Path Crosser 1";
    obstacle_configs[0].initial_mode = ObstacleBehavior::PATH_INTERSECT;
    obstacle_configs[0].available_modes = {"constant_velocity", "turn_left", "turn_right"};
    obstacle_configs[0].mode_switch_probability = 0.15;
    obstacle_configs[0].radius = 0.4;

    // Obstacle 1: Erratic turner
    obstacle_configs[1].name = "Erratic Turner";
    obstacle_configs[1].initial_mode = ObstacleBehavior::TURN_LEFT;
    obstacle_configs[1].available_modes = {"turn_left", "turn_right", "lane_change_left", "lane_change_right"};
    obstacle_configs[1].mode_switch_probability = 0.2;
    obstacle_configs[1].radius = 0.4;

    // Obstacle 2: Lane weaver
    obstacle_configs[2].name = "Lane Weaver";
    obstacle_configs[2].initial_mode = ObstacleBehavior::LANE_CHANGE_RIGHT;
    obstacle_configs[2].available_modes = {"lane_change_left", "lane_change_right", "constant_velocity"};
    obstacle_configs[2].mode_switch_probability = 0.25;
    obstacle_configs[2].radius = 0.4;

    // Obstacle 3: Another path crosser
    obstacle_configs[3].name = "Path Crosser 2";
    obstacle_configs[3].initial_mode = ObstacleBehavior::PATH_INTERSECT;
    obstacle_configs[3].available_modes = {"constant_velocity", "decelerating", "turn_left"};
    obstacle_configs[3].mode_switch_probability = 0.12;
    obstacle_configs[3].radius = 0.4;

    // Place dynamic obstacles
    std::vector<double> path_fractions = {0.18, 0.38, 0.58, 0.78};
    std::vector<double> lateral_offsets = {4.0, -4.0, 4.5, -4.5};
    std::vector<double> velocities = {0.35, 0.4, 0.3, 0.35};  // Dynamic but safe

    controller.place_obstacles_along_path(
        obstacle_configs, path_fractions, lateral_offsets, velocities
    );

    // Initial ego state
    EgoState ego_state(0.0, 0.0, 0.0, 0.0);
    double path_progress = 0.0;
    double reference_velocity = 2.5;  // Faster reference velocity

    // Collision threshold
    double collision_radius = config.ego_radius + config.obstacle_radius;

    // Simulation
    int num_steps = static_cast<int>(duration_seconds / config.dt);
    double total_solve_time = 0;
    std::vector<double> solve_times;

    // Open CSV for logging
    std::ofstream csv_file(test_name + "_trajectory.csv");
    csv_file << "step,time,ego_x,ego_y,ego_theta,ego_v,progress,lateral_error,";
    for (size_t i = 0; i < obstacle_configs.size(); ++i) {
        csv_file << "obs" << i << "_x,obs" << i << "_y,obs" << i << "_dist,";
    }
    csv_file << "solve_time_ms,num_scenarios" << std::endl;

    // Open scenarios CSV for logging predictions
    std::ofstream scenarios_file(test_name + "_scenarios.csv");
    scenarios_file << "step,scenario_id,obs_id,mode,timestep,pred_x,pred_y,probability" << std::endl;

    EgoDynamics dynamics(config.dt);
    const int max_scenarios_to_log = 5;  // Log up to 5 scenarios per step

    for (int step = 0; step < num_steps; ++step) {
        double time = step * config.dt;

        // Check if goal reached (within 0.1m of path end)
        if (path_progress >= config.path_length - 0.1) {
            if (verbose) {
                std::cout << "Goal reached at step " << step << "! (progress: "
                          << path_progress << "m)" << std::endl;
            }
            result.success = true;
            break;
        }

        // Solve MPC
        auto mpc_result = controller.solve(ego_state, path_progress, reference_velocity);

        total_solve_time += mpc_result.solve_time;
        solve_times.push_back(mpc_result.solve_time * 1000);

        // Check for collisions
        for (size_t i = 0; i < mpc_result.obstacle_distances.size(); ++i) {
            double dist = mpc_result.obstacle_distances[i];
            result.min_obstacle_distance = std::min(result.min_obstacle_distance, dist);
            if (dist < collision_radius) {
                result.num_collisions++;
            }
        }

        // Track lateral error
        result.max_lateral_error = std::max(result.max_lateral_error,
            std::abs(mpc_result.lateral_error));

        // Log to CSV
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

        // Log scenario predictions (subset of scenarios)
        const auto& scenarios = controller.scenarios();
        int scenarios_to_log = std::min(max_scenarios_to_log, static_cast<int>(scenarios.size()));
        for (int s = 0; s < scenarios_to_log; ++s) {
            const auto& scenario = scenarios[s];
            for (const auto& [obs_id, traj] : scenario.trajectories) {
                // Log every 3rd timestep to reduce file size
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

        // Print progress every 20 steps
        if (verbose && step % 20 == 0) {
            std::cout << "Step " << std::setw(4) << step
                      << " | Progress: " << std::fixed << std::setprecision(2)
                      << std::setw(6) << path_progress << "m"
                      << " | Lat Error: " << std::setw(6) << mpc_result.lateral_error << "m"
                      << " | Min Obs Dist: " << std::setw(5) << result.min_obstacle_distance << "m"
                      << " | Solve: " << std::setw(5) << mpc_result.solve_time * 1000 << "ms"
                      << std::endl;
        }

        // Apply control input
        if (mpc_result.success && mpc_result.first_input().has_value()) {
            ego_state = dynamics.propagate(ego_state, mpc_result.first_input().value());
        }

        // Update path progress (monotonic - never regress on S-curves)
        path_progress = controller.reference_path().find_closest_point(ego_state.position(), path_progress);

        // Simulate obstacles
        for (auto& [obs_id, obs] : controller.obstacles()) {
            controller.simulate_obstacle_step(obs_id, config.dt);
        }

        result.total_steps++;
    }

    csv_file.close();
    scenarios_file.close();

    // Compute statistics
    result.final_progress = path_progress;
    result.avg_solve_time = total_solve_time / result.total_steps * 1000;

    // Check success criteria - must reach within 0.5m of end
    if (path_progress >= config.path_length - 0.5) {
        result.success = true;
    }

    // Print summary
    if (verbose) {
        std::cout << "\n--- Test Summary: " << test_name << " ---" << std::endl;
        std::cout << "Success: " << (result.success ? "YES" : "NO") << std::endl;
        std::cout << "Final Progress: " << result.final_progress << " / "
                  << config.path_length << " m" << std::endl;
        std::cout << "Min Obstacle Distance: " << result.min_obstacle_distance << " m" << std::endl;
        std::cout << "Max Lateral Error: " << result.max_lateral_error << " m" << std::endl;
        std::cout << "Collisions: " << result.num_collisions << std::endl;
        std::cout << "Avg Solve Time: " << result.avg_solve_time << " ms" << std::endl;
        std::cout << "Total Steps: " << result.total_steps << std::endl;
    }

    // Generate GIF visualization
    std::string csv_filename = test_name + "_trajectory.csv";
    std::string gif_filename = test_name + ".gif";
    generate_gif(csv_filename, gif_filename);

    return result;
}

/**
 * @brief Test with uniform mode weights.
 */
TestResult test_uniform_weights() {
    return run_test("test_uniform_weights", WeightType::UNIFORM);
}

/**
 * @brief Test with frequency-based mode weights.
 */
TestResult test_frequency_weights() {
    return run_test("test_frequency_weights", WeightType::FREQUENCY);
}

/**
 * @brief Test with recency-based mode weights.
 */
TestResult test_recency_weights() {
    return run_test("test_recency_weights", WeightType::RECENCY);
}

/**
 * @brief Test with aggressive obstacle behavior.
 */
TestResult test_aggressive_obstacles() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running: test_aggressive_obstacles" << std::endl;
    std::cout << "========================================" << std::endl;

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

    ContouringMPC controller(config);

    // Create aggressive obstacles with mode switching
    std::vector<ObstacleConfiguration> obstacle_configs(4);

    obstacle_configs[0].name = "Aggressive Crosser 1";
    obstacle_configs[0].initial_mode = ObstacleBehavior::PATH_INTERSECT;
    obstacle_configs[0].available_modes = {"constant_velocity", "turn_left", "turn_right"};
    obstacle_configs[0].mode_switch_probability = 0.15;
    obstacle_configs[0].radius = 0.35;

    obstacle_configs[1].name = "Aggressive Crosser 2";
    obstacle_configs[1].initial_mode = ObstacleBehavior::PATH_INTERSECT;
    obstacle_configs[1].available_modes = {"constant_velocity", "lane_change_left", "lane_change_right"};
    obstacle_configs[1].mode_switch_probability = 0.18;
    obstacle_configs[1].radius = 0.35;

    obstacle_configs[2].name = "Fast Turner";
    obstacle_configs[2].initial_mode = ObstacleBehavior::TURN_RIGHT;
    obstacle_configs[2].available_modes = {"turn_left", "turn_right", "constant_velocity"};
    obstacle_configs[2].mode_switch_probability = 0.2;
    obstacle_configs[2].radius = 0.35;

    obstacle_configs[3].name = "Lane Weaver";
    obstacle_configs[3].initial_mode = ObstacleBehavior::LANE_CHANGE_LEFT;
    obstacle_configs[3].available_modes = {"lane_change_left", "lane_change_right", "decelerating"};
    obstacle_configs[3].mode_switch_probability = 0.18;
    obstacle_configs[3].radius = 0.35;

    std::vector<double> fractions = {0.20, 0.42, 0.64, 0.86};
    std::vector<double> offsets = {4.0, -4.0, 4.5, -4.5};
    std::vector<double> velocities = {0.35, 0.4, 0.3, 0.35};  // Dynamic but safe

    controller.place_obstacles_along_path(obstacle_configs, fractions, offsets, velocities);

    // Run simulation
    EgoState ego_state(0.0, 0.0, 0.0, 0.0);
    double path_progress = 0.0;
    double reference_velocity = 2.5;
    EgoDynamics dynamics(config.dt);

    TestResult result;
    result.test_name = "test_aggressive_obstacles";

    double collision_radius = config.ego_radius + config.obstacle_radius;
    int num_steps = 180;  // More steps for longer path

    // Open CSV for logging
    std::ofstream csv_file("test_aggressive_obstacles_trajectory.csv");
    csv_file << "step,time,ego_x,ego_y,ego_theta,ego_v,progress,lateral_error,";
    for (size_t i = 0; i < obstacle_configs.size(); ++i) {
        csv_file << "obs" << i << "_x,obs" << i << "_y,obs" << i << "_dist,";
    }
    csv_file << "solve_time_ms,num_scenarios" << std::endl;

    // Open scenarios CSV for logging predictions
    std::ofstream scenarios_file("test_aggressive_obstacles_scenarios.csv");
    scenarios_file << "step,scenario_id,obs_id,mode,timestep,pred_x,pred_y,probability" << std::endl;
    const int max_scenarios_to_log = 5;

    for (int step = 0; step < num_steps; ++step) {
        double time = step * config.dt;

        if (path_progress >= config.path_length - 0.1) {
            std::cout << "Goal reached at step " << step << "! (progress: "
                      << path_progress << "m)" << std::endl;
            result.success = true;
            break;
        }

        auto mpc_result = controller.solve(ego_state, path_progress, reference_velocity);

        for (double dist : mpc_result.obstacle_distances) {
            result.min_obstacle_distance = std::min(result.min_obstacle_distance, dist);
            if (dist < collision_radius) result.num_collisions++;
        }

        result.max_lateral_error = std::max(result.max_lateral_error,
            std::abs(mpc_result.lateral_error));

        // Log to CSV
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

        // Log scenario predictions
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

        if (mpc_result.success && mpc_result.first_input().has_value()) {
            ego_state = dynamics.propagate(ego_state, mpc_result.first_input().value());
        }

        path_progress = controller.reference_path().find_closest_point(ego_state.position(), path_progress);

        for (auto& [obs_id, _] : controller.obstacles()) {
            controller.simulate_obstacle_step(obs_id, config.dt);
        }

        result.total_steps++;

        if (step % 30 == 0) {
            std::cout << "Step " << step << " | Progress: " << path_progress
                      << "m | Min dist: " << result.min_obstacle_distance << "m" << std::endl;
        }
    }

    csv_file.close();
    scenarios_file.close();

    result.final_progress = path_progress;
    if (path_progress >= config.path_length - 0.5) {
        result.success = true;
    }

    std::cout << "\n--- Test Summary ---" << std::endl;
    std::cout << "Success: " << (result.success ? "YES" : "NO") << std::endl;
    std::cout << "Progress: " << result.final_progress << " / " << config.path_length << std::endl;
    std::cout << "Min Distance: " << result.min_obstacle_distance << std::endl;
    std::cout << "Collisions: " << result.num_collisions << std::endl;

    // Generate GIF visualization
    generate_gif("test_aggressive_obstacles_trajectory.csv", "test_aggressive_obstacles.gif");

    return result;
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Safe Horizon Adaptive MPC with Contouring - Test Suite          ║" << std::endl;
    std::cout << "║  Replicating Python test setup with 4 dynamic obstacles          ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;

    std::vector<TestResult> results;

    // Run tests
    results.push_back(test_uniform_weights());
    results.push_back(test_frequency_weights());
    results.push_back(test_recency_weights());
    results.push_back(test_aggressive_obstacles());

    // Summary
    std::cout << "\n\n╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                      TEST RESULTS SUMMARY                        ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;

    int passed = 0, failed = 0;

    std::cout << std::left << std::setw(30) << "Test Name"
              << std::setw(10) << "Status"
              << std::setw(12) << "Progress"
              << std::setw(12) << "Min Dist"
              << std::setw(12) << "Collisions" << std::endl;
    std::cout << std::string(76, '-') << std::endl;

    for (const auto& r : results) {
        std::cout << std::left << std::setw(30) << r.test_name
                  << std::setw(10) << (r.success ? "PASS" : "FAIL")
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.final_progress
                  << std::setw(12) << r.min_obstacle_distance
                  << std::setw(12) << r.num_collisions << std::endl;

        if (r.success && r.num_collisions == 0) {
            passed++;
        } else {
            failed++;
        }
    }

    std::cout << std::string(76, '-') << std::endl;
    std::cout << "Total: " << passed << " passed, " << failed << " failed" << std::endl;

    return failed > 0 ? 1 : 0;
}
