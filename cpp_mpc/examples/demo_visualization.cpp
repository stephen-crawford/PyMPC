/**
 * @file demo_visualization.cpp
 * @brief Demo that outputs trajectory data for visualization.
 *
 * Simulates an ego vehicle avoiding a dynamic obstacle while
 * navigating to a goal. Outputs CSV data for plotting.
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

#include "mpc_controller.hpp"

using namespace scenario_mpc;

int main() {
    std::cout << "=== Adaptive Scenario-Based MPC Demo ===" << std::endl;
    std::cout << "Simulating ego vehicle avoiding dynamic obstacle..." << std::endl;
    std::cout << std::endl;

    // Configure the controller
    ScenarioMPCConfig config;
    config.horizon = 15;
    config.dt = 0.1;
    config.num_scenarios = 8;
    config.confidence_level = 0.95;
    config.ego_radius = 0.5;
    config.obstacle_radius = 0.5;
    config.safety_margin = 0.2;
    config.goal_weight = 5.0;
    config.velocity_weight = 1.0;

    AdaptiveScenarioMPC controller(config);

    // Initial setup
    EgoState ego_state(0.0, 0.0, 0.0, 0.0);
    Eigen::Vector2d goal(12.0, 0.0);

    // Obstacle starts ahead and moves toward ego
    ObstacleState obstacle(8.0, 0.5, -1.5, 0.0);
    std::map<int, ObstacleState> obstacles = {{0, obstacle}};

    // Initialize obstacle tracking
    controller.initialize_obstacle(0);
    controller.update_mode_observation(0, "constant_velocity");

    // Open output file
    std::ofstream data_file("trajectory_data.csv");
    data_file << "time,ego_x,ego_y,ego_theta,ego_v,obs_x,obs_y,goal_x,goal_y,accel,steer" << std::endl;

    // Simulation loop
    int num_steps = 80;
    double time = 0.0;
    EgoDynamics dynamics(config.dt);

    std::cout << "Step  | Ego Position    | Velocity | Obstacle Pos   | Action" << std::endl;
    std::cout << "------|-----------------|----------|----------------|----------------" << std::endl;

    for (int step = 0; step < num_steps; ++step) {
        // Check if goal reached
        double dist_to_goal = (ego_state.position() - goal).norm();
        if (dist_to_goal < 0.5) {
            std::cout << "Goal reached at step " << step << "!" << std::endl;
            break;
        }

        // Solve MPC
        MPCResult result = controller.solve(ego_state, obstacles, goal, 2.0);

        // Get control input
        double accel = 0.0, steer = 0.0;
        if (result.success && result.first_input().has_value()) {
            accel = result.first_input()->a;
            steer = result.first_input()->delta;
        }

        // Log to file
        data_file << std::fixed << std::setprecision(4)
                  << time << ","
                  << ego_state.x << "," << ego_state.y << ","
                  << ego_state.theta << "," << ego_state.v << ","
                  << obstacles[0].x << "," << obstacles[0].y << ","
                  << goal(0) << "," << goal(1) << ","
                  << accel << "," << steer << std::endl;

        // Print status every 5 steps
        if (step % 5 == 0) {
            std::cout << std::setw(5) << step << " | "
                      << "(" << std::setw(5) << std::fixed << std::setprecision(2) << ego_state.x
                      << ", " << std::setw(5) << ego_state.y << ") | "
                      << std::setw(8) << std::setprecision(2) << ego_state.v << " | "
                      << "(" << std::setw(5) << obstacles[0].x
                      << ", " << std::setw(5) << obstacles[0].y << ") | "
                      << "a=" << std::setw(5) << std::setprecision(2) << accel
                      << " w=" << std::setw(5) << steer << std::endl;
        }

        // Apply control and update state
        if (result.success && result.first_input().has_value()) {
            ego_state = dynamics.propagate(ego_state, result.first_input().value());
        }

        // Update obstacle (constant velocity)
        obstacles[0].x += obstacles[0].vx * config.dt;
        obstacles[0].y += obstacles[0].vy * config.dt;

        // Update mode observation
        controller.update_mode_observation(0, "constant_velocity");

        time += config.dt;
    }

    data_file.close();

    // Print summary
    std::cout << std::endl;
    std::cout << "=== Simulation Complete ===" << std::endl;
    std::cout << "Final ego position: (" << ego_state.x << ", " << ego_state.y << ")" << std::endl;
    std::cout << "Final velocity: " << ego_state.v << " m/s" << std::endl;
    std::cout << "Data saved to: trajectory_data.csv" << std::endl;

    auto stats = controller.get_statistics();
    std::cout << std::endl;
    std::cout << "Controller Statistics:" << std::endl;
    std::cout << "  Total iterations: " << stats.iteration_count << std::endl;
    std::cout << "  Avg solve time: " << std::fixed << std::setprecision(2)
              << stats.avg_solve_time * 1000 << " ms" << std::endl;
    std::cout << "  Max solve time: " << stats.max_solve_time * 1000 << " ms" << std::endl;

    return 0;
}
