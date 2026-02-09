/**
 * @file example_mpc.cpp
 * @brief Example demonstrating the Adaptive Scenario-Based MPC.
 *
 * This example shows how to:
 * 1. Configure the MPC controller
 * 2. Set up obstacles with mode histories
 * 3. Solve the MPC problem
 * 4. Use the results
 */

#include <iostream>
#include <iomanip>

#include "mpc_controller.hpp"

using namespace scenario_mpc;

int main() {
    std::cout << "=== Adaptive Scenario-Based MPC Example ===" << std::endl;
    std::cout << std::endl;

    // Step 1: Configure the MPC controller
    ScenarioMPCConfig config;
    config.horizon = 20;          // 20 timesteps prediction horizon
    config.dt = 0.1;              // 100ms timestep
    config.num_scenarios = 10;    // Sample 10 scenarios
    config.confidence_level = 0.95;  // 95% chance constraint satisfaction
    config.ego_radius = 1.0;      // 1m ego vehicle radius
    config.obstacle_radius = 0.5; // 0.5m obstacle radius
    config.safety_margin = 0.1;   // 10cm additional safety margin

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Horizon: " << config.horizon << " steps" << std::endl;
    std::cout << "  Timestep: " << config.dt << " s" << std::endl;
    std::cout << "  Num scenarios: " << config.num_scenarios << std::endl;
    std::cout << "  Confidence: " << config.confidence_level * 100 << "%" << std::endl;
    std::cout << std::endl;

    // Step 2: Create the controller
    AdaptiveScenarioMPC controller(config);

    // Step 3: Set up the scenario
    // Ego vehicle starts at origin, heading in +x direction
    EgoState ego_state(0.0, 0.0, 0.0, 0.0);  // x, y, theta, v

    // One obstacle approaching from ahead
    std::map<int, ObstacleState> obstacles = {
        {0, ObstacleState(8.0, 0.0, -1.0, 0.0)},  // x, y, vx, vy
    };

    // Goal is ahead of the ego vehicle
    Eigen::Vector2d goal(15.0, 0.0);

    std::cout << "Scenario:" << std::endl;
    std::cout << "  Ego start: (" << ego_state.x << ", " << ego_state.y << ")" << std::endl;
    std::cout << "  Ego heading: " << ego_state.theta << " rad" << std::endl;
    std::cout << "  Goal: (" << goal(0) << ", " << goal(1) << ")" << std::endl;
    std::cout << "  Obstacle at: (" << obstacles[0].x << ", " << obstacles[0].y << ")" << std::endl;
    std::cout << "  Obstacle velocity: (" << obstacles[0].vx << ", " << obstacles[0].vy << ")" << std::endl;
    std::cout << std::endl;

    // Step 4: Initialize obstacle mode history
    // The controller learns obstacle behavior from observations
    controller.initialize_obstacle(0);
    controller.update_mode_observation(0, "constant_velocity");

    // Step 5: Solve the MPC problem
    std::cout << "Solving MPC..." << std::endl;
    double reference_velocity = 2.0;  // Desired velocity
    MPCResult result = controller.solve(ego_state, obstacles, goal, reference_velocity);

    // Step 6: Display results
    std::cout << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "  Success: " << (result.success ? "Yes" : "No") << std::endl;
    std::cout << "  Solve time: " << std::fixed << std::setprecision(3)
              << result.solve_time * 1000 << " ms" << std::endl;
    std::cout << "  Cost: " << result.cost << std::endl;
    std::cout << std::endl;

    if (result.success && result.first_input().has_value()) {
        EgoInput first_input = result.first_input().value();
        std::cout << "First control input:" << std::endl;
        std::cout << "  Acceleration: " << first_input.a << " m/s^2" << std::endl;
        std::cout << "  Angular velocity: " << first_input.delta << " rad/s" << std::endl;
        std::cout << std::endl;
    }

    // Step 7: Display planned trajectory
    std::cout << "Planned trajectory (first 5 steps):" << std::endl;
    std::cout << std::setw(6) << "Step"
              << std::setw(10) << "x"
              << std::setw(10) << "y"
              << std::setw(10) << "theta"
              << std::setw(10) << "v" << std::endl;
    std::cout << std::string(46, '-') << std::endl;

    int num_display = std::min(5, static_cast<int>(result.ego_trajectory.size()));
    for (int i = 0; i < num_display; ++i) {
        const auto& state = result.ego_trajectory[i];
        std::cout << std::setw(6) << i
                  << std::setw(10) << std::fixed << std::setprecision(3) << state.x
                  << std::setw(10) << state.y
                  << std::setw(10) << state.theta
                  << std::setw(10) << state.v << std::endl;
    }
    std::cout << std::endl;

    // Step 8: Display statistics
    auto stats = controller.get_statistics();
    std::cout << "Controller statistics:" << std::endl;
    std::cout << "  Iterations: " << stats.iteration_count << std::endl;
    std::cout << "  Avg solve time: " << stats.avg_solve_time * 1000 << " ms" << std::endl;
    std::cout << "  Num scenarios: " << stats.num_scenarios << std::endl;
    std::cout << std::endl;

    // Step 9: Simulate a few more steps
    std::cout << "Simulating 5 more MPC iterations..." << std::endl;
    EgoState current_state = ego_state;

    for (int iter = 0; iter < 5; ++iter) {
        // Apply first control input
        if (result.success && result.first_input().has_value()) {
            EgoDynamics dynamics(config.dt);
            current_state = dynamics.propagate(current_state, result.first_input().value());
        }

        // Update obstacle position (simple constant velocity)
        obstacles[0].x += obstacles[0].vx * config.dt;
        obstacles[0].y += obstacles[0].vy * config.dt;

        // Update mode observation
        controller.update_mode_observation(0, "constant_velocity");

        // Solve again
        result = controller.solve(current_state, obstacles, goal, reference_velocity);

        std::cout << "  Iter " << iter + 1 << ": ego=("
                  << std::fixed << std::setprecision(2)
                  << current_state.x << "," << current_state.y
                  << ") v=" << current_state.v
                  << " solve_time=" << result.solve_time * 1000 << "ms" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Example complete!" << std::endl;

    return 0;
}
