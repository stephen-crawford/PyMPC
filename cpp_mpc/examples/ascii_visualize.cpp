/**
 * @file ascii_visualize.cpp
 * @brief ASCII art visualization of MPC trajectory.
 *
 * Creates a simple terminal animation showing the ego vehicle
 * avoiding obstacles and reaching the goal.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <thread>
#include <chrono>
#include <vector>
#include <string>

#include "mpc_controller.hpp"

using namespace scenario_mpc;

// Terminal colors
const std::string RESET = "\033[0m";
const std::string RED = "\033[31m";
const std::string GREEN = "\033[32m";
const std::string BLUE = "\033[34m";
const std::string YELLOW = "\033[33m";
const std::string CYAN = "\033[36m";

void clear_screen() {
    std::cout << "\033[2J\033[H";
}

void draw_frame(const std::vector<std::pair<double, double>>& ego_history,
                double ego_x, double ego_y, double ego_theta,
                double obs_x, double obs_y,
                double goal_x, double goal_y,
                int step, double velocity, double solve_time) {

    const int WIDTH = 70;
    const int HEIGHT = 20;
    const double X_MIN = -2;
    const double X_MAX = 14;
    const double Y_MIN = -3;
    const double Y_MAX = 3;

    auto to_screen_x = [&](double x) {
        return static_cast<int>((x - X_MIN) / (X_MAX - X_MIN) * (WIDTH - 1));
    };
    auto to_screen_y = [&](double y) {
        return HEIGHT - 1 - static_cast<int>((y - Y_MIN) / (Y_MAX - Y_MIN) * (HEIGHT - 1));
    };

    // Create screen buffer
    std::vector<std::string> screen(HEIGHT, std::string(WIDTH, ' '));

    // Draw border
    for (int i = 0; i < WIDTH; ++i) {
        screen[0][i] = '-';
        screen[HEIGHT-1][i] = '-';
    }
    for (int i = 0; i < HEIGHT; ++i) {
        screen[i][0] = '|';
        screen[i][WIDTH-1] = '|';
    }

    // Draw trajectory history
    for (const auto& [hx, hy] : ego_history) {
        int sx = to_screen_x(hx);
        int sy = to_screen_y(hy);
        if (sx > 0 && sx < WIDTH-1 && sy > 0 && sy < HEIGHT-1) {
            screen[sy][sx] = '.';
        }
    }

    // Draw goal
    int gx = to_screen_x(goal_x);
    int gy = to_screen_y(goal_y);
    if (gx > 0 && gx < WIDTH-1 && gy > 0 && gy < HEIGHT-1) {
        screen[gy][gx] = 'G';
    }

    // Draw obstacle
    int ox = to_screen_x(obs_x);
    int oy = to_screen_y(obs_y);
    if (ox > 0 && ox < WIDTH-1 && oy > 0 && oy < HEIGHT-1) {
        screen[oy][ox] = 'O';
    }

    // Draw ego vehicle (with direction indicator)
    int ex = to_screen_x(ego_x);
    int ey = to_screen_y(ego_y);
    if (ex > 0 && ex < WIDTH-1 && ey > 0 && ey < HEIGHT-1) {
        char ego_char = '>';
        if (ego_theta > M_PI/4 && ego_theta < 3*M_PI/4) ego_char = '^';
        else if (ego_theta > 3*M_PI/4 || ego_theta < -3*M_PI/4) ego_char = '<';
        else if (ego_theta < -M_PI/4 && ego_theta > -3*M_PI/4) ego_char = 'v';
        screen[ey][ex] = ego_char;
    }

    // Print header
    std::cout << CYAN << "╔══════════════════════════════════════════════════════════════════════╗" << RESET << std::endl;
    std::cout << CYAN << "║" << RESET << "        " << YELLOW << "Adaptive Scenario-Based MPC - Live Demo" << RESET << "                   " << CYAN << "║" << RESET << std::endl;
    std::cout << CYAN << "╚══════════════════════════════════════════════════════════════════════╝" << RESET << std::endl;

    // Print screen with colors
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            char c = screen[y][x];
            if (c == 'G') std::cout << GREEN << c << RESET;
            else if (c == 'O') std::cout << RED << c << RESET;
            else if (c == '>' || c == '<' || c == '^' || c == 'v') std::cout << BLUE << c << RESET;
            else if (c == '.') std::cout << CYAN << c << RESET;
            else std::cout << c;
        }
        std::cout << std::endl;
    }

    // Print status
    std::cout << std::endl;
    std::cout << "Legend: " << BLUE << "> " << RESET << "Ego  "
              << RED << "O " << RESET << "Obstacle  "
              << GREEN << "G " << RESET << "Goal  "
              << CYAN << ". " << RESET << "Trail" << std::endl;
    std::cout << std::endl;
    std::cout << "Step: " << std::setw(3) << step
              << "  |  Ego: (" << std::fixed << std::setprecision(2) << std::setw(6) << ego_x
              << ", " << std::setw(6) << ego_y << ")"
              << "  |  Velocity: " << std::setw(5) << velocity << " m/s"
              << "  |  Solve: " << std::setw(5) << std::setprecision(2) << solve_time * 1000 << " ms" << std::endl;

    double dist_to_goal = std::sqrt(std::pow(ego_x - goal_x, 2) + std::pow(ego_y - goal_y, 2));
    double dist_to_obs = std::sqrt(std::pow(ego_x - obs_x, 2) + std::pow(ego_y - obs_y, 2));
    std::cout << "Distance to goal: " << std::setw(5) << dist_to_goal << " m"
              << "  |  Distance to obstacle: " << std::setw(5) << dist_to_obs << " m" << std::endl;
}

int main() {
    // Configure the controller
    ScenarioMPCConfig config;
    config.horizon = 15;
    config.dt = 0.1;
    config.num_scenarios = 8;
    config.ego_radius = 0.5;
    config.obstacle_radius = 0.5;
    config.safety_margin = 0.2;

    AdaptiveScenarioMPC controller(config);

    // Initial setup
    EgoState ego_state(0.0, 0.0, 0.0, 0.0);
    Eigen::Vector2d goal(12.0, 0.0);
    ObstacleState obstacle(8.0, 0.5, -1.5, 0.0);
    std::map<int, ObstacleState> obstacles = {{0, obstacle}};

    controller.initialize_obstacle(0);
    controller.update_mode_observation(0, "constant_velocity");

    EgoDynamics dynamics(config.dt);
    std::vector<std::pair<double, double>> ego_history;

    int num_steps = 80;

    for (int step = 0; step < num_steps; ++step) {
        // Check if goal reached
        double dist_to_goal = (ego_state.position() - goal).norm();
        if (dist_to_goal < 0.5) {
            clear_screen();
            draw_frame(ego_history, ego_state.x, ego_state.y, ego_state.theta,
                      obstacles[0].x, obstacles[0].y, goal(0), goal(1),
                      step, ego_state.v, 0);
            std::cout << std::endl;
            std::cout << GREEN << "*** GOAL REACHED! ***" << RESET << std::endl;
            break;
        }

        // Solve MPC
        MPCResult result = controller.solve(ego_state, obstacles, goal, 2.0);

        // Draw frame
        clear_screen();
        draw_frame(ego_history, ego_state.x, ego_state.y, ego_state.theta,
                  obstacles[0].x, obstacles[0].y, goal(0), goal(1),
                  step, ego_state.v, result.solve_time);

        // Store history
        ego_history.emplace_back(ego_state.x, ego_state.y);

        // Apply control
        if (result.success && result.first_input().has_value()) {
            ego_state = dynamics.propagate(ego_state, result.first_input().value());
        }

        // Update obstacle
        obstacles[0].x += obstacles[0].vx * config.dt;
        obstacles[0].y += obstacles[0].vy * config.dt;
        controller.update_mode_observation(0, "constant_velocity");

        // Sleep for animation
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Final statistics
    auto stats = controller.get_statistics();
    std::cout << std::endl;
    std::cout << "=== Final Statistics ===" << std::endl;
    std::cout << "Total iterations: " << stats.iteration_count << std::endl;
    std::cout << "Average solve time: " << std::fixed << std::setprecision(2)
              << stats.avg_solve_time * 1000 << " ms" << std::endl;
    std::cout << "Max solve time: " << stats.max_solve_time * 1000 << " ms" << std::endl;

    return 0;
}
