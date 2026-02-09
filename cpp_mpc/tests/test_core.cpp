/**
 * @file test_core.cpp
 * @brief Unit tests for scenario_mpc core functionality.
 *
 * Tests the mathematical formulation matching the Python implementation.
 * Uses a simple test framework for portability.
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>

#include "types.hpp"
#include "config.hpp"
#include "dynamics.hpp"
#include "mode_weights.hpp"
#include "trajectory_moments.hpp"
#include "scenario_sampler.hpp"
#include "collision_constraints.hpp"
#include "scenario_pruning.hpp"
#include "mpc_controller.hpp"

using namespace scenario_mpc;

// Simple test macros
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " #name "..." << std::flush; \
    try { \
        test_##name(); \
        std::cout << " PASSED" << std::endl; \
        passed++; \
    } catch (const std::exception& e) { \
        std::cout << " FAILED: " << e.what() << std::endl; \
        failed++; \
    } \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); \
} while(0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))

#define ASSERT_NEAR(a, b, tol) do { \
    if (std::abs((a) - (b)) > (tol)) { \
        throw std::runtime_error("Assertion failed: abs(" #a " - " #b ") <= " #tol); \
    } \
} while(0)

#define ASSERT_EQ(a, b) ASSERT_TRUE((a) == (b))

// =============================================================================
// Test Types
// =============================================================================

TEST(ego_state_to_array) {
    EgoState state(1.0, 2.0, 0.5, 1.5);
    Eigen::Vector4d arr = state.to_array();

    ASSERT_EQ(arr.size(), 4);
    ASSERT_NEAR(arr(0), 1.0, 1e-9);
    ASSERT_NEAR(arr(1), 2.0, 1e-9);
    ASSERT_NEAR(arr(2), 0.5, 1e-9);
    ASSERT_NEAR(arr(3), 1.5, 1e-9);

    // Test round-trip
    EgoState state2 = EgoState::from_array(arr);
    ASSERT_NEAR(state2.x, state.x, 1e-9);
    ASSERT_NEAR(state2.y, state.y, 1e-9);
    ASSERT_NEAR(state2.theta, state.theta, 1e-9);
    ASSERT_NEAR(state2.v, state.v, 1e-9);
}

TEST(obstacle_state) {
    ObstacleState obs(5.0, 3.0, 1.0, 0.5);

    Eigen::Vector2d pos = obs.position();
    ASSERT_NEAR(pos(0), 5.0, 1e-9);
    ASSERT_NEAR(pos(1), 3.0, 1e-9);

    Eigen::Vector2d vel = obs.velocity();
    ASSERT_NEAR(vel(0), 1.0, 1e-9);
    ASSERT_NEAR(vel(1), 0.5, 1e-9);
}

TEST(mode_model_propagate) {
    double dt = 0.1;
    Eigen::Matrix4d A;
    A << 1, 0, dt, 0,
         0, 1, 0, dt,
         0, 0, 1, 0,
         0, 0, 0, 1;
    Eigen::Vector4d b = Eigen::Vector4d::Zero();
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(4, 4) * 0.1;

    ModeModel mode("cv", A, b, G);
    ObstacleState state(0, 0, 1, 0);

    // Propagate without noise
    ObstacleState next_state = mode.propagate(state);

    ASSERT_NEAR(next_state.x, dt, 1e-9);
    ASSERT_NEAR(next_state.y, 0, 1e-9);
    ASSERT_NEAR(next_state.vx, 1, 1e-9);
}

TEST(collision_constraint_evaluate) {
    CollisionConstraint constraint(
        0, 0, 0,
        Eigen::Vector2d(1.0, 0.0),
        2.0
    );

    // Position satisfies constraint: 1*3 + 0*0 >= 2
    Eigen::Vector2d pos1(3.0, 0.0);
    ASSERT_TRUE(constraint.evaluate(pos1) > 0);

    // Position violates constraint: 1*1 + 0*0 < 2
    Eigen::Vector2d pos2(1.0, 0.0);
    ASSERT_TRUE(constraint.evaluate(pos2) < 0);
}

// =============================================================================
// Test Config
// =============================================================================

TEST(config_defaults) {
    ScenarioMPCConfig config;

    ASSERT_EQ(config.horizon, 20);
    ASSERT_NEAR(config.dt, 0.1, 1e-9);
    ASSERT_NEAR(config.confidence_level, 0.95, 1e-9);
    ASSERT_NEAR(config.epsilon(), 0.05, 1e-9);
}

TEST(config_validation) {
    ScenarioMPCConfig config;
    config.validate();  // Should not throw

    // Invalid horizon
    config.horizon = -1;
    bool threw = false;
    try {
        config.validate();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

TEST(required_scenarios_computation) {
    ScenarioMPCConfig config;
    config.confidence_level = 0.95;
    config.beta = 0.01;

    // For n_x = 10 decision variables
    int S = config.compute_required_scenarios(10);

    // S >= 2/epsilon * (ln(1/beta) + n_x)
    // S >= 2/0.05 * (ln(100) + 10) = 40 * (4.6 + 10) = 584
    ASSERT_TRUE(S >= 500);
}

// =============================================================================
// Test Dynamics
// =============================================================================

TEST(ego_dynamics_propagation) {
    EgoDynamics dynamics(0.1);
    EgoState state(0, 0, 0, 1);
    EgoInput input(0, 0);

    EgoState next_state = dynamics.propagate(state, input);

    // Moving forward at v=1, theta=0, should advance in x
    ASSERT_TRUE(next_state.x > state.x);
    ASSERT_NEAR(next_state.y, 0, 1e-6);
    ASSERT_NEAR(next_state.theta, 0, 1e-6);
    ASSERT_NEAR(next_state.v, 1, 1e-6);
}

TEST(ego_dynamics_turning) {
    EgoDynamics dynamics(0.1);
    EgoState state(0, 0, 0, 1);
    EgoInput input(0, 0.5);  // Turn left

    EgoState next_state = dynamics.propagate(state, input);

    // Should have turned
    ASSERT_TRUE(next_state.theta > 0);
}

TEST(obstacle_modes_creation) {
    auto modes = create_obstacle_mode_models(0.1);

    ASSERT_TRUE(modes.find("constant_velocity") != modes.end());
    ASSERT_TRUE(modes.find("turn_left") != modes.end());
    ASSERT_TRUE(modes.find("turn_right") != modes.end());

    // Check dimensions
    const auto& cv_mode = modes["constant_velocity"];
    ASSERT_EQ(cv_mode.A.rows(), 4);
    ASSERT_EQ(cv_mode.A.cols(), 4);
    ASSERT_EQ(cv_mode.b.size(), 4);
}

// =============================================================================
// Test Mode Weights
// =============================================================================

TEST(uniform_weights) {
    auto modes = create_obstacle_mode_models();
    ModeHistory history(0, modes);
    history.record_observation(0, "constant_velocity");
    history.record_observation(1, "turn_left");

    auto weights = compute_mode_weights(history, WeightType::UNIFORM);

    // All modes should have equal weight
    ASSERT_EQ(weights.size(), modes.size());
    double expected_weight = 1.0 / modes.size();
    for (const auto& [_, w] : weights) {
        ASSERT_NEAR(w, expected_weight, 1e-6);
    }
}

TEST(frequency_weights) {
    auto modes = create_obstacle_mode_models();
    ModeHistory history(0, modes);
    history.record_observation(0, "constant_velocity");
    history.record_observation(1, "constant_velocity");
    history.record_observation(2, "constant_velocity");
    history.record_observation(3, "turn_left");

    auto weights = compute_mode_weights(history, WeightType::FREQUENCY);

    // constant_velocity should have higher weight (3/4)
    ASSERT_TRUE(weights["constant_velocity"] > weights["turn_left"]);
    ASSERT_NEAR(weights["constant_velocity"], 0.75, 1e-6);
    ASSERT_NEAR(weights["turn_left"], 0.25, 1e-6);
}

TEST(recency_weights) {
    auto modes = create_obstacle_mode_models();
    ModeHistory history(0, modes);
    history.record_observation(0, "constant_velocity");  // Old
    history.record_observation(8, "turn_left");          // Recent
    history.record_observation(9, "turn_left");          // Most recent

    auto weights = compute_mode_weights(
        history, WeightType::RECENCY, 0.9, 10
    );

    // Recent observations should have more weight
    ASSERT_TRUE(weights["turn_left"] > weights["constant_velocity"]);
}

// =============================================================================
// Test Trajectory Moments
// =============================================================================

TEST(single_mode_trajectory) {
    auto modes = create_obstacle_mode_models();
    const auto& mode = modes["constant_velocity"];
    ObstacleState state(0, 0, 1, 0);

    auto traj = compute_single_mode_trajectory(state, mode, 10);

    ASSERT_EQ(traj.horizon(), 10);
    ASSERT_EQ(traj.steps.size(), 11);  // Including initial

    // Position should advance
    ASSERT_TRUE(traj.steps[10].mean(0) > traj.steps[0].mean(0));
}

TEST(trajectory_moments_computation) {
    auto modes = create_obstacle_mode_models();
    ObstacleState state(0, 0, 1, 0);
    std::map<std::string, double> weights = {
        {"constant_velocity", 0.5},
        {"turn_left", 0.5}
    };

    std::map<std::string, ModeModel> selected_modes;
    for (const auto& [name, _] : weights) {
        selected_modes[name] = modes[name];
    }

    auto moments = compute_trajectory_moments(state, weights, selected_modes, 10);

    ASSERT_EQ(moments.horizon(), 10);
    ASSERT_EQ(moments.means.rows(), 11);
    ASSERT_EQ(moments.covariances.size(), 11);

    // Covariance should grow over time (uncertainty increases)
    double cov_0 = moments.covariances[0].trace();
    double cov_10 = moments.covariances[10].trace();
    ASSERT_TRUE(cov_10 > cov_0);
}

// =============================================================================
// Test Scenario Sampling
// =============================================================================

TEST(sample_scenarios) {
    auto modes = create_obstacle_mode_models();
    std::map<int, ObstacleState> obstacles = {
        {0, ObstacleState(5, 0, -1, 0)},
        {1, ObstacleState(0, 5, 0, -1)},
    };
    std::map<int, ModeHistory> histories = {
        {0, ModeHistory(0, modes)},
        {1, ModeHistory(1, modes)},
    };
    histories[0].record_observation(0, "constant_velocity");
    histories[1].record_observation(0, "turn_left");

    auto scenarios = sample_scenarios(
        obstacles, histories, 10, 5
    );

    ASSERT_EQ(scenarios.size(), 5);
    for (const auto& scenario : scenarios) {
        ASSERT_EQ(scenario.trajectories.size(), 2);
        ASSERT_TRUE(scenario.trajectories.find(0) != scenario.trajectories.end());
        ASSERT_TRUE(scenario.trajectories.find(1) != scenario.trajectories.end());
    }
}

// =============================================================================
// Test Collision Constraints
// =============================================================================

TEST(ego_disc_positions) {
    EgoState state(0, 0, 0, 1);

    // Single disc
    auto positions = compute_ego_disc_positions(state, 1);
    ASSERT_EQ(positions.size(), 1);
    ASSERT_NEAR(positions[0](0), 0, 1e-9);
    ASSERT_NEAR(positions[0](1), 0, 1e-9);

    // Multiple discs
    positions = compute_ego_disc_positions(state, 3, 4.0);
    ASSERT_EQ(positions.size(), 3);
    // Should be along x-axis (theta=0)
    ASSERT_TRUE(positions[0](0) < positions[1](0));
    ASSERT_TRUE(positions[1](0) < positions[2](0));
}

TEST(linearized_constraints) {
    // Simple scenario with one obstacle
    std::vector<EgoState> ego_traj = {
        EgoState(0, 0, 0, 1),
        EgoState(1, 0, 0, 1),
    };

    std::vector<PredictionStep> obs_steps = {
        PredictionStep(0, Eigen::Vector2d(2, 0), Eigen::Matrix2d::Identity()),
        PredictionStep(1, Eigen::Vector2d(1.5, 0), Eigen::Matrix2d::Identity()),
    };
    ObstacleTrajectory obs_traj(0, "cv", obs_steps);
    Scenario scenario(0, {{0, obs_traj}});

    auto constraints = compute_linearized_constraints(
        ego_traj, {scenario}, 0.5, 0.5
    );

    ASSERT_TRUE(constraints.size() > 0);

    // Constraints should be roughly pointing from obstacle to ego
    for (const auto& c : constraints) {
        ASSERT_TRUE(c.a.norm() > 0);
        ASSERT_TRUE(std::isfinite(c.b));
    }
}

// =============================================================================
// Test Scenario Pruning
// =============================================================================

TEST(prune_dominated) {
    // Create two scenarios - one dominates the other
    std::vector<EgoState> ego_traj = {EgoState(0, 0, 0, 1)};

    // Scenario 1: obstacle close
    ObstacleTrajectory obs1(0, "cv",
        {PredictionStep(0, Eigen::Vector2d(2, 0), Eigen::Matrix2d::Identity())}
    );
    Scenario scenario1(0, {{0, obs1}});

    // Scenario 2: obstacle far (dominated)
    ObstacleTrajectory obs2(0, "cv",
        {PredictionStep(0, Eigen::Vector2d(5, 0), Eigen::Matrix2d::Identity())}
    );
    Scenario scenario2(1, {{0, obs2}});

    auto pruned = prune_dominated_scenarios(
        {scenario1, scenario2}, ego_traj, 0.5, 0.5
    );

    // Scenario 2 should be pruned (obstacle is farther)
    ASSERT_EQ(pruned.size(), 1);
    ASSERT_EQ(pruned[0].scenario_id, 0);
}

// =============================================================================
// Test Integration
// =============================================================================

TEST(full_pipeline) {
    ScenarioMPCConfig config;
    config.horizon = 10;
    config.num_scenarios = 5;
    config.dt = 0.1;

    AdaptiveScenarioMPC controller(config);

    // Setup
    EgoState ego_state(0, 0, 0, 0);
    std::map<int, ObstacleState> obstacles = {
        {0, ObstacleState(5, 0, -1, 0)}
    };
    Eigen::Vector2d goal(10, 0);

    // Initialize obstacle
    controller.initialize_obstacle(0);
    controller.update_mode_observation(0, "constant_velocity");

    // Solve
    MPCResult result = controller.solve(ego_state, obstacles, goal);

    ASSERT_TRUE(result.ego_trajectory.size() > 0);
    ASSERT_EQ(result.ego_trajectory.size(), config.horizon + 1);
    ASSERT_EQ(result.control_inputs.size(), config.horizon);

    // First control should be valid
    auto first_input = result.first_input();
    ASSERT_TRUE(first_input.has_value());
    ASSERT_TRUE(first_input->a >= config.min_acceleration);
    ASSERT_TRUE(first_input->a <= config.max_acceleration);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    int passed = 0;
    int failed = 0;

    std::cout << "Running scenario_mpc tests...\n" << std::endl;

    // Types tests
    RUN_TEST(ego_state_to_array);
    RUN_TEST(obstacle_state);
    RUN_TEST(mode_model_propagate);
    RUN_TEST(collision_constraint_evaluate);

    // Config tests
    RUN_TEST(config_defaults);
    RUN_TEST(config_validation);
    RUN_TEST(required_scenarios_computation);

    // Dynamics tests
    RUN_TEST(ego_dynamics_propagation);
    RUN_TEST(ego_dynamics_turning);
    RUN_TEST(obstacle_modes_creation);

    // Mode weights tests
    RUN_TEST(uniform_weights);
    RUN_TEST(frequency_weights);
    RUN_TEST(recency_weights);

    // Trajectory moments tests
    RUN_TEST(single_mode_trajectory);
    RUN_TEST(trajectory_moments_computation);

    // Scenario sampling tests
    RUN_TEST(sample_scenarios);

    // Collision constraints tests
    RUN_TEST(ego_disc_positions);
    RUN_TEST(linearized_constraints);

    // Scenario pruning tests
    RUN_TEST(prune_dominated);

    // Integration tests
    RUN_TEST(full_pipeline);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;

    return failed > 0 ? 1 : 0;
}
