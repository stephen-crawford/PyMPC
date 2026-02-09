/**
 * @file test_epsilon_guarantee.cpp
 * @brief Tests for epsilon guarantee enforcement and OT integration.
 *
 * Tests Parts 1-4:
 * - compute_effective_epsilon / compute_required_scenarios
 * - enforce_scenario_count in MPC solve
 * - WASSERSTEIN weight type
 * - estimate_mode_dynamics
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <numeric>

#include "types.hpp"
#include "config.hpp"
#include "mpc_controller.hpp"
#include "optimal_transport_predictor.hpp"

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

#define ASSERT_NEAR(a, b, tol) do { \
    if (std::abs((a) - (b)) > (tol)) { \
        throw std::runtime_error("Assertion failed: abs(" #a " - " #b ") <= " #tol); \
    } \
} while(0)

// =============================================================================
// Part 4: compute_effective_epsilon tests
// =============================================================================

TEST(effective_epsilon_basic) {
    ScenarioMPCConfig config;
    config.confidence_level = 0.95;  // epsilon = 0.05
    config.beta = 0.01;

    int d = 60;  // horizon=10, n_x=4, n_u=2
    double eps = config.compute_effective_epsilon(1000, d);
    // eps = 2*(ln(100) + 60) / 1000 = 2*(4.605+60)/1000 ~= 0.129
    ASSERT_TRUE(eps > 0.1 && eps < 0.2);
}

TEST(effective_epsilon_zero_scenarios) {
    ScenarioMPCConfig config;
    config.beta = 0.01;
    ASSERT_NEAR(config.compute_effective_epsilon(0, 60), 1.0, 1e-10);
}

TEST(effective_epsilon_more_scenarios_lower) {
    ScenarioMPCConfig config;
    config.beta = 0.01;
    int d = 60;
    double eps100 = config.compute_effective_epsilon(100, d);
    double eps1000 = config.compute_effective_epsilon(1000, d);
    double eps10000 = config.compute_effective_epsilon(10000, d);
    ASSERT_TRUE(eps100 > eps1000);
    ASSERT_TRUE(eps1000 > eps10000);
}

TEST(required_scenarios_with_removal) {
    ScenarioMPCConfig config;
    config.confidence_level = 0.9;  // epsilon = 0.1
    config.beta = 0.01;
    int d = 60;
    int S_no_removal = config.compute_required_scenarios(d, 0);
    int S_with_removal = config.compute_required_scenarios(d, 10);
    ASSERT_TRUE(S_with_removal > S_no_removal);
}

TEST(effective_epsilon_roundtrip) {
    ScenarioMPCConfig config;
    config.confidence_level = 0.9;  // epsilon = 0.1
    config.beta = 0.01;
    int d = 60;
    int S = config.compute_required_scenarios(d);
    double eps_back = config.compute_effective_epsilon(S, d);
    // Should be close to epsilon()
    ASSERT_NEAR(eps_back, config.epsilon(), 0.02);
}

// =============================================================================
// Part 1: Config flags
// =============================================================================

TEST(config_enforce_flags_default_false) {
    ScenarioMPCConfig config;
    ASSERT_TRUE(!config.enforce_all_scenarios);
    ASSERT_TRUE(!config.enforce_scenario_count);
    ASSERT_TRUE(!config.enable_dynamics_learning);
}

TEST(config_dynamics_learning_interval) {
    ScenarioMPCConfig config;
    ASSERT_TRUE(config.dynamics_learning_interval == 10);
    config.dynamics_learning_interval = 5;
    ASSERT_TRUE(config.dynamics_learning_interval == 5);
}

// =============================================================================
// Part 2: WASSERSTEIN weight type
// =============================================================================

TEST(wasserstein_weight_type_exists) {
    WeightType wt = WeightType::WASSERSTEIN;
    ASSERT_TRUE(wt == WeightType::WASSERSTEIN);
    ASSERT_TRUE(wt != WeightType::FREQUENCY);
}

TEST(ot_predictor_compute_mode_weights_uniform_fallback) {
    // With no observations, should return uniform weights
    auto predictor = create_ot_predictor_with_standard_modes(0.1, 1.0);
    std::vector<std::string> modes = {"constant_velocity", "decelerating"};
    auto weights = predictor.compute_mode_weights(0, modes);
    // Should be uniform since no observations
    ASSERT_NEAR(weights["constant_velocity"], 0.5, 1e-6);
    ASSERT_NEAR(weights["decelerating"], 0.5, 1e-6);
}

TEST(ot_predictor_mode_weights_with_observations) {
    auto predictor = create_ot_predictor_with_standard_modes(0.1, 1.0);

    // Feed observations simulating constant velocity motion
    for (int t = 0; t < 20; ++t) {
        Eigen::Vector2d pos(t * 0.1 * 1.0, 0.0);
        predictor.observe(0, pos, "constant_velocity");
        predictor.advance_timestep();
    }

    std::vector<std::string> modes = {"constant_velocity", "decelerating", "accelerating"};
    auto weights = predictor.compute_mode_weights(0, modes);

    // Weights should sum to 1
    double total = 0;
    for (auto& [k, v] : weights) total += v;
    ASSERT_NEAR(total, 1.0, 1e-6);
}

// =============================================================================
// Part 3: estimate_mode_dynamics
// =============================================================================

TEST(estimate_dynamics_insufficient_data) {
    OptimalTransportPredictor predictor(0.1);
    Eigen::Matrix4d A = Eigen::Matrix4d::Identity();
    auto result = predictor.estimate_mode_dynamics(0, "cv", A, 0.1);
    ASSERT_TRUE(!result.has_value());
}

TEST(estimate_dynamics_known_system) {
    double dt = 0.1;
    Eigen::Matrix4d A = Eigen::Matrix4d::Identity();
    A(0, 2) = dt;  // x += vx * dt
    A(1, 3) = dt;  // y += vy * dt

    Eigen::Vector4d b_true(0.0, 0.0, 0.05, 0.0);  // small acceleration

    OptimalTransportPredictor predictor(dt, 500, 0.1, 5);

    // Generate synthetic trajectory
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.01);

    Eigen::Vector4d state(0.0, 0.0, 1.0, 0.0);
    for (int t = 0; t < 100; ++t) {
        // Record observation
        Eigen::Vector2d pos(state(0), state(1));
        predictor.observe(0, pos, "test_mode");
        predictor.advance_timestep();

        // Propagate
        Eigen::Vector4d w(noise(rng), noise(rng), noise(rng), noise(rng));
        state = A * state + b_true + 0.01 * w;
    }

    auto result = predictor.estimate_mode_dynamics(0, "test_mode", A, dt);
    ASSERT_TRUE(result.has_value());

    auto [b_learned, G_learned] = result.value();
    // b_learned should be close to b_true
    ASSERT_NEAR(b_learned(2), b_true(2), 0.1);  // vx acceleration component
}

// =============================================================================
// Main
// =============================================================================

int main() {
    int passed = 0, failed = 0;

    std::cout << "=== Epsilon Guarantee & OT Integration Tests ===" << std::endl;

    // Part 4 tests
    RUN_TEST(effective_epsilon_basic);
    RUN_TEST(effective_epsilon_zero_scenarios);
    RUN_TEST(effective_epsilon_more_scenarios_lower);
    RUN_TEST(required_scenarios_with_removal);
    RUN_TEST(effective_epsilon_roundtrip);

    // Part 1 tests
    RUN_TEST(config_enforce_flags_default_false);
    RUN_TEST(config_dynamics_learning_interval);

    // Part 2 tests
    RUN_TEST(wasserstein_weight_type_exists);
    RUN_TEST(ot_predictor_compute_mode_weights_uniform_fallback);
    RUN_TEST(ot_predictor_mode_weights_with_observations);

    // Part 3 tests
    RUN_TEST(estimate_dynamics_insufficient_data);
    RUN_TEST(estimate_dynamics_known_system);

    std::cout << "\n=== Results: " << passed << " passed, "
              << failed << " failed ===" << std::endl;

    return failed > 0 ? 1 : 0;
}
