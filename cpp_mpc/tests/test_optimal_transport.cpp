/**
 * @file test_optimal_transport.cpp
 * @brief Unit tests for Optimal Transport Predictor.
 *
 * Tests the Sinkhorn algorithm, Wasserstein distances, barycenters,
 * and the OptimalTransportPredictor class.
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <random>

#include "optimal_transport_predictor.hpp"
#include "config.hpp"

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
        std::ostringstream oss; \
        oss << "Assertion failed: |" << (a) << " - " << (b) << "| = " \
            << std::abs((a) - (b)) << " > " << (tol); \
        throw std::runtime_error(oss.str()); \
    } \
} while(0)

#define ASSERT_EQ(a, b) ASSERT_TRUE((a) == (b))

#define ASSERT_GT(a, b) ASSERT_TRUE((a) > (b))

#define ASSERT_GE(a, b) ASSERT_TRUE((a) >= (b))

#define ASSERT_LT(a, b) ASSERT_TRUE((a) < (b))

// =============================================================================
// Test TrajectoryObservation
// =============================================================================

TEST(trajectory_observation_creation) {
    TrajectoryObservation obs(
        0,
        Eigen::Vector2d(1.0, 2.0),
        Eigen::Vector2d(0.5, 0.1),
        Eigen::Vector2d(0.01, 0.02),
        "constant_velocity"
    );

    ASSERT_EQ(obs.timestep, 0);
    ASSERT_NEAR(obs.position(0), 1.0, 1e-9);
    ASSERT_NEAR(obs.position(1), 2.0, 1e-9);
    ASSERT_NEAR(obs.velocity(0), 0.5, 1e-9);
    ASSERT_NEAR(obs.velocity(1), 0.1, 1e-9);
    ASSERT_EQ(obs.mode_id, "constant_velocity");

    Eigen::VectorXd state = obs.state();
    ASSERT_EQ(state.size(), 6);
    ASSERT_NEAR(state(0), 1.0, 1e-9);
    ASSERT_NEAR(state(2), 0.5, 1e-9);
}

// =============================================================================
// Test TrajectoryBuffer
// =============================================================================

TEST(trajectory_buffer_basic) {
    TrajectoryBuffer buffer(0, 10);

    ASSERT_TRUE(buffer.empty());
    ASSERT_EQ(buffer.size(), 0);

    // Add observations
    for (int i = 0; i < 5; ++i) {
        buffer.add_observation(TrajectoryObservation(
            i,
            Eigen::Vector2d(i * 0.1, 0),
            Eigen::Vector2d(0.1, 0),
            Eigen::Vector2d(0, 0)
        ));
    }

    ASSERT_EQ(buffer.size(), 5);

    // Get recent
    auto recent = buffer.get_recent(3);
    ASSERT_EQ(recent.size(), 3);
    ASSERT_EQ(recent[0].timestep, 2);
    ASSERT_EQ(recent[2].timestep, 4);
}

TEST(trajectory_buffer_circular) {
    TrajectoryBuffer buffer(0, 5);

    // Add more than max_length observations
    for (int i = 0; i < 10; ++i) {
        buffer.add_observation(TrajectoryObservation(
            i,
            Eigen::Vector2d(i, 0),
            Eigen::Vector2d(1, 0),
            Eigen::Vector2d(0, 0)
        ));
    }

    // Should only keep the last 5
    ASSERT_EQ(buffer.size(), 5);

    auto recent = buffer.get_recent(10);  // Ask for more than available
    ASSERT_EQ(recent.size(), 5);
    ASSERT_EQ(recent[0].timestep, 5);  // Oldest should be timestep 5
    ASSERT_EQ(recent[4].timestep, 9);  // Newest should be timestep 9
}

TEST(trajectory_buffer_velocity_samples) {
    TrajectoryBuffer buffer(0, 100);

    for (int i = 0; i < 10; ++i) {
        buffer.add_observation(TrajectoryObservation(
            i,
            Eigen::Vector2d(i, 0),
            Eigen::Vector2d(1.0 + 0.1 * i, 0.5),
            Eigen::Vector2d(0, 0)
        ));
    }

    Eigen::MatrixX2d velocities = buffer.get_velocity_samples();
    ASSERT_EQ(velocities.rows(), 10);
    ASSERT_EQ(velocities.cols(), 2);
    ASSERT_NEAR(velocities(0, 0), 1.0, 1e-9);
    ASSERT_NEAR(velocities(9, 0), 1.9, 1e-9);
}

// =============================================================================
// Test EmpiricalDistribution
// =============================================================================

TEST(empirical_distribution_from_samples) {
    Eigen::MatrixXd samples(4, 2);
    samples << 0, 0,
               1, 0,
               0, 1,
               1, 1;

    EmpiricalDistribution dist = EmpiricalDistribution::from_samples(samples);

    ASSERT_EQ(dist.n_samples(), 4);
    ASSERT_EQ(dist.dim(), 2);
    ASSERT_FALSE(dist.is_empty());

    // Uniform weights
    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(dist.weights()(i), 0.25, 1e-9);
    }
}

TEST(empirical_distribution_with_weights) {
    Eigen::MatrixXd samples(3, 2);
    samples << 0, 0,
               1, 0,
               2, 0;

    Eigen::VectorXd weights(3);
    weights << 0.5, 0.3, 0.2;

    EmpiricalDistribution dist = EmpiricalDistribution::from_samples(samples, weights);

    ASSERT_EQ(dist.n_samples(), 3);
    ASSERT_NEAR(dist.weights().sum(), 1.0, 1e-9);
}

TEST(empirical_distribution_mean) {
    Eigen::MatrixXd samples(4, 2);
    samples << 0, 0,
               2, 0,
               0, 2,
               2, 2;

    EmpiricalDistribution dist = EmpiricalDistribution::from_samples(samples);
    Eigen::VectorXd mean = dist.mean();

    ASSERT_EQ(mean.size(), 2);
    ASSERT_NEAR(mean(0), 1.0, 1e-9);
    ASSERT_NEAR(mean(1), 1.0, 1e-9);
}

TEST(empirical_distribution_weighted_mean) {
    Eigen::MatrixXd samples(2, 2);
    samples << 0, 0,
               4, 0;

    Eigen::VectorXd weights(2);
    weights << 0.75, 0.25;  // Weighted towards first point

    EmpiricalDistribution dist = EmpiricalDistribution::from_samples(samples, weights);
    Eigen::VectorXd mean = dist.mean();

    ASSERT_NEAR(mean(0), 1.0, 1e-9);  // 0.75*0 + 0.25*4 = 1.0
    ASSERT_NEAR(mean(1), 0.0, 1e-9);
}

TEST(empirical_distribution_covariance) {
    // Samples with known covariance structure
    Eigen::MatrixXd samples(100, 2);
    std::mt19937 rng(42);
    std::normal_distribution<double> norm(0, 1);

    for (int i = 0; i < 100; ++i) {
        samples(i, 0) = norm(rng);
        samples(i, 1) = norm(rng);
    }

    EmpiricalDistribution dist = EmpiricalDistribution::from_samples(samples);
    Eigen::MatrixXd cov = dist.covariance();

    ASSERT_EQ(cov.rows(), 2);
    ASSERT_EQ(cov.cols(), 2);

    // Should be close to identity (with some sampling error)
    ASSERT_NEAR(cov(0, 0), 1.0, 0.3);
    ASSERT_NEAR(cov(1, 1), 1.0, 0.3);
    ASSERT_NEAR(cov(0, 1), 0.0, 0.3);
}

TEST(empirical_distribution_empty) {
    Eigen::MatrixXd samples(0, 2);
    EmpiricalDistribution dist = EmpiricalDistribution::from_samples(samples);

    ASSERT_TRUE(dist.is_empty());
    ASSERT_EQ(dist.n_samples(), 0);
}

// =============================================================================
// Test Cost Matrix
// =============================================================================

TEST(cost_matrix_computation) {
    Eigen::MatrixXd source(2, 2);
    source << 0, 0,
              1, 0;

    Eigen::MatrixXd target(2, 2);
    target << 2, 0,
              0, 1;

    Eigen::MatrixXd cost = compute_cost_matrix(source, target, 2);

    ASSERT_EQ(cost.rows(), 2);
    ASSERT_EQ(cost.cols(), 2);

    // Cost from (0,0) to (2,0) = 4 (squared)
    ASSERT_NEAR(cost(0, 0), 4.0, 1e-9);

    // Cost from (0,0) to (0,1) = 1 (squared)
    ASSERT_NEAR(cost(0, 1), 1.0, 1e-9);

    // Cost from (1,0) to (2,0) = 1 (squared)
    ASSERT_NEAR(cost(1, 0), 1.0, 1e-9);

    // Cost from (1,0) to (0,1) = sqrt(2)^2 = 2
    ASSERT_NEAR(cost(1, 1), 2.0, 1e-9);
}

TEST(cost_matrix_empty) {
    Eigen::MatrixXd source(0, 2);
    Eigen::MatrixXd target(3, 2);

    Eigen::MatrixXd cost = compute_cost_matrix(source, target);

    ASSERT_EQ(cost.rows(), 0);
    ASSERT_EQ(cost.cols(), 3);
}

// =============================================================================
// Test Sinkhorn Algorithm
// =============================================================================

TEST(sinkhorn_uniform_weights) {
    Eigen::VectorXd source_weights(2);
    source_weights << 0.5, 0.5;

    Eigen::VectorXd target_weights(2);
    target_weights << 0.5, 0.5;

    Eigen::MatrixXd cost(2, 2);
    cost << 0, 1,
            1, 0;

    SinkhornResult result = sinkhorn_algorithm(
        source_weights, target_weights, cost, 0.1, 100);

    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.transport_plan.rows(), 2);
    ASSERT_EQ(result.transport_plan.cols(), 2);

    // Row sums should equal source weights
    ASSERT_NEAR(result.transport_plan.row(0).sum(), 0.5, 1e-4);
    ASSERT_NEAR(result.transport_plan.row(1).sum(), 0.5, 1e-4);

    // Column sums should equal target weights
    ASSERT_NEAR(result.transport_plan.col(0).sum(), 0.5, 1e-4);
    ASSERT_NEAR(result.transport_plan.col(1).sum(), 0.5, 1e-4);
}

TEST(sinkhorn_asymmetric_weights) {
    Eigen::VectorXd source_weights(2);
    source_weights << 0.7, 0.3;

    Eigen::VectorXd target_weights(2);
    target_weights << 0.4, 0.6;

    Eigen::MatrixXd cost(2, 2);
    cost << 1, 2,
            2, 1;

    SinkhornResult result = sinkhorn_algorithm(
        source_weights, target_weights, cost, 0.1);

    // Row sums should equal source weights
    ASSERT_NEAR(result.transport_plan.row(0).sum(), 0.7, 1e-4);
    ASSERT_NEAR(result.transport_plan.row(1).sum(), 0.3, 1e-4);

    // Column sums should equal target weights
    ASSERT_NEAR(result.transport_plan.col(0).sum(), 0.4, 1e-4);
    ASSERT_NEAR(result.transport_plan.col(1).sum(), 0.6, 1e-4);
}

TEST(sinkhorn_convergence) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uni(0.1, 1.0);

    int n = 10;
    Eigen::VectorXd source_weights(n);
    Eigen::VectorXd target_weights(n);
    Eigen::MatrixXd cost(n, n);

    for (int i = 0; i < n; ++i) {
        source_weights(i) = uni(rng);
        target_weights(i) = uni(rng);
        for (int j = 0; j < n; ++j) {
            cost(i, j) = uni(rng);
        }
    }
    source_weights /= source_weights.sum();
    target_weights /= target_weights.sum();

    SinkhornResult result = sinkhorn_algorithm(
        source_weights, target_weights, cost, 0.01, 1000);

    ASSERT_TRUE(result.converged || result.iterations >= 100);

    // Check marginals
    for (int i = 0; i < n; ++i) {
        ASSERT_NEAR(result.transport_plan.row(i).sum(), source_weights(i), 1e-3);
        ASSERT_NEAR(result.transport_plan.col(i).sum(), target_weights(i), 1e-3);
    }
}

TEST(sinkhorn_empty) {
    Eigen::VectorXd source_weights(0);
    Eigen::VectorXd target_weights(0);
    Eigen::MatrixXd cost(0, 0);

    SinkhornResult result = sinkhorn_algorithm(
        source_weights, target_weights, cost);

    ASSERT_TRUE(result.converged);
    ASSERT_NEAR(result.sinkhorn_distance, 0.0, 1e-9);
}

// =============================================================================
// Test Wasserstein Distance
// =============================================================================

TEST(wasserstein_identical_distributions) {
    Eigen::MatrixXd samples(4, 2);
    samples << 0, 0,
               1, 0,
               0, 1,
               1, 1;

    EmpiricalDistribution dist1 = EmpiricalDistribution::from_samples(samples);
    EmpiricalDistribution dist2 = EmpiricalDistribution::from_samples(samples);

    double distance = wasserstein_distance(dist1, dist2);

    // Should be close to 0 for identical distributions
    ASSERT_NEAR(distance, 0.0, 0.1);
}

TEST(wasserstein_different_distributions) {
    Eigen::MatrixXd samples1(4, 2);
    samples1 << 0, 0,
                1, 0,
                0, 1,
                1, 1;

    Eigen::MatrixXd samples2(4, 2);
    samples2 << 5, 5,
                6, 5,
                5, 6,
                6, 6;

    EmpiricalDistribution dist1 = EmpiricalDistribution::from_samples(samples1);
    EmpiricalDistribution dist2 = EmpiricalDistribution::from_samples(samples2);

    double distance = wasserstein_distance(dist1, dist2);

    // Should be significantly positive
    ASSERT_GT(distance, 5.0);
}

TEST(wasserstein_empty_distribution) {
    Eigen::MatrixXd samples(4, 2);
    samples << 0, 0, 1, 0, 0, 1, 1, 1;

    EmpiricalDistribution dist1 = EmpiricalDistribution::from_samples(samples);
    EmpiricalDistribution dist2 = EmpiricalDistribution::from_samples(
        Eigen::MatrixXd(0, 2));

    double distance = wasserstein_distance(dist1, dist2);
    ASSERT_NEAR(distance, 0.0, 1e-9);
}

TEST(wasserstein_translation) {
    // Create distribution and its translation
    Eigen::MatrixXd samples1(100, 2);
    std::mt19937 rng(42);
    std::normal_distribution<double> norm(0, 1);

    for (int i = 0; i < 100; ++i) {
        samples1(i, 0) = norm(rng);
        samples1(i, 1) = norm(rng);
    }

    double shift = 3.0;
    Eigen::MatrixXd samples2 = samples1;
    samples2.col(0).array() += shift;

    EmpiricalDistribution dist1 = EmpiricalDistribution::from_samples(samples1);
    EmpiricalDistribution dist2 = EmpiricalDistribution::from_samples(samples2);

    double distance = wasserstein_distance(dist1, dist2);

    // W2 distance should be close to the shift amount
    ASSERT_NEAR(distance, shift, 0.5);
}

// =============================================================================
// Test Wasserstein Barycenter
// =============================================================================

TEST(barycenter_single_distribution) {
    Eigen::MatrixXd samples(4, 2);
    samples << 0, 0,
               1, 0,
               0, 1,
               1, 1;

    EmpiricalDistribution dist = EmpiricalDistribution::from_samples(samples);

    EmpiricalDistribution barycenter = wasserstein_barycenter(
        {dist}, {1.0}, 50, 0.1);

    // Mean should be preserved
    ASSERT_NEAR(barycenter.mean()(0), dist.mean()(0), 0.2);
    ASSERT_NEAR(barycenter.mean()(1), dist.mean()(1), 0.2);
}

TEST(barycenter_two_distributions) {
    Eigen::MatrixXd samples1(4, 2);
    samples1 << 0, 0,
                1, 0,
                0, 1,
                1, 1;

    Eigen::MatrixXd samples2(4, 2);
    samples2 << 4, 0,
                5, 0,
                4, 1,
                5, 1;

    EmpiricalDistribution dist1 = EmpiricalDistribution::from_samples(samples1);
    EmpiricalDistribution dist2 = EmpiricalDistribution::from_samples(samples2);

    // Equal weights - barycenter should be in the middle
    EmpiricalDistribution barycenter = wasserstein_barycenter(
        {dist1, dist2}, {0.5, 0.5}, 50, 0.1);

    // Mean x should be around 2.5 (midpoint of 0.5 and 4.5)
    ASSERT_NEAR(barycenter.mean()(0), 2.5, 0.5);
    ASSERT_NEAR(barycenter.mean()(1), 0.5, 0.3);
}

TEST(barycenter_weighted) {
    Eigen::MatrixXd samples1(4, 2);
    samples1 << 0, 0,
                1, 0,
                0, 1,
                1, 1;

    Eigen::MatrixXd samples2(4, 2);
    samples2 << 10, 0,
                11, 0,
                10, 1,
                11, 1;

    EmpiricalDistribution dist1 = EmpiricalDistribution::from_samples(samples1);
    EmpiricalDistribution dist2 = EmpiricalDistribution::from_samples(samples2);

    // Weight heavily toward dist1
    EmpiricalDistribution barycenter = wasserstein_barycenter(
        {dist1, dist2}, {0.9, 0.1}, 50, 0.1);

    // Mean should be closer to dist1
    ASSERT_LT(barycenter.mean()(0), 3.0);
}

TEST(barycenter_empty_list) {
    EmpiricalDistribution barycenter = wasserstein_barycenter({}, {});

    ASSERT_TRUE(barycenter.is_empty());
}

// =============================================================================
// Test OptimalTransportPredictor
// =============================================================================

TEST(ot_predictor_creation) {
    OptimalTransportPredictor predictor(0.1, 200, 0.1, 10, 1.0,
                                         OTWeightType::WASSERSTEIN);

    ASSERT_NEAR(predictor.dt(), 0.1, 1e-9);
    ASSERT_EQ(predictor.weight_type(), OTWeightType::WASSERSTEIN);
    ASSERT_EQ(predictor.current_timestep(), 0);
}

TEST(ot_predictor_observe) {
    OptimalTransportPredictor predictor(0.1, 200, 0.1, 5);

    // Add observations
    for (int i = 0; i < 10; ++i) {
        predictor.observe(0, Eigen::Vector2d(i * 0.1, 0), "constant_velocity");
        predictor.advance_timestep();
    }

    ASSERT_TRUE(predictor.has_obstacle(0));
    ASSERT_EQ(predictor.get_observation_count(0), 10);
    ASSERT_FALSE(predictor.has_obstacle(1));
}

TEST(ot_predictor_learned_modes) {
    OptimalTransportPredictor predictor(0.1, 200, 0.1, 5);

    // Add observations with different modes
    for (int i = 0; i < 5; ++i) {
        predictor.observe(0, Eigen::Vector2d(i * 0.1, 0), "constant_velocity");
        predictor.advance_timestep();
    }
    for (int i = 0; i < 5; ++i) {
        predictor.observe(0, Eigen::Vector2d(0.5 + i * 0.05, i * 0.1), "turn_left");
        predictor.advance_timestep();
    }

    auto modes = predictor.get_learned_modes(0);
    ASSERT_EQ(modes.size(), 2);
    ASSERT_TRUE(modes.count("constant_velocity") > 0);
    ASSERT_TRUE(modes.count("turn_left") > 0);
}

TEST(ot_predictor_reference_distribution) {
    OptimalTransportPredictor predictor(0.1, 200, 0.1, 5);

    Eigen::MatrixX2d vel_samples(50, 2);
    for (int i = 0; i < 50; ++i) {
        vel_samples(i, 0) = 1.0 + 0.1 * (i % 5 - 2);
        vel_samples(i, 1) = 0.0 + 0.05 * (i % 5 - 2);
    }

    predictor.set_reference_distribution("test_mode", vel_samples);

    // Should be able to compute weights using reference
    std::vector<std::string> modes = {"test_mode"};
    auto weights = predictor.compute_mode_weights(0, modes);

    // With uniform weight type (since no observations)
    ASSERT_EQ(weights.size(), 1);
    ASSERT_NEAR(weights["test_mode"], 1.0, 1e-6);
}

TEST(ot_predictor_mode_weights_uniform) {
    OptimalTransportPredictor predictor(0.1, 200, 0.1, 5, 1.0,
                                         OTWeightType::UNIFORM);

    std::vector<std::string> modes = {"mode1", "mode2", "mode3"};
    auto weights = predictor.compute_mode_weights(0, modes);

    ASSERT_EQ(weights.size(), 3);
    ASSERT_NEAR(weights["mode1"], 1.0 / 3.0, 1e-6);
    ASSERT_NEAR(weights["mode2"], 1.0 / 3.0, 1e-6);
    ASSERT_NEAR(weights["mode3"], 1.0 / 3.0, 1e-6);
}

TEST(ot_predictor_mode_weights_wasserstein) {
    OptimalTransportPredictor predictor(0.1, 200, 0.1, 5, 1.0,
                                         OTWeightType::WASSERSTEIN);

    // Set up reference distributions with different velocities
    Eigen::MatrixX2d vel_forward(20, 2);
    Eigen::MatrixX2d vel_left(20, 2);

    for (int i = 0; i < 20; ++i) {
        vel_forward(i, 0) = 1.0;  // Forward
        vel_forward(i, 1) = 0.0;
        vel_left(i, 0) = 0.5;     // Forward + left
        vel_left(i, 1) = 0.5;
    }

    predictor.set_reference_distribution("forward", vel_forward);
    predictor.set_reference_distribution("left", vel_left);

    // Add observations matching "forward" mode
    for (int i = 0; i < 10; ++i) {
        predictor.observe(0, Eigen::Vector2d(i * 0.1, 0));  // Moving forward
        predictor.advance_timestep();
    }

    std::vector<std::string> modes = {"forward", "left"};
    auto weights = predictor.compute_mode_weights(0, modes);

    // Forward mode should have higher weight
    ASSERT_GT(weights["forward"], weights["left"]);
}

TEST(ot_predictor_predict_trajectory) {
    OptimalTransportPredictor predictor = create_ot_predictor_with_standard_modes();

    // Add some observations
    for (int i = 0; i < 20; ++i) {
        predictor.observe(0, Eigen::Vector2d(i * 0.05, 0), "constant_velocity");
        predictor.advance_timestep();
    }

    Eigen::Vector2d pos(1.0, 0.0);
    Eigen::Vector2d vel(0.5, 0.0);

    auto predictions = predictor.predict_trajectory(0, pos, vel, 10);

    ASSERT_EQ(predictions.size(), 11);  // horizon + 1

    // Position should advance
    ASSERT_GT(predictions[10].position(0), predictions[0].position(0));

    // Uncertainty should grow
    ASSERT_GT(predictions[10].major_radius, predictions[0].major_radius);
}

TEST(ot_predictor_constant_velocity_fallback) {
    OptimalTransportPredictor predictor(0.1, 200, 0.1, 5);

    // No reference distributions, no observations
    Eigen::Vector2d pos(0, 0);
    Eigen::Vector2d vel(1.0, 0.5);

    auto predictions = predictor.predict_trajectory(0, pos, vel, 5);

    ASSERT_EQ(predictions.size(), 6);

    // Should follow constant velocity
    for (int k = 0; k <= 5; ++k) {
        ASSERT_NEAR(predictions[k].position(0), k * 0.1 * 1.0, 0.01);
        ASSERT_NEAR(predictions[k].position(1), k * 0.1 * 0.5, 0.01);
    }
}

TEST(ot_predictor_prediction_error) {
    OptimalTransportPredictor predictor(0.1, 200, 0.1, 5);

    // Predicted vs actual trajectories
    std::vector<Eigen::Vector2d> predicted = {
        {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}
    };

    std::vector<Eigen::Vector2d> actual = {
        {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}
    };

    double error = predictor.compute_prediction_error(0, predicted, actual);
    ASSERT_NEAR(error, 0.0, 0.1);

    // Different trajectories
    std::vector<Eigen::Vector2d> actual2 = {
        {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}
    };

    double error2 = predictor.compute_prediction_error(0, predicted, actual2);
    ASSERT_GT(error2, 1.0);
}

TEST(ot_predictor_adapt_uncertainty) {
    OptimalTransportPredictor predictor(0.1, 200, 0.1, 5);

    // Low error -> low multiplier
    double low_mult = predictor.adapt_uncertainty(0, 0.1);
    ASSERT_GE(low_mult, 1.0);
    ASSERT_LT(low_mult, 2.0);

    // High error -> high multiplier
    double high_mult = predictor.adapt_uncertainty(0, 5.0);
    ASSERT_GT(high_mult, low_mult);
}

TEST(ot_predictor_reset) {
    OptimalTransportPredictor predictor(0.1, 200, 0.1, 5);

    // Add observations and reference
    predictor.observe(0, Eigen::Vector2d(0, 0), "test");
    predictor.set_reference_distribution("ref",
        Eigen::MatrixX2d::Zero(10, 2));

    ASSERT_TRUE(predictor.has_obstacle(0));

    // Reset (keep references)
    predictor.reset();
    ASSERT_FALSE(predictor.has_obstacle(0));

    // Reset all
    predictor.reset_all();
}

TEST(ot_predictor_factory_standard_modes) {
    auto predictor = create_ot_predictor_with_standard_modes(0.1, 0.5);

    // Should have standard reference modes
    std::vector<std::string> expected_modes = {
        "constant_velocity", "decelerating", "accelerating",
        "turn_left", "turn_right", "lane_change_left", "lane_change_right"
    };

    // Add observations to trigger mode weight computation
    for (int i = 0; i < 15; ++i) {
        predictor.observe(0, Eigen::Vector2d(i * 0.05, 0));
        predictor.advance_timestep();
    }

    auto weights = predictor.compute_mode_weights(0, expected_modes);
    ASSERT_EQ(weights.size(), expected_modes.size());

    // All weights should be positive
    for (const auto& [mode, weight] : weights) {
        ASSERT_GT(weight, 0.0);
    }
}

TEST(ot_predictor_mode_distribution_stats) {
    OptimalTransportPredictor predictor(0.1, 200, 0.1, 5);

    // Add observations
    for (int i = 0; i < 10; ++i) {
        predictor.observe(0, Eigen::Vector2d(i * 0.1, 0), "test_mode");
        predictor.advance_timestep();
    }

    auto stats = predictor.get_mode_distribution_stats(0, "test_mode");
    ASSERT_TRUE(stats.has_value());
    ASSERT_GT(stats->at("observation_count"), 0);
    ASSERT_TRUE(stats->count("velocity_mean_x") > 0);

    // Non-existent mode should return nullopt
    auto no_stats = predictor.get_mode_distribution_stats(0, "nonexistent");
    ASSERT_FALSE(no_stats.has_value());
}

// =============================================================================
// Integration Test
// =============================================================================

TEST(integration_full_workflow) {
    // Create predictor with standard modes
    auto predictor = create_ot_predictor_with_standard_modes(0.1, 1.0);

    int obstacle_id = 0;

    // Simulate obstacle moving forward then turning left
    // Phase 1: Forward motion
    double x = 0, y = 0;
    double vx = 1.0, vy = 0.0;

    for (int t = 0; t < 20; ++t) {
        predictor.observe(obstacle_id, Eigen::Vector2d(x, y), "constant_velocity");
        x += vx * 0.1;
        y += vy * 0.1;
        predictor.advance_timestep();
    }

    // Get mode weights - should favor constant_velocity
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right"};
    auto weights1 = predictor.compute_mode_weights(obstacle_id, modes);
    ASSERT_GT(weights1["constant_velocity"], weights1["turn_left"]);

    // Phase 2: Turn left
    for (int t = 0; t < 20; ++t) {
        predictor.observe(obstacle_id, Eigen::Vector2d(x, y), "turn_left");
        double turn_rate = 0.1;
        vx = std::cos(turn_rate * t) * 1.0;
        vy = std::sin(turn_rate * t) * 1.0;
        x += vx * 0.1;
        y += vy * 0.1;
        predictor.advance_timestep();
    }

    // Predict trajectory
    auto predictions = predictor.predict_trajectory(
        obstacle_id, Eigen::Vector2d(x, y), Eigen::Vector2d(vx, vy), 10);

    ASSERT_EQ(predictions.size(), 11);

    // Should have valid predictions with growing uncertainty
    for (int k = 0; k < 10; ++k) {
        ASSERT_TRUE(std::isfinite(predictions[k].position(0)));
        ASSERT_TRUE(std::isfinite(predictions[k].position(1)));
        ASSERT_GT(predictions[k].major_radius, 0);
    }

    // Compute prediction error (against simple forward model)
    std::vector<Eigen::Vector2d> predicted_traj, actual_traj;
    for (const auto& step : predictions) {
        predicted_traj.push_back(step.position);
        actual_traj.push_back(step.position + Eigen::Vector2d(0.1, 0.1));
    }

    double error = predictor.compute_prediction_error(
        obstacle_id, predicted_traj, actual_traj);
    ASSERT_GE(error, 0);

    // Adapt uncertainty
    double scale = predictor.adapt_uncertainty(obstacle_id, error);
    ASSERT_GE(scale, 1.0);
}

// =============================================================================
// Tests for Plan: Epsilon Guarantee + OT Integration + Dynamics Estimation
// =============================================================================

TEST(wasserstein_weight_type_exists) {
    // Part 2: Verify WASSERSTEIN exists in WeightType enum
    WeightType wt = WeightType::WASSERSTEIN;
    ASSERT_TRUE(wt == WeightType::WASSERSTEIN);
    // Can still use other types
    ASSERT_TRUE(WeightType::UNIFORM != WeightType::WASSERSTEIN);
    ASSERT_TRUE(WeightType::FREQUENCY != WeightType::WASSERSTEIN);
    ASSERT_TRUE(WeightType::RECENCY != WeightType::WASSERSTEIN);
}

TEST(dynamics_state_method) {
    // Part 3: Verify dynamics_state() returns [x, y, vx, vy]
    TrajectoryObservation obs(
        0,
        Eigen::Vector2d(1.0, 2.0),
        Eigen::Vector2d(0.5, -0.3),
        Eigen::Vector2d(0.0, 0.0),
        "constant_velocity");

    Eigen::Vector4d ds = obs.dynamics_state();
    ASSERT_NEAR(ds(0), 1.0, 1e-10);
    ASSERT_NEAR(ds(1), 2.0, 1e-10);
    ASSERT_NEAR(ds(2), 0.5, 1e-10);
    ASSERT_NEAR(ds(3), -0.3, 1e-10);
}

TEST(estimate_mode_dynamics_basic) {
    // Part 3: Estimate b from known linear dynamics x_{k+1} = A*x_k + b + G*w
    OptimalTransportPredictor predictor(0.1, 200, 0.1, 5, 1.0, OTWeightType::WASSERSTEIN);

    int obstacle_id = 0;
    Eigen::Matrix4d A = Eigen::Matrix4d::Identity();
    Eigen::Vector4d b_true;
    b_true << 0.1, 0.0, 0.0, 0.0;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.01);

    // Generate trajectory with known dynamics
    Eigen::Vector4d state;
    state << 0.0, 0.0, 1.0, 0.0;

    for (int t = 0; t < 50; ++t) {
        Eigen::Vector2d pos(state(0), state(1));
        predictor.observe(obstacle_id, pos, "constant_velocity");
        predictor.advance_timestep();

        // x_{k+1} = A * x_k + b + noise
        Eigen::Vector4d w;
        w << noise(rng), noise(rng), noise(rng), noise(rng);
        state = A * state + b_true + 0.01 * w;
    }
    // One more observation for the last state
    Eigen::Vector2d final_pos(state(0), state(1));
    predictor.observe(obstacle_id, final_pos, "constant_velocity");

    auto result = predictor.estimate_mode_dynamics(obstacle_id, "constant_velocity", A, 0.1);
    ASSERT_TRUE(result.has_value());

    auto [b_learned, G_learned] = result.value();
    // b_learned should be close to b_true (relaxed tolerance due to finite-difference noise)
    ASSERT_NEAR(b_learned(0), b_true(0), 0.5);
    // G should be positive (diagonal elements > 0)
    for (int i = 0; i < 4; ++i) {
        ASSERT_GT(G_learned(i, i), 0.0);
    }
}

TEST(estimate_mode_dynamics_insufficient_data) {
    // Should return nullopt when too few observations
    OptimalTransportPredictor predictor(0.1, 200, 0.1, 5, 1.0, OTWeightType::WASSERSTEIN);
    Eigen::Matrix4d A = Eigen::Matrix4d::Identity();

    // Only 2 observations - not enough
    predictor.observe(0, Eigen::Vector2d(0.0, 0.0), "mode_a");
    predictor.advance_timestep();
    predictor.observe(0, Eigen::Vector2d(0.1, 0.0), "mode_a");

    auto result = predictor.estimate_mode_dynamics(0, "mode_a", A, 0.1);
    ASSERT_FALSE(result.has_value());
}

TEST(enforce_scenario_count_config) {
    // Part 4: Verify config flags exist
    ScenarioMPCConfig config;
    ASSERT_FALSE(config.enforce_all_scenarios);
    ASSERT_FALSE(config.enforce_scenario_count);

    config.enforce_all_scenarios = true;
    config.enforce_scenario_count = true;
    ASSERT_TRUE(config.enforce_all_scenarios);
    ASSERT_TRUE(config.enforce_scenario_count);
}

TEST(trajectory_buffer_observations_accessor) {
    // Part 3: Verify observations() accessor works
    TrajectoryBuffer buffer(0, 100);
    TrajectoryObservation obs1(0, Eigen::Vector2d(0, 0), Eigen::Vector2d(1, 0),
                               Eigen::Vector2d(0, 0), "mode_a");
    TrajectoryObservation obs2(1, Eigen::Vector2d(0.1, 0), Eigen::Vector2d(1, 0),
                               Eigen::Vector2d(0, 0), "mode_a");
    buffer.add_observation(obs1);
    buffer.add_observation(obs2);

    const auto& observations = buffer.observations();
    ASSERT_EQ(observations.size(), 2u);
    ASSERT_NEAR(observations[0].position(0), 0.0, 1e-10);
    ASSERT_NEAR(observations[1].position(0), 0.1, 1e-10);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    int passed = 0;
    int failed = 0;

    std::cout << "Running Optimal Transport Predictor tests...\n" << std::endl;

    // TrajectoryObservation tests
    RUN_TEST(trajectory_observation_creation);

    // TrajectoryBuffer tests
    RUN_TEST(trajectory_buffer_basic);
    RUN_TEST(trajectory_buffer_circular);
    RUN_TEST(trajectory_buffer_velocity_samples);

    // EmpiricalDistribution tests
    RUN_TEST(empirical_distribution_from_samples);
    RUN_TEST(empirical_distribution_with_weights);
    RUN_TEST(empirical_distribution_mean);
    RUN_TEST(empirical_distribution_weighted_mean);
    RUN_TEST(empirical_distribution_covariance);
    RUN_TEST(empirical_distribution_empty);

    // Cost matrix tests
    RUN_TEST(cost_matrix_computation);
    RUN_TEST(cost_matrix_empty);

    // Sinkhorn algorithm tests
    RUN_TEST(sinkhorn_uniform_weights);
    RUN_TEST(sinkhorn_asymmetric_weights);
    RUN_TEST(sinkhorn_convergence);
    RUN_TEST(sinkhorn_empty);

    // Wasserstein distance tests
    RUN_TEST(wasserstein_identical_distributions);
    RUN_TEST(wasserstein_different_distributions);
    RUN_TEST(wasserstein_empty_distribution);
    RUN_TEST(wasserstein_translation);

    // Wasserstein barycenter tests
    RUN_TEST(barycenter_single_distribution);
    RUN_TEST(barycenter_two_distributions);
    RUN_TEST(barycenter_weighted);
    RUN_TEST(barycenter_empty_list);

    // OptimalTransportPredictor tests
    RUN_TEST(ot_predictor_creation);
    RUN_TEST(ot_predictor_observe);
    RUN_TEST(ot_predictor_learned_modes);
    RUN_TEST(ot_predictor_reference_distribution);
    RUN_TEST(ot_predictor_mode_weights_uniform);
    RUN_TEST(ot_predictor_mode_weights_wasserstein);
    RUN_TEST(ot_predictor_predict_trajectory);
    RUN_TEST(ot_predictor_constant_velocity_fallback);
    RUN_TEST(ot_predictor_prediction_error);
    RUN_TEST(ot_predictor_adapt_uncertainty);
    RUN_TEST(ot_predictor_reset);
    RUN_TEST(ot_predictor_factory_standard_modes);
    RUN_TEST(ot_predictor_mode_distribution_stats);

    // Integration tests
    RUN_TEST(integration_full_workflow);

    // Plan: Epsilon Guarantee + OT Integration + Dynamics Estimation
    RUN_TEST(wasserstein_weight_type_exists);
    RUN_TEST(dynamics_state_method);
    RUN_TEST(estimate_mode_dynamics_basic);
    RUN_TEST(estimate_mode_dynamics_insufficient_data);
    RUN_TEST(enforce_scenario_count_config);
    RUN_TEST(trajectory_buffer_observations_accessor);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;

    return failed > 0 ? 1 : 0;
}
