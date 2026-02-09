/**
 * @file config.hpp
 * @brief Configuration for Adaptive Scenario-Based MPC.
 *
 * Provides a clean interface for all hyperparameters.
 */

#ifndef SCENARIO_MPC_CONFIG_HPP
#define SCENARIO_MPC_CONFIG_HPP

#include "types.hpp"
#include <cmath>
#include <stdexcept>

namespace scenario_mpc {

/**
 * @brief Configuration parameters for Adaptive Scenario-Based MPC.
 */
struct ScenarioMPCConfig {
    // Horizon and timing
    int horizon = 20;           ///< Prediction horizon N
    double dt = 0.1;            ///< Timestep duration [s]

    // Ego vehicle parameters
    double ego_radius = 1.0;          ///< Collision radius for ego vehicle [m]
    double max_acceleration = 3.0;    ///< Maximum acceleration [m/s^2]
    double min_acceleration = -5.0;   ///< Minimum acceleration (braking) [m/s^2]
    double max_steering_rate = 0.8;   ///< Maximum steering rate [rad/s]

    // Obstacle parameters
    double obstacle_radius = 0.5;     ///< Default obstacle collision radius [m]

    // Scenario sampling (Theorem 1)
    int num_scenarios = 10;           ///< Number of scenarios to sample (S)
    double confidence_level = 0.95;   ///< Chance constraint confidence (1 - epsilon)
    double beta = 0.01;               ///< Risk parameter for sample size computation

    // Mode weights (Section 4)
    WeightType weight_type = WeightType::FREQUENCY;  ///< Strategy for mode weights
    double recency_decay = 0.9;       ///< Decay factor for recency weighting (lambda)

    // Epsilon guarantee enforcement (Part 4 + Part 1)
    bool enforce_all_scenarios = false;   ///< Use ALL scenarios as constraints (Theorem 1)
    bool enforce_scenario_count = false;  ///< Auto-increase scenarios if S < S_required
    bool ensure_mode_coverage = false;    ///< Guarantee ≥1 scenario per observed mode per obstacle
    bool enable_dynamics_learning = false; ///< Enable OT-based dynamics parameter learning
    int dynamics_learning_interval = 10;  ///< Update dynamics every N timesteps

    // Constraint parameters
    double safety_margin = 0.1;       ///< Additional safety margin [m]

    // Solver parameters
    int solver_max_iter = 500;        ///< Maximum solver iterations
    double solver_tolerance = 1e-4;   ///< Convergence tolerance

    // SQP solver parameters
    bool use_sqp_solver = true;           ///< Use SQP solver instead of heuristic
    int sqp_max_iterations = 5;           ///< Maximum SQP outer iterations
    double sqp_convergence_tol = 1e-3;    ///< SQP convergence tolerance on ||delta_u||
    int qp_max_iterations = 200;          ///< Maximum ADMM iterations per QP
    double qp_tolerance = 1e-4;           ///< ADMM absolute tolerance

    // Objective weights
    double goal_weight = 10.0;        ///< Weight for goal tracking
    double velocity_weight = 1.0;     ///< Weight for velocity tracking
    double acceleration_weight = 0.1; ///< Weight for acceleration penalty
    double steering_weight = 0.1;     ///< Weight for steering penalty

    // Progress-aware cost parameters
    double goal_weight_scale_max = 6.0;       ///< Max goal weight multiplier near end
    double goal_scale_start_fraction = 0.8;   ///< Start scaling goal weight at this progress fraction
    double min_velocity_penalty = 10.0;       ///< Penalty for velocity below threshold
    double min_velocity_threshold = 0.5;      ///< Minimum desired velocity [m/s]

    /// Violation probability (1 - confidence_level)
    double epsilon() const {
        return 1.0 - confidence_level;
    }

    /// Combined ego + obstacle radius for collision checking
    double combined_radius() const {
        return ego_radius + obstacle_radius;
    }

    /**
     * @brief Compute required number of scenarios using Theorem 1.
     *
     * S >= 2/epsilon * (ln(1/beta) + d + R)
     * where d = N * n_x + N * n_u
     *
     * @param num_constraints Number of decision variables (d)
     * @param num_removal Number of removed scenarios (R)
     * @return Minimum number of scenarios required
     */
    int compute_required_scenarios(int num_constraints, int num_removal = 0) const {
        return static_cast<int>(std::ceil(
            2.0 / epsilon() * (std::log(1.0 / beta) + num_constraints + num_removal)
        ));
    }

    /**
     * @brief Compute effective epsilon given actual scenario count.
     *
     * Inverse of Theorem 1: eps = 2*(ln(1/beta) + d + R) / S
     *
     * @param S_actual Actual number of scenarios used
     * @param d Decision variable dimension
     * @param R Number of removed scenarios
     * @return Effective violation probability bound
     */
    double compute_effective_epsilon(int S_actual, int d, int R = 0) const {
        if (S_actual <= 0) return 1.0;
        return 2.0 * (std::log(1.0 / beta) + d + R) / S_actual;
    }

    /// Validate configuration parameters
    void validate() const {
        if (horizon <= 0) {
            throw std::invalid_argument("horizon must be positive");
        }
        if (dt <= 0) {
            throw std::invalid_argument("dt must be positive");
        }
        if (confidence_level <= 0 || confidence_level >= 1) {
            throw std::invalid_argument("confidence_level must be in (0, 1)");
        }
        if (beta <= 0 || beta >= 1) {
            throw std::invalid_argument("beta must be in (0, 1)");
        }
        if (num_scenarios <= 0) {
            throw std::invalid_argument("num_scenarios must be positive");
        }
    }
};

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_CONFIG_HPP
