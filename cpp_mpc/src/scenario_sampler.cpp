/**
 * @file scenario_sampler.cpp
 * @brief Implementation of scenario sampling.
 */

#include "scenario_sampler.hpp"
#include "trajectory_moments.hpp"
#include <cmath>

namespace scenario_mpc {

namespace {

/**
 * @brief Sample a single obstacle trajectory.
 */
ObstacleTrajectory sample_obstacle_trajectory(
    int obstacle_id,
    const ObstacleState& initial_state,
    const std::map<std::string, ModeModel>& available_modes,
    const std::map<std::string, double>& mode_weights,
    int horizon,
    std::mt19937& rng
) {
    // Sample mode for this trajectory (constant mode over horizon for simplicity)
    std::string sampled_mode_id = sample_mode_from_weights(mode_weights, rng);
    const ModeModel& mode = available_modes.at(sampled_mode_id);

    // Sample noise sequence
    int noise_dim = mode.noise_dim();
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    Eigen::MatrixXd noise_samples(horizon, noise_dim);
    for (int k = 0; k < horizon; ++k) {
        for (int d = 0; d < noise_dim; ++d) {
            noise_samples(k, d) = normal_dist(rng);
        }
    }

    // Propagate trajectory
    std::vector<PredictionStep> steps;
    steps.reserve(horizon + 1);

    Eigen::Vector4d x = initial_state.to_array();
    Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();

    // Initial step
    steps.emplace_back(0, x.head<2>(), cov.block<2, 2>(0, 0));

    for (int k = 0; k < horizon; ++k) {
        // Propagate with sampled noise
        Eigen::VectorXd noise = noise_samples.row(k).transpose();
        x = mode.A * x + mode.b + mode.G * noise;

        // Update covariance (for uncertainty representation)
        cov = mode.A * cov * mode.A.transpose() + mode.G * mode.G.transpose();

        steps.emplace_back(k + 1, x.head<2>(), cov.block<2, 2>(0, 0));
    }

    return ObstacleTrajectory(
        obstacle_id, sampled_mode_id, steps, mode_weights.at(sampled_mode_id)
    );
}

/**
 * @brief Sample trajectory with mode switching at each timestep.
 */
ObstacleTrajectory sample_trajectory_with_mode_sequence(
    int obstacle_id,
    const ObstacleState& initial_state,
    const std::map<std::string, ModeModel>& available_modes,
    const std::map<std::string, double>& mode_weights,
    int horizon,
    std::mt19937& rng
) {
    std::vector<PredictionStep> steps;
    steps.reserve(horizon + 1);

    Eigen::Vector4d x = initial_state.to_array();
    Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();

    // Initial step
    steps.emplace_back(0, x.head<2>(), cov.block<2, 2>(0, 0));

    // Track which mode was used most (for trajectory labeling)
    std::map<std::string, int> mode_counts;
    for (const auto& [mode_id, _] : mode_weights) {
        mode_counts[mode_id] = 0;
    }

    double trajectory_prob = 1.0;
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    for (int k = 0; k < horizon; ++k) {
        // Sample mode for this timestep
        std::string mode_id = sample_mode_from_weights(mode_weights, rng);
        const ModeModel& mode = available_modes.at(mode_id);
        mode_counts[mode_id]++;
        trajectory_prob *= mode_weights.at(mode_id);

        // Sample noise
        int noise_dim = mode.noise_dim();
        Eigen::VectorXd noise(noise_dim);
        for (int d = 0; d < noise_dim; ++d) {
            noise(d) = normal_dist(rng);
        }

        // Propagate
        x = mode.A * x + mode.b + mode.G * noise;
        cov = mode.A * cov * mode.A.transpose() + mode.G * mode.G.transpose();

        steps.emplace_back(k + 1, x.head<2>(), cov.block<2, 2>(0, 0));
    }

    // Label trajectory with most frequent mode
    std::string dominant_mode;
    int max_count = 0;
    for (const auto& [mode_id, count] : mode_counts) {
        if (count > max_count) {
            max_count = count;
            dominant_mode = mode_id;
        }
    }

    return ObstacleTrajectory(obstacle_id, dominant_mode, steps, trajectory_prob);
}

}  // anonymous namespace

std::vector<Scenario> sample_scenarios(
    const std::map<int, ObstacleState>& obstacles,
    const std::map<int, ModeHistory>& mode_histories,
    int horizon,
    int num_scenarios,
    WeightType weight_type,
    double recency_decay,
    int current_timestep,
    std::mt19937* rng
) {
    // Create local RNG if not provided
    std::mt19937 local_rng;
    if (rng == nullptr) {
        std::random_device rd;
        local_rng = std::mt19937(rd());
        rng = &local_rng;
    }

    std::vector<Scenario> scenarios;
    scenarios.reserve(num_scenarios);

    for (int s = 0; s < num_scenarios; ++s) {
        std::map<int, ObstacleTrajectory> trajectories;

        for (const auto& [obs_id, obs_state] : obstacles) {
            auto hist_it = mode_histories.find(obs_id);
            if (hist_it == mode_histories.end()) {
                // No mode history - skip this obstacle
                continue;
            }

            const ModeHistory& mode_history = hist_it->second;

            // Step 1: Compute mode weights
            auto mode_weights = compute_mode_weights(
                mode_history, weight_type, recency_decay, current_timestep
            );

            if (mode_weights.empty()) {
                continue;
            }

            // Step 2 & 3 & 4: Sample trajectory
            ObstacleTrajectory trajectory = sample_obstacle_trajectory(
                obs_id, obs_state, mode_history.available_modes, mode_weights,
                horizon, *rng
            );

            trajectories[obs_id] = trajectory;
        }

        // Compute scenario probability as product of trajectory probabilities
        double scenario_prob = 1.0;
        for (const auto& [_, traj] : trajectories) {
            scenario_prob *= traj.probability;
        }

        scenarios.emplace_back(s, trajectories, scenario_prob);
    }

    return scenarios;
}

std::vector<Scenario> sample_scenarios_with_mode_sequences(
    const std::map<int, ObstacleState>& obstacles,
    const std::map<int, ModeHistory>& mode_histories,
    int horizon,
    int num_scenarios,
    WeightType weight_type,
    std::mt19937* rng
) {
    // Create local RNG if not provided
    std::mt19937 local_rng;
    if (rng == nullptr) {
        std::random_device rd;
        local_rng = std::mt19937(rd());
        rng = &local_rng;
    }

    std::vector<Scenario> scenarios;
    scenarios.reserve(num_scenarios);

    for (int s = 0; s < num_scenarios; ++s) {
        std::map<int, ObstacleTrajectory> trajectories;

        for (const auto& [obs_id, obs_state] : obstacles) {
            auto hist_it = mode_histories.find(obs_id);
            if (hist_it == mode_histories.end()) {
                continue;
            }

            const ModeHistory& mode_history = hist_it->second;
            auto mode_weights = compute_mode_weights(mode_history, weight_type);

            if (mode_weights.empty()) {
                continue;
            }

            // Sample trajectory with mode switching
            ObstacleTrajectory trajectory = sample_trajectory_with_mode_sequence(
                obs_id, obs_state, mode_history.available_modes, mode_weights,
                horizon, *rng
            );

            trajectories[obs_id] = trajectory;
        }

        // Uniform scenario weight for mode-switching version
        scenarios.emplace_back(s, trajectories, 1.0 / num_scenarios);
    }

    return scenarios;
}

std::vector<Scenario> sample_scenarios_with_mode_coverage(
    const std::map<int, ObstacleState>& obstacles,
    const std::map<int, ModeHistory>& mode_histories,
    int horizon,
    int num_scenarios,
    WeightType weight_type,
    double recency_decay,
    int current_timestep,
    std::mt19937* rng
) {
    // Create local RNG if not provided
    std::mt19937 local_rng;
    if (rng == nullptr) {
        std::random_device rd;
        local_rng = std::mt19937(rd());
        rng = &local_rng;
    }

    // Phase 0: Collect coverage modes (modes with nonzero weight) for each obstacle
    struct ObsCoverageInfo {
        std::map<std::string, double> mode_weights;
        std::vector<std::string> coverage_modes;  // modes with nonzero weight
    };
    std::map<int, ObsCoverageInfo> obs_info;

    int max_coverage_modes = 0;

    for (const auto& [obs_id, obs_state] : obstacles) {
        auto hist_it = mode_histories.find(obs_id);
        if (hist_it == mode_histories.end()) continue;

        const ModeHistory& mode_history = hist_it->second;
        auto mode_weights = compute_mode_weights(
            mode_history, weight_type, recency_decay, current_timestep
        );
        if (mode_weights.empty()) continue;

        ObsCoverageInfo info;
        info.mode_weights = mode_weights;
        for (const auto& [mode_id, w] : mode_weights) {
            if (w > 0.0) {
                info.coverage_modes.push_back(mode_id);
            }
        }
        max_coverage_modes = std::max(max_coverage_modes, static_cast<int>(info.coverage_modes.size()));
        obs_info[obs_id] = std::move(info);
    }

    int num_coverage = std::min(max_coverage_modes, num_scenarios);

    std::vector<Scenario> scenarios;
    scenarios.reserve(num_scenarios);

    // Phase 1: Coverage scenarios — force each mode to appear at least once
    for (int s = 0; s < num_coverage; ++s) {
        std::map<int, ObstacleTrajectory> trajectories;

        for (const auto& [obs_id, obs_state] : obstacles) {
            auto info_it = obs_info.find(obs_id);
            if (info_it == obs_info.end()) continue;

            const auto& info = info_it->second;
            auto hist_it = mode_histories.find(obs_id);
            const ModeHistory& mode_history = hist_it->second;

            if (s < static_cast<int>(info.coverage_modes.size())) {
                // Force this specific mode: create weights with 1.0 for the forced mode
                std::map<std::string, double> forced_weights;
                for (const auto& [mode_id, _] : info.mode_weights) {
                    forced_weights[mode_id] = 0.0;
                }
                forced_weights[info.coverage_modes[s]] = 1.0;

                ObstacleTrajectory trajectory = sample_obstacle_trajectory(
                    obs_id, obs_state, mode_history.available_modes, forced_weights,
                    horizon, *rng
                );
                // Set probability to the actual mode weight (not 1.0)
                trajectory.probability = info.mode_weights.at(info.coverage_modes[s]);
                trajectories[obs_id] = trajectory;
            } else {
                // This obstacle has fewer modes than num_coverage — sample normally
                ObstacleTrajectory trajectory = sample_obstacle_trajectory(
                    obs_id, obs_state, mode_history.available_modes, info.mode_weights,
                    horizon, *rng
                );
                trajectories[obs_id] = trajectory;
            }
        }

        double scenario_prob = 1.0;
        for (const auto& [_, traj] : trajectories) {
            scenario_prob *= traj.probability;
        }
        scenarios.emplace_back(s, trajectories, scenario_prob);
    }

    // Phase 2: Remaining scenarios — normal weight-based sampling
    for (int s = num_coverage; s < num_scenarios; ++s) {
        std::map<int, ObstacleTrajectory> trajectories;

        for (const auto& [obs_id, obs_state] : obstacles) {
            auto info_it = obs_info.find(obs_id);
            if (info_it == obs_info.end()) continue;

            const auto& info = info_it->second;
            auto hist_it = mode_histories.find(obs_id);
            const ModeHistory& mode_history = hist_it->second;

            ObstacleTrajectory trajectory = sample_obstacle_trajectory(
                obs_id, obs_state, mode_history.available_modes, info.mode_weights,
                horizon, *rng
            );
            trajectories[obs_id] = trajectory;
        }

        double scenario_prob = 1.0;
        for (const auto& [_, traj] : trajectories) {
            scenario_prob *= traj.probability;
        }
        scenarios.emplace_back(s, trajectories, scenario_prob);
    }

    return scenarios;
}

int compute_required_scenarios(double epsilon, double beta, int num_decision_vars) {
    return static_cast<int>(std::ceil(
        2.0 / epsilon * (std::log(1.0 / beta) + num_decision_vars)
    ));
}

}  // namespace scenario_mpc
