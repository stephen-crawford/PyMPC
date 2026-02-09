/**
 * @file test_paper_figures.cpp
 * @brief Generate CSV data for publication-quality paper figures.
 *
 * 8 tests, each writes CSV data to paper_fig_*.csv.
 * At the end, calls visualize_paper_figures.py to produce PNGs and GIFs.
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <random>
#include <set>
#include <sstream>
#include <algorithm>
#include <numeric>

#include "types.hpp"
#include "dynamics.hpp"
#include "mode_weights.hpp"
#include "scenario_sampler.hpp"
#include "trajectory_moments.hpp"
#include "collision_constraints.hpp"
#include "scenario_pruning.hpp"
#include "mpc_controller.hpp"
#include "config.hpp"
#include "optimal_transport_predictor.hpp"

using namespace scenario_mpc;

// ============================================================================
// Test macros (same pattern as test_core.cpp)
// ============================================================================

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

// Path to visualization script (relative to build directory)
const std::string VISUALIZE_SCRIPT = "../examples/visualize_paper_figures.py";

void run_visualization() {
    std::string cmd = "python3 " + VISUALIZE_SCRIPT + " all 2>&1";
    std::cout << "\nGenerating paper figure visualizations..." << std::endl;
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "Warning: Visualization script failed (return code " << ret << ")" << std::endl;
        std::cerr << "Make sure matplotlib is installed: pip install matplotlib pillow" << std::endl;
    }
}

// ============================================================================
// Helper: create standard 3-mode model set
// ============================================================================

static std::map<std::string, ModeModel> create_standard_modes(double dt = 0.1) {
    return create_obstacle_mode_models(dt);
}

// ============================================================================
// Fig 1: Scenario Fan — Mode Weight Comparison
// ============================================================================

TEST(scenario_fan_comparison) {
    double dt = 0.1;
    int horizon = 15;
    int num_scenarios = 20;
    auto modes = create_standard_modes(dt);

    // 1 obstacle at (5,0) moving left at 1 m/s
    ObstacleState obs(5.0, 0.0, -1.0, 0.0);
    int obs_id = 0;

    // Build mode history: 6x constant_velocity, 2x turn_left
    ModeHistory history(obs_id, modes);
    for (int t = 0; t < 6; ++t)
        history.record_observation(t, "constant_velocity");
    for (int t = 6; t < 8; ++t)
        history.record_observation(t, "turn_left");

    std::map<int, ObstacleState> obstacles = {{obs_id, obs}};
    std::map<int, ModeHistory> histories = {{obs_id, history}};

    std::ofstream csv("paper_fig_scenario_fan.csv");
    csv << "weight_type,scenario_id,obs_id,mode,timestep,x,y,probability" << std::endl;

    std::vector<std::pair<std::string, WeightType>> weight_configs = {
        {"UNIFORM", WeightType::UNIFORM},
        {"FREQUENCY", WeightType::FREQUENCY},
        {"RECENCY", WeightType::RECENCY}
    };

    for (const auto& [wt_name, wt] : weight_configs) {
        std::mt19937 rng(42);
        auto scenarios = sample_scenarios(
            obstacles, histories, horizon, num_scenarios,
            wt, 0.9, 8, &rng
        );

        for (const auto& scenario : scenarios) {
            for (const auto& [oid, traj] : scenario.trajectories) {
                for (size_t k = 0; k < traj.steps.size(); ++k) {
                    csv << wt_name << ","
                        << scenario.scenario_id << ","
                        << oid << ","
                        << traj.mode_id << ","
                        << k << ","
                        << std::fixed << std::setprecision(4)
                        << traj.steps[k].mean.x() << ","
                        << traj.steps[k].mean.y() << ","
                        << scenario.probability << std::endl;
                }
            }
        }
    }
    csv.close();

    std::cout << " [paper_fig_scenario_fan.csv]";
    ASSERT_TRUE(true);
}

// ============================================================================
// Fig 2: Trajectory Moments with Uncertainty Ellipses
// ============================================================================

TEST(trajectory_moments) {
    double dt = 0.1;
    int horizon = 20;
    auto modes = create_standard_modes(dt);

    // 1 obstacle at origin, v=(1,0)
    ObstacleState obs(0.0, 0.0, 1.0, 0.0);

    // 50/50 CV + turn_left
    std::map<std::string, double> weights = {
        {"constant_velocity", 0.5},
        {"turn_left", 0.5}
    };

    // Per-mode trajectories
    auto cv_traj = compute_single_mode_trajectory(obs, modes.at("constant_velocity"), horizon);
    auto tl_traj = compute_single_mode_trajectory(obs, modes.at("turn_left"), horizon);

    // Combined moments
    auto combined = compute_trajectory_moments(obs, weights, modes, horizon);

    std::ofstream csv("paper_fig_moments.csv");
    csv << "mode,timestep,mean_x,mean_y,cov_xx,cov_xy,cov_yy" << std::endl;

    // Write CV trajectory
    for (size_t k = 0; k < cv_traj.steps.size(); ++k) {
        csv << "constant_velocity," << k << ","
            << std::fixed << std::setprecision(6)
            << cv_traj.steps[k].mean.x() << ","
            << cv_traj.steps[k].mean.y() << ","
            << cv_traj.steps[k].covariance(0,0) << ","
            << cv_traj.steps[k].covariance(0,1) << ","
            << cv_traj.steps[k].covariance(1,1) << std::endl;
    }

    // Write turn_left trajectory
    for (size_t k = 0; k < tl_traj.steps.size(); ++k) {
        csv << "turn_left," << k << ","
            << std::fixed << std::setprecision(6)
            << tl_traj.steps[k].mean.x() << ","
            << tl_traj.steps[k].mean.y() << ","
            << tl_traj.steps[k].covariance(0,0) << ","
            << tl_traj.steps[k].covariance(0,1) << ","
            << tl_traj.steps[k].covariance(1,1) << std::endl;
    }

    // Write combined moments
    for (int k = 0; k <= combined.horizon(); ++k) {
        auto mean = combined.get_mean_at(k);
        auto cov = combined.get_covariance_at(k);
        csv << "combined," << k << ","
            << std::fixed << std::setprecision(6)
            << mean.x() << ","
            << mean.y() << ","
            << cov(0,0) << ","
            << cov(0,1) << ","
            << cov(1,1) << std::endl;
    }
    csv.close();

    std::cout << " [paper_fig_moments.csv]";
    ASSERT_TRUE(combined.horizon() == horizon);
}

// ============================================================================
// Fig 3: Linearized Collision Constraints
// ============================================================================

TEST(collision_constraints_visualization) {
    double dt = 0.1;
    int horizon = 10;
    int num_scenarios = 5;
    auto modes = create_standard_modes(dt);

    // Ego at origin heading right at v=1
    EgoState ego_start(0.0, 0.0, 0.0, 1.0);
    EgoDynamics ego_dyn(dt);

    // Generate straight-line reference trajectory
    std::vector<EgoState> ref_traj;
    ref_traj.push_back(ego_start);
    for (int k = 0; k < horizon; ++k) {
        EgoInput u(0.0, 0.0); // constant velocity
        ref_traj.push_back(ego_dyn.propagate(ref_traj.back(), u));
    }

    // 1 obstacle at (4,1) heading left at v=0.5
    ObstacleState obs(4.0, 1.0, -0.5, 0.0);
    int obs_id = 0;

    ModeHistory history(obs_id, modes);
    for (int t = 0; t < 5; ++t)
        history.record_observation(t, "constant_velocity");
    history.record_observation(5, "turn_left");
    history.record_observation(6, "turn_right");

    std::map<int, ObstacleState> obstacles = {{obs_id, obs}};
    std::map<int, ModeHistory> histories = {{obs_id, history}};

    std::mt19937 rng(42);
    auto scenarios = sample_scenarios(
        obstacles, histories, horizon, num_scenarios,
        WeightType::FREQUENCY, 0.9, 7, &rng
    );

    // Compute constraints
    double ego_radius = 1.0;
    double obs_radius = 0.5;
    auto constraints = compute_linearized_constraints(
        ref_traj, scenarios, ego_radius, obs_radius, 0.1
    );

    // Write CSV
    std::ofstream csv("paper_fig_constraints.csv");
    csv << "constraint_id,timestep,a_x,a_y,b,ego_x,ego_y,obs_x,obs_y,scenario_id" << std::endl;

    for (size_t i = 0; i < constraints.size(); ++i) {
        const auto& c = constraints[i];
        int k = c.k;
        // Find ego and obstacle positions at this timestep
        double ego_x = ref_traj[k].x;
        double ego_y = ref_traj[k].y;
        double ox = 0, oy = 0;
        // Find the obstacle position from the scenario
        for (const auto& s : scenarios) {
            if (s.scenario_id == c.scenario_id) {
                auto it = s.trajectories.find(c.obstacle_id);
                if (it != s.trajectories.end() && k < static_cast<int>(it->second.steps.size())) {
                    ox = it->second.steps[k].mean.x();
                    oy = it->second.steps[k].mean.y();
                }
                break;
            }
        }
        csv << i << "," << k << ","
            << std::fixed << std::setprecision(6)
            << c.a.x() << "," << c.a.y() << ","
            << c.b << ","
            << ego_x << "," << ego_y << ","
            << ox << "," << oy << ","
            << c.scenario_id << std::endl;
    }
    csv.close();

    std::cout << " [paper_fig_constraints.csv, " << constraints.size() << " constraints]";
    ASSERT_TRUE(constraints.size() > 0);
}

// ============================================================================
// Fig 4: Scenario Pruning Before/After
// ============================================================================

TEST(scenario_pruning) {
    double dt = 0.1;
    int horizon = 10;
    int num_scenarios = 30;
    auto modes = create_standard_modes(dt);

    // Ego at origin heading right
    EgoState ego_start(0.0, 0.0, 0.0, 1.0);
    EgoDynamics ego_dyn(dt);

    std::vector<EgoState> ref_traj;
    ref_traj.push_back(ego_start);
    for (int k = 0; k < horizon; ++k) {
        ref_traj.push_back(ego_dyn.propagate(ref_traj.back(), EgoInput(0.0, 0.0)));
    }

    // 2 obstacles
    ObstacleState obs0(4.0, 1.0, -0.5, 0.0);
    ObstacleState obs1(6.0, -1.5, -0.3, 0.2);

    ModeHistory hist0(0, modes);
    ModeHistory hist1(1, modes);
    for (int t = 0; t < 5; ++t) {
        hist0.record_observation(t, "constant_velocity");
        hist1.record_observation(t, "turn_left");
    }
    hist0.record_observation(5, "turn_left");
    hist1.record_observation(5, "constant_velocity");

    std::map<int, ObstacleState> obstacles = {{0, obs0}, {1, obs1}};
    std::map<int, ModeHistory> histories = {{0, hist0}, {1, hist1}};

    std::mt19937 rng(42);
    auto original = sample_scenarios(
        obstacles, histories, horizon, num_scenarios,
        WeightType::FREQUENCY, 0.9, 6, &rng
    );

    double ego_radius = 1.0;
    double obs_radius = 0.5;

    // Stage 1: Dominance pruning
    auto after_dominance = prune_dominated_scenarios(
        original, ref_traj, ego_radius, obs_radius
    );

    // Stage 2: Compute constraints and inactive removal
    auto constraints = compute_linearized_constraints(
        ref_traj, after_dominance, ego_radius, obs_radius, 0.1
    );
    auto [after_inactive, active_ids] = remove_inactive_scenarios(
        after_dominance, constraints, ref_traj, 1.0 // looser tolerance for demonstration
    );

    // Write CSV with all three stages
    std::ofstream csv("paper_fig_pruning.csv");
    csv << "stage,scenario_id,obs_id,timestep,x,y,probability" << std::endl;

    auto write_scenarios = [&](const std::string& stage, const std::vector<Scenario>& scens) {
        for (const auto& s : scens) {
            for (const auto& [oid, traj] : s.trajectories) {
                for (size_t k = 0; k < traj.steps.size(); k += 2) { // every other step
                    csv << stage << ","
                        << s.scenario_id << ","
                        << oid << "," << k << ","
                        << std::fixed << std::setprecision(4)
                        << traj.steps[k].mean.x() << ","
                        << traj.steps[k].mean.y() << ","
                        << s.probability << std::endl;
                }
            }
        }
    };

    write_scenarios("original", original);
    write_scenarios("after_dominance", after_dominance);
    write_scenarios("after_inactive", after_inactive);
    csv.close();

    std::cout << " [paper_fig_pruning.csv, " << original.size() << " -> "
              << after_dominance.size() << " -> " << after_inactive.size() << "]";
    ASSERT_TRUE(original.size() == static_cast<size_t>(num_scenarios));
}

// ============================================================================
// Fig 5: Mode Weight Adaptation Over Time
// ============================================================================

TEST(mode_weight_evolution) {
    double dt = 0.1;
    auto modes = create_standard_modes(dt);

    int total_steps = 60;

    std::ofstream csv("paper_fig_weight_evolution.csv");
    csv << "step,weight_type,mode,weight,true_mode" << std::endl;

    // For each step, determine the true mode and record an observation
    auto get_true_mode = [](int step) -> std::string {
        if (step < 20) return "constant_velocity";
        else if (step < 40) return "turn_left";
        else return "decelerating";
    };

    std::vector<std::pair<std::string, WeightType>> weight_configs = {
        {"UNIFORM", WeightType::UNIFORM},
        {"FREQUENCY", WeightType::FREQUENCY},
        {"RECENCY", WeightType::RECENCY}
    };

    for (const auto& [wt_name, wt] : weight_configs) {
        ModeHistory history(0, modes);

        for (int step = 0; step < total_steps; ++step) {
            std::string true_mode = get_true_mode(step);
            history.record_observation(step, true_mode);

            auto weights = compute_mode_weights(history, wt, 0.85, step);

            for (const auto& [mode_id, w] : weights) {
                csv << step << ","
                    << wt_name << ","
                    << mode_id << ","
                    << std::fixed << std::setprecision(6)
                    << w << ","
                    << true_mode << std::endl;
            }
        }
    }
    csv.close();

    std::cout << " [paper_fig_weight_evolution.csv]";
    ASSERT_TRUE(true);
}

// ============================================================================
// Fig 6: Full MPC Avoidance Loop (Animated)
// ============================================================================

TEST(mpc_avoidance_loop) {
    double dt = 0.1;
    int total_steps = 80;
    auto modes = create_standard_modes(dt);

    // Configure MPC
    ScenarioMPCConfig config;
    config.horizon = 15;
    config.dt = dt;
    config.num_scenarios = 15;
    config.weight_type = WeightType::FREQUENCY;
    config.recency_decay = 0.9;
    config.ego_radius = 0.8;
    config.obstacle_radius = 0.5;
    config.safety_margin = 0.2;
    config.goal_weight = 10.0;
    config.velocity_weight = 2.0;

    AdaptiveScenarioMPC controller(config);

    // Initialize obstacle 0
    int obs_id = 0;
    controller.initialize_obstacle(obs_id, modes);

    // Ego starts at origin, goal at (15,0)
    EgoState ego(0.0, 0.0, 0.0, 1.0);
    Eigen::Vector2d goal(15.0, 0.0);

    // Obstacle at (7, 0.5) moving left at 0.8 m/s (head-on scenario)
    ObstacleState obs(7.0, 0.5, -0.8, 0.0);
    EgoDynamics ego_dyn(dt);

    // Open CSVs
    std::ofstream csv_main("paper_fig_mpc_loop.csv");
    csv_main << "step,ego_x,ego_y,ego_theta,ego_v,obs_x,obs_y,goal_x,goal_y,"
             << "plan_k,plan_x,plan_y,num_scenarios,num_constraints" << std::endl;

    std::ofstream csv_scen("paper_fig_mpc_scenarios.csv");
    csv_scen << "step,scenario_id,obs_id,mode,timestep,pred_x,pred_y" << std::endl;

    for (int step = 0; step < total_steps; ++step) {
        // Record observation (assume constant_velocity mode most of the time)
        std::string obs_mode = "constant_velocity";
        controller.update_mode_observation(obs_id, obs_mode, step);

        // Solve MPC
        std::map<int, ObstacleState> obstacles_map = {{obs_id, obs}};
        auto result = controller.solve(ego, obstacles_map, goal, 1.5);

        int num_constraints = 0;
        if (result.success) {
            // Count constraints from scenarios
            auto ref_traj = result.ego_trajectory;
            auto scenarios = controller.scenarios();
            auto constraints = compute_linearized_constraints(
                ref_traj, scenarios, config.ego_radius, config.obstacle_radius, config.safety_margin
            );
            num_constraints = static_cast<int>(constraints.size());
        }

        // Write main CSV: one row per plan step
        int plan_len = result.success ? std::min(5, static_cast<int>(result.ego_trajectory.size())) : 0;
        for (int pk = 0; pk < std::max(1, plan_len); ++pk) {
            double px = pk < plan_len ? result.ego_trajectory[pk].x : ego.x;
            double py = pk < plan_len ? result.ego_trajectory[pk].y : ego.y;
            csv_main << step << ","
                     << std::fixed << std::setprecision(4)
                     << ego.x << "," << ego.y << ","
                     << ego.theta << "," << ego.v << ","
                     << obs.x << "," << obs.y << ","
                     << goal.x() << "," << goal.y() << ","
                     << pk << "," << px << "," << py << ","
                     << static_cast<int>(controller.scenarios().size()) << ","
                     << num_constraints << std::endl;
        }

        // Write scenario predictions (top 3)
        const auto& scenarios = controller.scenarios();
        int scen_to_log = std::min(3, static_cast<int>(scenarios.size()));
        for (int s = 0; s < scen_to_log; ++s) {
            const auto& scen = scenarios[s];
            for (const auto& [oid, traj] : scen.trajectories) {
                for (size_t k = 0; k < traj.steps.size(); k += 2) {
                    csv_scen << step << ","
                             << scen.scenario_id << ","
                             << oid << ","
                             << traj.mode_id << ","
                             << k << ","
                             << std::fixed << std::setprecision(4)
                             << traj.steps[k].mean.x() << ","
                             << traj.steps[k].mean.y() << std::endl;
                }
            }
        }

        // Apply control
        if (result.success && result.first_input().has_value()) {
            ego = ego_dyn.propagate(ego, result.first_input().value());
        } else {
            // Gentle braking if MPC fails
            ego = ego_dyn.propagate(ego, EgoInput(-0.5, 0.0));
        }

        // Simulate obstacle (constant velocity)
        obs.x += obs.vx * dt;
        obs.y += obs.vy * dt;
    }

    csv_main.close();
    csv_scen.close();

    std::cout << " [paper_fig_mpc_loop.csv + paper_fig_mpc_scenarios.csv]";
    ASSERT_TRUE(true);
}

// ============================================================================
// Fig 7: Epsilon Guarantee Curve
// ============================================================================

TEST(epsilon_guarantee_curve) {
    std::ofstream csv("paper_fig_epsilon.csv");
    csv << "num_scenarios,dimension,horizon,epsilon_effective,epsilon_required,beta" << std::endl;

    ScenarioMPCConfig cfg;

    // Sweep S from 10 to 500 for different dimensions
    std::vector<int> dimensions = {30, 60, 120};
    std::vector<int> horizons = {5, 10, 20};

    for (size_t di = 0; di < dimensions.size(); ++di) {
        int d = dimensions[di];
        int N = horizons[di];
        cfg.beta = 0.01;

        for (int S = 10; S <= 500; S += 5) {
            double eps_eff = cfg.compute_effective_epsilon(S, d);
            int S_required = cfg.compute_required_scenarios(d);

            csv << S << "," << d << "," << N << ","
                << std::fixed << std::setprecision(6)
                << eps_eff << ","
                << 0.05 << ","  // target epsilon
                << cfg.beta << std::endl;
        }
    }

    // Sweep beta for d=60
    std::vector<double> betas = {0.01, 0.05, 0.1};
    for (double beta : betas) {
        cfg.beta = beta;
        int d = 60;
        for (int S = 10; S <= 500; S += 5) {
            double eps_eff = cfg.compute_effective_epsilon(S, d);
            csv << S << "," << d << "," << 10 << ","
                << std::fixed << std::setprecision(6)
                << eps_eff << ","
                << 0.05 << ","
                << beta << std::endl;
        }
    }

    csv.close();

    std::cout << " [paper_fig_epsilon.csv]";
    ASSERT_TRUE(true);
}

// ============================================================================
// Fig 8: OT Weight Comparison
// ============================================================================

TEST(ot_weight_comparison) {
    double dt = 0.1;
    int total_steps = 30;
    auto modes = create_standard_modes(dt);

    // Create OT predictor with standard mode references
    auto ot_predictor = create_ot_predictor_with_standard_modes(dt, 1.0, 200, 0.1);

    int obs_id = 0;
    ModeHistory history(obs_id, modes);

    // Simulate obstacle doing constant_velocity
    ObstacleState obs(0.0, 0.0, 1.0, 0.0);

    std::ofstream csv("paper_fig_ot_weights.csv");
    csv << "step,method,mode,weight,wasserstein_distance" << std::endl;

    for (int step = 0; step < total_steps; ++step) {
        // Record observations
        history.record_observation(step, "constant_velocity");
        ot_predictor.observe(obs_id, obs.position(), "constant_velocity");
        ot_predictor.advance_timestep();

        // Propagate obstacle with CV mode
        obs = modes.at("constant_velocity").propagate(obs);

        // Compute weights from all three methods
        // Frequency
        auto freq_weights = compute_mode_weights(history, WeightType::FREQUENCY, 0.9, step);
        for (const auto& [mode_id, w] : freq_weights) {
            csv << step << ",frequency," << mode_id << ","
                << std::fixed << std::setprecision(6)
                << w << ",0.0" << std::endl;
        }

        // Recency
        auto rec_weights = compute_mode_weights(history, WeightType::RECENCY, 0.85, step);
        for (const auto& [mode_id, w] : rec_weights) {
            csv << step << ",recency," << mode_id << ","
                << std::fixed << std::setprecision(6)
                << w << ",0.0" << std::endl;
        }

        // Wasserstein (OT) - only if enough samples
        if (step >= 10) {
            std::vector<std::string> mode_ids;
            for (const auto& [mid, _] : modes) mode_ids.push_back(mid);
            auto ot_weights = ot_predictor.compute_mode_weights(obs_id, mode_ids);
            for (const auto& [mode_id, w] : ot_weights) {
                csv << step << ",wasserstein," << mode_id << ","
                    << std::fixed << std::setprecision(6)
                    << w << ",0.0" << std::endl;
            }
        }
    }

    csv.close();

    std::cout << " [paper_fig_ot_weights.csv]";
    ASSERT_TRUE(true);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Paper Figure Data Generation Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    int passed = 0, failed = 0;

    RUN_TEST(scenario_fan_comparison);
    RUN_TEST(trajectory_moments);
    RUN_TEST(collision_constraints_visualization);
    RUN_TEST(scenario_pruning);
    RUN_TEST(mode_weight_evolution);
    RUN_TEST(mpc_avoidance_loop);
    RUN_TEST(epsilon_guarantee_curve);
    RUN_TEST(ot_weight_comparison);

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    // Generate visualizations
    if (failed == 0) {
        run_visualization();
    }

    return failed > 0 ? 1 : 0;
}
