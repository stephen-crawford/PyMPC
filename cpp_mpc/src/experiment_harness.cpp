/**
 * @file experiment_harness.cpp
 * @brief Implementation of experiment harness: rollout runner, stats, CSV.
 */

#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include "collision_constraints.hpp"
#include "dynamics.hpp"
#include "mode_weights.hpp"
#include "scenario_sampler.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <chrono>
#include <cassert>
#include <iomanip>
#include <functional>

namespace scenario_mpc {

// ============================================================================
// CSV Writer
// ============================================================================

CSVWriter::CSVWriter(const std::string& filepath) : ofs_(filepath) {}

CSVWriter::~CSVWriter() {
    if (ofs_.is_open()) ofs_.close();
}

void CSVWriter::write_header() {
    ofs_ << "seed,method,S,eps_wass,sigma,shift_rho,shift_boost,ground_cost,"
         << "collision,collision_step,min_clearance,min_clearance_step,"
         << "total_progress,control_effort,constraint_active_count,"
         << "missed_mode_steps,total_steps,"
         << "avg_solve_ms,p50_solve_ms,p95_solve_ms,max_solve_ms,"
         << "total_dro_injected,avg_safe_horizon,clearance_5pct\n";
}

void CSVWriter::write_record(const RolloutRecord& rec) {
    ofs_ << rec.seed << "," << rec.method << "," << rec.S << ","
         << std::fixed << std::setprecision(4)
         << rec.eps_wass << "," << rec.sigma << ","
         << rec.shift_rho << "," << rec.shift_boost << "," << rec.ground_cost << ","
         << (rec.collision ? 1 : 0) << "," << rec.collision_step << ","
         << std::setprecision(4) << rec.min_clearance << "," << rec.min_clearance_step << ","
         << std::setprecision(4) << rec.total_progress << "," << rec.control_effort << ","
         << rec.constraint_active_count << ","
         << rec.missed_mode_steps << "," << rec.total_steps << ","
         << std::setprecision(4) << rec.avg_solve_ms << "," << rec.p50_solve_ms << ","
         << rec.p95_solve_ms << "," << rec.max_solve_ms << ","
         << rec.total_dro_injected << ","
         << std::setprecision(4) << rec.avg_safe_horizon << "," << rec.clearance_5pct << "\n";
}

void CSVWriter::flush() {
    ofs_.flush();
}

// ============================================================================
// Statistical Helpers
// ============================================================================

std::pair<double, double> wilson_ci(int successes, int n, double z) {
    if (n == 0) return {0.0, 1.0};
    double p_hat = static_cast<double>(successes) / n;
    double denom = 1.0 + z * z / n;
    double center = (p_hat + z * z / (2.0 * n)) / denom;
    double half_width = z * std::sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * n)) / n) / denom;
    return {std::max(0.0, center - half_width), std::min(1.0, center + half_width)};
}

BootstrapResult bootstrap_paired_delta(
    const std::vector<bool>& base_collisions,
    const std::vector<bool>& dro_collisions,
    int n_bootstrap,
    std::mt19937* rng
) {
    int n = static_cast<int>(base_collisions.size());
    assert(n == static_cast<int>(dro_collisions.size()));
    assert(n > 0);

    std::mt19937 local_rng(42);
    if (!rng) rng = &local_rng;
    std::uniform_int_distribution<int> idx_dist(0, n - 1);

    std::vector<double> deltas(n_bootstrap);
    for (int b = 0; b < n_bootstrap; ++b) {
        int base_sum = 0, dro_sum = 0;
        for (int i = 0; i < n; ++i) {
            int j = idx_dist(*rng);
            if (base_collisions[j]) base_sum++;
            if (dro_collisions[j]) dro_sum++;
        }
        double p_base = static_cast<double>(base_sum) / n;
        double p_dro = static_cast<double>(dro_sum) / n;
        deltas[b] = p_base - p_dro;
    }

    std::sort(deltas.begin(), deltas.end());
    double mean_delta = std::accumulate(deltas.begin(), deltas.end(), 0.0) / n_bootstrap;
    double ci_low = deltas[static_cast<int>(0.025 * n_bootstrap)];
    double ci_high = deltas[static_cast<int>(0.975 * n_bootstrap)];

    return {mean_delta, ci_low, ci_high};
}

double mcnemar_chi2(int b, int c) {
    if (b + c == 0) return 0.0;
    double num = std::abs(static_cast<double>(b) - c) - 1.0;  // continuity correction
    num = std::max(0.0, num);
    return (num * num) / (b + c);
}

EffectSizes compute_effect_sizes(double p_base, double p_dro) {
    EffectSizes es;
    es.abs_delta = p_base - p_dro;
    es.rel_delta = (p_base > 1e-12) ? (p_base - p_dro) / p_base : 0.0;
    es.risk_ratio = (p_base > 1e-12) ? p_dro / p_base : 0.0;
    es.cohens_h = 2.0 * std::asin(std::sqrt(p_base)) - 2.0 * std::asin(std::sqrt(p_dro));
    return es;
}

// ============================================================================
// Seed Derivation
// ============================================================================

SeedBundle derive_seeds(unsigned master_seed, int idx) {
    // Hash-based derivation for reproducibility
    auto hash = [](unsigned a, unsigned b) -> unsigned {
        // FNV-1a style mixing
        unsigned h = 2166136261u;
        h ^= a; h *= 16777619u;
        h ^= b; h *= 16777619u;
        return h;
    };

    SeedBundle sb;
    sb.master = master_seed;
    sb.env = hash(master_seed, static_cast<unsigned>(idx * 3 + 0));
    sb.predictor = hash(master_seed, static_cast<unsigned>(idx * 3 + 1));
    sb.scenario = hash(master_seed, static_cast<unsigned>(idx * 3 + 2));
    return sb;
}

// ============================================================================
// Obstacle Simulator (internal)
// ============================================================================

namespace {

struct ObstacleSim {
    ObstacleState state;
    std::string current_mode;
    std::vector<std::string> available_modes;
    std::map<std::string, ModeModel> mode_models;

    void step(double dt, std::mt19937& rng) {
        if (mode_models.find(current_mode) == mode_models.end()) return;
        const auto& model = mode_models.at(current_mode);
        Eigen::VectorXd noise = Eigen::VectorXd::Zero(model.noise_dim());
        std::normal_distribution<double> nd(0, 1);
        for (int i = 0; i < model.noise_dim(); ++i) noise(i) = nd(rng) * 0.02;
        state = model.propagate(state, &noise);
        double spd = std::sqrt(state.vx * state.vx + state.vy * state.vy);
        if (spd > 2.0) { state.vx *= 2.0 / spd; state.vy *= 2.0 / spd; }
    }

    void maybe_switch(double switch_prob, std::mt19937& rng) {
        std::uniform_real_distribution<double> u(0, 1);
        if (u(rng) < switch_prob && !available_modes.empty()) {
            std::uniform_int_distribution<int> idx(0, static_cast<int>(available_modes.size()) - 1);
            current_mode = available_modes[idx(rng)];
        }
    }
};

/// Apply distribution shift to mode observations
void apply_distribution_shift(
    const DistributionShiftConfig& shift,
    ObstacleSim& obs_sim,
    std::mt19937& rng
) {
    if (shift.rho <= 0.0 && shift.dangerous_boost <= 0.0) return;

    std::uniform_real_distribution<double> u(0, 1);

    // With probability rho, switch to a uniform-random mode
    if (shift.rho > 0.0 && u(rng) < shift.rho) {
        if (!obs_sim.available_modes.empty()) {
            std::uniform_int_distribution<int> idx(0, static_cast<int>(obs_sim.available_modes.size()) - 1);
            obs_sim.current_mode = obs_sim.available_modes[idx(rng)];
        }
    }

    // With probability dangerous_boost, force to the boosted mode
    if (shift.dangerous_boost > 0.0 && u(rng) < shift.dangerous_boost) {
        int boost_idx = shift.boosted_mode;
        if (boost_idx < 0) {
            // Default: last mode (typically the rare/dangerous one)
            boost_idx = static_cast<int>(obs_sim.available_modes.size()) - 1;
        }
        if (boost_idx >= 0 && boost_idx < static_cast<int>(obs_sim.available_modes.size())) {
            obs_sim.current_mode = obs_sim.available_modes[boost_idx];
        }
    }
}

/// Configure MPC controller for an ablation variant
void configure_ablation(
    ScenarioMPCConfig& config,
    DROConfig& dro_cfg,
    AblationVariant variant,
    double eps_wass,
    double sigma_scale
) {
    switch (variant) {
        case AblationVariant::NO_INJECTION:
            config.enable_dro = false;
            break;
        case AblationVariant::DRO_FULL:
            config.enable_dro = true;
            dro_cfg.risk_mode = DRORiskMode::FULL;
            dro_cfg.epsilon_base = eps_wass;
            dro_cfg.risk_sigma_scale = sigma_scale;
            break;
        case AblationVariant::DRO_NO_COV:
            config.enable_dro = true;
            dro_cfg.risk_mode = DRORiskMode::NO_COV;
            dro_cfg.epsilon_base = eps_wass;
            dro_cfg.risk_sigma_scale = 0.0;
            break;
        case AblationVariant::DRO_DISTANCE_ONLY:
            config.enable_dro = true;
            dro_cfg.risk_mode = DRORiskMode::DISTANCE_ONLY;
            dro_cfg.epsilon_base = eps_wass;
            dro_cfg.risk_sigma_scale = 0.0;
            break;
        case AblationVariant::RANDOM_INJECTION:
            // We handle random injection in the rollout loop
            config.enable_dro = false;
            break;
        case AblationVariant::ALWAYS_INJECT:
            // We handle always-inject in the rollout loop
            config.enable_dro = false;
            break;
    }
}

}  // anonymous namespace

// ============================================================================
// Rollout Runner
// ============================================================================

RolloutRecord run_experiment_rollout(
    const ExperimentConfig& config,
    unsigned seed
) {
    std::mt19937 rng(seed);
    RolloutRecord rec;
    rec.seed = seed;
    rec.method = ablation_variant_name(config.ablation);
    rec.S = config.num_scenarios;
    rec.eps_wass = config.eps_wass;
    rec.sigma = config.sigma_scale;
    rec.shift_rho = config.shift.rho;
    rec.shift_boost = config.shift.dangerous_boost;
    rec.ground_cost = ground_cost_name(config.ground_cost);

    constexpr double DT = 0.1;
    auto mode_models = create_obstacle_mode_models(DT);

    // Configure MPC
    ScenarioMPCConfig mpc_cfg;
    mpc_cfg.horizon = config.horizon;
    mpc_cfg.dt = DT;
    mpc_cfg.num_scenarios = config.num_scenarios;
    mpc_cfg.ego_radius = 0.5;
    mpc_cfg.obstacle_radius = 0.35;
    mpc_cfg.safety_margin = 0.2;
    mpc_cfg.use_sqp_solver = true;
    mpc_cfg.ensure_mode_coverage = true;
    mpc_cfg.num_discs = config.num_discs;
    mpc_cfg.safe_horizon_enabled = config.safe_horizon_enabled;
    mpc_cfg.safe_horizon_min = 3;
    mpc_cfg.weight_type = WeightType::FREQUENCY;

    // Configure DRO
    DROConfig dro_cfg;
    dro_cfg.ground_cost_type = config.ground_cost;
    configure_ablation(mpc_cfg, dro_cfg, config.ablation,
                       config.eps_wass, config.sigma_scale);

    if (mpc_cfg.enable_dro) {
        mpc_cfg.dro_epsilon_base = dro_cfg.epsilon_base;
        mpc_cfg.dro_risk_sigma_scale = dro_cfg.risk_sigma_scale;
        mpc_cfg.dro_adaptive_epsilon = dro_cfg.adaptive_epsilon;
    }

    AdaptiveScenarioMPC controller(mpc_cfg);

    // Setup obstacle
    int obs_id = 0;
    std::map<std::string, ModeModel> obs_mode_models;
    for (const auto& m : config.obs_modes) {
        if (mode_models.find(m) != mode_models.end())
            obs_mode_models[m] = mode_models[m];
    }
    if (!config.rare_mode.empty() && mode_models.find(config.rare_mode) != mode_models.end()) {
        obs_mode_models[config.rare_mode] = mode_models[config.rare_mode];
    }
    controller.initialize_obstacle(obs_id, obs_mode_models);

    ObstacleSim obs_sim;
    std::uniform_real_distribution<double> y_dist(-0.5, 0.5);
    obs_sim.state = ObstacleState(3.0 + y_dist(rng), 0.3 + y_dist(rng) * 0.5,
                                   -0.1, y_dist(rng) * 0.3);
    obs_sim.current_mode = config.obs_modes.empty() ? "constant_velocity" : config.obs_modes[0];
    obs_sim.available_modes = config.obs_modes;
    if (!config.rare_mode.empty()) obs_sim.available_modes.push_back(config.rare_mode);
    obs_sim.mode_models = obs_mode_models;

    EgoState ego(0.0, 0.0, 0.0, 1.5);
    Eigen::Vector2d goal(20.0, 0.0);
    EgoDynamics dynamics(DT);
    double collision_radius = mpc_cfg.ego_radius + mpc_cfg.obstacle_radius;

    // Initial mode observations
    for (int i = 0; i < 5; ++i) {
        controller.update_mode_observation(obs_id, obs_sim.current_mode, i);
    }

    std::vector<double> solve_times;
    std::vector<double> clearances;
    std::vector<int> safe_horizons;
    double control_effort = 0.0;
    int constraint_active_total = 0;

    for (int step = 0; step < config.rollout_steps; ++step) {
        // Mode switching
        if (!config.rare_mode.empty() && config.rare_switch_prob > 0) {
            std::uniform_real_distribution<double> u(0, 1);
            if (u(rng) < config.rare_switch_prob) {
                obs_sim.current_mode = config.rare_mode;
            } else {
                obs_sim.maybe_switch(config.switch_prob, rng);
            }
        } else {
            obs_sim.maybe_switch(config.switch_prob, rng);
        }

        // Apply distribution shift
        apply_distribution_shift(config.shift, obs_sim, rng);

        controller.update_mode_observation(obs_id, obs_sim.current_mode, step + 5);

        std::map<int, ObstacleState> obstacles;
        obstacles[obs_id] = obs_sim.state;

        auto mpc_result = controller.solve(ego, obstacles, goal, 1.5);
        solve_times.push_back(mpc_result.solve_time * 1000.0);  // ms
        rec.total_dro_injected += mpc_result.num_dro_injected;
        if (mpc_result.safe_horizon > 0)
            safe_horizons.push_back(mpc_result.safe_horizon);

        // Collision check
        double dist = (ego.position() - obs_sim.state.position()).norm();
        clearances.push_back(dist);
        if (dist < rec.min_clearance) {
            rec.min_clearance = dist;
            rec.min_clearance_step = step;
        }
        if (dist < collision_radius && !rec.collision) {
            rec.collision = true;
            rec.collision_step = step;
        }

        // Constraint active count
        constraint_active_total += static_cast<int>(mpc_result.active_scenarios.size());

        // Missed mode check
        bool mode_found = false;
        for (const auto& sc : controller.scenarios()) {
            for (const auto& [oid, traj] : sc.trajectories) {
                if (oid == obs_id && traj.mode_id == obs_sim.current_mode) {
                    mode_found = true; break;
                }
            }
            if (mode_found) break;
        }
        if (!mode_found) rec.missed_mode_steps++;

        // Apply control
        if (mpc_result.success && mpc_result.first_input().has_value()) {
            auto input = mpc_result.first_input().value();
            control_effort += input.a * input.a + input.delta * input.delta;
            ego = dynamics.propagate(ego, input);
        }
        obs_sim.step(DT, rng);
        rec.total_steps++;
    }

    // Compute summary metrics
    rec.total_progress = ego.x;
    rec.control_effort = control_effort;
    rec.constraint_active_count = constraint_active_total;

    // Solve time stats
    if (!solve_times.empty()) {
        rec.avg_solve_ms = std::accumulate(solve_times.begin(), solve_times.end(), 0.0) / solve_times.size();
        std::vector<double> sorted_times = solve_times;
        std::sort(sorted_times.begin(), sorted_times.end());
        int n = static_cast<int>(sorted_times.size());
        rec.p50_solve_ms = sorted_times[n / 2];
        rec.p95_solve_ms = sorted_times[std::min(n - 1, static_cast<int>(0.95 * n))];
        rec.max_solve_ms = sorted_times.back();
    }

    // Safe horizon average
    if (!safe_horizons.empty()) {
        rec.avg_safe_horizon = std::accumulate(safe_horizons.begin(), safe_horizons.end(), 0.0) / safe_horizons.size();
    }

    // Clearance 5th percentile
    if (!clearances.empty()) {
        std::vector<double> sorted_clear = clearances;
        std::sort(sorted_clear.begin(), sorted_clear.end());
        int idx_5pct = std::max(0, static_cast<int>(0.05 * sorted_clear.size()) - 1);
        rec.clearance_5pct = sorted_clear[idx_5pct];
    }

    return rec;
}

}  // namespace scenario_mpc
