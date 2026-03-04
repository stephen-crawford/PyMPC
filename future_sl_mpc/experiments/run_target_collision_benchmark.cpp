/**
 * @file run_target_collision_benchmark.cpp
 * @brief Sweep method knobs until collision rate target is met, then report compute cost.
 *
 * Goal: Find, for each method, the cheapest configuration that achieves a target
 * collision rate (default 2%) with a statistically meaningful criterion:
 *   - Use 95% Wilson confidence interval for Bernoulli collision indicator.
 *   - Declare "meets target" when the *upper* CI bound <= target.
 *
 * This avoids misleading conclusions from tiny-sample estimates (e.g. 0/10).
 *
 * Usage (from cpp_mpc/build):
 *   ./target_collision_benchmark <out_dir> <max_rollouts_per_config> <target_collision_rate>
 *
 * Output:
 *   - target_collision_benchmark.csv
 *   - Prints per-method best config summary.
 */
#include "experiment_harness.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::pair<double, double> wilson_ci(int successes, int n, double z = 1.96) {
    if (n <= 0) return {0.0, 1.0};
    const double p_hat = static_cast<double>(successes) / static_cast<double>(n);
    const double denom = 1.0 + (z * z) / static_cast<double>(n);
    const double center = (p_hat + (z * z) / (2.0 * static_cast<double>(n))) / denom;
    const double half = z * std::sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * n)) / n) / denom;
    return {std::max(0.0, center - half), std::min(1.0, center + half)};
}

static double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = q * static_cast<double>(v.size() - 1);
    const size_t i0 = static_cast<size_t>(std::floor(idx));
    const size_t i1 = std::min(v.size() - 1, i0 + 1);
    const double t = idx - static_cast<double>(i0);
    return (1.0 - t) * v[i0] + t * v[i1];
}

static scenario_mpc::ExperimentConfig make_baseline_config(const std::string& out_dir) {
    scenario_mpc::ExperimentConfig c;
    c.num_scenarios = 30;
    c.eps_wass = 0.1;
    c.sigma_scale = 1.0;
    c.horizon = 20;
    c.num_discs = 3;
    c.safe_horizon_enabled = true;
    c.switch_prob = 0.1;
    c.rollout_steps = 80;
    c.rare_switch_prob = 0.05;
    c.obs_modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
    c.rare_mode = "lane_change_left";
    c.scenario_tag = "target_collision";
    c.gan_scenario_csv_path = out_dir + "gan_scenarios.csv";
    c.reservoir_scenario_csv_path = out_dir + "reservoir_scenarios.csv";
    c.seek_avoid_scenario_csv_path = out_dir + "seek_avoid_scenarios.csv";
    c.seek_avoid_ml_scenario_csv_path = out_dir + "seek_avoid_ml_scenarios.csv";
    c.use_gan_cache = true;
    return c;
}

struct SweepRow {
    std::string method;
    std::string knob;
    int knob_value = 0;
    int n = 0;
    int collisions = 0;
    double p_hat = 0.0;
    double ci_lo = 0.0;
    double ci_hi = 1.0;
    double solve_ms_mean = 0.0;
    double solve_ms_p95 = 0.0;
    double solve_ms_max = 0.0;
    bool meets_target = false;
};

static SweepRow run_until_certified(
    scenario_mpc::ExperimentConfig cfg,
    const std::string& method,
    const std::string& knob,
    int knob_value,
    int max_rollouts,
    double target_collision_rate,
    unsigned seed_base
) {
    constexpr int MIN_ROLLOUTS_FOR_DECISION = 60;  // avoid declaring success/failure too early

    int collisions = 0;
    std::vector<double> solve_ms;
    solve_ms.reserve(static_cast<size_t>(max_rollouts));

    SweepRow row;
    row.method = method;
    row.knob = knob;
    row.knob_value = knob_value;

    for (int r = 0; r < max_rollouts; ++r) {
        const unsigned seed = seed_base + static_cast<unsigned>(r);
        auto rec = scenario_mpc::run_experiment_rollout_future_sl(cfg, seed, method);
        if (rec.collision) collisions++;
        solve_ms.push_back(rec.avg_solve_ms);

        const int n = r + 1;
        auto [lo, hi] = wilson_ci(collisions, n);
        const double p_hat = static_cast<double>(collisions) / n;

        // Success criterion: upper CI bound is below target.
        const bool can_decide = (n >= MIN_ROLLOUTS_FOR_DECISION);
        if (can_decide && hi <= target_collision_rate) {
            row.meets_target = true;
            row.n = n;
            row.collisions = collisions;
            row.p_hat = p_hat;
            row.ci_lo = lo;
            row.ci_hi = hi;
            break;
        }

        // Early failure criterion: even lower CI bound is above target.
        if (can_decide && lo > target_collision_rate) {
            row.meets_target = false;
            row.n = n;
            row.collisions = collisions;
            row.p_hat = p_hat;
            row.ci_lo = lo;
            row.ci_hi = hi;
            break;
        }

        // If we reach max_rollouts, record the final estimate.
        if (n == max_rollouts) {
            row.meets_target = (hi <= target_collision_rate);
            row.n = n;
            row.collisions = collisions;
            row.p_hat = p_hat;
            row.ci_lo = lo;
            row.ci_hi = hi;
        }
    }

    // Compute solve time stats over the n rollouts used.
    if (!solve_ms.empty()) {
        row.solve_ms_mean = std::accumulate(solve_ms.begin(), solve_ms.end(), 0.0) / solve_ms.size();
        row.solve_ms_p95 = percentile(solve_ms, 0.95);
        row.solve_ms_max = *std::max_element(solve_ms.begin(), solve_ms.end());
    }
    return row;
}

int main(int argc, char** argv) {
    std::string out_dir = "../future_sl_mpc/experiments/results/";
    if (argc >= 2) out_dir = argv[1];
    int max_rollouts_per_config = 250;
    if (argc >= 3) max_rollouts_per_config = std::atoi(argv[2]);
    double target = 0.02;
    if (argc >= 4) target = std::atof(argv[3]);

    fs::create_directories(out_dir);
    std::string csv_path = out_dir + "target_collision_benchmark.csv";
    std::ofstream csv(csv_path);
    csv.setf(std::ios::unitbuf);
    csv << "method,knob,knob_value,n,collisions,p_hat,ci_lo,ci_hi,solve_ms_mean,solve_ms_p95,solve_ms_max,meets_target\n";

    // Methods to compare.
    const std::vector<std::string> methods = {
        "SHMPC",
        "SHMPC_Conformal",
        "SHMPC_Hazard",
        "SHMPC_Bandit",
        "SHMPC_Compiler",
        "SHMPC_RTA",
        "SHMPC_QuotientSpace",
        "SHMPC_DoubleDual",
        "SHMPC_GAN",
        "SHMPC_GAN_Reduced"
    };

    // Sweep grids. Keep them modest: target certification already needs many rollouts.
    // For very low targets (e.g. 2%), non-adversarial variants typically won’t reach the goal
    // within reasonable compute budgets. Keep this grid small; methods that can meet the target
    // (GAN variants) get their own dedicated grids below.
    const std::vector<int> S_grid_common = {20, 30, 50};
    const std::vector<int> S_grid_gan = {8, 10, 12, 15, 18, 20, 25, 30};
    const std::vector<int> S_grid_gan_reduced = {6, 8, 10, 12, 15, 18};

    std::vector<SweepRow> all_rows;

    std::cout.setf(std::ios::unitbuf);
    std::cout << "Target collision benchmark: target=" << (target * 100.0)
              << "%, max_rollouts_per_config=" << max_rollouts_per_config << "\n";
    std::cout << "Criterion: 95% Wilson CI upper bound <= target.\n\n";

    unsigned seed_base = 700000u;
    for (const auto& method : methods) {
        std::cout << "Method: " << method << "\n";

        std::vector<int> grid = S_grid_common;
        std::string knob = "S";
        if (method == "SHMPC_GAN") {
            grid = S_grid_gan;
            knob = "gan_num_scenarios_override";
        } else if (method == "SHMPC_GAN_Reduced") {
            grid = S_grid_gan_reduced;
            knob = "gan_reduced_num_scenarios";
        } else {
            knob = "num_scenarios";
        }

        SweepRow best;
        bool has_best = false;

        for (int S : grid) {
            auto cfg = make_baseline_config(out_dir);
            cfg.num_scenarios = S;

            if (method == "SHMPC_GAN") {
                cfg.gan_num_scenarios_override = S;
                cfg.num_scenarios = 30;  // keep SHMPC sampling budget fixed; GAN override controls the scenario count used
            } else if (method == "SHMPC_GAN_Reduced") {
                cfg.gan_reduced_num_scenarios = S;
                cfg.num_scenarios = 30;
            }

            const unsigned local_seed_base = seed_base + static_cast<unsigned>(std::hash<std::string>{}(method)) + static_cast<unsigned>(S * 100);
            SweepRow row = run_until_certified(cfg, method, knob, S, max_rollouts_per_config, target, local_seed_base);
            all_rows.push_back(row);

            csv << row.method << "," << row.knob << "," << row.knob_value << ","
                << row.n << "," << row.collisions << ","
                << std::setprecision(8) << row.p_hat << "," << row.ci_lo << "," << row.ci_hi << ","
                << std::setprecision(6) << row.solve_ms_mean << "," << row.solve_ms_p95 << "," << row.solve_ms_max << ","
                << (row.meets_target ? 1 : 0) << "\n";

            std::cout << "  " << knob << "=" << S
                      << " -> p=" << std::fixed << std::setprecision(3) << (row.p_hat * 100.0) << "% "
                      << "CI=[" << (row.ci_lo * 100.0) << ", " << (row.ci_hi * 100.0) << "] "
                      << "solve=" << std::setprecision(3) << row.solve_ms_mean << " ms "
                      << (row.meets_target ? "MEETS" : "     ") << " (n=" << row.n << ")\n";

            if (row.meets_target) {
                if (!has_best || row.solve_ms_mean < best.solve_ms_mean) {
                    best = row;
                    has_best = true;
                }
                // We can keep sweeping to see if a different S is cheaper.
            }
        }

        if (has_best) {
            std::cout << "  BEST: " << best.knob << "=" << best.knob_value
                      << " solve_mean=" << best.solve_ms_mean << " ms, "
                      << "p_hat=" << (best.p_hat * 100.0) << "%, "
                      << "CI_hi=" << (best.ci_hi * 100.0) << "%, "
                      << "n=" << best.n << "\n\n";
        } else {
            std::cout << "  BEST: (no config met target within cap)\n\n";
        }
    }

    csv.close();
    std::cout << "Wrote " << csv_path << "\n";
    return 0;
}

