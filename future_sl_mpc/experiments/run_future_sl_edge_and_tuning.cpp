/**
 * @file run_future_sl_edge_and_tuning.cpp
 * @brief Run Future SL methods under edge-case scenarios and tuning sweeps.
 *
 * Edge cases: baseline, high_switch, rare_heavy, low_S, distribution_shift.
 * Tuning: certificate radius, RTA threshold, bandit beta.
 * Output: future_sl_rollouts_edge_tuning.csv (same schema with scenario column).
 *
 * Usage: run from cpp_mpc/build, or pass [out_dir] [rollouts_per_cell]
 *   rollouts_per_cell default 25.
 */

#include "experiment_harness.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <functional>
#include <filesystem>

namespace fs = std::filesystem;

static scenario_mpc::ExperimentConfig make_baseline() {
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
    c.scenario_tag = "baseline";
    return c;
}

int main(int argc, char** argv) {
    std::string out_dir = "../future_sl_mpc/experiments/results/";
    if (argc >= 2) out_dir = argv[1];
    int rollouts_per_cell = 25;
    if (argc >= 3) rollouts_per_cell = std::atoi(argv[2]);

    fs::create_directories(out_dir);
    std::string csv_path = out_dir + "future_sl_rollouts_edge_tuning.csv";
    scenario_mpc::CSVWriter writer(csv_path);
    writer.write_header();

    const std::vector<std::string> all_methods = {
        "SHMPC", "SHMPC_DRO", "SHMPC_AdaptiveDRO", "SHMPC_RTA",
        "SHMPC_Conformal", "SHMPC_Hazard", "SHMPC_Bandit", "SHMPC_Certificate", "SHMPC_Compiler",
        "CertificateFirst", "ScenarioCompiler", "SHMPC_GAN", "SHMPC_GAN_Reduced", "SHMPC_GAN_Quotient",
        "SHMPC_Reservoir", "SHMPC_SeekAvoid", "SHMPC_SeekAvoidML", "SHMPC_QuotientSpace", "SHMPC_DoubleDual"
    };
    unsigned seed_base = 40000u;

    // --- Edge-case scenarios ---
    struct ScenarioSpec {
        std::string tag;
        std::function<void(scenario_mpc::ExperimentConfig&)> apply;
    };
    std::vector<ScenarioSpec> scenarios = {
        {"baseline", [](scenario_mpc::ExperimentConfig& c) { c.scenario_tag = "baseline"; }},
        {"high_switch", [](scenario_mpc::ExperimentConfig& c) {
            c.scenario_tag = "high_switch";
            c.switch_prob = 0.25;
        }},
        {"rare_heavy", [](scenario_mpc::ExperimentConfig& c) {
            c.scenario_tag = "rare_heavy";
            c.rare_switch_prob = 0.15;
        }},
        {"low_S", [](scenario_mpc::ExperimentConfig& c) {
            c.scenario_tag = "low_S";
            c.num_scenarios = 12;
        }},
        {"distribution_shift", [](scenario_mpc::ExperimentConfig& c) {
            c.scenario_tag = "distribution_shift";
            c.shift.rho = 0.3;
            c.shift.dangerous_boost = 0.1;
        }}
    };

    std::cout << "Edge-case scenarios (" << rollouts_per_cell << " rollouts per method)\n";
    for (const auto& spec : scenarios) {
        auto config = make_baseline();
        spec.apply(config);
        config.gan_scenario_csv_path = out_dir + "gan_scenarios.csv";
        config.reservoir_scenario_csv_path = out_dir + "reservoir_scenarios.csv";
        config.seek_avoid_scenario_csv_path = out_dir + "seek_avoid_scenarios.csv";
        config.seek_avoid_ml_scenario_csv_path = out_dir + "seek_avoid_ml_scenarios.csv";
        for (const auto& method : all_methods) {
            for (int r = 0; r < rollouts_per_cell; ++r) {
                unsigned seed = seed_base + 1000u * static_cast<unsigned>(r) + static_cast<unsigned>(std::hash<std::string>{}(spec.tag + method));
                auto rec = scenario_mpc::run_experiment_rollout_future_sl(config, seed, method);
                writer.write_record(rec);
            }
            std::cout << "  " << spec.tag << " " << method << "\n";
        }
    }
    seed_base += 50000u;

    // --- Tuning: Certificate radius (SHMPC_Certificate, CertificateFirst) ---
    std::vector<double> cert_radii = {0.10, 0.15, 0.25};
    std::vector<std::string> cert_methods = {"SHMPC_Certificate", "CertificateFirst"};
    for (double radius : cert_radii) {
        auto config = make_baseline();
        config.scenario_tag = "tune_cert_" + std::to_string(static_cast<int>(radius * 100));  // 10, 15, 25
        config.certificate_radius_override = radius;
        for (const auto& method : cert_methods) {
            for (int r = 0; r < rollouts_per_cell; ++r) {
                unsigned seed = seed_base + 100u * r + static_cast<unsigned>(static_cast<int>(radius * 100));
                auto rec = scenario_mpc::run_experiment_rollout_future_sl(config, seed, method);
                writer.write_record(rec);
            }
            std::cout << "  tune_cert_" << radius << " " << method << "\n";
        }
    }
    seed_base += 10000u;

    // --- Tuning: RTA threshold (SHMPC_RTA) ---
    std::vector<double> rta_thresholds = {1.0, 1.5, 2.0};
    for (double thresh : rta_thresholds) {
        auto config = make_baseline();
        config.scenario_tag = "tune_rta_" + std::to_string(static_cast<int>(thresh * 10));
        config.rta_threshold_override = thresh;
        for (int r = 0; r < rollouts_per_cell; ++r) {
            unsigned seed = seed_base + 100u * r + static_cast<unsigned>(static_cast<int>(thresh * 10));
            auto rec = scenario_mpc::run_experiment_rollout_future_sl(config, seed, "SHMPC_RTA");
            writer.write_record(rec);
        }
        std::cout << "  tune_rta_" << thresh << " SHMPC_RTA\n";
    }
    seed_base += 10000u;

    // --- Tuning: Bandit beta (SHMPC_Bandit) ---
    std::vector<double> bandit_betas = {0.5, 1.0, 2.0};
    for (double beta : bandit_betas) {
        auto config = make_baseline();
        config.scenario_tag = "tune_bandit_" + std::to_string(static_cast<int>(beta * 10));
        config.bandit_beta_override = beta;
        for (int r = 0; r < rollouts_per_cell; ++r) {
            unsigned seed = seed_base + 100u * r + static_cast<unsigned>(static_cast<int>(beta * 10));
            auto rec = scenario_mpc::run_experiment_rollout_future_sl(config, seed, "SHMPC_Bandit");
            writer.write_record(rec);
        }
        std::cout << "  tune_bandit_" << beta << " SHMPC_Bandit\n";
    }

    writer.flush();
    std::cout << "Wrote " << csv_path << "\n";
    return 0;
}
