/**
 * @file run_future_sl_experiments.cpp
 * @brief Run rollouts for SHMPC baseline and Future SL extensions; write CSV for analysis.
 *
 * Methods: SHMPC, SHMPC_DRO, SHMPC_AdaptiveDRO, SHMPC_RTA.
 * Compare efficacy (collision rate, missed mode rate, progress, solve time) vs SHMPC.
 *
 * Build from cpp_mpc: add this source and link scenario_mpc. Run from cpp_mpc/build
 * or pass output dir. Output: future_sl_mpc/experiments/results/future_sl_rollouts.csv
 */

#include "experiment_harness.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    std::string out_dir = "../future_sl_mpc/experiments/results/";  // when run from cpp_mpc/build
    if (argc >= 2) out_dir = argv[1];
    int num_rollouts = 100;
    if (argc >= 3) num_rollouts = std::atoi(argv[2]);

    scenario_mpc::ExperimentConfig config;
    config.num_scenarios = 30;
    config.eps_wass = 0.1;
    config.sigma_scale = 1.0;
    config.horizon = 20;
    config.num_discs = 3;
    config.safe_horizon_enabled = true;
    config.switch_prob = 0.1;
    config.rollout_steps = 80;
    config.rare_switch_prob = 0.05;
    config.obs_modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
    config.rare_mode = "lane_change_left";

    fs::create_directories(out_dir);
    std::string csv_path = out_dir + "future_sl_rollouts.csv";
    scenario_mpc::CSVWriter writer(csv_path);
    writer.write_header();

    std::vector<std::string> methods = {
        "SHMPC", "SHMPC_DRO", "SHMPC_AdaptiveDRO", "SHMPC_RTA",
        "SHMPC_Conformal", "SHMPC_Hazard", "SHMPC_Bandit", "SHMPC_Certificate", "SHMPC_Compiler",
        "CertificateFirst", "ScenarioCompiler"
    };
    unsigned seed_base = 12345u;

    std::cout << "Running " << num_rollouts << " rollouts per method (SHMPC baseline vs extensions)\n";
    for (const auto& method : methods) {
        for (int r = 0; r < num_rollouts; ++r) {
            unsigned seed = seed_base + static_cast<unsigned>(r);
            auto rec = scenario_mpc::run_experiment_rollout_future_sl(config, seed, method);
            writer.write_record(rec);
            if ((r + 1) % 25 == 0)
                std::cout << "  " << method << " " << (r + 1) << "/" << num_rollouts << "\n";
        }
    }
    writer.flush();
    std::cout << "Wrote " << csv_path << "\n";
    return 0;
}
