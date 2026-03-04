/**
 * Sweep GAN scenario count to find config meeting:
 *   - >= 50% collision improvement vs SHMPC
 *   - Solve time < 10 ms (real-time for self-driving: sub-20ms target, 10ms margin)
 *
 * Usage: ./run_gan_realtime_sweep <output_dir> <rollouts_per_config>
 * Output: gan_realtime_sweep.csv + summary to stdout
 */
#include "experiment_harness.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    std::string out_dir = "../future_sl_mpc/experiments/results/";
    if (argc >= 2) out_dir = argv[1];
    int rollouts = 40;
    if (argc >= 3) rollouts = std::atoi(argv[2]);

    fs::create_directories(out_dir);
    std::string gan_csv = out_dir + "gan_scenarios.csv";
    scenario_mpc::ExperimentConfig base;
    base.num_scenarios = 30;
    base.horizon = 20;
    base.rollout_steps = 80;
    base.gan_scenario_csv_path = gan_csv;
    base.use_gan_cache = true;

    const std::vector<int> scenario_counts = {8, 10, 12, 15, 18, 20, 25, 30};
    constexpr double TARGET_IMPROVE = 0.50;   // 50% relative improvement
    constexpr double TARGET_SOLVE_MS = 10.0; // real-time: sub-10ms for margin

    std::ofstream csv(out_dir + "gan_realtime_sweep.csv");
    csv << "S,collision_rate,collision_count,n,solve_ms_mean,solve_ms_p95,rel_improve_pct,meets_safety,meets_realtime\n";

    double shmpc_collision = 0.0;
    double shmpc_solve = 0.0;
    int shmpc_n = 0;

    // First: run SHMPC baseline
    std::cout << "Running SHMPC baseline (" << rollouts << " rollouts)...\n";
    for (int r = 0; r < rollouts; ++r) {
        auto rec = scenario_mpc::run_experiment_rollout_future_sl(base, 10000u + r, "SHMPC");
        shmpc_collision += rec.collision ? 1.0 : 0.0;
        shmpc_solve += rec.avg_solve_ms;
        shmpc_n++;
    }
    shmpc_collision /= shmpc_n;
    shmpc_solve /= shmpc_n;
    std::cout << "  SHMPC: collision=" << (shmpc_collision * 100) << "%, solve=" << shmpc_solve << " ms\n\n";

    int best_S = -1;
    double best_improve = 0;
    double best_solve = 1e9;

    for (int S : scenario_counts) {
        base.gan_num_scenarios_override = S;
        int collisions = 0;
        double solve_sum = 0;
        std::vector<double> solve_times;
        for (int r = 0; r < rollouts; ++r) {
            auto rec = scenario_mpc::run_experiment_rollout_future_sl(base, 20000u + S * 1000u + r, "SHMPC_GAN");
            if (rec.collision) collisions++;
            solve_sum += rec.avg_solve_ms;
            solve_times.push_back(rec.avg_solve_ms);
        }
        double rate = static_cast<double>(collisions) / rollouts;
        double solve_mean = solve_sum / rollouts;
        std::sort(solve_times.begin(), solve_times.end());
        double solve_p95 = solve_times.empty() ? 0 : solve_times[static_cast<size_t>(0.95 * solve_times.size())];
        double rel_improve = (shmpc_collision - rate) / (shmpc_collision > 1e-9 ? shmpc_collision : 1e-9);
        bool meets_safety = rel_improve >= TARGET_IMPROVE;
        bool meets_realtime = solve_mean < TARGET_SOLVE_MS;

        csv << S << "," << rate << "," << collisions << "," << rollouts << ","
            << solve_mean << "," << solve_p95 << "," << (rel_improve * 100) << ","
            << (meets_safety ? 1 : 0) << "," << (meets_realtime ? 1 : 0) << "\n";

        std::cout << "S=" << S << ": collision=" << (rate * 100) << "%, solve=" << solve_mean << " ms, "
                  << "rel_improve=" << (rel_improve * 100) << "%, "
                  << (meets_safety ? "SAFE " : "     ") << (meets_realtime ? "RT" : "  ") << "\n";

        if (meets_safety && meets_realtime && (best_S < 0 || S < best_S)) {
            best_S = S;
            best_improve = rel_improve;
            best_solve = solve_mean;
        }
    }

    csv.close();
    std::cout << "\n--- Summary ---\n";
    if (best_S >= 0) {
        std::cout << "Recommended: S=" << best_S << " (rel_improve=" << (best_improve * 100)
                  << "%, solve=" << best_solve << " ms)\n";
        std::cout << "Meets: >=50% collision improvement, <10ms solve (real-time)\n";
    } else {
        std::cout << "No config met both constraints. Try lower S for speed or higher S for safety.\n";
    }
    std::cout << "Wrote " << out_dir << "gan_realtime_sweep.csv\n";
    return 0;
}
