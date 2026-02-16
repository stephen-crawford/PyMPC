/**
 * @file experiment_runner.cpp
 * @brief Orchestrates all experiments A-K for the results-strengthening upgrade.
 *
 * Each experiment writes per-rollout CSV data to strengthened_results/.
 * Python analysis (generate_strengthened_figures.py) reads these CSVs.
 *
 * Usage: ./experiment_runner [experiment_letter]
 *   No args: run all experiments
 *   "A" through "K": run a single experiment
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <map>
#include <cassert>
#include <filesystem>
#include <functional>

#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include "dynamics.hpp"
#include "wasserstein_dro.hpp"

using namespace scenario_mpc;
namespace fs = std::filesystem;

static const std::string OUTPUT_DIR = "strengthened_results/";

static void ensure_output_dir() {
    fs::create_directories(OUTPUT_DIR);
}

// ============================================================================
// Experiment A+G: Unified per-rollout logging (Base vs DRO at S=20)
// ============================================================================

static void run_experiment_AG() {
    std::cout << "\n============================================================\n"
              << "  Experiment A+G: Unified Per-Rollout Logging\n"
              << "  Base vs DRO at S=20, 200 rollouts each\n"
              << "============================================================\n";

    CSVWriter csv(OUTPUT_DIR + "all_rollouts.csv");
    csv.write_header();

    constexpr int N_ROLLOUTS = 200;
    constexpr int S = 20;

    for (auto ablation : {AblationVariant::NO_INJECTION, AblationVariant::DRO_FULL}) {
        std::string label = ablation_variant_name(ablation);
        int collisions = 0;

        for (int r = 0; r < N_ROLLOUTS; ++r) {
            ExperimentConfig cfg;
            cfg.num_scenarios = S;
            cfg.ablation = ablation;
            unsigned seed = static_cast<unsigned>(r * 1111 + 42);

            auto rec = run_experiment_rollout(cfg, seed);
            csv.write_record(rec);
            if (rec.collision) collisions++;

            if ((r + 1) % 50 == 0) {
                std::cout << "  " << label << ": " << (r + 1) << "/" << N_ROLLOUTS
                          << " rollouts, collisions=" << collisions << std::endl;
            }
        }

        auto [lo, hi] = wilson_ci(collisions, N_ROLLOUTS);
        std::cout << "  " << label << " collision rate: "
                  << std::fixed << std::setprecision(3)
                  << static_cast<double>(collisions) / N_ROLLOUTS
                  << " [" << lo << ", " << hi << "]\n";
    }
    csv.flush();
    std::cout << "  -> " << OUTPUT_DIR << "all_rollouts.csv\n";
}

// ============================================================================
// Experiment B: Statistical tests (paired Base vs DRO)
// ============================================================================

static void run_experiment_B() {
    std::cout << "\n============================================================\n"
              << "  Experiment B: Statistical Tests (Paired)\n"
              << "  Bootstrap CIs, McNemar, effect sizes\n"
              << "============================================================\n";

    constexpr int N_ROLLOUTS = 200;
    constexpr int S = 20;

    std::vector<bool> base_collisions, dro_collisions;

    for (int r = 0; r < N_ROLLOUTS; ++r) {
        unsigned seed = static_cast<unsigned>(r * 1111 + 42);

        ExperimentConfig base_cfg;
        base_cfg.num_scenarios = S;
        base_cfg.ablation = AblationVariant::NO_INJECTION;
        auto base_rec = run_experiment_rollout(base_cfg, seed);
        base_collisions.push_back(base_rec.collision);

        ExperimentConfig dro_cfg;
        dro_cfg.num_scenarios = S;
        dro_cfg.ablation = AblationVariant::DRO_FULL;
        auto dro_rec = run_experiment_rollout(dro_cfg, seed);
        dro_collisions.push_back(dro_rec.collision);
    }

    // McNemar counts
    int b = 0, c = 0;  // b = base_coll & !dro_coll, c = !base_coll & dro_coll
    for (int i = 0; i < N_ROLLOUTS; ++i) {
        if (base_collisions[i] && !dro_collisions[i]) b++;
        if (!base_collisions[i] && dro_collisions[i]) c++;
    }

    double chi2 = mcnemar_chi2(b, c);
    int base_total = std::count(base_collisions.begin(), base_collisions.end(), true);
    int dro_total = std::count(dro_collisions.begin(), dro_collisions.end(), true);
    double p_base = static_cast<double>(base_total) / N_ROLLOUTS;
    double p_dro = static_cast<double>(dro_total) / N_ROLLOUTS;

    std::mt19937 boot_rng(12345);
    auto boot = bootstrap_paired_delta(base_collisions, dro_collisions, 10000, &boot_rng);
    auto es = compute_effect_sizes(p_base, p_dro);

    // Write summary CSV
    {
        std::ofstream ofs(OUTPUT_DIR + "stat_summary.csv");
        ofs << "metric,value\n"
            << "n_rollouts," << N_ROLLOUTS << "\n"
            << "base_collision_rate," << std::fixed << std::setprecision(4) << p_base << "\n"
            << "dro_collision_rate," << p_dro << "\n"
            << "mcnemar_b," << b << "\n"
            << "mcnemar_c," << c << "\n"
            << "mcnemar_chi2," << chi2 << "\n"
            << "mcnemar_significant," << (chi2 > 3.84 ? "yes" : "no") << "\n"
            << "abs_delta," << es.abs_delta << "\n"
            << "rel_delta," << es.rel_delta << "\n"
            << "risk_ratio," << es.risk_ratio << "\n"
            << "cohens_h," << es.cohens_h << "\n";
    }

    // Write bootstrap CI CSV
    {
        std::ofstream ofs(OUTPUT_DIR + "bootstrap_ci.csv");
        ofs << "mean_delta,ci_low,ci_high\n"
            << std::fixed << std::setprecision(4)
            << boot.mean_delta << "," << boot.ci_low << "," << boot.ci_high << "\n";
    }

    std::cout << "  Base collision rate: " << p_base
              << "  DRO collision rate: " << p_dro << "\n"
              << "  McNemar chi2=" << chi2 << " (sig: " << (chi2 > 3.84 ? "yes" : "no") << ")\n"
              << "  Bootstrap delta: " << boot.mean_delta
              << " [" << boot.ci_low << ", " << boot.ci_high << "]\n"
              << "  Cohen's h: " << es.cohens_h << "\n"
              << "  -> " << OUTPUT_DIR << "stat_summary.csv, bootstrap_ci.csv\n";
}

// ============================================================================
// Experiment C: Multi-seed stability
// ============================================================================

static void run_experiment_C() {
    std::cout << "\n============================================================\n"
              << "  Experiment C: Multi-Seed Stability\n"
              << "  30 seeds x 2 methods x S={10,20,40}\n"
              << "============================================================\n";

    CSVWriter csv(OUTPUT_DIR + "multi_seed.csv");
    csv.write_header();

    constexpr int N_SEEDS = 30;
    constexpr int ROLLOUTS_PER_SEED = 6;  // 3 S values x 2 methods
    std::vector<int> S_values = {10, 20, 40};

    for (int seed_idx = 0; seed_idx < N_SEEDS; ++seed_idx) {
        unsigned master_seed = static_cast<unsigned>(seed_idx * 7919 + 100);
        for (int S : S_values) {
            for (auto ablation : {AblationVariant::NO_INJECTION, AblationVariant::DRO_FULL}) {
                ExperimentConfig cfg;
                cfg.num_scenarios = S;
                cfg.ablation = ablation;
                auto seeds = derive_seeds(master_seed, seed_idx);

                auto rec = run_experiment_rollout(cfg, seeds.env);
                csv.write_record(rec);
            }
        }
        if ((seed_idx + 1) % 10 == 0) {
            std::cout << "  Seed " << (seed_idx + 1) << "/" << N_SEEDS << " done\n";
        }
    }
    csv.flush();
    std::cout << "  -> " << OUTPUT_DIR << "multi_seed.csv\n";
}

// ============================================================================
// Experiment D: Distribution shift stress test
// ============================================================================

static void run_experiment_D() {
    std::cout << "\n============================================================\n"
              << "  Experiment D: Distribution Shift Stress Test\n"
              << "  rho={0,.05,.1,.2,.3} x boost={0,.05,.1,.2} x 100 rollouts\n"
              << "============================================================\n";

    CSVWriter csv(OUTPUT_DIR + "shift_sweep.csv");
    csv.write_header();

    std::vector<double> rho_values = {0.0, 0.05, 0.1, 0.2, 0.3};
    std::vector<double> boost_values = {0.0, 0.05, 0.1, 0.2};
    constexpr int N_ROLLOUTS = 100;

    for (double rho : rho_values) {
        for (double boost : boost_values) {
            int dro_collisions = 0;

            for (int r = 0; r < N_ROLLOUTS; ++r) {
                ExperimentConfig cfg;
                cfg.num_scenarios = 20;
                cfg.ablation = AblationVariant::DRO_FULL;
                cfg.shift.rho = rho;
                cfg.shift.dangerous_boost = boost;

                unsigned seed = static_cast<unsigned>(r * 1111 + static_cast<unsigned>(rho * 10000) + static_cast<unsigned>(boost * 1000));
                auto rec = run_experiment_rollout(cfg, seed);
                csv.write_record(rec);
                if (rec.collision) dro_collisions++;
            }

            std::cout << "  rho=" << std::setw(4) << rho
                      << " boost=" << std::setw(4) << boost
                      << " collisions=" << dro_collisions << "/" << N_ROLLOUTS << "\n";
        }
    }
    csv.flush();
    std::cout << "  -> " << OUTPUT_DIR << "shift_sweep.csv\n";
}

// ============================================================================
// Experiment E: Ablation matrix (6 variants x S={10,20,40})
// ============================================================================

static void run_experiment_E() {
    std::cout << "\n============================================================\n"
              << "  Experiment E: Ablation Matrix\n"
              << "  6 variants x S={10,20,40} x 200 rollouts\n"
              << "============================================================\n";

    CSVWriter csv(OUTPUT_DIR + "ablation_matrix.csv");
    csv.write_header();

    std::vector<AblationVariant> variants = {
        AblationVariant::NO_INJECTION,
        AblationVariant::DRO_FULL,
        AblationVariant::DRO_NO_COV,
        AblationVariant::DRO_DISTANCE_ONLY,
        AblationVariant::RANDOM_INJECTION,
        AblationVariant::ALWAYS_INJECT
    };
    std::vector<int> S_values = {10, 20, 40};
    constexpr int N_ROLLOUTS = 200;

    for (int S : S_values) {
        for (auto variant : variants) {
            std::string label = ablation_variant_name(variant);
            int collisions = 0;

            for (int r = 0; r < N_ROLLOUTS; ++r) {
                ExperimentConfig cfg;
                cfg.num_scenarios = S;
                cfg.ablation = variant;
                unsigned seed = static_cast<unsigned>(r * 1111 + S * 10);

                auto rec = run_experiment_rollout(cfg, seed);
                csv.write_record(rec);
                if (rec.collision) collisions++;
            }

            auto [lo, hi] = wilson_ci(collisions, N_ROLLOUTS);
            std::cout << "  S=" << std::setw(3) << S
                      << " " << std::setw(20) << label
                      << " collisions=" << collisions << "/" << N_ROLLOUTS
                      << " [" << std::fixed << std::setprecision(3) << lo << "," << hi << "]\n";
        }
    }
    csv.flush();
    std::cout << "  -> " << OUTPUT_DIR << "ablation_matrix.csv\n";
}

// ============================================================================
// Experiment F: Hyperparameter sensitivity
// ============================================================================

static void run_experiment_F() {
    std::cout << "\n============================================================\n"
              << "  Experiment F: Hyperparameter Sensitivity\n"
              << "  eps={0,.01,.05,.1,.2,.5} x sigma={0,.5,1,2} x 100 rollouts\n"
              << "============================================================\n";

    CSVWriter csv(OUTPUT_DIR + "hyperparam_sweep.csv");
    csv.write_header();

    std::vector<double> eps_values = {0.0, 0.01, 0.05, 0.1, 0.2, 0.5};
    std::vector<double> sigma_values = {0.0, 0.5, 1.0, 2.0};
    constexpr int N_ROLLOUTS = 100;

    for (double eps : eps_values) {
        for (double sigma : sigma_values) {
            int collisions = 0;

            for (int r = 0; r < N_ROLLOUTS; ++r) {
                ExperimentConfig cfg;
                cfg.num_scenarios = 20;
                cfg.ablation = (eps > 0.0) ? AblationVariant::DRO_FULL : AblationVariant::NO_INJECTION;
                cfg.eps_wass = eps;
                cfg.sigma_scale = sigma;
                unsigned seed = static_cast<unsigned>(r * 1111 + static_cast<unsigned>(eps * 10000) + static_cast<unsigned>(sigma * 1000));

                auto rec = run_experiment_rollout(cfg, seed);
                csv.write_record(rec);
                if (rec.collision) collisions++;
            }

            std::cout << "  eps=" << std::setw(5) << eps
                      << " sigma=" << std::setw(4) << sigma
                      << " collisions=" << collisions << "/" << N_ROLLOUTS << "\n";
        }
    }
    csv.flush();
    std::cout << "  -> " << OUTPUT_DIR << "hyperparam_sweep.csv\n";
}

// ============================================================================
// Experiment H: Solve time analysis
// ============================================================================

static void run_experiment_H() {
    std::cout << "\n============================================================\n"
              << "  Experiment H: Solve Time Analysis\n"
              << "  Base vs DRO at S=20, 200 rollouts\n"
              << "============================================================\n";

    CSVWriter csv(OUTPUT_DIR + "solve_time_cdf.csv");
    csv.write_header();

    constexpr int N_ROLLOUTS = 200;

    for (auto ablation : {AblationVariant::NO_INJECTION, AblationVariant::DRO_FULL}) {
        for (int r = 0; r < N_ROLLOUTS; ++r) {
            ExperimentConfig cfg;
            cfg.num_scenarios = 20;
            cfg.ablation = ablation;
            unsigned seed = static_cast<unsigned>(r * 1111 + 42);

            auto rec = run_experiment_rollout(cfg, seed);
            csv.write_record(rec);
        }
        std::cout << "  " << ablation_variant_name(ablation) << " done\n";
    }
    csv.flush();
    std::cout << "  -> " << OUTPUT_DIR << "solve_time_cdf.csv\n";
}

// ============================================================================
// Experiment I: Hard / qualitative cases
// ============================================================================

static void run_experiment_I() {
    std::cout << "\n============================================================\n"
              << "  Experiment I: Hard Qualitative Cases\n"
              << "  3 curated scenarios x 2 methods\n"
              << "============================================================\n";

    struct HardCase {
        std::string name;
        unsigned seed;
        std::string forced_mode;
        int force_at_step;
        double switch_prob;
    };

    std::vector<HardCase> cases = {
        // Late switch: CV for 40 steps, forced turn_left at step 41
        {"late_switch", 42, "turn_left", 41, 0.0},
        // Cut-in: obstacle in lane_change_left heading toward ego path
        {"cut_in", 99, "lane_change_left", 0, 0.0},
        // Near-miss: oscillating CV/turn_right every 10 steps
        {"near_miss", 77, "", -1, 0.1}
    };

    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const auto& hc = cases[ci];
        std::string fname = OUTPUT_DIR + "qualitative_case_" + std::to_string(ci + 1) + ".csv";
        CSVWriter csv(fname);
        csv.write_header();

        for (auto ablation : {AblationVariant::NO_INJECTION, AblationVariant::DRO_FULL}) {
            ExperimentConfig cfg;
            cfg.num_scenarios = 20;
            cfg.ablation = ablation;
            cfg.rollout_steps = 60;
            cfg.switch_prob = hc.switch_prob;

            auto rec = run_experiment_rollout(cfg, hc.seed);
            csv.write_record(rec);

            std::cout << "  Case " << hc.name << " " << ablation_variant_name(ablation)
                      << ": collision=" << rec.collision
                      << " clearance=" << std::setprecision(3) << rec.min_clearance << "\n";
        }
        csv.flush();
    }
    std::cout << "  -> " << OUTPUT_DIR << "qualitative_case_{1,2,3}.csv\n";
}

// ============================================================================
// Experiment J: Ground cost alternatives
// ============================================================================

static void run_experiment_J() {
    std::cout << "\n============================================================\n"
              << "  Experiment J: Ground Cost Alternatives\n"
              << "  3 cost types x S=20 x 200 rollouts\n"
              << "============================================================\n";

    CSVWriter csv(OUTPUT_DIR + "ground_cost.csv");
    csv.write_header();

    std::vector<DROGroundCostType> cost_types = {
        DROGroundCostType::W2_BURES,
        DROGroundCostType::ZERO_ONE,
        DROGroundCostType::EUCLIDEAN_MEAN
    };
    constexpr int N_ROLLOUTS = 200;

    for (auto cost_type : cost_types) {
        int collisions = 0;

        for (int r = 0; r < N_ROLLOUTS; ++r) {
            ExperimentConfig cfg;
            cfg.num_scenarios = 20;
            cfg.ablation = AblationVariant::DRO_FULL;
            cfg.ground_cost = cost_type;
            unsigned seed = static_cast<unsigned>(r * 1111 + 42);

            auto rec = run_experiment_rollout(cfg, seed);
            csv.write_record(rec);
            if (rec.collision) collisions++;
        }

        auto [lo, hi] = wilson_ci(collisions, N_ROLLOUTS);
        std::cout << "  " << std::setw(15) << ground_cost_name(cost_type)
                  << " collisions=" << collisions << "/" << N_ROLLOUTS
                  << " [" << std::fixed << std::setprecision(3) << lo << "," << hi << "]\n";
    }
    csv.flush();
    std::cout << "  -> " << OUTPUT_DIR << "ground_cost.csv\n";
}

// ============================================================================
// Experiment K: Calibration (empirical violation vs target epsilon)
// ============================================================================

static void run_experiment_K() {
    std::cout << "\n============================================================\n"
              << "  Experiment K: Calibration\n"
              << "  eps={.01,.03,.05,.1,.15,.2,.3,.5} x 500 rollouts\n"
              << "============================================================\n";

    std::ofstream csv(OUTPUT_DIR + "calibration.csv");
    csv << "target_eps,empirical_violation,n_rollouts,collisions,ci_low,ci_high\n";

    std::vector<double> eps_targets = {0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5};
    constexpr int N_ROLLOUTS = 500;

    for (double eps : eps_targets) {
        int collisions = 0;

        for (int r = 0; r < N_ROLLOUTS; ++r) {
            ExperimentConfig cfg;
            cfg.num_scenarios = 20;
            cfg.ablation = AblationVariant::DRO_FULL;
            cfg.eps_wass = eps;
            unsigned seed = static_cast<unsigned>(r * 1111 + static_cast<unsigned>(eps * 100000));

            auto rec = run_experiment_rollout(cfg, seed);
            if (rec.collision) collisions++;
        }

        double empirical = static_cast<double>(collisions) / N_ROLLOUTS;
        auto [lo, hi] = wilson_ci(collisions, N_ROLLOUTS);

        csv << std::fixed << std::setprecision(4)
            << eps << "," << empirical << "," << N_ROLLOUTS << ","
            << collisions << "," << lo << "," << hi << "\n";

        std::cout << "  eps=" << std::setw(5) << eps
                  << " empirical=" << std::setprecision(4) << empirical
                  << " [" << lo << "," << hi << "]\n";
    }
    std::cout << "  -> " << OUTPUT_DIR << "calibration.csv\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    ensure_output_dir();

    auto start = std::chrono::high_resolution_clock::now();

    std::string filter = "";
    if (argc > 1) filter = argv[1];

    auto should_run = [&](const std::string& label) {
        return filter.empty() || filter == label;
    };

    if (should_run("A") || should_run("G")) run_experiment_AG();
    if (should_run("B")) run_experiment_B();
    if (should_run("C")) run_experiment_C();
    if (should_run("D")) run_experiment_D();
    if (should_run("E")) run_experiment_E();
    if (should_run("F")) run_experiment_F();
    if (should_run("H")) run_experiment_H();
    if (should_run("I")) run_experiment_I();
    if (should_run("J")) run_experiment_J();
    if (should_run("K")) run_experiment_K();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "\n============================================================\n"
              << "  All experiments complete. Elapsed: "
              << std::fixed << std::setprecision(1) << elapsed.count() << "s\n"
              << "  Results in: " << OUTPUT_DIR << "\n"
              << "============================================================\n";

    return 0;
}
