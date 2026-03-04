/**
 * Tests for GAN scenario loader efficiency: cache (load once, materialize many).
 */
#include "gan_scenario_loader.hpp"
#include "types.hpp"
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

using namespace scenario_mpc;

#define ASSERT_TRUE(x) do { if (!(x)) { std::cerr << "FAIL: " #x << std::endl; return 1; } } while(0)

static int write_mini_gan_csv(const std::string& path, int horizon, int num_scenarios) {
    std::ofstream f(path);
    if (!f) return -1;
    f << "scenario_id,obstacle_id,k,dx,dy\n";
    for (int s = 0; s < num_scenarios; ++s) {
        for (int k = 0; k <= horizon; ++k) {
            double dx = 0.1 * s + 0.01 * k;
            double dy = 0.05 * s - 0.01 * k;
            f << s << ",0," << k << "," << dx << "," << dy << "\n";
        }
    }
    return 0;
}

int main() {
    const int horizon = 5;
    const int num_scenarios = 4;
    const std::string csv_path = "test_gan_cache_mini.csv";
    if (write_mini_gan_csv(csv_path, horizon, num_scenarios) != 0) {
        std::cerr << "Could not write test CSV\n";
        return 1;
    }

    std::map<int, ObstacleState> obs1;
    obs1[0] = ObstacleState(3.0, 0.5, 0.0, 0.0);
    std::map<int, ObstacleState> obs2;
    obs2[0] = ObstacleState(5.0, -0.3, 0.0, 0.0);

    // Load via legacy API
    auto from_file = load_scenarios_from_gan_csv(csv_path, obs1, horizon);
    ASSERT_TRUE(from_file.size() == static_cast<size_t>(num_scenarios));

    // Cache: load once
    GANScenarioCache cache;
    ASSERT_TRUE(cache.load(csv_path, horizon));
    ASSERT_TRUE(!cache.waypoints.empty());

    // Materialize with obs1
    auto mat1 = cache.materialize(obs1, horizon, num_scenarios);
    ASSERT_TRUE(mat1.size() == static_cast<size_t>(num_scenarios));
    ASSERT_TRUE(mat1[0].trajectories.count(0) == 1);
    double x0_1 = mat1[0].get_obstacle_position_at(0, 0).x();
    ASSERT_TRUE(std::abs(x0_1 - (3.0 + 0.0)) < 1e-6);  // obs1.x + dx

    // Materialize with obs2: positions should differ
    auto mat2 = cache.materialize(obs2, horizon, num_scenarios);
    ASSERT_TRUE(mat2.size() == static_cast<size_t>(num_scenarios));
    double x0_2 = mat2[0].get_obstacle_position_at(0, 0).x();
    ASSERT_TRUE(std::abs(x0_2 - 5.0) < 1e-6);
    ASSERT_TRUE(std::abs(x0_2 - x0_1) > 0.5);

    // Materialize with max_scenarios = 2
    auto mat_small = cache.materialize(obs1, horizon, 2);
    ASSERT_TRUE(mat_small.size() == 2u);

    std::cout << "GAN cache tests passed (load once, materialize many; max_scenarios cap)." << std::endl;
    return 0;
}
