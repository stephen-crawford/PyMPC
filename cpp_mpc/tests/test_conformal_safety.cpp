/**
 * @file test_conformal_safety.cpp
 * @brief Tests for conformal safety wrappers (Section 6.1).
 */

#include "conformal_safety.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace scenario_mpc;

static void test_conformal_quantile() {
    std::vector<double> residuals = {0.1, 0.5, 1.0, 1.5, 2.0};
    double q = conformal_quantile(residuals, 0.1);
    assert(q >= 0.0 && q <= 2.0 + 1e-6);
    double q_hi = conformal_quantile(residuals, 0.9);
    assert(q_hi >= q);
    std::cout << "  conformal_quantile: q_0.1=" << q << " q_0.9=" << q_hi << "\n";
}

static void test_quantiles_per_mode() {
    std::map<std::string, std::vector<double>> per_mode;
    per_mode["A"] = {0.2, 0.4, 0.6};
    per_mode["B"] = {1.0, 2.0, 3.0};
    auto qm = conformal_quantiles_per_mode(per_mode, 0.2);
    assert(qm.count("A") && qm.count("B"));
    assert(qm.at("B") >= qm.at("A"));
    std::cout << "  conformal_quantiles_per_mode: q_A=" << qm.at("A") << " q_B=" << qm.at("B") << "\n";
}

static void test_boundary_scores() {
    std::map<std::string, std::vector<double>> val_at_mean, grad_norm;
    std::map<std::string, double> radius;
    std::vector<std::string> ids = {"m1", "m2"};
    val_at_mean["m1"] = {0.5, 1.0};
    val_at_mean["m2"] = {-0.5, 2.0};
    grad_norm["m1"] = {0.1, 0.2};
    grad_norm["m2"] = {0.5, 0.5};
    radius["m1"] = 0.5;
    radius["m2"] = 1.0;
    auto scores = boundary_scores(val_at_mean, grad_norm, radius, ids);
    assert(scores.at("m2") > scores.at("m1"));
    std::cout << "  boundary_scores: s_m1=" << scores.at("m1") << " s_m2=" << scores.at("m2") << "\n";
}

static void test_allocate_scenarios() {
    std::map<std::string, double> scores;
    scores["rare"] = 2.0;
    scores["common"] = 0.5;
    auto alloc = allocate_scenarios_boundary_aware(scores, 10, 1.0);
    int sum = 0;
    for (const auto& [_, Sm] : alloc) sum += Sm;
    assert(sum == 10);
    assert(alloc.at("rare") >= alloc.at("common"));
    std::cout << "  allocate_scenarios: rare=" << alloc.at("rare") << " common=" << alloc.at("common") << "\n";
}

int main() {
    std::cout << "Conformal safety tests\n";
    test_conformal_quantile();
    test_quantiles_per_mode();
    test_boundary_scores();
    test_allocate_scenarios();
    std::cout << "All conformal safety tests passed.\n";
    return 0;
}
