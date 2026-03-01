/**
 * @file test_hazard_switch_sampling.cpp
 * @brief Tests for hazard-triggered switch-aware sampling (Section 6.2).
 */

#include "hazard_switch_sampling.hpp"
#include <cassert>
#include <iostream>

using namespace scenario_mpc;

int main() {
    std::map<std::string, double> nominal = {{"A", 0.7}, {"B", 0.3}};
    std::map<std::string, double> hazard = {{"A", 0.1}, {"B", 0.9}};
    auto reweighted = reweight_by_hazard(nominal, hazard, 1.0);
    assert(reweighted.at("B") > reweighted.at("A"));
    auto alloc = allocate_with_hazard_trigger(reweighted, 10, 0.2, 0.5, 1);
    assert(alloc.size() == 2);
    int sum = 0;
    for (const auto& [_, Sm] : alloc) sum += Sm;
    assert(sum == 10);
    std::cout << "Hazard reweight: A=" << reweighted.at("A") << " B=" << reweighted.at("B")
              << "; alloc A=" << alloc.at("A") << " B=" << alloc.at("B") << "\n";

    HazardModel model;
    model.set_mode_coefficients({{"A", 1.0}, {"B", 2.0}});
    HazardFeatures h;
    h.time_since_last_switch = 2.0;
    assert(model.hazard("B", h) >= model.hazard("A", h));
    std::cout << "All hazard switch sampling tests passed.\n";
    return 0;
}
