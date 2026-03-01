/**
 * @file test_risk_directed_bandit.cpp
 * @brief Tests for risk-directed UCB bandit allocation (Section 6.3).
 */

#include "risk_directed_bandit.hpp"
#include <cassert>
#include <iostream>

using namespace scenario_mpc;

int main() {
    RiskDirectedBandit bandit(1.0);
    bandit.set_modes({"A", "B", "C"});
    for (int i = 0; i < 5; ++i) bandit.update(i, "A", false);
    for (int i = 0; i < 5; ++i) bandit.update(i, "B", true);
    assert(bandit.ucb("B") > bandit.ucb("A"));
    auto alloc = bandit.allocate(12);
    int sum = 0;
    for (const auto& [_, Sm] : alloc) sum += Sm;
    assert(sum == 12);
    std::cout << "UCB A=" << bandit.ucb("A") << " B=" << bandit.ucb("B")
              << "; alloc A=" << alloc["A"] << " B=" << alloc["B"] << "\n";
    std::cout << "All risk-directed bandit tests passed.\n";
    return 0;
}
