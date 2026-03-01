/**
 * @file test_adaptive_dro_shift.cpp
 * @brief Tests for adaptive DRO radius from shift detection (Section 6.5).
 */

#include "adaptive_dro_shift.hpp"
#include <cassert>
#include <iostream>

using namespace scenario_mpc;

int main() {
    AdaptiveDROShift shift(0.01, 1.0, 0.5, 10);
    for (int i = 0; i < 5; ++i) shift.push_residual(0.3);
    double rho_low = shift.get_rho();
    assert(rho_low <= 0.02);  // below baseline -> near rho_min
    for (int i = 0; i < 10; ++i) shift.push_residual(1.0);
    double rho_high = shift.get_rho();
    assert(rho_high > rho_low);
    std::cout << "Adaptive DRO: rho_low=" << rho_low << " rho_high=" << rho_high << "\n";
    std::cout << "All adaptive DRO shift tests passed.\n";
    return 0;
}
