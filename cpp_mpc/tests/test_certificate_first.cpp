/**
 * @file test_certificate_first.cpp
 * @brief Tests for certificate-first schema (Section 7.1).
 */

#include "certificate_first.hpp"
#include <cassert>
#include <iostream>
#include <vector>

using namespace scenario_mpc;

int main() {
    std::vector<std::vector<double>> res_per_t = {{0.1, 0.2, 0.5}, {0.2, 0.3, 0.4}};
    auto radii = certificate_radii_from_residuals(res_per_t, 0.2);
    assert(radii.size() == 2);
    assert(radii[0] >= 0.0 && radii[1] >= 0.0);
    Certificate cert;
    cert.radii = radii;
    cert.delta = 0.2;
    double vol = certificate_volume(cert);
    assert(vol >= 0.0);
    double b_robust = tighten_offset(1.0, 1.0, 0.5);
    assert(b_robust == 0.5);
    std::cout << "Certificate-first tests passed.\n";
    return 0;
}
