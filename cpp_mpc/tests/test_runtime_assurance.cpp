/**
 * @file test_runtime_assurance.cpp
 * @brief Tests for runtime assurance wrapper (Section 7.4).
 */

#include "runtime_assurance.hpp"
#include <cassert>
#include <iostream>

using namespace scenario_mpc;

int main() {
    auto monitor = [](const EgoState&, const std::optional<double>&) { return true; };
    EgoInput fallback(0.0, 0.0);
    RuntimeAssurance rta(monitor, fallback);
    EgoInput learned(1.0, 0.1);
    EgoInput out = rta.apply(EgoState(), learned, std::nullopt);
    assert(out.a == learned.a && out.delta == learned.delta);

    auto unsafe_monitor = [](const EgoState&, const std::optional<double>&) { return false; };
    RuntimeAssurance rta_unsafe(unsafe_monitor, fallback);
    out = rta_unsafe.apply(EgoState(), learned, std::nullopt);
    assert(out.a == fallback.a && out.delta == fallback.delta);
    std::cout << "Runtime assurance tests passed.\n";
    return 0;
}
