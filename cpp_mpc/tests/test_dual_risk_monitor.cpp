/**
 * @file test_dual_risk_monitor.cpp
 * @brief Tests for dual-variable risk monitor (Section 6.8).
 */

#include "dual_risk_monitor.hpp"
#include <cassert>
#include <iostream>

using namespace scenario_mpc;

int main() {
    DualRiskMonitor monitor(0.0, 1.0, 0.1, 0.5);
    MPCTraceFeatures loose = {{1.0, 1.0}, {0.0, 0.0}, 10};
    MPCTraceFeatures tight = {{-0.5, 0.1}, {2.0, 1.0}, 100};
    double r_loose = monitor.predict_risk(loose);
    double r_tight = monitor.predict_risk(tight);
    assert(r_tight > r_loose);
    assert(!monitor.should_trigger_intervention(loose) || r_loose > 0.5);
    assert(monitor.should_trigger_intervention(tight));
    std::cout << "Dual risk: r_loose=" << r_loose << " r_tight=" << r_tight << "\n";
    std::cout << "All dual risk monitor tests passed.\n";
    return 0;
}
