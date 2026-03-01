/**
 * @file test_scenario_compiler.cpp
 * @brief Tests for scenario compiler (Section 7.2).
 */

#include "scenario_compiler.hpp"
#include "types.hpp"
#include <cassert>
#include <iostream>

using namespace scenario_mpc;

int main() {
    ScenarioCompilerState state;
    state.witness_set = {};
    state.iterations = 0;
    std::vector<EgoState> ref(2);
    ref[0].x = 0; ref[0].y = 0;
    ref[1].x = 1; ref[1].y = 0;
    int call_count = 0;
    AdversaryFn adv = [&call_count](const std::vector<EgoState>&, const std::vector<Scenario>&) {
        call_count++;
        if (call_count <= 1) {
            Scenario s;
            s.scenario_id = 1;
            return std::optional<Scenario>(s);
        }
        return std::optional<Scenario>();
    };
    auto next = scenario_compiler_step(state, ref, adv, 5);
    assert(next.witness_set.size() == 1);
    assert(next.iterations == 1);
    auto next2 = scenario_compiler_step(next, ref, adv, 5);
    assert(next2.certificate_valid);
    std::cout << "Scenario compiler tests passed.\n";
    return 0;
}
