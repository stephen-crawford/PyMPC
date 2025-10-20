# Scenario Constraints Rework Plan

## Problem Analysis

The current Python implementation of scenario-based MPC constraints is fundamentally incompatible with the C++ reference implementation from https://github.com/oscardegroot/scenario_module.

### Current Issues

1. **Incorrect constraint generation**: Python version tries to create halfspace constraints for EVERY scenario at EVERY timestep, leading to:
   - Massive number of constraints (scenarios × timesteps × discs)
   - Infeasible optimization problems
   - "list index out of range" errors when accessing constraint arrays

2. **Missing Safe Horizon logic**: The C++ version uses a "safe horizon" approach where:
   - Only feasible scenarios are considered
   - Constraints are only added when necessary
   - Polytopes are constructed around feasible regions

3. **Data structure mismatches**: The scenario sample structure doesn't match what the constraint generation expects

## C++ Implementation Architecture (from reference)

Based on https://github.com/oscardegroot/scenario_module, the correct flow is:

```
1. ScenarioSampler generates samples:
   samples[timestep][obstacle_id][dimension][sample_id]
   
2. SafeHorizon processes each timestep:
   - compute_distances(): Calculate distance from robot to each scenario sample
   - check_feasibility_by_distance(): Mark scenarios as feasible/infeasible
   - compute_halfspaces(): For infeasible scenarios, compute separating hyperplanes
   - construct_polytopes(): Build convex polytopes around feasible regions
   
3. ScenarioModule coordinates multiple discs:
   - Runs scenario optimization for each disc
   - Selects best polytope configuration
   - Returns constraint coefficients (a1, a2, b) for MPC
   
4. ScenarioConstraints applies constraints in MPC:
   - Only adds constraints where polytopes exist
   - Uses dummy constraints (100.0) where no constraints needed
```

## Required Changes

### Phase 1: Fix Data Structures
- [ ] Ensure sample structure is: `[timestep][obstacle_id][0/1][sample_id]` where 0=x, 1=y
- [ ] Fix SafeHorizon to correctly access this structure
- [ ] Add proper bounds checking everywhere

### Phase 2: Implement Proper Feasibility Checking
- [ ] Implement distance-based feasibility (scenarios far from robot are feasible)
- [ ] Only process infeasible scenarios for constraint generation
- [ ] Use risk parameter (e.g., 5% of scenarios can be infeasible)

### Phase 3: Implement Polytope Construction  
- [ ] Build convex polytopes around feasible regions
- [ ] Generate minimal set of halfspace constraints
- [ ] Handle cases where no polytope exists (use dummy constraints)

### Phase 4: Fix Constraint Application
- [ ] Only add constraints where polytopes exist
- [ ] Use large dummy values (100.0) for unconstrained regions
- [ ] Properly size constraint arrays

### Phase 5: Test Incrementally
- [ ] Test with no obstacles (should work like basic MPC)
- [ ] Test with static obstacles
- [ ] Test with dynamic obstacles
- [ ] Test full integration with contouring

## Implementation Strategy

Since this is a major rework, I'll implement it in stages:

1. **Temporarily disable scenario constraints** - Get contouring + goal reaching working first
2. **Create minimal scenario constraints** - Simple distance-based constraints only
3. **Add polytope logic** - Implement full scenario-based optimization
4. **Integrate and test** - Verify everything works together

## References

- C++ Scenario Module: https://github.com/oscardegroot/scenario_module
- C++ MPC Planner: https://github.com/tud-amr/mpc_planner  
- ROS Tools: https://github.com/oscardegroot/ros_tools
- DecompUtil: https://github.com/oscardegroot/DecompUtil

## Next Steps

1. Create a simplified scenario constraints module that uses distance-based constraints only
2. Fix the test to work without full scenario optimization
3. Gradually add back the complex polytope logic

This will be an iterative process, but following the C++ architecture closely will ensure correctness.

