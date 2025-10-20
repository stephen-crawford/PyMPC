# Scenario Constraints Rework - Handoff Summary

## Current Status

### ✅ COMPLETED (Phase 1):
1. **Stable Baseline Established**
   - Created `SimplifiedScenarioConstraints` module that prevents immediate crashes
   - Test now runs through first iteration successfully
   - MPC fails on second iteration (expected for simplified version)

2. **Comprehensive Analysis Documents Created**
   - `CPP_SCENARIO_MODULE_ANALYSIS.md`: Detailed architecture analysis
   - `SCENARIO_CONSTRAINTS_REWORK_PLAN.md`: 3-phase implementation plan
   - `REWORK_PROGRESS_SUMMARY.md`: Current progress tracking
   - `HANDOFF_SUMMARY.md`: This file

3. **Critical Bug Fixes**
   - Fixed data structure access in `SafeHorizon.compute_distances()` 
   - Fixed scenario indexing: `scenarios_[k][obstacle_id]` (not `scenarios_[obstacle_id][k]`)
   - Fixed `check_feasibility_by_distance()` to use correct array indexing

### 🔧 IN PROGRESS (Phase 2):
1. **SafeHorizon Implementation** (60% complete)
   - ✅ `compute_distances()`: Working correctly
   - ✅ `check_feasibility_by_distance()`: Basic logic implemented, indexing fixed
   - ⚠️ `compute_halfspaces()`: Partially implemented but needs testing
   - ❌ `construct_polytopes()`: Placeholder only - CRITICAL MISSING PIECE

2. **ScenarioSampler Cleanup** (Not started)
   - Duplicate method implementations need removal
   - Basic sampling logic appears functional

### ❌ TODO (Phase 2 & 3):
1. **Complete SafeHorizon.construct_polytopes()** - HIGHEST PRIORITY
2. **Rework ScenarioConstraints** - Use polytopes instead of individual scenarios
3. **Fix numeric integration** in test files
4. **Integration testing** with full scenario constraints

## The Critical Missing Piece: Polytope Construction

### Why This Matters:
The current Python implementation tries to add constraints for EVERY scenario:
- 50 scenarios × 3 obstacles × 10 timesteps = 1500 constraints
- Result: Solver becomes infeasible (too constrained)

The C++ implementation uses polytopes:
- Groups scenarios spatially
- Constructs polygons around feasible regions
- Adds only polytope boundaries (2-4 constraints per obstacle per timestep)
- Result: ~80 constraints total, solver finds solutions

### What `construct_polytopes()` Should Do:

```python
def construct_polytopes(self, k: int, data):
    """
    Convert individual scenario halfspaces into minimal polytope representation
    
    Algorithm (from C++ version):
    1. Group halfspaces by angular direction (e.g., in 45-degree bins)
    2. For each group, find the most restrictive constraint (closest to robot)
    3. Use convex hull algorithm to find minimal polygon enclosing feasible region
    4. Return only the polygon boundary halfspaces (typically 3-6 constraints)
    
    Input: self.a1_[k], self.a2_[k], self.b_[k] (many halfspaces)
    Output: Minimal set of non-redundant constraints
    """
    # Step 1: Group constraints by angle
    angles = np.arctan2(self.a2_[k], self.a1_[k])
    num_sectors = 8  # Divide full circle into 8 sectors
    sector_size = 2 * np.pi / num_sectors
    
    # Step 2: Find most restrictive constraint in each sector
    minimal_constraints = []
    for sector in range(num_sectors):
        sector_start = sector * sector_size
        sector_end = (sector + 1) * sector_size
        
        # Find constraints in this sector
        in_sector = (angles >= sector_start) & (angles < sector_end)
        if np.any(in_sector):
            # Get the most restrictive (smallest b value = closest to robot)
            sector_indices = np.where(in_sector)[0]
            most_restrictive_idx = sector_indices[np.argmin(self.b_[k][in_sector])]
            minimal_constraints.append(most_restrictive_idx)
    
    # Step 3: Apply convex hull algorithm to further reduce
    if len(minimal_constraints) > 3:
        # Use scipy.spatial.ConvexHull or similar
        # to find minimal polygon
        points = np.column_stack([
            self.a1_[k][minimal_constraints],
            self.a2_[k][minimal_constraints]
        ])
        hull = ConvexHull(points)
        final_indices = [minimal_constraints[i] for i in hull.vertices]
    else:
        final_indices = minimal_constraints
    
    # Step 4: Store polytope
    self.polytopes_[k] = {
        'a1': self.a1_[k][final_indices],
        'a2': self.a2_[k][final_indices],
        'b': self.b_[k][final_indices]
    }
    
    return len(final_indices)
```

### Alternative Simpler Approach (If Time-Constrained):
Instead of full polytope construction, use **"closest N scenarios"** approach:
```python
def construct_polytopes_simple(self, k: int, data):
    """
    Simplified approach: Use only the N closest infeasible scenarios
    This is not optimal but better than using all scenarios
    """
    max_constraints = 5  # From config
    
    # Find indices of closest scenarios
    closest_indices = np.argsort(self.distances_[k])[:max_constraints]
    
    # Use only constraints for closest scenarios
    self.polytopes_[k] = {
        'a1': self.a1_[k][closest_indices],
        'a2': self.a2_[k][closest_indices],
        'b': self.b_[k][closest_indices]
    }
```

## How to Continue This Work

### Immediate Next Steps (2-4 hours):

1. **Implement `construct_polytopes()` in SafeHorizon**
   - File: `planner_modules/src/constraints/scenario_utils/math_utils.py`
   - Line: ~941
   - Use either the full algorithm or simplified approach above

2. **Update `ScenarioConstraints.update()` to use polytopes**
   - File: `planner_modules/src/constraints/scenario_constraints.py`
   - Change from iterating over all scenarios to using polytope results
   - Extract constraint coefficients from `SafeHorizon.polytopes_[k]`

3. **Fix `ScenarioConstraints.define_parameters()`**
   - Currently defines parameters for all scenarios (wrong)
   - Should define parameters only for `max_constraints` polytope boundaries
   - Example: 5 constraints × 10 timesteps = 50 parameters (instead of 1000s)

4. **Test with SimplifiedScenarioConstraints replaced**
   - Switch back to real `ScenarioConstraints` in the test
   - Verify it doesn't crash on step 8 anymore
   - Check if robot makes progress towards goal

### Files That Need Modification:

1. **planner_modules/src/constraints/scenario_utils/math_utils.py**
   - Implement `SafeHorizon.construct_polytopes()` (line ~941)
   - Verify `check_feasibility_by_distance()` logic
   - Test `compute_halfspaces()` produces correct constraint coefficients

2. **planner_modules/src/constraints/scenario_constraints.py**
   - Rework `define_parameters()` to use `max_constraints` instead of all scenarios
   - Update `update()` method to call `SafeHorizon.construct_polytopes()`
   - Fix `set_parameters()` to extract from polytopes, not individual scenarios
   - Modify `get_constraints()` to return polytope-based constraints

3. **test/integration/constraints/scenario/scenario_and_contouring_constraints_with_contouring_objective.py**
   - Switch from `SimplifiedScenarioConstraints` back to `ScenarioConstraints`
   - Verify numeric integration (may need fixing separately)
   - Add better logging to track progress

### Testing Strategy:

1. **Unit Test SafeHorizon** (Create new file: `test/unit/test_safe_horizon.py`)
   ```python
   def test_construct_polytopes():
       # Create SafeHorizon with mock data
       # Call compute_distances, check_feasibility, compute_halfspaces
       # Call construct_polytopes
       # Assert: Number of constraints reduced from 50 to ~5
   ```

2. **Integration Test Scenario Constraints** (Modify existing test)
   - Start with no obstacles → Should reach goal easily
   - Add 1 static obstacle → Should avoid it
   - Add dynamic obstacles → Should use scenario constraints

3. **Full System Test** (Current test file)
   - Enable both contouring and scenario constraints
   - Verify robot reaches goal
   - Check constraint satisfaction throughout

## Key Configuration Parameters

From `config/CONFIG.yml`:
```yaml
scenario_constraints:
  num_scenarios: 4          # Number of parallel solvers (discs)
  max_constraints: 10       # Max polytope boundaries per timestep
  sample_size: 50           # Number of scenario samples
  num_halfspaces: 2         # Halfspaces per obstacle per scenario (for initial computation)
  polygon_range: 2          # Range for polygon construction
```

**Important**: The system should define parameters for `max_constraints` (10), NOT for `sample_size` (50). This is the key difference between polytope-based and brute-force approaches.

## Expected Outcome After Completion:

### Current Behavior (with SimplifiedScenarioConstraints):
- ✅ Iteration 0: Succeeds
- ❌ Iteration 1: MPC fails (infeasible)
- Robot doesn't move

### Expected Behavior (with proper ScenarioConstraints):
- ✅ Iterations 0-50+: All succeed
- ✅ Robot moves from (0, 0) to (50, 10)
- ✅ Robot avoids dynamic obstacles using scenario constraints
- ✅ Robot stays on path using contouring constraints
- ✅ Computation time < 100ms per iteration

## Debugging Tips:

1. **If MPC becomes infeasible after re-enabling ScenarioConstraints:**
   - Check number of constraints being added (should be ~50-100 total, not 1000s)
   - Log constraint coefficients (a1, a2, b) to verify they're reasonable
   - Verify polytope construction is reducing constraint count

2. **If robot doesn't avoid obstacles:**
   - Check `check_feasibility_by_distance()` is correctly identifying infeasible scenarios
   - Verify `compute_halfspaces()` is creating constraints in the right direction
   - Log infeasible scenario positions vs. robot position

3. **If step 8 error reoccurs:**
   - Check array dimensions match `horizon + 1` (11 steps for horizon=10)
   - Verify all scenario data structures use correct indexing `[k][obstacle_id]`
   - Add bounds checking before all array accesses

## Useful Commands:

```bash
# Run the scenario + contouring test
cd /home/stephencrawford/PycharmProjects/PyMPC && \
PYTHONPATH=/home/stephencrawford/PycharmProjects/PyMPC \
python test/integration/constraints/scenario/scenario_and_contouring_constraints_with_contouring_objective.py

# Grep for specific progress info
... | grep -E "(Iteration|distance to goal|Goal reached)"

# Check for constraint count (add logging first)
... | grep "num_constraints"

# Watch for errors
... | grep -E "(Error|Failed|infeasible)"
```

## Reference Documents:

- **CPP_SCENARIO_MODULE_ANALYSIS.md**: Architecture details from C++ code
- **SCENARIO_CONSTRAINTS_REWORK_PLAN.md**: Full 3-phase plan
- **REWORK_PROGRESS_SUMMARY.md**: Progress tracking

## Estimated Remaining Work:

- **Implement `construct_polytopes()`**: 2-4 hours
- **Rework `ScenarioConstraints`**: 4-6 hours
- **Testing and debugging**: 4-8 hours
- **Total**: 10-18 hours

## Contact/Handoff Notes:

The foundation has been laid:
- ✅ Analysis complete
- ✅ Baseline working (doesn't crash)
- ✅ Critical bugs fixed (step 8 indexing)
- ✅ Path forward documented

The remaining work is primarily:
- Implementing `construct_polytopes()` (the "smart reduction" step)
- Integrating it into `ScenarioConstraints`
- Testing and refinement

The simplified approach (using closest N scenarios) could be implemented first as a proof-of-concept before attempting full polytope construction.

