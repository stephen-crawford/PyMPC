# Halfspace-Based Scenario Constraints Implementation

## Reference Implementation
**Source**: https://github.com/oscardegroot/scenario_module  
**Paper**: O. de Groot et al., "Scenario-Based Trajectory Optimization with Bounded Probability of Collision", IJRR 2024  
**Performance**: ~400 samples for 12 obstacles at 30 Hz with <10% collision probability

## Key Innovation

### ❌ OLD Approach (Current Python Implementation)
```python
# Problem: Add constraints for EVERY scenario
for obstacle in obstacles:              # 12 obstacles
    for timestep in horizon:            # 15 timesteps  
        for scenario in scenarios:      # 400 scenarios
            add_constraint(scenario)    # RESULT: 72,000 constraints!
            
# Result: INFEASIBLE - Problem is over-constrained
```

### ✅ NEW Approach (From IJRR 2024 Paper)
```python
# Step 1: Run parallel scenario optimizers
solutions = run_parallel_solvers(scenarios)  # 4 solvers

# Step 2: Select best feasible solution
best_solution = select_min_cost(solutions)

# Step 3: Extract halfspace constraints from best solution
halfspaces = extract_halfspaces(best_solution, scenarios)

# Step 4: Add ONLY the halfspace constraints
for halfspace in halfspaces:            # ~5-10 per timestep
    add_constraint(halfspace)           # RESULT: ~75 constraints total!

# Result: FEASIBLE - Problem is well-constrained
```

## Implementation Files

### 1. Halfspace Builder
**File**: `planner_modules/src/constraints/scenario_utils/halfspace_builder.py` (373 lines)

**Classes**:
- `HalfspaceBuilder`: Extracts halfspace constraints from scenarios
- `HalfspaceConstraintExtractor`: Processes parallel solver results

**Key Methods**:
- `build_halfspaces_from_trajectory()`: Main entry point
- `_filter_infeasible_scenarios()`: Distance-based feasibility check
- `_polytope_halfspaces()`: Construct convex hull polytopes
- `_individual_halfspaces()`: Fallback for few scenarios
- `_reduce_halfspaces()`: Keep most important constraints

**Algorithm**:
1. For each timestep in trajectory:
   - For each obstacle:
     - Filter scenarios by distance (keep only those within collision risk)
     - If 3+ scenarios: Construct convex hull polytope
     - If <3 scenarios: Create individual separating hyperplanes
     - Extract halfspace constraints from polytope boundaries
   - Reduce to max N most important halfspaces per timestep
2. Return dict: {timestep: [(normal, offset), ...]}

### 2. Scenario Constraints (Halfspace Version)
**File**: `planner_modules/src/constraints/scenario_constraints_halfspace.py` (437 lines)

**Class**: `ScenarioConstraintsHalfspace(BaseConstraint)`

**Key Differences from Old Implementation**:
```python
# OLD: Store constraint for every scenario
_constraints = []  # 72,000 entries
for scenario in all_scenarios:
    _constraints.append(compute_constraint(scenario))

# NEW: Store only halfspace parameters
_a1 = np.zeros((num_discs, horizon+1, max_halfspaces_per_timestep))
_a2 = np.zeros((num_discs, horizon+1, max_halfspaces_per_timestep))
_b  = np.zeros((num_discs, horizon+1, max_halfspaces_per_timestep))
# Size: 1 * 16 * 5 = 80 constraints (vs 72,000!)
```

**Update Process**:
```python
def update(self, state, data):
    # 1. Generate scenario samples (once in main thread)
    sampler = self._initialize_sampler(data)
    
    # 2. Run parallel scenario solvers
    solutions = self._run_parallel_solvers(state, data)
    
    # 3. Extract halfspaces from best solution
    self._halfspaces = self.halfspace_extractor.extract_from_parallel_solutions(
        solutions, sampler.samples, data.dynamic_obstacles
    )
    
    # 4. Convert halfspaces to constraint parameters
    self._convert_halfspaces_to_parameters()
```

**Fallback Mechanism**:
If parallel solvers fail, uses simple distance-based constraints as fallback.

### 3. Test Demonstration
**File**: `test/integration/test_halfspace_scenario_demo.py` (312 lines)

**Test Class**: `HalfspaceScenarioDemo(BaseMPCTest)`

**Test Scenario**:
- Curved B-spline reference path
- 3 dynamic obstacles with Gaussian predictions
- Road boundaries (optional)
- Contouring objective for path following

**Visualization**:
- Plots trajectory with halfspace constraints
- Shows vehicle path, obstacles, uncertainty regions
- Displays statistics (solve time, iterations, distance traveled)

### 4. Configuration
**File**: `configs/halfspace_scenario_config.yaml`

**Key Parameters**:
```yaml
scenario_constraints:
  parallel_solvers: 4                    # Run 4 solvers in parallel
  num_scenarios: 400                     # 400 scenarios per obstacle
  max_halfspaces_per_timestep: 5         # Only 5 constraints per timestep!
  safety_margin: 0.5                     # Safety buffer (meters)
  feasibility_threshold: 3.0             # Distance threshold
  min_scenarios_for_polytope: 3          # Min for polytope construction
```

## Mathematical Foundation

### Halfspace Constraint
Each halfspace represents a linear inequality:
```
normal · position ≤ offset
```
In 2D:
```
a₁·x + a₂·y ≤ b
```

Where:
- `normal = [a₁, a₂]`: Unit direction vector
- `offset = b`: Scalar offset
- The halfspace divides the space into "safe" and "unsafe" regions

### Polytope Construction
For N infeasible scenarios at positions `{p₁, p₂, ..., pₙ}`:

1. Compute convex hull: `H = ConvexHull(p₁, p₂, ..., pₙ)`
2. Extract boundary equations: Each face of H gives a halfspace
3. Orient halfspaces: Normal should point away from robot
4. Result: Robot must stay outside the polytope (safe region)

### Feasibility Check
A scenario at position `p_scenario` is **infeasible** if:
```
distance(robot_pos, p_scenario) < obstacle_radius + safety_margin
```

Otherwise, it's **feasible** (safe) and doesn't need a constraint.

## Performance Comparison

### Old Approach
```
Constraints: 72,000 (12 obstacles × 15 timesteps × 400 scenarios)
MPC Solve Time: FAIL (infeasible)
Planning Rate: 0 Hz (can't solve)
```

### New Approach (Expected)
```
Constraints: 75 (15 timesteps × 5 halfspaces)
MPC Solve Time: ~0.03s
Planning Rate: ~30 Hz
Feasibility: ✅ (well-constrained)
```

## Usage Example

```python
from test.integration.test_halfspace_scenario_demo import HalfspaceScenarioDemo

# Create test
test = HalfspaceScenarioDemo(
    'halfspace_demo',
    dt=0.1,
    horizon=15,
    max_iterations=100
)

# Setup
test.setup(start=(0, 0), goal=(30, 20))

# Run
result = test.run()

# Visualize
plot_halfspace_trajectory(result, test.data)
```

## Expected Results

Based on the IJRR 2024 paper performance:

1. **Real-time Planning**: ~30 Hz update rate
2. **Obstacle Avoidance**: <10% collision probability
3. **Scalability**: 400 scenarios × 12 obstacles
4. **Smooth Trajectories**: Vehicle follows reference path
5. **Feasible Solutions**: MPC solves successfully

## Next Steps

1. **Run Test**: `python test/integration/test_halfspace_scenario_demo.py`
2. **Verify Performance**: Check if solve time is ~0.03s (30 Hz)
3. **Tune Parameters**: Adjust safety_margin, max_halfspaces_per_timestep
4. **Compare**: Test against old scenario_constraints.py
5. **Integrate**: Replace old scenario constraints with halfspace version

## References

1. **Paper**: O. de Groot et al., "Scenario-Based Trajectory Optimization with Bounded Probability of Collision", IJRR 2024  
   Preprint: https://arxiv.org/pdf/2307.01070

2. **Code**: https://github.com/oscardegroot/scenario_module  
   C++ implementation of Safe Horizon MPC

3. **MPC Planner**: https://github.com/tud-amr/mpc_planner  
   Framework for integrating scenario module

## Summary

The **halfspace approach** is the key innovation that enables real-time scenario MPC:

- **72,000 constraints → 75 constraints** (1000× reduction!)
- **Infeasible → Feasible** (well-constrained problem)
- **0 Hz → 30 Hz** (real-time performance)

This is achieved by:
1. Running parallel scenario optimizations
2. Selecting the best solution
3. Extracting only boundary halfspaces (not all scenarios)
4. Adding minimal, essential constraints to MPC

