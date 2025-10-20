# C++ Scenario Module Analysis

## Goal
Understand the architecture of the C++ `scenario_module` from oscardegroot/scenario_module to properly implement it in Python.

## Key Components

### 1. ScenarioModule (Main Orchestrator)
**Purpose**: Manages multiple disc managers and orchestrates scenario optimization.

**Key Responsibilities**:
- Manages multiple `ScenarioBase` instances (one per disc)
- Coordinates sampling of scenarios
- Runs parallel scenario solvers
- Aggregates results from all discs
- Provides constraint parameters to the MPC solver

**Key Data Structures**:
- `disc_manager_`: Vector of `ScenarioBase` objects, one per disc
- `sampler_`: `ScenarioSampler` instance for generating scenario samples
- `safe_horizon_`: `SafeHorizon` instance for computing feasible constraints

### 2. ScenarioBase (Per-Disc Manager)
**Purpose**: Represents a single disc (robot body segment) and its associated scenarios.

**Key Responsibilities**:
- Manages scenarios for a specific disc
- Stores constraint parameters (a1, a2, b) for halfspace constraints
- Tracks feasibility status
- Interfaces with the SafeHorizon for disc-specific computations

**Key Data**:
- Constraint coefficients (a1, a2, b) for each scenario at each timestep
- Feasibility flags
- Disc-specific properties (radius, position offset from robot center)

### 3. ScenarioSampler
**Purpose**: Generates and manages scenario samples for obstacle predictions.

**Key Responsibilities**:
- Samples from prediction distributions (Gaussian, Multimodal, etc.)
- Propagates scenarios forward in time
- Prunes scenarios to maintain computational efficiency
- Provides scenario positions for all timesteps and obstacles

**Key Methods**:
- `sample_standard_normal()`: Samples from standard normal distribution
- `sample_truncated_standard_normal()`: Samples with truncation
- `prune()`: Reduces the number of scenarios while maintaining coverage
- `integrate_and_translate_to_mean_and_variance()`: Transforms samples to match prediction

**Data Structure**:
- `samples[step][obstacle_id][dimension][sample_id]`:
  - `step`: Timestep in the horizon (0 to horizon)
  - `obstacle_id`: Index of the obstacle
  - `dimension`: 0 for x, 1 for y
  - `sample_id`: Index of the scenario sample

### 4. SafeHorizon
**Purpose**: Performs disc-wise computations for scenario MPC, including distance calculations, feasibility checks, and polytope construction.

**Key Responsibilities**:
- Compute distances from robot prediction to all scenarios
- Check feasibility of scenarios based on distance thresholds
- Compute halfspaces (separating hyperplanes) for feasible scenarios
- Construct polytopes around feasible scenario sets
- Identify and track infeasible scenarios

**Key Methods**:
- `compute_distances(data, k, obstacle_id)`: Computes Euclidean distance from predicted robot position to all scenario samples
- `check_feasibility_by_distance(k, obstacle_id)`: Determines which scenarios are feasible (far enough from robot)
- `compute_halfspaces(k, obstacle_id)`: Creates separating hyperplanes between robot and scenarios
- `construct_polytopes()`: Builds polytopes from halfspaces
- `is_data_ready()`: Checks if scenario data is available
- `reset()`: Clears all internal state

**Key Data Structures**:
- `scenarios_`: Reference to sampler's scenario samples
- `distances_[k]`: Distance to each scenario at timestep k
- `diffs_x_[k]`, `diffs_y_[k]`: Direction vectors to scenarios
- `feasible_[k]`: Boolean array indicating feasible scenarios
- `infeasible_scenario_poses_[k]`: Positions of infeasible scenarios
- `constraints_[k]`: Generated halfspace constraints

**Critical Insight**: The SafeHorizon does NOT create individual constraints for every scenario. Instead, it:
1. Identifies feasible scenarios (those far enough from the robot)
2. Groups them into regions
3. Constructs polytopes around these regions
4. Only the polytope boundaries become constraints in the MPC

### 5. ScenarioSolver (Parallel Worker)
**Purpose**: Solves scenario optimization for a specific disc.

**Key Responsibilities**:
- Runs scenario-specific optimization
- Computes constraint coefficients for its disc
- Returns feasibility status and constraint parameters

**Key Design Pattern**: Multiple `ScenarioSolver` instances run in parallel (one per disc or per solver thread), each solving a subset of the overall problem.

## Critical Architecture Insights

### A. Constraint Generation Philosophy
The C++ implementation does NOT:
- Add explicit constraints for every scenario at every timestep
- Create N*M*H constraints (N scenarios × M obstacles × H horizon)

The C++ implementation DOES:
- Identify feasible vs. infeasible scenarios based on distance
- Construct polytopes that enclose groups of feasible scenarios
- Only add constraints for polytope boundaries (typically 2-4 per disc per timestep)
- Use "safe horizon" concept: if a timestep has many feasible scenarios, it's "safe" and may need fewer constraints

### B. Data Flow
1. **Sampling Phase** (before MPC):
   - `ScenarioSampler` generates scenarios for all obstacles, timesteps
   - Scenarios are propagated forward using obstacle prediction models
   
2. **Update Phase** (before each MPC solve):
   - `SafeHorizon.compute_distances()`: Calculate distances to all scenarios
   - `SafeHorizon.check_feasibility_by_distance()`: Identify feasible scenarios
   - `SafeHorizon.compute_halfspaces()`: Generate separating hyperplanes
   - `SafeHorizon.construct_polytopes()`: Build polytopes from halfspaces
   
3. **Parameter Setting Phase** (during MPC setup):
   - `ScenarioModule` extracts polytope constraints from `SafeHorizon`
   - Constraint coefficients (a1, a2, b) are set as MPC parameters
   - MPC solver uses these as numeric parameters in symbolic constraints

4. **Solve Phase**:
   - MPC solver enforces constraints: a1*x + a2*y <= b
   - Robot trajectory avoids infeasible regions (where scenarios are too close)

### C. Feasibility Logic
A scenario is "feasible" if:
```
distance(robot_prediction, scenario_position) > safety_threshold
```

Where:
```
safety_threshold = disc_radius + obstacle_radius + safety_margin
```

If all scenarios for an obstacle are feasible at a timestep, no constraints are needed (the obstacle is far enough away).

If some scenarios are infeasible, the `SafeHorizon`:
1. Identifies the boundary between feasible and infeasible regions
2. Constructs halfspace constraints that separate the robot from infeasible scenarios
3. Optionally groups constraints into polytopes for efficiency

### D. Polytope Construction
The polytope approach:
- Takes multiple halfspaces (a1*x + a2*y <= b)
- Groups them by spatial proximity or angular direction
- Reduces redundant constraints
- Results in a smaller, tighter set of constraints

This is key to computational efficiency: instead of 100+ constraints (one per scenario), you might have 5-10 polytope boundaries.

## Current Python Implementation Issues

### Issue 1: Wrong Constraint Generation Pattern
**Current Python**: Tries to add constraints for every scenario at every timestep.
**Correct C++ Pattern**: Only add constraints for polytope boundaries of infeasible scenarios.

### Issue 2: Missing Feasibility Logic
**Current Python**: `SafeHorizon.check_feasibility_by_distance()` is a placeholder.
**Correct C++ Pattern**: Must properly identify feasible vs. infeasible scenarios before generating any constraints.

### Issue 3: Incorrect Data Structure Access
**Fixed**: The Python code was trying to access `scenarios_[k][obstacle_id]` directly, but scenarios are stored as `samples[step][obstacle_id][0/1][sample_id]`.

### Issue 4: No Polytope Logic
**Current Python**: Attempts to generate individual halfspaces but never constructs polytopes.
**Correct C++ Pattern**: Must implement `construct_polytopes()` to group and reduce constraints.

### Issue 5: Parallel Solver Structure Mismatch
**Current Python**: `ScenarioConstraints` has a list of `ScenarioSolver` instances but doesn't properly orchestrate them.
**Correct C++ Pattern**: Each solver should work on a specific disc and return aggregated results.

## Implementation Roadmap

### Phase 1: Baseline (COMPLETED)
- Created `SimplifiedScenarioConstraints` that provides non-binding dummy constraints
- Verified the MPC can run without crashing (achieved as of most recent test)

### Phase 2: Core Scenario Logic (IN PROGRESS)
#### Step 2.1: Fix ScenarioSampler
- [ ] Implement proper sampling methods
- [ ] Ensure correct data structure initialization
- [ ] Add scenario propagation logic

#### Step 2.2: Fix SafeHorizon
- [ ] Implement `compute_distances()` correctly (PARTIALLY DONE - indexing fixed)
- [ ] Implement `check_feasibility_by_distance()` properly
- [ ] Implement `compute_halfspaces()` for separating hyperplanes
- [ ] Implement `construct_polytopes()` for constraint reduction
- [ ] Add proper data structure management

#### Step 2.3: Rework ScenarioConstraints
- [ ] Change constraint generation to use polytopes, not individual scenarios
- [ ] Integrate properly with SafeHorizon
- [ ] Fix parallel solver orchestration
- [ ] Ensure constraint parameters are correctly sized and indexed

#### Step 2.4: Fix Numeric Integration
- [ ] Verify state propagation in tests uses correct control inputs
- [ ] Ensure `numeric_rk4` or equivalent is properly called

### Phase 3: Testing and Refinement
- [ ] Test scenario constraints in isolation (no contouring)
- [ ] Test contouring + scenario constraints together
- [ ] Verify goal-reaching with all constraints enabled
- [ ] Performance optimization

## Key Takeaways

1. **The Python adaptation was trying to solve the problem by brute force** (individual constraints per scenario), while the C++ implementation uses a **smart polytope-based approach** that dramatically reduces the constraint count.

2. **Feasibility is key**: The C++ code doesn't try to avoid all scenarios; it identifies which ones are actually problematic (infeasible) and only constrains those.

3. **Safe Horizon concept**: The name "safe horizon" comes from the idea that if scenarios are far away (safe), you need fewer or no constraints. The horizon is "safe" when most scenarios are feasible.

4. **Data structure consistency**: The scenarios must be accessed correctly as `samples[step][obstacle_id][dim][sample_id]`.

5. **Computational efficiency**: The polytope approach is essential for real-time MPC. Without it, the problem becomes intractable with many scenarios and obstacles.

