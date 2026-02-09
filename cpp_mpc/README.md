# Adaptive Scenario-Based MPC (C++ Implementation)

A C++ implementation of an Adaptive Scenario-Based Model Predictive Controller for autonomous vehicle motion planning under uncertainty.

## Overview

This library implements a scenario-based robust MPC framework that enables vehicles to plan safe, efficient trajectories while avoiding dynamic obstacles with uncertain behaviors. The implementation follows the mathematical formulation from the TUD-AMR MPC Planner.

### Key Features

- **Scenario-Based Robust MPC**: Samples multiple obstacle motion scenarios based on historical observations
- **Adaptive Mode Weighting**: Uses mode history (frequency, recency) to weight obstacle behavior predictions
- **Linearized Collision Constraints**: Efficient constraint formulation for real-time optimization
- **Scenario Pruning**: Reduces computation through dominance pruning and inactive scenario removal
- **Modular Design**: Clean separation of dynamics, sampling, constraints, and optimization

## Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2019+)
- CMake 3.14+
- Eigen3 3.3+

### Installing Dependencies

Ubuntu/Debian:
```bash
sudo apt-get install cmake libeigen3-dev
```

macOS:
```bash
brew install cmake eigen
```

## Building

```bash
cd cpp_mpc
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build Options

- `BUILD_TESTS=ON/OFF` - Build unit tests (default: ON)
- `BUILD_EXAMPLES=ON/OFF` - Build example programs (default: ON)

```bash
cmake -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON ..
```

## Running Tests

```bash
cd build
./test_core
# Or using ctest
ctest --output-on-failure
```

## Usage

### Basic Example

```cpp
#include "mpc_controller.hpp"

using namespace scenario_mpc;

int main() {
    // Configure the controller
    ScenarioMPCConfig config;
    config.horizon = 20;        // 20 timestep horizon
    config.dt = 0.1;            // 100ms timestep
    config.num_scenarios = 10;  // Sample 10 scenarios
    config.confidence_level = 0.95;  // 95% constraint satisfaction

    // Create controller
    AdaptiveScenarioMPC controller(config);

    // Set up scenario
    EgoState ego_state(0, 0, 0, 0);  // x, y, theta, v
    std::map<int, ObstacleState> obstacles = {
        {0, ObstacleState(5, 0, -1, 0)}  // Approaching obstacle
    };
    Eigen::Vector2d goal(10, 0);

    // Initialize obstacle mode tracking
    controller.initialize_obstacle(0);
    controller.update_mode_observation(0, "constant_velocity");

    // Solve MPC
    MPCResult result = controller.solve(ego_state, obstacles, goal);

    if (result.success) {
        // Apply first control input
        EgoInput input = result.first_input().value();
        std::cout << "Acceleration: " << input.a << " m/s^2\n";
        std::cout << "Angular velocity: " << input.delta << " rad/s\n";
    }

    return 0;
}
```

### Running the Example

```bash
cd build
./example_mpc
```

## Architecture

```
cpp_mpc/
├── include/
│   ├── types.hpp              # Core data structures
│   ├── config.hpp             # Configuration parameters
│   ├── dynamics.hpp           # Ego vehicle dynamics
│   ├── mode_weights.hpp       # Mode weight computation
│   ├── trajectory_moments.hpp # Moment propagation
│   ├── scenario_sampler.hpp   # Scenario sampling (Algorithm 1)
│   ├── collision_constraints.hpp  # Linearized constraints
│   ├── scenario_pruning.hpp   # Pruning algorithms (3-4)
│   └── mpc_controller.hpp     # Main MPC controller (Algorithm 2)
├── src/
│   └── [implementations]
├── tests/
│   └── test_core.cpp          # Unit tests
├── examples/
│   └── example_mpc.cpp        # Example program
└── CMakeLists.txt
```

## Core Components

### 1. Types (`types.hpp`)

- `EgoState`: Vehicle state (x, y, theta, v)
- `EgoInput`: Control input (acceleration, angular velocity)
- `ObstacleState`: Obstacle state (x, y, vx, vy)
- `ModeModel`: Mode-dependent dynamics (A, b, G matrices)
- `ModeHistory`: Observed mode history for an obstacle
- `Scenario`: Collection of obstacle trajectories
- `CollisionConstraint`: Linearized collision constraint
- `MPCResult`: Optimization result

### 2. Dynamics (`dynamics.hpp`)

Unicycle model for ego vehicle:
```
dx/dt = v * cos(theta)
dy/dt = v * sin(theta)
dtheta/dt = w (angular velocity)
dv/dt = a (acceleration)
```

Obstacle mode models:
- `constant_velocity`: Constant velocity motion
- `turn_left`/`turn_right`: Turning motions
- `lane_change_left`/`lane_change_right`: Lane change maneuvers
- `decelerating`: Braking motion

### 3. Mode Weights (`mode_weights.hpp`)

Three weighting strategies:
- **Uniform** (Eq. 4): Equal weights for all modes
- **Frequency** (Eq. 6): Based on observation counts
- **Recency** (Eq. 5): Exponential decay favoring recent observations

### 4. Scenario Sampling (`scenario_sampler.hpp`)

Implements Algorithm 1: SampleScenarios
- Samples mode sequences from categorical distribution
- Propagates obstacle trajectories with process noise
- Creates joint scenarios across all obstacles

### 5. Collision Constraints (`collision_constraints.hpp`)

Linearized half-space constraints:
```
a^T * p_ego >= b
```
where `a` is the separating hyperplane normal and `b` is the offset.

### 6. Scenario Pruning (`scenario_pruning.hpp`)

- **Algorithm 3**: Geometric dominance pruning
- **Algorithm 4**: Support-based scenario removal (after optimization)
- Clustering for diverse scenario selection

### 7. MPC Controller (`mpc_controller.hpp`)

Main Algorithm 2: AdaptiveScenarioMPC
1. Initialize reference trajectory (warmstart)
2. Sample scenarios
3. Prune dominated scenarios
4. Compute linearized constraints
5. Solve optimization
6. Remove inactive scenarios

## Mathematical Foundation

### Theorem 1: Sample Complexity

For ε-chance constraint satisfaction with confidence 1-β:
```
S >= 2/ε * (ln(1/β) + n_x)
```
where S is the number of scenarios and n_x is the number of decision variables.

### Constraint Formulation (Eq. 17-18)

Direction vector:
```
a = (p_ego - p_obs) / ||p_ego - p_obs||
```

Linearized constraint:
```
a^T * p_ego >= a^T * p_obs + r_combined
```

### Moment Propagation (Proposition 1)

For multi-modal obstacle predictions:
```
μ_k = Σ_m w_m * (A_m * μ_{k-1} + b_m)
Σ_k = Σ_m w_m * (A_m * Σ_{k-1} * A_m^T + G_m * G_m^T) + mode_mixing_term
```

## Configuration

Key parameters in `ScenarioMPCConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | 20 | Prediction horizon |
| `dt` | 0.1 | Timestep [s] |
| `num_scenarios` | 10 | Number of scenarios to sample |
| `confidence_level` | 0.95 | Chance constraint confidence |
| `ego_radius` | 1.0 | Ego collision radius [m] |
| `obstacle_radius` | 0.5 | Obstacle radius [m] |
| `safety_margin` | 0.1 | Additional safety margin [m] |
| `weight_type` | FREQUENCY | Mode weight strategy |
| `recency_decay` | 0.9 | Decay for recency weighting |

## Comparison with Python Implementation

This C++ implementation provides:
- Equivalent mathematical algorithms
- Same test coverage
- Faster execution (no interpreter overhead)
- No Python/CasADi dependency
- Suitable for embedded systems

The API mirrors the Python version for easy migration.

## License

Same as the parent PyMPC project.

## References

1. Campi, M. C., & Garatti, S. (2018). Introduction to the scenario approach.
2. TUD-AMR MPC Planner documentation.
