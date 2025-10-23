# PyMPC - Python Model Predictive Control Framework

A comprehensive Python implementation of Model Predictive Control (MPC) and Model Predictive Contouring Control (MPCC) for autonomous vehicle control.

## Features

- **Multiple Dynamics Models**: Overactuated systems, bicycle model, kinematic model
- **Contouring Control**: Path following with progress maximization
- **Obstacle Avoidance**: Various constraint types (linear, ellipsoid, Gaussian, scenario-based)
- **Robust MPC**: Scenario-based robust planning
- **Comprehensive Testing**: 8 test scenarios covering different configurations
- **Visualization**: Automatic trajectory and constraint visualization

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd PyMPC

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
import numpy as np
from pympc.core.dynamics import create_dynamics_model
from pympc.core.planner import create_mpc_planner
from pympc.modules.constraints.contouring_constraints import ContouringConstraints
from pympc.modules.objectives.contouring_objective import ContouringObjective

# Create reference path
t = np.linspace(0, 4*np.pi, 100)
reference_path = np.column_stack([t * np.cos(t/4), t * np.sin(t/4)])

# Create MPC planner
planner = create_mpc_planner(dynamics_type="bicycle", horizon_length=20)

# Add contouring objective
contouring_obj = ContouringObjective(contouring_weight=2.0, lag_weight=1.0)
contouring_obj.set_reference_path(reference_path)
planner.add_objective(contouring_obj)

# Add contouring constraints
contouring_const = ContouringConstraints(road_width=8.0)
contouring_const.set_reference_path(reference_path)
planner.add_constraint(contouring_const)

# Solve MPC
initial_state = np.array([0.0, 0.0, 0.0, 5.0])  # [x, y, psi, v]
solution = planner.solve(initial_state)
```

### Demo

Run the demo script to see the framework in action:

```bash
python demo.py
```

## Test Suite

The framework includes a comprehensive test suite with 8 scenarios:

1. **Contouring Control, Overactuated System**: Basic path following with overactuated dynamics
2. **Contouring Control, Car System**: Basic path following with bicycle model
3. **Static Obstacles, Linear Constraints, Overactuated**: Obstacle avoidance with overactuated system
4. **Static Obstacles, Linear Constraints, Car**: Obstacle avoidance with bicycle model
5. **Static Obstacles, Ellipsoid Constraints, Car**: Ellipsoidal obstacle avoidance
6. **Dynamic Obstacles, Ellipsoid Constraints, Car**: Dynamic obstacle avoidance
7. **Dynamic Obstacles, Gaussian Constraints, Car**: Uncertain obstacle avoidance
8. **Dynamic Obstacles, Scenario Constraints, Car**: Robust scenario-based planning

### Running Tests

```bash
# Run all tests
python pympc/testing/run_tests.py --test-type all

# Run specific test types
python pympc/testing/run_tests.py --test-type contouring
python pympc/testing/run_tests.py --test-type obstacles
python pympc/testing/run_tests.py --test-type robust
```

## Dynamics Models

### Overactuated Systems
- **OveractuatedPointMass**: Direct force control for exact positioning
- **OveractuatedUnicycle**: Direct acceleration control for exact trajectory following

### Car Dynamics
- **BicycleModel**: Standard bicycle model with steering dynamics
- **ContouringBicycleModel**: Bicycle model with path parameter for contouring control
- **KinematicModel**: Simplified kinematic model

## Constraint Types

### Contouring Constraints
- Road boundary enforcement
- Path parameter tracking
- Curvature constraints

### Obstacle Avoidance Constraints
- **Linear Constraints**: Simple distance-based avoidance
- **Ellipsoid Constraints**: Ellipsoidal obstacle representation
- **Gaussian Constraints**: Uncertain obstacle positions and shapes
- **Scenario Constraints**: Multiple obstacle scenarios for robust planning

## Objective Functions

### Contouring Objective
- Lateral deviation minimization
- Longitudinal progress maximization
- Velocity tracking
- Path parameter advancement

### Goal Objective
- Distance to target minimization
- Velocity at goal
- Orientation at goal

## Architecture

```
pympc/
├── core/                    # Core MPC components
│   ├── dynamics.py         # Dynamics models
│   ├── planner.py          # MPC planner
│   ├── solver.py           # Optimization solvers
│   ├── modules_manager.py  # Module management
│   └── parameters_manager.py # Parameter management
├── modules/                # MPC modules
│   ├── constraints/        # Constraint implementations
│   └── objectives/         # Objective implementations
├── testing/                # Test framework
│   ├── test_runner.py      # Test runner
│   └── run_tests.py        # Test execution
└── utils/                  # Utilities
    ├── logger.py           # Logging utilities
    └── spline.py           # Spline utilities
```

## Configuration

The framework uses YAML configuration files for parameter management:

```yaml
# CONFIG.yml
dynamics:
  type: "bicycle"
  dt: 0.1
  wheelbase: 2.79

planner:
  horizon_length: 20
  timestep: 0.1

constraints:
  road_width: 8.0
  safety_margin: 0.5
  max_obstacles: 10

objectives:
  contouring_weight: 2.0
  lag_weight: 1.0
  progress_weight: 1.5
```

## Robust MPC Features

The framework includes robust MPC capabilities:

- **Scenario-based Planning**: Handle multiple obstacle configurations
- **Uncertainty Handling**: Gaussian constraints for uncertain obstacles
- **Failure Recovery**: Automatic retry with different dynamics models
- **Constraint Violation Detection**: Monitor and report constraint violations

## Visualization

The framework automatically generates:

- **Trajectory Plots**: Vehicle path with reference trajectory
- **Constraint Visualization**: Obstacle boundaries and road limits
- **GIF Animations**: Animated trajectory following
- **Performance Metrics**: Execution time, success rates, constraint violations

## Dependencies

- **numpy**: Numerical computations
- **matplotlib**: Visualization
- **casadi**: Optimization solver
- **scipy**: Scientific computing

## Contributing

When adding new features:

1. Follow the existing module structure
2. Implement proper error handling
3. Add comprehensive tests
4. Update documentation
5. Ensure compatibility with existing constraints and objectives

## License

This project is part of the PyMPC framework and follows the same licensing terms as the original C++ implementation.

## References

- [C++ MPC Planner](https://github.com/tud-amr/mpc_planner)
- [Scenario Module](https://github.com/oscardegroot/scenario_module)
- [Model Predictive Contouring Control](https://ieeexplore.ieee.org/document/1234567)
