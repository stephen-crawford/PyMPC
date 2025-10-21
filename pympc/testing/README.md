# Standardized MPCC Testing Framework

This directory contains a comprehensive, standardized testing framework for Model Predictive Contouring Control (MPCC) that allows for easy configuration and testing of various scenarios.

## Features

### Core Capabilities
- **Standardized Test Framework**: Easy-to-use configuration system for MPCC tests
- **Curving Road Generation**: Automatic generation of curved roads with configurable parameters
- **Car Dynamics Model**: Bicycle model implementation for realistic vehicle dynamics
- **Contouring Objectives**: MPCC objective functions for path following and progress maximization
- **Contouring Constraints**: Road boundary enforcement and obstacle avoidance
- **Dynamic Obstacles**: Configurable obstacles with randomized trajectories
- **Perception Areas**: Multiple perception shapes (circle, cone, rectangle, etc.)
- **Automatic Visualization**: GIF generation with constraint overlays
- **C++ Reference Implementation**: Mathematical consistency with the original C++ codebase

### Test Categories
1. **Basic MPCC Tests**: Standard functionality testing
2. **Perception Tests**: Different perception area configurations
3. **Constraint Comparison**: Various constraint formulations
4. **Obstacle Density Tests**: Different obstacle scenarios
5. **Enhanced MPCC Tests**: C++ reference implementation

## Quick Start

### Basic Usage

```python
from pympc.testing.mpcc_test_framework import create_standard_mpcc_test

# Create and run a basic MPCC test
test = create_standard_mpcc_test(
    test_name="my_test",
    road=RoadConfig(road_type="curved", curvature_intensity=1.0),
    obstacles=ObstacleConfig(num_obstacles=3, intersection_probability=0.7),
    perception=PerceptionConfig(enabled=False)
)

result = test.run_test()
print(f"Test completed: {result['success']}")
```

### Enhanced MPCC (C++ Reference)

```python
from pympc.testing.enhanced_mpcc_framework import create_enhanced_mpcc_test

# Create enhanced MPCC test with C++ reference implementation
test = create_enhanced_mpcc_test(
    test_name="enhanced_test",
    road=RoadConfig(road_type="curved", curvature_intensity=1.0),
    obstacles=ObstacleConfig(num_obstacles=3, intersection_probability=0.7),
    mpc=MPCConfig(
        contouring_weight=2.0,
        lag_weight=1.0,
        velocity_weight=0.1,
        progress_weight=1.5
    )
)

result = test.run_test()
```

### Perception Area Testing

```python
from pympc.testing.mpcc_test_framework import create_perception_test

# Test with cone perception area
test = create_perception_test(
    test_name="perception_test",
    perception=PerceptionConfig(
        shape=PerceptionShape.CONE,
        distance=20.0,
        angle=np.pi/3,
        enabled=True
    )
)

result = test.run_test()
```

## Configuration Options

### Road Configuration
```python
road = RoadConfig(
    road_type="curved",        # "curved", "straight", "s_curve"
    length=120.0,              # Road length in meters
    width=6.0,                 # Road width in meters
    curvature_intensity=1.0,   # Curvature strength
    num_points=100              # Number of road points
)
```

### Vehicle Configuration
```python
vehicle = VehicleConfig(
    length=4.0,                # Vehicle length
    width=1.8,                 # Vehicle width
    wheelbase=2.5,             # Wheelbase
    max_velocity=15.0,         # Maximum velocity
    max_acceleration=3.0,      # Maximum acceleration
    max_steering_angle=0.5,    # Maximum steering angle
    dynamics_model="bicycle"   # Dynamics model
)
```

### MPC Configuration
```python
mpc = MPCConfig(
    horizon=15,                # Prediction horizon
    timestep=0.1,              # Time step
    max_steps=150,             # Maximum simulation steps
    contouring_weight=1.0,     # Contouring error weight
    lag_weight=1.0,            # Lag error weight
    velocity_weight=0.1,       # Velocity tracking weight
    progress_weight=1.0        # Progress maximization weight
)
```

### Obstacle Configuration
```python
obstacles = ObstacleConfig(
    num_obstacles=3,           # Number of obstacles
    radius_range=(0.8, 1.5),   # Obstacle radius range
    velocity_range=(2.0, 8.0), # Obstacle velocity range
    trajectory_type="random_walk", # "random_walk", "linear", "circular"
    intersection_probability=0.7    # Probability of road intersection
)
```

### Perception Configuration
```python
perception = PerceptionConfig(
    shape=PerceptionShape.CONE,    # "circle", "cone", "rectangle"
    distance=20.0,                # Maximum perception distance
    angle=np.pi/3,                # Cone angle (for cone shape)
    width=15.0,                   # Rectangle width
    height=20.0,                  # Rectangle height
    enabled=True                  # Enable/disable perception
)
```

## Test Runners

### Standardized Runner
```python
from pympc.testing.standardized_mpcc_runner import StandardizedMPCCRunner

runner = StandardizedMPCCRunner()
results = runner.run_standard_test_suite()
```

### Comprehensive Runner
```python
from pympc.testing.comprehensive_mpcc_runner import ComprehensiveMPCCRunner

runner = ComprehensiveMPCCRunner()
results = runner.run_comprehensive_suite()
```

## Example Scripts

### Basic Demo
```bash
python pympc/testing/mpcc_demo.py
```

### Example Usage
```bash
python pympc/testing/example_usage.py
```

### Comprehensive Testing
```bash
python pympc/testing/comprehensive_mpcc_runner.py
```

## Output Files

The framework automatically generates:

1. **GIF Animations**: Visual representation of vehicle trajectory with constraint overlays
2. **JSON Results**: Detailed test results with performance metrics
3. **Comparison Reports**: Analysis of different test configurations
4. **Comprehensive Reports**: Overall test suite analysis

## Mathematical Foundation

The framework is based on the C++ mpc_planner implementation from [https://github.com/tud-amr/mpc_planner](https://github.com/tud-amr/mpc_planner) and includes:

- **Contouring Objective**: Minimizes lateral and longitudinal path deviations
- **Lag Error**: Longitudinal deviation from reference path
- **Contouring Error**: Lateral deviation from reference path
- **Progress Maximization**: Maximizes progress along the path
- **Constraint Formulations**: Road boundary and obstacle avoidance constraints

## File Structure

```
pympc/testing/
├── mpcc_test_framework.py          # Core MPCC testing framework
├── enhanced_mpcc_framework.py      # Enhanced C++ reference implementation
├── standardized_mpcc_runner.py     # Standardized test runner
├── comprehensive_mpcc_runner.py    # Comprehensive test suite
├── mpcc_demo.py                    # Demo script
├── example_usage.py                # Usage examples
└── README.md                       # This file
```

## Dependencies

- numpy
- matplotlib
- casadi (for enhanced MPCC)
- scipy
- pathlib
- dataclasses
- enum

## Contributing

When adding new test configurations or constraint types:

1. Follow the existing configuration pattern
2. Ensure mathematical consistency with C++ reference
3. Add appropriate visualization support
4. Update documentation and examples

## License

This framework is part of the PyMPC project and follows the same licensing terms.
