# PyMPC Integration Test Framework

## Overview

The PyMPC Integration Test Framework provides a standardized way to run integration tests with configurable components. Each test outputs a timestamped folder containing all test artifacts.

## Features

- **Standardized Test Structure**: All tests follow the same pattern
- **Configurable Components**: Reference path, objectives, constraints, vehicle dynamics, obstacles
- **Automatic Output Generation**: CSV files, logs, animations
- **Timestamped Folders**: Each test creates a unique output folder
- **CONFIG.yml Integration**: All parameters read from configuration file

## Test Output Structure

Each test creates a folder named: `<timestamp>_objective_type_constraints_vehicle_type`

Example: `20241201_143022_contouring_safe_horizon_contouring_bicycle`

### Output Contents

- **`test.log`** - Complete execution log
- **`vehicle_states.csv`** - Vehicle trajectory data
- **`obstacle_*_states.csv`** - Individual obstacle trajectories
- **`animation.gif`** - Animated visualization of the test
- **`<test_script>.py`** - Copy of the test script

## Usage

### 1. Using the Test Runner Script

```bash
# List available predefined tests
python test/integration/run_integration_test.py --list-tests

# Run a predefined test
python test/integration/run_integration_test.py --test safe_horizon_basic

# Run a custom test
python test/integration/run_integration_test.py \
    --name "My Custom Test" \
    --objective contouring \
    --constraints safe_horizon contouring \
    --vehicle bicycle \
    --obstacles 3 \
    --obstacle-dynamics gaussian \
    --path-type curve \
    --path-length 25.0 \
    --duration 15.0
```

### 2. Using the Framework Directly

```python
from test.integration.integration_test_framework import IntegrationTestFramework, TestConfig, create_reference_path

# Create framework
framework = IntegrationTestFramework()

# Create test configuration
test_config = TestConfig(
    reference_path=create_reference_path("straight", 20.0),
    objective_module="contouring",
    constraint_modules=["safe_horizon", "contouring"],
    vehicle_dynamics="bicycle",
    num_obstacles=3,
    obstacle_dynamics=["gaussian", "gaussian", "gaussian"],
    test_name="My Test",
    duration=10.0,
    timestep=0.1
)

# Run test
result = framework.run_test(test_config)
```

### 3. Running Specific Tests

```bash
# Safe Horizon tests
python test/integration/safe_horizon_integration_test.py

# Individual framework tests
python test/integration/integration_test_framework.py
```

## Configuration Options

### Reference Paths

- **`straight`** - Straight line path
- **`curve`** - Curved path with sine wave
- **`s_curve`** - S-curve path

### Objective Modules

- **`contouring`** - Contouring objective
- **`goal`** - Goal-reaching objective

### Constraint Modules

- **`safe_horizon`** - Safe Horizon scenario-based constraints
- **`contouring`** - Contouring constraints
- **`gaussian`** - Gaussian uncertainty constraints
- **`linear`** - Linear constraints
- **`ellipsoid`** - Ellipsoid constraints

### Vehicle Dynamics

- **`bicycle`** - Bicycle model
- **`unicycle`** - Unicycle model

### Obstacle Dynamics

- **`gaussian`** - Gaussian prediction uncertainty
- **`deterministic`** - Deterministic prediction

## Predefined Tests

### 1. `safe_horizon_basic`
- **Purpose**: Basic Safe Horizon constraint test
- **Configuration**: Straight path, 3 Gaussian obstacles, bicycle model
- **Duration**: 10 seconds

### 2. `safe_horizon_advanced`
- **Purpose**: Advanced Safe Horizon with multiple constraints
- **Configuration**: Curved path, 4 mixed obstacles, bicycle model
- **Duration**: 15 seconds

### 3. `gaussian_constraints`
- **Purpose**: Traditional Gaussian constraints test
- **Configuration**: Straight path, 2 mixed obstacles, unicycle model
- **Duration**: 8 seconds

### 4. `multi_constraint`
- **Purpose**: Multiple constraint types test
- **Configuration**: S-curve path, 5 obstacles, bicycle model
- **Duration**: 12 seconds

### 5. `comparison_test`
- **Purpose**: Safe Horizon vs traditional comparison
- **Configuration**: Curved path, 3 Gaussian obstacles, bicycle model
- **Duration**: 10 seconds

## Example Test Results

### Successful Test Output
```
✅ TEST COMPLETED SUCCESSFULLY
📁 Output folder: test_outputs/20241201_143022_contouring_safe_horizon_contouring_bicycle
📊 Vehicle states recorded: 101
📊 Obstacle states recorded: [101, 101, 101]
⏱️  Average computation time: 0.045s
⚠️  Constraint violations: 0

📁 Generated files:
  - test.log (execution log)
  - vehicle_states.csv (vehicle trajectory)
  - obstacle_*_states.csv (obstacle trajectories)
  - animation.gif (trajectory visualization)
```

### CSV File Format

**vehicle_states.csv**:
```csv
time,x,y,theta,velocity
0.0,0.0,0.0,0.0,1.0
0.1,0.1,0.0,0.0,1.0
0.2,0.2,0.0,0.0,1.0
...
```

**obstacle_0_states.csv**:
```csv
time,x,y
0.0,5.0,0.0
0.1,5.2,0.1
0.2,5.4,0.2
...
```

## Configuration File Integration

The framework automatically reads from `config/CONFIG.yml`:

```yaml
# Key parameters used by the framework
horizon: 10
timestep: 0.1
robot:
  radius: 0.5
obstacle_radius: 0.35
safe_horizon:
  epsilon_p: 0.1
  beta: 0.01
  n_bar: 10
  num_scenarios: 100
  parallel_solvers: 4
```

## Extending the Framework

### Adding New Constraint Types

1. Add the constraint to `create_constraint_modules()` in `integration_test_framework.py`
2. Update the choices in `run_integration_test.py`
3. Add predefined test configurations

### Adding New Vehicle Dynamics

1. Add the dynamics model to `create_vehicle_dynamics()` in `integration_test_framework.py`
2. Update the choices in `run_integration_test.py`

### Adding New Objective Types

1. Add the objective to `create_objective_module()` in `integration_test_framework.py`
2. Update the choices in `run_integration_test.py`

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all required modules are installed
2. **Configuration Errors**: Check `CONFIG.yml` syntax
3. **Animation Issues**: Install `pillow` for GIF generation
4. **Memory Issues**: Reduce test duration or number of obstacles

### Debug Mode

Enable verbose logging by modifying the framework:

```python
logger.setLevel(logging.DEBUG)
```

## Performance Considerations

- **Computation Time**: Safe Horizon constraints are computationally intensive
- **Memory Usage**: Large numbers of scenarios require more memory
- **Animation Generation**: GIF creation can be slow for long tests
- **Parallel Solvers**: Configure `parallel_solvers` in CONFIG.yml for better performance

## Best Practices

1. **Test Naming**: Use descriptive test names
2. **Duration**: Keep tests under 20 seconds for reasonable execution time
3. **Obstacles**: Start with 2-3 obstacles and increase as needed
4. **Constraints**: Test individual constraints before combining multiple types
5. **Documentation**: Document custom test configurations
