# Safe Horizon Constraint Module

This module implements scenario-based safe horizon constraints with support tracking for Model Predictive Control (MPC). It provides probabilistic safety guarantees through scenario optimization theory.

## Overview

The Safe Horizon Constraint module integrates scenario-based constraints into the MPC framework, allowing the robot to navigate safely in dynamic environments with uncertain obstacle predictions. The module uses scenario optimization to provide probabilistic safety guarantees.

## Key Features

- **Scenario Sampling**: Samples scenarios from obstacle predictions (Gaussian, multimodal)
- **Collision Constraint Formulation**: Linearizes collision constraints into half-spaces
- **Free-Space Polytope Construction**: Builds polytopes representing safe regions
- **Support Tracking**: Tracks active constraints and maintains support sets
- **Big-M Relaxation**: Optional scenario removal for computational efficiency
- **MPC Integration**: Seamless integration with existing MPC framework
- **Parallel Processing**: Multi-threaded scenario optimization

## Architecture

### Core Components

1. **SafeHorizonConstraint**: Main constraint class inheriting from BaseConstraint
2. **SafeHorizonModule**: Core scenario optimization logic
3. **ScenarioSampler**: Handles scenario sampling from obstacle predictions
4. **DiscManager**: Manages constraints for robot discs
5. **Math Utils**: Mathematical utilities for polytope operations

### File Structure

```
planner_modules/src/constraints/
├── safe_horizon_constraint.py          # Main constraint class
└── scenario_utils/
    ├── __init__.py
    ├── scenario_module.py             # Core scenario module
    ├── sampler.py                     # Scenario sampling
    └── math_utils.py                  # Mathematical utilities

test/
└── integration/constraints/safe_horizon/
    ├── __init__.py
    └── safe_horizon_integration_test.py
```

## Configuration

Add the following to your `CONFIG.yml`:

```yaml
safe_horizon:
  epsilon_p: 0.1                    # Constraint violation probability
  beta: 0.01                       # Confidence level (1 - beta is confidence)
  n_bar: 10                        # Support dimension
  num_removal: 0                   # Number of scenarios to remove
  max_constraints_per_disc: 24     # Maximum constraints per disc
  use_slack: true                  # Enable slack variables
  num_scenarios: 100               # Number of scenarios to sample
  enable_outlier_removal: true     # Enable outlier removal
  parallel_solvers: 4              # Number of parallel solvers
```

## Usage

### Basic Integration

```python
from planner_modules.src.constraints.safe_horizon_constraint import SafeHorizonConstraint
from solver.src.casadi_solver import CasADiSolver

# Create solver and constraint
solver = CasADiSolver()
safe_horizon_constraint = SafeHorizonConstraint(solver)

# Add to module manager
module_manager = solver.get_module_manager()
module_manager.add_module(safe_horizon_constraint)
```

### Data Requirements

The constraint requires dynamic obstacles with Gaussian or multimodal predictions:

```python
from planning.src.types import DynamicObstacle, PredictionType, PredictionStep

# Create dynamic obstacle
obstacle = DynamicObstacle(
    index=0,
    position=np.array([5.0, 0.0]),
    angle=0.0,
    radius=0.5
)

# Set Gaussian prediction
obstacle.prediction.type = PredictionType.GAUSSIAN
obstacle.prediction.steps = []

# Add prediction steps
for i in range(horizon_length):
    step = PredictionStep(
        position=np.array([5.0 + i * 0.2, 0.0]),
        angle=0.0,
        major_radius=0.5,
        minor_radius=0.5
    )
    obstacle.prediction.steps.append(step)
```

### MPC Integration Hooks

The constraint provides hooks for MPC integration:

```python
# Prepare iteration
safe_horizon_constraint.prepare_iteration(x_init, perception_data)

# Run optimization (called by MPC solver)
result = safe_horizon_constraint.optimize(state, data)

# Post-solve processing
safe_horizon_constraint.post_solve(x_solution, u_solution)
```

## Theory

### Scenario Optimization

The module uses scenario optimization theory to provide probabilistic safety guarantees:

- **Sample Size**: Computed as `n ≥ (2/ε) * ln(1/β) + 2*n̄ + (2*n̄/ε) * ln(2/ε)`
- **Violation Probability**: ε (epsilon_p) - probability of constraint violation
- **Confidence Level**: β (beta) - confidence in the probabilistic guarantee
- **Support Dimension**: n̄ (n_bar) - dimension of the support set

### Collision Constraints

Collision constraints are linearized as:
```
a₁x + a₂y ≤ b
```
where `(a₁, a₂)` is the normal vector and `b` is the safety margin.

### Free-Space Polytopes

The intersection of all scenario constraints forms a polytope representing the free space:
```
P = {x ∈ ℝ² : Aᵢx ≤ bᵢ, ∀i ∈ scenarios}
```

## Testing

### Unit Tests

Run the unit tests:
```bash
python -m pytest planner_modules/test/safe_horizon_test.py -v
```

### Integration Tests

Run the comprehensive integration tests:
```bash
python -m pytest test/integration/constraints/safe_horizon/safe_horizon_integration_test.py -v
```

### Demo

Run the demonstration:
```python
from test.integration.constraints.safe_horizon.safe_horizon_integration_test import SafeHorizonMPCDemo

demo = SafeHorizonMPCDemo()
demo.run_demo()
```

## Performance Considerations

- **Parallel Processing**: Uses multiple scenario solvers for faster optimization
- **Scenario Removal**: Big-M relaxation reduces computational load
- **Outlier Removal**: Statistical filtering improves constraint quality
- **Support Tracking**: Early stopping when support limits exceeded

## Troubleshooting

### Common Issues

1. **Data Not Ready**: Ensure obstacles have valid Gaussian predictions
2. **Parameter Validation Failed**: Check epsilon_p, beta, and n_bar values
3. **No Feasible Solution**: Try increasing num_scenarios or adjusting parameters
4. **Support Exceeded**: Reduce n_bar or increase num_removal

### Debugging

Enable debug logging:
```python
from utils.utils import LOG_DEBUG
LOG_DEBUG("Safe Horizon debugging enabled")
```

Check constraint info:
```python
info = safe_horizon_constraint.get_constraint_info()
print(f"Constraint info: {info}")
```

## Future Enhancements

- **Adaptive Sampling**: Dynamic scenario sampling based on environment complexity
- **Multi-Modal Predictions**: Enhanced support for complex prediction models
- **Real-Time Adaptation**: Online parameter tuning
- **Visualization**: Real-time constraint visualization tools

## References

- Calafiore, G. C., & Campi, M. C. (2006). The scenario approach to robust control design.
- Schildbach, G., et al. (2014). Scenario model predictive control for lane change assistance.
- Hewing, L., et al. (2020). Cautious model predictive control using Gaussian process regression.
