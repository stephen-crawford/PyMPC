# MPC Visualization and Logging Framework

## Overview

This comprehensive framework provides easy-to-use tools for setting up MPC tests with contouring objectives and arbitrary constraints. It includes advanced visualization, logging, performance monitoring, and demo capabilities.

## Features

### 🎯 **Easy Test Configuration**
- **TestConfigBuilder**: Fluent API for creating test configurations
- **Predefined Configurations**: Ready-to-use test scenarios
- **Flexible Constraint Setup**: Easy addition of multiple constraint types
- **Custom Objectives**: Support for contouring, goal-reaching, and custom objectives

### 📊 **Advanced Visualization**
- **2D/3D Trajectory Plots**: High-quality trajectory visualization
- **State Evolution Plots**: Time-series plots of state variables
- **Control Evolution Plots**: Control input visualization
- **Constraint Analysis**: Visual representation of constraints
- **Performance Metrics**: Bar charts and trend analysis
- **Animations**: Animated trajectory playback

### 📝 **Comprehensive Logging**
- **Structured Logging**: Detailed optimization logs
- **Performance Tracking**: Solve times, iterations, convergence
- **Trajectory Analysis**: Path length, tracking error, velocity analysis
- **Session Management**: Organized log files by session
- **JSON Export**: Machine-readable log data

### ⚡ **Performance Monitoring**
- **Real-time Metrics**: Live performance tracking
- **Benchmark Analysis**: Comparative performance analysis
- **Problem Size Analysis**: Variables vs. constraints analysis
- **Trajectory Quality**: Path quality metrics
- **Memory Usage**: Resource utilization tracking

### 🚀 **Demo Framework**
- **Predefined Demos**: Ready-to-run demonstration scenarios
- **Custom Demos**: Easy creation of custom test scenarios
- **Benchmark Suites**: Systematic performance testing
- **Comparison Reports**: Side-by-side performance analysis
- **Session Reports**: Comprehensive analysis summaries

## Quick Start

### 1. Basic Usage

```python
from utils.demo_framework import create_demo_framework
from utils.test_config import TestConfigBuilder

# Create demo framework
demo = create_demo_framework(
	output_dir="my_demo_outputs",
	enable_logging=True,
	enable_visualization=True,
	enable_performance_monitoring=True
)

# Create test configuration
config = (TestConfigBuilder("my_test")
		  .set_mpc_params(horizon_length=20, dt=0.1)
		  .add_contouring_objective(reference_path)
		  .add_linear_constraints(state_bounds={'min': [...], 'max': [...]})
		  .add_ellipsoid_constraints(obstacles, safety_margin=0.3)
		  .set_initial_state([0.0, 0.0, 0.0, 1.0, 0.0])
		  .build())

# Run demo
result = demo.run_demo(config, mpc_solver)
```

### 2. Using Predefined Configurations

```python
from utils.test_config import PredefinedTestConfigs

# Use predefined configuration
config = PredefinedTestConfigs.curving_road_ellipsoid()
result = demo.run_demo(config, mpc_solver)
```

### 3. Running Multiple Demos

```python
# Run predefined demos
demo_names = ["curving_road_ellipsoid", "curving_road_gaussian", "goal_reaching"]
results = demo.run_predefined_demos(demo_names, mpc_solver)

# Generate comparison report
comparison_report = demo.create_comparison_report(demo_names)
```

## Framework Components

### 1. Test Configuration System (`pympc.utils.test_config`)

#### TestConfigBuilder
```python
config = (TestConfigBuilder("test_name")
          .set_mpc_params(horizon_length=20, dt=0.1)
          .set_dynamics("bicycle", wheelbase=2.5)
          .add_contouring_objective(reference_path)
          .add_goal_objective(goal_state)
          .add_linear_constraints(state_bounds, control_bounds)
          .add_ellipsoid_constraints(obstacles)
          .add_gaussian_constraints(uncertain_obstacles)
          .add_scenario_constraints(scenarios)
          .set_initial_state([0, 0, 0, 1, 0])
          .build())
```

#### Predefined Configurations
- `curving_road_ellipsoid()`: Curving road with ellipsoid obstacles
- `curving_road_gaussian()`: Curving road with uncertain obstacles
- `curving_road_scenario()`: Curving road with scenario constraints
- `goal_reaching()`: Goal reaching with obstacles
- `combined_constraints()`: All constraint types combined

### 2. Logging System (`pympc.utils.logger`)

#### MPCLogger
```python
logger = MPCLogger(log_dir="logs", log_level="INFO")

# Log optimization start
logger.log_optimization_start(test_name, horizon_length, dt, state_dim, control_dim)

# Log optimization end
logger.log_optimization_end(success=True, solve_time=1.5, objective_value=10.2)

# Log trajectory analysis
logger.log_trajectory_analysis(trajectory, reference_path, obstacles)

# Get session summary
logger.print_session_summary()
```

#### MPCProfiler
```python
profiler = MPCProfiler(logger)

# Profile sections
profiler.start_profile("setup")
# ... setup code ...
profiler.end_profile("setup")

# Get profile summary
profile_summary = profiler.get_profile_summary()
```

### 3. Visualization System (`pympc.utils.advanced_visualizer`)

#### MPCVisualizer
```python
visualizer = MPCVisualizer(figsize=(12, 8), save_dir="plots")

# 2D trajectory plot
visualizer.plot_trajectory_2d(
    trajectory, reference_path, obstacles,
    title="MPC Trajectory", save=True, show=True
)

# State evolution plot
visualizer.plot_state_evolution(
    trajectory, dt=0.1, state_names=['x', 'y', 'yaw', 'v', 'delta']
)

# Control evolution plot
visualizer.plot_control_evolution(
    controls, dt=0.1, control_names=['acceleration', 'steering_rate']
)

# Performance metrics plot
visualizer.plot_performance_metrics(metrics)

# Create animation
animation = visualizer.create_animation(trajectory, obstacles)
```

### 4. Performance Monitoring (`pympc.utils.performance_monitor`)

#### PerformanceMonitor
```python
monitor = PerformanceMonitor(log_dir="performance_logs")

# Start monitoring
monitor.start_test(test_name, test_config)

# Record metrics
monitor.record_solve_time(solve_time)
monitor.record_optimization_metrics(iterations, objective_value)
monitor.record_trajectory_metrics(trajectory, reference_path)

# End monitoring
metrics = monitor.end_test(success=True)

# Generate reports
summary = monitor.get_performance_summary()
monitor.plot_performance_comparison()
monitor.generate_performance_report()
```

### 5. Demo Framework (`pympc.utils.demo_framework`)

#### MPCDemoFramework
```python
demo = create_demo_framework(
    output_dir="demo_outputs",
    enable_logging=True,
    enable_visualization=True,
    enable_performance_monitoring=True
)

# Run single demo
result = demo.run_demo(config, mpc_solver)

# Run predefined demos
results = demo.run_predefined_demos(demo_names, mpc_solver)

# Run benchmark suite
benchmark_results = demo.run_benchmark_suite(solver_func, benchmark_configs)

# Generate reports
comparison_report = demo.create_comparison_report(demo_names)
session_report = demo.generate_session_report()
```

## Example Scenarios

### 1. Simple Contouring Control

```python
# Create curving road
t = np.linspace(0, 4*np.pi, 100)
x = 0.3 * t
y = 2 * np.sin(0.3 * t)
road_path = np.column_stack([x, y])

# Create obstacles
obstacles = []
for i in range(20):
    x_obs = 1.0 + 0.4 * i
    y_obs = 1.0 + 0.5 * np.sin(0.1 * i)
    obstacles.append({
        'center': np.array([x_obs, y_obs]),
        'shape': np.array([0.5, 0.3]),
        'rotation': 0.0
    })

# Create configuration
config = (TestConfigBuilder("simple_contouring")
          .set_mpc_params(horizon_length=20, dt=0.1)
          .add_contouring_objective(road_path)
          .add_linear_constraints(
              state_bounds={
                  'min': [-20, -10, -np.pi, 0, -0.6],
                  'max': [20, 10, np.pi, 4, 0.6]
              }
          )
          .add_ellipsoid_constraints(obstacles, safety_margin=0.3)
          .set_initial_state([0.0, 0.0, 0.0, 1.5, 0.0])
          .build())

# Run demo
result = demo.run_demo(config, mpc_solver)
```

### 2. Multiple Constraint Types

```python
config = (TestConfigBuilder("multiple_constraints")
          .set_mpc_params(horizon_length=25, dt=0.1)
          .add_contouring_objective(road_path)
          .add_linear_constraints(state_bounds, control_bounds)
          .add_ellipsoid_constraints(ellipsoid_obstacles)
          .add_gaussian_constraints(uncertain_obstacles)
          .add_scenario_constraints(scenarios)
          .set_initial_state([0.0, 0.0, 0.0, 2.0, 0.0])
          .build())
```

### 3. Goal Reaching

```python
config = (TestConfigBuilder("goal_reaching")
          .set_mpc_params(horizon_length=30, dt=0.1)
          .add_goal_objective(goal_state, goal_weight=1.0, terminal_weight=10.0)
          .add_linear_constraints(state_bounds)
          .add_ellipsoid_constraints(obstacles)
          .set_initial_state([0.0, 0.0, 0.0, 1.0, 0.0])
          .build())
```

## Output Structure

```
demo_outputs/
├── demo_session_YYYYMMDD_HHMMSS/
│   ├── logs/
│   │   ├── mpc_optimization.log
│   │   ├── optimization_data.json
│   │   └── session_summary.json
│   ├── plots/
│   │   ├── trajectory_2d_001.png
│   │   ├── state_evolution_002.png
│   │   ├── control_evolution_003.png
│   │   └── performance_comparison.png
│   ├── performance/
│   │   ├── performance_metrics.json
│   │   ├── performance_report.json
│   │   └── performance_data.csv
│   └── session_report.json
```

## Advanced Features

### 1. Custom Solver Integration

```python
def my_mpc_solver(config):
    """Custom MPC solver function."""
    # Your MPC implementation here
    return {
        'success': True,
        'states': trajectory,
        'controls': controls,
        'solve_time': solve_time,
        'iterations': iterations,
        'objective_value': objective_value
    }

# Use with demo framework
result = demo.run_demo(config, my_mpc_solver)
```

### 2. Performance Analysis

```python
# Generate performance plots
monitor.plot_performance_comparison(metric='solve_time')
monitor.plot_solve_time_trend()
monitor.plot_problem_size_analysis()
monitor.plot_trajectory_quality()

# Export data
csv_file = monitor.export_to_csv()
```

### 3. Custom Visualizations

```python
# Create custom plots
visualizer.plot_constraint_analysis(trajectory, constraints)
visualizer.create_animation(trajectory, obstacles, reference_path)
```

## Best Practices

### 1. Configuration Management
- Use descriptive test names
- Set appropriate horizon lengths and time steps
- Choose constraint parameters carefully
- Save configurations for reuse

### 2. Performance Optimization
- Monitor solve times and iterations
- Analyze problem size vs. performance
- Use appropriate solver options
- Consider constraint complexity

### 3. Visualization
- Save plots for documentation
- Use consistent color schemes
- Include reference paths and obstacles
- Animate complex scenarios

### 4. Logging
- Enable detailed logging for debugging
- Use structured log formats
- Monitor convergence and feasibility
- Track performance trends

## Troubleshooting

### Common Issues

1. **Optimization Fails**
   - Check constraint feasibility
   - Verify initial conditions
   - Adjust solver options
   - Reduce problem complexity

2. **Visualization Errors**
   - Ensure trajectory data is valid
   - Check obstacle format
   - Verify reference path dimensions
   - Handle empty data gracefully

3. **Performance Issues**
   - Monitor memory usage
   - Optimize constraint formulation
   - Use appropriate horizon lengths
   - Consider problem scaling

### Debug Tips

1. **Enable Verbose Logging**
   ```python
   logger = MPCLogger(log_level="DEBUG")
   ```

2. **Check Constraint Feasibility**
   ```python
   # Verify constraints are feasible
   for constraint in config.constraints:
       print(f"Constraint type: {constraint['type']}")
   ```

3. **Monitor Performance**
   ```python
   # Track solve times
   monitor.plot_solve_time_trend()
   ```

## Examples

See the following example scripts:
- `simple_demo_example.py`: Basic usage examples
- `demo_with_visualization.py`: Comprehensive demo framework
- `test_curving_road_simple.py`: Curving road test scenarios

## Conclusion

This framework provides a comprehensive solution for MPC testing, visualization, and analysis. It makes it easy to set up tests with contouring objectives and arbitrary constraints while providing detailed logging, performance monitoring, and visualization capabilities.

The framework is designed to be:
- **Easy to use**: Simple API for common tasks
- **Flexible**: Support for custom configurations and solvers
- **Comprehensive**: Full logging, visualization, and analysis
- **Production-ready**: Robust error handling and performance monitoring

Start with the simple examples and gradually explore the advanced features to get the most out of the framework.
