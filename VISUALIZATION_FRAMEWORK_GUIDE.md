# PyMPC Visualization Framework Guide

## Overview

The PyMPC visualization framework provides comprehensive visualization capabilities for MPC tests, including real-time plotting, constraint overlays, and animation generation. This guide covers all aspects of the framework and how to use it effectively.

## Table of Contents

1. [Framework Components](#framework-components)
2. [Visualization Configuration](#visualization-configuration)
3. [Constraint Overlays](#constraint-overlays)
4. [Test Integration](#test-integration)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Framework Components

### Core Components

- **`StandardizedVisualizer`**: Main visualization engine
- **`VisualizationConfig`**: Configuration management
- **`TestVisualizationManager`**: Test-specific visualization management
- **`BaseConstraint.get_visualization_overlay()`**: Constraint overlay interface

### Visualization Modes

- **`STATIC`**: Generate static plots (default)
- **`REALTIME`**: Real-time plot updates
- **`DEBUG`**: Enhanced debugging visualization

## Visualization Configuration

### Basic Configuration

```python
from utils.standardized_visualization import VisualizationConfig, VisualizationMode

# Basic configuration
config = VisualizationConfig(
    mode=VisualizationMode.REALTIME,
    save_plots=True,
    show_plots=True,
    output_dir="test_results/my_test/visualizations"
)
```

### Advanced Configuration

```python
# Enhanced configuration with all options
config = VisualizationConfig(
    mode=VisualizationMode.REALTIME,
    realtime=True,                    # Enable real-time updates
    show_constraint_projection=True,   # Show constraint overlays
    save_animation=True,               # Generate GIF animations
    save_plots=True,                  # Save static plots
    show_plots=True,                  # Display plots during execution
    fps=10,                           # Animation frame rate
    dpi=100,                          # Plot resolution
    output_dir="test_results/my_test/visualizations",
    figure_size=(12, 8),              # Plot dimensions
    colors={                          # Custom color scheme
        'vehicle': '#1f77b4',
        'trajectory': '#ff7f0e',
        'constraints': '#ff9896'
    }
)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mode` | `VisualizationMode` | `STATIC` | Visualization mode |
| `realtime` | `bool` | `False` | Enable real-time updates |
| `show_constraint_projection` | `bool` | `False` | Show constraint overlays |
| `save_animation` | `bool` | `False` | Generate GIF animations |
| `save_plots` | `bool` | `False` | Save static plots |
| `show_plots` | `bool` | `False` | Display plots during execution |
| `fps` | `int` | `10` | Animation frame rate |
| `dpi` | `int` | `100` | Plot resolution |
| `output_dir` | `str` | `"test_outputs"` | Output directory |
| `figure_size` | `Tuple[int, int]` | `(12, 8)` | Plot dimensions |

## Constraint Overlays

### Supported Constraint Types

The framework supports visualization overlays for all constraint types:

- **Scenario Constraints**: Halfspace projections
- **Linear Constraints**: Linear halfspace lines
- **Gaussian Constraints**: Elliptical uncertainty regions
- **Ellipsoid Constraints**: Ellipsoidal constraint boundaries
- **Decomposition Constraints**: Corridor polygons
- **Contouring Constraints**: Road corridor visualization

### Overlay Data Format

Constraint overlays use a standardized format:

```python
overlay = {
    'halfspaces': [  # List of (a, b, c) tuples for ax + by + c = 0
        (1.0, 0.0, -5.0),  # Vertical line at x = 5
        (0.0, 1.0, -3.0)   # Horizontal line at y = 3
    ],
    'polygons': [    # List of polygon dictionaries
        {
            'x': [0, 1, 1, 0],      # X coordinates
            'y': [0, 0, 1, 1],      # Y coordinates
            'color': '#66ccff',     # Fill color
            'alpha': 0.15           # Transparency
        }
    ],
    'points': [      # List of (x, y) tuples
        (2.5, 3.0),
        (4.0, 1.5)
    ]
}
```

### Implementing Constraint Overlays

To add visualization overlays to a constraint module:

```python
class MyConstraint(BaseConstraint):
    def get_visualization_overlay(self):
        """Return visualization overlay for this constraint."""
        try:
            # Collect constraint data
            halfspaces = []
            polygons = []
            points = []
            
            # Add constraint-specific visualization data
            # ... (constraint-specific logic)
            
            return {
                'halfspaces': halfspaces,
                'polygons': polygons,
                'points': points
            }
        except Exception:
            return None  # Safe fallback
```

## Test Integration

### Standardized Test Framework

All integration tests use the standardized framework with built-in visualization support:

```python
from test.framework.standardized_test import BaseMPCTest, TestConfig
from utils.standardized_visualization import VisualizationConfig, VisualizationMode

class MyTest(BaseMPCTest):
    def __init__(self):
        config = TestConfig(
            test_name="my_test",
            enable_visualization=True,
            visualization_mode=VisualizationMode.REALTIME
        )
        super().__init__(config)
        
        # Enhanced visualization configuration
        self.viz_config = VisualizationConfig(
            mode=VisualizationMode.REALTIME,
            realtime=True,
            show_constraint_projection=True,
            save_animation=True,
            save_plots=True,
            fps=10,
            dpi=100,
            output_dir=f"test_results/{config.test_name}/visualizations"
        )
        
        # Initialize enhanced visualizer
        if config.enable_visualization:
            self.visualizer = TestVisualizationManager(config.test_name)
            self.visualizer.initialize(self.viz_config)
```

### Constraint Overlay Collection

The framework automatically collects constraint overlays from active modules:

```python
def _collect_constraint_overlays(self, planner):
    """Collect constraint overlays from active modules."""
    overlays = {'halfspaces': [], 'polygons': [], 'points': []}
    
    try:
        if hasattr(planner, 'solver') and hasattr(planner.solver, 'module_manager'):
            modules = getattr(planner.solver.module_manager, 'modules', [])
            for module in modules:
                if hasattr(module, 'get_visualization_overlay'):
                    overlay = module.get_visualization_overlay()
                    if overlay:
                        if 'halfspaces' in overlay:
                            overlays['halfspaces'].extend(overlay['halfspaces'])
                        if 'polygons' in overlay:
                            overlays['polygons'].extend(overlay['polygons'])
                        if 'points' in overlay:
                            overlays['points'].extend(overlay['points'])
    except Exception as e:
        self.logger.log_debug(f"Could not collect constraint overlays: {e}")
    
    return overlays
```

## Usage Examples

### Example 1: Basic Test with Visualization

```python
from test.framework.standardized_test import BaseMPCTest, TestConfig
from utils.standardized_visualization import VisualizationConfig, VisualizationMode

class BasicTest(BaseMPCTest):
    def __init__(self):
        config = TestConfig(
            test_name="basic_test",
            enable_visualization=True,
            visualization_mode=VisualizationMode.REALTIME
        )
        super().__init__(config)
    
    def setup_test_environment(self):
        # Setup test environment
        return {
            'start': (0, 0),
            'goal': (10, 10),
            'obstacles': [...]
        }
    
    def setup_mpc_system(self, data):
        # Setup MPC system
        return planner, solver
    
    def execute_mpc_iteration(self, planner, data, iteration):
        # Execute MPC iteration
        return new_state
    
    def check_goal_reached(self, state, goal):
        # Check if goal reached
        return distance_to_goal < tolerance
```

### Example 2: Advanced Test with Constraint Overlays

```python
class AdvancedTest(BaseMPCTest):
    def __init__(self):
        config = TestConfig(
            test_name="advanced_test",
            enable_visualization=True,
            visualization_mode=VisualizationMode.REALTIME
        )
        super().__init__(config)
        
        # Enhanced visualization with constraint overlays
        self.viz_config = VisualizationConfig(
            mode=VisualizationMode.REALTIME,
            realtime=True,
            show_constraint_projection=True,
            save_animation=True,
            save_plots=True,
            fps=15,
            dpi=150,
            output_dir=f"test_results/{config.test_name}/visualizations"
        )
        
        if config.enable_visualization:
            self.visualizer = TestVisualizationManager(config.test_name)
            self.visualizer.initialize(self.viz_config)
```

### Example 3: Custom Constraint Overlay

```python
class CustomConstraint(BaseConstraint):
    def get_visualization_overlay(self):
        """Custom constraint overlay implementation."""
        try:
            # Get current constraint data
            halfspaces = []
            for i in range(self.num_active_constraints):
                a1 = self._a1[i]
                a2 = self._a2[i]
                b = self._b[i]
                if a1 is not None and a2 is not None and b is not None:
                    halfspaces.append((a1, a2, -b))
            
            return {'halfspaces': halfspaces}
        except Exception:
            return None
```

## Best Practices

### 1. Performance Optimization

- Use `realtime=False` for large-scale testing
- Set appropriate `fps` values (10-15 for most cases)
- Use `dpi=100` for standard quality, `dpi=150` for high quality
- Disable `show_plots=True` for headless environments

### 2. Memory Management

- Use `save_animation=False` unless specifically needed
- Clear visualization data after test completion
- Use appropriate `figure_size` for your use case

### 3. Constraint Overlay Design

- Keep overlays lightweight and informative
- Use consistent colors across constraint types
- Provide meaningful alpha values for transparency
- Handle exceptions gracefully in overlay methods

### 4. Test Organization

- Use descriptive test names
- Organize output directories by test type
- Include timestamps in output filenames
- Document visualization configuration choices

## Troubleshooting

### Common Issues

#### 1. Import Errors

```
ModuleNotFoundError: No module named 'utils.standardized_visualization'
```

**Solution**: Ensure project root is in Python path:
```python
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
```

#### 2. Visualization Not Showing

**Check**:
- `show_plots=True` in configuration
- `enable_visualization=True` in test config
- Proper visualizer initialization

#### 3. Constraint Overlays Not Appearing

**Check**:
- `show_constraint_projection=True` in configuration
- Constraint modules implement `get_visualization_overlay()`
- Overlay data format is correct

#### 4. Animation Generation Fails

**Check**:
- `save_animation=True` in configuration
- Sufficient disk space for output
- Proper animation data collection

### Debug Mode

Enable debug mode for detailed visualization information:

```python
config = VisualizationConfig(
    mode=VisualizationMode.DEBUG,
    show_plots=True,
    save_plots=True
)
```

### Performance Monitoring

Monitor visualization performance:

```python
# Check visualization timing
if hasattr(self.visualizer, 'plot_times'):
    avg_time = np.mean(self.visualizer.plot_times)
    print(f"Average plot time: {avg_time:.3f}s")
```

## Advanced Features

### Custom Color Schemes

```python
colors = {
    'vehicle': '#1f77b4',
    'trajectory': '#ff7f0e',
    'reference_path': '#2ca02c',
    'road_boundaries': '#d62728',
    'obstacles': '#9467bd',
    'goal': '#2ca02c',
    'start': '#ff7f0e',
    'constraints': '#ff9896',
    'violations': '#ff0000'
}

config = VisualizationConfig(colors=colors)
```

### Multi-Panel Layouts

The framework supports various layout configurations:

- **Trajectory Analysis**: Main trajectory + performance metrics
- **Constraint Analysis**: Constraint violations + solver diagnostics
- **Debug Mode**: Enhanced debugging information

### Export Options

```python
# Export static plots
visualizer.save_plot("trajectory_analysis.png")

# Export animations
visualizer.create_animation("test_animation.gif")

# Export comprehensive report
report = visualizer.generate_report()
```

## Conclusion

The PyMPC visualization framework provides comprehensive visualization capabilities for MPC testing. By following this guide, you can effectively use the framework to create informative visualizations, debug constraint issues, and generate professional test reports.

For additional support or feature requests, refer to the project documentation or contact the development team.
