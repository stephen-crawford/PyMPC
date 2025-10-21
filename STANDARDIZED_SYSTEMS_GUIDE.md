# PyMPC Standardized Systems Guide

This guide explains how to use the new standardized logging, visualization, and testing systems in PyMPC.

## Overview

The PyMPC codebase has been reworked to include three main standardized systems:

1. **Standardized Logging System** - Comprehensive logging with clear levels and diagnostics
2. **Standardized Visualization System** - Unified plotting interface for all tests
3. **Standardized Test Framework** - Easy test implementation with clear failure explanations

## 1. Standardized Logging System

### Features
- **Colored console output** with clear log levels
- **Test-specific logging** with automatic context
- **Performance monitoring** with timing information
- **Error tracking** with full context and suggestions
- **Diagnostic logging** for solver and constraint analysis

### Basic Usage

```python
from utils.standardized_logging import get_test_logger, LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR

# Get test logger
logger = get_test_logger("my_test", "INFO")

# Start test
logger.start_test()

# Log different types of information
logger.log_success("Operation completed successfully")
logger.log_warning("Potential issue detected", {"details": "value"})
logger.log_error("Operation failed", exception, {"context": "data"})
logger.log_debug("Debug information", data)

# End test
logger.end_test(success=True)
```

### Advanced Usage

```python
from utils.standardized_logging import PerformanceMonitor, DiagnosticLogger

# Performance monitoring
with PerformanceMonitor(logger, "expensive_operation"):
    result = expensive_operation()

# Diagnostic logging
diag_logger = DiagnosticLogger(logger)
diag_logger.log_solver_state("CasADiSolver", solver_state)
diag_logger.log_constraint_violation("road_boundary", 0.5, (0, 1), 1.5)
diag_logger.log_vehicle_state(vehicle_state, iteration)
```

## 2. Standardized Visualization System

### Features
- **Unified plotting interface** for all test types
- **Real-time and static visualization**
- **Automatic layout management**
- **Export capabilities** (PNG, GIF, MP4)
- **Interactive debugging tools**

### Basic Usage

```python
from utils.standardized_visualization import TestVisualizationManager, VisualizationConfig, VisualizationMode

# Create visualizer
visualizer = TestVisualizationManager("my_test")

# Configure visualization
config = VisualizationConfig(
    mode=VisualizationMode.REALTIME,
    save_plots=True,
    figure_size=(12, 8)
)
visualizer.initialize(config)

# Plot test setup
visualizer.plot_test_setup(environment_data)

# Update during test execution
visualizer.update_test_progress(state, trajectory_x, trajectory_y, iteration)

# Finalize
visualizer.finalize_test(success=True)
```

### Advanced Usage

```python
from utils.standardized_visualization import StandardizedVisualizer

# Create custom visualizer
visualizer = StandardizedVisualizer("my_test", config)

# Create different layouts
visualizer.create_figure("trajectory_analysis")  # 2x2 grid
visualizer.create_figure("mpc_debug")           # Debug layout

# Plot specific elements
visualizer.plot_environment(data)
visualizer.plot_vehicle(state, iteration)
visualizer.plot_trajectory(trajectory_x, trajectory_y)
visualizer.plot_constraints(constraint_data)
visualizer.plot_performance(performance_data)

# Save and animate
visualizer.save_plot("final_result.png")
visualizer.create_animation("test_animation.gif")
```

## 3. Standardized Test Framework

### Features
- **Easy test implementation** with abstract base class
- **Clear failure explanations** with diagnostic context
- **Automatic test discovery** and execution
- **Performance monitoring** and reporting
- **Integration** with logging and visualization

### Basic Test Implementation

```python
from test.framework.standardized_test import BaseMPCTest, TestConfig, TestResult

class MyMPCTest(BaseMPCTest):
    def __init__(self):
        config = TestConfig(
            test_name="my_mpc_test",
            description="Test MPC with scenario constraints",
            timeout=60.0,
            max_iterations=100,
            goal_tolerance=1.0,
            enable_visualization=True
        )
        super().__init__(config)
    
    def setup_test_environment(self):
        """Setup roads, obstacles, start/goal positions."""
        return {
            'start': (0, 0),
            'goal': (50, 0),
            'reference_path': generate_reference_path(),
            'dynamic_obstacles': create_obstacles(),
            'left_bound': create_road_boundaries(),
            'right_bound': create_road_boundaries()
        }
    
    def setup_mpc_system(self, data):
        """Setup solver, planner, constraints, objectives."""
        # Create solver
        solver = CasADiSolver()
        
        # Create planner
        planner = Planner(solver, vehicle)
        
        # Add constraints and objectives
        solver.module_manager.add_module(ContouringConstraints(solver))
        solver.module_manager.add_module(ScenarioConstraints(solver))
        solver.module_manager.add_module(ContouringObjective(solver))
        
        return planner, solver
    
    def execute_mpc_iteration(self, planner, data, iteration):
        """Execute one MPC iteration."""
        # Get current state
        current_state = planner.get_state()
        
        # Update data
        planner.update_data(data)
        
        # Solve MPC
        result = planner.solve()
        
        # Extract control inputs
        control_inputs = result.control_inputs
        
        # Apply control and return new state
        new_state = self.apply_control(current_state, control_inputs)
        planner.set_state(new_state)
        
        return new_state
    
    def check_goal_reached(self, state, goal):
        """Check if goal has been reached."""
        distance = np.linalg.norm([state['x'] - goal[0], state['y'] - goal[1]])
        return distance <= self.config.goal_tolerance

# Run the test
test = MyMPCTest()
result = test.run_test()
```

### Test Suite Usage

```python
from test.framework.standardized_test import TestSuite

# Create test suite
suite = TestSuite("MPC Integration Tests")

# Add tests
suite.add_test(MyMPCTest())
suite.add_test(AnotherMPCTest())
suite.add_test(YetAnotherMPCTest())

# Run all tests
results = suite.run_all_tests()
```

## 4. Debugging Tools

### Features
- **Constraint analysis** and violation detection
- **Solver diagnostics** and performance monitoring
- **Trajectory analysis** for optimization
- **Automatic problem detection** with solutions

### Usage

```python
from utils.debugging_tools import ConstraintAnalyzer, SolverDiagnostics, TrajectoryAnalyzer, ProblemDetector

# Initialize analyzers
constraint_analyzer = ConstraintAnalyzer()
solver_diagnostics = SolverDiagnostics()
trajectory_analyzer = TrajectoryAnalyzer()
problem_detector = ProblemDetector()

# During test execution
for iteration in range(max_iterations):
    # Analyze constraints
    violations = constraint_analyzer.analyze_constraint_violations(
        constraints, bounds, iteration
    )
    
    # Analyze solver performance
    diagnostic = solver_diagnostics.analyze_solver_performance(
        solver, solve_time, iteration
    )
    
    # Analyze trajectory
    trajectory_analysis = trajectory_analyzer.analyze_trajectory(
        trajectory_x, trajectory_y, reference_path
    )

# Detect problems
problems = problem_detector.detect_common_problems(
    solver_diagnostics, constraint_analyzer, trajectory_analyzer
)

# Generate reports
constraint_summary = constraint_analyzer.get_violation_summary()
solver_summary = solver_diagnostics.get_performance_summary()
problem_report = problem_detector.generate_problem_report()
```

## 5. Example: Complete Test Implementation

Here's a complete example showing how to implement a test using all the standardized systems:

```python
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test.framework.standardized_test import BaseMPCTest, TestConfig
from utils.standardized_logging import get_test_logger
from utils.standardized_visualization import VisualizationConfig, VisualizationMode
from utils.debugging_tools import ConstraintAnalyzer, SolverDiagnostics, TrajectoryAnalyzer


class ScenarioContouringTest(BaseMPCTest):
	"""Example test implementing scenario constraints with contouring."""

	def __init__(self):
		config = TestConfig(
			test_name="scenario_contouring_test",
			description="Test scenario constraints with contouring objective",
			timeout=120.0,
			max_iterations=200,
			goal_tolerance=1.0,
			enable_visualization=True,
			visualization_mode=VisualizationMode.REALTIME
		)
		super().__init__(config)

		# Initialize debugging tools
		self.constraint_analyzer = ConstraintAnalyzer()
		self.solver_diagnostics = SolverDiagnostics()
		self.trajectory_analyzer = TrajectoryAnalyzer()

	def setup_test_environment(self):
		"""Setup test environment with curved road and obstacles."""
		self.logger.log_phase("Environment Setup", "Creating curved road with obstacles")

		# Create curved reference path
		t = np.linspace(0, 1, 50)
		x_path = np.linspace(0, 50, 50)
		y_path = 3 * np.sin(2 * np.pi * t)
		s_path = np.linspace(0, 1, 50)

		reference_path = {
			'x': x_path, 'y': y_path, 's': s_path
		}

		# Create road boundaries
		normals = calculate_path_normals(reference_path)
		road_width = 8.0
		half_width = road_width / 2

		left_bound = {
			'x': x_path + normals[:, 0] * half_width,
			'y': y_path + normals[:, 1] * half_width,
			's': s_path
		}

		right_bound = {
			'x': x_path - normals[:, 0] * half_width,
			'y': y_path - normals[:, 1] * half_width,
			's': s_path
		}

		# Create dynamic obstacles
		obstacles = [
			{'x': 20, 'y': 2, 'radius': 1.0, 'type': 'gaussian'},
			{'x': 35, 'y': -1, 'radius': 0.8, 'type': 'gaussian'}
		]

		return {
			'start': (0, 0),
			'goal': (50, 0),
			'reference_path': reference_path,
			'left_bound': left_bound,
			'right_bound': right_bound,
			'dynamic_obstacles': obstacles
		}

	def setup_mpc_system(self, data):
		"""Setup MPC system with scenario and contouring constraints."""
		self.logger.log_phase("MPC Setup", "Initializing solver and modules")

		# Import required modules
		from solver.src.casadi_solver import CasADiSolver
		from planning.src.planner import Planner
		from planning.src.dynamic_models import ContouringSecondOrderUnicycleModel
		from planner.src.planner_modules.src.constraints.contouring_constraints import ContouringConstraints
		from planner.src.planner_modules.src.constraints.fixed_scenario_constraints import FixedScenarioConstraints
		from planner.src.planner_modules.src.objectives.contouring_objective import ContouringObjective

		# Create vehicle model
		vehicle = ContouringSecondOrderUnicycleModel()

		# Create solver
		solver = CasADiSolver()
		solver.set_dynamics_model(vehicle)

		# Create planner
		planner = Planner(solver, vehicle)

		# Add modules
		contouring_constraints = ContouringConstraints(solver)
		scenario_constraints = FixedScenarioConstraints(solver)
		contouring_objective = ContouringObjective(solver)

		solver.module_manager.add_module(contouring_constraints)
		solver.module_manager.add_module(scenario_constraints)
		solver.module_manager.add_module(contouring_objective)

		# Pass data to constraints
		contouring_constraints.on_data_received(data)

		# Initialize solver
		solver.define_parameters()

		self.logger.log_success("MPC system setup completed")

		return planner, solver

	def execute_mpc_iteration(self, planner, data, iteration):
		"""Execute one MPC iteration with diagnostics."""
		iteration_start = time.time()

		try:
			# Get current state
			current_state = planner.get_state()

			# Update data
			planner.update_data(data)

			# Solve MPC
			with PerformanceMonitor(self.logger, f"MPC_Solve_{iteration}"):
				result = planner.solve()

			# Analyze solver performance
			solve_time = time.time() - iteration_start
			diagnostic = self.solver_diagnostics.analyze_solver_performance(
				planner.solver, solve_time, iteration
			)

			# Extract control inputs
			if hasattr(result, 'control_inputs') and result.control_inputs:
				control_inputs = result.control_inputs
			else:
				# Fallback control
				self.logger.log_warning(f"No control inputs at iteration {iteration}, using fallback")
				control_inputs = self.generate_fallback_control(current_state, data)

			# Apply control
			new_state = self.apply_control(current_state, control_inputs)
			planner.set_state(new_state)

			# Log progress
			if iteration % 10 == 0:
				distance = np.linalg.norm([
					new_state.get('x', 0) - data['goal'][0],
					new_state.get('y', 0) - data['goal'][1]
				])
				self.logger.log_debug(f"Iteration {iteration}: Distance to goal: {distance:.3f}")

			return new_state

		except Exception as e:
			self.logger.log_error(f"MPC iteration {iteration} failed", e)
			# Use fallback control
			return self.execute_fallback_control(planner, data, iteration)

	def check_goal_reached(self, state, goal):
		"""Check if goal has been reached."""
		distance = np.linalg.norm([state.get('x', 0) - goal[0], state.get('y', 0) - goal[1]])
		return distance <= self.config.goal_tolerance

	def apply_control(self, state, control_inputs):
		"""Apply control inputs to get new state."""
		dt = 0.1

		# Extract control inputs
		if isinstance(control_inputs, dict):
			a = control_inputs.get('a', 0)
			w = control_inputs.get('w', 0)
		else:
			a, w = control_inputs[0], control_inputs[1]

		# Apply dynamics
		x = state.get('x', 0)
		y = state.get('y', 0)
		psi = state.get('psi', 0)
		v = state.get('v', 0)

		new_x = x + v * np.cos(psi) * dt
		new_y = y + v * np.sin(psi) * dt
		new_psi = psi + w * dt
		new_v = max(0, v + a * dt)
		new_spline = state.get('spline', 0) + v * dt

		return {
			'x': new_x, 'y': new_y, 'psi': new_psi,
			'v': new_v, 'spline': new_spline
		}

	def generate_fallback_control(self, state, data):
		"""Generate fallback control when MPC fails."""
		goal = data['goal']
		dx = goal[0] - state.get('x', 0)
		dy = goal[1] - state.get('y', 0)
		goal_angle = np.arctan2(dy, dx)

		angle_error = goal_angle - state.get('psi', 0)

		# Normalize angle error
		while angle_error > np.pi:
			angle_error -= 2 * np.pi
		while angle_error < -np.pi:
			angle_error += 2 * np.pi

		return {'a': 1.0, 'w': angle_error * 2.0}

	def execute_fallback_control(self, planner, data, iteration):
		"""Execute fallback control when MPC fails."""
		current_state = planner.get_state()
		control_inputs = self.generate_fallback_control(current_state, data)
		new_state = self.apply_control(current_state, control_inputs)
		planner.set_state(new_state)

		self.logger.log_warning(f"Using fallback control at iteration {iteration}")
		return new_state


# Helper functions
def calculate_path_normals(reference_path):
	"""Calculate path normals for road boundaries."""
	x = np.array(reference_path['x'])
	y = np.array(reference_path['y'])

	dx = np.gradient(x)
	dy = np.gradient(y)

	# Normalize
	norm = np.sqrt(dx ** 2 + dy ** 2)
	dx_norm = dx / norm
	dy_norm = dy / norm

	# Perpendicular vectors (normals)
	normals_x = -dy_norm
	normals_y = dx_norm

	return np.column_stack([normals_x, normals_y])


# Run the test
if __name__ == "__main__":
	test = ScenarioContouringTest()
	result = test.run_test()

	print(f"Test {'PASSED' if result.success else 'FAILED'}")
	print(f"Duration: {result.duration:.2f}s")
	print(f"Iterations: {result.iterations_completed}")
	print(f"Final distance: {result.final_distance_to_goal:.3f}")
```

## 6. Best Practices

### Logging
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- Include context and diagnostic information
- Use performance monitoring for expensive operations
- Log test phases and progress

### Visualization
- Choose appropriate layout for your test type
- Update visualization in real-time for debugging
- Save plots and animations for analysis
- Use consistent colors and styling

### Testing
- Implement all required abstract methods
- Handle MPC failures gracefully with fallback control
- Include comprehensive error checking
- Use debugging tools for problem detection

### Debugging
- Monitor constraint violations
- Track solver performance
- Analyze trajectory quality
- Use automatic problem detection

## 7. Migration Guide

To migrate existing tests to the new standardized systems:

1. **Replace logging calls**:
   ```python
   # Old
   print("Debug message")
   
   # New
   logger.log_debug("Debug message")
   ```

2. **Replace visualization code**:
   ```python
   # Old
   plt.plot(x, y)
   plt.show()
   
   # New
   visualizer.plot_trajectory(x, y)
   visualizer.update_test_progress(state, x, y, iteration)
   ```

3. **Convert to test framework**:
   ```python
   # Old
   def run_my_test():
       # test code here
   
   # New
   class MyTest(BaseMPCTest):
       def setup_test_environment(self):
           # environment setup
       def setup_mpc_system(self, data):
           # MPC setup
       # ... implement other methods
   ```

This standardized system makes tests easier to implement, modify, and debug while providing comprehensive diagnostics and clear failure explanations.
