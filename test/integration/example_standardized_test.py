"""
Example Standardized Test Implementation

This example demonstrates how to use the new standardized logging,
visualization, and testing systems for PyMPC tests.
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from test.framework.standardized_test import BaseMPCTest, TestConfig
from utils.standardized_logging import get_test_logger, PerformanceMonitor
from utils.standardized_visualization import VisualizationConfig, VisualizationMode
from utils.debugging_tools import ConstraintAnalyzer, SolverDiagnostics, TrajectoryAnalyzer


class ExampleStandardizedTest(BaseMPCTest):
    """
    Example test demonstrating the standardized systems.
    
    This test shows how to:
    - Use standardized logging with clear diagnostics
    - Implement real-time visualization
    - Handle MPC failures gracefully
    - Use debugging tools for problem detection
    """
    
    def __init__(self):
        config = TestConfig(
            test_name="example_standardized_test",
            description="Example test demonstrating standardized systems",
            timeout=60.0,
            max_iterations=100,
            goal_tolerance=1.0,
            enable_visualization=True,
            visualization_mode=VisualizationMode.REALTIME,
            log_level="DEBUG"
        )
        super().__init__(config)
        
        # Initialize debugging tools
        self.constraint_analyzer = ConstraintAnalyzer()
        self.solver_diagnostics = SolverDiagnostics()
        self.trajectory_analyzer = TrajectoryAnalyzer()
        
        self.logger.log_success("Example standardized test initialized")
    
    def setup_test_environment(self):
        """Setup test environment with curved road and obstacles."""
        self.logger.log_phase("Environment Setup", "Creating test environment")
        
        # Create curved reference path
        t = np.linspace(0, 1, 50)
        x_path = np.linspace(0, 50, 50)
        y_path = 3 * np.sin(2 * np.pi * t)
        s_path = np.linspace(0, 1, 50)
        
        reference_path = {
            'x': x_path, 'y': y_path, 's': s_path
        }
        
        # Create road boundaries
        normals = self.calculate_path_normals(reference_path)
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
        
        environment_data = {
            'start': (0, 0),
            'goal': (50, 0),
            'reference_path': reference_path,
            'left_bound': left_bound,
            'right_bound': right_bound,
            'dynamic_obstacles': obstacles
        }
        
        self.logger.log_success("Environment setup completed", {
            'path_length': len(x_path),
            'obstacles': len(obstacles),
            'road_width': road_width
        })
        
        return environment_data
    
    def setup_mpc_system(self, data):
        """Setup MPC system with constraints and objectives."""
        self.logger.log_phase("MPC System Setup", "Initializing solver and modules")
        
        try:
            # Import required modules
            from solver.src.casadi_solver import CasADiSolver
            from planning.src.planner import Planner
            from planning.src.dynamic_models import ContouringSecondOrderUnicycleModel
            from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
            from planner_modules.src.constraints.fixed_scenario_constraints import FixedScenarioConstraints
            from planner_modules.src.objectives.contouring_objective import ContouringObjective
            
            # Create vehicle model
            vehicle = ContouringSecondOrderUnicycleModel()
            self.logger.log_debug("Created vehicle model", {
                'model_type': type(vehicle).__name__
            })
            
            # Create solver
            solver = CasADiSolver()
            solver.set_dynamics_model(vehicle)
            self.logger.log_debug("Created solver", {
                'solver_type': type(solver).__name__
            })
            
            # Create planner
            planner = Planner(solver, vehicle)
            self.logger.log_debug("Created planner", {
                'planner_type': type(planner).__name__
            })
            
            # Add modules
            contouring_constraints = ContouringConstraints(solver)
            scenario_constraints = FixedScenarioConstraints(solver)
            contouring_objective = ContouringObjective(solver)
            
            solver.module_manager.add_module(contouring_constraints)
            solver.module_manager.add_module(scenario_constraints)
            solver.module_manager.add_module(contouring_objective)
            
            self.logger.log_debug("Added modules", {
                'modules': [
                    type(contouring_constraints).__name__,
                    type(scenario_constraints).__name__,
                    type(contouring_objective).__name__
                ]
            })
            
            # Pass data to constraints
            contouring_constraints.on_data_received(data)
            scenario_constraints.on_data_received(data)
            
            # Initialize solver
            solver.define_parameters()
            
            self.logger.log_success("MPC system setup completed")
            
            return planner, solver
            
        except Exception as e:
            self.logger.log_error("Failed to setup MPC system", e)
            raise
    
    def execute_mpc_iteration(self, planner, data, iteration):
        """Execute one MPC iteration with comprehensive diagnostics."""
        iteration_start = time.time()
        
        try:
            # Get current state
            current_state = planner.get_state()
            self.logger.log_debug(f"Current state at iteration {iteration}", {
                'position': (current_state.get('x', 0), current_state.get('y', 0)),
                'velocity': current_state.get('v', 0),
                'heading': current_state.get('psi', 0)
            })
            
            # Update data
            planner.update_data(data)
            
            # Solve MPC with performance monitoring
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
                self.logger.log_debug("Extracted control inputs", {
                    'inputs': control_inputs
                })
            else:
                # Fallback control
                self.logger.log_warning(f"No control inputs at iteration {iteration}, using fallback")
                control_inputs = self.generate_fallback_control(current_state, data)
            
            # Apply control
            new_state = self.apply_control(current_state, control_inputs)
            planner.set_state(new_state)
            
            # Update trajectory data for analysis
            self.trajectory_analyzer.trajectory_data = {
                'x': self.trajectory_x + [new_state.get('x', 0)],
                'y': self.trajectory_y + [new_state.get('y', 0)]
            }
            
            # Log progress
            if iteration % 10 == 0:
                distance = np.linalg.norm([
                    new_state.get('x', 0) - data['goal'][0],
                    new_state.get('y', 0) - data['goal'][1]
                ])
                self.logger.log_info(f"Iteration {iteration}: Distance to goal: {distance:.3f}")
                
                # Analyze trajectory quality
                if len(self.trajectory_x) > 5:
                    trajectory_analysis = self.trajectory_analyzer.analyze_trajectory(
                        self.trajectory_x, self.trajectory_y, data.get('reference_path')
                    )
                    if trajectory_analysis['issues']:
                        self.logger.log_warning("Trajectory quality issues detected", trajectory_analysis['issues'])
            
            return new_state
            
        except Exception as e:
            self.logger.log_error(f"MPC iteration {iteration} failed", e)
            # Use fallback control
            return self.execute_fallback_control(planner, data, iteration)
    
    def check_goal_reached(self, state, goal):
        """Check if goal has been reached."""
        distance = np.linalg.norm([state.get('x', 0) - goal[0], state.get('y', 0) - goal[1]])
        reached = distance <= self.config.goal_tolerance
        
        if reached:
            self.logger.log_success(f"Goal reached! Final distance: {distance:.3f}")
        
        return reached
    
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
        
        new_state = {
            'x': new_x, 'y': new_y, 'psi': new_psi, 
            'v': new_v, 'spline': new_spline
        }
        
        self.logger.log_debug("Applied control", {
            'control': (a, w),
            'new_position': (new_x, new_y),
            'new_velocity': new_v
        })
        
        return new_state
    
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
        
        control = {'a': 1.0, 'w': angle_error * 2.0}
        
        self.logger.log_debug("Generated fallback control", {
            'goal_angle': goal_angle,
            'angle_error': angle_error,
            'control': control
        })
        
        return control
    
    def execute_fallback_control(self, planner, data, iteration):
        """Execute fallback control when MPC fails."""
        current_state = planner.get_state()
        control_inputs = self.generate_fallback_control(current_state, data)
        new_state = self.apply_control(current_state, control_inputs)
        planner.set_state(new_state)
        
        self.logger.log_warning(f"Using fallback control at iteration {iteration}")
        return new_state
    
    def calculate_path_normals(self, reference_path):
        """Calculate path normals for road boundaries."""
        x = np.array(reference_path['x'])
        y = np.array(reference_path['y'])
        
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        # Normalize
        norm = np.sqrt(dx**2 + dy**2)
        dx_norm = dx / norm
        dy_norm = dy / norm
        
        # Perpendicular vectors (normals)
        normals_x = -dy_norm
        normals_y = dx_norm
        
        return np.column_stack([normals_x, normals_y])


def run_example_test():
    """Run the example standardized test."""
    print("🚀 Running Example Standardized Test")
    print("=" * 50)
    
    # Create and run test
    test = ExampleStandardizedTest()
    result = test.run_test()
    
    # Print results
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    print(f"Test: {result.test_name}")
    print(f"Success: {'✅ PASSED' if result.success else '❌ FAILED'}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Iterations: {result.iterations_completed}")
    print(f"Final distance to goal: {result.final_distance_to_goal:.3f}")
    
    if not result.success:
        print(f"Failure reason: {result.failure_reason}")
        print(f"Errors: {len(result.errors)}")
        print(f"Warnings: {len(result.warnings)}")
    
    # Print performance metrics
    if result.performance_metrics:
        print(f"\n📈 PERFORMANCE METRICS")
        print(f"MPC failures: {result.performance_metrics.get('mpc_failures', 0)}")
        if result.performance_metrics.get('iteration_times'):
            avg_time = np.mean(result.performance_metrics['iteration_times'])
            print(f"Average iteration time: {avg_time:.3f}s")
    
    return result


if __name__ == "__main__":
    result = run_example_test()
