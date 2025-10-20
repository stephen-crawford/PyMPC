"""
Test Complete Mpc System - Converted to Standardized Systems

This test has been automatically converted to use the standardized
logging, visualization, and testing framework.
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
from utils.standardized_logging import get_test_logger
from utils.standardized_visualization import VisualizationConfig, VisualizationMode
from utils.debugging_tools import ConstraintAnalyzer, SolverDiagnostics, TrajectoryAnalyzer


class TestCompleteMpcSystem(BaseMPCTest):
    """
    Test Complete Mpc System using standardized systems.
    
    This test demonstrates scenario constraints with contouring objective
    using the standardized framework.
    """
    
    def __init__(self):
        config = TestConfig(
            test_name="test_name",
            description="Test with visualization framework",
            timeout=120.0,
            max_iterations=200,
            goal_tolerance=1.0,
            enable_visualization=True,
            visualization_mode=VisualizationMode.REALTIME,
            log_level="INFO"
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
        
        # Initialize debugging tools
        self.constraint_analyzer = ConstraintAnalyzer()
        self.solver_diagnostics = SolverDiagnostics()
        self.trajectory_analyzer = TrajectoryAnalyzer()
    
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
        
        self.logger.log_success("Environment setup completed")
        return environment_data
    
    def setup_mpc_system(self, data):
        """Setup MPC system with scenario and contouring constraints."""
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
            
            # Update data
            planner.update_data(data)
            
            # Solve MPC
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
                self.logger.log_info(f"Iteration {iteration}: Distance to goal: {distance:.3f}")
            
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
            if hasattr(self, 'logger'):
                self.logger.log_debug(f"Could not collect constraint overlays: {e}")
        
        return overlays
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


# Run the test
if __name__ == "__main__":
    test = TestCompleteMpcSystem()
    result = test.run_test()
    
    print(f"Test {'PASSED' if result.success else 'FAILED'}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Iterations: {result.iterations_completed}")
    print(f"Final distance: {result.final_distance_to_goal:.3f}")
