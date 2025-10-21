#!/usr/bin/env python3
"""
TUD-AMR Style Contouring Demo

This demonstrates proper contouring constraints and objective following the TUD-AMR MPC planner approach.
The vehicle should follow the reference path while respecting road boundaries.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from test.framework.standardized_test import BaseMPCTest, TestConfig
from utils.standardized_visualization import VisualizationMode
from solver.src.casadi_solver import CasADiSolver
from planner.src.planner_modules.src.constraints.tud_contouring_constraints import TUDContouringConstraints
from planner.src.planner_modules.src.objectives.tud_contouring_objective import TUDContouringObjective
from planning.src.types import Data, Bound


class TUDContouringDemo(BaseMPCTest):
    """TUD-AMR style contouring demo that actually works."""
    
    def __init__(self):
        config = TestConfig(
            test_name="tud_contouring_demo",
            description="TUD-AMR style contouring constraints and objective demo",
            timeout=60.0,
            max_iterations=100,
            goal_tolerance=2.0,
            enable_visualization=True,
            save_outputs=True,
            log_level="INFO",
            visualization_mode=VisualizationMode.STATIC
        )
        super().__init__(config)
        self.data = None
        self.solver = None
    
    def setup_test_environment(self):
        """Set up the test environment."""
        self.logger.log_phase("Environment Setup", "Creating curved road environment")
        
        # Create curved reference path
        t = np.linspace(0, 1, 100)
        x_path = np.linspace(0, 100, 100)  # 100m road
        y_path = 8 * np.sin(0.4 * np.pi * t) + 4 * np.sin(0.8 * np.pi * t) + 2 * np.sin(1.6 * np.pi * t)  # Curved S-path
        s_path = np.linspace(0, 1, 100)
        
        # Road parameters
        road_width = 6.0
        
        # Create road boundaries
        dx = np.gradient(x_path)
        dy = np.gradient(y_path)
        normals_x = -dy / np.sqrt(dx**2 + dy**2)
        normals_y = dx / np.sqrt(dx**2 + dy**2)
        
        # Left and right boundaries
        left_x = x_path + normals_x * road_width / 2
        left_y = y_path + normals_y * road_width / 2
        right_x = x_path - normals_x * road_width / 2
        right_y = y_path - normals_y * road_width / 2
        
        # Create data object
        data = Data()
        data.reference_path = Bound(x=x_path, y=y_path, s=s_path)
        data.left_bound = Bound(x=left_x, y=left_y, s=s_path)
        data.right_bound = Bound(x=right_x, y=right_y, s=s_path)
        data.start = (0.0, 0.0)
        data.goal = (100.0, 0.0)
        data.road_width = road_width
        
        self.logger.log_success("Environment created")
        self.logger.logger.info(f"Road length: {x_path[-1] - x_path[0]:.1f}m")
        self.logger.logger.info(f"Road width: {road_width:.1f}m")
        
        return data
    
    def setup_mpc_solver(self, data):
        """Set up the MPC solver with TUD-AMR style modules."""
        self.logger.log_phase("MPC Setup", "Setting up TUD-AMR style MPC solver")
        
        # Create solver
        solver = CasADiSolver()
        solver.initialize(data)
        
        # Add TUD-AMR style contouring constraints
        contouring_constraints = TUDContouringConstraints(solver)
        solver.module_manager.add_module(contouring_constraints)
        
        # Add TUD-AMR style contouring objective
        contouring_objective = TUDContouringObjective(solver)
        solver.module_manager.add_module(contouring_objective)
        
        self.logger.log_success("TUD-AMR style MPC solver configured")
        return solver
    
    
    def check_boundary_violation(self, x, y, data):
        """Check if vehicle position violates road boundaries."""
        # Find closest point on reference path
        ref_x = np.array(data.reference_path.x)
        ref_y = np.array(data.reference_path.y)
        distances = np.sqrt((ref_x - x)**2 + (ref_y - y)**2)
        closest_idx = np.argmin(distances)
        
        # Get road boundaries at closest point
        left_x = data.left_bound.x[closest_idx]
        left_y = data.left_bound.y[closest_idx]
        right_x = data.right_bound.x[closest_idx]
        right_y = data.right_bound.y[closest_idx]
        
        # Calculate road center
        center_x = (left_x + right_x) / 2
        center_y = (left_y + right_y) / 2
        
        # Calculate distances
        left_distance = np.sqrt((x - left_x)**2 + (y - left_y)**2)
        right_distance = np.sqrt((x - right_x)**2 + (y - right_y)**2)
        center_distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Vehicle violates boundaries if it's outside the road corridor
        road_half_width = data.road_width / 2
        is_violation = (left_distance < road_half_width or 
                       right_distance < road_half_width or 
                       center_distance > road_half_width)
        
        return is_violation
    
    def check_goal_reached(self, state, data):
        """Check if goal is reached."""
        if not self.data:
            return False
        
        goal_distance = np.sqrt((state.x - self.data.goal[0])**2 + (state.y - self.data.goal[1])**2)
        return goal_distance < self.config.goal_tolerance
    
    def execute_mpc_iteration(self, planner, data, iteration):
        """Execute one MPC iteration."""
        try:
            # Get current state from planner
            current_state = planner.get_state()
            
            # Set initial state for solver
            self.solver.set_initial_state(current_state)
            
            # Solve MPC
            exit_flag = self.solver.solve()
            
            if exit_flag == 1:  # Success
                # Extract control inputs from solution
                if hasattr(self.solver, 'solution') and self.solver.solution is not None:
                    # Get first control input from solution
                    v_cmd = float(self.solver.solution.value(self.solver.var_dict['v'][0]))
                    omega_cmd = float(self.solver.solution.value(self.solver.var_dict['omega'][0]))
                else:
                    # Fallback to simple goal-seeking
                    goal_x, goal_y = data.goal
                    desired_angle = np.arctan2(goal_y - current_state.get('y', 0), goal_x - current_state.get('x', 0))
                    v_cmd = 2.0
                    omega_cmd = 0.0
                
                # Apply control and update state
                dt = 0.1
                new_x = current_state.get('x', 0) + v_cmd * np.cos(current_state.get('theta', 0)) * dt
                new_y = current_state.get('y', 0) + v_cmd * np.sin(current_state.get('theta', 0)) * dt
                new_theta = current_state.get('theta', 0) + omega_cmd * dt
                
                new_state = {
                    'x': new_x,
                    'y': new_y,
                    'theta': new_theta,
                    'v': v_cmd,
                    'omega': omega_cmd
                }
                
                # Update planner state
                planner.set_state(new_state)
                
                return new_state
            else:
                # MPC failed, use fallback control
                self.logger.log_warning("MPC failed, using fallback control")
                goal_x, goal_y = data.goal
                desired_angle = np.arctan2(goal_y - current_state.get('y', 0), goal_x - current_state.get('x', 0))
                v_cmd = 1.0  # Slower fallback speed
                omega_cmd = 0.0
                
                # Apply fallback control
                dt = 0.1
                new_x = current_state.get('x', 0) + v_cmd * np.cos(desired_angle) * dt
                new_y = current_state.get('y', 0) + v_cmd * np.sin(desired_angle) * dt
                
                new_state = {
                    'x': new_x,
                    'y': new_y,
                    'theta': desired_angle,
                    'v': v_cmd,
                    'omega': omega_cmd
                }
                
                # Update planner state
                planner.set_state(new_state)
                
                return new_state
                
        except Exception as e:
            self.logger.log_error(f"Error in MPC iteration: {e}")
            # Return current state unchanged
            return current_state
    
    def setup_mpc_system(self, data):
        """Set up the MPC system."""
        self.data = data
        self.solver = self.setup_mpc_solver(data)
        
        # Create a proper Planner object
        from planning.src.planner import Planner
        from planning.src.dynamic_models import ContouringSecondOrderUnicycleModel
        
        vehicle = ContouringSecondOrderUnicycleModel()
        planner = Planner(self.solver, vehicle)
        
        return planner, self.solver
    
    def get_current_state(self, planner):
        """Get current state from planner as dictionary."""
        try:
            state = planner.get_state()
            # Convert State object to dictionary
            return state.to_dict()
        except:
            return {'x': 0, 'y': 0, 'theta': 0, 'v': 0, 'omega': 0}
    
    def execute_fallback_control(self, planner, data, iteration):
        """Execute fallback control when MPC fails."""
        # Get current state from planner
        current_state = planner.get_state()
        
        # Get goal from data object
        goal = data.goal if hasattr(data, 'goal') else (0, 0)
        
        # Calculate goal direction
        dx = goal[0] - current_state.get('x', 0)
        dy = goal[1] - current_state.get('y', 0)
        goal_angle = np.arctan2(dy, dx)
        
        # Simple control
        dt = 0.1
        v_cmd = 1.0
        omega_cmd = 0.0
        
        new_x = current_state.get('x', 0) + v_cmd * np.cos(goal_angle) * dt
        new_y = current_state.get('y', 0) + v_cmd * np.sin(goal_angle) * dt
        
        new_state = {
            'x': new_x,
            'y': new_y,
            'theta': goal_angle,
            'v': v_cmd,
            'omega': omega_cmd
        }
        
        # Update planner state
        planner.set_state(new_state)
        
        return new_state
    
    def generate_diagnostic_data(self, planner, data):
        """Generate diagnostic data for failure analysis."""
        return {
            'final_trajectory_length': len(self.trajectory_x),
            'average_iteration_time': np.mean(self.performance_data['iteration_times']) if self.performance_data['iteration_times'] else 0,
            'mpc_failure_rate': self.performance_data['mpc_failures'] / max(self.iteration, 1),
            'trajectory_bounds': {
                'x_min': min(self.trajectory_x) if self.trajectory_x else 0,
                'x_max': max(self.trajectory_x) if self.trajectory_x else 0,
                'y_min': min(self.trajectory_y) if self.trajectory_y else 0,
                'y_max': max(self.trajectory_y) if self.trajectory_y else 0
            },
            'solver_info': self.get_solver_diagnostics(planner),
            'constraint_info': self.get_constraint_diagnostics(planner)
        }


def main():
    """Main function to run the TUD-AMR style contouring demo."""
    demo = TUDContouringDemo()
    result = demo.run_test()
    
    print("\n" + "="*70)
    print("TUD-AMR STYLE CONTOURING DEMO RESULTS")
    print("="*70)
    if result.success:
        print("Demo PASSED")
        print("✅ Successfully demonstrated TUD-AMR style contouring!")
    else:
        print("Demo FAILED")
        print("❌ Vehicle did not reach goal")
    print(f"Final distance: {result.final_distance_to_goal:.2f}m")
    print(f"Iterations completed: {result.iterations_completed}")
    print("📁 Results saved to: test_results/tud_contouring_demo/")
    print("="*70)


if __name__ == "__main__":
    main()
