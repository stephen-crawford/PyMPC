"""
Simple Standardized Test Runner

This script demonstrates the standardized test framework with proper visualization.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test.framework.standardized_test import BaseMPCTest, TestConfig
from utils.standardized_visualization import VisualizationConfig, VisualizationMode

# Import MPC components
from solver.src.casadi_solver import CasADiSolver
from planning.src.planner import Planner
from planning.src.dynamic_models import ContouringSecondOrderUnicycleModel
from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
from planner_modules.src.constraints.scenario_constraints import ScenarioConstraints
from planner_modules.src.objectives.goal_objective import GoalObjective
from planner_modules.src.objectives.contouring_objective import ContouringObjective
from planning.src.types import Data, ReferencePath


class SimpleStandardizedTest(BaseMPCTest):
    """
    Simple standardized test to demonstrate the framework.
    """
    
    def __init__(self):
        config = TestConfig(
            test_name="simple_standardized_test",
            description="Simple test to demonstrate standardized framework",
            timeout=60.0,
            max_iterations=50,
            goal_tolerance=1.0,
            enable_visualization=True,
            visualization_mode=VisualizationMode.REALTIME,
            log_level="INFO"
        )
        super().__init__(config)
        
        # Test state
        self.mpc_system = None
        self.current_state = None
        self.reference_path = None
        self.obstacles = []
    
    def setup_test_environment(self):
        """Setup simple test environment."""
        self.logger.log_phase("Environment Setup", "Creating simple test environment")
        
        # Create simple reference path
        t = np.linspace(0, 1, 50)
        x_path = np.linspace(0, 20, 50)
        y_path = 2 * np.sin(2 * np.pi * t)
        s_path = np.linspace(0, 1, 50)
        
        self.reference_path = {
            'x': x_path, 'y': y_path, 's': s_path
        }
        
        # Create simple obstacles with proper structure
        self.obstacles = [
            {
                'x': 10, 'y': 1, 'radius': 1.0, 'type': 'static',
                'prediction': {'type': 'static', 'uncertainty': [0.1, 0.1]}
            },
            {
                'x': 15, 'y': -1, 'radius': 0.8, 'type': 'dynamic', 
                'velocity': [0.2, 0.1],
                'prediction': {'type': 'dynamic', 'uncertainty': [0.2, 0.2]}
            }
        ]
        
        # Update visualization
        if self.visualizer:
            self.visualizer.update_reference_path(self.reference_path)
            self.visualizer.update_obstacles(self.obstacles)
            self.visualizer.update_goal((20, 0))
        
        self.logger.log_success("Environment setup complete")
    
    def create_mpc_system(self):
        """Create simple MPC system."""
        self.logger.log_phase("System Creation", "Creating MPC system")
        
        # Create solver
        solver = CasADiSolver(timestep=0.1, horizon=10)
        
        # Create vehicle model
        model = ContouringSecondOrderUnicycleModel()
        solver.set_dynamics_model(model)
        
        # Create planner
        planner = Planner(solver, model)
        
        # Create modules
        goal_module = GoalObjective(solver)
        contouring_objective = ContouringObjective(solver)
        contouring_constraints = ContouringConstraints(solver)
        scenario_constraints = ScenarioConstraints(solver)
        
        # Add modules to solver
        solver.module_manager.add_module(goal_module)
        solver.module_manager.add_module(contouring_objective)
        solver.module_manager.add_module(contouring_constraints)
        solver.module_manager.add_module(scenario_constraints)
        
        # Create test data
        data = Data()
        data.start = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        data.goal = [20.0, 0.0]
        data.goal_received = True
        data.planning_start_time = 0.0
        
        # Set reference path
        ref_path = ReferencePath()
        ref_path.set('x', self.reference_path['x'])
        ref_path.set('y', self.reference_path['y'])
        ref_path.set('s', self.reference_path['s'])
        data.reference_path = ref_path
        
        # Set road boundaries
        left_bound = ReferencePath()
        left_bound.set('x', self.reference_path['x'])
        left_bound.set('y', self.reference_path['y'] + 2.0)
        right_bound = ReferencePath()
        right_bound.set('x', self.reference_path['x'])
        right_bound.set('y', self.reference_path['y'] - 2.0)
        
        data.left_bound = left_bound
        data.right_bound = right_bound
        
        # Set obstacles
        data.dynamic_obstacles = self.obstacles
        
        # Initialize system
        planner.initialize(data)
        
        self.mpc_system = {
            'planner': planner,
            'solver': solver,
            'data': data,
            'model': model,
            'contouring_constraints': contouring_constraints,
            'scenario_constraints': scenario_constraints
        }
        
        self.logger.log_success("MPC system created successfully")
        return self.mpc_system
    
    def run_mpc_iteration(self, iteration: int) -> tuple[bool, float, dict]:
        """Run a single MPC iteration."""
        try:
            # Update obstacles
            self._update_dynamic_obstacles(iteration)
            
            # Run MPC solve
            result = self.mpc_system['planner'].solve_mpc(self.mpc_system['data'])
            
            if not result.success:
                self.logger.log_warning(f"Iteration {iteration}: MPC solve failed")
                return False, float('inf'), {}
            
            # Extract state information
            if result.trajectory_history:
                current_traj = result.trajectory_history[0]
                self.current_state = {
                    'x': current_traj[0],
                    'y': current_traj[1],
                    'psi': current_traj[2],
                    'vx': current_traj[3],
                    'vy': current_traj[4],
                    'omega': current_traj[5]
                }
            else:
                self.current_state = {
                    'x': 0, 'y': 0, 'psi': 0, 'vx': 0, 'vy': 0, 'omega': 0
                }
            
            # Calculate error
            error = self._calculate_error()
            
            # Get constraint information
            constraints = self._get_constraint_info()
            constraint_projections = self._get_constraint_projections()
            
            # Update constraint overlays
            self._update_constraint_overlays()
            
            # Create state dictionary
            state = {
                **self.current_state,
                'constraints': constraints,
                'constraint_projections': constraint_projections,
                'iteration': iteration,
                'error': error
            }
            
            return True, error, state
            
        except Exception as e:
            self.logger.log_error(f"Iteration {iteration} failed: {str(e)}")
            return False, float('inf'), {}
    
    def _update_dynamic_obstacles(self, iteration: int):
        """Update dynamic obstacle positions."""
        dt = 0.1
        
        for obs in self.obstacles:
            if obs.get('type') == 'dynamic' and 'velocity' in obs:
                vx, vy = obs['velocity']
                obs['x'] += vx * dt
                obs['y'] += vy * dt
        
        # Update data
        self.mpc_system['data'].dynamic_obstacles = self.obstacles
        
        # Update visualization
        if self.visualizer:
            self.visualizer.update_obstacles(self.obstacles)
    
    def _calculate_error(self) -> float:
        """Calculate current error."""
        if not self.current_state:
            return float('inf')
        
        # Distance to goal
        goal_x, goal_y = 20.0, 0.0
        current_x, current_y = self.current_state['x'], self.current_state['y']
        distance_to_goal = np.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)
        
        return distance_to_goal
    
    def _get_constraint_info(self) -> list:
        """Get constraint information for visualization."""
        constraints = []
        
        # Add scenario constraints
        try:
            scenario_module = self.mpc_system['scenario_constraints']
            params = scenario_module.parameter_manager.get_parameters()
            
            for i in range(len(self.obstacles)):
                if f'scenario_a1_{i}' in params:
                    a1 = params[f'scenario_a1_{i}']
                    a2 = params[f'scenario_a2_{i}']
                    b = params[f'scenario_b_{i}']
                    
                    constraints.append({
                        'type': 'halfspace',
                        'A': [a1, a2],
                        'b': b,
                        'module': 'scenario',
                        'obstacle_id': i
                    })
        except Exception as e:
            self.logger.log_debug(f"Error getting scenario constraints: {e}")
        
        return constraints
    
    def _get_constraint_projections(self) -> list:
        """Get constraint projections for visualization."""
        projections = []
        
        if not self.current_state:
            return projections
        
        x, y = self.current_state['x'], self.current_state['y']
        
        # Add projections for each obstacle
        for i, obs in enumerate(self.obstacles):
            obs_x, obs_y = obs['x'], obs['y']
            radius = obs['radius']
            
            # Calculate distance to obstacle
            distance = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            
            # Create projection point
            if distance > 0:
                proj_x = obs_x + (x - obs_x) * radius / distance
                proj_y = obs_y + (y - obs_y) * radius / distance
                
                projections.append({
                    'type': 'point',
                    'x': proj_x,
                    'y': proj_y,
                    'color': 'cyan',
                    'size': 30,
                    'obstacle_id': i
                })
        
        return projections
    
    def _update_constraint_overlays(self):
        """Update constraint overlays from modules."""
        if not self.visualizer:
            return
        
        # Get overlays from constraint modules
        for module_name in ['contouring_constraints', 'scenario_constraints']:
            if module_name in self.mpc_system:
                try:
                    module = self.mpc_system[module_name]
                    overlay = module.get_visualization_overlay()
                    if overlay:
                        self.add_constraint_overlay(module_name, overlay)
                except Exception as e:
                    self.logger.log_debug(f"Error getting overlay from {module_name}: {e}")


def run_simple_test():
    """Run the simple standardized test."""
    print("Running Simple Standardized Test")
    print("=" * 40)
    
    # Create and run test
    test = SimpleStandardizedTest()
    result = test.run_test()
    
    # Print results
    print("\nTest Results:")
    print(test.get_test_summary())
    
    return result


if __name__ == "__main__":
    run_simple_test()
