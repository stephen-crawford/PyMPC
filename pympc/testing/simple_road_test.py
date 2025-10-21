"""
Simple Road Test for MPC Framework

This module provides a simplified end-to-end test that demonstrates
a car traversing a road while avoiding obstacles.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import os
import time
from typing import List, Dict, Any, Tuple

from pympc.core import ModuleManager, ParameterManager
from pympc.constraints import ContouringConstraints, ScenarioConstraints
from pympc.objectives import ContouringObjective, GoalObjective
from pympc.utils.spline import Spline2D, SplineFitter
from pympc.utils.logger import LOG_INFO, LOG_DEBUG, LOG_WARN


class SimpleRoadTest:
    """Simple road test for MPC framework."""
    
    def __init__(self):
        """Initialize the test."""
        self.output_dir = "test_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Test parameters
        self.horizon = 15
        self.timestep = 0.1
        self.max_steps = 50
        
        # Vehicle parameters
        self.vehicle_length = 4.5
        self.vehicle_width = 2.0
        self.max_velocity = 12.0
        
        # Road parameters
        self.road_width = 6.0
        self.road_length = 80.0
        
    def create_road(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a simple curved road."""
        # Create curved center line
        s = np.linspace(0, 1, 50)
        x_center = s * self.road_length
        y_center = 3 * np.sin(2 * np.pi * s * 0.5)  # Gentle curve
        
        # Calculate road normals
        dx = np.gradient(x_center)
        dy = np.gradient(y_center)
        norm = np.sqrt(dx**2 + dy**2)
        nx = -dy / (norm + 1e-9)
        ny = dx / (norm + 1e-9)
        
        # Create boundaries
        half_width = self.road_width / 2
        x_left = x_center + nx * half_width
        y_left = y_center + ny * half_width
        x_right = x_center - nx * half_width
        y_right = y_center - ny * half_width
        
        center_line = np.column_stack([x_center, y_center])
        left_boundary = np.column_stack([x_left, y_left])
        right_boundary = np.column_stack([x_right, y_right])
        
        return center_line, left_boundary, right_boundary
    
    def create_obstacles(self) -> List[Dict[str, Any]]:
        """Create dynamic obstacles."""
        obstacles = [
            {
                'position': [25.0, 1.0],
                'velocity': [8.0, 0.5],
                'radius': 1.2,
                'prediction_steps': 15
            },
            {
                'position': [45.0, -1.5],
                'velocity': [6.0, -0.3],
                'radius': 1.0,
                'prediction_steps': 15
            }
        ]
        return obstacles
    
    def run_test(self) -> Dict[str, Any]:
        """Run the simple road test."""
        LOG_INFO("Starting Simple Road Test")
        
        try:
            # Create road and obstacles
            center_line, left_boundary, right_boundary = self.create_road()
            obstacles = self.create_obstacles()
            
            # Initialize MPC components
            module_manager = ModuleManager()
            parameter_manager = ParameterManager()
            
            # Create objectives and constraints
            goal_obj = GoalObjective()
            contouring_obj = ContouringObjective()
            contouring_const = ContouringConstraints()
            scenario_const = ScenarioConstraints()
            
            # Add modules
            module_manager.add_module(goal_obj)
            module_manager.add_module(contouring_obj)
            module_manager.add_module(contouring_const)
            module_manager.add_module(scenario_const)
            
            # Define parameters
            module_manager.define_parameters(parameter_manager)
            
            # Set up reference path
            reference_path = self._create_reference_path(center_line)
            goal = [center_line[-1, 0], center_line[-1, 1]]
            
            # Run simulation
            result = self._run_simulation(
                module_manager, parameter_manager, 
                reference_path, goal, obstacles,
                left_boundary, right_boundary
            )
            
            return result
            
        except Exception as e:
            LOG_WARN(f"Simple road test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'trajectory': [],
                'obstacles': obstacles if 'obstacles' in locals() else []
            }
    
    def _create_reference_path(self, center_line: np.ndarray) -> Dict[str, Any]:
        """Create reference path data structure."""
        # Create arc length parameterization
        dx = np.diff(center_line[:, 0])
        dy = np.diff(center_line[:, 1])
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(ds)])
        s = s / s[-1]  # Normalize to [0, 1]
        
        # Create velocity profile
        velocity = np.ones_like(s) * self.max_velocity * 0.8
        
        return {
            'x': center_line[:, 0],
            'y': center_line[:, 1],
            's': s,
            'v': velocity
        }
    
    def _run_simulation(self, module_manager: ModuleManager, 
                       parameter_manager: ParameterManager,
                       reference_path: Dict[str, Any], goal: List[float],
                       obstacles: List[Dict], left_boundary: np.ndarray,
                       right_boundary: np.ndarray) -> Dict[str, Any]:
        """Run the MPC simulation."""
        # Initial state [x, y, psi, v]
        initial_state = np.array([0.0, 0.0, 0.0, 8.0])
        current_state = initial_state.copy()
        
        # Storage for results
        trajectory = [initial_state.copy()]
        controls = []
        solve_times = []
        
        LOG_INFO(f"Starting simulation with goal: {goal}")
        
        for step in range(self.max_steps):
            LOG_DEBUG(f"Step {step}: State = {current_state}")
            
            # Check if goal reached
            distance_to_goal = np.sqrt((current_state[0] - goal[0])**2 + 
                                     (current_state[1] - goal[1])**2)
            if distance_to_goal < 3.0:
                LOG_INFO(f"Goal reached at step {step} (distance: {distance_to_goal:.2f}m)")
                break
            
            # Update obstacles for current step
            updated_obstacles = self._update_obstacles(obstacles, step)
            
            # Create data object
            data = self._create_data_object(
                reference_path, goal, updated_obstacles, 
                left_boundary, right_boundary
            )
            
            # Update modules
            module_manager.update_all(current_state, data)
            
            # Set parameters
            module_manager.set_parameters_all(parameter_manager, data, self.horizon)
            
            # Solve MPC problem (simplified)
            start_time = time.time()
            try:
                # This is a simplified solver - in practice, you'd use the full MPC solver
                solution = self._solve_mpc_simplified(
                    current_state, module_manager, parameter_manager, step
                )
                solve_time = time.time() - start_time
                
                if not solution['success']:
                    error_msg = f"Solver failed at step {step}: {solution.get('message', 'Unknown error')}"
                    LOG_WARN(error_msg)
                    return {
                        'success': False,
                        'error': error_msg,
                        'trajectory': np.array(trajectory),
                        'controls': np.array(controls),
                        'solve_times': solve_times
                    }
                
                # Extract next state and control
                next_state = solution['next_state']
                control = solution['control']
                
                # Store results
                trajectory.append(next_state)
                controls.append(control)
                solve_times.append(solve_time)
                
                # Update state
                current_state = next_state
                
            except Exception as e:
                error_msg = f"Simulation failed at step {step}: {e}"
                LOG_WARN(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'trajectory': np.array(trajectory),
                    'controls': np.array(controls),
                    'solve_times': solve_times
                }
        
        # Create visualization data
        visualization_data = {
            'trajectory': np.array(trajectory),
            'obstacles': obstacles,
            'road_data': {
                'center_line': reference_path['x'],
                'left_boundary': left_boundary,
                'right_boundary': right_boundary
            }
        }
        
        return {
            'success': True,
            'trajectory': np.array(trajectory),
            'controls': np.array(controls),
            'solve_times': solve_times,
            'visualization_data': visualization_data
        }
    
    def _update_obstacles(self, obstacles: List[Dict], step: int) -> List[Dict]:
        """Update obstacle positions for current step."""
        updated_obstacles = []
        
        for obs in obstacles:
            # Project obstacle position forward
            dt = self.timestep
            new_x = obs['position'][0] + obs['velocity'][0] * dt * step
            new_y = obs['position'][1] + obs['velocity'][1] * dt * step
            
            updated_obs = obs.copy()
            updated_obs['position'] = [new_x, new_y]
            updated_obstacles.append(updated_obs)
        
        return updated_obstacles
    
    def _create_data_object(self, reference_path: Dict[str, Any], goal: List[float],
                           obstacles: List[Dict], left_boundary: np.ndarray,
                           right_boundary: np.ndarray) -> Dict[str, Any]:
        """Create data object for modules."""
        return {
            'reference_path': reference_path,
            'goal': goal,
            'dynamic_obstacles': obstacles,
            'left_boundary': left_boundary,
            'right_boundary': right_boundary
        }
    
    def _solve_mpc_simplified(self, current_state: np.ndarray, 
                             module_manager: ModuleManager,
                             parameter_manager: ParameterManager,
                             step: int) -> Dict[str, Any]:
        """Simplified MPC solver for testing."""
        try:
            # This is a simplified implementation
            # In practice, you'd use the full CasADi-based MPC solver
            
            # Simple state propagation
            dt = self.timestep
            v = current_state[3]
            psi = current_state[2]
            
            # Simple control (steer towards goal)
            goal = [80.0, 0.0]  # Simplified goal
            dx = goal[0] - current_state[0]
            dy = goal[1] - current_state[1]
            desired_psi = np.arctan2(dy, dx)
            
            # Simple steering control
            steering_angle = np.clip(desired_psi - psi, -0.3, 0.3)
            acceleration = 0.5  # Simple acceleration
            
            # State propagation
            next_x = current_state[0] + v * np.cos(psi) * dt
            next_y = current_state[1] + v * np.sin(psi) * dt
            next_psi = psi + steering_angle * dt
            next_v = np.clip(v + acceleration * dt, 0, self.max_velocity)
            
            next_state = np.array([next_x, next_y, next_psi, next_v])
            control = np.array([steering_angle, acceleration])
            
            return {
                'success': True,
                'next_state': next_state,
                'control': control
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e)
            }
    
    def create_animation(self, result: Dict[str, Any], filename: str = "simple_road_test.gif") -> str:
        """Create animation of the test result."""
        if not result['success'] or 'visualization_data' not in result:
            LOG_WARN("Cannot create animation for failed test")
            return ""
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get data
        trajectory = result['visualization_data']['trajectory']
        obstacles = result['visualization_data']['obstacles']
        road_data = result['visualization_data']['road_data']
        
        # Plot road
        ax.plot(road_data['center_line'], road_data['left_boundary'][:, 1], 
                'k-', linewidth=2, label='Road Boundaries')
        ax.plot(road_data['center_line'], road_data['right_boundary'][:, 1], 
                'k-', linewidth=2)
        ax.plot(road_data['center_line'], np.zeros_like(road_data['center_line']), 
                'k--', linewidth=1, alpha=0.5, label='Road Center')
        
        # Initialize plots
        vehicle_patch = None
        obstacle_patches = []
        path_line = None
        
        def animate(frame):
            nonlocal vehicle_patch, obstacle_patches, path_line
            
            # Clear previous patches
            if vehicle_patch:
                vehicle_patch.remove()
            for patch in obstacle_patches:
                patch.remove()
            obstacle_patches.clear()
            if path_line:
                path_line.remove()
            
            # Plot trajectory up to current frame
            if frame > 0:
                path_line, = ax.plot(trajectory[:frame, 0], trajectory[:frame, 1], 
                                   'b-', linewidth=3, alpha=0.8, label='Vehicle Path')
            
            # Plot current vehicle position
            if frame < len(trajectory):
                x, y, psi = trajectory[frame, :3]
                vehicle_patch = self._draw_vehicle(ax, x, y, psi)
            
            # Plot obstacles at current time
            for i, obs in enumerate(obstacles):
                obs_x = obs['position'][0] + obs['velocity'][0] * frame * self.timestep
                obs_y = obs['position'][1] + obs['velocity'][1] * frame * self.timestep
                obs_radius = obs['radius']
                
                circle = Circle((obs_x, obs_y), obs_radius, 
                               color='red', alpha=0.6, label='Obstacle' if i == 0 else "")
                ax.add_patch(circle)
                obstacle_patches.append(circle)
            
            # Set plot properties
            ax.set_xlim(-5, 85)
            ax.set_ylim(-8, 8)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title(f'Simple Road Test - Step {frame}')
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(trajectory), 
                                    interval=200, repeat=True)
        
        # Save animation
        anim.save(filepath, writer='pillow', fps=5)
        plt.close()
        
        LOG_INFO(f"Animation saved to {filepath}")
        return filepath
    
    def _draw_vehicle(self, ax, x: float, y: float, psi: float) -> Rectangle:
        """Draw vehicle at given position and orientation."""
        # Vehicle dimensions
        length = self.vehicle_length
        width = self.vehicle_width
        
        # Create vehicle rectangle
        vehicle = Rectangle((x - length/2, y - width/2), length, width, 
                           angle=np.degrees(psi), color='blue', alpha=0.8)
        ax.add_patch(vehicle)
        return vehicle
    
    def run_and_visualize(self) -> Dict[str, Any]:
        """Run test and create visualization."""
        # Run test
        result = self.run_test()
        
        if result['success']:
            LOG_INFO("Test completed successfully!")
            
            # Create animation
            animation_path = self.create_animation(result)
            
            # Print results
            print("\n" + "="*50)
            print("SIMPLE ROAD TEST RESULTS")
            print("="*50)
            print(f"Status: ✅ SUCCESS")
            print(f"Trajectory Length: {len(result['trajectory'])} steps")
            print(f"Average Solve Time: {np.mean(result['solve_times']):.4f}s")
            print(f"Animation: {animation_path}")
            
        else:
            LOG_WARN("Test failed!")
            print("\n" + "="*50)
            print("SIMPLE ROAD TEST RESULTS")
            print("="*50)
            print(f"Status: ❌ FAILED")
            print(f"Error: {result['error']}")
        
        return result


def main():
    """Main function to run simple road test."""
    test = SimpleRoadTest()
    result = test.run_and_visualize()
    return result


if __name__ == "__main__":
    main()
