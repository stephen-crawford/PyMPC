"""
Unified Constraint Framework for MPC Testing

This module provides a standardized framework for testing different constraint types
with configurable perception areas and obstacle memory, following the C++ MPC planner formulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Polygon
from typing import List, Dict, Any, Tuple, Optional
import os
import time


class PerceptionArea:
    """Configurable perception area with obstacle memory."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize perception area.
        
        Args:
            config: Dictionary containing perception area configuration
        """
        self.config = config
        self.obstacle_memory = {}  # Track obstacles with memory
        self.memory_duration = config.get('memory_duration', 3.0)  # seconds
        
    def update_obstacle_memory(self, obstacles: List[Dict], vehicle_state: np.ndarray, 
                              timestep: float) -> List[Dict]:
        """
        Update obstacle memory and filter obstacles by perception area.
        
        Args:
            obstacles: List of current obstacles
            vehicle_state: Current vehicle state [x, y, psi, v]
            timestep: Current timestep
            
        Returns:
            List of obstacles within perception area with memory
        """
        vehicle_x, vehicle_y, vehicle_psi = vehicle_state[0], vehicle_state[1], vehicle_state[2]
        current_time = timestep
        
        # Update memory for existing obstacles
        for obs_id in list(self.obstacle_memory.keys()):
            memory_entry = self.obstacle_memory[obs_id]
            if current_time - memory_entry['last_seen'] > self.memory_duration:
                # Remove from memory if too old
                del self.obstacle_memory[obs_id]
        
        # Check current obstacles against perception area
        filtered_obstacles = []
        for obs in obstacles:
            obs_x, obs_y = obs['position']
            obs_id = obs.get('id', f"obs_{obs_x}_{obs_y}")
            
            # Check if obstacle is in perception area
            if self._is_in_perception_area(obs_x, obs_y, vehicle_x, vehicle_y, vehicle_psi):
                # Add to memory or update existing entry
                self.obstacle_memory[obs_id] = {
                    'obstacle': obs.copy(),
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'predictions': self._generate_obstacle_predictions(obs, vehicle_state)
                }
                filtered_obstacles.append(obs)
            elif obs_id in self.obstacle_memory:
                # Obstacle was in perception area before, update memory
                self.obstacle_memory[obs_id]['last_seen'] = current_time
                # Use memory predictions
                memory_obs = self.obstacle_memory[obs_id]['obstacle']
                filtered_obstacles.append(memory_obs)
        
        return filtered_obstacles
    
    def _is_in_perception_area(self, obs_x: float, obs_y: float, 
                              vehicle_x: float, vehicle_y: float, 
                              vehicle_psi: float) -> bool:
        """Check if obstacle is within perception area."""
        perception_type = self.config['type']
        
        # Calculate relative position
        rel_x = obs_x - vehicle_x
        rel_y = obs_y - vehicle_y
        distance = np.sqrt(rel_x**2 + rel_y**2)
        
        # Calculate angle relative to vehicle heading
        angle_to_obstacle = np.arctan2(rel_y, rel_x)
        angle_diff = angle_to_obstacle - vehicle_psi
        # Normalize angle to [-pi, pi]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        if perception_type == 'radius':
            return distance <= self.config['radius']
            
        elif perception_type == 'sphere':
            return distance <= self.config['radius']
            
        elif perception_type == 'forward_cone':
            forward_angle = self.config['forward_angle']
            forward_distance = self.config['forward_distance']
            return (distance <= forward_distance and 
                   abs(angle_diff) <= forward_angle/2)
                   
        elif perception_type == 'bidirectional_cone':
            forward_angle = self.config['forward_angle']
            backward_angle = self.config['backward_angle']
            forward_distance = self.config['forward_distance']
            backward_distance = self.config['backward_distance']
            
            # Forward cone
            if (distance <= forward_distance and 
                abs(angle_diff) <= forward_angle/2):
                return True
            # Backward cone
            elif (distance <= backward_distance and 
                  abs(angle_diff) >= np.pi - backward_angle/2):
                return True
            return False
                    
        elif perception_type == 'rectangle':
            forward_distance = self.config['forward_distance']
            backward_distance = self.config['backward_distance']
            width = self.config['width']
            
            # Transform to vehicle coordinate system
            cos_psi = np.cos(vehicle_psi)
            sin_psi = np.sin(vehicle_psi)
            
            # Rotate obstacle position to vehicle frame
            obs_x_vehicle = rel_x * cos_psi + rel_y * sin_psi
            obs_y_vehicle = -rel_x * sin_psi + rel_y * cos_psi
            
            # Check if obstacle is within rectangular bounds
            return (-backward_distance <= obs_x_vehicle <= forward_distance and 
                    -width/2 <= obs_y_vehicle <= width/2)
        
        return False
    
    def _generate_obstacle_predictions(self, obstacle: Dict, vehicle_state: np.ndarray) -> List[Dict]:
        """Generate obstacle predictions for different scenarios."""
        predictions = []
        obs_x, obs_y = obstacle['position']
        obs_radius = obstacle['radius']
        
        # Generate multiple scenario predictions
        scenarios = [
            {'name': 'continue', 'weight': 0.4, 'time': 2.0, 'factor': 1.0},
            {'name': 'accelerate', 'weight': 0.25, 'time': 1.5, 'factor': 1.2},
            {'name': 'turn_left', 'weight': 0.15, 'time': 1.8, 'factor': 1.0, 'turn': np.pi/6},
            {'name': 'turn_right', 'weight': 0.1, 'time': 1.8, 'factor': 1.0, 'turn': -np.pi/6},
            {'name': 'decelerate', 'weight': 0.1, 'time': 2.2, 'factor': 0.7}
        ]
        
        for scenario in scenarios:
            pred_x, pred_y = self._predict_obstacle_position(obstacle, scenario)
            predictions.append({
                'scenario': scenario['name'],
                'weight': scenario['weight'],
                'position': [pred_x, pred_y],
                'radius': obs_radius + scenario.get('radius_offset', 0.0),
                'time': scenario['time']
            })
        
        return predictions
    
    def _predict_obstacle_position(self, obstacle: Dict, scenario: Dict) -> Tuple[float, float]:
        """Predict obstacle position for a given scenario."""
        obs_x, obs_y = obstacle['position']
        velocity = obstacle['velocity']
        
        if scenario['name'] == 'continue':
            pred_x = obs_x + velocity[0] * scenario['time'] * scenario['factor']
            pred_y = obs_y + velocity[1] * scenario['time'] * scenario['factor']
            
        elif scenario['name'] == 'accelerate':
            pred_x = obs_x + velocity[0] * scenario['time'] * scenario['factor']
            pred_y = obs_y + velocity[1] * scenario['time'] * scenario['factor']
            
        elif scenario['name'] in ['turn_left', 'turn_right']:
            turn_angle = scenario['turn']
            cos_turn = np.cos(turn_angle)
            sin_turn = np.sin(turn_angle)
            vel_x = velocity[0] * cos_turn - velocity[1] * sin_turn
            vel_y = velocity[0] * sin_turn + velocity[1] * cos_turn
            pred_x = obs_x + vel_x * scenario['time']
            pred_y = obs_y + vel_y * scenario['time']
            
        elif scenario['name'] == 'decelerate':
            pred_x = obs_x + velocity[0] * scenario['time'] * scenario['factor']
            pred_y = obs_y + velocity[1] * scenario['time'] * scenario['factor']
            
        else:
            # Default to continue
            pred_x = obs_x + velocity[0] * scenario['time']
            pred_y = obs_y + velocity[1] * scenario['time']
        
        return pred_x, pred_y
    
    def get_obstacle_predictions(self, obstacle_id: str) -> List[Dict]:
        """Get predictions for a specific obstacle from memory."""
        if obstacle_id in self.obstacle_memory:
            return self.obstacle_memory[obstacle_id]['predictions']
        return []
    
    def draw_perception_area(self, ax, vehicle_pos: np.ndarray, vehicle_psi: float) -> List:
        """Draw the perception area visualization."""
        patches = []
        vehicle_x, vehicle_y = vehicle_pos[0], vehicle_pos[1]
        perception_type = self.config['type']
        
        if perception_type == 'radius':
            radius = self.config['radius']
            circle = Circle((vehicle_x, vehicle_y), radius, 
                          fill=False, edgecolor='cyan', linewidth=2, 
                          linestyle='--', alpha=0.7, label='Perception Area')
            ax.add_patch(circle)
            patches.append(circle)
            
        elif perception_type == 'forward_cone':
            forward_angle = self.config['forward_angle']
            forward_distance = self.config['forward_distance']
            
            # Calculate cone vertices
            half_angle = forward_angle / 2
            left_angle = vehicle_psi - half_angle
            right_angle = vehicle_psi + half_angle
            
            left_x = vehicle_x + forward_distance * np.cos(left_angle)
            left_y = vehicle_y + forward_distance * np.sin(left_angle)
            right_x = vehicle_x + forward_distance * np.cos(right_angle)
            right_y = vehicle_y + forward_distance * np.sin(right_angle)
            
            # Draw cone
            cone_points = [[vehicle_x, vehicle_y], [left_x, left_y], [right_x, right_y], [vehicle_x, vehicle_y]]
            cone_polygon = plt.Polygon(cone_points, fill=False, edgecolor='cyan', 
                                     linewidth=2, linestyle='--', alpha=0.7, label='Perception Area')
            ax.add_patch(cone_polygon)
            patches.append(cone_polygon)
            
        elif perception_type == 'bidirectional_cone':
            # Forward cone
            forward_angle = self.config['forward_angle']
            forward_distance = self.config['forward_distance']
            half_angle = forward_angle / 2
            left_angle = vehicle_psi - half_angle
            right_angle = vehicle_psi + half_angle
            
            left_x = vehicle_x + forward_distance * np.cos(left_angle)
            left_y = vehicle_y + forward_distance * np.sin(left_angle)
            right_x = vehicle_x + forward_distance * np.cos(right_angle)
            right_y = vehicle_y + forward_distance * np.sin(right_angle)
            
            forward_cone_points = [[vehicle_x, vehicle_y], [left_x, left_y], [right_x, right_y], [vehicle_x, vehicle_y]]
            forward_cone = plt.Polygon(forward_cone_points, fill=False, edgecolor='cyan', 
                                     linewidth=2, linestyle='--', alpha=0.7, label='Forward Perception')
            ax.add_patch(forward_cone)
            patches.append(forward_cone)
            
            # Backward cone
            backward_angle = self.config['backward_angle']
            backward_distance = self.config['backward_distance']
            half_angle = backward_angle / 2
            left_angle = vehicle_psi + np.pi - half_angle
            right_angle = vehicle_psi + np.pi + half_angle
            
            left_x = vehicle_x + backward_distance * np.cos(left_angle)
            left_y = vehicle_y + backward_distance * np.sin(left_angle)
            right_x = vehicle_x + backward_distance * np.cos(right_angle)
            right_y = vehicle_y + backward_distance * np.sin(right_angle)
            
            backward_cone_points = [[vehicle_x, vehicle_y], [left_x, left_y], [right_x, right_y], [vehicle_x, vehicle_y]]
            backward_cone = plt.Polygon(backward_cone_points, fill=False, edgecolor='magenta', 
                                      linewidth=2, linestyle='--', alpha=0.7, label='Backward Perception')
            ax.add_patch(backward_cone)
            patches.append(backward_cone)
            
        elif perception_type == 'rectangle':
            forward_distance = self.config['forward_distance']
            backward_distance = self.config['backward_distance']
            width = self.config['width']
            
            # Calculate rectangle corners in vehicle frame
            cos_psi = np.cos(vehicle_psi)
            sin_psi = np.sin(vehicle_psi)
            
            # Rectangle corners in vehicle frame
            corners_vehicle = [
                [-backward_distance, -width/2],
                [forward_distance, -width/2],
                [forward_distance, width/2],
                [-backward_distance, width/2]
            ]
            
            # Transform to world frame
            corners_world = []
            for corner in corners_vehicle:
                x_world = vehicle_x + corner[0] * cos_psi - corner[1] * sin_psi
                y_world = vehicle_y + corner[0] * sin_psi + corner[1] * cos_psi
                corners_world.append([x_world, y_world])
            
            # Draw rectangle
            rect_polygon = plt.Polygon(corners_world, fill=False, edgecolor='cyan', 
                                     linewidth=2, linestyle='--', alpha=0.7, label='Perception Area')
            ax.add_patch(rect_polygon)
            patches.append(rect_polygon)
        
        return patches


class UnifiedConstraintTest:
    """Unified framework for testing different constraint types."""
    
    def __init__(self, constraint_type: str, perception_config: Dict[str, Any]):
        """
        Initialize unified constraint test.
        
        Args:
            constraint_type: Type of constraint ('linearized', 'ellipsoid', 'gaussian', 'scenario')
            perception_config: Perception area configuration
        """
        self.constraint_type = constraint_type
        self.perception_area = PerceptionArea(perception_config)
        
        # Test parameters
        self.horizon = 15
        self.timestep = 0.1
        self.max_steps = 150
        
        # Vehicle parameters
        self.vehicle_length = 4.0
        self.vehicle_width = 1.8
        self.max_velocity = 12.0
        
        # Road parameters
        self.road_width = 8.0
        self.road_length = 120.0
        
        # Output directory
        self.output_dir = f"{constraint_type}_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_curved_road(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a curved road that requires significant turning."""
        s = np.linspace(0, 1, 80)
        x = s * self.road_length
        
        # Complex S-curve with multiple frequencies
        y = (10 * np.sin(2 * np.pi * s * 0.3) + 
             5 * np.sin(2 * np.pi * s * 0.6) + 
             2.5 * np.sin(2 * np.pi * s * 1.2))
        
        # Create road boundaries
        half_width = self.road_width / 2
        left_boundary = np.column_stack([x, y - half_width])
        right_boundary = np.column_stack([x, y + half_width])
        center_line = np.column_stack([x, y])
        
        return center_line, left_boundary, right_boundary
    
    def create_obstacles(self) -> List[Dict[str, Any]]:
        """Create obstacles with random positions and various dynamic models."""
        obstacles = []
        
        for i in range(4):
            start_x = np.random.uniform(20, 100)
            start_y = np.random.uniform(-15, 15)
            
            radius = np.random.uniform(0.8, 1.5)
            speed = np.random.uniform(3.0, 6.0)
            
            # Different movement models
            if i == 0:
                velocity = [speed, np.random.uniform(-2, 2)]
                movement_model = 'linear'
            elif i == 1:
                angular_velocity = np.random.uniform(0.1, 0.3)
                velocity = [speed, angular_velocity]
                movement_model = 'circular'
            elif i == 2:
                velocity = [speed, np.random.uniform(-1, 1)]
                movement_model = 'oscillating'
            else:
                velocity = [speed, np.random.uniform(-1.5, 1.5)]
                movement_model = 'random_walk'
            
            obstacle = {
                'id': f'obstacle_{i}',
                'position': [start_x, start_y],
                'velocity': velocity,
                'radius': radius,
                'type': 'dynamic',
                'color': 'red',
                'alpha': 0.6,
                'movement_model': movement_model,
                'initial_position': [start_x, start_y],
                'angular_velocity': angular_velocity if i == 1 else 0.0,
                'oscillation_amplitude': np.random.uniform(1.0, 3.0) if i == 2 else 0.0,
                'oscillation_frequency': np.random.uniform(0.1, 0.3) if i == 2 else 0.0,
                'random_walk_std': np.random.uniform(0.1, 0.3) if i == 3 else 0.0
            }
            obstacles.append(obstacle)
        
        return obstacles
    
    def update_obstacles(self, obstacles: List[Dict], step: int) -> List[Dict]:
        """Update obstacle positions for current step with various movement models."""
        updated_obstacles = []
        dt = self.timestep
        time = step * dt
        
        for obs in obstacles:
            movement_model = obs.get('movement_model', 'linear')
            
            if movement_model == 'linear':
                new_x = obs['position'][0] + obs['velocity'][0] * dt
                new_y = obs['position'][1] + obs['velocity'][1] * dt
                
            elif movement_model == 'circular':
                angular_vel = obs.get('angular_velocity', 0.1)
                radius = 5.0
                center_x = obs['initial_position'][0]
                center_y = obs['initial_position'][1]
                
                angle = angular_vel * time
                new_x = center_x + radius * np.cos(angle)
                new_y = center_y + radius * np.sin(angle)
                
            elif movement_model == 'oscillating':
                amplitude = obs.get('oscillation_amplitude', 2.0)
                frequency = obs.get('oscillation_frequency', 0.2)
                
                new_x = obs['position'][0] + obs['velocity'][0] * dt
                new_y = obs['initial_position'][1] + amplitude * np.sin(frequency * time)
                
            elif movement_model == 'random_walk':
                std = obs.get('random_walk_std', 0.2)
                new_x = obs['position'][0] + obs['velocity'][0] * dt
                new_y = obs['position'][1] + obs['velocity'][1] * dt + np.random.normal(0, std)
                
            else:
                new_x = obs['position'][0] + obs['velocity'][0] * dt
                new_y = obs['position'][1] + obs['velocity'][1] * dt
            
            updated_obs = obs.copy()
            updated_obs['position'] = [new_x, new_y]
            updated_obstacles.append(updated_obs)
        
        return updated_obstacles
    
    def apply_constraint_avoidance(self, current_state: np.ndarray, obstacles: List[Dict], 
                                 step: int) -> Dict[str, Any]:
        """
        Apply constraint-specific avoidance logic.
        
        This method should be overridden by specific constraint implementations.
        """
        raise NotImplementedError("Subclasses must implement apply_constraint_avoidance")
    
    def draw_constraint_visualization(self, ax, obstacle_pos: np.ndarray, 
                                    vehicle_pos: np.ndarray, frame: int) -> List:
        """
        Draw constraint-specific visualization.
        
        This method should be overridden by specific constraint implementations.
        """
        raise NotImplementedError("Subclasses must implement draw_constraint_visualization")
    
    def run_test(self) -> Dict[str, Any]:
        """Run the unified constraint test."""
        print(f"Starting {self.constraint_type.title()} Constraints Test")
        print("This test demonstrates obstacle avoidance with configurable perception areas.")
        print()
        
        # Create road and obstacles
        center_line, left_boundary, right_boundary = self.create_curved_road()
        obstacles = self.create_obstacles()
        
        # Calculate goal position
        goal = np.array([self.road_length, center_line[-1, 1]])
        
        print(f"Road length: {self.road_length}m")
        print(f"Goal position: ({goal[0]:.1f}, {goal[1]:.1f})")
        print(f"Number of obstacles: {len(obstacles)}")
        print(f"Constraint type: {self.constraint_type.title()}")
        print(f"Perception area: {self.perception_area.config['type']}")
        print()
        
        # Run simulation
        result = self._run_simulation(center_line, left_boundary, right_boundary, 
                                   obstacles, goal)
        
        # Create animation
        if result['success']:
            self._create_animation(result['trajectory'], obstacles, 
                                center_line, left_boundary, right_boundary, goal)
        
        return result
    
    def _run_simulation(self, center_line: np.ndarray, left_boundary: np.ndarray, 
                       right_boundary: np.ndarray, obstacles: List[Dict], 
                       goal: np.ndarray) -> Dict[str, Any]:
        """Run the MPC simulation."""
        print(f"Starting simulation with goal: ({goal[0]:.1f}, {goal[1]:.1f})")
        
        # Initialize vehicle state
        current_state = np.array([0.0, 0.0, 0.0, 5.0])  # [x, y, psi, v]
        trajectory = [current_state.copy()]
        
        print("Step-by-step progress:")
        
        for step in range(self.max_steps):
            # Update obstacles
            obstacles = self.update_obstacles(obstacles, step)
            
            # Update perception area and get filtered obstacles
            filtered_obstacles = self.perception_area.update_obstacle_memory(
                obstacles, current_state, step * self.timestep)
            
            # Apply constraint avoidance
            result = self.apply_constraint_avoidance(current_state, filtered_obstacles, step)
            
            if not result['success']:
                print(f"  Error: Solver failed at step {step}: {result['message']}")
                return {
                    'success': False,
                    'error': result['message'],
                    'trajectory': np.array(trajectory)
                }
            
            # Update state
            current_state = result['next_state']
            trajectory.append(current_state.copy())
            
            # Check goal reaching
            distance_to_goal = np.sqrt((current_state[0] - goal[0])**2 + 
                                     (current_state[1] - goal[1])**2)
            
            if step % 10 == 0:
                print(f"  Step {step}: Position ({current_state[0]:.1f}, {current_state[1]:.1f}), "
                      f"Distance to goal: {distance_to_goal:.1f}m")
            
            if distance_to_goal < 2.0:
                print(f"  Step {step}: Goal reached! (distance: {distance_to_goal:.2f}m)")
                break
        
        if distance_to_goal >= 2.0:
            print(f"  Warning: Goal not reached (final distance: {distance_to_goal:.2f}m)")
        
        print(f"\n🎉 {self.constraint_type.title()} constraints test completed successfully!")
        print(f"Animation saved to {self.output_dir}/{self.constraint_type}_constraints_test.gif")
        
        return {
            'success': True,
            'trajectory': np.array(trajectory),
            'obstacles': obstacles,
            'road_data': {
                'center_line': center_line,
                'left_boundary': left_boundary,
                'right_boundary': right_boundary
            },
            'goal': goal
        }
    
    def _create_animation(self, trajectory: np.ndarray, obstacles: List[Dict],
                         center_line: np.ndarray, left_boundary: np.ndarray,
                         right_boundary: np.ndarray, goal: np.ndarray):
        """Create animation of the test."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot road
        ax.plot(center_line[:, 0], center_line[:, 1], 'k--', linewidth=2, label='Road Center')
        ax.plot(left_boundary[:, 0], left_boundary[:, 1], 'k-', linewidth=2, label='Road Boundaries')
        ax.plot(right_boundary[:, 0], right_boundary[:, 1], 'k-', linewidth=2)
        
        # Plot goal
        goal_circle = Circle((goal[0], goal[1]), 2.0, 
                           facecolor='green', alpha=0.7, 
                           edgecolor='darkgreen', linewidth=2,
                           label='Goal')
        ax.add_patch(goal_circle)
        ax.text(goal[0], goal[1] + 3, 'GOAL', ha='center', va='bottom', 
               fontsize=12, fontweight='bold', color='darkgreen')
        
        # Set plot properties
        ax.set_xlim(-10, 130)
        ax.set_ylim(-30, 30)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'{self.constraint_type.title()} Constraints Test - Configurable Perception Areas')
        
        # Animation variables
        vehicle_patch = None
        obstacle_patches = []
        constraint_patches = []
        perception_patches = []
        
        def animate(frame):
            nonlocal vehicle_patch, obstacle_patches, constraint_patches, perception_patches
            
            ax.clear()
            
            # Replot road
            ax.plot(center_line[:, 0], center_line[:, 1], 'k--', linewidth=2, label='Road Center')
            ax.plot(left_boundary[:, 0], left_boundary[:, 1], 'k-', linewidth=2, label='Road Boundaries')
            ax.plot(right_boundary[:, 0], right_boundary[:, 1], 'k-', linewidth=2)
            
            # Plot goal
            goal_circle = Circle((goal[0], goal[1]), 2.0, 
                               facecolor='green', alpha=0.7, 
                               edgecolor='darkgreen', linewidth=2,
                               label='Goal')
            ax.add_patch(goal_circle)
            ax.text(goal[0], goal[1] + 3, 'GOAL', ha='center', va='bottom', 
                   fontsize=12, fontweight='bold', color='darkgreen')
            
            # Draw perception area for current vehicle position
            if frame < len(trajectory):
                current_state = trajectory[frame]
                vehicle_pos = current_state[:2]
                vehicle_psi = current_state[2]
                perception_patches = self.perception_area.draw_perception_area(ax, vehicle_pos, vehicle_psi)
            
            # Plot trajectory up to current frame
            if frame > 0:
                ax.plot(trajectory[:frame, 0], trajectory[:frame, 1], 
                       'b-', linewidth=3, alpha=0.8, label='Vehicle Path')
            
            # Plot current vehicle position
            if frame < len(trajectory):
                current_state = trajectory[frame]
                vehicle_x, vehicle_y = current_state[0], current_state[1]
                vehicle_psi = current_state[2]
                
                # Draw vehicle
                vehicle_corners = self._get_vehicle_corners(vehicle_x, vehicle_y, vehicle_psi)
                vehicle_patch = plt.Polygon(vehicle_corners, facecolor='blue', alpha=0.7, 
                                          edgecolor='darkblue', linewidth=2, label='Vehicle')
                ax.add_patch(vehicle_patch)
                
                # Draw obstacles
                obstacle_patches = []
                for obs in obstacles:
                    obs_x, obs_y = obs['position']
                    obs_radius = obs['radius']
                    
                    # Check if obstacle is in perception area
                    if self.perception_area._is_in_perception_area(obs_x, obs_y, vehicle_x, vehicle_y, vehicle_psi):
                        # Draw obstacle
                        obstacle_circle = Circle((obs_x, obs_y), obs_radius, 
                                               facecolor='red', alpha=0.6, 
                                               edgecolor='darkred', linewidth=2)
                        ax.add_patch(obstacle_circle)
                        obstacle_patches.append(obstacle_circle)
                        
                        # Draw constraint visualization
                        constraint_patches = self.draw_constraint_visualization(
                            ax, np.array([obs_x, obs_y]), np.array([vehicle_x, vehicle_y]), frame)
            
            # Set plot limits
            ax.set_xlim(-10, 130)
            ax.set_ylim(-30, 30)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_title(f'{self.constraint_type.title()} Constraints Test - Configurable Perception Areas')
            
            return vehicle_patch, obstacle_patches, constraint_patches, perception_patches
        
        # Create animation
        anim = plt.FuncAnimation(fig, animate, frames=len(trajectory), 
                               interval=100, blit=False, repeat=True)
        
        # Save animation
        output_path = f"{self.output_dir}/{self.constraint_type}_constraints_test.gif"
        anim.save(output_path, writer='pillow', fps=10)
        plt.close()
    
    def _get_vehicle_corners(self, x: float, y: float, psi: float) -> List[List[float]]:
        """Get vehicle corner coordinates for visualization."""
        half_length = self.vehicle_length / 2
        half_width = self.vehicle_width / 2
        
        # Vehicle corners in local frame
        corners_local = [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width]
        ]
        
        # Transform to world frame
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        
        corners_world = []
        for corner in corners_local:
            x_world = x + corner[0] * cos_psi - corner[1] * sin_psi
            y_world = y + corner[0] * sin_psi + corner[1] * cos_psi
            corners_world.append([x_world, y_world])
        
        return corners_world
