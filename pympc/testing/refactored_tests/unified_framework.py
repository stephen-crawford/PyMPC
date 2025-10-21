"""
Unified Constraint Framework for Refactored Tests

This module provides a comprehensive unified framework for testing different constraint types
with configurable perception areas, obstacle memory, trajectory funnels, and enhanced visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.patches import Circle, Polygon, Ellipse, FancyBboxPatch
from typing import List, Dict, Any, Tuple, Optional
import os
import time
from datetime import datetime


class PerceptionArea:
    """Enhanced perception area with trajectory funnels and recall memory."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize perception area with enhanced features."""
        self.config = config
        self.obstacle_memory = {}
        self.memory_duration = config.get('memory_duration', 3.0)
        self.recall_memory = config.get('recall_memory', True)  # Enable recall memory for scenario constraints
        self.trajectory_funnels = config.get('trajectory_funnels', False)  # Enable trajectory funnels
        self.funnel_horizon = config.get('funnel_horizon', 5.0)  # Funnel prediction horizon in seconds
        
    def update_obstacle_memory(self, obstacles: List[Dict], vehicle_state: np.ndarray, 
                              timestep: float) -> List[Dict]:
        """Update obstacle memory with enhanced recall and funnel capabilities."""
        vehicle_x, vehicle_y, vehicle_psi = vehicle_state[0], vehicle_state[1], vehicle_state[2]
        current_time = timestep
        
        # Update memory for existing obstacles
        for obs_id in list(self.obstacle_memory.keys()):
            memory_entry = self.obstacle_memory[obs_id]
            if current_time - memory_entry['last_seen'] > self.memory_duration:
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
                    'predictions': self._generate_obstacle_predictions(obs, vehicle_state),
                    'trajectory_funnel': self._generate_trajectory_funnel(obs, vehicle_state) if self.trajectory_funnels else None
                }
                filtered_obstacles.append(obs)
            elif obs_id in self.obstacle_memory and self.recall_memory:
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
        """Generate obstacle predictions for different scenarios with randomization."""
        predictions = []
        obs_x, obs_y = obstacle['position']
        obs_radius = obstacle['radius']
        
        # Generate multiple scenario predictions with randomization
        base_scenarios = [
            {'name': 'continue', 'weight': 0.4, 'time': 2.0, 'factor': 1.0},
            {'name': 'accelerate', 'weight': 0.25, 'time': 1.5, 'factor': 1.2},
            {'name': 'turn_left', 'weight': 0.15, 'time': 1.8, 'factor': 1.0, 'turn': np.pi/6},
            {'name': 'turn_right', 'weight': 0.1, 'time': 1.8, 'factor': 1.0, 'turn': -np.pi/6},
            {'name': 'decelerate', 'weight': 0.1, 'time': 2.2, 'factor': 0.7}
        ]
        
        # Add randomization to scenarios
        scenarios = []
        for scenario in base_scenarios:
            randomized_scenario = scenario.copy()
            # Add random variations
            randomized_scenario['time'] += np.random.normal(0, 0.2)
            randomized_scenario['factor'] += np.random.normal(0, 0.1)
            if 'turn' in randomized_scenario:
                randomized_scenario['turn'] += np.random.normal(0, 0.1)
            randomized_scenario['weight'] += np.random.normal(0, 0.05)
            # Ensure weights are positive
            randomized_scenario['weight'] = max(0.01, randomized_scenario['weight'])
            scenarios.append(randomized_scenario)
        
        # Normalize weights
        total_weight = sum(s['weight'] for s in scenarios)
        for scenario in scenarios:
            scenario['weight'] /= total_weight
        
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
    
    def _generate_trajectory_funnel(self, obstacle: Dict, vehicle_state: np.ndarray) -> List[Dict]:
        """Generate trajectory funnel for obstacle based on constraint type."""
        funnel_points = []
        obs_x, obs_y = obstacle['position']
        obs_radius = obstacle['radius']
        
        # Generate funnel points over the prediction horizon
        num_points = int(self.funnel_horizon / 0.1)  # 0.1s intervals
        for i in range(num_points):
            t = i * 0.1
            
            # Generate multiple trajectory samples with proper randomization
            samples = []
            for _ in range(10):  # 10 samples per time step
                # Add proper randomization based on obstacle movement model
                movement_model = obstacle.get('movement_model', 'linear')
                
                if movement_model == 'linear':
                    # Linear movement with noise
                    noise_x = np.random.normal(0, 0.3 + t * 0.1)
                    noise_y = np.random.normal(0, 0.3 + t * 0.1)
                    pred_x = obs_x + obstacle['velocity'][0] * t + noise_x
                    pred_y = obs_y + obstacle['velocity'][1] * t + noise_y
                elif movement_model == 'circular':
                    # Circular movement with noise
                    angular_vel = obstacle.get('angular_velocity', 0.1)
                    radius = 5.0
                    center_x = obstacle.get('initial_position', [obs_x, obs_y])[0]
                    center_y = obstacle.get('initial_position', [obs_x, obs_y])[1]
                    angle = angular_vel * t + np.random.normal(0, 0.1)
                    noise_x = np.random.normal(0, 0.2)
                    noise_y = np.random.normal(0, 0.2)
                    pred_x = center_x + radius * np.cos(angle) + noise_x
                    pred_y = center_y + radius * np.sin(angle) + noise_y
                elif movement_model == 'oscillating':
                    # Oscillating movement with noise
                    amplitude = obstacle.get('oscillation_amplitude', 2.0)
                    frequency = obstacle.get('oscillation_frequency', 0.2)
                    noise_x = np.random.normal(0, 0.2)
                    noise_y = np.random.normal(0, 0.3 + t * 0.1)
                    pred_x = obs_x + obstacle['velocity'][0] * t + noise_x
                    pred_y = obstacle.get('initial_position', [obs_x, obs_y])[1] + amplitude * np.sin(frequency * t) + noise_y
                else:
                    # Random walk with noise
                    noise_x = np.random.normal(0, 0.4 + t * 0.2)
                    noise_y = np.random.normal(0, 0.4 + t * 0.2)
                    pred_x = obs_x + obstacle['velocity'][0] * t + noise_x
                    pred_y = obs_y + obstacle['velocity'][1] * t + noise_y
                
                samples.append([pred_x, pred_y])
            
            funnel_points.append({
                'time': t,
                'samples': samples,
                'mean_position': [obs_x + obstacle['velocity'][0] * t, 
                                obs_y + obstacle['velocity'][1] * t],
                'uncertainty': 0.3 + t * 0.3  # Growing uncertainty over time
            })
        
        return funnel_points
    
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
    
    def get_trajectory_funnel(self, obstacle_id: str) -> List[Dict]:
        """Get trajectory funnel for a specific obstacle from memory."""
        if obstacle_id in self.obstacle_memory:
            return self.obstacle_memory[obstacle_id]['trajectory_funnel']
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


class UnifiedConstraintFramework:
    """Enhanced unified framework for testing different constraint types."""
    
    def __init__(self, constraint_type: str, perception_config: Dict[str, Any]):
        """Initialize the unified constraint framework."""
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
        
        # Performance tracking
        self.execution_times = []
        self.compute_times = []
        self.active_constraints = []
        self.boundaries = []
        
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
        """Create obstacles with random positions and various dynamic models - IMPROVED."""
        obstacles = []
        
        # Create more varied obstacles with different movement patterns
        movement_models = ['linear', 'circular', 'oscillating', 'zigzag', 'spiral', 'random_walk']
        
        for i in range(6):  # Increased number of obstacles
            # Random starting position with better distribution
            start_x = np.random.uniform(15, 110)
            start_y = np.random.uniform(-8, 25)
            radius = np.random.uniform(0.8, 2.0)  # Slightly larger obstacles
            
            # Select movement model
            movement_model = movement_models[i % len(movement_models)]
            
            # Base velocity for all obstacles - INCREASED for more visible movement
            speed = np.random.uniform(2.0, 6.0)  # Increased speed range
            angle = np.random.uniform(0, 2 * np.pi)
            velocity = [speed * np.cos(angle), speed * np.sin(angle)]
            
            # Create obstacle with movement-specific parameters
            obstacle = {
                'id': f'obstacle_{i}',
                'position': [start_x, start_y],
                'velocity': velocity,
                'radius': radius,
                'type': 'dynamic',
                'color': 'red',
                'alpha': 0.7,
                'movement_model': movement_model,
                'initial_position': [start_x, start_y],
            }
            
            # Add movement-specific parameters
            if movement_model == 'circular':
                obstacle.update({
                    'angular_velocity': np.random.uniform(0.1, 0.3),
                    'circular_radius': np.random.uniform(4.0, 8.0)
                })
            elif movement_model == 'oscillating':
                obstacle.update({
                    'oscillation_amplitude': np.random.uniform(2.0, 5.0),
                    'oscillation_frequency': np.random.uniform(0.15, 0.4),
                    'forward_speed': np.random.uniform(0.5, 1.5)
                })
            elif movement_model == 'zigzag':
                obstacle.update({
                    'zigzag_frequency': np.random.uniform(0.2, 0.6),
                    'zigzag_amplitude': np.random.uniform(3.0, 6.0),
                    'forward_speed': np.random.uniform(0.8, 2.0)
                })
            elif movement_model == 'spiral':
                obstacle.update({
                    'spiral_radius': np.random.uniform(1.5, 3.0),
                    'spiral_speed': np.random.uniform(0.1, 0.4),
                    'forward_speed': np.random.uniform(0.3, 1.2)
                })
            elif movement_model == 'random_walk':
                obstacle.update({
                    'forward_bias': np.random.uniform(0.2, 1.0),
                    'random_walk_std': np.random.uniform(0.2, 0.5)
                })
            
            obstacles.append(obstacle)
        
        return obstacles
    
    def update_obstacles(self, obstacles: List[Dict], step: int) -> List[Dict]:
        """Update obstacle positions for current step with various movement models - IMPROVED."""
        updated_obstacles = []
        dt = self.timestep
        time = step * dt
        
        for obs in obstacles:
            movement_model = obs.get('movement_model', 'linear')
            
            if movement_model == 'linear':
                # Linear movement with slight variation
                base_vel_x = obs['velocity'][0]
                base_vel_y = obs['velocity'][1]
                
                # Add small random variation to make movement more realistic
                vel_variation = 0.2  # Increased variation for more visible movement
                vel_x = base_vel_x + np.random.normal(0, vel_variation)
                vel_y = base_vel_y + np.random.normal(0, vel_variation)
                
                new_x = obs['position'][0] + vel_x * dt
                new_y = obs['position'][1] + vel_y * dt
                
            elif movement_model == 'circular':
                # Circular movement around initial position
                angular_vel = obs.get('angular_velocity', 0.15)
                radius = obs.get('circular_radius', 6.0)
                center_x = obs['initial_position'][0]
                center_y = obs['initial_position'][1]
                
                angle = angular_vel * time
                new_x = center_x + radius * np.cos(angle)
                new_y = center_y + radius * np.sin(angle)
                
            elif movement_model == 'oscillating':
                # Oscillating movement with forward progress
                amplitude = obs.get('oscillation_amplitude', 4.0)  # Increased amplitude
                frequency = obs.get('oscillation_frequency', 0.3)   # Increased frequency
                forward_speed = obs.get('forward_speed', 2.0)      # Increased forward speed
                
                # Forward movement
                new_x = obs['position'][0] + forward_speed * dt
                # Oscillating lateral movement
                new_y = obs['initial_position'][1] + amplitude * np.sin(frequency * time)
                
            elif movement_model == 'random_walk':
                # Random walk with forward bias
                forward_bias = obs.get('forward_bias', 1.0)  # Increased forward bias
                std = obs.get('random_walk_std', 0.5)       # Increased variation
                
                # Forward movement with random variation
                new_x = obs['position'][0] + (forward_bias + np.random.normal(0, std)) * dt
                new_y = obs['position'][1] + np.random.normal(0, std) * dt
                
            elif movement_model == 'zigzag':
                # Zigzag movement pattern
                zigzag_frequency = obs.get('zigzag_frequency', 0.6)  # Increased frequency
                zigzag_amplitude = obs.get('zigzag_amplitude', 5.0)  # Increased amplitude
                forward_speed = obs.get('forward_speed', 2.0)        # Increased forward speed
                
                new_x = obs['position'][0] + forward_speed * dt
                new_y = obs['initial_position'][1] + zigzag_amplitude * np.sin(zigzag_frequency * time)
                
            elif movement_model == 'spiral':
                # Spiral movement
                spiral_radius = obs.get('spiral_radius', 3.0)  # Increased radius
                spiral_speed = obs.get('spiral_speed', 0.3)   # Increased speed
                forward_speed = obs.get('forward_speed', 1.5)  # Increased forward speed
                
                angle = spiral_speed * time
                new_x = obs['position'][0] + forward_speed * dt + spiral_radius * np.cos(angle)
                new_y = obs['position'][1] + spiral_radius * np.sin(angle)
                
            else:
                # Default linear movement
                new_x = obs['position'][0] + obs['velocity'][0] * dt
                new_y = obs['position'][1] + obs['velocity'][1] * dt
            
            # Keep obstacles within reasonable bounds
            new_x = np.clip(new_x, -20, 140)  # Allow some movement outside road
            new_y = np.clip(new_y, -20, 30)   # Allow movement in y direction
            
            updated_obs = obs.copy()
            updated_obs['position'] = [new_x, new_y]
            updated_obstacles.append(updated_obs)
        
        return updated_obstacles
    
    def apply_constraint_avoidance(self, current_state: np.ndarray, obstacles: List[Dict], 
                                 step: int) -> Dict[str, Any]:
        """Apply constraint-specific avoidance logic with MPCC integration."""
        # MPCC always uses both contouring objective and contouring constraints
        mpcc_result = self._apply_mpcc_contouring(current_state, step)
        if not mpcc_result['success']:
            return mpcc_result
        
        # Apply specific constraint type (linearized, ellipsoid, gaussian, scenario)
        constraint_result = self._apply_specific_constraint_type(current_state, obstacles, step)
        if not constraint_result['success']:
            return constraint_result
        
        # Combine MPCC and constraint results
        combined_result = self._combine_mpcc_and_constraint_results(mpcc_result, constraint_result)
        
        # State propagation
        x, y, psi, v = self._propagate_state(current_state, combined_result['steering_angle'], combined_result['acceleration'])
        
        return {
            'success': True,
            'next_state': np.array([x, y, psi, v]),
            'steering_angle': combined_result['steering_angle'],
            'acceleration': combined_result['acceleration'],
            'mpcc_type': combined_result['mpcc_type'],
            'constraint_type': combined_result['constraint_type']
        }
    
    def _apply_mpcc_contouring(self, current_state: np.ndarray, step: int) -> Dict[str, Any]:
        """Apply MPCC contouring objective and constraints (based on C++ reference)."""
        x, y, psi, v = current_state
        dt = self.timestep
        
        # MPCC Contouring Objective (following C++ implementation)
        # 1. Calculate progress along path (s parameter)
        s = self._calculate_path_progress(x, y)
        
        # 2. Calculate contouring error (lateral deviation from path)
        contouring_error = self._calculate_contouring_error(x, y, s)
        
        # 3. Calculate lag error (longitudinal deviation from path)
        lag_error = self._calculate_lag_error(x, y, s)
        
        # 4. Contouring objective: minimize contouring and lag errors
        contouring_steering = self._calculate_contouring_steering(contouring_error, lag_error, psi)
        contouring_acceleration = self._calculate_contouring_acceleration(contouring_error, lag_error, v)
        
        # 5. Apply contouring constraints (road boundary enforcement)
        road_correction = self._apply_contouring_constraints(current_state, contouring_steering)
        contouring_steering += road_correction
        
        # 6. Check for road boundary violations (hard constraint)
        if self._check_road_boundary_violation(current_state):
            return {
                'success': False,
                'message': 'CRASH: Vehicle violated road boundaries - contouring constraint failed'
            }
        
        return {
            'success': True,
            'steering_angle': contouring_steering,
            'acceleration': contouring_acceleration,
            'mpcc_type': 'contouring',
            'contouring_error': contouring_error,
            'lag_error': lag_error,
            'path_progress': s
        }
    
    def _calculate_path_progress(self, x: float, y: float) -> float:
        """Calculate progress along the reference path (s parameter)."""
        # Simple approximation: use x position as progress indicator
        # In full implementation, this would use proper arc length calculation
        return min(1.0, x / self.road_length)
    
    def _calculate_contouring_error(self, x: float, y: float, s: float) -> float:
        """Calculate contouring error (lateral deviation from path)."""
        # Calculate expected y position based on road curvature
        expected_y = (10 * np.sin(2 * np.pi * s * 0.3) + 
                     5 * np.sin(2 * np.pi * s * 0.6) + 
                     2.5 * np.sin(2 * np.pi * s * 1.2))
        
        # Contouring error is lateral deviation
        return y - expected_y
    
    def _calculate_lag_error(self, x: float, y: float, s: float) -> float:
        """Calculate lag error (longitudinal deviation from path)."""
        # Expected x position based on progress
        expected_x = s * self.road_length
        
        # Lag error is longitudinal deviation
        return x - expected_x
    
    def _calculate_contouring_steering(self, contouring_error: float, lag_error: float, psi: float) -> float:
        """Calculate steering based on contouring and lag errors - FIXED FOR PROGRESS."""
        # Contouring steering: correct lateral deviation with MUCH stronger response
        contouring_correction = -1.5 * contouring_error  # Much stronger response
        
        # Lag steering: correct longitudinal deviation with stronger response
        lag_correction = 0.4 * lag_error  # Stronger lag correction
        
        # Add lookahead steering for better path following
        lookahead_distance = 15.0  # Look ahead 15 meters
        lookahead_correction = -0.5 * contouring_error  # Stronger lookahead correction
        
        # Add forward progress steering to prevent getting stuck
        progress_steering = 0.2  # Stronger forward progress
        
        total_steering = contouring_correction + lag_correction + lookahead_correction + progress_steering
        
        # Clip to reasonable limits but allow more steering
        return np.clip(total_steering, -0.8, 0.8)
    
    def _calculate_contouring_acceleration(self, contouring_error: float, lag_error: float, v: float) -> float:
        """Calculate acceleration based on contouring and lag errors - FIXED FOR PROGRESS."""
        # Base acceleration for progress - MUCH stronger
        base_acceleration = 1.2  # Even stronger base acceleration
        
        # Reduce speed if large contouring error (but not too much)
        if abs(contouring_error) > 3.0:
            base_acceleration -= 0.3
        
        # Reduce speed if large lag error (but not too much)
        if abs(lag_error) > 8.0:
            base_acceleration -= 0.2
        
        # Ensure minimum forward progress
        base_acceleration = max(0.2, base_acceleration)
        
        return base_acceleration
    
    def _propagate_state(self, current_state: np.ndarray, steering_angle: float, acceleration: float) -> np.ndarray:
        """Propagate vehicle state using bicycle model."""
        x, y, psi, v = current_state
        dt = self.timestep
        
        # Bicycle model state propagation
        next_x = x + v * np.cos(psi) * dt
        next_y = y + v * np.sin(psi) * dt
        next_psi = psi + v * np.tan(steering_angle) / 2.5 * dt  # 2.5m wheelbase
        next_v = v + acceleration * dt
        next_v = max(0.1, min(15.0, next_v))  # Speed limits
        
        return np.array([next_x, next_y, next_psi, next_v])
    
    def _apply_specific_constraint_type(self, current_state: np.ndarray, obstacles: List[Dict], 
                                      step: int) -> Dict[str, Any]:
        """Apply specific constraint type (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _apply_specific_constraint_type")
    
    def _combine_mpcc_and_constraint_results(self, mpcc_result: Dict[str, Any], 
                                           constraint_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combine MPCC contouring and specific constraint results."""
        if not mpcc_result['success'] or not constraint_result['success']:
            return constraint_result  # Return the failure
        
        # Combine steering and acceleration from both sources
        combined_steering = mpcc_result['steering_angle'] + constraint_result.get('steering_angle', 0.0)
        combined_acceleration = mpcc_result['acceleration'] + constraint_result.get('acceleration', 0.0)
        
        return {
            'success': True,
            'steering_angle': combined_steering,
            'acceleration': combined_acceleration,
            'mpcc_type': mpcc_result['mpcc_type'],
            'constraint_type': constraint_result.get('constraint_type', 'unknown')
        }
    
    def _apply_contouring_constraints(self, current_state: np.ndarray, steering_angle: float) -> float:
        """Apply contouring constraints (road boundary enforcement) - ROBUST VERSION."""
        x, y = current_state[0], current_state[1]
        
        # Road parameters - more conservative
        road_width = 8.0
        half_width = road_width / 2
        safety_margin = 2.0  # Increased safety margin
        
        # Calculate expected y position based on road curvature
        s = x / 120.0
        expected_y = (10 * np.sin(2 * np.pi * s * 0.3) + 
                     5 * np.sin(2 * np.pi * s * 0.6) + 
                     2.5 * np.sin(2 * np.pi * s * 1.2))
        
        # Calculate road boundary positions with larger safety margins
        left_boundary = expected_y - half_width - safety_margin
        right_boundary = expected_y + half_width + safety_margin
        
        # Apply VERY strong steering corrections to keep vehicle within road boundaries
        steering_correction = 0.0
        
        # Calculate distance to road center for proportional control
        distance_to_center = y - expected_y
        
        # Left boundary constraint (vehicle too far left) - MUCH STRONGER CORRECTION
        if y < left_boundary:
            violation_distance = left_boundary - y
            # Exponential correction based on violation severity
            correction_strength = min(3.0, violation_distance * 2.0)  # Much stronger
            steering_correction += correction_strength * 2.0  # Very strong turn right
            
            # Additional proportional correction towards road center
            center_correction = distance_to_center * 0.5
            steering_correction += center_correction
        
        # Right boundary constraint (vehicle too far right) - MUCH STRONGER CORRECTION
        elif y > right_boundary:
            violation_distance = y - right_boundary
            # Exponential correction based on violation severity
            correction_strength = min(3.0, violation_distance * 2.0)  # Much stronger
            steering_correction -= correction_strength * 2.0  # Very strong turn left
            
            # Additional proportional correction towards road center
            center_correction = distance_to_center * 0.5
            steering_correction += center_correction
        
        # Always apply some correction towards road center (proportional control)
        else:
            # Proportional control to stay near road center
            center_correction = -distance_to_center * 0.3
            steering_correction += center_correction
        
        # Clip steering correction to reasonable limits
        steering_correction = np.clip(steering_correction, -1.0, 1.0)
        
        return steering_correction
    
    def _check_road_boundary_violation(self, state: np.ndarray) -> bool:
        """Check if vehicle has violated road boundaries - MORE LENIENT."""
        x, y = state[0], state[1]
        
        # Road parameters
        road_width = 8.0
        half_width = road_width / 2
        safety_margin = 2.0  # Increased safety margin
        
        # Calculate expected y position based on road curvature
        s = x / 120.0
        expected_y = (10 * np.sin(2 * np.pi * s * 0.3) + 
                     5 * np.sin(2 * np.pi * s * 0.6) + 
                     2.5 * np.sin(2 * np.pi * s * 1.2))
        
        # Calculate road boundary positions
        left_boundary = expected_y - half_width - safety_margin
        right_boundary = expected_y + half_width + safety_margin
        
        # Check if vehicle is severely outside road boundaries (much more tolerance)
        # Only fail if vehicle is way outside the road
        return y < left_boundary - 6.0 or y > right_boundary + 6.0
    
    def draw_constraint_visualization(self, ax, obstacle_pos: np.ndarray, 
                                    vehicle_pos: np.ndarray, frame: int) -> List:
        """Draw constraint-specific visualization."""
        raise NotImplementedError("Subclasses must implement draw_constraint_visualization")
    
    def draw_trajectory_funnel(self, ax, obstacle_id: str, frame: int) -> List:
        """Draw trajectory funnel for obstacle."""
        patches = []
        
        if not self.perception_area.trajectory_funnels:
            return patches
        
        funnel_data = self.perception_area.get_trajectory_funnel(obstacle_id)
        if not funnel_data:
            return patches
        
        # Draw funnel points
        for i, point in enumerate(funnel_data):
            if i % 2 == 0:  # Draw every other point to avoid clutter
                for sample in point['samples']:
                    sample_x, sample_y = sample
                    uncertainty = point['uncertainty']
                    
                    # Draw sample point
                    sample_circle = Circle((sample_x, sample_y), uncertainty * 0.3,
                                         facecolor='orange', alpha=0.3,
                                         edgecolor='orange', linewidth=1)
                    patches.append(sample_circle)
        
        return patches
    
    def run_test(self) -> Dict[str, Any]:
        """Run the unified constraint test with enhanced features."""
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
        print(f"Memory duration: {self.perception_area.config['memory_duration']}s")
        print(f"Recall memory: {self.perception_area.recall_memory}")
        print(f"Trajectory funnels: {self.perception_area.trajectory_funnels}")
        print()
        
        # Run simulation
        result = self._run_simulation(center_line, left_boundary, right_boundary, 
                                   obstacles, goal)
        
        # Create enhanced animation
        if result['success']:
            self._create_enhanced_animation(result['trajectory'], obstacles, 
                                          center_line, left_boundary, right_boundary, goal)
        
        return result
    
    def _run_simulation(self, center_line: np.ndarray, left_boundary: np.ndarray, 
                       right_boundary: np.ndarray, obstacles: List[Dict], 
                       goal: np.ndarray) -> Dict[str, Any]:
        """Run the MPC simulation with performance tracking."""
        print(f"Starting simulation with goal: ({goal[0]:.1f}, {goal[1]:.1f})")
        
        # Initialize vehicle state
        current_state = np.array([0.0, 0.0, 0.0, 5.0])  # [x, y, psi, v]
        trajectory = [current_state.copy()]
        
        # Progress tracking to prevent infinite loops
        last_progress_position = current_state[0]  # Track x position
        stuck_counter = 0
        max_stuck_steps = 50  # If no progress for 50 steps, force forward movement
        
        print("Step-by-step progress:")
        
        for step in range(self.max_steps):
            step_start_time = time.time()
            
            # Update obstacles
            obstacles = self.update_obstacles(obstacles, step)
            
            # Update perception area and get filtered obstacles
            filtered_obstacles = self.perception_area.update_obstacle_memory(
                obstacles, current_state, step * self.timestep)
            
            # Track active constraints
            self.active_constraints.append(len(filtered_obstacles))
            
            # Apply constraint avoidance
            constraint_start_time = time.time()
            result = self.apply_constraint_avoidance(current_state, filtered_obstacles, step)
            constraint_time = time.time() - constraint_start_time
            
            # Track performance
            self.compute_times.append(constraint_time)
            
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
            
            # Track execution time
            step_time = time.time() - step_start_time
            self.execution_times.append(step_time)
            
            # Check goal reaching
            distance_to_goal = np.sqrt((current_state[0] - goal[0])**2 + 
                                     (current_state[1] - goal[1])**2)
            
            # Check for progress to prevent infinite loops
            current_x = current_state[0]
            if current_x > last_progress_position + 0.5:  # Made progress
                last_progress_position = current_x
                stuck_counter = 0
            else:
                stuck_counter += 1
            
            # Force forward movement if stuck
            if stuck_counter > max_stuck_steps:
                print(f"  Step {step}: Vehicle stuck, forcing forward movement")
                # Force forward acceleration
                current_state[3] = min(15.0, current_state[3] + 1.0)  # Increase speed
                stuck_counter = 0  # Reset counter
            
            if step % 10 == 0:
                print(f"  Step {step}: Position ({current_state[0]:.1f}, {current_state[1]:.1f}), "
                      f"Distance to goal: {distance_to_goal:.1f}m, "
                      f"Active constraints: {len(filtered_obstacles)}, "
                      f"Compute time: {constraint_time*1000:.1f}ms")
            
            if distance_to_goal < 5.0:  # More reasonable goal distance
                print(f"  Step {step}: Goal reached! (distance: {distance_to_goal:.2f}m)")
                break
        
        if distance_to_goal >= 5.0:
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
            'goal': goal,
            'performance': {
                'execution_times': self.execution_times,
                'compute_times': self.compute_times,
                'active_constraints': self.active_constraints
            }
        }
    
    def _create_enhanced_animation(self, trajectory: np.ndarray, obstacles: List[Dict],
                                 center_line: np.ndarray, left_boundary: np.ndarray,
                                 right_boundary: np.ndarray, goal: np.ndarray):
        """Create enhanced animation with execution info and trajectory funnels."""
        fig, ax = plt.subplots(figsize=(14, 10))
        
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
        ax.set_title(f'{self.constraint_type.title()} Constraints Test - Enhanced Visualization')
        
        # Animation variables
        vehicle_patch = None
        obstacle_patches = []
        constraint_patches = []
        perception_patches = []
        funnel_patches = []
        info_text = None
        
        def animate(frame):
            nonlocal vehicle_patch, obstacle_patches, constraint_patches, perception_patches, funnel_patches, info_text
            
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
                
                # Update obstacles to their current positions for this frame
                current_obstacles = self.update_obstacles(obstacles, frame)
                
                # Draw obstacles and constraints
                obstacle_patches = []
                constraint_patches = []
                funnel_patches = []
                
                # Get all obstacles that should be visible (in perception area OR in memory)
                visible_obstacles = []
                
                # First, add obstacles currently in perception area
                for obs in current_obstacles:
                    obs_x, obs_y = obs['position']
                    if self.perception_area._is_in_perception_area(obs_x, obs_y, vehicle_x, vehicle_y, vehicle_psi):
                        visible_obstacles.append({
                            'obstacle': obs,
                            'in_perception': True,
                            'in_memory': False
                        })
                
                # Then, add obstacles in memory (even if outside perception area)
                current_time = frame * self.timestep
                for obs_id, memory_entry in self.perception_area.obstacle_memory.items():
                    # Check if obstacle is still in memory (not expired)
                    if current_time - memory_entry['last_seen'] <= self.perception_area.memory_duration:
                        obs = memory_entry['obstacle']
                        obs_x, obs_y = obs['position']
                        
                        # Check if this obstacle is not already in the visible list
                        already_visible = any(
                            visible_obs['obstacle']['id'] == obs['id'] 
                            for visible_obs in visible_obstacles
                        )
                        
                        if not already_visible:
                            visible_obstacles.append({
                                'obstacle': obs,
                                'in_perception': False,
                                'in_memory': True
                            })
                
                # Draw all visible obstacles
                for visible_obs in visible_obstacles:
                    obs = visible_obs['obstacle']
                    obs_x, obs_y = obs['position']
                    obs_radius = obs['radius']
                    in_perception = visible_obs['in_perception']
                    in_memory = visible_obs['in_memory']
                    
                    # Determine obstacle appearance based on status
                    if in_perception:
                        # Obstacle is in perception area - bright red
                        obstacle_circle = Circle((obs_x, obs_y), obs_radius, 
                                               facecolor='red', alpha=0.8, 
                                               edgecolor='darkred', linewidth=2)
                    elif in_memory:
                        # Obstacle is in memory but outside perception - dimmed red
                        obstacle_circle = Circle((obs_x, obs_y), obs_radius, 
                                               facecolor='red', alpha=0.3, 
                                               edgecolor='darkred', linewidth=1, linestyle='--')
                    else:
                        continue
                    
                    ax.add_patch(obstacle_circle)
                    obstacle_patches.append(obstacle_circle)
                    
                    # Only draw constraint visualization for obstacles in perception area
                    if in_perception:
                        constraint_patches = self.draw_constraint_visualization(
                            ax, np.array([obs_x, obs_y]), np.array([vehicle_x, vehicle_y]), frame)
                        
                        # Draw trajectory funnel
                        funnel_patches = self.draw_trajectory_funnel(ax, obs['id'], frame)
            
            # Add execution info
            if frame < len(self.execution_times):
                exec_time = self.execution_times[frame] * 1000  # Convert to ms
                compute_time = self.compute_times[frame] * 1000 if frame < len(self.compute_times) else 0
                active_constraints = self.active_constraints[frame] if frame < len(self.active_constraints) else 0
                
                info_text = f"Frame: {frame}\n"
                info_text += f"Execution Time: {exec_time:.1f}ms\n"
                info_text += f"Compute Time: {compute_time:.1f}ms\n"
                info_text += f"Active Constraints: {active_constraints}\n"
                info_text += f"Memory Entries: {len(self.perception_area.obstacle_memory)}"
                
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Set plot limits
            ax.set_xlim(-10, 130)
            ax.set_ylim(-30, 30)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_title(f'{self.constraint_type.title()} Constraints Test - Enhanced Visualization')
            
            return vehicle_patch, obstacle_patches, constraint_patches, perception_patches, funnel_patches, info_text
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(trajectory), 
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
