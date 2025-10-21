"""
Linearized Constraint Test using Unified Framework

This module implements linearized half-space constraints using the unified framework
with configurable perception areas and obstacle memory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from typing import List, Dict, Any
from .unified_constraint_framework import UnifiedConstraintTest


class LinearizedConstraintTest(UnifiedConstraintTest):
    """Test for linearized half-space constraints with unified framework."""
    
    def __init__(self, perception_config: Dict[str, Any]):
        """Initialize linearized constraint test."""
        super().__init__('linearized', perception_config)
    
    def apply_constraint_avoidance(self, current_state: np.ndarray, obstacles: List[Dict], 
                                 step: int) -> Dict[str, Any]:
        """Apply linearized half-space constraint avoidance."""
        try:
            x, y, psi, v = current_state
            dt = self.timestep
            
            # Calculate desired heading towards goal
            goal_x = self.road_length
            goal_y = 0.0  # Approximate goal position
            dx = goal_x - x
            dy = goal_y - y
            desired_psi = np.arctan2(dy, dx)
            
            # Linearized constraint avoidance
            steering_angle = (desired_psi - psi) * 0.5
            acceleration = 0.3
            
            # Apply hard road boundary constraints FIRST
            road_correction = self._apply_road_boundary_constraints(current_state, steering_angle)
            steering_angle += road_correction
            
            # Apply linearized constraints for obstacles in perception area
            for obs in obstacles:
                obs_x, obs_y = obs['position']
                obs_radius = obs['radius']
                
                # Calculate distance to obstacle
                dist_to_obs = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                
                if dist_to_obs < 15.0:  # Detection range
                    # Linearized constraint: n^T * (x - x_obs) >= d_min
                    # where n is the normal vector and d_min is minimum safe distance
                    d_min = obs_radius + 0.2  # Minimum safe distance
                    
                    # Calculate normal vector (pointing from obstacle to vehicle)
                    if dist_to_obs > 1e-6:  # Avoid division by zero
                        obs_normal_x = (x - obs_x) / dist_to_obs
                        obs_normal_y = (y - obs_y) / dist_to_obs
                        
                        # Check constraint violation
                        constraint_violation = (obs_normal_x * (x - obs_x) + obs_normal_y * (y - obs_y)) - d_min
                        
                        if constraint_violation < 0:  # Constraint violated
                            # Apply corrective steering
                            violation_magnitude = abs(constraint_violation) / d_min
                            correction_strength = min(0.3, violation_magnitude * 0.5)
                            
                            # Calculate avoidance direction (perpendicular to normal)
                            avoid_direction = np.arctan2(-obs_normal_y, obs_normal_x)
                            steering_angle += correction_strength * np.sin(avoid_direction - psi)
                            
                            # Speed reduction
                            if violation_magnitude > 0.3:
                                acceleration = -0.05  # Gentle deceleration
                
                # Check for crash
                if dist_to_obs < obs_radius + 0.1:
                    return {
                        'success': False,
                        'message': 'CRASH: Linearized constraint violated - collision imminent'
                    }
            
            # Clip steering angle
            steering_angle = np.clip(steering_angle, -0.4, 0.4)
            
            # State propagation
            next_x = x + v * np.cos(psi) * dt
            next_y = y + v * np.sin(psi) * dt
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
    
    def _apply_road_boundary_constraints(self, state: np.ndarray, current_steering: float) -> float:
        """Apply hard road boundary constraints (contouring constraints)."""
        x, y = state[0], state[1]
        
        # Road parameters
        road_width = 8.0
        half_width = road_width / 2
        safety_margin = 2.0
        
        # Calculate expected y position based on road curvature
        s = x / 120.0  # Normalize x position
        expected_y = (10 * np.sin(2 * np.pi * s * 0.3) + 
                     5 * np.sin(2 * np.pi * s * 0.6) + 
                     2.5 * np.sin(2 * np.pi * s * 1.2))
        
        # Calculate road boundary positions
        left_boundary = expected_y - half_width - safety_margin
        right_boundary = expected_y + half_width + safety_margin
        
        # Apply steering corrections to keep vehicle within road boundaries
        steering_correction = 0.0
        
        # Left boundary constraint (vehicle too far left)
        if y < left_boundary:
            correction_strength = max(0.1, 1.0 - (left_boundary - y) / 1.5)
            steering_correction += correction_strength * 0.2  # Gentle turn right
        
        # Right boundary constraint (vehicle too far right)
        if y > right_boundary:
            correction_strength = max(0.1, 1.0 - (y - right_boundary) / 1.5)
            steering_correction -= correction_strength * 0.2  # Gentle turn left
        
        return steering_correction
    
    def draw_constraint_visualization(self, ax, obstacle_pos: np.ndarray, 
                                    vehicle_pos: np.ndarray, frame: int) -> List:
        """Draw linearized constraints (rotating half-space representation)."""
        patches = []
        obs_x, obs_y = obstacle_pos[0], obstacle_pos[1]
        veh_x, veh_y = vehicle_pos[0], vehicle_pos[1]
        
        distance = np.sqrt((veh_x - obs_x)**2 + (veh_y - obs_y)**2)
        
        if distance < 12.0:
            angle = np.arctan2(obs_y - veh_y, obs_x - veh_x)
            vehicle_rotation = 0.1 * np.sin(frame * 0.1)
            time_rotation = frame * 0.05
            total_rotation = angle + vehicle_rotation + time_rotation
            
            constraint_radius = 8.0
            half_angle_range = np.pi/3
            
            polygon_points = []
            for ang in np.linspace(total_rotation - half_angle_range, total_rotation + half_angle_range, 20):
                px = obs_x + constraint_radius * np.cos(ang)
                py = obs_y + constraint_radius * np.sin(ang)
                polygon_points.append([px, py])
            
            polygon_points.append([obs_x, obs_y])
            
            constraint_polygon = Polygon(polygon_points, 
                                       facecolor='red', alpha=0.3,
                                       edgecolor='red', linewidth=2)
            patches.append(constraint_polygon)
            
            line_length = 6.0
            line_start_x = obs_x - line_length * np.cos(total_rotation)
            line_start_y = obs_y - line_length * np.sin(total_rotation)
            line_end_x = obs_x + line_length * np.cos(total_rotation)
            line_end_y = obs_y + line_length * np.sin(total_rotation)
            
            ax.plot([line_start_x, line_end_x], [line_start_y, line_end_y], 
                   'r--', linewidth=3, alpha=0.8)
            
            arrow_length = 3.0
            arrow_x = obs_x + arrow_length * np.cos(total_rotation)
            arrow_y = obs_y + arrow_length * np.sin(total_rotation)
            ax.annotate('', xy=(arrow_x, arrow_y), xytext=(obs_x, obs_y),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.8))
        
        return patches


def run_linearized_test():
    """Run the linearized constraint test with configurable perception area."""
    # Configure perception area
    perception_config = {
        'type': 'forward_cone',  # 'radius', 'sphere', 'forward_cone', 'bidirectional_cone', 'rectangle'
        'radius': 20.0,  # For radius/sphere perception
        'forward_angle': np.pi/3,  # For cone perception (60 degrees)
        'backward_angle': np.pi/6,  # For bidirectional cone (30 degrees)
        'forward_distance': 25.0,  # For cone/rectangle perception
        'backward_distance': 10.0,  # For bidirectional cone/rectangle perception
        'width': 15.0,  # For rectangle perception
        'height': 20.0,  # For rectangle perception
        'memory_duration': 3.0  # Obstacle memory duration in seconds
    }
    
    # Create and run test
    test = LinearizedConstraintTest(perception_config)
    result = test.run_test()
    
    # Print results
    print("\n" + "="*60)
    print("LINEARIZED CONSTRAINTS TEST RESULTS")
    print("="*60)
    print(f"Status: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
    if result['success']:
        print(f"Trajectory Length: {len(result['trajectory'])} steps")
        print(f"Total Distance: {test.road_length}m")
        print(f"Animation: {test.output_dir}/linearized_constraints_test.gif")
        print(f"Constraint Type: Linearized (Half-space constraints)")
        print(f"Perception Area: {perception_config['type']}")
        print(f"Memory Duration: {perception_config['memory_duration']}s")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("\n🎉 Test completed successfully!")
    print(f"Check the animation at: {test.output_dir}/linearized_constraints_test.gif")
    
    return result


if __name__ == "__main__":
    run_linearized_test()
