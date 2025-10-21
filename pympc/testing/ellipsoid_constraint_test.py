"""
Ellipsoid Constraint Test using Unified Framework

This module implements ellipsoid constraints using the unified framework
with configurable perception areas and obstacle memory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Ellipse
from typing import List, Dict, Any
from .unified_constraint_framework import UnifiedConstraintTest


class EllipsoidConstraintTest(UnifiedConstraintTest):
    """Test for ellipsoid constraints with unified framework."""
    
    def __init__(self, perception_config: Dict[str, Any]):
        """Initialize ellipsoid constraint test."""
        super().__init__('ellipsoid', perception_config)
    
    def apply_constraint_avoidance(self, current_state: np.ndarray, obstacles: List[Dict], 
                                 step: int) -> Dict[str, Any]:
        """Apply ellipsoid constraint avoidance."""
        try:
            x, y, psi, v = current_state
            dt = self.timestep
            
            # Calculate desired heading towards goal
            goal_x = self.road_length
            goal_y = 0.0  # Approximate goal position
            dx = goal_x - x
            dy = goal_y - y
            desired_psi = np.arctan2(dy, dx)
            
            # Ellipsoid constraint avoidance
            steering_angle = (desired_psi - psi) * 0.5
            acceleration = 0.3
            
            # Apply hard road boundary constraints FIRST
            road_correction = self._apply_road_boundary_constraints(current_state, steering_angle)
            steering_angle += road_correction
            
            # Apply ellipsoid constraints for obstacles in perception area
            for obs in obstacles:
                obs_x, obs_y = obs['position']
                obs_radius = obs['radius']
                
                # Calculate distance to obstacle
                dist_to_obs = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                
                if dist_to_obs < 15.0:  # Detection range
                    # Calculate relative position vector
                    rel_x = x - obs_x
                    rel_y = y - obs_y
                    
                    # Ellipsoid shape parameters (major and minor axes)
                    major_axis = obs_radius + 1.0  # Ellipsoid major axis
                    minor_axis = obs_radius + 0.5  # Ellipsoid minor axis
                    
                    # Ellipsoid constraint: (x - x_obs)^T * P * (x - x_obs) >= 1
                    # where P = diag(1/a^2, 1/b^2) for ellipse with axes a, b
                    ellipsoid_value = (rel_x**2 / major_axis**2) + (rel_y**2 / minor_axis**2)
                    
                    if ellipsoid_value < 1.0:  # Constraint violated
                        # Apply corrective steering
                        violation_magnitude = (1.0 - ellipsoid_value) / 1.0
                        correction_strength = min(0.4, violation_magnitude * 0.8)
                        
                        # Calculate gradient of ellipsoid constraint
                        grad_x = 2 * rel_x / major_axis**2
                        grad_y = 2 * rel_y / minor_axis**2
                        avoid_direction = np.arctan2(grad_y, grad_x)
                        
                        # Apply steering correction
                        steering_angle += correction_strength * np.sin(avoid_direction - psi)
                        
                        # Speed reduction based on constraint violation
                        if violation_magnitude > 0.3:
                            acceleration = -0.1  # Moderate deceleration
                
                # Check for crash
                if dist_to_obs < obs_radius + 0.2:
                    return {
                        'success': False,
                        'message': 'CRASH: Ellipsoid constraint violated - collision imminent'
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
        """Draw ellipsoid constraints (moving elliptical boundaries)."""
        patches = []
        obs_x, obs_y = obstacle_pos[0], obstacle_pos[1]
        veh_x, veh_y = vehicle_pos[0], vehicle_pos[1]
        
        distance = np.sqrt((veh_x - obs_x)**2 + (veh_y - obs_y)**2)
        
        if distance < 12.0:
            # Main ellipsoid constraint
            major_axis = 3.0 + 1.0 * np.sin(frame * 0.1)  # Oscillating size
            minor_axis = 2.0 + 0.5 * np.cos(frame * 0.15)  # Oscillating size
            
            # Create ellipsoid
            ellipse = Ellipse((obs_x, obs_y), major_axis * 2, minor_axis * 2,
                            angle=frame * 2,  # Rotating ellipsoid
                            facecolor='orange', alpha=0.3,
                            edgecolor='red', linewidth=2)
            patches.append(ellipse)
            
            # Inner safety ellipsoid
            safety_major = major_axis * 0.7
            safety_minor = minor_axis * 0.7
            safety_ellipse = Ellipse((obs_x, obs_y), safety_major * 2, safety_minor * 2,
                                   angle=frame * 2,
                                   facecolor='yellow', alpha=0.2,
                                   edgecolor='orange', linewidth=1)
            patches.append(safety_ellipse)
            
            # Gradient lines showing constraint direction
            num_gradients = 8
            for i in range(num_gradients):
                angle = i * 2 * np.pi / num_gradients + frame * 0.1
                grad_length = 2.0
                grad_start_x = obs_x + major_axis * 0.8 * np.cos(angle)
                grad_start_y = obs_y + minor_axis * 0.8 * np.sin(angle)
                grad_end_x = grad_start_x + grad_length * np.cos(angle)
                grad_end_y = grad_start_y + grad_length * np.sin(angle)
                
                ax.plot([grad_start_x, grad_end_x], [grad_start_y, grad_end_y],
                       'r-', linewidth=1, alpha=0.6)
            
            # Dynamic boundary ellipse
            boundary_angle = frame * 0.05
            boundary_major = major_axis + 0.5 * np.sin(boundary_angle)
            boundary_minor = minor_axis + 0.3 * np.cos(boundary_angle)
            
            boundary_ellipse = Ellipse((obs_x, obs_y), boundary_major * 2, boundary_minor * 2,
                                     angle=boundary_angle * 10,
                                     fill=False, edgecolor='red', linewidth=3,
                                     linestyle='--', alpha=0.8)
            patches.append(boundary_ellipse)
        
        return patches


def run_ellipsoid_test():
    """Run the ellipsoid constraint test with configurable perception area."""
    # Configure perception area
    perception_config = {
        'type': 'bidirectional_cone',  # 'radius', 'sphere', 'forward_cone', 'bidirectional_cone', 'rectangle'
        'radius': 20.0,  # For radius/sphere perception
        'forward_angle': np.pi/3,  # For cone perception (60 degrees)
        'backward_angle': np.pi/6,  # For bidirectional cone (30 degrees)
        'forward_distance': 25.0,  # For cone/rectangle perception
        'backward_distance': 10.0,  # For bidirectional cone/rectangle perception
        'width': 15.0,  # For rectangle perception
        'height': 20.0,  # For rectangle perception
        'memory_duration': 4.0  # Obstacle memory duration in seconds
    }
    
    # Create and run test
    test = EllipsoidConstraintTest(perception_config)
    result = test.run_test()
    
    # Print results
    print("\n" + "="*60)
    print("ELLIPSOID CONSTRAINTS TEST RESULTS")
    print("="*60)
    print(f"Status: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
    if result['success']:
        print(f"Trajectory Length: {len(result['trajectory'])} steps")
        print(f"Total Distance: {test.road_length}m")
        print(f"Animation: {test.output_dir}/ellipsoid_constraints_test.gif")
        print(f"Constraint Type: Ellipsoid (Smooth elliptical boundaries)")
        print(f"Perception Area: {perception_config['type']}")
        print(f"Memory Duration: {perception_config['memory_duration']}s")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("\n🎉 Test completed successfully!")
    print(f"Check the animation at: {test.output_dir}/ellipsoid_constraints_test.gif")
    
    return result


if __name__ == "__main__":
    run_ellipsoid_test()
