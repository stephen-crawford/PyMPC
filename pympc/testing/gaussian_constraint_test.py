"""
Gaussian Constraint Test using Unified Framework

This module implements Gaussian constraints using the unified framework
with configurable perception areas and obstacle memory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Ellipse
from typing import List, Dict, Any
from .unified_constraint_framework import UnifiedConstraintTest


class GaussianConstraintTest(UnifiedConstraintTest):
    """Test for Gaussian constraints with unified framework."""
    
    def __init__(self, perception_config: Dict[str, Any]):
        """Initialize Gaussian constraint test."""
        super().__init__('gaussian', perception_config)
    
    def apply_constraint_avoidance(self, current_state: np.ndarray, obstacles: List[Dict], 
                                 step: int) -> Dict[str, Any]:
        """Apply Gaussian constraint avoidance."""
        try:
            x, y, psi, v = current_state
            dt = self.timestep
            
            # Calculate desired heading towards goal
            goal_x = self.road_length
            goal_y = 0.0  # Approximate goal position
            dx = goal_x - x
            dy = goal_y - y
            desired_psi = np.arctan2(dy, dx)
            
            # Gaussian constraint avoidance
            steering_angle = (desired_psi - psi) * 0.5
            acceleration = 0.3
            
            # Apply hard road boundary constraints FIRST
            road_correction = self._apply_road_boundary_constraints(current_state, steering_angle)
            steering_angle += road_correction
            
            # Apply Gaussian constraints for obstacles in perception area
            for obs in obstacles:
                obs_x, obs_y = obs['position']
                obs_radius = obs['radius']
                
                # Calculate distance to obstacle
                dist_to_obs = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                
                if dist_to_obs < 15.0:  # Detection range
                    # Calculate relative position vector
                    rel_x = x - obs_x
                    rel_y = y - obs_y
                    
                    # Gaussian constraint parameters
                    sigma_x = obs_radius + 1.0  # Standard deviation in x
                    sigma_y = obs_radius + 0.8  # Standard deviation in y
                    
                    # Gaussian constraint: (x - x_obs)^T * Sigma^(-1) * (x - x_obs) >= threshold
                    # where Sigma is the covariance matrix
                    gaussian_value = (rel_x**2 / sigma_x**2) + (rel_y**2 / sigma_y**2)
                    threshold = 2.0  # 2-sigma confidence level
                    
                    if gaussian_value < threshold:  # Constraint violated
                        # Apply corrective steering
                        violation_magnitude = (threshold - gaussian_value) / threshold
                        correction_strength = min(0.4, violation_magnitude * 0.6)
                        
                        # Calculate gradient of Gaussian constraint
                        grad_x = 2 * rel_x / sigma_x**2
                        grad_y = 2 * rel_y / sigma_y**2
                        avoid_direction = np.arctan2(grad_y, grad_x)
                        
                        # Apply steering correction
                        steering_angle += correction_strength * np.sin(avoid_direction - psi)
                        
                        # Speed reduction based on constraint violation
                        if violation_magnitude > 0.3:
                            acceleration = -0.1  # Moderate deceleration
                
                # Check for crash
                if dist_to_obs < obs_radius + 0.5:
                    return {
                        'success': False,
                        'message': 'CRASH: Gaussian constraint violated - collision imminent'
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
        """Draw Gaussian constraints (moving probabilistic boundaries)."""
        patches = []
        obs_x, obs_y = obstacle_pos[0], obstacle_pos[1]
        veh_x, veh_y = vehicle_pos[0], vehicle_pos[1]
        
        distance = np.sqrt((veh_x - obs_x)**2 + (veh_y - obs_y)**2)
        
        if distance < 12.0:
            # Multiple confidence levels for Gaussian constraints
            confidence_levels = [1.0, 1.5, 2.0, 2.5]  # 1-sigma, 1.5-sigma, 2-sigma, 2.5-sigma
            colors = ['green', 'yellow', 'orange', 'red']
            alphas = [0.1, 0.2, 0.3, 0.4]
            
            for i, (level, color, alpha) in enumerate(zip(confidence_levels, colors, alphas)):
                # Oscillating Gaussian parameters
                sigma_x = 2.0 + 0.5 * np.sin(frame * 0.1 + i * 0.5)
                sigma_y = 1.5 + 0.3 * np.cos(frame * 0.15 + i * 0.3)
                
                # Create Gaussian ellipse
                ellipse = Ellipse((obs_x, obs_y), sigma_x * level * 2, sigma_y * level * 2,
                                angle=frame * 1,  # Rotating Gaussian
                                facecolor=color, alpha=alpha,
                                edgecolor=color, linewidth=1)
                patches.append(ellipse)
            
            # Rotating prediction arrows
            num_arrows = 6
            for i in range(num_arrows):
                angle = i * 2 * np.pi / num_arrows + frame * 0.1
                arrow_length = 3.0 + 1.0 * np.sin(frame * 0.2 + i)
                arrow_x = obs_x + arrow_length * np.cos(angle)
                arrow_y = obs_y + arrow_length * np.sin(angle)
                
                ax.annotate('', xy=(arrow_x, arrow_y), xytext=(obs_x, obs_y),
                           arrowprops=dict(arrowstyle='->', color='purple', lw=1, alpha=0.6))
            
            # Dynamic Gaussian boundary
            boundary_sigma_x = 2.5 + 0.8 * np.sin(frame * 0.05)
            boundary_sigma_y = 2.0 + 0.6 * np.cos(frame * 0.08)
            boundary_level = 2.0
            
            boundary_ellipse = Ellipse((obs_x, obs_y), 
                                     boundary_sigma_x * boundary_level * 2, 
                                     boundary_sigma_y * boundary_level * 2,
                                     angle=frame * 0.5,
                                     fill=False, edgecolor='purple', linewidth=3,
                                     linestyle='--', alpha=0.8)
            patches.append(boundary_ellipse)
        
        return patches


def run_gaussian_test():
    """Run the Gaussian constraint test with configurable perception area."""
    # Configure perception area
    perception_config = {
        'type': 'rectangle',  # 'radius', 'sphere', 'forward_cone', 'bidirectional_cone', 'rectangle'
        'radius': 20.0,  # For radius/sphere perception
        'forward_angle': np.pi/3,  # For cone perception (60 degrees)
        'backward_angle': np.pi/6,  # For bidirectional cone (30 degrees)
        'forward_distance': 25.0,  # For cone/rectangle perception
        'backward_distance': 10.0,  # For bidirectional cone/rectangle perception
        'width': 15.0,  # For rectangle perception
        'height': 20.0,  # For rectangle perception
        'memory_duration': 5.0  # Obstacle memory duration in seconds
    }
    
    # Create and run test
    test = GaussianConstraintTest(perception_config)
    result = test.run_test()
    
    # Print results
    print("\n" + "="*60)
    print("GAUSSIAN CONSTRAINTS TEST RESULTS")
    print("="*60)
    print(f"Status: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
    if result['success']:
        print(f"Trajectory Length: {len(result['trajectory'])} steps")
        print(f"Total Distance: {test.road_length}m")
        print(f"Animation: {test.output_dir}/gaussian_constraints_test.gif")
        print(f"Constraint Type: Gaussian (Probabilistic uncertainty)")
        print(f"Perception Area: {perception_config['type']}")
        print(f"Memory Duration: {perception_config['memory_duration']}s")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("\n🎉 Test completed successfully!")
    print(f"Check the animation at: {test.output_dir}/gaussian_constraints_test.gif")
    
    return result


if __name__ == "__main__":
    run_gaussian_test()
