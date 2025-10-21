#!/usr/bin/env python3
"""
Ellipsoid constraint test implementation.
"""

import numpy as np
from typing import Dict, List, Any
from matplotlib.patches import Ellipse
from .unified_framework import UnifiedConstraintFramework


class EllipsoidConstraintTest(UnifiedConstraintFramework):
    """Test for ellipsoid constraints with unified framework."""
    
    def __init__(self, perception_config: Dict[str, Any]):
        """Initialize ellipsoid constraint test."""
        super().__init__('ellipsoid', perception_config)
    
    def _apply_specific_constraint_type(self, current_state: np.ndarray, obstacles: List[Dict], 
                                      step: int) -> Dict[str, Any]:
        """Apply ellipsoid constraint avoidance (MPCC integration)."""
        try:
            x, y, psi, v = current_state
            
            # Ellipsoid constraint avoidance
            steering_angle = 0.0
            acceleration = 0.0
            
            # Apply ellipsoid constraints for obstacles in perception area
            for obs in obstacles:
                obs_x, obs_y = obs['position']
                obs_radius = obs['radius']
                
                # Calculate distance to obstacle
                dist_to_obs = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                
                # HARD CONSTRAINT: Check for immediate collision first
                if dist_to_obs < obs_radius + 0.1:
                    return {
                        'success': False,
                        'message': 'CRASH: Collision with obstacle - ellipsoid constraint violated'
                    }
                
                if dist_to_obs < 15.0:  # Detection range
                    # Ellipsoid constraint: (x - x_obs)^T * P * (x - x_obs) >= 1
                    # where P is the ellipsoid matrix
                    rel_x = x - obs_x
                    rel_y = y - obs_y
                    
                    # Define ellipsoid parameters with increased safety margin
                    major_axis = obs_radius + 1.5  # Increased safety margin
                    minor_axis = obs_radius + 1.0
                    
                    # Calculate ellipsoid constraint value
                    ellipsoid_value = (rel_x**2 / major_axis**2) + (rel_y**2 / minor_axis**2)
                    
                    if ellipsoid_value < 1.0:  # Constraint violated - HARD CONSTRAINT
                        violation_magnitude = 1.0 - ellipsoid_value
                        correction_strength = min(0.8, violation_magnitude * 1.0)  # Stronger correction
                        
                        # Calculate gradient of ellipsoid function for avoidance direction
                        grad_x = 2 * rel_x / major_axis**2
                        grad_y = 2 * rel_y / minor_axis**2
                        
                        # Avoidance direction is along the gradient
                        avoid_direction = np.arctan2(grad_y, grad_x)
                        steering_angle += correction_strength * np.sin(avoid_direction - psi)
                        
                        # Strong speed reduction for constraint violation
                        acceleration = -0.3
                        
                        # If violation is severe, treat as potential collision
                        if violation_magnitude > 0.5:
                            return {
                                'success': False,
                                'message': 'CRASH: Severe ellipsoid constraint violation - collision imminent'
                            }
            
            # Clip steering angle
            steering_angle = np.clip(steering_angle, -0.4, 0.4)
            
            return {
                'success': True,
                'steering_angle': steering_angle,
                'acceleration': acceleration,
                'constraint_type': 'ellipsoid'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Ellipsoid constraint error: {str(e)}'
            }
    
    def draw_constraint_visualization(self, ax, obstacle_pos: np.ndarray, 
                                    vehicle_pos: np.ndarray, frame: int) -> List:
        """Draw ellipsoid constraints (moving elliptical boundaries)."""
        patches = []
        obs_x, obs_y = obstacle_pos[0], obstacle_pos[1]
        veh_x, veh_y = vehicle_pos[0], vehicle_pos[1]
        
        distance = np.sqrt((veh_x - obs_x)**2 + (veh_y - obs_y)**2)
        
        if distance < 12.0:
            # Main ellipsoid boundary
            major_axis = 3.0 + 0.5 * np.sin(frame * 0.1)
            minor_axis = 2.0 + 0.3 * np.cos(frame * 0.15)
            
            ellipse = Ellipse((obs_x, obs_y), major_axis * 2, minor_axis * 2,
                          facecolor='red', alpha=0.3, edgecolor='red', linewidth=2)
            patches.append(ellipse)
            
            # Inner safety ellipse
            safety_major = major_axis * 0.7
            safety_minor = minor_axis * 0.7
            safety_ellipse = Ellipse((obs_x, obs_y), safety_major * 2, safety_minor * 2,
                                  facecolor='orange', alpha=0.2, edgecolor='orange', linewidth=1)
            patches.append(safety_ellipse)
            
            # Gradient lines showing constraint direction
            for i in range(8):
                angle = i * np.pi / 4
                line_length = 2.0 + 0.5 * np.sin(frame * 0.1 + angle)
                end_x = obs_x + line_length * np.cos(angle)
                end_y = obs_y + line_length * np.sin(angle)
                ax.plot([obs_x, end_x], [obs_y, end_y], 'r-', linewidth=1, alpha=0.6)
            
            # Dynamic boundary ellipse
            boundary_angle = frame * 0.05
            boundary_major = 4.0 + 0.5 * np.sin(frame * 0.2)
            boundary_minor = 2.5 + 0.3 * np.cos(frame * 0.25)
            
            # Rotate ellipse
            cos_angle = np.cos(boundary_angle)
            sin_angle = np.sin(boundary_angle)
            
            # Create rotated ellipse points
            t = np.linspace(0, 2*np.pi, 50)
            ellipse_x = boundary_major * np.cos(t)
            ellipse_y = boundary_minor * np.sin(t)
            
            # Apply rotation
            rotated_x = obs_x + ellipse_x * cos_angle - ellipse_y * sin_angle
            rotated_y = obs_y + ellipse_x * sin_angle + ellipse_y * cos_angle
            
            ax.plot(rotated_x, rotated_y, 'r--', linewidth=2, alpha=0.7)
        
        return patches