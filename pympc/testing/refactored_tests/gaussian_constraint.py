#!/usr/bin/env python3
"""
Gaussian constraint test implementation.
"""

import numpy as np
from typing import Dict, List, Any
from .unified_framework import UnifiedConstraintFramework


class GaussianConstraintTest(UnifiedConstraintFramework):
    """Test for Gaussian constraints with unified framework."""
    
    def __init__(self, perception_config: Dict[str, Any]):
        """Initialize Gaussian constraint test."""
        super().__init__('gaussian', perception_config)
    
    def _apply_specific_constraint_type(self, current_state: np.ndarray, obstacles: List[Dict], 
                                      step: int) -> Dict[str, Any]:
        """Apply Gaussian constraint avoidance (MPCC integration)."""
        try:
            x, y, psi, v = current_state
            
            # Gaussian constraint avoidance
            steering_angle = 0.0
            acceleration = 0.0
            
            # Apply Gaussian constraints for obstacles in perception area
            for obs in obstacles:
                obs_x, obs_y = obs['position']
                obs_radius = obs['radius']
                
                # Calculate distance to obstacle
                dist_to_obs = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                
                # HARD CONSTRAINT: Check for immediate collision first
                if dist_to_obs < obs_radius + 0.1:
                    return {
                        'success': False,
                        'message': 'CRASH: Collision with obstacle - Gaussian constraint violated'
                    }
                
                if dist_to_obs < 15.0:  # Detection range
                    # Gaussian constraint: (x - x_obs)^T * Sigma^(-1) * (x - x_obs) >= threshold
                    # where Sigma is the covariance matrix
                    rel_x = x - obs_x
                    rel_y = y - obs_y
                    
                    # Define Gaussian parameters with increased safety margin
                    sigma_x = obs_radius + 1.5  # Increased safety margin
                    sigma_y = obs_radius + 1.2
                    threshold = 2.5  # Higher confidence level for safety
                    
                    # Calculate Gaussian constraint value
                    gaussian_value = (rel_x**2 / sigma_x**2) + (rel_y**2 / sigma_y**2)
                    
                    if gaussian_value < threshold:  # Constraint violated - HARD CONSTRAINT
                        violation_magnitude = threshold - gaussian_value
                        correction_strength = min(0.8, violation_magnitude * 1.0)  # Stronger correction
                        
                        # Calculate gradient of Gaussian function for avoidance direction
                        grad_x = 2 * rel_x / sigma_x**2
                        grad_y = 2 * rel_y / sigma_y**2
                        
                        # Avoidance direction is along the gradient
                        avoid_direction = np.arctan2(grad_y, grad_x)
                        steering_angle += correction_strength * np.sin(avoid_direction - psi)
                        
                        # Strong speed reduction for constraint violation
                        acceleration = -0.3
                        
                        # If violation is severe, treat as potential collision
                        if violation_magnitude > 0.5:
                            return {
                                'success': False,
                                'message': 'CRASH: Severe Gaussian constraint violation - collision imminent'
                            }
            
            # Clip steering angle
            steering_angle = np.clip(steering_angle, -0.4, 0.4)
            
            return {
                'success': True,
                'steering_angle': steering_angle,
                'acceleration': acceleration,
                'constraint_type': 'gaussian'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Gaussian constraint error: {str(e)}'
            }
    
    def draw_constraint_visualization(self, ax, obstacle_pos: np.ndarray, 
                                    vehicle_pos: np.ndarray, frame: int) -> List:
        """Draw Gaussian constraints (moving probabilistic boundaries)."""
        patches = []
        obs_x, obs_y = obstacle_pos[0], obstacle_pos[1]
        veh_x, veh_y = vehicle_pos[0], vehicle_pos[1]
        
        distance = np.sqrt((veh_x - obs_x)**2 + (veh_y - obs_y)**2)
        
        if distance < 12.0:
            # Define Gaussian parameters
            sigma_x = 3.0 + 0.5 * np.sin(frame * 0.1)
            sigma_y = 2.5 + 0.3 * np.cos(frame * 0.15)
            threshold = 2.5 # Corresponds to the threshold in _solve_mpc_step
            
            # Create a grid for visualization
            x_grid = np.linspace(obs_x - 10, obs_x + 10, 100)
            y_grid = np.linspace(obs_y - 10, obs_y + 10, 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            # Calculate Gaussian values
            Z = ((X - obs_x)**2 / sigma_x**2) + ((Y - obs_y)**2 / sigma_y**2)
            
            # Plot contour lines for different confidence levels
            contour_levels = [1.0, threshold, 4.0] # Inner, constraint, outer
            colors = ['orange', 'red', 'darkred']
            linestyles = ['--', '-', '--']
            
            for i, level in enumerate(contour_levels):
                cs = ax.contour(X, Y, Z, levels=[level], colors=colors[i], 
                                linestyles=linestyles[i], linewidths=2, alpha=0.7)
                for collection in cs.collections:
                    patches.append(collection)
            
            # Draw prediction arrows (dynamic rotation)
            for i in range(4):
                angle = i * np.pi / 2 + frame * 0.05
                line_length = 3.0 + 0.5 * np.sin(frame * 0.1 + angle)
                end_x = obs_x + line_length * np.cos(angle)
                end_y = obs_y + line_length * np.sin(angle)
                ax.arrow(obs_x, obs_y, end_x - obs_x, end_y - obs_y, 
                        head_width=0.8, head_length=1.0, fc='red', ec='red', alpha=0.6)
        
        return patches