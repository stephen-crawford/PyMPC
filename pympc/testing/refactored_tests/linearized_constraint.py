#!/usr/bin/env python3
"""
Linearized constraint test implementation.
"""

import numpy as np
from typing import Dict, List, Any
from .unified_framework import UnifiedConstraintFramework


class LinearizedConstraintTest(UnifiedConstraintFramework):
    """Test for linearized constraints with unified framework."""
    
    def __init__(self, perception_config: Dict[str, Any]):
        """Initialize linearized constraint test."""
        super().__init__('linearized', perception_config)
    
    def _apply_specific_constraint_type(self, current_state: np.ndarray, obstacles: List[Dict], 
                                      step: int) -> Dict[str, Any]:
        """Apply linearized constraint avoidance (MPCC integration)."""
        try:
            x, y, psi, v = current_state
            
            # Linearized constraint avoidance
            steering_angle = 0.0
            acceleration = 0.0
            
            # Apply linearized constraints for obstacles in perception area
            for obs in obstacles:
                obs_x, obs_y = obs['position']
                obs_radius = obs['radius']
                
                # Calculate distance to obstacle
                dist_to_obs = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                
                # HARD CONSTRAINT: Check for immediate collision first
                if dist_to_obs < obs_radius + 0.1:
                    return {
                        'success': False,
                        'message': 'CRASH: Collision with obstacle - linearized constraint violated'
                    }
                
                if dist_to_obs < 15.0:  # Detection range
                    # Linearized constraint: n^T * (x - x_obs) >= d_min
                    # where n is the normal vector and d_min is minimum safe distance
                    d_min = obs_radius + 0.5  # Increased minimum safe distance
                    
                    # Calculate normal vector (pointing from obstacle to vehicle)
                    if dist_to_obs > 1e-6:  # Avoid division by zero
                        obs_normal_x = (x - obs_x) / dist_to_obs
                        obs_normal_y = (y - obs_y) / dist_to_obs
                        
                        # Check constraint violation
                        constraint_violation = (obs_normal_x * (x - obs_x) + obs_normal_y * (y - obs_y)) - d_min
                        
                        if constraint_violation < 0:  # Constraint violated - HARD CONSTRAINT
                            # Apply strong corrective steering
                            violation_magnitude = abs(constraint_violation) / d_min
                            correction_strength = min(0.8, violation_magnitude * 1.0)  # Stronger correction
                            
                            # Calculate avoidance direction (perpendicular to normal)
                            avoid_direction = np.arctan2(-obs_normal_y, obs_normal_x)
                            steering_angle += correction_strength * np.sin(avoid_direction - psi)
                            
                            # Strong speed reduction for constraint violation
                            acceleration = -0.3  # Strong deceleration
                            
                            # If violation is severe, treat as potential collision
                            if violation_magnitude > 0.5:
                                return {
                                    'success': False,
                                    'message': 'CRASH: Severe linearized constraint violation - collision imminent'
                                }
            
            # Clip steering angle
            steering_angle = np.clip(steering_angle, -0.4, 0.4)
            
            return {
                'success': True,
                'steering_angle': steering_angle,
                'acceleration': acceleration,
                'constraint_type': 'linearized'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Linearized constraint error: {str(e)}'
            }
    
    def draw_constraint_visualization(self, ax, obstacle_pos: np.ndarray, 
                                    vehicle_pos: np.ndarray, frame: int) -> List:
        """Draw linearized constraints (rotating half-space representation)."""
        patches = []
        obs_x, obs_y = obstacle_pos[0], obstacle_pos[1]
        veh_x, veh_y = vehicle_pos[0], vehicle_pos[1]
        
        distance = np.sqrt((veh_x - obs_x)**2 + (veh_y - obs_y)**2)
        
        if distance < 12.0: # Only draw if within a certain range
            # Calculate normal vector (pointing from obstacle to vehicle)
            if distance > 1e-6:
                obs_normal_x = (veh_x - obs_x) / distance
                obs_normal_y = (veh_y - obs_y) / distance
            else:
                obs_normal_x, obs_normal_y = 1.0, 0.0 # Default if at same position
            
            # Minimum safe distance
            d_min = 0.5 # Corresponds to obs_radius + 0.5 from _solve_mpc_step
            
            # Half-space line: n_x * (x - obs_x) + n_y * (y - obs_y) = d_min
            # Rotate the half-space based on vehicle's relative position and time
            # This creates a dynamic, "rotating" effect as the vehicle moves
            
            # Angle of the normal vector
            normal_angle = np.arctan2(obs_normal_y, obs_normal_x)
            
            # Add a time-based oscillation for visual dynamism
            oscillation_angle = np.sin(frame * 0.1) * 0.2 # +/- ~11 degrees
            
            # Combine for dynamic rotation
            dynamic_normal_angle = normal_angle + oscillation_angle
            
            # Recalculate normal components for visualization
            vis_normal_x = np.cos(dynamic_normal_angle)
            vis_normal_y = np.sin(dynamic_normal_angle)
            
            # Define points for the half-space line
            # The line is perpendicular to the normal vector
            line_angle = dynamic_normal_angle + np.pi / 2
            
            line_length = 10.0 # Length of the visualized line
            
            # Point on the line (d_min away from obstacle along normal)
            point_on_line_x = obs_x + vis_normal_x * d_min
            point_on_line_y = obs_y + vis_normal_y * d_min
            
            # End points of the line segment
            line_start_x = point_on_line_x - line_length * np.cos(line_angle)
            line_start_y = point_on_line_y - line_length * np.sin(line_angle)
            line_end_x = point_on_line_x + line_length * np.cos(line_angle)
            line_end_y = point_on_line_y + line_length * np.sin(line_angle)
            
            ax.plot([line_start_x, line_end_x], [line_start_y, line_end_y], 
                    'r-', linewidth=2, alpha=0.7, label='Linearized Constraint Boundary')
            
            # Draw an arrow to indicate the "safe" side (away from obstacle)
            arrow_length = 3.0
            arrow_start_x = point_on_line_x
            arrow_start_y = point_on_line_y
            arrow_end_x = point_on_line_x + vis_normal_x * arrow_length
            arrow_end_y = point_on_line_y + vis_normal_y * arrow_length
            
            ax.arrow(arrow_start_x, arrow_start_y, 
                    arrow_end_x - arrow_start_x, arrow_end_y - arrow_start_y, 
                    head_width=0.8, head_length=1.0, fc='red', ec='red', alpha=0.7)
            
        return patches