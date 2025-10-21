#!/usr/bin/env python3
"""
Scenario constraint test implementation.
"""

import numpy as np
from typing import Dict, List, Any
from matplotlib.patches import Circle
from .unified_framework import UnifiedConstraintFramework


class ScenarioConstraintTest(UnifiedConstraintFramework):
    """Test for scenario constraints with unified framework."""
    
    def __init__(self, perception_config: Dict[str, Any]):
        """Initialize scenario constraint test."""
        super().__init__('scenario', perception_config)
        
        # Scenario-specific parameters
        self.num_scenarios = 5
        self.scenario_weights = [0.4, 0.25, 0.15, 0.1, 0.1]
        self.scenario_names = ['continue', 'accelerate', 'turn_left', 'turn_right', 'decelerate']
        
        # Recall memory configuration
        self.recall_memory_enabled = perception_config.get('recall_memory', True)
        self.recall_duration = perception_config.get('recall_duration', 8.0)  # Extended recall for scenarios
        
        # Trajectory funnel configuration
        self.funnel_enabled = perception_config.get('trajectory_funnels', True)
        self.funnel_horizon = perception_config.get('funnel_horizon', 5.0)
    
    def _apply_specific_constraint_type(self, current_state: np.ndarray, obstacles: List[Dict], 
                                      step: int) -> Dict[str, Any]:
        """Apply scenario constraint avoidance (MPCC integration)."""
        try:
            x, y, psi, v = current_state
            
            # Scenario constraint avoidance
            steering_angle = 0.0
            acceleration = 0.0
            
            # Apply scenario constraints for obstacles in perception area
            for obs in obstacles:
                obs_x, obs_y = obs['position']
                obs_radius = obs['radius']
                
                # Calculate distance to obstacle
                dist_to_obs = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                
                # HARD CONSTRAINT: Check for immediate collision first
                if dist_to_obs < obs_radius + 0.1:
                    return {
                        'success': False,
                        'message': 'CRASH: Collision with obstacle - scenario constraint violated'
                    }
                
                if dist_to_obs < 20.0:  # Extended detection range for scenarios
                    # Apply scenario constraints (multiple possible outcomes)
                    for scenario_idx in range(self.num_scenarios):
                        scenario_weight = self.scenario_weights[scenario_idx]
                        scenario_name = self.scenario_names[scenario_idx]
                        
                        # Generate scenario prediction
                        scenario_pred = self._generate_scenario_prediction(obs, scenario_name, scenario_idx)
                        scenario_x, scenario_y = scenario_pred['position']
                        scenario_radius = scenario_pred['radius']
                        
                        # Calculate distance to scenario prediction
                        scenario_dist = np.sqrt((x - scenario_x)**2 + (y - scenario_y)**2)
                        scenario_safety_margin = scenario_radius + 1.5  # Increased safety margin
                        
                        # Check scenario constraint violation
                        constraint_violation = scenario_safety_margin - scenario_dist
                        
                        if constraint_violation > 0:  # Constraint violated - HARD CONSTRAINT
                            violation_magnitude = constraint_violation / scenario_safety_margin
                            correction_strength = min(0.8, violation_magnitude * scenario_weight * 2.0)  # Stronger correction
                            
                            # Calculate avoidance direction
                            if scenario_dist > 1e-6:
                                grad_x = (x - scenario_x) / scenario_dist
                                grad_y = (y - scenario_y) / scenario_dist
                                avoid_direction = np.arctan2(grad_y, grad_x)
                                steering_angle += correction_strength * np.sin(avoid_direction - psi)
                            
                            # Strong speed reduction for constraint violation
                            acceleration = -0.3
                            
                            # If violation is severe, treat as potential collision
                            if violation_magnitude > 0.5:
                                return {
                                    'success': False,
                                    'message': f'CRASH: Severe scenario constraint violation ({scenario_name}) - collision imminent'
                                }
            
            # Clip steering angle
            steering_angle = np.clip(steering_angle, -0.4, 0.4)
            
            return {
                'success': True,
                'steering_angle': steering_angle,
                'acceleration': acceleration,
                'constraint_type': 'scenario'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Scenario constraint error: {str(e)}'
            }
    
    def _generate_scenario_prediction(self, obs: Dict, scenario_name: str, scenario_idx: int) -> Dict:
        """Generate scenario prediction for obstacle."""
        obs_x, obs_y = obs['position']
        obs_radius = obs['radius']
        velocity = obs.get('velocity', [0.0, 0.0])
        
        # Prediction time (how far into the future to predict)
        prediction_time = 2.0 + np.random.normal(0, 0.5)  # 2 seconds ± 0.5s
        
        if scenario_name == 'continue':
            # Most likely: continues current trajectory
            pred_x = obs_x + velocity[0] * prediction_time
            pred_y = obs_y + velocity[1] * prediction_time
            pred_radius = obs_radius + 0.1
        elif scenario_name == 'accelerate':
            # Accelerates in current direction
            accel_factor = 1.5
            pred_x = obs_x + velocity[0] * prediction_time * accel_factor
            pred_y = obs_y + velocity[1] * prediction_time * accel_factor
            pred_radius = obs_radius + 0.2
        elif scenario_name == 'turn_left':
            # Turns left
            turn_angle = np.pi/6
            cos_turn = np.cos(turn_angle)
            sin_turn = np.sin(turn_angle)
            vel_x = velocity[0] * cos_turn - velocity[1] * sin_turn
            vel_y = velocity[0] * sin_turn + velocity[1] * cos_turn
            pred_x = obs_x + vel_x * prediction_time
            pred_y = obs_y + vel_y * prediction_time
            pred_radius = obs_radius + 0.15
        elif scenario_name == 'turn_right':
            # Turns right
            turn_angle = -np.pi/6
            cos_turn = np.cos(turn_angle)
            sin_turn = np.sin(turn_angle)
            vel_x = velocity[0] * cos_turn - velocity[1] * sin_turn
            vel_y = velocity[0] * sin_turn + velocity[1] * cos_turn
            pred_x = obs_x + vel_x * prediction_time
            pred_y = obs_y + vel_y * prediction_time
            pred_radius = obs_radius + 0.15
        elif scenario_name == 'decelerate':
            # Decelerates
            decel_factor = 0.5
            pred_x = obs_x + velocity[0] * prediction_time * decel_factor
            pred_y = obs_y + velocity[1] * prediction_time * decel_factor
            pred_radius = obs_radius + 0.05
        else:
            # Default to continue
            pred_x = obs_x + velocity[0] * prediction_time
            pred_y = obs_y + velocity[1] * prediction_time
            pred_radius = obs_radius
        
        return {
            'position': [pred_x, pred_y],
            'radius': pred_radius,
            'scenario': scenario_name,
            'weight': self.scenario_weights[scenario_idx],
            'time': prediction_time
        }
    
    def get_obstacle_predictions(self, obstacle_id: str) -> List[Dict]:
        """Get obstacle predictions for visualization."""
        # This would normally get predictions from memory
        # For now, return empty list
        return []
    
    def draw_constraint_visualization(self, ax, obstacle_pos: np.ndarray, 
                                    vehicle_pos: np.ndarray, frame: int) -> List:
        """Draw scenario constraints (chosen scenario + alternative predictions)."""
        patches = []
        obs_x, obs_y = obstacle_pos[0], obstacle_pos[1]
        veh_x, veh_y = vehicle_pos[0], vehicle_pos[1]
        
        # Get predictions for this obstacle from memory
        predictions = self.get_obstacle_predictions(str(obstacle_pos[0]) + "_" + str(obstacle_pos[1]))
        
        if not predictions:
            return patches
        
        # Draw the chosen scenario as a hard constraint
        chosen_scenario_pred = predictions[0] # Assuming the first one is the "chosen"
        scenario_x, scenario_y = chosen_scenario_pred['position']
        scenario_radius = chosen_scenario_pred['radius']
        
        # Hard constraint boundary (thick, solid)
        chosen_circle = Circle((scenario_x, scenario_y), scenario_radius + 1.0, # Increased for visibility
                            facecolor='red', alpha=0.5, edgecolor='darkred', linewidth=3, linestyle='-')
        ax.add_patch(chosen_circle)
        patches.append(chosen_circle)
        
        # Overlay alternative predictions from parallel solvers as different colors
        colors = ['blue', 'green', 'purple', 'orange'] # Different colors for other scenarios
        
        for i, pred in enumerate(predictions[1:]): # Skip the first (chosen) scenario
            pred_x, pred_y = pred['position']
            pred_radius = pred['radius']
            
            # Draw alternative prediction circle
            alt_circle = Circle((pred_x, pred_y), pred_radius, 
                                facecolor=colors[i % len(colors)], alpha=0.3, 
                                edgecolor=colors[i % len(colors)], linewidth=1, linestyle='--')
            ax.add_patch(alt_circle)
            patches.append(alt_circle)
            
            # Draw a line connecting the obstacle to its prediction
            ax.plot([obs_x, pred_x], [obs_y, pred_y], 
                    color=colors[i % len(colors)], linestyle=':', linewidth=1, alpha=0.5)
            
            # Add probability/weight indicator (optional)
            # ax.text(pred_x, pred_y + pred_radius + 0.5, f"{pred['weight']:.2f}", 
            #         color=colors[i % len(colors)], fontsize=8, ha='center')
        
        return patches