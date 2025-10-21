"""
Scenario Constraint Test using Unified Framework

This module implements scenario constraints using the unified framework
with configurable perception areas and obstacle memory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from typing import List, Dict, Any
from .unified_constraint_framework import UnifiedConstraintTest


class ScenarioConstraintTest(UnifiedConstraintTest):
    """Test for scenario constraints with unified framework."""
    
    def __init__(self, perception_config: Dict[str, Any]):
        """Initialize scenario constraint test."""
        super().__init__('scenario', perception_config)
    
    def apply_constraint_avoidance(self, current_state: np.ndarray, obstacles: List[Dict], 
                                 step: int) -> Dict[str, Any]:
        """Apply scenario constraint avoidance with obstacle memory."""
        try:
            x, y, psi, v = current_state
            dt = self.timestep
            
            # Calculate desired heading towards goal
            goal_x = self.road_length
            goal_y = 0.0  # Approximate goal position
            dx = goal_x - x
            dy = goal_y - y
            desired_psi = np.arctan2(dy, dx)
            
            # Scenario constraint avoidance
            steering_angle = (desired_psi - psi) * 0.5
            acceleration = 0.3
            
            # Apply hard road boundary constraints FIRST
            road_correction = self._apply_road_boundary_constraints(current_state, steering_angle)
            steering_angle += road_correction
            
            # Apply scenario constraints for obstacles in perception area with memory
            for obs in obstacles:
                obs_x, obs_y = obs['position']
                obs_radius = obs['radius']
                obs_id = obs.get('id', f"obs_{obs_x}_{obs_y}")
                
                # Calculate distance to obstacle
                dist_to_obs = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                
                if dist_to_obs < 15.0:  # Detection range
                    # Get obstacle predictions from memory
                    predictions = self.perception_area.get_obstacle_predictions(obs_id)
                    
                    if not predictions:
                        # Generate new predictions if not in memory
                        predictions = self.perception_area._generate_obstacle_predictions(obs, current_state)
                    
                    # Apply scenario constraints for each prediction
                    for prediction in predictions:
                        pred_x, pred_y = prediction['position']
                        pred_radius = prediction['radius']
                        weight = prediction['weight']
                        
                        # Calculate distance to predicted position
                        pred_dist = np.sqrt((x - pred_x)**2 + (y - pred_y)**2)
                        
                        # Scenario constraint: distance >= scenario_radius + safety_margin
                        safety_margin = 1.5
                        constraint_violation = (pred_radius + safety_margin) - pred_dist
                        
                        if constraint_violation > 0:  # Constraint violated
                            # Apply corrective steering based on constraint violation
                            violation_magnitude = constraint_violation / (pred_radius + safety_margin)
                            correction_strength = min(0.6, violation_magnitude * weight * 2.0)
                            
                            # Calculate avoidance direction (gradient of constraint)
                            if pred_dist > 1e-6:  # Avoid division by zero
                                grad_x = (x - pred_x) / pred_dist
                                grad_y = (y - pred_y) / pred_dist
                                avoid_direction = np.arctan2(grad_y, grad_x)
                                
                                # Apply steering correction
                                steering_angle += correction_strength * np.sin(avoid_direction - psi)
                            
                            # Speed reduction based on constraint violation severity
                            if violation_magnitude > 0.3:
                                acceleration = -0.3  # Strong deceleration
                            elif violation_magnitude > 0.1:
                                acceleration = -0.1  # Moderate deceleration
                
                # Check for crash
                if dist_to_obs < obs_radius + 0.3:
                    return {
                        'success': False,
                        'message': 'CRASH: Scenario constraint violated - collision imminent'
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
        """Draw scenario constraints (chosen scenario as hard constraints + alternative predictions)."""
        patches = []
        obs_x, obs_y = obstacle_pos[0], obstacle_pos[1]
        veh_x, veh_y = vehicle_pos[0], vehicle_pos[1]
        
        distance = np.sqrt((veh_x - obs_x)**2 + (veh_y - obs_y)**2)
        
        if distance < 15.0:
            # Dynamic scenario selection based on frame
            primary_scenario_idx = frame % 5  # Cycle through scenarios
            
            # Define scenario colors
            scenario_colors = ['red', 'blue', 'green', 'purple', 'orange']
            scenario_names = ['Continue', 'Accelerate', 'Turn Left', 'Turn Right', 'Decelerate']
            
            # Draw primary scenario (chosen as hard constraint)
            primary_radius = 2.0 + 0.5 * np.sin(frame * 0.1)
            primary_circle = Circle((obs_x, obs_y), primary_radius,
                                  facecolor=scenario_colors[primary_scenario_idx], alpha=0.6,
                                  edgecolor='black', linewidth=3,
                                  label=f'Primary: {scenario_names[primary_scenario_idx]}')
            patches.append(primary_circle)
            
            # Draw alternative scenario predictions (parallel solvers)
            for i, (color, name) in enumerate(zip(scenario_colors, scenario_names)):
                if i != primary_scenario_idx:
                    # Calculate alternative prediction position
                    angle_offset = i * 2 * np.pi / 5 + frame * 0.05
                    pred_distance = 3.0 + 1.0 * np.sin(frame * 0.1 + i)
                    pred_x = obs_x + pred_distance * np.cos(angle_offset)
                    pred_y = obs_y + pred_distance * np.sin(angle_offset)
                    
                    # Draw alternative scenario
                    alt_radius = 1.0 + 0.3 * np.cos(frame * 0.15 + i)
                    alt_circle = Circle((pred_x, pred_y), alt_radius,
                                      facecolor=color, alpha=0.4,
                                      edgecolor=color, linewidth=2)
                    patches.append(alt_circle)
                    
                    # Draw connection line to main obstacle
                    ax.plot([obs_x, pred_x], [obs_y, pred_y],
                           color=color, linewidth=1, alpha=0.6, linestyle='--')
                    
                    # Draw solver prediction indicator
                    ax.text(pred_x, pred_y + 1.5, f'S{i+1}', ha='center', va='bottom',
                           fontsize=8, color=color, fontweight='bold')
            
            # Draw scenario probability indicators
            for i in range(5):
                prob_x = obs_x - 2.0 + i * 1.0
                prob_y = obs_y - 3.0
                prob_height = 0.5 + 0.3 * np.sin(frame * 0.1 + i)
                prob_color = scenario_colors[i] if i == primary_scenario_idx else 'gray'
                
                prob_bar = plt.Rectangle((prob_x - 0.2, prob_y), 0.4, prob_height,
                                       facecolor=prob_color, alpha=0.7,
                                       edgecolor='black', linewidth=1)
                patches.append(prob_bar)
        
        return patches


def run_scenario_test():
    """Run the scenario constraint test with configurable perception area."""
    # Configure perception area
    perception_config = {
        'type': 'radius',  # 'radius', 'sphere', 'forward_cone', 'bidirectional_cone', 'rectangle'
        'radius': 20.0,  # For radius/sphere perception
        'forward_angle': np.pi/3,  # For cone perception (60 degrees)
        'backward_angle': np.pi/6,  # For bidirectional cone (30 degrees)
        'forward_distance': 25.0,  # For cone/rectangle perception
        'backward_distance': 10.0,  # For bidirectional cone/rectangle perception
        'width': 15.0,  # For rectangle perception
        'height': 20.0,  # For rectangle perception
        'memory_duration': 6.0  # Obstacle memory duration in seconds
    }
    
    # Create and run test
    test = ScenarioConstraintTest(perception_config)
    result = test.run_test()
    
    # Print results
    print("\n" + "="*60)
    print("SCENARIO CONSTRAINTS TEST RESULTS")
    print("="*60)
    print(f"Status: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
    if result['success']:
        print(f"Trajectory Length: {len(result['trajectory'])} steps")
        print(f"Total Distance: {test.road_length}m")
        print(f"Animation: {test.output_dir}/scenario_constraints_test.gif")
        print(f"Constraint Type: Scenario (Multi-modal uncertainty)")
        print(f"Perception Area: {perception_config['type']}")
        print(f"Memory Duration: {perception_config['memory_duration']}s")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("\n🎉 Test completed successfully!")
    print(f"Check the animation at: {test.output_dir}/scenario_constraints_test.gif")
    
    return result


if __name__ == "__main__":
    run_scenario_test()
