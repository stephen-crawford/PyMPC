"""
Gaussian constraints for obstacle avoidance in MPC planning.

This module implements Gaussian obstacle avoidance constraints
that handle uncertain obstacle positions and shapes.
"""

import numpy as np
import casadi as cs
from typing import Dict, List, Tuple, Optional, Any
from .base_constraint import BaseConstraint


class GaussianConstraints(BaseConstraint):
    """
    Gaussian constraints for uncertain obstacle avoidance.
    
    These constraints handle obstacles with uncertain positions and shapes
    using Gaussian distributions and chance constraints.
    """
    
    def __init__(self, safety_margin: float = 1.0, max_obstacles: int = 10,
                 confidence_level: float = 0.95, enabled: bool = True):
        """
        Initialize Gaussian constraints.
        
        Args:
            safety_margin: Safety margin around obstacles
            max_obstacles: Maximum number of obstacles
            confidence_level: Confidence level for chance constraints
            enabled: Whether constraints are enabled
        """
        super().__init__("gaussian_constraints", enabled)
        
        self.safety_margin = safety_margin
        self.max_obstacles = max_obstacles
        self.confidence_level = confidence_level
        
        # Obstacle data
        self.obstacles = []
        self.obstacle_data = {}
        
        # Constraint parameters
        self.parameters = {
            'safety_margin': safety_margin,
            'max_obstacles': max_obstacles,
            'confidence_level': confidence_level
        }
    
    def add_obstacle(self, mean_center: np.ndarray, covariance: np.ndarray,
                     mean_radius: float, radius_variance: float = 0.0,
                     obstacle_id: Optional[str] = None) -> str:
        """
        Add Gaussian obstacle.
        
        Args:
            mean_center: Mean obstacle center [x, y]
            covariance: Position covariance matrix (2x2)
            mean_radius: Mean obstacle radius
            radius_variance: Radius variance
            obstacle_id: Optional obstacle ID
            
        Returns:
            Obstacle ID
        """
        if len(self.obstacles) >= self.max_obstacles:
            print(f"Warning: Maximum number of obstacles ({self.max_obstacles}) reached")
            return None
        
        if obstacle_id is None:
            obstacle_id = f"gaussian_obstacle_{len(self.obstacles)}"
        
        obstacle = {
            'id': obstacle_id,
            'mean_center': np.array(mean_center),
            'covariance': np.array(covariance),
            'mean_radius': mean_radius,
            'radius_variance': radius_variance,
            'enabled': True
        }
        
        self.obstacles.append(obstacle)
        return obstacle_id
    
    def add_circular_obstacle(self, mean_center: np.ndarray, mean_radius: float,
                             position_variance: float = 0.0, radius_variance: float = 0.0,
                             obstacle_id: Optional[str] = None) -> str:
        """
        Add circular Gaussian obstacle.
        
        Args:
            mean_center: Mean obstacle center [x, y]
            mean_radius: Mean obstacle radius
            position_variance: Position variance
            radius_variance: Radius variance
            obstacle_id: Optional obstacle ID
            
        Returns:
            Obstacle ID
        """
        # Create isotropic covariance matrix
        covariance = np.eye(2) * position_variance
        return self.add_obstacle(mean_center, covariance, mean_radius, 
                                radius_variance, obstacle_id)
    
    def remove_obstacle(self, obstacle_id: str) -> bool:
        """
        Remove obstacle by ID.
        
        Args:
            obstacle_id: Obstacle ID
            
        Returns:
            True if removed, False otherwise
        """
        for i, obstacle in enumerate(self.obstacles):
            if obstacle['id'] == obstacle_id:
                del self.obstacles[i]
                return True
        return False
    
    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        self.obstacles.clear()
    
    def set_obstacle_enabled(self, obstacle_id: str, enabled: bool) -> bool:
        """
        Set obstacle enabled state.
        
        Args:
            obstacle_id: Obstacle ID
            enabled: Whether obstacle is enabled
            
        Returns:
            True if found, False otherwise
        """
        for obstacle in self.obstacles:
            if obstacle['id'] == obstacle_id:
                obstacle['enabled'] = enabled
                return True
        return False
    
    def add_constraints(self, x: cs.SX, u: cs.SX, k: int, **kwargs) -> List[cs.SX]:
        """
        Add Gaussian constraints for time step k.
        
        Args:
            x: State variables [x, y, psi, v, ...]
            u: Input variables
            k: Time step
            **kwargs: Additional parameters
            
        Returns:
            List of constraint expressions
        """
        if not self.enabled or not self.obstacles:
            return []
        
        constraints = []
        
        # Get vehicle position
        vehicle_x = x[0]  # x position
        vehicle_y = x[1]  # y position
        
        # Add constraint for each obstacle
        for obstacle in self.obstacles:
            if not obstacle['enabled']:
                continue
            
            # Get obstacle parameters
            mean_center = obstacle['mean_center']
            covariance = obstacle['covariance']
            mean_radius = obstacle['mean_radius']
            radius_variance = obstacle['radius_variance']
            
            # Compute relative position
            dx = vehicle_x - mean_center[0]
            dy = vehicle_y - mean_center[1]
            
            # Chance constraint: P(distance <= safe_radius) <= 1 - confidence_level
            # This is approximated as: distance >= safe_radius + safety_margin
            
            # Compute safe radius (mean + safety margin + uncertainty)
            safe_radius = mean_radius + self.safety_margin
            
            # Add uncertainty term based on confidence level
            # For Gaussian distribution, use quantile function
            from scipy.stats import norm
            quantile = norm.ppf(self.confidence_level)
            
            # Position uncertainty
            position_uncertainty = quantile * cs.sqrt(covariance[0, 0] + covariance[1, 1])
            
            # Radius uncertainty
            radius_uncertainty = quantile * cs.sqrt(radius_variance)
            
            # Total safe distance
            total_safe_distance = safe_radius + position_uncertainty + radius_uncertainty
            
            # Distance constraint: distance >= total_safe_distance
            distance = cs.sqrt(dx*dx + dy*dy)
            constraint = distance - total_safe_distance
            constraints.append(constraint)
        
        return constraints
    
    def update(self, state: np.ndarray, data: Dict[str, Any], 
               module_data: Dict[str, Any]) -> None:
        """
        Update Gaussian constraints.
        
        Args:
            state: Current state
            data: Real-time data
            module_data: Module-specific data
        """
        # Update obstacle data from real-time data
        if 'gaussian_obstacles' in data:
            self._update_obstacles_from_data(data['gaussian_obstacles'])
        
        # Update constraint data
        self.constraint_data.update({
            'safety_margin': self.safety_margin,
            'confidence_level': self.confidence_level,
            'obstacle_count': len(self.obstacles),
            'enabled_obstacles': sum(1 for obs in self.obstacles if obs['enabled'])
        })
    
    def _update_obstacles_from_data(self, obstacles_data: List[Dict[str, Any]]) -> None:
        """
        Update obstacles from real-time data.
        
        Args:
            obstacles_data: List of obstacle data dictionaries
        """
        # Clear existing obstacles
        self.clear_obstacles()
        
        # Add obstacles from data
        for obs_data in obstacles_data:
            if 'mean_center' in obs_data and 'covariance' in obs_data:
                self.add_obstacle(
                    obs_data['mean_center'],
                    obs_data['covariance'],
                    obs_data.get('mean_radius', 1.0),
                    obs_data.get('radius_variance', 0.0),
                    obs_data.get('id')
                )
            elif 'mean_center' in obs_data and 'mean_radius' in obs_data:
                # Circular obstacle with position variance
                position_variance = obs_data.get('position_variance', 0.0)
                radius_variance = obs_data.get('radius_variance', 0.0)
                self.add_circular_obstacle(
                    obs_data['mean_center'],
                    obs_data['mean_radius'],
                    position_variance,
                    radius_variance,
                    obs_data.get('id')
                )
    
    def visualize(self, data: Dict[str, Any], module_data: Dict[str, Any]) -> None:
        """
        Visualize Gaussian constraints.
        
        Args:
            data: Real-time data
            module_data: Module-specific data
        """
        print(f"Gaussian Constraints:")
        print(f"  Safety margin: {self.safety_margin}")
        print(f"  Confidence level: {self.confidence_level}")
        print(f"  Obstacle count: {len(self.obstacles)}")
        print(f"  Enabled obstacles: {sum(1 for obs in self.obstacles if obs['enabled'])}")
        print(f"  Enabled: {self.enabled}")
        
        for i, obstacle in enumerate(self.obstacles):
            print(f"  Obstacle {i}: {obstacle['id']} at {obstacle['mean_center']} "
                  f"(radius={obstacle['mean_radius']:.2f}, "
                  f"pos_var={np.trace(obstacle['covariance']):.2f}, "
                  f"radius_var={obstacle['radius_variance']:.2f})")
    
    def get_obstacle_info(self) -> List[Dict[str, Any]]:
        """
        Get obstacle information.
        
        Returns:
            List of obstacle information dictionaries
        """
        return [
            {
                'id': obs['id'],
                'mean_center': obs['mean_center'].tolist(),
                'covariance': obs['covariance'].tolist(),
                'mean_radius': obs['mean_radius'],
                'radius_variance': obs['radius_variance'],
                'enabled': obs['enabled']
            }
            for obs in self.obstacles
        ]
    
    def get_constraint_info(self) -> Dict[str, Any]:
        """
        Get constraint information.
        
        Returns:
            Constraint information dictionary
        """
        return {
            'constraint_name': self.constraint_name,
            'safety_margin': self.safety_margin,
            'confidence_level': self.confidence_level,
            'max_obstacles': self.max_obstacles,
            'obstacle_count': len(self.obstacles),
            'enabled_obstacles': sum(1 for obs in self.obstacles if obs['enabled']),
            'enabled': self.enabled
        }
    
    def set_safety_margin(self, margin: float) -> None:
        """
        Set safety margin.
        
        Args:
            margin: Safety margin
        """
        self.safety_margin = margin
        self.parameters['safety_margin'] = margin
    
    def set_confidence_level(self, level: float) -> None:
        """
        Set confidence level.
        
        Args:
            level: Confidence level (0.0 to 1.0)
        """
        self.confidence_level = max(0.0, min(1.0, level))
        self.parameters['confidence_level'] = self.confidence_level
    
    def set_max_obstacles(self, max_obs: int) -> None:
        """
        Set maximum number of obstacles.
        
        Args:
            max_obs: Maximum number of obstacles
        """
        self.max_obstacles = max_obs
        self.parameters['max_obstacles'] = max_obs
        
        # Remove excess obstacles if necessary
        while len(self.obstacles) > max_obs:
            self.obstacles.pop()
    
    def check_collision_probability(self, position: np.ndarray) -> Tuple[float, List[str]]:
        """
        Check collision probability with obstacles.
        
        Args:
            position: Position to check [x, y]
            
        Returns:
            (max_collision_probability, colliding_obstacle_ids)
        """
        max_probability = 0.0
        colliding_obstacles = []
        
        for obstacle in self.obstacles:
            if not obstacle['enabled']:
                continue
            
            # Get obstacle parameters
            mean_center = obstacle['mean_center']
            covariance = obstacle['covariance']
            mean_radius = obstacle['mean_radius']
            radius_variance = obstacle['radius_variance']
            
            # Compute relative position
            dx = position[0] - mean_center[0]
            dy = position[1] - mean_center[1]
            
            # Distance to obstacle center
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Collision probability (simplified)
            # This is a rough approximation - in practice, you'd use proper
            # multivariate Gaussian probability calculations
            position_uncertainty = np.sqrt(np.trace(covariance))
            radius_uncertainty = np.sqrt(radius_variance)
            
            # Approximate collision probability
            safe_distance = mean_radius + self.safety_margin
            if distance < safe_distance:
                # High collision probability
                probability = 1.0 - (distance / safe_distance)
                max_probability = max(max_probability, probability)
                colliding_obstacles.append(obstacle['id'])
        
        return max_probability, colliding_obstacles
    
    def get_safe_distance(self, position: np.ndarray) -> float:
        """
        Get minimum safe distance to any obstacle.
        
        Args:
            position: Position to check [x, y]
            
        Returns:
            Minimum safe distance
        """
        min_distance = float('inf')
        
        for obstacle in self.obstacles:
            if not obstacle['enabled']:
                continue
            
            # Get obstacle parameters
            mean_center = obstacle['mean_center']
            covariance = obstacle['covariance']
            mean_radius = obstacle['mean_radius']
            radius_variance = obstacle['radius_variance']
            
            # Compute relative position
            dx = position[0] - mean_center[0]
            dy = position[1] - mean_center[1]
            
            # Distance to obstacle center
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Safe distance including uncertainty
            position_uncertainty = np.sqrt(np.trace(covariance))
            radius_uncertainty = np.sqrt(radius_variance)
            
            safe_distance = mean_radius + self.safety_margin + position_uncertainty + radius_uncertainty
            
            min_distance = min(min_distance, distance - safe_distance)
        
        return min_distance
