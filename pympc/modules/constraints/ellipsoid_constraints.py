"""
Ellipsoid constraints for obstacle avoidance in MPC planning.

This module implements ellipsoidal obstacle avoidance constraints
that ensure the vehicle maintains a safe distance from obstacles.
"""

import numpy as np
import casadi as cs
from typing import Dict, List, Tuple, Optional, Any
from .base_constraint import BaseConstraint


class EllipsoidConstraints(BaseConstraint):
    """
    Ellipsoid constraints for obstacle avoidance.
    
    These constraints ensure the vehicle maintains a safe distance
    from obstacles represented as ellipsoids.
    """
    
    def __init__(self, safety_margin: float = 1.0, max_obstacles: int = 10,
                 enabled: bool = True):
        """
        Initialize ellipsoid constraints.
        
        Args:
            safety_margin: Safety margin around obstacles
            max_obstacles: Maximum number of obstacles
            enabled: Whether constraints are enabled
        """
        super().__init__("ellipsoid_constraints", enabled)
        
        self.safety_margin = safety_margin
        self.max_obstacles = max_obstacles
        
        # Obstacle data
        self.obstacles = []
        self.obstacle_data = {}
        
        # Constraint parameters
        self.parameters = {
            'safety_margin': safety_margin,
            'max_obstacles': max_obstacles
        }
    
    def add_obstacle(self, center: np.ndarray, semi_major: float, semi_minor: float,
                     angle: float = 0.0, obstacle_id: Optional[str] = None) -> str:
        """
        Add ellipsoidal obstacle.
        
        Args:
            center: Obstacle center [x, y]
            semi_major: Semi-major axis length
            semi_minor: Semi-minor axis length
            angle: Rotation angle (radians)
            obstacle_id: Optional obstacle ID
            
        Returns:
            Obstacle ID
        """
        if len(self.obstacles) >= self.max_obstacles:
            print(f"Warning: Maximum number of obstacles ({self.max_obstacles}) reached")
            return None
        
        if obstacle_id is None:
            obstacle_id = f"obstacle_{len(self.obstacles)}"
        
        obstacle = {
            'id': obstacle_id,
            'center': np.array(center),
            'semi_major': semi_major,
            'semi_minor': semi_minor,
            'angle': angle,
            'enabled': True
        }
        
        self.obstacles.append(obstacle)
        return obstacle_id
    
    def add_circular_obstacle(self, center: np.ndarray, radius: float,
                             obstacle_id: Optional[str] = None) -> str:
        """
        Add circular obstacle.
        
        Args:
            center: Obstacle center [x, y]
            radius: Obstacle radius
            obstacle_id: Optional obstacle ID
            
        Returns:
            Obstacle ID
        """
        return self.add_obstacle(center, radius, radius, 0.0, obstacle_id)
    
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
        Add ellipsoid constraints for time step k.
        
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
            center = obstacle['center']
            semi_major = obstacle['semi_major']
            semi_minor = obstacle['semi_minor']
            angle = obstacle['angle']
            
            # Compute relative position
            dx = vehicle_x - center[0]
            dy = vehicle_y - center[1]
            
            # Rotate to obstacle frame
            cos_angle = cs.cos(angle)
            sin_angle = cs.sin(angle)
            
            dx_rot = dx * cos_angle + dy * sin_angle
            dy_rot = -dx * sin_angle + dy * cos_angle
            
            # Ellipsoid constraint: (dx_rot/a)^2 + (dy_rot/b)^2 >= 1
            # where a = semi_major + safety_margin, b = semi_minor + safety_margin
            a = semi_major + self.safety_margin
            b = semi_minor + self.safety_margin
            
            # Constraint: (dx_rot/a)^2 + (dy_rot/b)^2 - 1 >= 0
            constraint = (dx_rot/a)**2 + (dy_rot/b)**2 - 1.0
            constraints.append(constraint)
        
        return constraints
    
    def update(self, state: np.ndarray, data: Dict[str, Any], 
               module_data: Dict[str, Any]) -> None:
        """
        Update ellipsoid constraints.
        
        Args:
            state: Current state
            data: Real-time data
            module_data: Module-specific data
        """
        # Update obstacle data from real-time data
        if 'obstacles' in data:
            self._update_obstacles_from_data(data['obstacles'])
        
        # Update constraint data
        self.constraint_data.update({
            'safety_margin': self.safety_margin,
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
            if 'center' in obs_data and 'radius' in obs_data:
                # Circular obstacle
                self.add_circular_obstacle(
                    obs_data['center'],
                    obs_data['radius'],
                    obs_data.get('id')
                )
            elif 'center' in obs_data and 'semi_major' in obs_data and 'semi_minor' in obs_data:
                # Ellipsoidal obstacle
                self.add_obstacle(
                    obs_data['center'],
                    obs_data['semi_major'],
                    obs_data['semi_minor'],
                    obs_data.get('angle', 0.0),
                    obs_data.get('id')
                )
    
    def visualize(self, data: Dict[str, Any], module_data: Dict[str, Any]) -> None:
        """
        Visualize ellipsoid constraints.
        
        Args:
            data: Real-time data
            module_data: Module-specific data
        """
        print(f"Ellipsoid Constraints:")
        print(f"  Safety margin: {self.safety_margin}")
        print(f"  Obstacle count: {len(self.obstacles)}")
        print(f"  Enabled obstacles: {sum(1 for obs in self.obstacles if obs['enabled'])}")
        print(f"  Enabled: {self.enabled}")
        
        for i, obstacle in enumerate(self.obstacles):
            print(f"  Obstacle {i}: {obstacle['id']} at {obstacle['center']} "
                  f"(a={obstacle['semi_major']:.2f}, b={obstacle['semi_minor']:.2f}, "
                  f"angle={obstacle['angle']:.2f})")
    
    def get_obstacle_info(self) -> List[Dict[str, Any]]:
        """
        Get obstacle information.
        
        Returns:
            List of obstacle information dictionaries
        """
        return [
            {
                'id': obs['id'],
                'center': obs['center'].tolist(),
                'semi_major': obs['semi_major'],
                'semi_minor': obs['semi_minor'],
                'angle': obs['angle'],
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
    
    def check_collision(self, position: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Check if position collides with any obstacle.
        
        Args:
            position: Position to check [x, y]
            
        Returns:
            (collision_detected, colliding_obstacle_ids)
        """
        colliding_obstacles = []
        
        for obstacle in self.obstacles:
            if not obstacle['enabled']:
                continue
            
            # Get obstacle parameters
            center = obstacle['center']
            semi_major = obstacle['semi_major']
            semi_minor = obstacle['semi_minor']
            angle = obstacle['angle']
            
            # Compute relative position
            dx = position[0] - center[0]
            dy = position[1] - center[1]
            
            # Rotate to obstacle frame
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            
            dx_rot = dx * cos_angle + dy * sin_angle
            dy_rot = -dx * sin_angle + dy * cos_angle
            
            # Check if inside ellipsoid
            a = semi_major + self.safety_margin
            b = semi_minor + self.safety_margin
            
            if (dx_rot/a)**2 + (dy_rot/b)**2 <= 1.0:
                colliding_obstacles.append(obstacle['id'])
        
        return len(colliding_obstacles) > 0, colliding_obstacles
    
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
            center = obstacle['center']
            semi_major = obstacle['semi_major']
            semi_minor = obstacle['semi_minor']
            angle = obstacle['angle']
            
            # Compute relative position
            dx = position[0] - center[0]
            dy = position[1] - center[1]
            
            # Rotate to obstacle frame
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            
            dx_rot = dx * cos_angle + dy * sin_angle
            dy_rot = -dx * sin_angle + dy * cos_angle
            
            # Distance to ellipsoid boundary
            a = semi_major + self.safety_margin
            b = semi_minor + self.safety_margin
            
            # Approximate distance to ellipsoid
            distance = np.sqrt((dx_rot/a)**2 + (dy_rot/b)**2) * min(a, b)
            min_distance = min(min_distance, distance)
        
        return min_distance
