"""
Linearized constraints for MPC planning.

This module implements linearized constraints that approximate
nonlinear obstacle avoidance constraints using linear approximations.
"""

import numpy as np
import casadi as cs
from typing import Dict, List, Optional, Any
from .base_constraint import BaseConstraint


class LinearizedConstraints(BaseConstraint):
    """
    Linearized constraints for obstacle avoidance.
    
    These constraints use linear approximations of nonlinear
    obstacle avoidance constraints for computational efficiency.
    """
    
    def __init__(self, safety_margin: float = 1.0, max_obstacles: int = 10,
                 linearization_point: Optional[np.ndarray] = None, enabled: bool = True):
        """
        Initialize linearized constraints.
        
        Args:
            safety_margin: Safety margin around obstacles
            max_obstacles: Maximum number of obstacles
            linearization_point: Point for linearization (if None, use current state)
            enabled: Whether constraints are enabled
        """
        super().__init__("linearized_constraints", enabled)
        
        self.safety_margin = safety_margin
        self.max_obstacles = max_obstacles
        self.linearization_point = linearization_point
        
        # Obstacle data
        self.obstacles = []
        self.linearization_data = {}
        
        # Constraint parameters
        self.parameters = {
            'safety_margin': safety_margin,
            'max_obstacles': max_obstacles,
            'linearization_point': linearization_point
        }
    
    def add_obstacle(self, center: np.ndarray, radius: float,
                     obstacle_id: Optional[str] = None) -> str:
        """
        Add obstacle to linearized constraints.
        
        Args:
            center: Obstacle center [x, y]
            radius: Obstacle radius
            obstacle_id: Optional obstacle ID
            
        Returns:
            Obstacle ID
        """
        if len(self.obstacles) >= self.max_obstacles:
            print(f"Warning: Maximum number of obstacles ({self.max_obstacles}) reached")
            return None
        
        if obstacle_id is None:
            obstacle_id = f"linear_obstacle_{len(self.obstacles)}"
        
        obstacle = {
            'id': obstacle_id,
            'center': np.array(center),
            'radius': radius,
            'enabled': True
        }
        
        self.obstacles.append(obstacle)
        return obstacle_id
    
    def set_linearization_point(self, point: np.ndarray) -> None:
        """
        Set linearization point.
        
        Args:
            point: Linearization point [x, y]
        """
        self.linearization_point = np.array(point)
        self.parameters['linearization_point'] = self.linearization_point.tolist()
    
    def add_constraints(self, x: cs.SX, u: cs.SX, k: int, **kwargs) -> List[cs.SX]:
        """
        Add linearized constraints for time step k.
        
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
        
        # Get linearization point
        if self.linearization_point is not None:
            lin_x, lin_y = self.linearization_point[0], self.linearization_point[1]
        else:
            # Use current state as linearization point
            lin_x, lin_y = vehicle_x, vehicle_y
        
        # Add linearized constraints for each obstacle
        for obstacle in self.obstacles:
            if not obstacle['enabled']:
                continue
            
            # Get obstacle parameters
            obs_x, obs_y = obstacle['center'][0], obstacle['center'][1]
            radius = obstacle['radius']
            
            # Linearize distance constraint around linearization point
            # Original constraint: sqrt((x - obs_x)^2 + (y - obs_y)^2) >= radius + safety_margin
            # Linearized: d_lin + grad_d_lin * (x - lin_x) + grad_d_lin * (y - lin_y) >= radius + safety_margin
            
            # Distance at linearization point
            dx_lin = lin_x - obs_x
            dy_lin = lin_y - obs_y
            d_lin = cs.sqrt(dx_lin*dx_lin + dy_lin*dy_lin)
            
            # Avoid division by zero
            d_lin_safe = cs.fmax(d_lin, 1e-6)
            
            # Gradient of distance function
            grad_x = dx_lin / d_lin_safe
            grad_y = dy_lin / d_lin_safe
            
            # Linearized constraint
            safe_radius = radius + self.safety_margin
            linearized_distance = d_lin + grad_x * (vehicle_x - lin_x) + grad_y * (vehicle_y - lin_y)
            constraint = linearized_distance - safe_radius
            constraints.append(constraint)
        
        return constraints
    
    def update(self, state: np.ndarray, data: Dict[str, Any], 
               module_data: Dict[str, Any]) -> None:
        """
        Update linearized constraints.
        
        Args:
            state: Current state
            data: Real-time data
            module_data: Module-specific data
        """
        # Update linearization point from current state
        if self.linearization_point is None:
            self.set_linearization_point(state[:2])
        
        # Update obstacle data from real-time data
        if 'obstacles' in data:
            self._update_obstacles_from_data(data['obstacles'])
        
        # Update constraint data
        self.linearization_data.update({
            'safety_margin': self.safety_margin,
            'obstacle_count': len(self.obstacles),
            'enabled_obstacles': sum(1 for obs in self.obstacles if obs['enabled']),
            'linearization_point': self.linearization_point.tolist() if self.linearization_point is not None else None
        })
    
    def _update_obstacles_from_data(self, obstacles_data: List[Dict[str, Any]]) -> None:
        """
        Update obstacles from real-time data.
        
        Args:
            obstacles_data: List of obstacle data dictionaries
        """
        # Clear existing obstacles
        self.obstacles.clear()
        
        # Add obstacles from data
        for obs_data in obstacles_data:
            self.add_obstacle(
                obs_data.get('center', [0.0, 0.0]),
                obs_data.get('radius', 1.0),
                obs_data.get('id')
            )
    
    def visualize(self, data: Dict[str, Any], module_data: Dict[str, Any]) -> None:
        """
        Visualize linearized constraints.
        
        Args:
            data: Real-time data
            module_data: Module-specific data
        """
        print("Linearized Constraints:")
        print(f"  Safety margin: {self.safety_margin}")
        print(f"  Obstacle count: {len(self.obstacles)}")
        print(f"  Enabled obstacles: {sum(1 for obs in self.obstacles if obs['enabled'])}")
        print(f"  Linearization point: {self.linearization_point}")
        print(f"  Enabled: {self.enabled}")
        
        for i, obstacle in enumerate(self.obstacles):
            print(f"  Obstacle {i}: {obstacle['id']} at {obstacle['center']} "
                  f"(radius={obstacle['radius']:.2f})")
    
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
            'linearization_point': self.linearization_point.tolist() if self.linearization_point is not None else None,
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
                'radius': obs['radius'],
                'enabled': obs['enabled']
            }
            for obs in self.obstacles
        ]
    
    def compute_linearization_error(self, position: np.ndarray) -> float:
        """
        Compute linearization error at given position.
        
        Args:
            position: Position to check [x, y]
            
        Returns:
            Maximum linearization error
        """
        if self.linearization_point is None:
            return 0.0
        
        max_error = 0.0
        
        for obstacle in self.obstacles:
            if not obstacle['enabled']:
                continue
            
            # True distance
            dx = position[0] - obstacle['center'][0]
            dy = position[1] - obstacle['center'][1]
            true_distance = np.sqrt(dx*dx + dy*dy)
            
            # Linearized distance
            lin_x, lin_y = self.linearization_point[0], self.linearization_point[1]
            dx_lin = lin_x - obstacle['center'][0]
            dy_lin = lin_y - obstacle['center'][1]
            d_lin = np.sqrt(dx_lin*dx_lin + dy_lin*dy_lin)
            
            if d_lin > 1e-6:
                grad_x = dx_lin / d_lin
                grad_y = dy_lin / d_lin
                linearized_distance = d_lin + grad_x * (position[0] - lin_x) + grad_y * (position[1] - lin_y)
                
                error = abs(true_distance - linearized_distance)
                max_error = max(max_error, error)
        
        return max_error
    
    def update_linearization_point(self, position: np.ndarray) -> None:
        """
        Update linearization point based on current position.
        
        Args:
            position: Current position [x, y]
        """
        self.set_linearization_point(position)
    
    def get_linearization_statistics(self) -> Dict[str, Any]:
        """
        Get linearization statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'linearization_point': self.linearization_point.tolist() if self.linearization_point is not None else None,
            'obstacle_count': len(self.obstacles),
            'enabled_obstacles': sum(1 for obs in self.obstacles if obs['enabled']),
            'safety_margin': self.safety_margin
        }
