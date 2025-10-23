"""
Goal objective for MPC planning.

This module implements the goal objective that encourages
the vehicle to reach a target goal position.
"""

import numpy as np
import casadi as cs
from typing import Dict, List, Tuple, Optional, Any
from .base_objective import BaseObjective


class GoalObjective(BaseObjective):
    """
    Goal objective for reaching target positions.
    
    This objective encourages the vehicle to reach a target goal
    position with appropriate velocity and orientation.
    """
    
    def __init__(self, goal_position: np.ndarray, goal_velocity: float = 0.0,
                 goal_orientation: Optional[float] = None, distance_weight: float = 1.0,
                 velocity_weight: float = 0.1, orientation_weight: float = 0.1,
                 enabled: bool = True):
        """
        Initialize goal objective.
        
        Args:
            goal_position: Target goal position [x, y]
            goal_velocity: Target velocity at goal
            goal_orientation: Target orientation at goal (optional)
            distance_weight: Weight for distance to goal
            velocity_weight: Weight for velocity at goal
            orientation_weight: Weight for orientation at goal
            enabled: Whether objective is enabled
        """
        super().__init__("goal_objective", enabled)
        
        self.goal_position = np.array(goal_position)
        self.goal_velocity = goal_velocity
        self.goal_orientation = goal_orientation
        
        self.distance_weight = distance_weight
        self.velocity_weight = velocity_weight
        self.orientation_weight = orientation_weight
        
        # Objective parameters
        self.parameters = {
            'distance_weight': distance_weight,
            'velocity_weight': velocity_weight,
            'orientation_weight': orientation_weight,
            'goal_position': self.goal_position.tolist(),
            'goal_velocity': goal_velocity,
            'goal_orientation': goal_orientation
        }
    
    def set_goal_position(self, position: np.ndarray) -> None:
        """
        Set goal position.
        
        Args:
            position: Goal position [x, y]
        """
        self.goal_position = np.array(position)
        self.parameters['goal_position'] = self.goal_position.tolist()
    
    def set_goal_velocity(self, velocity: float) -> None:
        """
        Set goal velocity.
        
        Args:
            velocity: Goal velocity
        """
        self.goal_velocity = velocity
        self.parameters['goal_velocity'] = velocity
    
    def set_goal_orientation(self, orientation: Optional[float]) -> None:
        """
        Set goal orientation.
        
        Args:
            orientation: Goal orientation (radians) or None
        """
        self.goal_orientation = orientation
        self.parameters['goal_orientation'] = orientation
    
    def add_objective(self, x: cs.SX, u: cs.SX, k: int, **kwargs) -> cs.SX:
        """
        Add goal objective for time step k.
        
        Args:
            x: State variables [x, y, psi, v, ...]
            u: Input variables
            k: Time step
            **kwargs: Additional parameters
            
        Returns:
            Objective function expression
        """
        if not self.enabled:
            return 0.0
        
        # Get vehicle state
        vehicle_x = x[0]  # x position
        vehicle_y = x[1]  # y position
        vehicle_psi = x[2]  # heading angle
        vehicle_v = x[3]  # velocity
        
        # Distance to goal
        dx = vehicle_x - self.goal_position[0]
        dy = vehicle_y - self.goal_position[1]
        distance_to_goal = cs.sqrt(dx*dx + dy*dy)
        
        # Distance objective
        distance_objective = self.distance_weight * distance_to_goal**2
        
        # Velocity objective
        velocity_error = vehicle_v - self.goal_velocity
        velocity_objective = self.velocity_weight * velocity_error**2
        
        # Orientation objective (if specified)
        orientation_objective = 0.0
        if self.goal_orientation is not None:
            orientation_error = vehicle_psi - self.goal_orientation
            # Handle angle wrapping
            orientation_error = cs.atan2(cs.sin(orientation_error), cs.cos(orientation_error))
            orientation_objective = self.orientation_weight * orientation_error**2
        
        # Combine objectives
        objective = distance_objective + velocity_objective + orientation_objective
        
        return objective
    
    def update(self, state: np.ndarray, data: Dict[str, Any], 
               module_data: Dict[str, Any]) -> None:
        """
        Update goal objective.
        
        Args:
            state: Current state
            data: Real-time data
            module_data: Module-specific data
        """
        # Update objective data
        self.objective_data.update({
            'distance_weight': self.distance_weight,
            'velocity_weight': self.velocity_weight,
            'orientation_weight': self.orientation_weight,
            'goal_position': self.goal_position.tolist(),
            'goal_velocity': self.goal_velocity,
            'goal_orientation': self.goal_orientation
        })
    
    def visualize(self, data: Dict[str, Any], module_data: Dict[str, Any]) -> None:
        """
        Visualize goal objective.
        
        Args:
            data: Real-time data
            module_data: Module-specific data
        """
        print(f"Goal Objective:")
        print(f"  Goal position: {self.goal_position}")
        print(f"  Goal velocity: {self.goal_velocity}")
        print(f"  Goal orientation: {self.goal_orientation}")
        print(f"  Distance weight: {self.distance_weight}")
        print(f"  Velocity weight: {self.velocity_weight}")
        print(f"  Orientation weight: {self.orientation_weight}")
        print(f"  Enabled: {self.enabled}")
    
    def get_objective_info(self) -> Dict[str, Any]:
        """
        Get objective information.
        
        Returns:
            Objective information dictionary
        """
        return {
            'objective_name': self.objective_name,
            'goal_position': self.goal_position.tolist(),
            'goal_velocity': self.goal_velocity,
            'goal_orientation': self.goal_orientation,
            'distance_weight': self.distance_weight,
            'velocity_weight': self.velocity_weight,
            'orientation_weight': self.orientation_weight,
            'enabled': self.enabled
        }
    
    def set_distance_weight(self, weight: float) -> None:
        """
        Set distance weight.
        
        Args:
            weight: Distance weight
        """
        self.distance_weight = weight
        self.parameters['distance_weight'] = weight
    
    def set_velocity_weight(self, weight: float) -> None:
        """
        Set velocity weight.
        
        Args:
            weight: Velocity weight
        """
        self.velocity_weight = weight
        self.parameters['velocity_weight'] = weight
    
    def set_orientation_weight(self, weight: float) -> None:
        """
        Set orientation weight.
        
        Args:
            weight: Orientation weight
        """
        self.orientation_weight = weight
        self.parameters['orientation_weight'] = weight
    
    def set_weights(self, distance: float = None, velocity: float = None,
                   orientation: float = None) -> None:
        """
        Set multiple weights at once.
        
        Args:
            distance: Distance weight
            velocity: Velocity weight
            orientation: Orientation weight
        """
        if distance is not None:
            self.set_distance_weight(distance)
        if velocity is not None:
            self.set_velocity_weight(velocity)
        if orientation is not None:
            self.set_orientation_weight(orientation)
    
    def get_distance_to_goal(self, position: np.ndarray) -> float:
        """
        Get distance to goal from position.
        
        Args:
            position: Position to check [x, y]
            
        Returns:
            Distance to goal
        """
        dx = position[0] - self.goal_position[0]
        dy = position[1] - self.goal_position[1]
        return np.sqrt(dx*dx + dy*dy)
    
    def is_goal_reached(self, position: np.ndarray, velocity: float = None,
                       orientation: float = None, tolerance: float = 0.5) -> bool:
        """
        Check if goal is reached.
        
        Args:
            position: Current position [x, y]
            velocity: Current velocity (optional)
            orientation: Current orientation (optional)
            tolerance: Position tolerance
            
        Returns:
            True if goal is reached, False otherwise
        """
        # Check position
        distance = self.get_distance_to_goal(position)
        if distance > tolerance:
            return False
        
        # Check velocity (if specified)
        if velocity is not None and self.goal_velocity is not None:
            velocity_error = abs(velocity - self.goal_velocity)
            if velocity_error > 0.5:  # 0.5 m/s tolerance
                return False
        
        # Check orientation (if specified)
        if orientation is not None and self.goal_orientation is not None:
            orientation_error = abs(orientation - self.goal_orientation)
            # Handle angle wrapping
            orientation_error = min(orientation_error, 2*np.pi - orientation_error)
            if orientation_error > 0.1:  # 0.1 rad tolerance
                return False
        
        return True
    
    def get_goal_direction(self, position: np.ndarray) -> np.ndarray:
        """
        Get direction to goal from position.
        
        Args:
            position: Position to check [x, y]
            
        Returns:
            Unit direction vector to goal
        """
        dx = self.goal_position[0] - position[0]
        dy = self.goal_position[1] - position[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance > 1e-6:
            return np.array([dx/distance, dy/distance])
        else:
            return np.array([0.0, 0.0])
    
    def get_goal_angle(self, position: np.ndarray) -> float:
        """
        Get angle to goal from position.
        
        Args:
            position: Position to check [x, y]
            
        Returns:
            Angle to goal (radians)
        """
        direction = self.get_goal_direction(position)
        return np.arctan2(direction[1], direction[0])
