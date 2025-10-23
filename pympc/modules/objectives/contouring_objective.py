"""
Contouring objective for MPC planning.

This module implements the contouring objective that encourages
the vehicle to follow a reference path while maximizing progress.
"""

import numpy as np
import casadi as cs
from typing import Dict, List, Tuple, Optional, Any
from .base_objective import BaseObjective


class ContouringObjective(BaseObjective):
    """
    Contouring objective for path following and progress maximization.
    
    This objective combines contouring error (lateral deviation) and
    lag error (longitudinal deviation) with progress maximization.
    """
    
    def __init__(self, contouring_weight: float = 2.0, lag_weight: float = 1.0,
                 progress_weight: float = 1.5, velocity_weight: float = 0.1,
                 enabled: bool = True):
        """
        Initialize contouring objective.
        
        Args:
            contouring_weight: Weight for contouring error
            lag_weight: Weight for lag error
            progress_weight: Weight for progress maximization
            velocity_weight: Weight for velocity tracking
            enabled: Whether objective is enabled
        """
        super().__init__("contouring_objective", enabled)
        
        self.contouring_weight = contouring_weight
        self.lag_weight = lag_weight
        self.progress_weight = progress_weight
        self.velocity_weight = velocity_weight
        
        # Reference path data
        self.reference_path = None
        self.path_length = 0.0
        self.path_points = None
        self.path_tangents = None
        self.path_curvatures = None
        self.path_velocities = None
        
        # Objective parameters
        self.parameters = {
            'contouring_weight': contouring_weight,
            'lag_weight': lag_weight,
            'progress_weight': progress_weight,
            'velocity_weight': velocity_weight
        }
    
    def set_reference_path(self, path: np.ndarray, velocities: Optional[np.ndarray] = None) -> None:
        """
        Set reference path.
        
        Args:
            path: Reference path as Nx2 array (x, y coordinates)
            velocities: Reference velocities (optional)
        """
        self.reference_path = path.copy()
        self.path_length = self._compute_path_length(path)
        self.path_points = path
        self.path_tangents = self._compute_tangents(path)
        self.path_curvatures = self._compute_curvatures(path)
        
        if velocities is not None:
            self.path_velocities = velocities.copy()
        else:
            # Default velocity profile
            self.path_velocities = np.ones(len(path)) * 5.0  # 5 m/s default
    
    def _compute_path_length(self, path: np.ndarray) -> float:
        """Compute total path length."""
        if len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(1, len(path)):
            dx = path[i, 0] - path[i-1, 0]
            dy = path[i, 1] - path[i-1, 1]
            length += np.sqrt(dx*dx + dy*dy)
        
        return length
    
    def _compute_tangents(self, path: np.ndarray) -> np.ndarray:
        """Compute path tangents."""
        if len(path) < 2:
            return np.array([[0.0, 1.0]])
        
        tangents = np.zeros_like(path)
        
        # First point
        dx = path[1, 0] - path[0, 0]
        dy = path[1, 1] - path[0, 1]
        norm = np.sqrt(dx*dx + dy*dy)
        if norm > 1e-6:
            tangents[0, 0] = dx / norm
            tangents[0, 1] = dy / norm
        else:
            tangents[0, 0] = 1.0
            tangents[0, 1] = 0.0
        
        # Middle points
        for i in range(1, len(path)-1):
            dx = path[i+1, 0] - path[i-1, 0]
            dy = path[i+1, 1] - path[i-1, 1]
            norm = np.sqrt(dx*dx + dy*dy)
            if norm > 1e-6:
                tangents[i, 0] = dx / norm
                tangents[i, 1] = dy / norm
            else:
                tangents[i, 0] = tangents[i-1, 0]
                tangents[i, 1] = tangents[i-1, 1]
        
        # Last point
        dx = path[-1, 0] - path[-2, 0]
        dy = path[-1, 1] - path[-2, 1]
        norm = np.sqrt(dx*dx + dy*dy)
        if norm > 1e-6:
            tangents[-1, 0] = dx / norm
            tangents[-1, 1] = dy / norm
        else:
            tangents[-1, 0] = tangents[-2, 0]
            tangents[-1, 1] = tangents[-2, 1]
        
        return tangents
    
    def _compute_curvatures(self, path: np.ndarray) -> np.ndarray:
        """Compute path curvatures."""
        if len(path) < 3:
            return np.zeros(len(path))
        
        curvatures = np.zeros(len(path))
        
        for i in range(1, len(path)-1):
            # Three consecutive points
            p1 = path[i-1]
            p2 = path[i]
            p3 = path[i+1]
            
            # Vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Cross product for curvature
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            norm1 = np.sqrt(v1[0]*v1[0] + v1[1]*v1[1])
            norm2 = np.sqrt(v2[0]*v2[0] + v2[1]*v2[1])
            
            if norm1 > 1e-6 and norm2 > 1e-6:
                curvatures[i] = cross / (norm1 * norm2)
            else:
                curvatures[i] = 0.0
        
        # Set boundary curvatures
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
        
        return curvatures
    
    def _get_path_point_at_s(self, s: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Get path point, tangent, curvature, and velocity at path parameter s.
        
        Args:
            s: Path parameter (0 to path_length)
            
        Returns:
            (point, tangent, curvature, velocity)
        """
        if self.reference_path is None or len(self.reference_path) < 2:
            return np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.0, 5.0
        
        # Clamp s to valid range
        s = max(0.0, min(s, self.path_length))
        
        # Find segment containing s
        cumulative_length = 0.0
        for i in range(1, len(self.reference_path)):
            dx = self.reference_path[i, 0] - self.reference_path[i-1, 0]
            dy = self.reference_path[i, 1] - self.reference_path[i-1, 1]
            segment_length = np.sqrt(dx*dx + dy*dy)
            
            if cumulative_length + segment_length >= s:
                # Interpolate within segment
                alpha = (s - cumulative_length) / segment_length
                point = (1 - alpha) * self.reference_path[i-1] + alpha * self.reference_path[i]
                tangent = self.path_tangents[i-1]
                curvature = self.path_curvatures[i-1]
                velocity = self.path_velocities[i-1]
                return point, tangent, curvature, velocity
            
            cumulative_length += segment_length
        
        # Return last point
        return self.reference_path[-1], self.path_tangents[-1], self.path_curvatures[-1], self.path_velocities[-1]
    
    def add_objective(self, x: cs.SX, u: cs.SX, k: int, **kwargs) -> cs.SX:
        """
        Add contouring objective for time step k.
        
        Args:
            x: State variables [x, y, psi, v, s, ...]
            u: Input variables
            k: Time step
            **kwargs: Additional parameters
            
        Returns:
            Objective function expression
        """
        if not self.enabled or self.reference_path is None:
            return 0.0
        
        # Get vehicle state
        vehicle_x = x[0]  # x position
        vehicle_y = x[1]  # y position
        vehicle_psi = x[2]  # heading angle
        vehicle_v = x[3]  # velocity
        
        # Get path parameter (if available in state)
        if x.shape[0] > 4:  # Assuming spline parameter is at index 4
            s = x[4]  # Path parameter
        else:
            # Estimate path parameter from position
            s = self._estimate_path_parameter(vehicle_x, vehicle_y)
        
        # Get path information at parameter s
        path_point, path_tangent, path_curvature, path_velocity = self._get_path_point_at_s(s)
        
        # Compute contouring error (lateral deviation)
        dx = vehicle_x - path_point[0]
        dy = vehicle_y - path_point[1]
        
        # Lateral distance (signed)
        contouring_error = -dx * path_tangent[1] + dy * path_tangent[0]
        
        # Compute lag error (longitudinal deviation)
        lag_error = dx * path_tangent[0] + dy * path_tangent[1]
        
        # Progress maximization (negative lag error to maximize progress)
        progress_term = -lag_error
        
        # Velocity tracking
        velocity_error = vehicle_v - path_velocity
        
        # Combine objectives
        objective = (
            self.contouring_weight * contouring_error**2 +
            self.lag_weight * lag_error**2 +
            self.progress_weight * progress_term +
            self.velocity_weight * velocity_error**2
        )
        
        return objective
    
    def _estimate_path_parameter(self, x: float, y: float) -> float:
        """
        Estimate path parameter from vehicle position.
        
        Args:
            x: Vehicle x position
            y: Vehicle y position
            
        Returns:
            Estimated path parameter
        """
        if self.reference_path is None or len(self.reference_path) < 2:
            return 0.0
        
        # Find closest point on path
        min_distance = float('inf')
        closest_s = 0.0
        
        cumulative_length = 0.0
        for i in range(1, len(self.reference_path)):
            # Distance to line segment
            p1 = self.reference_path[i-1]
            p2 = self.reference_path[i]
            
            # Vector from p1 to p2
            v = p2 - p1
            # Vector from p1 to vehicle
            w = np.array([x, y]) - p1
            
            # Project w onto v
            if np.dot(v, v) > 1e-6:
                t = max(0.0, min(1.0, np.dot(w, v) / np.dot(v, v)))
                closest_point = p1 + t * v
                distance = np.sqrt((x - closest_point[0])**2 + (y - closest_point[1])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_s = cumulative_length + t * np.sqrt(np.dot(v, v))
            
            cumulative_length += np.sqrt(np.dot(v, v))
        
        return closest_s
    
    def update(self, state: np.ndarray, data: Dict[str, Any], 
               module_data: Dict[str, Any]) -> None:
        """
        Update contouring objective.
        
        Args:
            state: Current state
            data: Real-time data
            module_data: Module-specific data
        """
        # Update objective data
        self.objective_data.update({
            'contouring_weight': self.contouring_weight,
            'lag_weight': self.lag_weight,
            'progress_weight': self.progress_weight,
            'velocity_weight': self.velocity_weight,
            'path_length': self.path_length,
            'reference_path': self.reference_path
        })
    
    def visualize(self, data: Dict[str, Any], module_data: Dict[str, Any]) -> None:
        """
        Visualize contouring objective.
        
        Args:
            data: Real-time data
            module_data: Module-specific data
        """
        print(f"Contouring Objective:")
        print(f"  Contouring weight: {self.contouring_weight}")
        print(f"  Lag weight: {self.lag_weight}")
        print(f"  Progress weight: {self.progress_weight}")
        print(f"  Velocity weight: {self.velocity_weight}")
        print(f"  Path length: {self.path_length}")
        print(f"  Enabled: {self.enabled}")
        
        if self.reference_path is not None:
            print(f"  Reference path points: {len(self.reference_path)}")
    
    def get_objective_info(self) -> Dict[str, Any]:
        """
        Get objective information.
        
        Returns:
            Objective information dictionary
        """
        return {
            'objective_name': self.objective_name,
            'contouring_weight': self.contouring_weight,
            'lag_weight': self.lag_weight,
            'progress_weight': self.progress_weight,
            'velocity_weight': self.velocity_weight,
            'path_length': self.path_length,
            'enabled': self.enabled,
            'reference_path_set': self.reference_path is not None
        }
    
    def set_contouring_weight(self, weight: float) -> None:
        """
        Set contouring weight.
        
        Args:
            weight: Contouring weight
        """
        self.contouring_weight = weight
        self.parameters['contouring_weight'] = weight
    
    def set_lag_weight(self, weight: float) -> None:
        """
        Set lag weight.
        
        Args:
            weight: Lag weight
        """
        self.lag_weight = weight
        self.parameters['lag_weight'] = weight
    
    def set_progress_weight(self, weight: float) -> None:
        """
        Set progress weight.
        
        Args:
            weight: Progress weight
        """
        self.progress_weight = weight
        self.parameters['progress_weight'] = weight
    
    def set_velocity_weight(self, weight: float) -> None:
        """
        Set velocity weight.
        
        Args:
            weight: Velocity weight
        """
        self.velocity_weight = weight
        self.parameters['velocity_weight'] = weight
    
    def set_weights(self, contouring: float = None, lag: float = None,
                   progress: float = None, velocity: float = None) -> None:
        """
        Set multiple weights at once.
        
        Args:
            contouring: Contouring weight
            lag: Lag weight
            progress: Progress weight
            velocity: Velocity weight
        """
        if contouring is not None:
            self.set_contouring_weight(contouring)
        if lag is not None:
            self.set_lag_weight(lag)
        if progress is not None:
            self.set_progress_weight(progress)
        if velocity is not None:
            self.set_velocity_weight(velocity)
