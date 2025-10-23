"""
Contouring constraints for MPC planning.

This module implements contouring constraints that enforce the vehicle
to stay within road boundaries while following a reference path.
"""

import numpy as np
import casadi as cs
from typing import Dict, List, Tuple, Optional, Any
from .base_constraint import BaseConstraint


class ContouringConstraints(BaseConstraint):
    """
    Contouring constraints for road boundary enforcement.
    
    These constraints ensure the vehicle stays within the road boundaries
    while following a reference path. They are essential for MPCC.
    """
    
    def __init__(self, road_width: float = 6.0, safety_margin: float = 0.5,
                 max_curvature: float = 0.5, enabled: bool = True):
        """
        Initialize contouring constraints.
        
        Args:
            road_width: Road width
            safety_margin: Safety margin from road edges
            max_curvature: Maximum road curvature
            enabled: Whether constraints are enabled
        """
        super().__init__("contouring_constraints", enabled)
        
        self.road_width = road_width
        self.safety_margin = safety_margin
        self.max_curvature = max_curvature
        
        # Reference path data
        self.reference_path = None
        self.path_length = 0.0
        self.path_points = None
        self.path_tangents = None
        self.path_curvatures = None
        
        # Constraint parameters
        self.parameters = {
            'road_width': road_width,
            'safety_margin': safety_margin,
            'max_curvature': max_curvature
        }
    
    def set_reference_path(self, path: np.ndarray) -> None:
        """
        Set reference path.
        
        Args:
            path: Reference path as Nx2 array (x, y coordinates)
        """
        self.reference_path = path.copy()
        self.path_length = self._compute_path_length(path)
        self.path_points = path
        self.path_tangents = self._compute_tangents(path)
        self.path_curvatures = self._compute_curvatures(path)
    
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
    
    def _get_path_point_at_s(self, s: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get path point, tangent, and curvature at path parameter s.
        
        Args:
            s: Path parameter (0 to path_length)
            
        Returns:
            (point, tangent, curvature)
        """
        if self.reference_path is None or len(self.reference_path) < 2:
            return np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.0
        
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
                return point, tangent, curvature
            
            cumulative_length += segment_length
        
        # Return last point
        return self.reference_path[-1], self.path_tangents[-1], self.path_curvatures[-1]
    
    def add_constraints(self, x: cs.SX, u: cs.SX, k: int, **kwargs) -> List[cs.SX]:
        """
        Add contouring constraints for time step k.
        
        Args:
            x: State variables [x, y, psi, v, ...]
            u: Input variables
            k: Time step
            **kwargs: Additional parameters
            
        Returns:
            List of constraint expressions
        """
        if not self.enabled or self.reference_path is None:
            return []
        
        constraints = []
        
        # Get vehicle position
        vehicle_x = x[0]  # x position
        vehicle_y = x[1]  # y position
        
        # Get path parameter (if available in state)
        if x.shape[0] > 4:  # Assuming spline parameter is at index 4
            s = x[4]  # Path parameter
        else:
            # Estimate path parameter from position
            s = self._estimate_path_parameter(vehicle_x, vehicle_y)
        
        # Get path information at parameter s
        path_point, path_tangent, path_curvature = self._get_path_point_at_s(s)
        
        # Compute lateral distance from path
        dx = vehicle_x - path_point[0]
        dy = vehicle_y - path_point[1]
        
        # Lateral distance (signed)
        lateral_distance = -dx * path_tangent[1] + dy * path_tangent[0]
        
        # Road width constraint
        half_width = (self.road_width - 2 * self.safety_margin) / 2.0
        
        # Left boundary constraint: lateral_distance >= -half_width
        constraints.append(lateral_distance + half_width)
        
        # Right boundary constraint: lateral_distance <= half_width
        constraints.append(half_width - lateral_distance)
        
        # Curvature constraint (if applicable)
        if abs(path_curvature) > self.max_curvature:
            # Add curvature-based constraint
            curvature_constraint = abs(path_curvature) - self.max_curvature
            constraints.append(curvature_constraint)
        
        return constraints
    
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
        Update contouring constraints.
        
        Args:
            state: Current state
            data: Real-time data
            module_data: Module-specific data
        """
        # Update constraint data
        self.constraint_data.update({
            'road_width': self.road_width,
            'safety_margin': self.safety_margin,
            'path_length': self.path_length,
            'reference_path': self.reference_path
        })
    
    def visualize(self, data: Dict[str, Any], module_data: Dict[str, Any]) -> None:
        """
        Visualize contouring constraints.
        
        Args:
            data: Real-time data
            module_data: Module-specific data
        """
        print(f"Contouring Constraints:")
        print(f"  Road width: {self.road_width}")
        print(f"  Safety margin: {self.safety_margin}")
        print(f"  Path length: {self.path_length}")
        print(f"  Enabled: {self.enabled}")
        
        if self.reference_path is not None:
            print(f"  Reference path points: {len(self.reference_path)}")
    
    def get_constraint_info(self) -> Dict[str, Any]:
        """
        Get constraint information.
        
        Returns:
            Constraint information dictionary
        """
        return {
            'constraint_name': self.constraint_name,
            'road_width': self.road_width,
            'safety_margin': self.safety_margin,
            'max_curvature': self.max_curvature,
            'path_length': self.path_length,
            'enabled': self.enabled,
            'reference_path_set': self.reference_path is not None
        }
    
    def set_road_width(self, width: float) -> None:
        """
        Set road width.
        
        Args:
            width: Road width
        """
        self.road_width = width
        self.parameters['road_width'] = width
    
    def set_safety_margin(self, margin: float) -> None:
        """
        Set safety margin.
        
        Args:
            margin: Safety margin
        """
        self.safety_margin = margin
        self.parameters['safety_margin'] = margin
    
    def set_max_curvature(self, curvature: float) -> None:
        """
        Set maximum curvature.
        
        Args:
            curvature: Maximum curvature
        """
        self.max_curvature = curvature
        self.parameters['max_curvature'] = curvature
