"""
Mathematical utilities for MPC.

This module provides various mathematical utility functions
for MPC computations.
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from scipy.spatial.distance import cdist
from scipy.optimize import minimize


class MathUtils:
    """
    Mathematical utilities for MPC computations.
    """
    
    @staticmethod
    def closest_point_on_path(point: np.ndarray, 
                             path: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        Find the closest point on a path to a given point.
        
        Args:
            point: Query point [x, y]
            path: Path points [N, 2]
            
        Returns:
            Tuple of (closest_point, distance, segment_index)
        """
        if len(path) < 2:
            return path[0], np.linalg.norm(point - path[0]), 0
        
        min_distance = float('inf')
        closest_point = path[0]
        segment_index = 0
        
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            
            # Project point onto line segment
            segment = end - start
            segment_length = np.linalg.norm(segment)
            
            if segment_length == 0:
                continue
            
            t = np.dot(point - start, segment) / (segment_length ** 2)
            t = np.clip(t, 0, 1)
            
            closest_on_segment = start + t * segment
            distance = np.linalg.norm(point - closest_on_segment)
            
            if distance < min_distance:
                min_distance = distance
                closest_point = closest_on_segment
                segment_index = i
        
        return closest_point, min_distance, segment_index
    
    @staticmethod
    def compute_path_progress(point: np.ndarray, 
                             path: np.ndarray) -> float:
        """
        Compute progress along a path.
        
        Args:
            point: Current point [x, y]
            path: Path points [N, 2]
            
        Returns:
            Progress value (0 to path_length)
        """
        if len(path) < 2:
            return 0.0
        
        # Find closest point on path
        closest_point, _, segment_index = MathUtils.closest_point_on_path(point, path)
        
        # Compute progress up to the closest segment
        progress = 0.0
        for i in range(segment_index):
            segment_length = np.linalg.norm(path[i + 1] - path[i])
            progress += segment_length
        
        # Add progress within the current segment
        segment_start = path[segment_index]
        segment_progress = np.linalg.norm(closest_point - segment_start)
        progress += segment_progress
        
        return progress
    
    @staticmethod
    def compute_curvature(path: np.ndarray) -> np.ndarray:
        """
        Compute curvature of a path.
        
        Args:
            path: Path points [N, 2]
            
        Returns:
            Curvature values [N-2]
        """
        if len(path) < 3:
            return np.array([])
        
        # Compute first and second derivatives using finite differences
        dx = np.gradient(path[:, 0])
        dy = np.gradient(path[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Compute curvature: k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = np.abs(dx[:-1] * ddy[:-1] - dy[:-1] * ddx[:-1])
        denominator = (dx[:-1]**2 + dy[:-1]**2)**(3/2)
        
        # Avoid division by zero
        curvature = np.zeros_like(numerator)
        valid_mask = denominator > 1e-6
        curvature[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        
        return curvature
    
    @staticmethod
    def interpolate_path(path: np.ndarray, 
                        num_points: int) -> np.ndarray:
        """
        Interpolate a path to have a specific number of points.
        
        Args:
            path: Original path points [N, 2]
            num_points: Desired number of points
            
        Returns:
            Interpolated path [num_points, 2]
        """
        if len(path) < 2:
            return path
        
        # Compute cumulative distances
        distances = np.zeros(len(path))
        for i in range(1, len(path)):
            distances[i] = distances[i-1] + np.linalg.norm(path[i] - path[i-1])
        
        # Create interpolation points
        total_distance = distances[-1]
        interpolation_distances = np.linspace(0, total_distance, num_points)
        
        # Interpolate x and y coordinates
        x_interp = np.interp(interpolation_distances, distances, path[:, 0])
        y_interp = np.interp(interpolation_distances, distances, path[:, 1])
        
        return np.column_stack([x_interp, y_interp])
    
    @staticmethod
    def smooth_path(path: np.ndarray, 
                   smoothing_factor: float = 0.1) -> np.ndarray:
        """
        Smooth a path using moving average.
        
        Args:
            path: Path points [N, 2]
            smoothing_factor: Smoothing factor (0 to 1)
            
        Returns:
            Smoothed path [N, 2]
        """
        if len(path) < 3:
            return path
        
        window_size = max(3, int(len(path) * smoothing_factor))
        if window_size % 2 == 0:
            window_size += 1
        
        # Apply moving average
        smoothed_path = np.zeros_like(path)
        half_window = window_size // 2
        
        for i in range(len(path)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(path), i + half_window + 1)
            smoothed_path[i] = np.mean(path[start_idx:end_idx], axis=0)
        
        return smoothed_path
    
    @staticmethod
    def compute_path_length(path: np.ndarray) -> float:
        """
        Compute the total length of a path.
        
        Args:
            path: Path points [N, 2]
            
        Returns:
            Total path length
        """
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(path)):
            total_length += np.linalg.norm(path[i] - path[i-1])
        
        return total_length
    
    @staticmethod
    def find_path_intersection(path1: np.ndarray, 
                              path2: np.ndarray) -> Optional[np.ndarray]:
        """
        Find intersection point between two paths.
        
        Args:
            path1: First path [N, 2]
            path2: Second path [M, 2]
            
        Returns:
            Intersection point [x, y] or None if no intersection
        """
        if len(path1) < 2 or len(path2) < 2:
            return None
        
        # Check each segment of path1 against each segment of path2
        for i in range(len(path1) - 1):
            for j in range(len(path2) - 1):
                intersection = MathUtils._line_segment_intersection(
                    path1[i], path1[i+1], path2[j], path2[j+1]
                )
                if intersection is not None:
                    return intersection
        
        return None
    
    @staticmethod
    def _line_segment_intersection(p1: np.ndarray, p2: np.ndarray,
                                  p3: np.ndarray, p4: np.ndarray) -> Optional[np.ndarray]:
        """
        Find intersection between two line segments.
        
        Args:
            p1, p2: First line segment endpoints
            p3, p4: Second line segment endpoints
            
        Returns:
            Intersection point or None
        """
        # Line 1: p1 + t * (p2 - p1)
        # Line 2: p3 + s * (p4 - p3)
        
        d1 = p2 - p1
        d2 = p4 - p3
        d3 = p1 - p3
        
        # Compute cross products
        cross_d1_d2 = d1[0] * d2[1] - d1[1] * d2[0]
        
        if abs(cross_d1_d2) < 1e-10:  # Lines are parallel
            return None
        
        # Compute intersection parameters
        t = (d3[0] * d2[1] - d3[1] * d2[0]) / cross_d1_d2
        s = (d3[0] * d1[1] - d3[1] * d1[0]) / cross_d1_d2
        
        # Check if intersection is within both segments
        if 0 <= t <= 1 and 0 <= s <= 1:
            return p1 + t * d1
        
        return None
    
    @staticmethod
    def compute_heading_angle(path: np.ndarray) -> np.ndarray:
        """
        Compute heading angles along a path.
        
        Args:
            path: Path points [N, 2]
            
        Returns:
            Heading angles [N-1] in radians
        """
        if len(path) < 2:
            return np.array([])
        
        headings = np.zeros(len(path) - 1)
        for i in range(len(path) - 1):
            dx = path[i+1, 0] - path[i, 0]
            dy = path[i+1, 1] - path[i, 1]
            headings[i] = np.arctan2(dy, dx)
        
        return headings
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        Normalize an angle to [-pi, pi].
        
        Args:
            angle: Angle in radians
            
        Returns:
            Normalized angle
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    @staticmethod
    def angle_difference(angle1: float, angle2: float) -> float:
        """
        Compute the shortest angular difference between two angles.
        
        Args:
            angle1: First angle in radians
            angle2: Second angle in radians
            
        Returns:
            Angular difference in [-pi, pi]
        """
        diff = angle1 - angle2
        return MathUtils.normalize_angle(diff)
