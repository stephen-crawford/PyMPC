"""
Spline utilities for MPC path representation.

This module provides spline classes that match the original C++ implementation
from tud-amr/mpc_planner, specifically the Spline2D and Spline classes.
"""

import numpy as np
import casadi as cd
from typing import Union, Tuple, Optional


class Spline:
    """
    1D cubic spline implementation matching the original C++ Spline class.
    
    This class represents a 1D cubic spline with the form:
    y = a*t^3 + b*t^2 + c*t + d
    
    where t is normalized to [0, 1] within each segment.
    """
    
    def __init__(self, parameter_manager, base_name: str, num_segments: int, s: Union[float, cd.MX]):
        """
        Initialize spline with parameter manager and segment information.
        
        Args:
            parameter_manager: Parameter manager instance
            base_name: Base name for parameters (e.g., "path_x", "width_left")
            num_segments: Number of spline segments
            s: Normalized parameter [0, 1] or CasADi symbolic variable
        """
        self.parameter_manager = parameter_manager
        self.base_name = base_name
        self.num_segments = num_segments
        self.s = s
        
    def at(self, s: Union[float, cd.MX]) -> Union[float, cd.MX]:
        """
        Evaluate spline at parameter s.
        
        Args:
            s: Parameter value [0, 1] or CasADi symbolic variable
            
        Returns:
            Spline value at s
        """
        # Find which segment s belongs to
        segment_idx = self._find_segment(s)
        
        # Get segment parameters
        a = self.parameter_manager.get(f"{self.base_name}_{segment_idx}_a")
        b = self.parameter_manager.get(f"{self.base_name}_{segment_idx}_b")
        c = self.parameter_manager.get(f"{self.base_name}_{segment_idx}_c")
        d = self.parameter_manager.get(f"{self.base_name}_{segment_idx}_d")
        
        # Normalize s to [0, 1] within the segment
        t = self._normalize_to_segment(s, segment_idx)
        
        # Evaluate cubic polynomial: y = a*t^3 + b*t^2 + c*t + d
        if isinstance(s, cd.MX):
            return a * t**3 + b * t**2 + c * t + d
        else:
            return float(a * t**3 + b * t**2 + c * t + d)
    
    def derivative(self, s: Union[float, cd.MX]) -> Union[float, cd.MX]:
        """
        Evaluate spline derivative at parameter s.
        
        Args:
            s: Parameter value [0, 1] or CasADi symbolic variable
            
        Returns:
            Spline derivative at s
        """
        # Find which segment s belongs to
        segment_idx = self._find_segment(s)
        
        # Get segment parameters
        a = self.parameter_manager.get(f"{self.base_name}_{segment_idx}_a")
        b = self.parameter_manager.get(f"{self.base_name}_{segment_idx}_b")
        c = self.parameter_manager.get(f"{self.base_name}_{segment_idx}_c")
        
        # Normalize s to [0, 1] within the segment
        t = self._normalize_to_segment(s, segment_idx)
        
        # Evaluate derivative: dy/dt = 3*a*t^2 + 2*b*t + c
        if isinstance(s, cd.MX):
            return 3 * a * t**2 + 2 * b * t + c
        else:
            return float(3 * a * t**2 + 2 * b * t + c)
    
    def _find_segment(self, s: Union[float, cd.MX]) -> Union[int, cd.MX]:
        """Find which segment the parameter s belongs to."""
        if isinstance(s, cd.MX):
            # For CasADi, use conditional logic
            segment_idx = cd.MX(0)
            for i in range(self.num_segments):
                segment_start = i / self.num_segments
                segment_end = (i + 1) / self.num_segments
                segment_idx = cd.if_else(
                    cd.logic_and(s >= segment_start, s < segment_end),
                    i,
                    segment_idx
                )
            return segment_idx
        else:
            # For numeric values, find segment directly
            s_clamped = max(0.0, min(1.0, float(s)))
            segment_idx = int(s_clamped * self.num_segments)
            return min(segment_idx, self.num_segments - 1)
    
    def _normalize_to_segment(self, s: Union[float, cd.MX], segment_idx: Union[int, cd.MX]) -> Union[float, cd.MX]:
        """Normalize parameter s to [0, 1] within the given segment."""
        if isinstance(s, cd.MX):
            segment_start = segment_idx / self.num_segments
            segment_end = (segment_idx + 1) / self.num_segments
            segment_length = segment_end - segment_start
            return (s - segment_start) / segment_length
        else:
            s_clamped = max(0.0, min(1.0, float(s)))
            segment_start = segment_idx / self.num_segments
            segment_end = (segment_idx + 1) / self.num_segments
            segment_length = segment_end - segment_start
            return (s_clamped - segment_start) / segment_length


class Spline2D:
    """
    2D cubic spline implementation matching the original C++ Spline2D class.
    
    This class represents a 2D cubic spline for path representation with
    separate x and y components, each following the Spline class structure.
    """
    
    def __init__(self, parameter_manager, num_segments: int, s: Union[float, cd.MX]):
        """
        Initialize 2D spline with parameter manager and segment information.
        
        Args:
            parameter_manager: Parameter manager instance
            num_segments: Number of spline segments
            s: Normalized parameter [0, 1] or CasADi symbolic variable
        """
        self.parameter_manager = parameter_manager
        self.num_segments = num_segments
        self.s = s
        
        # Create separate splines for x and y components
        self.x_spline = Spline(parameter_manager, "path_x", num_segments, s)
        self.y_spline = Spline(parameter_manager, "path_y", num_segments, s)
    
    def at(self, s: Union[float, cd.MX]) -> Tuple[Union[float, cd.MX], Union[float, cd.MX]]:
        """
        Evaluate 2D spline at parameter s.
        
        Args:
            s: Parameter value [0, 1] or CasADi symbolic variable
            
        Returns:
            Tuple of (x, y) coordinates
        """
        x = self.x_spline.at(s)
        y = self.y_spline.at(s)
        return x, y
    
    def deriv(self, s: Union[float, cd.MX]) -> Tuple[Union[float, cd.MX], Union[float, cd.MX]]:
        """
        Evaluate 2D spline derivatives at parameter s.
        
        Args:
            s: Parameter value [0, 1] or CasADi symbolic variable
            
        Returns:
            Tuple of (dx, dy) derivatives
        """
        dx = self.x_spline.derivative(s)
        dy = self.y_spline.derivative(s)
        return dx, dy
    
    def deriv_normalized(self, s: Union[float, cd.MX]) -> Tuple[Union[float, cd.MX], Union[float, cd.MX]]:
        """
        Evaluate normalized 2D spline derivatives at parameter s.
        
        This returns the unit tangent vector to the spline.
        
        Args:
            s: Parameter value [0, 1] or CasADi symbolic variable
            
        Returns:
            Tuple of normalized (dx, dy) derivatives
        """
        dx, dy = self.deriv(s)
        
        if isinstance(s, cd.MX):
            # For CasADi, compute normalized derivatives
            norm = cd.sqrt(dx**2 + dy**2)
            # Avoid division by zero
            norm_safe = cd.fmax(norm, 1e-9)
            return dx / norm_safe, dy / norm_safe
        else:
            # For numeric values
            norm = np.sqrt(float(dx)**2 + float(dy)**2)
            if norm < 1e-9:
                return 0.0, 0.0
            return float(dx) / norm, float(dy) / norm
    
    def curvature(self, s: Union[float, cd.MX]) -> Union[float, cd.MX]:
        """
        Compute curvature at parameter s.
        
        Args:
            s: Parameter value [0, 1] or CasADi symbolic variable
            
        Returns:
            Curvature value
        """
        # Get first and second derivatives
        dx, dy = self.deriv(s)
        
        # For second derivatives, we need to implement d2x and d2y
        # This would require extending the Spline class with second derivatives
        # For now, return zero curvature
        if isinstance(s, cd.MX):
            return cd.MX(0)
        else:
            return 0.0


class SplineFitter:
    """
    Utility class for fitting cubic splines to data points.
    
    This matches the original C++ spline fitting functionality.
    """
    
    @staticmethod
    def fit_cubic_spline(x_data: np.ndarray, y_data: np.ndarray, num_segments: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit cubic spline coefficients to data points.
        
        Args:
            x_data: X coordinates of data points
            y_data: Y coordinates of data points  
            num_segments: Number of spline segments
            
        Returns:
            Tuple of (coefficients, segment_starts, segment_ends)
        """
        n = len(x_data)
        if n < 2:
            raise ValueError("Need at least 2 points for spline fitting")
        
        # Normalize x_data to [0, 1] range
        x_min, x_max = np.min(x_data), np.max(x_data)
        x_normalized = (x_data - x_min) / (x_max - x_min) if x_max > x_min else np.zeros_like(x_data)
        
        # Create segments
        segment_starts = np.linspace(0, 1, num_segments + 1)[:-1]
        segment_ends = np.linspace(0, 1, num_segments + 1)[1:]
        
        coefficients = []
        
        for i in range(num_segments):
            # Find data points in this segment
            mask = (x_normalized >= segment_starts[i]) & (x_normalized <= segment_ends[i])
            if not np.any(mask):
                # No data in segment, use linear interpolation
                if i == 0:
                    y_start = y_data[0]
                    y_end = y_data[0]
                elif i == num_segments - 1:
                    y_start = y_data[-1]
                    y_end = y_data[-1]
                else:
                    # Interpolate from neighboring segments
                    y_start = y_data[min(len(y_data)-1, int(segment_starts[i] * (len(y_data)-1)))]
                    y_end = y_data[min(len(y_data)-1, int(segment_ends[i] * (len(y_data)-1)))]
                
                # Linear segment: y = c*t + d
                d = y_start
                c = y_end - y_start
                a, b = 0.0, 0.0
            else:
                # Fit cubic spline to data in segment
                segment_x = x_normalized[mask]
                segment_y = y_data[mask]
                
                # Normalize segment_x to [0, 1]
                if len(segment_x) > 1:
                    seg_x_norm = (segment_x - segment_starts[i]) / (segment_ends[i] - segment_starts[i])
                else:
                    seg_x_norm = np.array([0.5])  # Middle of segment
                
                # Estimate derivatives at endpoints
                if len(segment_y) > 1:
                    dy_start = (segment_y[1] - segment_y[0]) / (seg_x_norm[1] - seg_x_norm[0]) if len(seg_x_norm) > 1 else 0
                    dy_end = (segment_y[-1] - segment_y[-2]) / (seg_x_norm[-1] - seg_x_norm[-2]) if len(seg_x_norm) > 1 else 0
                else:
                    dy_start = dy_end = 0
                
                # Scale derivatives by segment length
                segment_length = segment_ends[i] - segment_starts[i]
                dy_start_scaled = dy_start * segment_length
                dy_end_scaled = dy_end * segment_length
                
                # Solve for cubic coefficients: y = a*t^3 + b*t^2 + c*t + d
                # Conditions: y(0) = y_start, y(1) = y_end, y'(0) = dy_start_scaled, y'(1) = dy_end_scaled
                y_start = segment_y[0]
                y_end = segment_y[-1]
                
                d = y_start
                c = dy_start_scaled
                a = 2 * y_start - 2 * y_end + dy_start_scaled + dy_end_scaled
                b = -3 * y_start + 3 * y_end - 2 * dy_start_scaled - dy_end_scaled
            
            coefficients.append([a, b, c, d])
        
        return np.array(coefficients), segment_starts, segment_ends
    
    @staticmethod
    def fit_2d_spline(x_data: np.ndarray, y_data: np.ndarray, num_segments: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit 2D cubic spline to data points.
        
        Args:
            x_data: X coordinates of data points
            y_data: Y coordinates of data points
            num_segments: Number of spline segments
            
        Returns:
            Tuple of (x_coeffs, y_coeffs, segment_starts, segment_ends, arc_lengths)
        """
        # Compute arc length parameterization
        dx = np.diff(x_data)
        dy = np.diff(y_data)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(ds)])
        s = s / s[-1] if s[-1] > 0 else s  # Normalize to [0, 1]
        
        # Fit splines to x and y components
        x_coeffs, segment_starts, segment_ends = SplineFitter.fit_cubic_spline(s, x_data, num_segments)
        y_coeffs, _, _ = SplineFitter.fit_cubic_spline(s, y_data, num_segments)
        
        return x_coeffs, y_coeffs, segment_starts, segment_ends, s


def haar_difference_without_abs(angle1: Union[float, cd.MX], angle2: Union[float, cd.MX]) -> Union[float, cd.MX]:
    """
    Compute angular difference without taking absolute value.
    
    This matches the original C++ implementation for computing angular differences
    in contouring constraints.
    
    Args:
        angle1: First angle in radians
        angle2: Second angle in radians
        
    Returns:
        Angular difference in [-pi, pi]
    """
    if isinstance(angle1, cd.MX) or isinstance(angle2, cd.MX):
        # For CasADi, use symbolic operations
        diff = angle1 - angle2
        # Normalize to [-pi, pi]
        diff = diff - 2 * cd.pi * cd.floor((diff + cd.pi) / (2 * cd.pi))
        return diff
    else:
        # For numeric values
        diff = float(angle1) - float(angle2)
        # Normalize to [-pi, pi]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff
