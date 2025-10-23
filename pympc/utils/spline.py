"""
Spline utilities for MPC framework.

This module provides spline interpolation and manipulation utilities
for smooth path representation and contouring control.
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt


class Spline:
    """
    Spline class for smooth path representation.
    
    This class provides methods for creating, evaluating, and manipulating
    splines for use in MPC path following.
    """
    
    def __init__(self, points: np.ndarray, kind: str = 'cubic'):
        """
        Initialize spline from points.
        
        Args:
            points: Array of points (Nx2) [x, y]
            kind: Type of spline interpolation ('cubic', 'linear', 'quadratic')
        """
        self.points = np.array(points)
        self.kind = kind
        self.spline_x = None
        self.spline_y = None
        self.arc_lengths = None
        self.total_length = 0.0
        
        if len(self.points) < 2:
            raise ValueError("At least 2 points required for spline")
        
        self._create_spline()
    
    def _create_spline(self) -> None:
        """Create the spline interpolation."""
        if len(self.points) < 2:
            return
        
        # Compute arc lengths
        diffs = np.diff(self.points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        self.arc_lengths = np.concatenate([[0], np.cumsum(distances)])
        self.total_length = self.arc_lengths[-1]
        
        # Create splines for x and y coordinates
        if self.kind == 'cubic' and len(self.points) >= 4:
            self.spline_x = CubicSpline(self.arc_lengths, self.points[:, 0])
            self.spline_y = CubicSpline(self.arc_lengths, self.points[:, 1])
        else:
            # Fall back to linear interpolation
            self.spline_x = interp1d(self.arc_lengths, self.points[:, 0], 
                                   kind='linear', bounds_error=False, 
                                   fill_value='extrapolate')
            self.spline_y = interp1d(self.arc_lengths, self.points[:, 1], 
                                   kind='linear', bounds_error=False, 
                                   fill_value='extrapolate')
    
    def evaluate(self, s: float) -> Tuple[float, float]:
        """
        Evaluate spline at arc length s.
        
        Args:
            s: Arc length parameter (0 to total_length)
            
        Returns:
            Tuple of (x, y) coordinates
        """
        if self.spline_x is None or self.spline_y is None:
            return (0.0, 0.0)
        
        # Clamp s to valid range
        s = np.clip(s, 0, self.total_length)
        
        x = float(self.spline_x(s))
        y = float(self.spline_y(s))
        
        return (x, y)
    
    def evaluate_derivative(self, s: float) -> Tuple[float, float]:
        """
        Evaluate spline derivative at arc length s.
        
        Args:
            s: Arc length parameter
            
        Returns:
            Tuple of (dx/ds, dy/ds)
        """
        if self.spline_x is None or self.spline_y is None:
            return (0.0, 0.0)
        
        # Clamp s to valid range
        s = np.clip(s, 0, self.total_length)
        
        if hasattr(self.spline_x, 'derivative'):
            dx_ds = float(self.spline_x.derivative()(s))
            dy_ds = float(self.spline_y.derivative()(s))
        else:
            # For linear interpolation, use finite differences
            eps = 1e-6
            s1 = max(0, s - eps)
            s2 = min(self.total_length, s + eps)
            
            x1, y1 = self.evaluate(s1)
            x2, y2 = self.evaluate(s2)
            
            dx_ds = (x2 - x1) / (s2 - s1) if s2 > s1 else 0.0
            dy_ds = (y2 - y1) / (s2 - s1) if s2 > s1 else 0.0
        
        return (dx_ds, dy_ds)
    
    def evaluate_second_derivative(self, s: float) -> Tuple[float, float]:
        """
        Evaluate spline second derivative at arc length s.
        
        Args:
            s: Arc length parameter
            
        Returns:
            Tuple of (d²x/ds², d²y/ds²)
        """
        if self.spline_x is None or self.spline_y is None:
            return (0.0, 0.0)
        
        # Clamp s to valid range
        s = np.clip(s, 0, self.total_length)
        
        if hasattr(self.spline_x, 'derivative') and hasattr(self.spline_x.derivative(), 'derivative'):
            d2x_ds2 = float(self.spline_x.derivative().derivative()(s))
            d2y_ds2 = float(self.spline_y.derivative().derivative()(s))
        else:
            # For linear interpolation, second derivative is zero
            d2x_ds2 = 0.0
            d2y_ds2 = 0.0
        
        return (d2x_ds2, d2y_ds2)
    
    def get_tangent(self, s: float) -> Tuple[float, float]:
        """
        Get tangent vector at arc length s.
        
        Args:
            s: Arc length parameter
            
        Returns:
            Normalized tangent vector (tx, ty)
        """
        dx_ds, dy_ds = self.evaluate_derivative(s)
        norm = np.sqrt(dx_ds**2 + dy_ds**2)
        
        if norm > 1e-8:
            return (dx_ds / norm, dy_ds / norm)
        else:
            return (1.0, 0.0)  # Default tangent
    
    def get_normal(self, s: float) -> Tuple[float, float]:
        """
        Get normal vector at arc length s.
        
        Args:
            s: Arc length parameter
            
        Returns:
            Normalized normal vector (nx, ny)
        """
        tx, ty = self.get_tangent(s)
        return (-ty, tx)  # Rotate tangent by 90 degrees
    
    def get_curvature(self, s: float) -> float:
        """
        Get curvature at arc length s.
        
        Args:
            s: Arc length parameter
            
        Returns:
            Curvature value
        """
        dx_ds, dy_ds = self.evaluate_derivative(s)
        d2x_ds2, d2y_ds2 = self.evaluate_second_derivative(s)
        
        # Curvature formula: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
        numerator = dx_ds * d2y_ds2 - dy_ds * d2x_ds2
        denominator = (dx_ds**2 + dy_ds**2)**(3/2)
        
        if abs(denominator) > 1e-8:
            return numerator / denominator
        else:
            return 0.0
    
    def find_closest_point(self, x: float, y: float) -> Tuple[float, float, float]:
        """
        Find the closest point on the spline to a given point.
        
        Args:
            x: Query x coordinate
            y: Query y coordinate
            
        Returns:
            Tuple of (s, closest_x, closest_y)
        """
        if self.spline_x is None or self.spline_y is None:
            return (0.0, 0.0, 0.0)
        
        # Sample the spline at many points to find closest
        num_samples = max(100, len(self.points) * 10)
        s_samples = np.linspace(0, self.total_length, num_samples)
        
        distances = []
        for s in s_samples:
            sx, sy = self.evaluate(s)
            dist = np.sqrt((x - sx)**2 + (y - sy)**2)
            distances.append(dist)
        
        # Find minimum distance
        min_idx = np.argmin(distances)
        s_closest = s_samples[min_idx]
        closest_x, closest_y = self.evaluate(s_closest)
        
        return (s_closest, closest_x, closest_y)
    
    def get_contouring_error(self, x: float, y: float) -> Tuple[float, float]:
        """
        Compute contouring error (lateral deviation from path).
        
        Args:
            x: Vehicle x position
            y: Vehicle y position
            
        Returns:
            Tuple of (lag_error, contouring_error)
        """
        s, closest_x, closest_y = self.find_closest_point(x, y)
        
        # Compute lag error (longitudinal deviation)
        lag_error = s
        
        # Compute contouring error (lateral deviation)
        tx, ty = self.get_tangent(s)
        contouring_error = (x - closest_x) * (-ty) + (y - closest_y) * tx
        
        return (lag_error, contouring_error)
    
    def sample_points(self, num_points: int = None, ds: float = None) -> np.ndarray:
        """
        Sample points along the spline.
        
        Args:
            num_points: Number of points to sample
            ds: Arc length step size
            
        Returns:
            Array of sampled points (Nx2)
        """
        if num_points is not None:
            s_values = np.linspace(0, self.total_length, num_points)
        elif ds is not None:
            s_values = np.arange(0, self.total_length + ds, ds)
        else:
            s_values = self.arc_lengths
        
        points = []
        for s in s_values:
            x, y = self.evaluate(s)
            points.append([x, y])
        
        return np.array(points)
    
    def get_length(self) -> float:
        """
        Get total arc length of the spline.
        
        Returns:
            Total arc length
        """
        return self.total_length
    
    def visualize(self, ax: plt.Axes, num_points: int = 100, **kwargs) -> None:
        """
        Visualize the spline on a matplotlib axes.
        
        Args:
            ax: Matplotlib axes object
            num_points: Number of points to plot
            **kwargs: Additional plotting options
        """
        s_values = np.linspace(0, self.total_length, num_points)
        x_values = []
        y_values = []
        
        for s in s_values:
            x, y = self.evaluate(s)
            x_values.append(x)
            y_values.append(y)
        
        ax.plot(x_values, y_values, **kwargs)
        
        # Plot original points
        ax.scatter(self.points[:, 0], self.points[:, 1], 
                  color='red', s=50, zorder=5, label='Control Points')
    
    def __len__(self) -> int:
        """Return number of control points."""
        return len(self.points)
    
    def __str__(self) -> str:
        """String representation."""
        return f"Spline({self.kind}, {len(self.points)} points, length={self.total_length:.2f})"