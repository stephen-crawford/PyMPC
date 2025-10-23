"""
Utility functions for MPC framework.
"""

import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Optional, Tuple, List, Dict, Union
import time
import json
import os
from datetime import datetime


# Set up logging
def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


# Create logger
logger = logging.getLogger('pympc')


def LOG_DEBUG(message: str) -> None:
    """Log debug message."""
    logger.debug(message)


def LOG_INFO(message: str) -> None:
    """Log info message."""
    logger.info(message)


def LOG_WARN(message: str) -> None:
    """Log warning message."""
    logger.warning(message)


def LOG_ERROR(message: str) -> None:
    """Log error message."""
    logger.error(message)


# Initialize logging
setup_logging()


# Mathematical utilities
def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-pi, pi].
    
    Args:
        angle: Angle in radians
        
    Returns:
        Normalized angle
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def wrap_angle(angle: float) -> float:
    """
    Wrap angle to [0, 2*pi].
    
    Args:
        angle: Angle in radians
        
    Returns:
        Wrapped angle
    """
    return angle % (2 * np.pi)


def rotation_matrix(angle: float) -> np.ndarray:
    """
    Create 2D rotation matrix.
    
    Args:
        angle: Rotation angle in radians
        
    Returns:
        2x2 rotation matrix
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]])


def distance_point_to_line(point: np.ndarray, line_start: np.ndarray, 
                          line_end: np.ndarray) -> float:
    """
    Compute distance from point to line segment.
    
    Args:
        point: Point [x, y]
        line_start: Line start point [x, y]
        line_end: Line end point [x, y]
        
    Returns:
        Distance to line segment
    """
    # Vector from line start to point
    point_vec = point - line_start
    
    # Vector along line
    line_vec = line_end - line_start
    line_length = np.linalg.norm(line_vec)
    
    if line_length < 1e-8:
        return np.linalg.norm(point_vec)
    
    # Project point onto line
    t = np.dot(point_vec, line_vec) / (line_length * line_length)
    t = np.clip(t, 0, 1)  # Clamp to line segment
    
    # Closest point on line
    closest_point = line_start + t * line_vec
    
    return np.linalg.norm(point - closest_point)


def distance_point_to_polygon(point: np.ndarray, vertices: np.ndarray) -> float:
    """
    Compute distance from point to polygon.
    
    Args:
        point: Point [x, y]
        vertices: Polygon vertices (Nx2 array)
        
    Returns:
        Distance to polygon
    """
    min_distance = float('inf')
    n = len(vertices)
    
    for i in range(n):
        j = (i + 1) % n
        dist = distance_point_to_line(point, vertices[i], vertices[j])
        min_distance = min(min_distance, dist)
    
    return min_distance


def is_point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
    """
    Check if point is inside polygon.
    
    Args:
        point: Point [x, y]
        vertices: Polygon vertices (Nx2 array)
        
    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(vertices)
    inside = False
    
    for i in range(n):
        j = (i + 1) % n
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
    
    return inside


# Visualization utilities
def create_figure(size: Tuple[int, int] = (10, 8), 
                 title: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a matplotlib figure and axes.
    
    Args:
        size: Figure size (width, height)
        title: Figure title
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=size)
    if title:
        ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_trajectory(ax: plt.Axes, trajectory: np.ndarray, 
                   color: str = 'blue', label: str = 'Trajectory',
                   linewidth: int = 2) -> None:
    """
    Plot a trajectory on axes.
    
    Args:
        ax: Matplotlib axes
        trajectory: Trajectory points (Nx2 array)
        color: Line color
        label: Line label
        linewidth: Line width
    """
    if len(trajectory) > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 1], 
                color=color, linewidth=linewidth, label=label)


def plot_vehicle(ax: plt.Axes, position: np.ndarray, heading: float = 0.0,
                length: float = 4.0, width: float = 1.8,
                color: str = 'red', alpha: float = 0.7) -> None:
    """
    Plot vehicle as rectangle.
    
    Args:
        ax: Matplotlib axes
        position: Vehicle position [x, y]
        heading: Vehicle heading angle
        length: Vehicle length
        width: Vehicle width
        color: Vehicle color
        alpha: Transparency
    """
    # Create vehicle rectangle
    from matplotlib.patches import Rectangle
    
    # Vehicle corners in local frame
    corners = np.array([
        [-length/2, -width/2],
        [length/2, -width/2],
        [length/2, width/2],
        [-length/2, width/2]
    ])
    
    # Rotate corners
    R = rotation_matrix(heading)
    rotated_corners = corners @ R.T
    
    # Translate to position
    world_corners = rotated_corners + position
    
    # Create rectangle patch
    vehicle_patch = plt.Polygon(world_corners, color=color, alpha=alpha)
    ax.add_patch(vehicle_patch)
    
    # Add heading arrow
    arrow_length = length/2
    arrow_end = position + arrow_length * np.array([np.cos(heading), np.sin(heading)])
    ax.arrow(position[0], position[1], 
             arrow_end[0] - position[0], arrow_end[1] - position[1],
             head_width=0.5, head_length=0.5, fc=color, ec=color)


def plot_obstacles(ax: plt.Axes, obstacles: List[Dict[str, Any]], 
                  color: str = 'red', alpha: float = 0.5) -> None:
    """
    Plot obstacles on axes.
    
    Args:
        ax: Matplotlib axes
        obstacles: List of obstacle dictionaries
        color: Obstacle color
        alpha: Transparency
    """
    for obstacle in obstacles:
        if obstacle['type'] == 'circle':
            circle = plt.Circle(obstacle['position'], obstacle['radius'],
                              color=color, alpha=alpha)
            ax.add_patch(circle)
        elif obstacle['type'] == 'ellipse':
            ellipse = plt.Ellipse(obstacle['position'], 
                                obstacle['width'], obstacle['height'],
                                angle=np.degrees(obstacle['angle']),
                                color=color, alpha=alpha)
            ax.add_patch(ellipse)
        elif obstacle['type'] == 'polygon':
            polygon = plt.Polygon(obstacle['vertices'], 
                                color=color, alpha=alpha)
            ax.add_patch(polygon)


# File I/O utilities
def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data dictionary
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Data dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_numpy_array(array: np.ndarray, filepath: str) -> None:
    """
    Save numpy array to file.
    
    Args:
        array: Numpy array
        filepath: Output file path
    """
    np.save(filepath, array)


def load_numpy_array(filepath: str) -> np.ndarray:
    """
    Load numpy array from file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Numpy array
    """
    return np.load(filepath)


# Performance utilities
class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        LOG_INFO(f"{self.name} took {duration:.4f} seconds")
    
    def elapsed_time(self) -> float:
        """Get elapsed time."""
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or time.time()
        return end_time - self.start_time


def time_function(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        LOG_DEBUG(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper


# Data validation utilities
def validate_trajectory(trajectory: np.ndarray) -> bool:
    """
    Validate trajectory data.
    
    Args:
        trajectory: Trajectory array (Nx2 or Nx3)
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(trajectory, np.ndarray):
        return False
    
    if trajectory.ndim != 2:
        return False
    
    if trajectory.shape[1] < 2:
        return False
    
    if np.any(np.isnan(trajectory)) or np.any(np.isinf(trajectory)):
        return False
    
    return True


def validate_control_sequence(controls: np.ndarray, 
                            max_velocity: float = None,
                            max_acceleration: float = None) -> bool:
    """
    Validate control sequence.
    
    Args:
        controls: Control sequence (NxM array)
        max_velocity: Maximum velocity constraint
        max_acceleration: Maximum acceleration constraint
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(controls, np.ndarray):
        return False
    
    if controls.ndim != 2:
        return False
    
    if np.any(np.isnan(controls)) or np.any(np.isinf(controls)):
        return False
    
    if max_velocity is not None and controls.shape[1] > 0:
        velocities = controls[:, 0]
        if np.any(np.abs(velocities) > max_velocity):
            return False
    
    if max_acceleration is not None and controls.shape[1] > 1:
        accelerations = controls[:, 1]
        if np.any(np.abs(accelerations) > max_acceleration):
            return False
    
    return True


# Configuration utilities
def create_output_directory(base_dir: str, test_name: str) -> str:
    """
    Create output directory for test results.
    
    Args:
        base_dir: Base directory
        test_name: Test name
        
    Returns:
        Created directory path
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f"{test_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


# Error handling utilities
class MPCError(Exception):
    """Base exception for MPC framework."""
    pass


class SolverError(MPCError):
    """Exception for solver errors."""
    pass


class ConstraintError(MPCError):
    """Exception for constraint errors."""
    pass


class ObjectiveError(MPCError):
    """Exception for objective errors."""
    pass


def handle_mpc_error(error: Exception, context: str = "") -> None:
    """
    Handle MPC errors with logging.
    
    Args:
        error: Exception to handle
        context: Additional context string
    """
    error_msg = f"MPC Error in {context}: {str(error)}"
    LOG_ERROR(error_msg)
    raise MPCError(error_msg) from error
