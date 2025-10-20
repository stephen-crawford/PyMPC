"""
Standardized Visualization Framework for PyMPC

This module provides a comprehensive visualization system for MPC testing
with real-time constraint visualization, trajectory plotting, and interactive debugging.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.collections import LineCollection
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path


class VisualizationMode(Enum):
    """Visualization modes for different use cases."""
    REALTIME = "realtime"
    SAVE_ANIMATION = "save_animation"
    SAVE_PLOTS = "save_plots"
    INTERACTIVE = "interactive"
    HEADLESS = "headless"


@dataclass
class VisualizationConfig:
    """Configuration for visualization system."""
    mode: VisualizationMode = VisualizationMode.REALTIME
    realtime: bool = True
    show_constraint_projection: bool = True
    show_trajectory: bool = True
    show_vehicle: bool = True
    show_obstacles: bool = True
    show_constraints: bool = True
    show_path: bool = True
    save_animation: bool = False
    save_plots: bool = False
    fps: int = 10
    dpi: int = 100
    output_dir: str = "test_results/visualizations"
    figure_size: Tuple[float, float] = (12, 8)
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    aspect_equal: bool = True
    grid: bool = True
    legend: bool = True


class TestVisualizationManager:
    """
    Comprehensive visualization manager for MPC testing.
    
    Features:
    - Real-time constraint visualization
    - Trajectory plotting with history
    - Interactive debugging tools
    - Animation generation
    - Multi-constraint overlay support
    """
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.config = None
        self.fig = None
        self.ax = None
        self.animation = None
        self.artists = {}
        self.trajectory_history = []
        self.constraint_overlays = {}
        self.vehicle_state = None
        self.reference_path = None
        self.obstacles = []
        self.constraints = []
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
    def initialize(self, config: VisualizationConfig):
        """Initialize the visualization system."""
        self.config = config
        
        # Create output directory
        if config.save_animation or config.save_plots:
            os.makedirs(config.output_dir, exist_ok=True)
        
        # Setup matplotlib
        if config.mode == VisualizationMode.HEADLESS:
            plt.switch_backend('Agg')
        else:
            plt.ion()
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=config.figure_size)
        self.ax.set_title(f"MPC Test: {self.test_name}")
        
        if config.aspect_equal:
            self.ax.set_aspect('equal')
        
        if config.grid:
            self.ax.grid(True, alpha=0.3)
        
        if config.legend:
            self.ax.legend()
        
        # Initialize artists
        self._initialize_artists()
        
        if config.mode == VisualizationMode.REALTIME:
            plt.show(block=False)
            plt.pause(0.1)
    
    def _initialize_artists(self):
        """Initialize matplotlib artists."""
        self.artists = {
            'reference_path': None,
            'vehicle': None,
            'trajectory': None,
            'obstacles': [],
            'constraints': [],
            'constraint_projection': [],
            'goal': None,
            'road_bounds': []
        }
    
    def update_vehicle_state(self, state: Dict[str, float]):
        """Update vehicle state for visualization."""
        self.vehicle_state = state
        
        # Update vehicle position
        if self.artists['vehicle'] is not None:
            self.artists['vehicle'].remove()
        
        x, y = state.get('x', 0), state.get('y', 0)
        psi = state.get('psi', 0)
        
        # Draw vehicle as arrow
        dx = 0.5 * np.cos(psi)
        dy = 0.5 * np.sin(psi)
        
        self.artists['vehicle'] = self.ax.arrow(
            x, y, dx, dy,
            head_width=0.3, head_length=0.2,
            fc='red', ec='red', alpha=0.8,
            label='Vehicle'
        )
    
    def update_trajectory(self, trajectory: List[Dict[str, float]]):
        """Update trajectory visualization."""
        if not trajectory:
            return
        
        # Add to history
        self.trajectory_history.extend(trajectory)
        
        # Limit history length
        max_history = 100
        if len(self.trajectory_history) > max_history:
            self.trajectory_history = self.trajectory_history[-max_history:]
        
        # Update trajectory plot
        if self.artists['trajectory'] is not None:
            self.artists['trajectory'].remove()
        
        if self.trajectory_history:
            x_traj = [point.get('x', 0) for point in self.trajectory_history]
            y_traj = [point.get('y', 0) for point in self.trajectory_history]
            
            self.artists['trajectory'] = self.ax.plot(
                x_traj, y_traj,
                'b-', alpha=0.7, linewidth=2,
                label='Trajectory'
            )[0]
    
    def update_reference_path(self, path: Dict[str, List[float]]):
        """Update reference path visualization."""
        self.reference_path = path
        
        if self.artists['reference_path'] is not None:
            self.artists['reference_path'].remove()
        
        if path and 'x' in path and 'y' in path:
            self.artists['reference_path'] = self.ax.plot(
                path['x'], path['y'],
                'g--', alpha=0.8, linewidth=2,
                label='Reference Path'
            )[0]
    
    def update_road_bounds(self, left_bound: Dict[str, List[float]], 
                          right_bound: Dict[str, List[float]]):
        """Update road boundary visualization."""
        # Clear existing bounds
        for artist in self.artists['road_bounds']:
            artist.remove()
        self.artists['road_bounds'] = []
        
        if left_bound and 'x' in left_bound and 'y' in left_bound:
            left_line = self.ax.plot(
                left_bound['x'], left_bound['y'],
                'k-', alpha=0.6, linewidth=1,
                label='Left Bound'
            )[0]
            self.artists['road_bounds'].append(left_line)
        
        if right_bound and 'x' in right_bound and 'y' in right_bound:
            right_line = self.ax.plot(
                right_bound['x'], right_bound['y'],
                'k-', alpha=0.6, linewidth=1,
                label='Right Bound'
            )[0]
            self.artists['road_bounds'].append(right_line)
    
    def update_obstacles(self, obstacles: List[Dict[str, Any]]):
        """Update obstacle visualization."""
        # Clear existing obstacles
        for artist in self.artists['obstacles']:
            artist.remove()
        self.artists['obstacles'] = []
        
        for i, obs in enumerate(obstacles):
            x, y = obs.get('x', 0), obs.get('y', 0)
            radius = obs.get('radius', 0.5)
            obs_type = obs.get('type', 'static')
            
            # Choose color based on type
            color = 'orange' if obs_type == 'dynamic' else 'red'
            alpha = 0.6 if obs_type == 'dynamic' else 0.8
            
            circle = Circle((x, y), radius, color=color, alpha=alpha)
            self.ax.add_patch(circle)
            self.artists['obstacles'].append(circle)
            
            # Add label
            self.ax.text(x, y, f'O{i}', ha='center', va='center', fontsize=8)
    
    def update_constraints(self, constraints: List[Dict[str, Any]]):
        """Update constraint visualization."""
        if not self.config.show_constraints:
            return
        
        # Clear existing constraints
        for artist in self.artists['constraints']:
            artist.remove()
        self.artists['constraints'] = []
        
        for i, constraint in enumerate(constraints):
            constraint_type = constraint.get('type', 'halfspace')
            
            if constraint_type == 'halfspace':
                self._draw_halfspace_constraint(constraint, i)
            elif constraint_type == 'circle':
                self._draw_circle_constraint(constraint, i)
            elif constraint_type == 'polygon':
                self._draw_polygon_constraint(constraint, i)
    
    def _draw_halfspace_constraint(self, constraint: Dict[str, Any], index: int):
        """Draw halfspace constraint visualization."""
        A = constraint.get('A', [0, 0])
        b = constraint.get('b', 0)
        
        # Create line representing the constraint boundary
        # For visualization, we'll draw a line perpendicular to the normal
        if len(A) >= 2:
            # Normalize A
            norm = np.linalg.norm(A)
            if norm > 1e-6:
                A_norm = np.array(A) / norm
                b_norm = b / norm
                
                # Create line perpendicular to A
                perp = np.array([-A_norm[1], A_norm[0]])
                
                # Get current plot limits
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                
                # Find intersection points with plot boundaries
                points = []
                for x in [xlim[0], xlim[1]]:
                    y = (b_norm - A_norm[0] * x) / A_norm[1] if abs(A_norm[1]) > 1e-6 else ylim[0]
                    if ylim[0] <= y <= ylim[1]:
                        points.append([x, y])
                
                for y in [ylim[0], ylim[1]]:
                    x = (b_norm - A_norm[1] * y) / A_norm[0] if abs(A_norm[0]) > 1e-6 else xlim[0]
                    if xlim[0] <= x <= xlim[1]:
                        points.append([x, y])
                
                if len(points) >= 2:
                    points = np.array(points)
                    line = self.ax.plot(
                        points[:, 0], points[:, 1],
                        'r--', alpha=0.5, linewidth=1,
                        label=f'Constraint {index}' if index == 0 else ""
                    )[0]
                    self.artists['constraints'].append(line)
    
    def _draw_circle_constraint(self, constraint: Dict[str, Any], index: int):
        """Draw circle constraint visualization."""
        center = constraint.get('center', [0, 0])
        radius = constraint.get('radius', 1.0)
        
        circle = Circle(center, radius, fill=False, color='purple', 
                       linestyle='--', alpha=0.7, linewidth=1)
        self.ax.add_patch(circle)
        self.artists['constraints'].append(circle)
    
    def _draw_polygon_constraint(self, constraint: Dict[str, Any], index: int):
        """Draw polygon constraint visualization."""
        vertices = constraint.get('vertices', [])
        if len(vertices) >= 3:
            polygon = Polygon(vertices, fill=False, color='brown',
                            linestyle='--', alpha=0.7, linewidth=1)
            self.ax.add_patch(polygon)
            self.artists['constraints'].append(polygon)
    
    def update_constraint_projection(self, projections: List[Dict[str, Any]]):
        """Update constraint projection visualization."""
        if not self.config.show_constraint_projection:
            return
        
        # Clear existing projections
        for artist in self.artists['constraint_projection']:
            artist.remove()
        self.artists['constraint_projection'] = []
        
        for i, proj in enumerate(projections):
            proj_type = proj.get('type', 'point')
            
            if proj_type == 'point':
                x, y = proj.get('x', 0), proj.get('y', 0)
                color = proj.get('color', 'cyan')
                size = proj.get('size', 20)
                
                point = self.ax.scatter(x, y, c=color, s=size, alpha=0.8,
                                      label=f'Projection {i}' if i == 0 else "")
                self.artists['constraint_projection'].append(point)
    
    def update_goal(self, goal: Tuple[float, float]):
        """Update goal visualization."""
        if self.artists['goal'] is not None:
            self.artists['goal'].remove()
        
        x, y = goal
        self.artists['goal'] = self.ax.scatter(
            x, y, c='green', s=100, marker='*',
            label='Goal', zorder=10
        )
    
    def add_constraint_overlay(self, module_name: str, overlay: Dict[str, Any]):
        """Add constraint overlay from a specific module."""
        self.constraint_overlays[module_name] = overlay
        
        # Process overlay data
        if 'halfspaces' in overlay:
            for i, halfspace in enumerate(overlay['halfspaces']):
                constraint = {
                    'type': 'halfspace',
                    'A': halfspace.get('A', [0, 0]),
                    'b': halfspace.get('b', 0)
                }
                self._draw_halfspace_constraint(constraint, i)
        
        if 'polygons' in overlay:
            for i, polygon in enumerate(overlay['polygons']):
                constraint = {
                    'type': 'polygon',
                    'vertices': polygon.get('vertices', [])
                }
                self._draw_polygon_constraint(constraint, i)
        
        if 'points' in overlay:
            for i, point in enumerate(overlay['points']):
                self.ax.scatter(
                    point.get('x', 0), point.get('y', 0),
                    c=point.get('color', 'blue'),
                    s=point.get('size', 20),
                    alpha=point.get('alpha', 0.8),
                    label=f'{module_name} Point {i}' if i == 0 else ""
                )
    
    def refresh_display(self):
        """Refresh the display with current data."""
        if self.config.mode == VisualizationMode.REALTIME:
            # Update limits
            if self.vehicle_state:
                x, y = self.vehicle_state.get('x', 0), self.vehicle_state.get('y', 0)
                if self.config.xlim is None:
                    self.ax.set_xlim(x - 10, x + 10)
                if self.config.ylim is None:
                    self.ax.set_ylim(y - 10, y + 10)
            
            # Update legend
            if self.config.legend:
                handles, labels = self.ax.get_legend_handles_labels()
                if handles:
                    self.ax.legend(handles, labels, loc='upper right')
            
            # Refresh
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
    
    def save_frame(self, frame_name: str = None):
        """Save current frame as image."""
        if not self.config.save_plots:
            return
        
        if frame_name is None:
            frame_name = f"frame_{self.frame_count:04d}"
        
        filename = os.path.join(self.config.output_dir, f"{frame_name}.png")
        self.fig.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
        self.frame_count += 1
    
    def create_animation(self, duration: float = 10.0):
        """Create animation from trajectory history."""
        if not self.config.save_animation or not self.trajectory_history:
            return
        
        def animate(frame):
            # Clear axis
            self.ax.clear()
            
            # Plot reference path
            if self.reference_path:
                self.ax.plot(
                    self.reference_path['x'], self.reference_path['y'],
                    'g--', alpha=0.8, linewidth=2, label='Reference Path'
                )
            
            # Plot trajectory up to current frame
            if frame < len(self.trajectory_history):
                traj = self.trajectory_history[:frame+1]
                x_traj = [point.get('x', 0) for point in traj]
                y_traj = [point.get('y', 0) for point in traj]
                self.ax.plot(x_traj, y_traj, 'b-', alpha=0.7, linewidth=2, label='Trajectory')
                
                # Plot current vehicle position
                if traj:
                    current = traj[-1]
                    x, y = current.get('x', 0), current.get('y', 0)
                    psi = current.get('psi', 0)
                    dx = 0.5 * np.cos(psi)
                    dy = 0.5 * np.sin(psi)
                    self.ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.2,
                               fc='red', ec='red', alpha=0.8, label='Vehicle')
            
            # Plot obstacles
            for obs in self.obstacles:
                x, y = obs.get('x', 0), obs.get('y', 0)
                radius = obs.get('radius', 0.5)
                circle = Circle((x, y), radius, color='orange', alpha=0.6)
                self.ax.add_patch(circle)
            
            # Set limits and labels
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_title(f'MPC Test: {self.test_name} - Frame {frame}')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            
            if self.config.aspect_equal:
                self.ax.set_aspect('equal')
        
        # Create animation
        frames = min(len(self.trajectory_history), int(duration * self.config.fps))
        self.animation = animation.FuncAnimation(
            self.fig, animate, frames=frames,
            interval=1000/self.config.fps, repeat=False
        )
        
        # Save animation
        filename = os.path.join(self.config.output_dir, f"{self.test_name}_animation.gif")
        self.animation.save(filename, writer='pillow', fps=self.config.fps)
        
        return filename
    
    def cleanup(self):
        """Cleanup visualization resources."""
        if self.animation:
            self.animation = None
        
        if self.fig:
            plt.close(self.fig)
            self.fig = None
        
        self.artists.clear()
        self.trajectory_history.clear()
        self.constraint_overlays.clear()
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'frames_rendered': self.frame_count,
            'elapsed_time': elapsed_time,
            'average_fps': fps
        }