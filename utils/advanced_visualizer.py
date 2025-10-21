"""
Advanced visualization system for MPC trajectories and constraints.

This module provides comprehensive visualization tools for MPC optimization,
constraint analysis, and performance monitoring.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime

# Optional imports
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Set style for better plots
if SEABORN_AVAILABLE:
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
else:
    plt.style.use('default')


class MPCVisualizer:
    """
    Advanced visualizer for MPC trajectories and constraints.
    """
    
    def __init__(self, 
                 figsize: Tuple[int, int] = (12, 8),
                 dpi: int = 100,
                 save_dir: str = "plots"):
        """
        Initialize MPC visualizer.
        
        Args:
            figsize: Figure size (width, height)
            dpi: Dots per inch
            save_dir: Directory to save plots
        """
        self.figsize = figsize
        self.dpi = dpi
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Create session directory
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.save_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        # Color schemes
        self.colors = {
            'trajectory': '#2E86AB',
            'reference': '#A23B72',
            'obstacle': '#F18F01',
            'goal': '#C73E1D',
            'start': '#2E8B57',
            'constraint': '#8B4513',
            'uncertainty': '#9370DB'
        }
        
        self.plot_count = 0
    
    def plot_trajectory_2d(self, 
                          trajectory: np.ndarray,
                          reference_path: Optional[np.ndarray] = None,
                          obstacles: Optional[List[Dict]] = None,
                          goal: Optional[np.ndarray] = None,
                          title: str = "MPC Trajectory",
                          save: bool = True,
                          show: bool = True) -> plt.Figure:
        """
        Plot 2D trajectory with obstacles and reference path.
        
        Args:
            trajectory: Vehicle trajectory (state_dim, N)
            reference_path: Reference path (2, N) or (N, 2)
            obstacles: List of obstacle dictionaries
            goal: Goal position (2,)
            title: Plot title
            save: Save plot to file
            show: Display plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Extract positions
        if trajectory.shape[0] >= 2:
            positions = trajectory[:2, :]
        else:
            positions = trajectory
        
        # Plot reference path
        if reference_path is not None:
            if reference_path.shape[0] == 2:
                ref_x, ref_y = reference_path[0, :], reference_path[1, :]
            else:
                ref_x, ref_y = reference_path[:, 0], reference_path[:, 1]
            
            ax.plot(ref_x, ref_y, '--', color=self.colors['reference'], 
                   linewidth=2, alpha=0.7, label='Reference Path')
        
        # Plot obstacles
        if obstacles is not None:
            self._plot_obstacles(ax, obstacles)
        
        # Plot trajectory
        ax.plot(positions[0, :], positions[1, :], '-', 
               color=self.colors['trajectory'], linewidth=3, label='Trajectory')
        
        # Plot start and end points
        ax.plot(positions[0, 0], positions[1, 0], 'o', 
               color=self.colors['start'], markersize=10, label='Start')
        ax.plot(positions[0, -1], positions[1, -1], 's', 
               color=self.colors['goal'], markersize=10, label='End')
        
        # Plot goal if provided
        if goal is not None:
            ax.plot(goal[0], goal[1], '*', 
                   color=self.colors['goal'], markersize=15, label='Goal')
        
        # Formatting
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        if save:
            filename = f"trajectory_2d_{self.plot_count:03d}.png"
            filepath = self.session_dir / filename
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            self.plot_count += 1
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_trajectory_3d(self, 
                          trajectory: np.ndarray,
                          reference_path: Optional[np.ndarray] = None,
                          title: str = "MPC Trajectory 3D",
                          save: bool = True,
                          show: bool = True) -> plt.Figure:
        """
        Plot 3D trajectory.
        
        Args:
            trajectory: Vehicle trajectory (state_dim, N)
            reference_path: Reference path (3, N) or (N, 3)
            title: Plot title
            save: Save plot to file
            show: Display plot
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract positions
        if trajectory.shape[0] >= 3:
            positions = trajectory[:3, :]
        else:
            # Pad with zeros for 2D trajectories
            positions = np.vstack([trajectory[:2, :], np.zeros(trajectory.shape[1])])
        
        # Plot reference path
        if reference_path is not None:
            if reference_path.shape[0] == 3:
                ref_x, ref_y, ref_z = reference_path[0, :], reference_path[1, :], reference_path[2, :]
            else:
                ref_x, ref_y, ref_z = reference_path[:, 0], reference_path[:, 1], reference_path[:, 2]
            
            ax.plot(ref_x, ref_y, ref_z, '--', color=self.colors['reference'], 
                   linewidth=2, alpha=0.7, label='Reference Path')
        
        # Plot trajectory
        ax.plot(positions[0, :], positions[1, :], positions[2, :], '-', 
               color=self.colors['trajectory'], linewidth=3, label='Trajectory')
        
        # Plot start and end points
        ax.plot(positions[0, 0], positions[1, 0], positions[2, 0], 'o', 
               color=self.colors['start'], markersize=10, label='Start')
        ax.plot(positions[0, -1], positions[1, -1], positions[2, -1], 's', 
               color=self.colors['goal'], markersize=10, label='End')
        
        # Formatting
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            filename = f"trajectory_3d_{self.plot_count:03d}.png"
            filepath = self.session_dir / filename
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            self.plot_count += 1
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_state_evolution(self, 
                           trajectory: np.ndarray,
                           dt: float = 0.1,
                           state_names: Optional[List[str]] = None,
                           title: str = "State Evolution",
                           save: bool = True,
                           show: bool = True) -> plt.Figure:
        """
        Plot state evolution over time.
        
        Args:
            trajectory: Vehicle trajectory (state_dim, N)
            dt: Time step
            state_names: Names for state variables
            title: Plot title
            save: Save plot to file
            show: Display plot
            
        Returns:
            matplotlib Figure object
        """
        time_steps = np.arange(trajectory.shape[1]) * dt
        
        if state_names is None:
            state_names = [f'State {i}' for i in range(trajectory.shape[0])]
        
        fig, axes = plt.subplots(trajectory.shape[0], 1, figsize=(12, 3*trajectory.shape[0]), dpi=self.dpi)
        if trajectory.shape[0] == 1:
            axes = [axes]
        
        for i, (state, name) in enumerate(zip(trajectory, state_names)):
            axes[i].plot(time_steps, state, color=self.colors['trajectory'], linewidth=2)
            axes[i].set_ylabel(name)
            axes[i].grid(True, alpha=0.3)
            if i == trajectory.shape[0] - 1:
                axes[i].set_xlabel('Time (s)')
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save:
            filename = f"state_evolution_{self.plot_count:03d}.png"
            filepath = self.session_dir / filename
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            self.plot_count += 1
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_control_evolution(self, 
                              controls: np.ndarray,
                              dt: float = 0.1,
                              control_names: Optional[List[str]] = None,
                              title: str = "Control Evolution",
                              save: bool = True,
                              show: bool = True) -> plt.Figure:
        """
        Plot control evolution over time.
        
        Args:
            controls: Control trajectory (control_dim, N)
            dt: Time step
            control_names: Names for control variables
            title: Plot title
            save: Save plot to file
            show: Display plot
            
        Returns:
            matplotlib Figure object
        """
        time_steps = np.arange(controls.shape[1]) * dt
        
        if control_names is None:
            control_names = [f'Control {i}' for i in range(controls.shape[0])]
        
        fig, axes = plt.subplots(controls.shape[0], 1, figsize=(12, 3*controls.shape[0]), dpi=self.dpi)
        if controls.shape[0] == 1:
            axes = [axes]
        
        for i, (control, name) in enumerate(zip(controls, control_names)):
            axes[i].plot(time_steps, control, color=self.colors['constraint'], linewidth=2)
            axes[i].set_ylabel(name)
            axes[i].grid(True, alpha=0.3)
            if i == controls.shape[0] - 1:
                axes[i].set_xlabel('Time (s)')
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save:
            filename = f"control_evolution_{self.plot_count:03d}.png"
            filepath = self.session_dir / filename
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            self.plot_count += 1
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_constraint_analysis(self, 
                               trajectory: np.ndarray,
                               constraints: List[Dict[str, Any]],
                               title: str = "Constraint Analysis",
                               save: bool = True,
                               show: bool = True) -> plt.Figure:
        """
        Plot constraint analysis.
        
        Args:
            trajectory: Vehicle trajectory (state_dim, N)
            constraints: List of constraint dictionaries
            title: Plot title
            save: Save plot to file
            show: Display plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Extract positions
        positions = trajectory[:2, :] if trajectory.shape[0] >= 2 else trajectory
        
        # Plot trajectory
        ax.plot(positions[0, :], positions[1, :], '-', 
               color=self.colors['trajectory'], linewidth=3, label='Trajectory')
        
        # Plot constraints
        for i, constraint in enumerate(constraints):
            self._plot_constraint(ax, constraint, i)
        
        # Formatting
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        if save:
            filename = f"constraint_analysis_{self.plot_count:03d}.png"
            filepath = self.session_dir / filename
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            self.plot_count += 1
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_performance_metrics(self, 
                               metrics: Dict[str, Any],
                               title: str = "Performance Metrics",
                               save: bool = True,
                               show: bool = True) -> plt.Figure:
        """
        Plot performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            title: Plot title
            save: Save plot to file
            show: Display plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create bar plot
        names = list(metrics.keys())
        values = list(metrics.values())
        
        bars = ax.bar(names, values, color=self.colors['constraint'], alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            filename = f"performance_metrics_{self.plot_count:03d}.png"
            filepath = self.session_dir / filename
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            self.plot_count += 1
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def create_animation(self, 
                        trajectory: np.ndarray,
                        obstacles: Optional[List[Dict]] = None,
                        reference_path: Optional[np.ndarray] = None,
                        title: str = "MPC Animation",
                        interval: int = 100,
                        save: bool = True,
                        show: bool = True) -> FuncAnimation:
        """
        Create animated trajectory.
        
        Args:
            trajectory: Vehicle trajectory (state_dim, N)
            obstacles: List of obstacle dictionaries
            reference_path: Reference path
            title: Animation title
            interval: Animation interval in ms
            save: Save animation to file
            show: Display animation
            
        Returns:
            matplotlib FuncAnimation object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Extract positions
        positions = trajectory[:2, :] if trajectory.shape[0] >= 2 else trajectory
        
        # Plot static elements
        if reference_path is not None:
            if reference_path.shape[0] == 2:
                ref_x, ref_y = reference_path[0, :], reference_path[1, :]
            else:
                ref_x, ref_y = reference_path[:, 0], reference_path[:, 1]
            ax.plot(ref_x, ref_y, '--', color=self.colors['reference'], 
                   linewidth=2, alpha=0.7, label='Reference Path')
        
        if obstacles is not None:
            self._plot_obstacles(ax, obstacles)
        
        # Initialize animated elements
        line, = ax.plot([], [], '-', color=self.colors['trajectory'], linewidth=3, label='Trajectory')
        point, = ax.plot([], [], 'o', color=self.colors['start'], markersize=10, label='Vehicle')
        
        def animate(frame):
            # Update trajectory
            line.set_data(positions[0, :frame+1], positions[1, :frame+1])
            point.set_data([positions[0, frame]], [positions[1, frame]])
            return line, point
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=positions.shape[1], 
                           interval=interval, blit=True, repeat=True)
        
        # Formatting
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        if save:
            filename = f"animation_{self.plot_count:03d}.gif"
            filepath = self.session_dir / filename
            anim.save(filepath, writer='pillow', fps=10)
            self.plot_count += 1
        
        if show:
            plt.show()
        
        return anim
    
    def _plot_obstacles(self, ax: plt.Axes, obstacles: List[Dict]) -> None:
        """Plot obstacles on axes."""
        for i, obs in enumerate(obstacles):
            if 'center' in obs and 'shape' in obs:
                center = obs['center']
                shape = obs['shape']
                
                if len(shape) == 2:  # Ellipsoid
                    ellipse = patches.Ellipse(center, 2*shape[0], 2*shape[1], 
                                            alpha=0.5, color=self.colors['obstacle'])
                    ax.add_patch(ellipse)
                elif len(shape) == 4:  # Rectangle
                    rect = patches.Rectangle((center[0]-shape[0]/2, center[1]-shape[1]/2), 
                                          shape[0], shape[1], alpha=0.5, 
                                          color=self.colors['obstacle'])
                    ax.add_patch(rect)
    
    def _plot_constraint(self, ax: plt.Axes, constraint: Dict[str, Any], index: int) -> None:
        """Plot constraint on axes."""
        constraint_type = constraint.get('type', 'unknown')
        
        if constraint_type == 'ellipsoid':
            center = constraint.get('center', [0, 0])
            shape = constraint.get('shape', [1, 1])
            ellipse = patches.Ellipse(center, 2*shape[0], 2*shape[1], 
                                    alpha=0.3, color=self.colors['constraint'],
                                    label=f'Constraint {index}')
            ax.add_patch(ellipse)
        elif constraint_type == 'rectangle':
            center = constraint.get('center', [0, 0])
            shape = constraint.get('shape', [1, 1])
            rect = patches.Rectangle((center[0]-shape[0]/2, center[1]-shape[1]/2), 
                                   shape[0], shape[1], alpha=0.3, 
                                   color=self.colors['constraint'],
                                   label=f'Constraint {index}')
            ax.add_patch(rect)
    
    def save_session_summary(self, session_data: Dict[str, Any]) -> None:
        """Save session summary."""
        summary_file = self.session_dir / "session_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
    
    def get_session_dir(self) -> Path:
        """Get session directory path."""
        return self.session_dir
