"""
Visualization utilities for MPC.

This module provides utilities for visualizing MPC trajectories,
constraints, and optimization results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict, Any, Optional, Tuple
import time


class MPCVisualizer:
    """
    Visualization utilities for MPC.
    
    This class provides methods for visualizing MPC trajectories,
    constraints, and optimization results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), **kwargs):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size (width, height)
            **kwargs: Additional parameters
        """
        self.figsize = figsize
        self.parameters = kwargs
        
    def plot_trajectory(self, 
                       trajectory: np.ndarray,
                       reference_path: Optional[np.ndarray] = None,
                       obstacles: Optional[List[Dict[str, Any]]] = None,
                       constraints: Optional[List[Dict[str, Any]]] = None,
                       title: str = "MPC Trajectory",
                       save_path: Optional[str] = None) -> None:
        """
        Plot MPC trajectory with optional reference path and obstacles.
        
        Args:
            trajectory: State trajectory [state_dim, horizon+1]
            reference_path: Reference path [N, 2]
            obstacles: List of obstacle dictionaries
            constraints: List of constraint dictionaries
            title: Plot title
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Extract position coordinates (assuming first two states are x, y)
        x_traj = trajectory[0, :]
        y_traj = trajectory[1, :]
        
        # Plot trajectory
        ax.plot(x_traj, y_traj, 'b-', linewidth=2, label='MPC Trajectory')
        ax.scatter(x_traj[0], y_traj[0], c='green', s=100, marker='o', 
                  label='Start', zorder=5)
        ax.scatter(x_traj[-1], y_traj[-1], c='red', s=100, marker='s', 
                  label='End', zorder=5)
        
        # Plot reference path if provided
        if reference_path is not None:
            ax.plot(reference_path[:, 0], reference_path[:, 1], 
                   'r--', linewidth=2, alpha=0.7, label='Reference Path')
        
        # Plot obstacles if provided
        if obstacles is not None:
            self._plot_obstacles(ax, obstacles)
        
        # Plot constraints if provided
        if constraints is not None:
            self._plot_constraints(ax, constraints)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_contouring_analysis(self, 
                                trajectory: np.ndarray,
                                reference_path: np.ndarray,
                                contouring_errors: Optional[np.ndarray] = None,
                                progress_values: Optional[np.ndarray] = None,
                                title: str = "Contouring Analysis",
                                save_path: Optional[str] = None) -> None:
        """
        Plot contouring analysis showing progress and error.
        
        Args:
            trajectory: State trajectory [state_dim, horizon+1]
            reference_path: Reference path [N, 2]
            contouring_errors: Contouring errors at each time step
            progress_values: Progress values at each time step
            title: Plot title
            save_path: Path to save the plot
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Extract position coordinates
        x_traj = trajectory[0, :]
        y_traj = trajectory[1, :]
        
        # Plot trajectory and reference path
        ax1.plot(reference_path[:, 0], reference_path[:, 1], 
                 'r--', linewidth=2, alpha=0.7, label='Reference Path')
        ax1.plot(x_traj, y_traj, 'b-', linewidth=2, label='MPC Trajectory')
        ax1.scatter(x_traj[0], y_traj[0], c='green', s=100, marker='o', 
                   label='Start', zorder=5)
        ax1.scatter(x_traj[-1], y_traj[-1], c='red', s=100, marker='s', 
                   label='End', zorder=5)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('Trajectory vs Reference')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot contouring errors
        if contouring_errors is not None:
            time_steps = np.arange(len(contouring_errors))
            ax2.plot(time_steps, contouring_errors, 'g-', linewidth=2)
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Contouring Error')
            ax2.set_title('Contouring Error Over Time')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No contouring error data', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Contouring Error')
        
        # Plot progress values
        if progress_values is not None:
            time_steps = np.arange(len(progress_values))
            ax3.plot(time_steps, progress_values, 'b-', linewidth=2)
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Progress')
            ax3.set_title('Progress Over Time')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No progress data', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Progress')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_scenario_analysis(self, 
                              trajectory: np.ndarray,
                              scenarios: List[Dict[str, Any]],
                              scenario_predictions: Optional[List[np.ndarray]] = None,
                              title: str = "Scenario Analysis",
                              save_path: Optional[str] = None) -> None:
        """
        Plot scenario analysis showing different scenario predictions.
        
        Args:
            trajectory: State trajectory [state_dim, horizon+1]
            scenarios: List of scenario dictionaries
            scenario_predictions: List of scenario prediction trajectories
            title: Plot title
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Extract position coordinates
        x_traj = trajectory[0, :]
        y_traj = trajectory[1, :]
        
        # Plot main trajectory
        ax.plot(x_traj, y_traj, 'b-', linewidth=3, label='MPC Trajectory')
        ax.scatter(x_traj[0], y_traj[0], c='green', s=100, marker='o', 
                  label='Start', zorder=5)
        ax.scatter(x_traj[-1], y_traj[-1], c='red', s=100, marker='s', 
                  label='End', zorder=5)
        
        # Plot scenario predictions if provided
        if scenario_predictions is not None:
            for i, pred_traj in enumerate(scenario_predictions):
                x_pred = pred_traj[0, :]
                y_pred = pred_traj[1, :]
                ax.plot(x_pred, y_pred, '--', alpha=0.6, 
                       label=f'Scenario {i+1} Prediction')
        
        # Plot scenario obstacles
        for i, scenario in enumerate(scenarios):
            obstacles = scenario.get('obstacles', [])
            for j, obstacle in enumerate(obstacles):
                self._plot_obstacle(ax, obstacle, 
                                  color=f'C{i+1}', alpha=0.3,
                                  label=f'Scenario {i+1} Obstacle {j+1}' if j == 0 else "")
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_animation(self, 
                        trajectory: np.ndarray,
                        reference_path: Optional[np.ndarray] = None,
                        obstacles: Optional[List[Dict[str, Any]]] = None,
                        title: str = "MPC Animation",
                        save_path: Optional[str] = None,
                        interval: int = 100) -> None:
        """
        Create an animation of the MPC trajectory.
        
        Args:
            trajectory: State trajectory [state_dim, horizon+1]
            reference_path: Reference path [N, 2]
            obstacles: List of obstacle dictionaries
            title: Animation title
            save_path: Path to save the animation
            interval: Animation interval in milliseconds
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Extract position coordinates
        x_traj = trajectory[0, :]
        y_traj = trajectory[1, :]
        
        # Plot reference path if provided
        if reference_path is not None:
            ax.plot(reference_path[:, 0], reference_path[:, 1], 
                   'r--', linewidth=2, alpha=0.7, label='Reference Path')
        
        # Plot obstacles if provided
        if obstacles is not None:
            self._plot_obstacles(ax, obstacles)
        
        # Initialize animation elements
        line, = ax.plot([], [], 'b-', linewidth=2, label='MPC Trajectory')
        point, = ax.plot([], [], 'bo', markersize=8, label='Current Position')
        
        def animate(frame):
            # Update trajectory line
            line.set_data(x_traj[:frame+1], y_traj[:frame+1])
            
            # Update current position
            point.set_data([x_traj[frame]], [y_traj[frame]])
            
            return line, point
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(x_traj),
                                    interval=interval, blit=True, repeat=True)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Set axis limits
        ax.set_xlim(np.min(x_traj) - 1, np.max(x_traj) + 1)
        ax.set_ylim(np.min(y_traj) - 1, np.max(y_traj) + 1)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
        
        plt.show()
    
    def _plot_obstacles(self, ax, obstacles: List[Dict[str, Any]]) -> None:
        """Plot obstacles on the given axes."""
        for obstacle in obstacles:
            self._plot_obstacle(ax, obstacle)
    
    def _plot_obstacle(self, ax, obstacle: Dict[str, Any], 
                      color: str = 'red', alpha: float = 0.5,
                      label: str = 'Obstacle') -> None:
        """Plot a single obstacle on the given axes."""
        if 'center' in obstacle and 'shape' in obstacle:
            # Ellipsoid obstacle
            center = obstacle['center']
            shape = obstacle['shape']
            rotation = obstacle.get('rotation', 0.0)
            
            # Generate points on unit circle
            theta = np.linspace(0, 2*np.pi, 100)
            unit_circle = np.array([np.cos(theta), np.sin(theta)])
            
            # Transform to ellipsoid
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)
            R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            
            # Scale by eigenvalues
            eigenvalues, eigenvectors = np.linalg.eigh(shape)
            scale_matrix = np.diag(np.sqrt(eigenvalues))
            
            transformed_points = R @ scale_matrix @ unit_circle + center[:, np.newaxis]
            
            ax.fill(transformed_points[0], transformed_points[1], 
                   color=color, alpha=alpha, label=label)
        
        elif 'vertices' in obstacle:
            # Polytope obstacle
            vertices = obstacle['vertices']
            closed_vertices = np.vstack([vertices, vertices[0]])
            
            ax.fill(closed_vertices[:, 0], closed_vertices[:, 1], 
                   color=color, alpha=alpha, label=label)
    
    def _plot_constraints(self, ax, constraints: List[Dict[str, Any]]) -> None:
        """Plot constraints on the given axes."""
        for constraint in constraints:
            if constraint.get('type') == 'linear':
                # Plot linear constraints as lines
                A = constraint['A']
                B = constraint['B']
                c = constraint['c']
                
                # This is a simplified visualization
                # In practice, you might want to plot constraint boundaries
                pass
