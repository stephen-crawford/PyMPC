"""
Real-time visualization system for MPC vehicle progress and constraints.

This module provides real-time visualization capabilities that show
vehicle progress, constraints, and optimization results as they happen.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime
import time

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


class RealtimeVisualizer:
    """
    Real-time visualizer for MPC vehicle progress and constraints.
    """
    
    def __init__(self, 
                 figsize: Tuple[int, int] = (14, 10),
                 dpi: int = 100,
                 save_dir: str = "realtime_plots",
                 fps: int = 10):
        """
        Initialize real-time visualizer.
        
        Args:
            figsize: Figure size (width, height)
            dpi: Dots per inch
            save_dir: Directory to save plots and GIFs
            fps: Frames per second for animation
        """
        self.figsize = figsize
        self.dpi = dpi
        self.save_dir = Path(save_dir)
        self.fps = fps
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Animation data
        self.animation_data = {
            'trajectory': [],
            'reference_path': None,
            'obstacles': [],
            'constraints': [],
            'vehicle_states': [],
            'control_inputs': [],
            'timestamps': [],
            'objective_values': [],
            'solve_times': [],
            'constraint_violations': []
        }
        
        # Current frame data
        self.current_frame = 0
        self.max_frames = 0
        
        # Figure and axes
        self.fig = None
        self.ax_main = None
        self.ax_states = None
        self.ax_controls = None
        self.ax_performance = None
        
        # Animation objects
        self.animation = None
        self.vehicle_patch = None
        self.trajectory_line = None
        self.reference_line = None
        self.obstacle_patches = []
        self.constraint_patches = []
        
    def initialize_plot(self, 
                      reference_path: np.ndarray,
                      obstacles: List[Dict] = None,
                      xlim: Tuple[float, float] = None,
                      ylim: Tuple[float, float] = None) -> None:
        """
        Initialize the real-time plot.
        
        Args:
            reference_path: Reference path as array of [x, y] points
            obstacles: List of obstacle dictionaries
            xlim: X-axis limits
            ylim: Y-axis limits
        """
        # Create figure with subplots
        self.fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        self.ax_main, self.ax_states = axes[0, 0], axes[0, 1]
        self.ax_controls, self.ax_performance = axes[1, 0], axes[1, 1]
        
        # Set up main trajectory plot
        self.ax_main.set_title('Real-time Vehicle Progress', fontsize=14, fontweight='bold')
        self.ax_main.set_xlabel('X Position (m)')
        self.ax_main.set_ylabel('Y Position (m)')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_aspect('equal')
        
        # Set axis limits
        if xlim:
            self.ax_main.set_xlim(xlim)
        if ylim:
            self.ax_main.set_ylim(ylim)
        else:
            # Auto-scale based on reference path
            if len(reference_path) > 0:
                margin = 2.0
                x_min, x_max = np.min(reference_path[:, 0]) - margin, np.max(reference_path[:, 0]) + margin
                y_min, y_max = np.min(reference_path[:, 1]) - margin, np.max(reference_path[:, 1]) + margin
                self.ax_main.set_xlim(x_min, x_max)
                self.ax_main.set_ylim(y_min, y_max)
        
        # Plot reference path
        if len(reference_path) > 0:
            self.reference_line, = self.ax_main.plot(
                reference_path[:, 0], reference_path[:, 1], 
                'k--', linewidth=2, alpha=0.7, label='Reference Path'
            )
        
        # Initialize trajectory line
        self.trajectory_line, = self.ax_main.plot([], [], 'b-', linewidth=2, label='Vehicle Trajectory')
        
        # Initialize vehicle patch (triangle) with default position
        default_vehicle = np.array([[0, 0], [0, 0], [0, 0]])  # Default triangle
        self.vehicle_patch = patches.Polygon(default_vehicle, closed=True, 
                                            facecolor='red', edgecolor='black', 
                                            linewidth=2, alpha=0.8, label='Vehicle')
        self.ax_main.add_patch(self.vehicle_patch)
        
        # Initialize obstacle patches
        self.obstacle_patches = []
        if obstacles:
            for i, obs in enumerate(obstacles):
                if 'center' in obs and 'shape' in obs:
                    # Ellipsoid obstacle
                    center = obs['center']
                    shape = obs['shape']
                    safety_margin = obs.get('safety_margin', 0.0)
                    
                    # Create ellipse patch
                    eigenvals, eigenvecs = np.linalg.eigh(shape)
                    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                    width = 2 * np.sqrt(eigenvals[0]) + 2 * safety_margin
                    height = 2 * np.sqrt(eigenvals[1]) + 2 * safety_margin
                    
                    ellipse = patches.Ellipse(center, width, height, angle=angle,
                                            facecolor='red', alpha=0.3, edgecolor='red',
                                            linewidth=2, label=f'Obstacle {i+1}')
                    self.obstacle_patches.append(ellipse)
                    self.ax_main.add_patch(ellipse)
        
        # Set up state evolution plot
        self.ax_states.set_title('Vehicle States', fontsize=12)
        self.ax_states.set_xlabel('Time Step')
        self.ax_states.set_ylabel('State Value')
        self.ax_states.grid(True, alpha=0.3)
        
        # Set up control inputs plot
        self.ax_controls.set_title('Control Inputs', fontsize=12)
        self.ax_controls.set_xlabel('Time Step')
        self.ax_controls.set_ylabel('Control Value')
        self.ax_controls.grid(True, alpha=0.3)
        
        # Set up performance plot
        self.ax_performance.set_title('Performance Metrics', fontsize=12)
        self.ax_performance.set_xlabel('Time Step')
        self.ax_performance.set_ylabel('Value')
        self.ax_performance.grid(True, alpha=0.3)
        
        # Add legend to main plot
        self.ax_main.legend(loc='upper right')
        
        # Store reference data
        self.animation_data['reference_path'] = reference_path
        if obstacles:
            self.animation_data['obstacles'] = obstacles
        
        plt.tight_layout()
    
    def update_frame(self, 
                    vehicle_state: np.ndarray,
                    control_input: np.ndarray,
                    trajectory: np.ndarray,
                    objective_value: float = None,
                    solve_time: float = None,
                    constraint_violations: Dict = None,
                    timestamp: float = None) -> None:
        """
        Update the visualization with new frame data.
        
        Args:
            vehicle_state: Current vehicle state [x, y, theta, v, delta]
            control_input: Current control input [a, delta_dot]
            trajectory: Trajectory history
            objective_value: Current objective value
            solve_time: Solve time for current step
            constraint_violations: Constraint violation information
            timestamp: Current timestamp
        """
        # Store data
        self.animation_data['trajectory'].append(trajectory.copy())
        self.animation_data['vehicle_states'].append(vehicle_state.copy())
        self.animation_data['control_inputs'].append(control_input.copy())
        self.animation_data['timestamps'].append(timestamp or time.time())
        
        if objective_value is not None:
            self.animation_data['objective_values'].append(objective_value)
        if solve_time is not None:
            self.animation_data['solve_times'].append(solve_time)
        if constraint_violations:
            self.animation_data['constraint_violations'].append(constraint_violations)
        
        self.current_frame += 1
    
    def _update_vehicle_patch(self, x: float, y: float, theta: float, 
                              vehicle_length: float = 2.0, vehicle_width: float = 1.0) -> None:
        """Update vehicle patch position and orientation."""
        # Create vehicle triangle
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Vehicle corners (triangle pointing forward)
        front_x = x + vehicle_length/2 * cos_theta
        front_y = y + vehicle_length/2 * sin_theta
        back_left_x = x - vehicle_length/2 * cos_theta + vehicle_width/2 * sin_theta
        back_left_y = y - vehicle_length/2 * sin_theta - vehicle_width/2 * cos_theta
        back_right_x = x - vehicle_length/2 * cos_theta - vehicle_width/2 * sin_theta
        back_right_y = y - vehicle_length/2 * sin_theta + vehicle_width/2 * cos_theta
        
        # Update patch
        self.vehicle_patch.set_xy([
            [front_x, front_y],
            [back_left_x, back_left_y],
            [back_right_x, back_right_y]
        ])
    
    def _animate_frame(self, frame: int) -> List:
        """Animate a single frame."""
        if frame >= len(self.animation_data['trajectory']):
            return []
        
        # Get current data
        trajectory = self.animation_data['trajectory'][frame]
        vehicle_state = self.animation_data['vehicle_states'][frame]
        control_input = self.animation_data['control_inputs'][frame]
        
        # Update trajectory line
        if len(trajectory) > 1:
            self.trajectory_line.set_data(trajectory[:, 0], trajectory[:, 1])
        
        # Update vehicle position
        x, y, theta = vehicle_state[0], vehicle_state[1], vehicle_state[2]
        self._update_vehicle_patch(x, y, theta)
        
        # Update state plot
        if frame > 0:
            states_data = np.array(self.animation_data['vehicle_states'][:frame+1])
            time_steps = np.arange(len(states_data))
            
            self.ax_states.clear()
            self.ax_states.set_title('Vehicle States', fontsize=12)
            self.ax_states.set_xlabel('Time Step')
            self.ax_states.set_ylabel('State Value')
            self.ax_states.grid(True, alpha=0.3)
            
            # Plot each state
            state_labels = ['x', 'y', 'θ', 'v', 'δ']
            colors = ['blue', 'green', 'red', 'orange', 'purple']
            for i in range(min(states_data.shape[1], 5)):
                self.ax_states.plot(time_steps, states_data[:, i], 
                                  color=colors[i], label=state_labels[i], linewidth=2)
            self.ax_states.legend()
        
        # Update control plot
        if frame > 0:
            controls_data = np.array(self.animation_data['control_inputs'][:frame+1])
            time_steps = np.arange(len(controls_data))
            
            self.ax_controls.clear()
            self.ax_controls.set_title('Control Inputs', fontsize=12)
            self.ax_controls.set_xlabel('Time Step')
            self.ax_controls.set_ylabel('Control Value')
            self.ax_controls.grid(True, alpha=0.3)
            
            # Plot each control
            control_labels = ['a', 'δ̇']
            colors = ['blue', 'red']
            for i in range(min(controls_data.shape[1], 2)):
                self.ax_controls.plot(time_steps, controls_data[:, i], 
                                    color=colors[i], label=control_labels[i], linewidth=2)
            self.ax_controls.legend()
        
        # Update performance plot
        if frame > 0 and len(self.animation_data['objective_values']) > 0:
            obj_values = self.animation_data['objective_values'][:frame+1]
            solve_times = self.animation_data['solve_times'][:frame+1] if len(self.animation_data['solve_times']) > 0 else []
            time_steps = np.arange(len(obj_values))
            
            # Clear and reset performance plot
            self.ax_performance.clear()
            self.ax_performance.set_title('Performance Metrics', fontsize=12)
            self.ax_performance.set_xlabel('Time Step')
            self.ax_performance.set_ylabel('Objective Value')
            self.ax_performance.grid(True, alpha=0.3)
            
            # Plot objective values
            self.ax_performance.plot(time_steps, obj_values, 'b-', 
                                   label='Objective Value', linewidth=2)
            
            # Plot solve times on separate y-axis if available
            if len(solve_times) > 0:
                # Create twin axis only once and reuse
                if not hasattr(self, 'ax_performance_twin'):
                    self.ax_performance_twin = self.ax_performance.twinx()
                    self.ax_performance_twin.set_ylabel('Solve Time (s)', color='r')
                    self.ax_performance_twin.tick_params(axis='y', labelcolor='r')
                
                # Clear the twin axis and plot new data
                self.ax_performance_twin.clear()
                self.ax_performance_twin.set_ylabel('Solve Time (s)', color='r')
                self.ax_performance_twin.tick_params(axis='y', labelcolor='r')
                self.ax_performance_twin.plot(time_steps, solve_times, 'r-', 
                                            label='Solve Time (s)', linewidth=2)
                
                # Create combined legend
                lines1, labels1 = self.ax_performance.get_legend_handles_labels()
                lines2, labels2 = self.ax_performance_twin.get_legend_handles_labels()
                self.ax_performance.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            else:
                self.ax_performance.legend(loc='upper left')
        
        # Update main plot title with current step
        self.ax_main.set_title(f'Real-time Vehicle Progress - Step {frame}', 
                              fontsize=14, fontweight='bold')
        
        return []
    
    def start_animation(self, 
                       total_frames: int,
                       save_gif: bool = True,
                       gif_filename: str = None) -> None:
        """
        Start the real-time animation.
        
        Args:
            total_frames: Total number of frames to animate
            save_gif: Whether to save as GIF
            gif_filename: Custom GIF filename
        """
        self.max_frames = total_frames
        
        if gif_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            gif_filename = f"realtime_animation_{timestamp}.gif"
        
        gif_path = self.save_dir / gif_filename
        
        # Create animation
        self.animation = FuncAnimation(
            self.fig, self._animate_frame, 
            frames=total_frames, 
            interval=1000//self.fps,  # Convert fps to interval
            blit=False, repeat=False
        )
        
        if save_gif:
            print(f"Saving animation as GIF: {gif_path}")
            self.animation.save(str(gif_path), writer='pillow', fps=self.fps)
            print(f"GIF saved successfully: {gif_path}")
        
        return gif_path
    
    def show_animation(self) -> None:
        """Show the animation."""
        if self.animation:
            plt.show()
    
    def save_frame(self, frame: int, filename: str = None) -> str:
        """
        Save a single frame as an image.
        
        Args:
            frame: Frame number to save
            filename: Custom filename
            
        Returns:
            Path to saved image
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"frame_{frame:03d}_{timestamp}.png"
        
        frame_path = self.save_dir / filename
        
        # Update to specific frame
        self._animate_frame(frame)
        
        # Save figure
        self.fig.savefig(str(frame_path), dpi=self.dpi, bbox_inches='tight')
        
        return str(frame_path)
    
    def export_data(self, filename: str = None) -> str:
        """
        Export animation data to JSON.
        
        Args:
            filename: Custom filename
            
        Returns:
            Path to saved JSON file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"animation_data_{timestamp}.json"
        
        data_path = self.save_dir / filename
        
        # Simple conversion function
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        # Convert all data
        export_data = convert_to_serializable(self.animation_data)
        
        with open(data_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return str(data_path)
    
    def close(self) -> None:
        """Close the visualizer."""
        if self.fig:
            plt.close(self.fig)
        self.animation = None
