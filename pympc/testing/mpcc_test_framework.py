"""
Standardized MPCC Testing Framework

This module provides a comprehensive testing framework for Model Predictive Contouring Control (MPCC)
with configurable parameters, perception areas, and dynamic obstacles.

Based on the C++ mpc_planner implementation from https://github.com/tud-amr/mpc_planner
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
import time
import json

from ..core.dynamics import BicycleModel


class PerceptionShape(Enum):
    """Perception area shapes."""
    CIRCLE = "circle"
    SPHERE = "sphere"
    CONE = "cone"
    FORWARD_CONE = "forward_cone"
    BACKWARD_CONE = "backward_cone"
    RECTANGLE = "rectangle"


@dataclass
class PerceptionConfig:
    """Configuration for vehicle perception area."""
    shape: PerceptionShape = PerceptionShape.CIRCLE
    distance: float = 20.0  # Maximum perception distance
    angle: float = np.pi / 3  # For cone shapes (half-angle)
    width: float = 10.0  # For rectangle shape
    height: float = 20.0  # For rectangle shape
    enabled: bool = True


@dataclass
class ObstacleConfig:
    """Configuration for dynamic obstacles."""
    num_obstacles: int = 3
    radius_range: Tuple[float, float] = (0.8, 1.5)
    velocity_range: Tuple[float, float] = (2.0, 8.0)
    trajectory_type: str = "random_walk"  # "random_walk", "linear", "circular"
    intersection_probability: float = 0.7  # Probability of intersecting with road


@dataclass
class RoadConfig:
    """Configuration for road generation."""
    road_type: str = "curved"  # "curved", "straight", "s_curve"
    length: float = 120.0
    width: float = 6.0
    curvature_intensity: float = 1.0
    num_points: int = 100


@dataclass
class VehicleConfig:
    """Configuration for vehicle dynamics."""
    length: float = 4.0
    width: float = 1.8
    wheelbase: float = 2.5
    max_velocity: float = 15.0
    max_acceleration: float = 3.0
    max_steering_angle: float = 0.5
    dynamics_model: str = "bicycle"  # "bicycle", "kinematic"


@dataclass
class MPCConfig:
    """Configuration for MPC parameters."""
    horizon: int = 15
    timestep: float = 0.1
    max_steps: int = 150
    solver_tolerance: float = 1e-6
    contouring_weight: float = 1.0
    lag_weight: float = 1.0
    velocity_weight: float = 0.1
    progress_weight: float = 1.0


@dataclass
class TestConfig:
    """Complete test configuration."""
    test_name: str = "mpcc_test"
    road: RoadConfig = field(default_factory=RoadConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    output_dir: str = "mpcc_test_outputs"
    generate_gif: bool = True
    gif_fps: int = 10
    save_data: bool = True


class PerceptionArea:
    """Vehicle perception area implementation."""
    
    def __init__(self, config: PerceptionConfig):
        self.config = config
        self.enabled = config.enabled
    
    def is_obstacle_visible(self, vehicle_pos: np.ndarray, vehicle_heading: float, 
                          obstacle_pos: np.ndarray) -> bool:
        """Check if obstacle is within perception area."""
        if not self.enabled:
            return True
        
        # Calculate relative position
        rel_pos = obstacle_pos - vehicle_pos
        distance = np.linalg.norm(rel_pos)
        
        if distance > self.config.distance:
            return False
        
        if self.config.shape == PerceptionShape.CIRCLE:
            return distance <= self.config.distance
        
        elif self.config.shape == PerceptionShape.CONE:
            # Calculate angle from vehicle heading
            angle = np.arctan2(rel_pos[1], rel_pos[0]) - vehicle_heading
            angle = np.abs(np.arctan2(np.sin(angle), np.cos(angle)))  # Normalize to [-pi, pi]
            return angle <= self.config.angle
        
        elif self.config.shape == PerceptionShape.FORWARD_CONE:
            angle = np.arctan2(rel_pos[1], rel_pos[0]) - vehicle_heading
            angle = np.abs(np.arctan2(np.sin(angle), np.cos(angle)))
            return angle <= self.config.angle and rel_pos[0] >= 0  # Only forward
        
        elif self.config.shape == PerceptionShape.BACKWARD_CONE:
            angle = np.arctan2(rel_pos[1], rel_pos[0]) - vehicle_heading
            angle = np.abs(np.arctan2(np.sin(angle), np.cos(angle)))
            return angle <= self.config.angle and rel_pos[0] <= 0  # Only backward
        
        elif self.config.shape == PerceptionShape.RECTANGLE:
            # Transform to vehicle frame
            cos_heading = np.cos(vehicle_heading)
            sin_heading = np.sin(vehicle_heading)
            x_local = rel_pos[0] * cos_heading + rel_pos[1] * sin_heading
            y_local = -rel_pos[0] * sin_heading + rel_pos[1] * cos_heading
            
            return (abs(x_local) <= self.config.width/2 and 
                    abs(y_local) <= self.config.height/2)
        
        return True
    
    def get_visualization_patch(self, vehicle_pos: np.ndarray, vehicle_heading: float) -> patches.Patch:
        """Get matplotlib patch for visualization."""
        if not self.enabled:
            return None
        
        if self.config.shape == PerceptionShape.CIRCLE:
            return patches.Circle(vehicle_pos, self.config.distance, 
                                fill=False, linestyle='--', alpha=0.5, color='blue')
        
        elif self.config.shape == PerceptionShape.CONE:
            # Create cone patch
            angle_range = np.linspace(-self.config.angle, self.config.angle, 20)
            x_points = [vehicle_pos[0]]
            y_points = [vehicle_pos[1]]
            
            for angle in angle_range:
                x = vehicle_pos[0] + self.config.distance * np.cos(vehicle_heading + angle)
                y = vehicle_pos[1] + self.config.distance * np.sin(vehicle_heading + angle)
                x_points.append(x)
                y_points.append(y)
            
            return patches.Polygon(list(zip(x_points, y_points)), 
                                 fill=False, linestyle='--', alpha=0.5, color='blue')
        
        elif self.config.shape == PerceptionShape.RECTANGLE:
            # Create rectangle in vehicle frame
            cos_heading = np.cos(vehicle_heading)
            sin_heading = np.sin(vehicle_heading)
            
            # Rectangle corners in local frame
            corners_local = np.array([
                [-self.config.width/2, -self.config.height/2],
                [self.config.width/2, -self.config.height/2],
                [self.config.width/2, self.config.height/2],
                [-self.config.width/2, self.config.height/2]
            ])
            
            # Transform to global frame
            corners_global = []
            for corner in corners_local:
                x_global = vehicle_pos[0] + corner[0] * cos_heading - corner[1] * sin_heading
                y_global = vehicle_pos[1] + corner[0] * sin_heading + corner[1] * cos_heading
                corners_global.append([x_global, y_global])
            
            return patches.Polygon(corners_global, fill=False, linestyle='--', 
                                 alpha=0.5, color='blue')
        
        return None


class DynamicObstacle:
    """Dynamic obstacle with configurable trajectory."""
    
    def __init__(self, config: ObstacleConfig, road_center: np.ndarray, 
                 road_bounds: Tuple[np.ndarray, np.ndarray]):
        self.config = config
        self.road_center = road_center
        self.road_bounds = road_bounds
        
        # Initialize obstacle properties
        self.radius = np.random.uniform(*config.radius_range)
        self.velocity = np.random.uniform(*config.velocity_range)
        self.position = np.array([0.0, 0.0])  # Initialize position
        
        # Initialize position (random along road or intersecting)
        self._initialize_position()
        
        # Initialize trajectory parameters
        self._initialize_trajectory()
    
    def _initialize_position(self):
        """Initialize obstacle position."""
        if np.random.random() < self.config.intersection_probability:
            # Place obstacle to intersect with road
            road_length = len(self.road_center)
            intersection_point = np.random.randint(0, road_length)
            road_pos = self.road_center[intersection_point]
            
            # Add some offset to ensure intersection
            offset_angle = np.random.uniform(0, 2*np.pi)
            offset_distance = np.random.uniform(0, 5.0)
            self.position = road_pos + offset_distance * np.array([
                np.cos(offset_angle), np.sin(offset_angle)
            ])
        else:
            # Place obstacle randomly in area
            x_range = (self.road_center[:, 0].min() - 20, self.road_center[:, 0].max() + 20)
            y_range = (self.road_center[:, 1].min() - 20, self.road_center[:, 1].max() + 20)
            self.position = np.array([
                np.random.uniform(*x_range),
                np.random.uniform(*y_range)
            ])
    
    def _initialize_trajectory(self):
        """Initialize trajectory parameters."""
        if self.config.trajectory_type == "random_walk":
            self.direction = np.random.uniform(0, 2*np.pi)
            self.direction_change_prob = 0.1
        elif self.config.trajectory_type == "linear":
            self.direction = np.random.uniform(0, 2*np.pi)
            self.direction_change_prob = 0.0
        elif self.config.trajectory_type == "circular":
            self.center = self.position.copy()
            self.radius = np.random.uniform(5, 15)
            self.angular_velocity = np.random.uniform(-0.1, 0.1)
            self.angle = 0
    
    def update(self, dt: float):
        """Update obstacle position."""
        if self.config.trajectory_type == "random_walk":
            # Random walk with occasional direction changes
            if np.random.random() < self.direction_change_prob:
                self.direction += np.random.uniform(-np.pi/4, np.pi/4)
            
            # Add some noise to velocity
            velocity_noise = np.random.normal(0, 0.5)
            current_velocity = self.velocity + velocity_noise
            
            # Update position
            self.position += current_velocity * dt * np.array([
                np.cos(self.direction), np.sin(self.direction)
            ])
        
        elif self.config.trajectory_type == "linear":
            # Linear motion
            self.position += self.velocity * dt * np.array([
                np.cos(self.direction), np.sin(self.direction)
            ])
        
        elif self.config.trajectory_type == "circular":
            # Circular motion
            self.angle += self.angular_velocity * dt
            self.position = self.center + self.radius * np.array([
                np.cos(self.angle), np.sin(self.angle)
            ])


class MPCCTestFramework:
    """Standardized MPCC testing framework."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.perception_area = PerceptionArea(config.perception)
        
        # Initialize dynamics model
        if config.vehicle.dynamics_model == "bicycle":
            self.dynamics = BicycleModel(
                dt=config.mpc.timestep, 
                wheelbase=config.vehicle.wheelbase
            )
        else:
            raise ValueError(f"Unsupported dynamics model: {config.vehicle.dynamics_model}")
        
        # Initialize road
        self.road_center, self.road_left, self.road_right = self._generate_road()
        
        # Initialize obstacles
        self.obstacles = self._generate_obstacles()
        
        # Initialize vehicle state
        self.vehicle_state = np.array([0.0, 0.0, 0.0, 0.0])  # [x, y, psi, v]
        
        # Results storage
        self.trajectory = []
        self.obstacle_trajectories = {i: [] for i in range(len(self.obstacles))}
        self.constraint_violations = []
        self.perception_history = []
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _generate_road(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate curved road."""
        s = np.linspace(0, 1, self.config.road.num_points)
        x = s * self.config.road.length
        
        if self.config.road.road_type == "curved":
            # S-curve road
            y = (8 * np.sin(2 * np.pi * s * 0.3) + 
                 4 * np.sin(2 * np.pi * s * 0.6) + 
                 2 * np.sin(2 * np.pi * s * 1.2)) * self.config.road.curvature_intensity
        elif self.config.road.road_type == "s_curve":
            # More complex S-curve
            y = (10 * np.sin(2 * np.pi * s * 0.2) + 
                 5 * np.sin(2 * np.pi * s * 0.4) + 
                 2.5 * np.sin(2 * np.pi * s * 0.8))
        else:  # straight
            y = np.zeros_like(x)
        
        # Create road boundaries
        half_width = self.config.road.width / 2
        
        # Calculate road normals for proper boundary placement
        dx = np.gradient(x)
        dy = np.gradient(y)
        norm = np.sqrt(dx**2 + dy**2)
        nx = -dy / norm
        ny = dx / norm
        
        left_boundary = np.column_stack([x - half_width * nx, y - half_width * ny])
        right_boundary = np.column_stack([x + half_width * nx, y + half_width * ny])
        center_line = np.column_stack([x, y])
        
        return center_line, left_boundary, right_boundary
    
    def _generate_obstacles(self) -> List[DynamicObstacle]:
        """Generate dynamic obstacles."""
        obstacles = []
        for i in range(self.config.obstacles.num_obstacles):
            obstacle = DynamicObstacle(
                self.config.obstacles, 
                self.road_center, 
                (self.road_left, self.road_right)
            )
            obstacles.append(obstacle)
        return obstacles
    
    def _apply_mpcc_control(self, current_state: np.ndarray, visible_obstacles: List[DynamicObstacle]) -> np.ndarray:
        """Apply MPCC control based on contouring objective and constraints."""
        x, y, psi, v = current_state
        
        # Calculate progress along path (s parameter)
        s = self._calculate_path_progress(x, y)
        
        # Calculate contouring error (lateral deviation from path)
        contouring_error = self._calculate_contouring_error(x, y, s)
        
        # Calculate lag error (longitudinal deviation from path)
        lag_error = self._calculate_lag_error(x, y, s)
        
        # MPCC control law (simplified)
        # Steering control for contouring
        k_contour = 2.0  # Contouring gain
        k_lag = 1.0     # Lag gain
        
        steering_angle = k_contour * contouring_error + k_lag * lag_error
        steering_angle = np.clip(steering_angle, -self.config.vehicle.max_steering_angle, 
                                self.config.vehicle.max_steering_angle)
        
        # Velocity control
        target_velocity = self.config.vehicle.max_velocity * 0.8  # 80% of max speed
        velocity_error = target_velocity - v
        k_vel = 1.0
        acceleration = k_vel * velocity_error
        acceleration = np.clip(acceleration, -self.config.vehicle.max_acceleration, 
                              self.config.vehicle.max_acceleration)
        
        # Apply obstacle avoidance constraints
        steering_angle, acceleration = self._apply_obstacle_constraints(
            current_state, steering_angle, acceleration, visible_obstacles
        )
        
        return np.array([acceleration, steering_angle])
    
    def _calculate_path_progress(self, x: float, y: float) -> float:
        """Calculate progress along the reference path (s parameter)."""
        # Find closest point on road center
        distances = np.linalg.norm(self.road_center - np.array([x, y]), axis=1)
        closest_idx = np.argmin(distances)
        return closest_idx / len(self.road_center)
    
    def _calculate_contouring_error(self, x: float, y: float, s: float) -> float:
        """Calculate contouring error (lateral deviation from path)."""
        # Get path point at progress s
        idx = int(s * (len(self.road_center) - 1))
        idx = np.clip(idx, 0, len(self.road_center) - 1)
        path_point = self.road_center[idx]
        
        # Calculate path tangent
        if idx < len(self.road_center) - 1:
            path_tangent = self.road_center[idx + 1] - path_point
        else:
            path_tangent = path_point - self.road_center[idx - 1]
        
        path_angle = np.arctan2(path_tangent[1], path_tangent[0])
        
        # Calculate lateral deviation
        rel_pos = np.array([x, y]) - path_point
        lateral_deviation = -rel_pos[0] * np.sin(path_angle) + rel_pos[1] * np.cos(path_angle)
        
        return lateral_deviation
    
    def _calculate_lag_error(self, x: float, y: float, s: float) -> float:
        """Calculate lag error (longitudinal deviation from path)."""
        # Get path point at progress s
        idx = int(s * (len(self.road_center) - 1))
        idx = np.clip(idx, 0, len(self.road_center) - 1)
        path_point = self.road_center[idx]
        
        # Calculate path tangent
        if idx < len(self.road_center) - 1:
            path_tangent = self.road_center[idx + 1] - path_point
        else:
            path_tangent = path_point - self.road_center[idx - 1]
        
        path_angle = np.arctan2(path_tangent[1], path_tangent[0])
        
        # Calculate longitudinal deviation
        rel_pos = np.array([x, y]) - path_point
        longitudinal_deviation = rel_pos[0] * np.cos(path_angle) + rel_pos[1] * np.sin(path_angle)
        
        return longitudinal_deviation
    
    def _apply_obstacle_constraints(self, state: np.ndarray, steering: float, 
                                  acceleration: float, obstacles: List[DynamicObstacle]) -> Tuple[float, float]:
        """Apply obstacle avoidance constraints."""
        x, y, _, _ = state
        
        for obstacle in obstacles:
            # Calculate distance to obstacle
            obstacle_pos = obstacle.position
            distance = np.linalg.norm(np.array([x, y]) - obstacle_pos)
            
            # If too close, apply avoidance
            if distance < obstacle.radius + self.config.vehicle.length/2 + 2.0:  # Safety margin
                # Calculate avoidance direction
                rel_pos = np.array([x, y]) - obstacle_pos
                avoidance_angle = np.arctan2(rel_pos[1], rel_pos[0])
                
                # Adjust steering for avoidance
                steering_adjustment = 0.3 * np.sin(avoidance_angle - psi)
                steering += steering_adjustment
                
                # Reduce speed when near obstacles
                acceleration *= 0.5
        
        return steering, acceleration
    
    def run_test(self) -> Dict[str, Any]:
        """Run the complete MPCC test."""
        print(f"Running MPCC test: {self.config.test_name}")
        print(f"Road type: {self.config.road.road_type}")
        print(f"Obstacles: {len(self.obstacles)}")
        print(f"Perception: {self.config.perception.shape.value if self.config.perception.enabled else 'disabled'}")
        
        start_time = time.time()
        
        for step in range(self.config.mpc.max_steps):
        # Update obstacles
        for obstacle in self.obstacles:
            obstacle.update(self.config.mpc.timestep)
        self.obstacle_trajectories[step] = [obs.position.copy() for obs in self.obstacles]
            
            # Check which obstacles are visible
            visible_obstacles = []
            for obstacle in self.obstacles:
                if self.perception_area.is_obstacle_visible(
                    self.vehicle_state[:2], self.vehicle_state[2], obstacle.position
                ):
                    visible_obstacles.append(obstacle)
            
            # Apply MPCC control
            control = self._apply_mpcc_control(self.vehicle_state, visible_obstacles)
            
            # Update vehicle state
            next_state = self.dynamics.step(self.vehicle_state, control)
            self.vehicle_state = next_state
            
            # Store trajectory
            self.trajectory.append(self.vehicle_state.copy())
            
            # Check for constraint violations
            self._check_constraint_violations()
            
            # Store perception data
            self.perception_history.append({
                'step': step,
                'visible_obstacles': len(visible_obstacles),
                'total_obstacles': len(self.obstacles)
            })
            
            # Check if goal reached
            if self._is_goal_reached():
                print(f"Goal reached at step {step}")
                break
        
        duration = time.time() - start_time
        
        # Generate results
        test_results = {
            'test_name': self.config.test_name,
            'duration': duration,
            'steps_completed': len(self.trajectory),
            'success': self._is_goal_reached(),
            'trajectory': np.array(self.trajectory),
            'obstacle_trajectories': self.obstacle_trajectories,
            'constraint_violations': self.constraint_violations,
            'perception_history': self.perception_history,
            'config': self.config
        }
        
        # Save results
        if self.config.save_data:
            self._save_results(test_results)
        
        # Generate visualization
        if self.config.generate_gif:
            self._generate_visualization(test_results)
        
        return test_results
    
    def _check_constraint_violations(self):
        """Check for constraint violations."""
        x, y, _, _ = self.vehicle_state
        
        # Check road boundary violations
        if self._is_outside_road(x, y):
            self.constraint_violations.append({
                'step': len(self.trajectory),
                'type': 'road_boundary',
                'position': [x, y]
            })
        
        # Check obstacle collisions
        for obstacle_id, obstacle in enumerate(self.obstacles):
            distance = np.linalg.norm(np.array([x, y]) - obstacle.position)
            if distance < obstacle.radius + self.config.vehicle.length/2:
                self.constraint_violations.append({
                    'step': len(self.trajectory),
                    'type': 'obstacle_collision',
                    'obstacle_id': obstacle_id,
                    'position': [x, y]
                })
    
    def _is_outside_road(self, x: float, y: float) -> bool:
        """Check if position is outside road boundaries."""
        # Find closest point on road
        distances = np.linalg.norm(self.road_center - np.array([x, y]), axis=1)
        closest_idx = np.argmin(distances)
        
        # Check distance to road boundaries
        dist_to_left = np.linalg.norm(np.array([x, y]) - self.road_left[closest_idx])
        dist_to_right = np.linalg.norm(np.array([x, y]) - self.road_right[closest_idx])
        
        # If closer to boundaries than center, might be outside
        min_dist_to_boundary = min(dist_to_left, dist_to_right)
        return min_dist_to_boundary > self.config.road.width / 2 + self.config.vehicle.width / 2
    
    def _is_goal_reached(self) -> bool:
        """Check if goal is reached."""
        if len(self.trajectory) < 2:
            return False
        
        # Check if vehicle has traveled most of the road
        total_distance = 0
        for i in range(1, len(self.trajectory)):
            pos1 = self.trajectory[i-1][:2]
            pos2 = self.trajectory[i][:2]
            total_distance += np.linalg.norm(pos2 - pos1)
        
        road_length = self.config.road.length
        return total_distance >= road_length * 0.8  # 80% of road length
    
    def _save_results(self, test_results: Dict[str, Any]):
        """Save test results to file."""
        # Convert numpy arrays to lists for JSON serialization
        results_copy = test_results.copy()
        results_copy['trajectory'] = test_results['trajectory'].tolist()
        
        # Save obstacle trajectories
        for step, obs_traj in test_results['obstacle_trajectories'].items():
            if isinstance(obs_traj, list):
                results_copy['obstacle_trajectories'][step] = [pos.tolist() for pos in obs_traj]
        
        # Save to JSON
        results_file = os.path.join(self.config.output_dir, f"{self.config.test_name}_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_copy, f, indent=2)
    
    def _generate_visualization(self, test_results: Dict[str, Any]):
        """Generate animated visualization of the test."""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Plot road
        ax.plot(self.road_center[:, 0], self.road_center[:, 1], 'k-', linewidth=2, label='Road Center')
        ax.plot(self.road_left[:, 0], self.road_left[:, 1], 'k--', linewidth=1, label='Road Boundaries')
        ax.plot(self.road_right[:, 0], self.road_right[:, 1], 'k--', linewidth=1)
        
        # Initialize vehicle and obstacle plots
        vehicle_plot, = ax.plot([], [], 'ro', markersize=8, label='Vehicle')
        obstacle_plots = []
        for _ in range(len(self.obstacles)):
            plot, = ax.plot([], [], 'bo', markersize=6, label='Obstacle')
            obstacle_plots.append(plot)
        
        def animate(frame):
            if frame >= len(self.trajectory):
                return vehicle_plot, *obstacle_plots
            
            # Update vehicle position
            vehicle_pos = self.trajectory[frame][:2]
            vehicle_plot.set_data([vehicle_pos[0]], [vehicle_pos[1]])
            
            # Update obstacle positions
            if frame in self.obstacle_trajectories:
                obs_positions = self.obstacle_trajectories[frame]
                for i, (plot, pos) in enumerate(zip(obstacle_plots, obs_positions)):
                    plot.set_data([pos[0]], [pos[1]])
            
            # Update perception area
            if self.config.perception.enabled:
                vehicle_heading = self.trajectory[frame][2]
                patch = self.perception_area.get_visualization_patch(vehicle_pos, vehicle_heading)
                if patch is not None:
                    # Remove old patch
                    for p in ax.patches:
                        if hasattr(p, '_perception_patch'):
                            p.remove()
                    # Add new patch
                    setattr(patch, '_perception_patch', True)
                    ax.add_patch(patch)
            
            # Add constraint overlays
            self._add_constraint_overlays(ax, frame)
            
            return vehicle_plot, *obstacle_plots
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(self.trajectory), 
                           interval=1000/self.config.gif_fps, blit=False, repeat=True)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'MPCC Test: {self.config.test_name}')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')
        
        # Save animation
        gif_file = os.path.join(self.config.output_dir, f"{self.config.test_name}.gif")
        anim.save(gif_file, writer='pillow', fps=self.config.gif_fps)
        
        print(f"Animation saved to: {gif_file}")
    
    def _add_constraint_overlays(self, ax, frame):
        """Add constraint overlays to visualization."""
        if frame >= len(self.trajectory):
            return
        
        # Add obstacle constraint circles
        if frame in self.obstacle_trajectories:
            for obstacle_id, obs_pos in enumerate(self.obstacle_trajectories[frame]):
                obstacle = self.obstacles[obstacle_id]
                circle = patches.Circle(obs_pos, obstacle.radius, 
                                      fill=False, linestyle=':', alpha=0.7, color='red')
                ax.add_patch(circle)


def create_standard_mpcc_test(test_name: str = "standard_mpcc", **kwargs) -> MPCCTestFramework:
    """Create a standard MPCC test with default configuration."""
    config = TestConfig(test_name=test_name, **kwargs)
    return MPCCTestFramework(config)


def create_curved_road_test(test_name: str = "curved_road_mpcc", **kwargs) -> MPCCTestFramework:
    """Create MPCC test with curved road."""
    config = TestConfig(
        test_name=test_name,
        road=RoadConfig(road_type="curved", curvature_intensity=1.5),
        obstacles=ObstacleConfig(num_obstacles=4, intersection_probability=0.8),
        **kwargs
    )
    return MPCCTestFramework(config)


def create_perception_test(test_name: str = "perception_mpcc", **kwargs) -> MPCCTestFramework:
    """Create MPCC test with perception limitations."""
    config = TestConfig(
        test_name=test_name,
        perception=PerceptionConfig(
            shape=PerceptionShape.CONE,
            distance=15.0,
            angle=np.pi/4,
            enabled=True
        ),
        obstacles=ObstacleConfig(num_obstacles=5, intersection_probability=0.9),
        **kwargs
    )
    return MPCCTestFramework(config)


if __name__ == "__main__":
    # Example usage
    test = create_standard_mpcc_test("example_test")
    results = test.run_test()
    print(f"Test completed: {results['success']}")
    print(f"Duration: {results['duration']:.2f}s")
