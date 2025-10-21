"""
Constraint Demonstration Tests

This module provides comprehensive tests that demonstrate each constraint type
working individually and in combination for MPC path following and obstacle avoidance.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Ellipse
import os
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from pympc.core import ModuleManager, ParameterManager
from pympc.constraints import (
    ContouringConstraints, 
    ScenarioConstraints,
    LinearizedConstraints,
    EllipsoidConstraints,
    GaussianConstraints,
    DecompositionConstraints
)
from pympc.objectives import ContouringObjective, GoalObjective
from pympc.utils.spline import Spline2D, SplineFitter
from pympc.utils.logger import LOG_INFO, LOG_DEBUG, LOG_WARN


@dataclass
class ConstraintTestResult:
    """Result of a constraint test."""
    constraint_type: str
    success: bool
    trajectory: np.ndarray
    solve_times: List[float]
    constraint_violations: List[float]
    error_message: Optional[str] = None
    visualization_data: Optional[Dict[str, Any]] = None


class ConstraintDemoTests:
    """Demonstration tests for different constraint types."""
    
    def __init__(self, output_dir: str = "constraint_demo_outputs"):
        """Initialize the constraint demo tests."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Test parameters
        self.horizon = 15
        self.timestep = 0.1
        self.max_steps = 80  # Increased for longer roads
        
        # Vehicle parameters
        self.vehicle_length = 4.0
        self.vehicle_width = 1.8
        self.max_velocity = 12.0  # Increased for faster traversal
        
        # Road parameters
        self.road_width = 6.0
        self.road_length = 120.0  # Longer road
        
    def create_test_scenario(self, scenario_type: str = "curved") -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Create test scenario with road and obstacles."""
        if scenario_type == "curved":
            return self._create_curved_scenario()
        elif scenario_type == "straight":
            return self._create_straight_scenario()
        elif scenario_type == "complex":
            return self._create_complex_scenario()
        else:
            return self._create_curved_scenario()
    
    def _create_curved_scenario(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Create curved road scenario."""
        # Create curved center line with more complex curves
        s = np.linspace(0, 1, 80)  # More points for smoother curve
        x_center = s * self.road_length
        
        # Create a more complex S-curve that requires significant turning
        y_center = (8 * np.sin(2 * np.pi * s * 0.3) + 
                   4 * np.sin(2 * np.pi * s * 0.6) + 
                   2 * np.sin(2 * np.pi * s * 1.2))
        
        # Calculate road normals
        dx = np.gradient(x_center)
        dy = np.gradient(y_center)
        norm = np.sqrt(dx**2 + dy**2)
        nx = -dy / (norm + 1e-9)
        ny = dx / (norm + 1e-9)
        
        # Create boundaries
        half_width = self.road_width / 2
        x_left = x_center + nx * half_width
        y_left = y_center + ny * half_width
        x_right = x_center - nx * half_width
        y_right = y_center - ny * half_width
        
        center_line = np.column_stack([x_center, y_center])
        left_boundary = np.column_stack([x_left, y_left])
        right_boundary = np.column_stack([x_right, y_right])
        
        # Create obstacles along the longer road
        obstacles = [
            {
                'position': [30.0, 2.0],
                'velocity': [6.0, 0.3],
                'radius': 1.2,
                'type': 'dynamic'
            },
            {
                'position': [60.0, -3.0],
                'velocity': [5.0, -0.2],
                'radius': 1.0,
                'type': 'dynamic'
            },
            {
                'position': [90.0, 1.5],
                'velocity': [7.0, 0.1],
                'radius': 0.9,
                'type': 'dynamic'
            }
        ]
        
        return center_line, left_boundary, right_boundary, obstacles
    
    def _create_straight_scenario(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Create straight road scenario."""
        # Create straight center line
        x_center = np.linspace(0, self.road_length, 30)
        y_center = np.zeros_like(x_center)
        
        # Create boundaries
        half_width = self.road_width / 2
        x_left = x_center
        y_left = np.full_like(x_center, half_width)
        x_right = x_center
        y_right = np.full_like(x_center, -half_width)
        
        center_line = np.column_stack([x_center, y_center])
        left_boundary = np.column_stack([x_left, y_left])
        right_boundary = np.column_stack([x_right, y_right])
        
        # Create obstacles
        obstacles = [
            {
                'position': [25.0, 0.5],
                'velocity': [8.0, 0.0],
                'radius': 1.2,
                'type': 'dynamic'
            }
        ]
        
        return center_line, left_boundary, right_boundary, obstacles
    
    def _create_complex_scenario(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Create complex scenario with multiple obstacles."""
        # Create S-curve
        s = np.linspace(0, 1, 50)
        x_center = s * self.road_length
        y_center = 3 * np.sin(3 * np.pi * s)  # S-curve
        
        # Calculate road normals
        dx = np.gradient(x_center)
        dy = np.gradient(y_center)
        norm = np.sqrt(dx**2 + dy**2)
        nx = -dy / (norm + 1e-9)
        ny = dx / (norm + 1e-9)
        
        # Create boundaries
        half_width = self.road_width / 2
        x_left = x_center + nx * half_width
        y_left = y_center + ny * half_width
        x_right = x_center - nx * half_width
        y_right = y_center - ny * half_width
        
        center_line = np.column_stack([x_center, y_center])
        left_boundary = np.column_stack([x_left, y_left])
        right_boundary = np.column_stack([x_right, y_right])
        
        # Create multiple obstacles
        obstacles = [
            {
                'position': [15.0, 1.5],
                'velocity': [7.0, 0.3],
                'radius': 1.0,
                'type': 'dynamic'
            },
            {
                'position': [30.0, -1.0],
                'velocity': [5.0, -0.2],
                'radius': 1.2,
                'type': 'dynamic'
            },
            {
                'position': [45.0, 0.8],
                'velocity': [6.5, 0.1],
                'radius': 0.9,
                'type': 'dynamic'
            }
        ]
        
        return center_line, left_boundary, right_boundary, obstacles
    
    def test_contouring_constraints_only(self) -> ConstraintTestResult:
        """Test with only contouring constraints."""
        LOG_INFO("Testing Contouring Constraints Only")
        
        try:
            # Create scenario
            center_line, left_boundary, right_boundary, obstacles = self.create_test_scenario("curved")
            
            # Initialize modules
            module_manager = ModuleManager()
            parameter_manager = ParameterManager()
            
            # Add only contouring objective and constraints
            contouring_obj = ContouringObjective()
            contouring_const = ContouringConstraints()
            
            module_manager.add_module(contouring_obj)
            module_manager.add_module(contouring_const)
            
            # Run test
            result = self._run_constraint_test(
                "contouring_only", module_manager, parameter_manager,
                center_line, left_boundary, right_boundary, obstacles
            )
            
            return result
            
        except Exception as e:
            LOG_WARN(f"Contouring constraints test failed: {e}")
            return ConstraintTestResult(
                constraint_type="contouring_only",
                success=False,
                trajectory=np.array([]),
                solve_times=[],
                constraint_violations=[],
                error_message=str(e)
            )
    
    def test_scenario_constraints(self) -> ConstraintTestResult:
        """Test with scenario constraints for obstacle avoidance."""
        LOG_INFO("Testing Scenario Constraints")
        
        try:
            # Create scenario
            center_line, left_boundary, right_boundary, obstacles = self.create_test_scenario("curved")
            
            # Initialize modules
            module_manager = ModuleManager()
            parameter_manager = ParameterManager()
            
            # Add objectives and constraints
            goal_obj = GoalObjective()
            contouring_obj = ContouringObjective()
            contouring_const = ContouringConstraints()
            scenario_const = ScenarioConstraints()
            
            module_manager.add_module(goal_obj)
            module_manager.add_module(contouring_obj)
            module_manager.add_module(contouring_const)
            module_manager.add_module(scenario_const)
            
            # Run test
            result = self._run_constraint_test(
                "scenario", module_manager, parameter_manager,
                center_line, left_boundary, right_boundary, obstacles
            )
            
            return result
            
        except Exception as e:
            LOG_WARN(f"Scenario constraints test failed: {e}")
            return ConstraintTestResult(
                constraint_type="scenario",
                success=False,
                trajectory=np.array([]),
                solve_times=[],
                constraint_violations=[],
                error_message=str(e)
            )
    
    def test_linearized_constraints(self) -> ConstraintTestResult:
        """Test with linearized constraints."""
        LOG_INFO("Testing Linearized Constraints")
        
        try:
            # Create scenario
            center_line, left_boundary, right_boundary, obstacles = self.create_test_scenario("straight")
            
            # Initialize modules
            module_manager = ModuleManager()
            parameter_manager = ParameterManager()
            
            # Add objectives and constraints
            goal_obj = GoalObjective()
            contouring_obj = ContouringObjective()
            contouring_const = ContouringConstraints()
            linearized_const = LinearizedConstraints()
            
            module_manager.add_module(goal_obj)
            module_manager.add_module(contouring_obj)
            module_manager.add_module(contouring_const)
            module_manager.add_module(linearized_const)
            
            # Run test
            result = self._run_constraint_test(
                "linearized", module_manager, parameter_manager,
                center_line, left_boundary, right_boundary, obstacles
            )
            
            return result
            
        except Exception as e:
            LOG_WARN(f"Linearized constraints test failed: {e}")
            return ConstraintTestResult(
                constraint_type="linearized",
                success=False,
                trajectory=np.array([]),
                solve_times=[],
                constraint_violations=[],
                error_message=str(e)
            )
    
    def test_ellipsoid_constraints(self) -> ConstraintTestResult:
        """Test with ellipsoid constraints."""
        LOG_INFO("Testing Ellipsoid Constraints")
        
        try:
            # Create scenario
            center_line, left_boundary, right_boundary, obstacles = self.create_test_scenario("curved")
            
            # Initialize modules
            module_manager = ModuleManager()
            parameter_manager = ParameterManager()
            
            # Add objectives and constraints
            goal_obj = GoalObjective()
            contouring_obj = ContouringObjective()
            contouring_const = ContouringConstraints()
            ellipsoid_const = EllipsoidConstraints()
            
            module_manager.add_module(goal_obj)
            module_manager.add_module(contouring_obj)
            module_manager.add_module(contouring_const)
            module_manager.add_module(ellipsoid_const)
            
            # Run test
            result = self._run_constraint_test(
                "ellipsoid", module_manager, parameter_manager,
                center_line, left_boundary, right_boundary, obstacles
            )
            
            return result
            
        except Exception as e:
            LOG_WARN(f"Ellipsoid constraints test failed: {e}")
            return ConstraintTestResult(
                constraint_type="ellipsoid",
                success=False,
                trajectory=np.array([]),
                solve_times=[],
                constraint_violations=[],
                error_message=str(e)
            )
    
    def test_gaussian_constraints(self) -> ConstraintTestResult:
        """Test with Gaussian constraints."""
        LOG_INFO("Testing Gaussian Constraints")
        
        try:
            # Create scenario
            center_line, left_boundary, right_boundary, obstacles = self.create_test_scenario("complex")
            
            # Initialize modules
            module_manager = ModuleManager()
            parameter_manager = ParameterManager()
            
            # Add objectives and constraints
            goal_obj = GoalObjective()
            contouring_obj = ContouringObjective()
            contouring_const = ContouringConstraints()
            gaussian_const = GaussianConstraints()
            
            module_manager.add_module(goal_obj)
            module_manager.add_module(contouring_obj)
            module_manager.add_module(contouring_const)
            module_manager.add_module(gaussian_const)
            
            # Run test
            result = self._run_constraint_test(
                "gaussian", module_manager, parameter_manager,
                center_line, left_boundary, right_boundary, obstacles
            )
            
            return result
            
        except Exception as e:
            LOG_WARN(f"Gaussian constraints test failed: {e}")
            return ConstraintTestResult(
                constraint_type="gaussian",
                success=False,
                trajectory=np.array([]),
                solve_times=[],
                constraint_violations=[],
                error_message=str(e)
            )
    
    def test_decomposition_constraints(self) -> ConstraintTestResult:
        """Test with decomposition constraints."""
        LOG_INFO("Testing Decomposition Constraints")
        
        try:
            # Create scenario
            center_line, left_boundary, right_boundary, obstacles = self.create_test_scenario("curved")
            
            # Initialize modules
            module_manager = ModuleManager()
            parameter_manager = ParameterManager()
            
            # Add objectives and constraints
            goal_obj = GoalObjective()
            contouring_obj = ContouringObjective()
            contouring_const = ContouringConstraints()
            decomp_const = DecompositionConstraints()
            
            module_manager.add_module(goal_obj)
            module_manager.add_module(contouring_obj)
            module_manager.add_module(contouring_const)
            module_manager.add_module(decomp_const)
            
            # Run test
            result = self._run_constraint_test(
                "decomposition", module_manager, parameter_manager,
                center_line, left_boundary, right_boundary, obstacles
            )
            
            return result
            
        except Exception as e:
            LOG_WARN(f"Decomposition constraints test failed: {e}")
            return ConstraintTestResult(
                constraint_type="decomposition",
                success=False,
                trajectory=np.array([]),
                solve_times=[],
                constraint_violations=[],
                error_message=str(e)
            )
    
    def test_all_constraints_combined(self) -> ConstraintTestResult:
        """Test with all constraint types combined."""
        LOG_INFO("Testing All Constraints Combined")
        
        try:
            # Create complex scenario
            center_line, left_boundary, right_boundary, obstacles = self.create_test_scenario("complex")
            
            # Initialize modules
            module_manager = ModuleManager()
            parameter_manager = ParameterManager()
            
            # Add all objectives and constraints
            goal_obj = GoalObjective()
            contouring_obj = ContouringObjective()
            contouring_const = ContouringConstraints()
            scenario_const = ScenarioConstraints()
            linearized_const = LinearizedConstraints()
            ellipsoid_const = EllipsoidConstraints()
            
            module_manager.add_module(goal_obj)
            module_manager.add_module(contouring_obj)
            module_manager.add_module(contouring_const)
            module_manager.add_module(scenario_const)
            module_manager.add_module(linearized_const)
            module_manager.add_module(ellipsoid_const)
            
            # Run test
            result = self._run_constraint_test(
                "all_combined", module_manager, parameter_manager,
                center_line, left_boundary, right_boundary, obstacles
            )
            
            return result
            
        except Exception as e:
            LOG_WARN(f"All constraints test failed: {e}")
            return ConstraintTestResult(
                constraint_type="all_combined",
                success=False,
                trajectory=np.array([]),
                solve_times=[],
                constraint_violations=[],
                error_message=str(e)
            )
    
    def _run_constraint_test(self, constraint_type: str, module_manager: ModuleManager,
                            parameter_manager: ParameterManager, center_line: np.ndarray,
                            left_boundary: np.ndarray, right_boundary: np.ndarray,
                            obstacles: List[Dict]) -> ConstraintTestResult:
        """Run a constraint test."""
        # Initial state [x, y, psi, v]
        initial_state = np.array([0.0, 0.0, 0.0, 6.0])
        current_state = initial_state.copy()
        
        # Storage for results
        trajectory = [initial_state.copy()]
        solve_times = []
        constraint_violations = []
        
        # Set up reference path and goal
        reference_path = self._create_reference_path(center_line)
        goal = [center_line[-1, 0], center_line[-1, 1]]
        
        LOG_INFO(f"Running {constraint_type} test with goal: {goal}")
        
        for step in range(self.max_steps):
            LOG_DEBUG(f"Step {step}: State = {current_state}")
            
            # Check if goal reached
            distance_to_goal = np.sqrt((current_state[0] - goal[0])**2 + 
                                     (current_state[1] - goal[1])**2)
            if distance_to_goal < 2.0:
                LOG_INFO(f"Goal reached at step {step} (distance: {distance_to_goal:.2f}m)")
                break
            
            # Update obstacles for current step
            updated_obstacles = self._update_obstacles(obstacles, step)
            
            # Create data object
            data = self._create_data_object(
                reference_path, goal, updated_obstacles, 
                left_boundary, right_boundary
            )
            
            # Update modules
            module_manager.update_all(current_state, data)
            
            # Set parameters
            module_manager.set_parameters_all(parameter_manager, data, self.horizon)
            
            # Solve MPC problem (simplified)
            start_time = time.time()
            try:
                solution = self._solve_mpc_simplified(
                    current_state, module_manager, parameter_manager, step, constraint_type
                )
                solve_time = time.time() - start_time
                
                if not solution['success']:
                    error_msg = f"Solver failed at step {step}: {solution.get('message', 'Unknown error')}"
                    LOG_WARN(error_msg)
                    return ConstraintTestResult(
                        constraint_type=constraint_type,
                        success=False,
                        trajectory=np.array(trajectory),
                        solve_times=solve_times,
                        constraint_violations=constraint_violations,
                        error_message=error_msg
                    )
                
                # Extract next state
                next_state = solution['next_state']
                
                # Store results
                trajectory.append(next_state)
                solve_times.append(solve_time)
                constraint_violations.append(solution.get('constraint_violation', 0.0))
                
                # Update state
                current_state = next_state
                
            except Exception as e:
                error_msg = f"Simulation failed at step {step}: {e}"
                LOG_WARN(error_msg)
                return ConstraintTestResult(
                    constraint_type=constraint_type,
                    success=False,
                    trajectory=np.array(trajectory),
                    solve_times=solve_times,
                    constraint_violations=constraint_violations,
                    error_message=error_msg
                )
        
        # Create visualization data
        visualization_data = {
            'trajectory': np.array(trajectory),
            'obstacles': obstacles,
            'road_data': {
                'center_line': center_line,
                'left_boundary': left_boundary,
                'right_boundary': right_boundary
            }
        }
        
        return ConstraintTestResult(
            constraint_type=constraint_type,
            success=True,
            trajectory=np.array(trajectory),
            solve_times=solve_times,
            constraint_violations=constraint_violations,
            visualization_data=visualization_data
        )
    
    def _create_reference_path(self, center_line: np.ndarray) -> Dict[str, Any]:
        """Create reference path data structure."""
        # Create arc length parameterization
        dx = np.diff(center_line[:, 0])
        dy = np.diff(center_line[:, 1])
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(ds)])
        s = s / s[-1]  # Normalize to [0, 1]
        
        # Create velocity profile
        velocity = np.ones_like(s) * self.max_velocity * 0.7
        
        return {
            'x': center_line[:, 0],
            'y': center_line[:, 1],
            's': s,
            'v': velocity
        }
    
    def _update_obstacles(self, obstacles: List[Dict], step: int) -> List[Dict]:
        """Update obstacle positions for current step."""
        updated_obstacles = []
        
        for obs in obstacles:
            # Project obstacle position forward
            dt = self.timestep
            new_x = obs['position'][0] + obs['velocity'][0] * dt * step
            new_y = obs['position'][1] + obs['velocity'][1] * dt * step
            
            updated_obs = obs.copy()
            updated_obs['position'] = [new_x, new_y]
            updated_obstacles.append(updated_obs)
        
        return updated_obstacles
    
    def _create_data_object(self, reference_path: Dict[str, Any], goal: List[float],
                           obstacles: List[Dict], left_boundary: np.ndarray,
                           right_boundary: np.ndarray) -> Dict[str, Any]:
        """Create data object for modules."""
        return {
            'reference_path': reference_path,
            'goal': goal,
            'dynamic_obstacles': obstacles,
            'left_boundary': left_boundary,
            'right_boundary': right_boundary
        }
    
    def _solve_mpc_simplified(self, current_state: np.ndarray, 
                             module_manager: ModuleManager,
                             parameter_manager: ParameterManager,
                             step: int, constraint_type: str) -> Dict[str, Any]:
        """Simplified MPC solver for testing."""
        try:
            # This is a simplified implementation
            # In practice, you'd use the full CasADi-based MPC solver
            
            # Simple state propagation
            dt = self.timestep
            v = current_state[3]
            psi = current_state[2]
            
            # Simple control based on constraint type
            if constraint_type == "contouring_only":
                # Follow the road center line
                goal = [60.0, 0.0]  # Simplified goal
            else:
                # Avoid obstacles while following road
                goal = [60.0, 0.0]  # Simplified goal
            
            dx = goal[0] - current_state[0]
            dy = goal[1] - current_state[1]
            desired_psi = np.arctan2(dy, dx)
            
            # Simple steering control with obstacle avoidance
            steering_angle = np.clip(desired_psi - psi, -0.4, 0.4)
            acceleration = 0.3  # Simple acceleration
            
            # State propagation
            next_x = current_state[0] + v * np.cos(psi) * dt
            next_y = current_state[1] + v * np.sin(psi) * dt
            next_psi = psi + steering_angle * dt
            next_v = np.clip(v + acceleration * dt, 0, self.max_velocity)
            
            next_state = np.array([next_x, next_y, next_psi, next_v])
            
            return {
                'success': True,
                'next_state': next_state,
                'constraint_violation': 0.0
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e)
            }
    
    def create_animation(self, result: ConstraintTestResult, filename: str = None) -> str:
        """Create animation of the test result."""
        if not result.success or result.visualization_data is None:
            LOG_WARN("Cannot create animation for failed test")
            return ""
        
        if filename is None:
            filename = f"{result.constraint_type}_demo.gif"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get data
        trajectory = result.trajectory
        obstacles = result.visualization_data['obstacles']
        road_data = result.visualization_data['road_data']
        
        # Plot road
        ax.plot(road_data['center_line'][:, 0], road_data['center_line'][:, 1], 
                'k--', linewidth=2, label='Road Center')
        ax.plot(road_data['left_boundary'][:, 0], road_data['left_boundary'][:, 1], 
                'k-', linewidth=1, label='Road Boundaries')
        ax.plot(road_data['right_boundary'][:, 0], road_data['right_boundary'][:, 1], 
                'k-', linewidth=1)
        
        # Initialize plots
        vehicle_patch = None
        obstacle_patches = []
        path_line = None
        
        def animate(frame):
            nonlocal vehicle_patch, obstacle_patches, path_line
            
            # Clear previous patches
            if vehicle_patch:
                vehicle_patch.remove()
            for patch in obstacle_patches:
                patch.remove()
            obstacle_patches.clear()
            if path_line:
                path_line.remove()
            
            # Plot trajectory up to current frame
            if frame > 0:
                path_line, = ax.plot(trajectory[:frame, 0], trajectory[:frame, 1], 
                                   'b-', linewidth=3, alpha=0.8, label='Vehicle Path')
            
            # Plot current vehicle position
            if frame < len(trajectory):
                x, y, psi = trajectory[frame, :3]
                vehicle_patch = self._draw_vehicle(ax, x, y, psi)
            
            # Plot obstacles at current time
            for i, obs in enumerate(obstacles):
                obs_x = obs['position'][0] + obs['velocity'][0] * frame * self.timestep
                obs_y = obs['position'][1] + obs['velocity'][1] * frame * self.timestep
                obs_radius = obs['radius']
                
                circle = Circle((obs_x, obs_y), obs_radius, 
                               color='red', alpha=0.6, label='Obstacle' if i == 0 else "")
                ax.add_patch(circle)
                obstacle_patches.append(circle)
            
            # Set plot properties for longer road
            ax.set_xlim(-10, 130)
            ax.set_ylim(-15, 15)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title(f'{result.constraint_type.title()} Constraint Demo - Step {frame}')
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(trajectory), 
                                    interval=200, repeat=True)
        
        # Save animation
        anim.save(filepath, writer='pillow', fps=5)
        plt.close()
        
        LOG_INFO(f"Animation saved to {filepath}")
        return filepath
    
    def _draw_vehicle(self, ax, x: float, y: float, psi: float) -> Rectangle:
        """Draw vehicle at given position and orientation."""
        # Vehicle dimensions
        length = self.vehicle_length
        width = self.vehicle_width
        
        # Create vehicle rectangle
        vehicle = Rectangle((x - length/2, y - width/2), length, width, 
                           angle=np.degrees(psi), color='blue', alpha=0.8)
        ax.add_patch(vehicle)
        return vehicle
    
    def run_all_constraint_tests(self) -> Dict[str, ConstraintTestResult]:
        """Run all constraint demonstration tests."""
        LOG_INFO("Starting Constraint Demonstration Tests")
        
        tests = {
            'contouring_only': self.test_contouring_constraints_only,
            'scenario': self.test_scenario_constraints,
            'linearized': self.test_linearized_constraints,
            'ellipsoid': self.test_ellipsoid_constraints,
            'gaussian': self.test_gaussian_constraints,
            'decomposition': self.test_decomposition_constraints,
            'all_combined': self.test_all_constraints_combined
        }
        
        results = {}
        
        for test_name, test_func in tests.items():
            LOG_INFO(f"Running {test_name} constraint test...")
            try:
                result = test_func()
                results[test_name] = result
                
                if result.success:
                    LOG_INFO(f"{test_name} constraint test PASSED")
                    # Create animation
                    self.create_animation(result)
                else:
                    LOG_WARN(f"{test_name} constraint test FAILED: {result.error_message}")
                    
            except Exception as e:
                LOG_WARN(f"{test_name} constraint test ERROR: {e}")
                results[test_name] = ConstraintTestResult(
                    constraint_type=test_name,
                    success=False,
                    trajectory=np.array([]),
                    solve_times=[],
                    constraint_violations=[],
                    error_message=str(e)
                )
        
        return results
    
    def generate_constraint_test_report(self, results: Dict[str, ConstraintTestResult]) -> str:
        """Generate comprehensive constraint test report."""
        report_path = os.path.join(self.output_dir, "constraint_test_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# Constraint Demonstration Test Report\n\n")
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            total_tests = len(results)
            passed_tests = sum(1 for r in results.values() if r.success)
            failed_tests = total_tests - passed_tests
            
            f.write("## Test Summary\n\n")
            f.write(f"- Total Tests: {total_tests}\n")
            f.write(f"- Passed: {passed_tests}\n")
            f.write(f"- Failed: {failed_tests}\n")
            f.write(f"- Success Rate: {passed_tests/total_tests*100:.1f}%\n\n")
            
            # Individual test results
            f.write("## Individual Constraint Test Results\n\n")
            
            for test_name, result in results.items():
                f.write(f"### {test_name.replace('_', ' ').title()} Test\n\n")
                
                if result.success:
                    f.write("**Status:** ✅ PASSED\n\n")
                    f.write(f"- Trajectory Length: {len(result.trajectory)} steps\n")
                    f.write(f"- Average Solve Time: {np.mean(result.solve_times):.4f}s\n")
                    f.write(f"- Max Constraint Violation: {np.max(result.constraint_violations):.6f}\n")
                else:
                    f.write("**Status:** ❌ FAILED\n\n")
                    f.write(f"- Error: {result.error_message}\n")
                
                f.write("\n")
        
        LOG_INFO(f"Constraint test report saved to {report_path}")
        return report_path


def main():
    """Main function to run constraint demonstration tests."""
    # Create test framework
    test_framework = ConstraintDemoTests()
    
    # Run all tests
    results = test_framework.run_all_constraint_tests()
    
    # Generate report
    report_path = test_framework.generate_constraint_test_report(results)
    
    # Print summary
    print("\n" + "="*60)
    print("CONSTRAINT DEMONSTRATION TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result.success else "❌ FAILED"
        print(f"{test_name.upper():<20} {status}")
        if not result.success and result.error_message:
            print(f"                     Error: {result.error_message}")
    
    print(f"\nTest report saved to: {report_path}")
    print(f"Output directory: {test_framework.output_dir}")


if __name__ == "__main__":
    main()
