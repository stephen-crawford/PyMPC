"""
End-to-End Tests for MPC Framework

This module provides comprehensive end-to-end tests that demonstrate
a car traversing a road while avoiding obstacles using various constraint types.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Polygon
import casadi as cd
from typing import List, Dict, Any, Tuple, Optional
import os
import time
from dataclasses import dataclass

from pympc.core import ModuleManager, ParameterManager, MPCCPlanner
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
class TestResult:
    """Test result data structure."""
    success: bool
    trajectory: np.ndarray
    controls: np.ndarray
    solve_times: List[float]
    iterations: List[int]
    constraint_violations: List[float]
    error_message: Optional[str] = None
    visualization_data: Optional[Dict[str, Any]] = None


class EndToEndTestFramework:
    """Framework for end-to-end MPC testing."""
    
    def __init__(self, test_name: str, output_dir: str = "test_outputs"):
        """
        Initialize the test framework.
        
        Args:
            test_name: Name of the test
            output_dir: Directory for output files
        """
        self.test_name = test_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Test configuration
        self.horizon = 20
        self.timestep = 0.1
        self.max_iterations = 150  # Increased for longer roads
        self.solver_tolerance = 1e-6
        
        # Vehicle parameters
        self.vehicle_length = 4.5
        self.vehicle_width = 2.0
        self.max_velocity = 15.0
        self.max_acceleration = 3.0
        self.max_steering_angle = 0.5
        
        # Road parameters
        self.road_width = 6.0
        self.road_length = 150.0  # Longer road
        
        # Results storage
        self.results = []
        
    def create_curved_road(self, num_points: int = 150) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a curved road with left and right boundaries.
        
        Args:
            num_points: Number of points along the road
            
        Returns:
            Tuple of (center_line, left_boundary, right_boundary)
        """
        # Create curved center line with more complex curves
        s = np.linspace(0, 1, num_points)
        x_center = s * self.road_length
        
        # Create a more complex S-curve that requires significant turning
        y_center = (10 * np.sin(2 * np.pi * s * 0.4) + 
                   5 * np.sin(2 * np.pi * s * 0.8) + 
                   3 * np.sin(2 * np.pi * s * 1.6))
        
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
        
        return center_line, left_boundary, right_boundary
    
    def create_dynamic_obstacles(self, num_obstacles: int = 3) -> List[Dict[str, Any]]:
        """
        Create dynamic obstacles for the test.
        
        Args:
            num_obstacles: Number of obstacles to create
            
        Returns:
            List of obstacle dictionaries
        """
        obstacles = []
        
        for i in range(num_obstacles):
            # Random obstacle properties
            start_x = 20 + i * 25 + np.random.uniform(-5, 5)
            start_y = np.random.uniform(-2, 2)
            velocity_x = np.random.uniform(5, 12)
            velocity_y = np.random.uniform(-1, 1)
            radius = np.random.uniform(0.8, 1.5)
            
            obstacle = {
                'position': [start_x, start_y],
                'velocity': [velocity_x, velocity_y],
                'radius': radius,
                'prediction_steps': 20,
                'uncertainty': [0.2, 0.1]  # Major and minor axis uncertainty
            }
            obstacles.append(obstacle)
        
        return obstacles
    
    def create_static_obstacles(self, num_obstacles: int = 2) -> List[Dict[str, Any]]:
        """
        Create static obstacles for the test.
        
        Args:
            num_obstacles: Number of obstacles to create
            
        Returns:
            List of static obstacle dictionaries
        """
        obstacles = []
        
        for i in range(num_obstacles):
            # Place obstacles along the road
            x = 30 + i * 20 + np.random.uniform(-3, 3)
            y = np.random.uniform(-1.5, 1.5)
            width = np.random.uniform(1.0, 2.0)
            height = np.random.uniform(1.0, 2.0)
            
            obstacle = {
                'position': [x, y],
                'size': [width, height],
                'type': 'rectangle'
            }
            obstacles.append(obstacle)
        
        return obstacles
    
    def run_contouring_scenario_test(self) -> TestResult:
        """Test with contouring constraints only."""
        LOG_INFO("Running Contouring Scenario Test")
        
        try:
            # Create road and obstacles
            center_line, left_boundary, right_boundary = self.create_curved_road()
            dynamic_obstacles = self.create_dynamic_obstacles(2)
            
            # Initialize planner
            planner = MPCCPlanner(horizon=self.horizon, timestep=self.timestep)
            
            # Add contouring objective and constraints
            contouring_obj = ContouringObjective(planner.solver)
            contouring_const = ContouringConstraints(planner.solver)
            
            planner.add_objective(contouring_obj)
            planner.add_constraint(contouring_const)
            
            # Set up reference path
            reference_path = self._create_reference_path(center_line)
            planner.set_reference_path(reference_path)
            
            # Set up road boundaries
            planner.set_road_boundaries(left_boundary, right_boundary)
            
            # Run simulation
            result = self._run_simulation(planner, dynamic_obstacles, "contouring")
            return result
            
        except Exception as e:
            LOG_WARN(f"Contouring test failed: {e}")
            return TestResult(success=False, trajectory=np.array([]), 
                           controls=np.array([]), solve_times=[], 
                           iterations=[], constraint_violations=[],
                           error_message=str(e))
    
    def run_scenario_constraints_test(self) -> TestResult:
        """Test with scenario constraints for obstacle avoidance."""
        LOG_INFO("Running Scenario Constraints Test")
        
        try:
            # Create road and obstacles
            center_line, left_boundary, right_boundary = self.create_curved_road()
            dynamic_obstacles = self.create_dynamic_obstacles(3)
            
            # Initialize planner
            planner = MPCCPlanner(horizon=self.horizon, timestep=self.timestep)
            
            # Add objectives and constraints
            goal_obj = GoalObjective(planner.solver)
            contouring_obj = ContouringObjective(planner.solver)
            contouring_const = ContouringConstraints(planner.solver)
            scenario_const = ScenarioConstraints(planner.solver)
            
            planner.add_objective(goal_obj)
            planner.add_objective(contouring_obj)
            planner.add_constraint(contouring_const)
            planner.add_constraint(scenario_const)
            
            # Set up reference path and goal
            reference_path = self._create_reference_path(center_line)
            goal = [center_line[-1, 0], center_line[-1, 1]]
            
            planner.set_reference_path(reference_path)
            planner.set_goal(goal)
            planner.set_road_boundaries(left_boundary, right_boundary)
            planner.set_dynamic_obstacles(dynamic_obstacles)
            
            # Run simulation
            result = self._run_simulation(planner, dynamic_obstacles, "scenario")
            return result
            
        except Exception as e:
            LOG_WARN(f"Scenario constraints test failed: {e}")
            return TestResult(success=False, trajectory=np.array([]), 
                           controls=np.array([]), solve_times=[], 
                           iterations=[], constraint_violations=[],
                           error_message=str(e))
    
    def run_linearized_constraints_test(self) -> TestResult:
        """Test with linearized constraints for obstacle avoidance."""
        LOG_INFO("Running Linearized Constraints Test")
        
        try:
            # Create road and obstacles
            center_line, left_boundary, right_boundary = self.create_curved_road()
            dynamic_obstacles = self.create_dynamic_obstacles(2)
            
            # Initialize planner
            planner = MPCCPlanner(horizon=self.horizon, timestep=self.timestep)
            
            # Add objectives and constraints
            goal_obj = GoalObjective(planner.solver)
            contouring_obj = ContouringObjective(planner.solver)
            contouring_const = ContouringConstraints(planner.solver)
            linearized_const = LinearizedConstraints(planner.solver)
            
            planner.add_objective(goal_obj)
            planner.add_objective(contouring_obj)
            planner.add_constraint(contouring_const)
            planner.add_constraint(linearized_const)
            
            # Set up reference path and goal
            reference_path = self._create_reference_path(center_line)
            goal = [center_line[-1, 0], center_line[-1, 1]]
            
            planner.set_reference_path(reference_path)
            planner.set_goal(goal)
            planner.set_road_boundaries(left_boundary, right_boundary)
            planner.set_dynamic_obstacles(dynamic_obstacles)
            
            # Run simulation
            result = self._run_simulation(planner, dynamic_obstacles, "linearized")
            return result
            
        except Exception as e:
            LOG_WARN(f"Linearized constraints test failed: {e}")
            return TestResult(success=False, trajectory=np.array([]), 
                           controls=np.array([]), solve_times=[], 
                           iterations=[], constraint_violations=[],
                           error_message=str(e))
    
    def run_ellipsoid_constraints_test(self) -> TestResult:
        """Test with ellipsoid constraints for smooth obstacle avoidance."""
        LOG_INFO("Running Ellipsoid Constraints Test")
        
        try:
            # Create road and obstacles
            center_line, left_boundary, right_boundary = self.create_curved_road()
            dynamic_obstacles = self.create_dynamic_obstacles(2)
            
            # Initialize planner
            planner = MPCCPlanner(horizon=self.horizon, timestep=self.timestep)
            
            # Add objectives and constraints
            goal_obj = GoalObjective(planner.solver)
            contouring_obj = ContouringObjective(planner.solver)
            contouring_const = ContouringConstraints(planner.solver)
            ellipsoid_const = EllipsoidConstraints(planner.solver)
            
            planner.add_objective(goal_obj)
            planner.add_objective(contouring_obj)
            planner.add_constraint(contouring_const)
            planner.add_constraint(ellipsoid_const)
            
            # Set up reference path and goal
            reference_path = self._create_reference_path(center_line)
            goal = [center_line[-1, 0], center_line[-1, 1]]
            
            planner.set_reference_path(reference_path)
            planner.set_goal(goal)
            planner.set_road_boundaries(left_boundary, right_boundary)
            planner.set_dynamic_obstacles(dynamic_obstacles)
            
            # Run simulation
            result = self._run_simulation(planner, dynamic_obstacles, "ellipsoid")
            return result
            
        except Exception as e:
            LOG_WARN(f"Ellipsoid constraints test failed: {e}")
            return TestResult(success=False, trajectory=np.array([]), 
                           controls=np.array([]), solve_times=[], 
                           iterations=[], constraint_violations=[],
                           error_message=str(e))
    
    def run_comprehensive_test(self) -> TestResult:
        """Test with all constraint types combined."""
        LOG_INFO("Running Comprehensive Test")
        
        try:
            # Create road and obstacles
            center_line, left_boundary, right_boundary = self.create_curved_road()
            dynamic_obstacles = self.create_dynamic_obstacles(3)
            static_obstacles = self.create_static_obstacles(2)
            
            # Initialize planner
            planner = MPCCPlanner(horizon=self.horizon, timestep=self.timestep)
            
            # Add all objectives and constraints
            goal_obj = GoalObjective(planner.solver)
            contouring_obj = ContouringObjective(planner.solver)
            contouring_const = ContouringConstraints(planner.solver)
            scenario_const = ScenarioConstraints(planner.solver)
            linearized_const = LinearizedConstraints(planner.solver)
            
            planner.add_objective(goal_obj)
            planner.add_objective(contouring_obj)
            planner.add_constraint(contouring_const)
            planner.add_constraint(scenario_const)
            planner.add_constraint(linearized_const)
            
            # Set up reference path and goal
            reference_path = self._create_reference_path(center_line)
            goal = [center_line[-1, 0], center_line[-1, 1]]
            
            planner.set_reference_path(reference_path)
            planner.set_goal(goal)
            planner.set_road_boundaries(left_boundary, right_boundary)
            planner.set_dynamic_obstacles(dynamic_obstacles)
            planner.set_static_obstacles(static_obstacles)
            
            # Run simulation
            result = self._run_simulation(planner, dynamic_obstacles, "comprehensive")
            return result
            
        except Exception as e:
            LOG_WARN(f"Comprehensive test failed: {e}")
            return TestResult(success=False, trajectory=np.array([]), 
                           controls=np.array([]), solve_times=[], 
                           iterations=[], constraint_violations=[],
                           error_message=str(e))
    
    def _create_reference_path(self, center_line: np.ndarray) -> Dict[str, Any]:
        """Create reference path data structure."""
        # Create arc length parameterization
        dx = np.diff(center_line[:, 0])
        dy = np.diff(center_line[:, 1])
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(ds)])
        s = s / s[-1]  # Normalize to [0, 1]
        
        # Create velocity profile
        velocity = np.ones_like(s) * self.max_velocity * 0.8
        
        return {
            'x': center_line[:, 0],
            'y': center_line[:, 1],
            's': s,
            'v': velocity
        }
    
    def _run_simulation(self, planner: MPCCPlanner, obstacles: List[Dict], 
                       test_type: str) -> TestResult:
        """Run the MPC simulation."""
        # Initial state
        initial_state = np.array([0.0, 0.0, 0.0, 5.0])  # [x, y, psi, v]
        
        # Storage for results
        trajectory = [initial_state]
        controls = []
        solve_times = []
        iterations = []
        constraint_violations = []
        
        current_state = initial_state.copy()
        
        for step in range(self.max_iterations):
            LOG_DEBUG(f"Step {step}: State = {current_state}")
            
            # Check if goal reached
            if self._is_goal_reached(current_state, planner.goal):
                LOG_INFO(f"Goal reached at step {step}")
                break
            
            # Update obstacles for current step
            updated_obstacles = self._update_obstacles(obstacles, step)
            planner.set_dynamic_obstacles(updated_obstacles)
            
            # Solve MPC problem
            start_time = time.time()
            try:
                solution = planner.solve(current_state)
                solve_time = time.time() - start_time
                
                if not solution['success']:
                    error_msg = f"Solver failed at step {step}: {solution.get('message', 'Unknown error')}"
                    LOG_WARN(error_msg)
                    return TestResult(success=False, trajectory=np.array(trajectory),
                                   controls=np.array(controls), solve_times=solve_times,
                                   iterations=iterations, constraint_violations=constraint_violations,
                                   error_message=error_msg)
                
                # Extract solution
                next_state = solution['state_trajectory'][1]  # Next state
                control = solution['control_trajectory'][0]  # Current control
                
                # Store results
                trajectory.append(next_state)
                controls.append(control)
                solve_times.append(solve_time)
                iterations.append(solution.get('iterations', 0))
                constraint_violations.append(solution.get('constraint_violation', 0.0))
                
                # Update state
                current_state = next_state
                
            except Exception as e:
                error_msg = f"Simulation failed at step {step}: {e}"
                LOG_WARN(error_msg)
                return TestResult(success=False, trajectory=np.array(trajectory),
                               controls=np.array(controls), solve_times=solve_times,
                               iterations=iterations, constraint_violations=constraint_violations,
                               error_message=error_msg)
        
        # Create visualization data
        visualization_data = self._create_visualization_data(trajectory, obstacles, test_type)
        
        return TestResult(
            success=True,
            trajectory=np.array(trajectory),
            controls=np.array(controls),
            solve_times=solve_times,
            iterations=iterations,
            constraint_violations=constraint_violations,
            visualization_data=visualization_data
        )
    
    def _is_goal_reached(self, state: np.ndarray, goal: List[float]) -> bool:
        """Check if goal is reached."""
        if goal is None:
            return False
        
        distance = np.sqrt((state[0] - goal[0])**2 + (state[1] - goal[1])**2)
        return distance < 2.0  # 2 meter tolerance
    
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
    
    def _create_visualization_data(self, trajectory: List[np.ndarray], 
                                 obstacles: List[Dict], test_type: str) -> Dict[str, Any]:
        """Create visualization data for the test."""
        return {
            'trajectory': trajectory,
            'obstacles': obstacles,
            'test_type': test_type,
            'road_data': self._get_road_visualization_data()
        }
    
    def _get_road_visualization_data(self) -> Dict[str, Any]:
        """Get road visualization data."""
        center_line, left_boundary, right_boundary = self.create_curved_road()
        return {
            'center_line': center_line,
            'left_boundary': left_boundary,
            'right_boundary': right_boundary
        }
    
    def create_animation(self, result: TestResult, filename: str = None) -> str:
        """Create animation of the test result."""
        if not result.success or result.visualization_data is None:
            LOG_WARN("Cannot create animation for failed test")
            return ""
        
        if filename is None:
            filename = f"{self.test_name}_{result.visualization_data['test_type']}.gif"
        
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
        
        # Initialize vehicle and obstacle plots
        vehicle_patch = None
        obstacle_patches = []
        
        def animate(frame):
            nonlocal vehicle_patch, obstacle_patches
            
            # Clear previous patches
            if vehicle_patch:
                vehicle_patch.remove()
            for patch in obstacle_patches:
                patch.remove()
            obstacle_patches.clear()
            
            # Plot trajectory up to current frame
            if frame > 0:
                ax.plot(trajectory[:frame, 0], trajectory[:frame, 1], 
                       'b-', linewidth=2, alpha=0.7, label='Vehicle Path')
            
            # Plot current vehicle position
            if frame < len(trajectory):
                x, y, psi = trajectory[frame, :3]
                vehicle_patch = self._draw_vehicle(ax, x, y, psi)
            
            # Plot obstacles at current time
            for obs in obstacles:
                obs_x = obs['position'][0] + obs['velocity'][0] * frame * self.timestep
                obs_y = obs['position'][1] + obs['velocity'][1] * frame * self.timestep
                obs_radius = obs['radius']
                
                circle = Circle((obs_x, obs_y), obs_radius, 
                               color='red', alpha=0.6, label='Obstacle')
                ax.add_patch(circle)
                obstacle_patches.append(circle)
            
            # Set plot properties for longer road
            ax.set_xlim(-10, 160)
            ax.set_ylim(-20, 20)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title(f'MPC Test: {result.visualization_data["test_type"].title()} - Step {frame}')
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(trajectory), 
                                    interval=100, repeat=True)
        
        # Save animation
        anim.save(filepath, writer='pillow', fps=10)
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
    
    def run_all_tests(self) -> Dict[str, TestResult]:
        """Run all end-to-end tests."""
        LOG_INFO("Starting End-to-End Tests")
        
        tests = {
            'contouring': self.run_contouring_scenario_test,
            'scenario': self.run_scenario_constraints_test,
            'linearized': self.run_linearized_constraints_test,
            'ellipsoid': self.run_ellipsoid_constraints_test,
            'comprehensive': self.run_comprehensive_test
        }
        
        results = {}
        
        for test_name, test_func in tests.items():
            LOG_INFO(f"Running {test_name} test...")
            try:
                result = test_func()
                results[test_name] = result
                
                if result.success:
                    LOG_INFO(f"{test_name} test PASSED")
                    # Create animation
                    self.create_animation(result, f"{test_name}_test.gif")
                else:
                    LOG_WARN(f"{test_name} test FAILED: {result.error_message}")
                    
            except Exception as e:
                LOG_WARN(f"{test_name} test ERROR: {e}")
                results[test_name] = TestResult(
                    success=False, trajectory=np.array([]), 
                    controls=np.array([]), solve_times=[], 
                    iterations=[], constraint_violations=[],
                    error_message=str(e)
                )
        
        return results
    
    def generate_test_report(self, results: Dict[str, TestResult]) -> str:
        """Generate comprehensive test report."""
        report_path = os.path.join(self.output_dir, "test_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# End-to-End MPC Test Report\n\n")
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
            f.write("## Individual Test Results\n\n")
            
            for test_name, result in results.items():
                f.write(f"### {test_name.title()} Test\n\n")
                
                if result.success:
                    f.write("**Status:** ✅ PASSED\n\n")
                    f.write(f"- Trajectory Length: {len(result.trajectory)} steps\n")
                    f.write(f"- Average Solve Time: {np.mean(result.solve_times):.4f}s\n")
                    f.write(f"- Average Iterations: {np.mean(result.iterations):.1f}\n")
                    f.write(f"- Max Constraint Violation: {np.max(result.constraint_violations):.6f}\n")
                else:
                    f.write("**Status:** ❌ FAILED\n\n")
                    f.write(f"- Error: {result.error_message}\n")
                
                f.write("\n")
        
        LOG_INFO(f"Test report saved to {report_path}")
        return report_path


def main():
    """Main function to run end-to-end tests."""
    # Create test framework
    test_framework = EndToEndTestFramework("mpc_end_to_end_tests")
    
    # Run all tests
    results = test_framework.run_all_tests()
    
    # Generate report
    report_path = test_framework.generate_test_report(results)
    
    # Print summary
    print("\n" + "="*60)
    print("END-TO-END TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result.success else "❌ FAILED"
        print(f"{test_name.upper():<15} {status}")
        if not result.success and result.error_message:
            print(f"                Error: {result.error_message}")
    
    print(f"\nTest report saved to: {report_path}")
    print(f"Output directory: {test_framework.output_dir}")


if __name__ == "__main__":
    main()
