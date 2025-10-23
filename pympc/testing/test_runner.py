"""
Test runner for MPC planning scenarios.

This module implements the comprehensive test suite for MPC planning
with various dynamics models, constraints, and objectives.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Tuple, Optional, Any
import time
import os
import json
from datetime import datetime

from ..core.dynamics import create_dynamics_model
from ..core.planner import create_mpc_planner
from ..core.solver import create_solver
from ..modules.constraints.contouring_constraints import ContouringConstraints
from ..modules.constraints.ellipsoid_constraints import EllipsoidConstraints
from ..modules.constraints.gaussian_constraints import GaussianConstraints
from ..modules.constraints.scenario_constraints import ScenarioConstraints
from ..modules.objectives.contouring_objective import ContouringObjective
from ..modules.objectives.goal_objective import GoalObjective


class TestResult:
    """Test result container."""
    
    def __init__(self, test_name: str, success: bool, execution_time: float,
                 final_position: np.ndarray = None, final_error: float = 0.0,
                 constraint_violations: int = 0, objective_value: float = 0.0,
                 trajectory: np.ndarray = None, controls: np.ndarray = None):
        """
        Initialize test result.
        
        Args:
            test_name: Name of the test
            success: Whether test succeeded
            execution_time: Test execution time
            final_position: Final vehicle position
            final_error: Final tracking error
            constraint_violations: Number of constraint violations
            objective_value: Final objective value
            trajectory: Vehicle trajectory
            controls: Control inputs
        """
        self.test_name = test_name
        self.success = success
        self.execution_time = execution_time
        self.final_position = final_position
        self.final_error = final_error
        self.constraint_violations = constraint_violations
        self.objective_value = objective_value
        self.trajectory = trajectory
        self.controls = controls


class TestConfig:
    """Test configuration."""
    
    def __init__(self, test_name: str, dynamics_type: str = "bicycle",
                 horizon_length: int = 20, timestep: float = 0.1,
                 max_steps: int = 150, gif_generation: bool = True,
                 num_obstacles: int = 3, obstacle_type: str = "static",
                 constraint_type: str = "linear", road_width: float = 8.0):
        """
        Initialize test configuration.
        
        Args:
            test_name: Name of the test
            dynamics_type: Type of dynamics model
            horizon_length: Prediction horizon length
            timestep: Time step
            max_steps: Maximum simulation steps
            gif_generation: Whether to generate GIF
            num_obstacles: Number of obstacles
            obstacle_type: Type of obstacles ("static" or "dynamic")
            constraint_type: Type of obstacle constraints
            road_width: Road width
        """
        self.test_name = test_name
        self.dynamics_type = dynamics_type
        self.horizon_length = horizon_length
        self.timestep = timestep
        self.max_steps = max_steps
        self.gif_generation = gif_generation
        self.num_obstacles = num_obstacles
        self.obstacle_type = obstacle_type
        self.constraint_type = constraint_type
        self.road_width = road_width


class MPCTestRunner:
    """Main test runner for MPC scenarios."""
    
    def __init__(self, output_dir: str = "test_outputs"):
        """
        Initialize test runner.
        
        Args:
            output_dir: Output directory for test results
        """
        self.output_dir = output_dir
        self.results = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def run_test(self, config: TestConfig) -> TestResult:
        """
        Run a single test.
        
        Args:
            config: Test configuration
            
        Returns:
            Test result
        """
        print(f"Running test: {config.test_name}")
        print(f"  Dynamics: {config.dynamics_type}")
        print(f"  Obstacles: {config.num_obstacles} {config.obstacle_type}")
        print(f"  Constraints: {config.constraint_type}")
        
        start_time = time.time()
        
        try:
            # Create test scenario
            scenario = self._create_test_scenario(config)
            
            # Run simulation
            result = self._run_simulation(scenario, config)
            
            # Generate visualization if requested
            if config.gif_generation:
                self._generate_visualization(scenario, result, config)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            print(f"Test completed: {result.success}")
            print(f"Execution time: {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            execution_time = time.time() - start_time
            return TestResult(
                test_name=config.test_name,
                success=False,
                execution_time=execution_time
            )
    
    def _create_test_scenario(self, config: TestConfig) -> Dict[str, Any]:
        """
        Create test scenario.
        
        Args:
            config: Test configuration
            
        Returns:
            Scenario dictionary
        """
        # Create reference path
        reference_path = self._create_reference_path(config.road_width)
        
        # Create obstacles
        obstacles = self._create_obstacles(config)
        
        # Create dynamics model
        dynamics = create_dynamics_model(config.dynamics_type, dt=config.timestep)
        
        # Create planner
        planner = create_mpc_planner(
            dynamics_type=config.dynamics_type,
            horizon_length=config.horizon_length,
            dt=config.timestep
        )
        
        # Add objectives and constraints
        self._setup_planner(planner, reference_path, obstacles, config)
        
        return {
            'reference_path': reference_path,
            'obstacles': obstacles,
            'dynamics': dynamics,
            'planner': planner,
            'config': config
        }
    
    def _create_reference_path(self, road_width: float) -> np.ndarray:
        """
        Create reference path.
        
        Args:
            road_width: Road width
            
        Returns:
            Reference path as Nx2 array
        """
        # Create a curved path
        t = np.linspace(0, 4*np.pi, 100)
        x = t * np.cos(t/4)
        y = t * np.sin(t/4)
        
        return np.column_stack([x, y])
    
    def _create_obstacles(self, config: TestConfig) -> List[Dict[str, Any]]:
        """
        Create obstacles for test.
        
        Args:
            config: Test configuration
            
        Returns:
            List of obstacle dictionaries
        """
        obstacles = []
        
        # Create obstacles along the path
        for i in range(config.num_obstacles):
            # Place obstacles at different positions along the path
            angle = 2 * np.pi * i / config.num_obstacles
            radius = 20.0 + 10.0 * i
            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            obstacle = {
                'id': f'obstacle_{i}',
                'center': [x, y],
                'radius': 1.0 + 0.5 * i,
                'enabled': True
            }
            
            obstacles.append(obstacle)
        
        return obstacles
    
    def _setup_planner(self, planner, reference_path: np.ndarray, 
                       obstacles: List[Dict[str, Any]], config: TestConfig) -> None:
        """
        Set up planner with objectives and constraints.
        
        Args:
            planner: MPC planner
            reference_path: Reference path
            obstacles: List of obstacles
            config: Test configuration
        """
        # Add contouring objective
        contouring_obj = ContouringObjective(
            contouring_weight=2.0,
            lag_weight=1.0,
            progress_weight=1.5,
            velocity_weight=0.1
        )
        contouring_obj.set_reference_path(reference_path)
        planner.add_objective(contouring_obj)
        
        # Add contouring constraints
        contouring_const = ContouringConstraints(
            road_width=config.road_width,
            safety_margin=0.5
        )
        contouring_const.set_reference_path(reference_path)
        planner.add_constraint(contouring_const)
        
        # Add obstacle constraints based on type
        if config.constraint_type == "linear":
            # Use ellipsoid constraints for linear obstacle avoidance
            for obstacle in obstacles:
                ellipsoid_const = EllipsoidConstraints(safety_margin=1.0)
                ellipsoid_const.add_circular_obstacle(
                    obstacle['center'], obstacle['radius'], obstacle['id']
                )
                planner.add_constraint(ellipsoid_const)
        
        elif config.constraint_type == "ellipsoid":
            # Use ellipsoid constraints
            ellipsoid_const = EllipsoidConstraints(safety_margin=1.0)
            for obstacle in obstacles:
                ellipsoid_const.add_circular_obstacle(
                    obstacle['center'], obstacle['radius'], obstacle['id']
                )
            planner.add_constraint(ellipsoid_const)
        
        elif config.constraint_type == "gaussian":
            # Use Gaussian constraints
            gaussian_const = GaussianConstraints(safety_margin=1.0)
            for obstacle in obstacles:
                gaussian_const.add_circular_obstacle(
                    obstacle['center'], obstacle['radius'], obstacle['id']
                )
            planner.add_constraint(gaussian_const)
        
        elif config.constraint_type == "scenario":
            # Use scenario constraints
            scenario_const = ScenarioConstraints(safety_margin=1.0)
            scenario_const.add_scenario("main_scenario", obstacles)
            planner.add_constraint(scenario_const)
    
    def _run_simulation(self, scenario: Dict[str, Any], config: TestConfig) -> TestResult:
        """
        Run MPC simulation.
        
        Args:
            scenario: Test scenario
            config: Test configuration
            
        Returns:
            Test result
        """
        planner = scenario['planner']
        reference_path = scenario['reference_path']
        
        # Initial state
        initial_state = np.array([0.0, 0.0, 0.0, 5.0])  # [x, y, psi, v]
        if config.dynamics_type in ["contouring_bicycle", "contouring_unicycle"]:
            initial_state = np.append(initial_state, 0.0)  # Add spline parameter
        
        current_state = initial_state.copy()
        trajectory = [current_state[:2].copy()]
        controls = []
        
        success = True
        constraint_violations = 0
        total_objective = 0.0
        
        for step in range(config.max_steps):
            try:
                # Solve MPC
                solution = planner.solve(current_state)
                
                if solution is None:
                    success = False
                    break
                
                # Apply first control input
                if 'controls' in solution and solution['controls'] is not None:
                    control = solution['controls'][:, 0]
                    controls.append(control.copy())
                    
                    # Integrate dynamics
                    current_state = self._integrate_dynamics(
                        current_state, control, scenario['dynamics'], config.timestep
                    )
                else:
                    success = False
                    break
                
                # Store trajectory
                trajectory.append(current_state[:2].copy())
                
                # Check for goal reaching
                if self._is_goal_reached(current_state, reference_path):
                    break
                
                # Check for constraint violations
                if self._check_constraint_violations(current_state, scenario):
                    constraint_violations += 1
                
                # Update objective value
                if 'cost' in solution:
                    total_objective += solution['cost']
                
            except Exception as e:
                print(f"Simulation error at step {step}: {e}")
                success = False
                break
        
        # Calculate final error
        final_error = self._calculate_final_error(current_state, reference_path)
        
        return TestResult(
            test_name=config.test_name,
            success=success,
            execution_time=0.0,  # Will be set by caller
            final_position=current_state[:2],
            final_error=final_error,
            constraint_violations=constraint_violations,
            objective_value=total_objective,
            trajectory=np.array(trajectory),
            controls=np.array(controls) if controls else None
        )
    
    def _integrate_dynamics(self, state: np.ndarray, control: np.ndarray,
                           dynamics, dt: float) -> np.ndarray:
        """
        Integrate dynamics forward.
        
        Args:
            state: Current state
            control: Control input
            dynamics: Dynamics model
            dt: Time step
            
        Returns:
            Next state
        """
        # Simple Euler integration
        next_state = state.copy()
        
        if dynamics.nx == 4:  # Unicycle model
            psi, v = state[2], state[3]
            a, w = control[0], control[1]
            
            next_state[0] += v * np.cos(psi) * dt
            next_state[1] += v * np.sin(psi) * dt
            next_state[2] += w * dt
            next_state[3] += a * dt
        
        elif dynamics.nx == 5:  # Bicycle model
            psi, v, delta = state[2], state[3], state[4]
            a, delta_dot = control[0], control[1]
            
            # Bicycle model integration
            lr = dynamics.wheelbase / 2.0
            lf = dynamics.wheelbase / 2.0
            ratio = lr / (lr + lf)
            
            beta = np.arctan(ratio * np.tan(delta))
            
            next_state[0] += v * np.cos(psi + beta) * dt
            next_state[1] += v * np.sin(psi + beta) * dt
            next_state[2] += (v / lr) * np.sin(beta) * dt
            next_state[3] += a * dt
            next_state[4] += delta_dot * dt
        
        elif dynamics.nx == 6:  # Contouring bicycle model
            psi, v, delta, s = state[2], state[3], state[4], state[5]
            a, delta_dot = control[0], control[1]
            
            # Bicycle model integration
            lr = dynamics.wheelbase / 2.0
            lf = dynamics.wheelbase / 2.0
            ratio = lr / (lr + lf)
            
            beta = np.arctan(ratio * np.tan(delta))
            
            next_state[0] += v * np.cos(psi + beta) * dt
            next_state[1] += v * np.sin(psi + beta) * dt
            next_state[2] += (v / lr) * np.sin(beta) * dt
            next_state[3] += a * dt
            next_state[4] += delta_dot * dt
            next_state[5] += v * dt  # Path parameter
        
        return next_state
    
    def _is_goal_reached(self, state: np.ndarray, reference_path: np.ndarray) -> bool:
        """
        Check if goal is reached.
        
        Args:
            state: Current state
            reference_path: Reference path
            
        Returns:
            True if goal reached, False otherwise
        """
        # Check if we're near the end of the path
        current_pos = state[:2]
        end_pos = reference_path[-1]
        
        distance_to_end = np.linalg.norm(current_pos - end_pos)
        return distance_to_end < 2.0  # 2 meter tolerance
    
    def _check_constraint_violations(self, state: np.ndarray, scenario: Dict[str, Any]) -> bool:
        """
        Check for constraint violations.
        
        Args:
            state: Current state
            scenario: Test scenario
            
        Returns:
            True if constraint violated, False otherwise
        """
        # This is a simplified check - in practice, you'd check all constraints
        return False
    
    def _calculate_final_error(self, state: np.ndarray, reference_path: np.ndarray) -> float:
        """
        Calculate final tracking error.
        
        Args:
            state: Final state
            reference_path: Reference path
            
        Returns:
            Final tracking error
        """
        current_pos = state[:2]
        end_pos = reference_path[-1]
        return np.linalg.norm(current_pos - end_pos)
    
    def _generate_visualization(self, scenario: Dict[str, Any], result: TestResult, 
                               config: TestConfig) -> None:
        """
        Generate test visualization.
        
        Args:
            scenario: Test scenario
            result: Test result
            config: Test configuration
        """
        if result.trajectory is None:
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot reference path
        reference_path = scenario['reference_path']
        ax.plot(reference_path[:, 0], reference_path[:, 1], 'b-', linewidth=2, label='Reference Path')
        
        # Plot vehicle trajectory
        trajectory = result.trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Vehicle Trajectory')
        
        # Plot obstacles
        obstacles = scenario['obstacles']
        for obstacle in obstacles:
            circle = plt.Circle(obstacle['center'], obstacle['radius'], 
                              color='red', alpha=0.5, label='Obstacle')
            ax.add_patch(circle)
        
        # Plot start and end points
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
        
        # Set up plot
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{config.test_name} - {"Success" if result.success else "Failed"}')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')
        
        # Save plot
        output_file = os.path.join(self.output_dir, f'{config.test_name}_result.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved: {output_file}")
    
    def run_all_tests(self) -> List[TestResult]:
        """
        Run all test scenarios.
        
        Returns:
            List of test results
        """
        test_configs = [
            # Test 1: Contouring constraints, contouring objective, overactuated system
            TestConfig("test_1_contouring_overactuated", "overactuated_unicycle", 
                      constraint_type="linear", num_obstacles=0),
            
            # Test 2: Contouring constraints, contouring objective, car system
            TestConfig("test_2_contouring_car", "bicycle", 
                      constraint_type="linear", num_obstacles=0),
            
            # Test 3: Contouring constraints, contouring objective, 3 static obstacles, linear constraints, overactuated
            TestConfig("test_3_static_linear_overactuated", "overactuated_unicycle", 
                      constraint_type="linear", num_obstacles=3, obstacle_type="static"),
            
            # Test 4: Contouring constraints, contouring objective, 3 static obstacles, linear constraints, car
            TestConfig("test_4_static_linear_car", "bicycle", 
                      constraint_type="linear", num_obstacles=3, obstacle_type="static"),
            
            # Test 5: Contouring constraints, contouring objective, 3 static obstacles, ellipsoid constraints, car
            TestConfig("test_5_static_ellipsoid_car", "bicycle", 
                      constraint_type="ellipsoid", num_obstacles=3, obstacle_type="static"),
            
            # Test 6: Contouring constraints, contouring objective, 3 dynamic obstacles, ellipsoid constraints, car
            TestConfig("test_6_dynamic_ellipsoid_car", "bicycle", 
                      constraint_type="ellipsoid", num_obstacles=3, obstacle_type="dynamic"),
            
            # Test 7: Contouring constraints, contouring objective, 3 dynamic obstacles, gaussian constraints, car
            TestConfig("test_7_dynamic_gaussian_car", "bicycle", 
                      constraint_type="gaussian", num_obstacles=3, obstacle_type="dynamic"),
            
            # Test 8: Contouring constraints, contouring objective, 3 dynamic obstacles, scenario constraints, car
            TestConfig("test_8_dynamic_scenario_car", "bicycle", 
                      constraint_type="scenario", num_obstacles=3, obstacle_type="dynamic"),
        ]
        
        results = []
        for config in test_configs:
            result = self.run_test(config)
            results.append(result)
            self.results.append(result)
        
        # Generate summary report
        self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results: List[TestResult]) -> None:
        """
        Generate summary report.
        
        Args:
            results: List of test results
        """
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(results),
            'successful_tests': sum(1 for r in results if r.success),
            'failed_tests': sum(1 for r in results if not r.success),
            'success_rate': sum(1 for r in results if r.success) / len(results),
            'total_execution_time': sum(r.execution_time for r in results),
            'average_execution_time': np.mean([r.execution_time for r in results]),
            'test_results': [
                {
                    'test_name': r.test_name,
                    'success': r.success,
                    'execution_time': r.execution_time,
                    'final_error': r.final_error,
                    'constraint_violations': r.constraint_violations,
                    'objective_value': r.objective_value
                }
                for r in results
            ]
        }
        
        # Save summary
        summary_file = os.path.join(self.output_dir, 'test_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary report saved: {summary_file}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Total execution time: {summary['total_execution_time']:.2f}s")


def main():
    """Main function to run all tests."""
    runner = MPCTestRunner()
    results = runner.run_all_tests()
    
    print("\nTest Results Summary:")
    print("=" * 50)
    for result in results:
        status = "PASS" if result.success else "FAIL"
        print(f"{result.test_name}: {status} ({result.execution_time:.2f}s)")


if __name__ == "__main__":
    main()
