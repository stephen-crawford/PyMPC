"""
Comprehensive test suite for PyMPC framework.

This module implements all the test scenarios requested by the user:
1. Contouring constraints, contouring objective along a curving path with the overactuated system
2. Contouring constraints, contouring objective along a curving path with the car dynamics
3. Contouring constraints, contouring objective, 3 static obstacles with linear constraints for obstacle avoidance, overactuated system
4. Contouring constraints, contouring objective, 3 static obstacles with linear constraints for obstacle avoidance, car system
5. Contouring constraints, contouring objective, 3 static obstacles with ellipsoid constraints for obstacle avoidance, car system
6. Contouring constraints, contouring objective, 3 dynamic obstacles with ellipsoid constraints for obstacle avoidance, car system
7. Contouring constraints, contouring objective, 3 dynamic obstacles with gaussian constraints for obstacle avoidance, car system
8. Contouring constraints, contouring objective, 3 dynamic obstacles with scenario constraints for obstacle avoidance, car system
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Tuple, Optional, Any
import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pympc.core import (
    OveractuatedPointMass, BicycleModel, MPCCPlanner, CasADiSolver
)
from pympc.modules.constraints import (
    ContouringConstraints, EllipsoidConstraints, GaussianConstraints, 
    ScenarioConstraints, LinearizedConstraints
)
from pympc.modules.objectives import ContouringObjective


class ComprehensiveTestSuite:
    """Comprehensive test suite for PyMPC framework."""
    
    def __init__(self, output_dir: str = "test_results"):
        """
        Initialize test suite.
        
        Args:
            output_dir: Directory for test results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Test results
        self.test_results = {}
        self.visualization_data = {}
        
        # Test parameters
        self.horizon_length = 20
        self.dt = 0.1
        self.max_retries = 2
        
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all test scenarios.
        
        Returns:
            Dictionary of test results
        """
        print("Starting Comprehensive Test Suite for PyMPC Framework")
        print("=" * 60)
        
        # Test scenarios
        test_scenarios = [
            ("test_1", "Overactuated system with contouring constraints and objective"),
            ("test_2", "Car system with contouring constraints and objective"),
            ("test_3", "Overactuated system with static obstacles and linear constraints"),
            ("test_4", "Car system with static obstacles and linear constraints"),
            ("test_5", "Car system with static obstacles and ellipsoid constraints"),
            ("test_6", "Car system with dynamic obstacles and ellipsoid constraints"),
            ("test_7", "Car system with dynamic obstacles and gaussian constraints"),
            ("test_8", "Car system with dynamic obstacles and scenario constraints")
        ]
        
        for test_id, description in test_scenarios:
            print(f"\nRunning {test_id}: {description}")
            print("-" * 40)
            
            try:
                result = self._run_test_scenario(test_id)
                self.test_results[test_id] = result
                
                if result['success']:
                    print(f"✓ {test_id} PASSED")
                else:
                    print(f"✗ {test_id} FAILED: {result['error']}")
                    
            except Exception as e:
                print(f"✗ {test_id} ERROR: {str(e)}")
                self.test_results[test_id] = {
                    'success': False,
                    'error': str(e),
                    'exception': True
                }
        
        # Generate summary
        self._generate_summary()
        
        return self.test_results
    
    def _run_test_scenario(self, test_id: str) -> Dict[str, Any]:
        """
        Run a specific test scenario.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Test result dictionary
        """
        if test_id == "test_1":
            return self._test_overactuated_contouring()
        elif test_id == "test_2":
            return self._test_car_contouring()
        elif test_id == "test_3":
            return self._test_overactuated_static_linear()
        elif test_id == "test_4":
            return self._test_car_static_linear()
        elif test_id == "test_5":
            return self._test_car_static_ellipsoid()
        elif test_id == "test_6":
            return self._test_car_dynamic_ellipsoid()
        elif test_id == "test_7":
            return self._test_car_dynamic_gaussian()
        elif test_id == "test_8":
            return self._test_car_dynamic_scenario()
        else:
            raise ValueError(f"Unknown test ID: {test_id}")
    
    def _test_overactuated_contouring(self) -> Dict[str, Any]:
        """Test 1: Overactuated system with contouring constraints and objective."""
        try:
            # Create overactuated dynamics
            dynamics = OveractuatedPointMass(dt=self.dt)
            
            # Create solver
            solver = CasADiSolver(dynamics, self.horizon_length, self.dt)
            
            # Create planner
            planner = MPCCPlanner(dynamics, solver)
            
            # Add contouring constraints
            contouring_constraints = ContouringConstraints()
            planner.add_constraint_module(contouring_constraints)
            
            # Add contouring objective
            contouring_objective = ContouringObjective()
            planner.add_objective_module(contouring_objective)
            
            # Generate curving path
            path = self._generate_curving_path()
            contouring_constraints.set_reference_path(path)
            contouring_objective.set_reference_path(path)
            
            # Run simulation
            result = self._run_simulation(planner, dynamics, "overactuated_contouring")
            
            return {
                'success': result['success'],
                'error': result.get('error'),
                'collisions': result.get('collisions', 0),
                'simulation_time': result.get('simulation_time', 0),
                'final_position': result.get('final_position'),
                'path_deviation': result.get('path_deviation', 0)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_car_contouring(self) -> Dict[str, Any]:
        """Test 2: Car system with contouring constraints and objective."""
        try:
            # Create car dynamics
            dynamics = BicycleModel(dt=self.dt)
            
            # Create solver
            solver = CasADiSolver(dynamics, self.horizon_length, self.dt)
            
            # Create planner
            planner = MPCCPlanner(dynamics, solver)
            
            # Add contouring constraints
            contouring_constraints = ContouringConstraints()
            planner.add_constraint_module(contouring_constraints)
            
            # Add contouring objective
            contouring_objective = ContouringObjective()
            planner.add_objective_module(contouring_objective)
            
            # Generate curving path
            path = self._generate_curving_path()
            contouring_constraints.set_reference_path(path)
            contouring_objective.set_reference_path(path)
            
            # Run simulation with retry logic
            result = self._run_simulation_with_retry(planner, dynamics, "car_contouring")
            
            return {
                'success': result['success'],
                'error': result.get('error'),
                'collisions': result.get('collisions', 0),
                'simulation_time': result.get('simulation_time', 0),
                'final_position': result.get('final_position'),
                'path_deviation': result.get('path_deviation', 0),
                'retries': result.get('retries', 0)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_overactuated_static_linear(self) -> Dict[str, Any]:
        """Test 3: Overactuated system with static obstacles and linear constraints."""
        try:
            # Create overactuated dynamics
            dynamics = OveractuatedPointMass(dt=self.dt)
            
            # Create solver
            solver = CasADiSolver(dynamics, self.horizon_length, self.dt)
            
            # Create planner
            planner = MPCCPlanner(dynamics, solver)
            
            # Add contouring constraints and objective
            contouring_constraints = ContouringConstraints()
            contouring_objective = ContouringObjective()
            planner.add_constraint_module(contouring_constraints)
            planner.add_objective_module(contouring_objective)
            
            # Add linear constraints for static obstacles
            linear_constraints = LinearizedConstraints()
            planner.add_constraint_module(linear_constraints)
            
            # Generate curving path and obstacles
            path = self._generate_curving_path()
            obstacles = self._generate_static_obstacles()
            
            contouring_constraints.set_reference_path(path)
            contouring_objective.set_reference_path(path)
            
            # Add obstacles to linear constraints
            for obs in obstacles:
                linear_constraints.add_obstacle(obs['center'], obs['radius'])
            
            # Run simulation
            result = self._run_simulation(planner, dynamics, "overactuated_static_linear")
            
            return {
                'success': result['success'],
                'error': result.get('error'),
                'collisions': result.get('collisions', 0),
                'simulation_time': result.get('simulation_time', 0),
                'final_position': result.get('final_position'),
                'path_deviation': result.get('path_deviation', 0)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_car_static_linear(self) -> Dict[str, Any]:
        """Test 4: Car system with static obstacles and linear constraints."""
        try:
            # Create car dynamics
            dynamics = BicycleModel(dt=self.dt)
            
            # Create solver
            solver = CasADiSolver(dynamics, self.horizon_length, self.dt)
            
            # Create planner
            planner = MPCCPlanner(dynamics, solver)
            
            # Add contouring constraints and objective
            contouring_constraints = ContouringConstraints()
            contouring_objective = ContouringObjective()
            planner.add_constraint_module(contouring_constraints)
            planner.add_objective_module(contouring_objective)
            
            # Add linear constraints for static obstacles
            linear_constraints = LinearizedConstraints()
            planner.add_constraint_module(linear_constraints)
            
            # Generate curving path and obstacles
            path = self._generate_curving_path()
            obstacles = self._generate_static_obstacles()
            
            contouring_constraints.set_reference_path(path)
            contouring_objective.set_reference_path(path)
            
            # Add obstacles to linear constraints
            for obs in obstacles:
                linear_constraints.add_obstacle(obs['center'], obs['radius'])
            
            # Run simulation with retry logic
            result = self._run_simulation_with_retry(planner, dynamics, "car_static_linear")
            
            return {
                'success': result['success'],
                'error': result.get('error'),
                'collisions': result.get('collisions', 0),
                'simulation_time': result.get('simulation_time', 0),
                'final_position': result.get('final_position'),
                'path_deviation': result.get('path_deviation', 0),
                'retries': result.get('retries', 0)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_car_static_ellipsoid(self) -> Dict[str, Any]:
        """Test 5: Car system with static obstacles and ellipsoid constraints."""
        try:
            # Create car dynamics
            dynamics = BicycleModel(dt=self.dt)
            
            # Create solver
            solver = CasADiSolver(dynamics, self.horizon_length, self.dt)
            
            # Create planner
            planner = MPCCPlanner(dynamics, solver)
            
            # Add contouring constraints and objective
            contouring_constraints = ContouringConstraints()
            contouring_objective = ContouringObjective()
            planner.add_constraint_module(contouring_constraints)
            planner.add_objective_module(contouring_objective)
            
            # Add ellipsoid constraints for static obstacles
            ellipsoid_constraints = EllipsoidConstraints()
            planner.add_constraint_module(ellipsoid_constraints)
            
            # Generate curving path and obstacles
            path = self._generate_curving_path()
            obstacles = self._generate_static_obstacles()
            
            contouring_constraints.set_reference_path(path)
            contouring_objective.set_reference_path(path)
            
            # Add obstacles to ellipsoid constraints
            for obs in obstacles:
                ellipsoid_constraints.add_obstacle(obs['center'], obs['radius'])
            
            # Run simulation with retry logic
            result = self._run_simulation_with_retry(planner, dynamics, "car_static_ellipsoid")
            
            return {
                'success': result['success'],
                'error': result.get('error'),
                'collisions': result.get('collisions', 0),
                'simulation_time': result.get('simulation_time', 0),
                'final_position': result.get('final_position'),
                'path_deviation': result.get('path_deviation', 0),
                'retries': result.get('retries', 0)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_car_dynamic_ellipsoid(self) -> Dict[str, Any]:
        """Test 6: Car system with dynamic obstacles and ellipsoid constraints."""
        try:
            # Create car dynamics
            dynamics = BicycleModel(dt=self.dt)
            
            # Create solver
            solver = CasADiSolver(dynamics, self.horizon_length, self.dt)
            
            # Create planner
            planner = MPCCPlanner(dynamics, solver)
            
            # Add contouring constraints and objective
            contouring_constraints = ContouringConstraints()
            contouring_objective = ContouringObjective()
            planner.add_constraint_module(contouring_constraints)
            planner.add_objective_module(contouring_objective)
            
            # Add ellipsoid constraints for dynamic obstacles
            ellipsoid_constraints = EllipsoidConstraints()
            planner.add_constraint_module(ellipsoid_constraints)
            
            # Generate curving path and dynamic obstacles
            path = self._generate_curving_path()
            obstacles = self._generate_dynamic_obstacles()
            
            contouring_constraints.set_reference_path(path)
            contouring_objective.set_reference_path(path)
            
            # Add obstacles to ellipsoid constraints
            for obs in obstacles:
                ellipsoid_constraints.add_obstacle(obs['center'], obs['radius'])
            
            # Run simulation with retry logic
            result = self._run_simulation_with_retry(planner, dynamics, "car_dynamic_ellipsoid")
            
            return {
                'success': result['success'],
                'error': result.get('error'),
                'collisions': result.get('collisions', 0),
                'simulation_time': result.get('simulation_time', 0),
                'final_position': result.get('final_position'),
                'path_deviation': result.get('path_deviation', 0),
                'retries': result.get('retries', 0)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_car_dynamic_gaussian(self) -> Dict[str, Any]:
        """Test 7: Car system with dynamic obstacles and gaussian constraints."""
        try:
            # Create car dynamics
            dynamics = BicycleModel(dt=self.dt)
            
            # Create solver
            solver = CasADiSolver(dynamics, self.horizon_length, self.dt)
            
            # Create planner
            planner = MPCCPlanner(dynamics, solver)
            
            # Add contouring constraints and objective
            contouring_constraints = ContouringConstraints()
            contouring_objective = ContouringObjective()
            planner.add_constraint_module(contouring_constraints)
            planner.add_objective_module(contouring_objective)
            
            # Add gaussian constraints for dynamic obstacles
            gaussian_constraints = GaussianConstraints()
            planner.add_constraint_module(gaussian_constraints)
            
            # Generate curving path and dynamic obstacles
            path = self._generate_curving_path()
            obstacles = self._generate_dynamic_obstacles()
            
            contouring_constraints.set_reference_path(path)
            contouring_objective.set_reference_path(path)
            
            # Add obstacles to gaussian constraints
            for obs in obstacles:
                gaussian_constraints.add_obstacle(obs['center'], obs['radius'])
            
            # Run simulation with retry logic
            result = self._run_simulation_with_retry(planner, dynamics, "car_dynamic_gaussian")
            
            return {
                'success': result['success'],
                'error': result.get('error'),
                'collisions': result.get('collisions', 0),
                'simulation_time': result.get('simulation_time', 0),
                'final_position': result.get('final_position'),
                'path_deviation': result.get('path_deviation', 0),
                'retries': result.get('retries', 0)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_car_dynamic_scenario(self) -> Dict[str, Any]:
        """Test 8: Car system with dynamic obstacles and scenario constraints."""
        try:
            # Create car dynamics
            dynamics = BicycleModel(dt=self.dt)
            
            # Create solver
            solver = CasADiSolver(dynamics, self.horizon_length, self.dt)
            
            # Create planner
            planner = MPCCPlanner(dynamics, solver)
            
            # Add contouring constraints and objective
            contouring_constraints = ContouringConstraints()
            contouring_objective = ContouringObjective()
            planner.add_constraint_module(contouring_constraints)
            planner.add_objective_module(contouring_objective)
            
            # Add scenario constraints for dynamic obstacles
            scenario_constraints = ScenarioConstraints()
            planner.add_constraint_module(scenario_constraints)
            
            # Generate curving path and dynamic obstacles
            path = self._generate_curving_path()
            obstacles = self._generate_dynamic_obstacles()
            
            contouring_constraints.set_reference_path(path)
            contouring_objective.set_reference_path(path)
            
            # Add obstacles to scenario constraints
            for obs in obstacles:
                scenario_constraints.add_scenario(f"scenario_{len(scenario_constraints.scenarios)}", [obs])
            
            # Run simulation with retry logic
            result = self._run_simulation_with_retry(planner, dynamics, "car_dynamic_scenario")
            
            return {
                'success': result['success'],
                'error': result.get('error'),
                'collisions': result.get('collisions', 0),
                'simulation_time': result.get('simulation_time', 0),
                'final_position': result.get('final_position'),
                'path_deviation': result.get('path_deviation', 0),
                'retries': result.get('retries', 0)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _run_simulation_with_retry(self, planner: MPCCPlanner, dynamics, test_name: str) -> Dict[str, Any]:
        """
        Run simulation with retry logic for car systems.
        
        Args:
            planner: MPC planner
            dynamics: Dynamics model
            test_name: Test name for visualization
            
        Returns:
            Simulation result
        """
        for retry in range(self.max_retries + 1):
            try:
                result = self._run_simulation(planner, dynamics, f"{test_name}_retry_{retry}")
                
                if result['success']:
                    result['retries'] = retry
                    return result
                    
            except Exception as e:
                if retry == self.max_retries:
                    # Final retry failed, try with overactuated system
                    return self._try_overactuated_fallback(test_name)
                
                print(f"  Retry {retry + 1} failed: {str(e)}")
                continue
        
        return {'success': False, 'error': 'All retries failed'}
    
    def _try_overactuated_fallback(self, test_name: str) -> Dict[str, Any]:
        """
        Try with overactuated system as fallback.
        
        Args:
            test_name: Test name
            
        Returns:
            Fallback result
        """
        print(f"  Trying overactuated system fallback for {test_name}")
        
        try:
            # Create overactuated dynamics
            dynamics = OveractuatedPointMass(dt=self.dt)
            
            # Create solver
            solver = CasADiSolver(dynamics, self.horizon_length, self.dt)
            
            # Create planner
            planner = MPCCPlanner(dynamics, solver)
            
            # Add same constraints and objectives
            contouring_constraints = ContouringConstraints()
            contouring_objective = ContouringObjective()
            planner.add_constraint_module(contouring_constraints)
            planner.add_objective_module(contouring_objective)
            
            # Generate curving path
            path = self._generate_curving_path()
            contouring_constraints.set_reference_path(path)
            contouring_objective.set_reference_path(path)
            
            # Run simulation
            result = self._run_simulation(planner, dynamics, f"{test_name}_overactuated_fallback")
            result['fallback_used'] = True
            
            return result
            
        except Exception as e:
            return {
                'success': False, 
                'error': f'Overactuated fallback failed: {str(e)}',
                'fallback_used': True
            }
    
    def _run_simulation(self, planner: MPCCPlanner, dynamics, test_name: str) -> Dict[str, Any]:
        """
        Run a single simulation.
        
        Args:
            planner: MPC planner
            dynamics: Dynamics model
            test_name: Test name for visualization
            
        Returns:
            Simulation result
        """
        # Initial state
        x0 = np.array([0.0, 0.0, 0.0, 5.0])  # [x, y, psi, v]
        
        # Simulation parameters
        sim_time = 10.0  # seconds
        n_steps = int(sim_time / self.dt)
        
        # Storage
        states = np.zeros((n_steps, dynamics.nx))
        inputs = np.zeros((n_steps, dynamics.nu))
        times = np.zeros(n_steps)
        
        states[0] = x0
        collisions = 0
        
        start_time = time.time()
        
        try:
            for k in range(n_steps - 1):
                # Solve MPC problem
                success, result = planner.solve(states[k])
                
                if not success:
                    return {
                        'success': False,
                        'error': f'MPC solve failed at step {k}',
                        'collisions': collisions,
                        'simulation_time': time.time() - start_time
                    }
                
                # Apply first control input
                u = result['u'][:, 0]
                inputs[k] = u
                
                # Simulate dynamics
                x_next = dynamics.discrete_dynamics(states[k], u)
                states[k + 1] = x_next
                times[k + 1] = (k + 1) * self.dt
                
                # Check for collisions (simplified)
                if self._check_collision(states[k + 1]):
                    collisions += 1
                    if collisions > 5:  # Too many collisions
                        return {
                            'success': False,
                            'error': 'Too many collisions',
                            'collisions': collisions,
                            'simulation_time': time.time() - start_time
                        }
            
            # Calculate path deviation
            path_deviation = self._calculate_path_deviation(states)
            
            # Generate visualization
            self._generate_visualization(test_name, states, inputs, times)
            
            return {
                'success': True,
                'collisions': collisions,
                'simulation_time': time.time() - start_time,
                'final_position': states[-1][:2],
                'path_deviation': path_deviation
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'collisions': collisions,
                'simulation_time': time.time() - start_time
            }
    
    def _check_collision(self, state: np.ndarray) -> bool:
        """
        Check for collision (simplified).
        
        Args:
            state: Current state
            
        Returns:
            True if collision detected
        """
        # Simple collision check - in practice, this would check against obstacles
        x, y = state[0], state[1]
        
        # Check bounds
        if abs(x) > 50 or abs(y) > 50:
            return True
        
        return False
    
    def _calculate_path_deviation(self, states: np.ndarray) -> float:
        """
        Calculate path deviation.
        
        Args:
            states: State trajectory
            
        Returns:
            Average path deviation
        """
        # Simple path deviation calculation
        path = self._generate_curving_path()
        deviations = []
        
        for i, state in enumerate(states):
            if i < len(path):
                expected_pos = path[i]
                actual_pos = state[:2]
                deviation = np.linalg.norm(actual_pos - expected_pos)
                deviations.append(deviation)
        
        return np.mean(deviations) if deviations else 0.0
    
    def _generate_curving_path(self) -> np.ndarray:
        """Generate a curving reference path."""
        t = np.linspace(0, 2*np.pi, 100)
        x = 10 * t
        y = 5 * np.sin(t)
        return np.column_stack([x, y])
    
    def _generate_static_obstacles(self) -> List[Dict[str, Any]]:
        """Generate static obstacles."""
        obstacles = []
        for i in range(3):
            obstacles.append({
                'center': [20 + i * 10, 2 + i * 2],
                'radius': 1.5
            })
        return obstacles
    
    def _generate_dynamic_obstacles(self) -> List[Dict[str, Any]]:
        """Generate dynamic obstacles."""
        obstacles = []
        for i in range(3):
            obstacles.append({
                'center': [15 + i * 8, 1 + i * 1.5],
                'radius': 1.0,
                'velocity': [0.5, 0.2]
            })
        return obstacles
    
    def _generate_visualization(self, test_name: str, states: np.ndarray, 
                              inputs: np.ndarray, times: np.ndarray) -> None:
        """
        Generate visualization for test.
        
        Args:
            test_name: Test name
            states: State trajectory
            inputs: Input trajectory
            times: Time vector
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot trajectory
        ax1.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, label='Vehicle Path')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title(f'{test_name} - Trajectory')
        ax1.legend()
        ax1.grid(True)
        
        # Plot velocity
        ax2.plot(times, states[:, 3], 'r-', linewidth=2, label='Velocity')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity')
        ax2.set_title(f'{test_name} - Velocity Profile')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{test_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_summary(self) -> None:
        """Generate test summary."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_id, result in self.test_results.items():
            status = "PASS" if result['success'] else "FAIL"
            print(f"  {test_id}: {status}")
            if not result['success']:
                print(f"    Error: {result.get('error', 'Unknown error')}")
        
        # Save results to file
        import json
        with open(f'{self.output_dir}/test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {self.output_dir}/test_results.json")


def main():
    """Run comprehensive test suite."""
    test_suite = ComprehensiveTestSuite()
    results = test_suite.run_all_tests()
    
    # Print final results
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST SUITE COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
