"""
End-to-end MPCC with scenario constraints test.

This test demonstrates a complete implementation of Model Predictive
Contouring Control with scenario constraints for vehicle navigation
along a road while avoiding dynamic obstacles.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import time

from ..core.planner import MPCCPlanner
from ..core.dynamics import BicycleModel
from ..constraints import ContouringConstraints, ScenarioConstraints
from ..objectives import ContouringObjective, GoalObjective
from .base_test import BaseMPCTest, TestConfig, TestResult


class MPCCScenarioTest(BaseMPCTest):
    """
    Test for MPCC with scenario constraints.
    
    This test creates a scenario where a vehicle must:
    1. Follow a curved road path (contouring objective)
    2. Avoid dynamic obstacles using scenario constraints
    3. Reach a goal position
    """

    def __init__(self, config: TestConfig):
        super().__init__(config)
        
        # Test parameters
        self.road_length = 100.0
        self.road_width = 4.0
        self.num_obstacles = 3
        self.obstacle_speed = 2.0
        
        # Initialize components
        self.planner = None
        self.dynamics = None
        self.reference_path = None
        self.obstacles = []
        self.vehicle_state = None

    def setup_test_environment(self):
        """Setup test environment with road and obstacles."""
        # Create curved road path
        self.reference_path = self._create_curved_road()
        
        # Create dynamic obstacles
        self.obstacles = self._create_dynamic_obstacles()
        
        # Initialize vehicle state
        self.vehicle_state = np.array([0.0, 0.0, 0.0, 5.0])  # [x, y, psi, v]
        
        # Create dynamics model
        self.dynamics = BicycleModel(dt=0.1, wheelbase=2.5)

    def create_mpc_system(self):
        """Create MPC system with objectives and constraints."""
        # Create planner
        self.planner = MPCCPlanner(
            dynamics=self.dynamics,
            horizon_length=20,
            dt=0.1
        )
        
        # Add objectives
        contouring_obj = ContouringObjective(
            reference_path=self.reference_path,
            weight=1.0
        )
        self.planner.add_objective(contouring_obj)
        
        goal_obj = GoalObjective(
            goal_position=[self.road_length, 0.0],
            weight=0.5
        )
        self.planner.add_objective(goal_obj)
        
        # Add constraints
        contouring_constraint = ContouringConstraints(
            reference_path=self.reference_path,
            road_width=self.road_width
        )
        self.planner.add_constraint(contouring_constraint)
        
        scenario_constraint = ScenarioConstraints(
            obstacles=self.obstacles,
            safety_margin=1.0
        )
        self.planner.add_constraint(scenario_constraint)
        
        return self.planner

    def run_mpc_iteration(self, iteration: int) -> Tuple[bool, float, Dict[str, Any]]:
        """Run a single MPC iteration."""
        try:
            # Update obstacle positions
            self._update_obstacles(iteration)
            
            # Solve MPC problem
            solution = self.planner.solve(
                initial_state=self.vehicle_state,
                reference_path=self.reference_path
            )
            
            if not solution or solution.get('status') != 'optimal':
                return False, float('inf'), {'error': 'MPC solve failed'}
            
            # Extract first control action
            controls = solution.get('controls')
            if controls is None or controls.size == 0:
                return False, float('inf'), {'error': 'No controls returned'}
            
            u = controls[:, 0]  # First control action
            
            # Simulate vehicle forward
            self.vehicle_state = self.dynamics.step(self.vehicle_state, u)
            
            # Calculate error metrics
            error = self._calculate_error()
            
            # Create state information
            state = {
                'x': self.vehicle_state[0],
                'y': self.vehicle_state[1],
                'psi': self.vehicle_state[2],
                'v': self.vehicle_state[3],
                'controls': u,
                'obstacles': self.obstacles.copy(),
                'error': error
            }
            
            return True, error, state
            
        except Exception as e:
            return False, float('inf'), {'error': str(e)}

    def _create_curved_road(self) -> np.ndarray:
        """Create a curved road reference path."""
        s = np.linspace(0, 1, 100)
        x = s * self.road_length
        y = 2.0 * np.sin(0.1 * x)  # Gentle curve
        
        return np.column_stack([x, y])

    def _create_dynamic_obstacles(self) -> List[Dict[str, Any]]:
        """Create dynamic obstacles."""
        obstacles = []
        
        for i in range(self.num_obstacles):
            # Random initial positions
            x0 = np.random.uniform(20, 80)
            y0 = np.random.uniform(-2, 2)
            
            # Random velocities
            vx = np.random.uniform(-self.obstacle_speed, self.obstacle_speed)
            vy = np.random.uniform(-0.5, 0.5)
            
            obstacle = {
                'position': [x0, y0],
                'velocity': [vx, vy],
                'radius': 1.0,
                'uncertainty': [0.2, 0.2]
            }
            obstacles.append(obstacle)
        
        return obstacles

    def _update_obstacles(self, iteration: int):
        """Update obstacle positions."""
        dt = 0.1
        for obstacle in self.obstacles:
            obstacle['position'][0] += obstacle['velocity'][0] * dt
            obstacle['position'][1] += obstacle['velocity'][1] * dt

    def _calculate_error(self) -> float:
        """Calculate current error from reference path."""
        if self.reference_path is None:
            return float('inf')
        
        # Find closest point on reference path
        vehicle_pos = self.vehicle_state[:2]
        distances = np.linalg.norm(self.reference_path - vehicle_pos, axis=1)
        min_distance = np.min(distances)
        
        return min_distance

    def _check_convergence(self, error: float) -> bool:
        """Check if the test has converged."""
        return error < 0.5  # Within 0.5m of reference path

    def _check_timeout(self) -> bool:
        """Check if the test has timed out."""
        elapsed = time.time() - self.start_time
        return elapsed > 30.0  # 30 second timeout

    def get_visualization_overlay(self):
        """Get visualization overlay for the test."""
        return {
            'reference_path': {
                'x': self.reference_path[:, 0].tolist(),
                'y': self.reference_path[:, 1].tolist(),
                'color': 'blue',
                'linewidth': 2,
                'label': 'Reference Path'
            },
            'obstacles': [
                {
                    'x': obs['position'][0],
                    'y': obs['position'][1],
                    'radius': obs['radius'],
                    'color': 'red',
                    'alpha': 0.7
                }
                for obs in self.obstacles
            ]
        }


def create_mpcc_scenario_test() -> MPCCScenarioTest:
    """Create and configure the MPCC scenario test."""
    config = TestConfig(
        test_name="mpcc_scenario_test",
        description="MPCC with scenario constraints for road following with dynamic obstacles",
        timeout=30.0,
        max_iterations=100,
        goal_tolerance=0.5,
        enable_visualization=True,
        save_outputs=True
    )
    
    return MPCCScenarioTest(config)


if __name__ == "__main__":
    # Run the test
    test = create_mpcc_scenario_test()
    result = test.run_test()
    
    print("Test Results:")
    print(f"Success: {result.success}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Iterations: {result.iterations}")
    print(f"Final Error: {result.final_error:.4f}")
    print(f"Convergence Rate: {result.convergence_rate:.2%}")
    
    if result.error_log:
        print("Errors:")
        for error in result.error_log:
            print(f"  - {error}")
