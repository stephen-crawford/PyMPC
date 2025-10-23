"""
Scenario optimization strategy for robust MPC planning.

This module implements the scenario optimization strategy based on
the C++ scenario_module implementation for robust constraint handling.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from enum import Enum


class ScenarioStatus(Enum):
    """Scenario status enumeration."""
    SUCCESS = 0
    PROJECTED_SUCCESS = 1
    BACKUP_PLAN = 2
    INFEASIBLE = 3
    DATA_MISSING = 4
    RESET = 5


class ScenarioSolveStatus(Enum):
    """Scenario solve status enumeration."""
    SUCCESS = 0
    INFEASIBLE = 1
    SUPPORT_EXCEEDED = 2
    NONZERO_SLACK = 3


class ObstacleType(Enum):
    """Obstacle type enumeration."""
    STATIC = 0
    DYNAMIC = 1
    RANGE = 2


class ConstraintSide(Enum):
    """Constraint side enumeration."""
    BOTTOM = 0
    TOP = 1
    UNDEFINED = 2


class Scenario:
    """Scenario data structure."""
    
    def __init__(self, idx: int, obstacle_idx: int):
        """
        Initialize scenario.
        
        Args:
            idx: Scenario index
            obstacle_idx: Obstacle index
        """
        self.idx = idx
        self.obstacle_idx = obstacle_idx


class ScenarioConstraint:
    """Scenario constraint data structure."""
    
    def __init__(self, scenario: Scenario, obstacle_type: ObstacleType, 
                 constraint_side: ConstraintSide):
        """
        Initialize scenario constraint.
        
        Args:
            scenario: Scenario reference
            obstacle_type: Type of obstacle
            constraint_side: Side of constraint
        """
        self.scenario = scenario
        self.obstacle_type = obstacle_type
        self.constraint_side = constraint_side
    
    def get_halfspace_index(self, sample_size: int) -> int:
        """
        Get halfspace index for constraint.
        
        Args:
            sample_size: Size of sample
            
        Returns:
            Halfspace index
        """
        if self.obstacle_type == ObstacleType.DYNAMIC:
            return sample_size * self.scenario.obstacle_idx + self.scenario.idx
        else:
            return self.scenario.idx


class SupportSubsample:
    """Support subsample data structure."""
    
    def __init__(self, initial_size: int = 150):
        """
        Initialize support subsample.
        
        Args:
            initial_size: Initial size for allocation
        """
        self.support_indices = []
        self.scenarios = []
        self.size = 0
        
        # Pre-allocate for efficiency (Python lists don't have reserve method)
        # self.support_indices.reserve(initial_size)
        # self.scenarios.reserve(initial_size)
    
    def add(self, scenario: Scenario) -> None:
        """
        Add scenario to support subsample.
        
        Args:
            scenario: Scenario to add
        """
        # No duplicates
        if self.contains_scenario(scenario):
            return
        
        self.support_indices.append(scenario.idx)
        self.scenarios.append(scenario)
        self.size += 1
    
    def reset(self) -> None:
        """Reset support subsample."""
        self.size = 0
        self.support_indices.clear()
        self.scenarios.clear()
    
    def contains_scenario(self, scenario: Scenario) -> bool:
        """
        Check if scenario is already in support.
        
        Args:
            scenario: Scenario to check
            
        Returns:
            True if scenario is in support
        """
        return scenario.idx in self.support_indices[:self.size]
    
    def merge_with(self, other: 'SupportSubsample') -> None:
        """
        Merge with another support subsample.
        
        Args:
            other: Other support subsample to merge with
        """
        for i in range(other.size):
            if not self.contains_scenario(other.scenarios[i]):
                self.add(other.scenarios[i])
    
    def print_info(self) -> None:
        """Print support subsample information."""
        print("Support Subsample:")
        for i in range(self.size):
            print(f"  Scenario {self.scenarios[i].idx}, Obstacle {self.scenarios[i].obstacle_idx}")
    
    def print_update(self, solver_id: int, bound: int, iterations: int) -> None:
        """
        Print update information.
        
        Args:
            solver_id: Solver ID
            bound: Bound value
            iterations: Number of iterations
        """
        print(f"[Solver {solver_id}] SQP ({iterations}): Support = {self.size}/{bound}")


class ScenarioOptimizer:
    """Scenario optimization strategy implementation."""
    
    def __init__(self, max_scenarios: int = 100, max_iterations: int = 10,
                 convergence_tolerance: float = 1e-6):
        """
        Initialize scenario optimizer.
        
        Args:
            max_scenarios: Maximum number of scenarios
            max_iterations: Maximum optimization iterations
            convergence_tolerance: Convergence tolerance
        """
        self.max_scenarios = max_scenarios
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        
        # Optimization state
        self.active_constraints = SupportSubsample()
        self.infeasible_scenarios = SupportSubsample()
        self.scenario_status = ScenarioStatus.SUCCESS
        self.solve_status = ScenarioSolveStatus.SUCCESS
        
        # Scenario data
        self.scenarios = []
        self.scenario_constraints = []
        self.obstacle_trajectories = []
        
        # Optimization parameters
        self.support_bound = 50
        self.removal_count = 10
        self.sampling_count = 20
    
    def add_scenario(self, obstacle_idx: int, trajectory: np.ndarray) -> int:
        """
        Add scenario to optimization.
        
        Args:
            obstacle_idx: Obstacle index
            trajectory: Obstacle trajectory
            
        Returns:
            Scenario index
        """
        scenario_idx = len(self.scenarios)
        scenario = Scenario(scenario_idx, obstacle_idx)
        self.scenarios.append(scenario)
        self.obstacle_trajectories.append(trajectory)
        
        return scenario_idx
    
    def compute_active_constraints(self, vehicle_position: np.ndarray, 
                                 vehicle_radius: float = 1.0) -> bool:
        """
        Compute active constraints for current vehicle position.
        
        Args:
            vehicle_position: Current vehicle position [x, y]
            vehicle_radius: Vehicle radius
            
        Returns:
            True if constraints computed successfully
        """
        self.active_constraints.reset()
        self.infeasible_scenarios.reset()
        
        # Check each scenario for constraint activity
        for i, scenario in enumerate(self.scenarios):
            trajectory = self.obstacle_trajectories[i]
            
            # Compute distance to obstacle at each time step
            distances = []
            for t in range(len(trajectory)):
                obstacle_pos = trajectory[t]
                distance = np.linalg.norm(vehicle_position - obstacle_pos)
                distances.append(distance)
            
            min_distance = min(distances)
            
            # Check if constraint is active
            if min_distance <= vehicle_radius + 0.5:  # Safety margin
                self.active_constraints.add(scenario)
            elif min_distance > vehicle_radius + 2.0:  # Far from obstacle
                # Scenario is infeasible (too far)
                self.infeasible_scenarios.add(scenario)
        
        return len(self.active_constraints.scenarios) > 0
    
    def optimize_scenarios(self, vehicle_position: np.ndarray, 
                          reference_trajectory: np.ndarray = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Optimize scenarios using iterative approach.
        
        Args:
            vehicle_position: Current vehicle position
            reference_trajectory: Reference trajectory
            
        Returns:
            (success, optimization_result)
        """
        self.scenario_status = ScenarioStatus.SUCCESS
        
        # Initialize optimization
        iteration = 0
        converged = False
        self._previous_active_size = 0
        
        while iteration < self.max_iterations and not converged:
            # Compute active constraints
            constraints_active = self.compute_active_constraints(vehicle_position)
            
            if not constraints_active:
                # No active constraints, optimization complete
                converged = True
                break
            
            # Check if support exceeds bound
            if self.active_constraints.size > self.support_bound:
                # Remove scenarios by distance
                self._remove_scenarios_by_distance(vehicle_position)
            
            # Check convergence
            if iteration > 0:
                # Check if active constraints changed significantly
                constraint_change = abs(self.active_constraints.size - self._previous_active_size)
                if constraint_change < self.convergence_tolerance:
                    converged = True
                    break
            
            self._previous_active_size = self.active_constraints.size
            iteration += 1
        
        # Determine final status
        if not converged and iteration >= self.max_iterations:
            self.scenario_status = ScenarioStatus.INFEASIBLE
            self.solve_status = ScenarioSolveStatus.INFEASIBLE
        elif self.active_constraints.size > self.support_bound:
            self.scenario_status = ScenarioStatus.BACKUP_PLAN
            self.solve_status = ScenarioSolveStatus.SUPPORT_EXCEEDED
        else:
            self.scenario_status = ScenarioStatus.SUCCESS
            self.solve_status = ScenarioSolveStatus.SUCCESS
        
        result = {
            'success': self.scenario_status == ScenarioStatus.SUCCESS,
            'status': self.scenario_status,
            'solve_status': self.solve_status,
            'active_constraints': self.active_constraints.size,
            'infeasible_scenarios': self.infeasible_scenarios.size,
            'iterations': iteration,
            'converged': converged
        }
        
        return self.scenario_status == ScenarioStatus.SUCCESS, result
    
    def _remove_scenarios_by_distance(self, vehicle_position: np.ndarray) -> None:
        """
        Remove scenarios based on distance from vehicle.
        
        Args:
            vehicle_position: Current vehicle position
        """
        # Compute distances for all active scenarios
        distances = []
        for scenario in self.active_constraints.scenarios:
            trajectory = self.obstacle_trajectories[scenario.idx]
            min_distance = float('inf')
            
            for t in range(len(trajectory)):
                obstacle_pos = trajectory[t]
                distance = np.linalg.norm(vehicle_position - obstacle_pos)
                min_distance = min(min_distance, distance)
            
            distances.append((min_distance, scenario))
        
        # Sort by distance (furthest first)
        distances.sort(key=lambda x: x[0], reverse=True)
        
        # Remove furthest scenarios
        removal_count = min(self.removal_count, len(distances))
        for i in range(removal_count):
            scenario_to_remove = distances[i][1]
            if scenario_to_remove in self.active_constraints.scenarios:
                self.active_constraints.scenarios.remove(scenario_to_remove)
                self.active_constraints.size -= 1
    
    def get_active_constraints(self) -> List[ScenarioConstraint]:
        """
        Get active scenario constraints.
        
        Returns:
            List of active scenario constraints
        """
        constraints = []
        
        for scenario in self.active_constraints.scenarios:
            # Create constraint for each active scenario
            constraint = ScenarioConstraint(
                scenario, 
                ObstacleType.DYNAMIC, 
                ConstraintSide.BOTTOM
            )
            constraints.append(constraint)
        
        return constraints
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """
        Get optimization statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_scenarios': len(self.scenarios),
            'active_constraints': self.active_constraints.size,
            'infeasible_scenarios': self.infeasible_scenarios.size,
            'scenario_status': self.scenario_status,
            'solve_status': self.solve_status,
            'support_bound': self.support_bound,
            'max_iterations': self.max_iterations
        }
    
    def reset_optimization(self) -> None:
        """Reset optimization state."""
        self.active_constraints.reset()
        self.infeasible_scenarios.reset()
        self.scenario_status = ScenarioStatus.RESET
        self.solve_status = ScenarioSolveStatus.SUCCESS
    
    def set_support_bound(self, bound: int) -> None:
        """
        Set support bound.
        
        Args:
            bound: Support bound value
        """
        self.support_bound = bound
    
    def set_removal_count(self, count: int) -> None:
        """
        Set removal count.
        
        Args:
            count: Number of scenarios to remove per iteration
        """
        self.removal_count = count
    
    def set_sampling_count(self, count: int) -> None:
        """
        Set sampling count.
        
        Args:
            count: Number of samples to generate
        """
        self.sampling_count = count
