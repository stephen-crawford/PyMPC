"""
Scenario constraints for robust MPC planning.

This module implements scenario-based constraints that handle
multiple possible obstacle configurations and uncertain environments.
Based on the C++ scenario_module implementation.
"""

import numpy as np
import casadi as cs
from typing import Dict, List, Tuple, Optional, Any
from .base_constraint import BaseConstraint
from .scenario_optimization import ScenarioOptimizer, ScenarioStatus, ScenarioSolveStatus


class ScenarioConstraints(BaseConstraint):
    """
    Scenario constraints for robust obstacle avoidance.
    
    These constraints handle multiple scenarios of obstacle configurations
    and ensure the vehicle avoids obstacles in all scenarios.
    """
    
    def __init__(self, safety_margin: float = 1.0, max_scenarios: int = 5,
                 max_obstacles_per_scenario: int = 10, support_bound: int = 50, enabled: bool = True):
        """
        Initialize scenario constraints.
        
        Args:
            safety_margin: Safety margin around obstacles
            max_scenarios: Maximum number of scenarios
            max_obstacles_per_scenario: Maximum obstacles per scenario
            support_bound: Support bound for optimization
            enabled: Whether constraints are enabled
        """
        super().__init__("scenario_constraints", enabled)
        
        self.safety_margin = safety_margin
        self.max_scenarios = max_scenarios
        self.max_obstacles_per_scenario = max_obstacles_per_scenario
        
        # Initialize scenario optimizer
        self.scenario_optimizer = ScenarioOptimizer(
            max_scenarios=max_scenarios,
            max_iterations=10,
            convergence_tolerance=1e-6
        )
        self.scenario_optimizer.set_support_bound(support_bound)
        
        # Scenario data
        self.scenarios = []
        self.scenario_data = {}
        
        # Constraint parameters
        self.parameters = {
            'safety_margin': safety_margin,
            'max_scenarios': max_scenarios,
            'max_obstacles_per_scenario': max_obstacles_per_scenario,
            'support_bound': support_bound
        }
    
    def add_scenario(self, scenario_id: str, obstacles: List[Dict[str, Any]],
                     probability: float = 1.0) -> bool:
        """
        Add scenario with obstacles.
        
        Args:
            scenario_id: Scenario identifier
            obstacles: List of obstacle dictionaries
            probability: Scenario probability
            
        Returns:
            True if added successfully, False otherwise
        """
        if len(self.scenarios) >= self.max_scenarios:
            print(f"Warning: Maximum number of scenarios ({self.max_scenarios}) reached")
            return False
        
        if len(obstacles) > self.max_obstacles_per_scenario:
            print(f"Warning: Too many obstacles for scenario {scenario_id}")
            return False
        
        scenario = {
            'id': scenario_id,
            'obstacles': obstacles.copy(),
            'probability': probability,
            'enabled': True
        }
        
        self.scenarios.append(scenario)
        return True
    
    def remove_scenario(self, scenario_id: str) -> bool:
        """
        Remove scenario by ID.
        
        Args:
            scenario_id: Scenario ID
            
        Returns:
            True if removed, False otherwise
        """
        for i, scenario in enumerate(self.scenarios):
            if scenario['id'] == scenario_id:
                del self.scenarios[i]
                return True
        return False
    
    def clear_scenarios(self) -> None:
        """Clear all scenarios."""
        self.scenarios.clear()
    
    def set_scenario_enabled(self, scenario_id: str, enabled: bool) -> bool:
        """
        Set scenario enabled state.
        
        Args:
            scenario_id: Scenario ID
            enabled: Whether scenario is enabled
            
        Returns:
            True if found, False otherwise
        """
        for scenario in self.scenarios:
            if scenario['id'] == scenario_id:
                scenario['enabled'] = enabled
                return True
        return False
    
    def add_constraints(self, x: cs.SX, u: cs.SX, k: int, **kwargs) -> List[cs.SX]:
        """
        Add scenario constraints for time step k.
        
        Args:
            x: State variables [x, y, psi, v, ...]
            u: Input variables
            k: Time step
            **kwargs: Additional parameters
            
        Returns:
            List of constraint expressions
        """
        if not self.enabled or not self.scenarios:
            return []
        
        constraints = []
        
        # Get vehicle position
        vehicle_position = np.array([float(x[0]), float(x[1])])
        
        # Use scenario optimizer to compute active constraints
        success, optimization_result = self.scenario_optimizer.optimize_scenarios(
            vehicle_position, 
            self._get_reference_trajectory()
        )
        
        if not success:
            # Handle optimization failure
            if optimization_result['solve_status'] == ScenarioSolveStatus.INFEASIBLE:
                # Return empty constraints for infeasible case
                return []
            elif optimization_result['solve_status'] == ScenarioSolveStatus.SUPPORT_EXCEEDED:
                # Use backup plan
                return self._get_backup_constraints(x, k)
        
        # Get active constraints from optimizer
        active_constraints = self.scenario_optimizer.get_active_constraints()
        
        # Convert to CasADi constraints
        vehicle_x = x[0]  # x position
        vehicle_y = x[1]  # y position
        
        for constraint in active_constraints:
            scenario = constraint.scenario
            if scenario.idx < len(self.scenarios):
                scenario_data = self.scenarios[scenario.idx]
                if scenario_data.get('enabled', True):
                    # Get obstacle trajectory for this scenario
                    obstacle_trajectory = self.scenario_optimizer.obstacle_trajectories[scenario.idx]
                    
                    if k < len(obstacle_trajectory):
                        obs_pos = obstacle_trajectory[k]
                        obs_x, obs_y = obs_pos[0], obs_pos[1]
                        radius = scenario_data.get('obstacle_radius', 1.0)
                        
                        # Distance constraint
                        dx = vehicle_x - obs_x
                        dy = vehicle_y - obs_y
                        distance = cs.sqrt(dx*dx + dy*dy)
                        
                        safe_radius = radius + self.safety_margin
                        constraint_expr = distance - safe_radius
                        constraints.append(constraint_expr)
        
        return constraints
    
    def _get_reference_trajectory(self) -> np.ndarray:
        """
        Get reference trajectory for scenario optimization.
        
        Returns:
            Reference trajectory array
        """
        # Simple reference trajectory - in practice, this would come from path planning
        if hasattr(self, '_reference_trajectory'):
            return self._reference_trajectory
        
        # Default straight line trajectory
        horizon = 20
        trajectory = np.zeros((horizon, 2))
        for i in range(horizon):
            trajectory[i] = [i * 0.5, 0.0]  # Move forward at 0.5 m/s
        
        return trajectory
    
    def _get_backup_constraints(self, x: cs.SX, k: int) -> List[cs.SX]:
        """
        Get backup constraints when scenario optimization fails.
        
        Args:
            x: State variables
            k: Time step
            
        Returns:
            List of backup constraint expressions
        """
        constraints = []
        
        # Simple backup: avoid all obstacles with basic distance constraints
        vehicle_x = x[0]
        vehicle_y = x[1]
        
        for scenario in self.scenarios:
            if scenario.get('enabled', True):
                for obstacle in scenario.get('obstacles', []):
                    if obstacle.get('enabled', True):
                        center = obstacle.get('center', [0.0, 0.0])
                        radius = obstacle.get('radius', 1.0)
                        
                        dx = vehicle_x - center[0]
                        dy = vehicle_y - center[1]
                        distance = cs.sqrt(dx*dx + dy*dy)
                        
                        safe_radius = radius + self.safety_margin
                        constraint = distance - safe_radius
                        constraints.append(constraint)
        
        return constraints
    
    def update(self, state: np.ndarray, data: Dict[str, Any], 
               module_data: Dict[str, Any]) -> None:
        """
        Update scenario constraints.
        
        Args:
            state: Current state
            data: Real-time data
            module_data: Module-specific data
        """
        # Update scenario data from real-time data
        if 'scenarios' in data:
            self._update_scenarios_from_data(data['scenarios'])
        
        # Update constraint data
        self.constraint_data.update({
            'safety_margin': self.safety_margin,
            'scenario_count': len(self.scenarios),
            'enabled_scenarios': sum(1 for s in self.scenarios if s['enabled']),
            'total_obstacles': sum(len(s['obstacles']) for s in self.scenarios)
        })
    
    def _update_scenarios_from_data(self, scenarios_data: List[Dict[str, Any]]) -> None:
        """
        Update scenarios from real-time data.
        
        Args:
            scenarios_data: List of scenario data dictionaries
        """
        # Clear existing scenarios
        self.clear_scenarios()
        
        # Add scenarios from data
        for scenario_data in scenarios_data:
            if 'id' in scenario_data and 'obstacles' in scenario_data:
                self.add_scenario(
                    scenario_data['id'],
                    scenario_data['obstacles'],
                    scenario_data.get('probability', 1.0)
                )
    
    def visualize(self, data: Dict[str, Any], module_data: Dict[str, Any]) -> None:
        """
        Visualize scenario constraints.
        
        Args:
            data: Real-time data
            module_data: Module-specific data
        """
        print(f"Scenario Constraints:")
        print(f"  Safety margin: {self.safety_margin}")
        print(f"  Scenario count: {len(self.scenarios)}")
        print(f"  Enabled scenarios: {sum(1 for s in self.scenarios if s['enabled'])}")
        print(f"  Total obstacles: {sum(len(s['obstacles']) for s in self.scenarios)}")
        print(f"  Enabled: {self.enabled}")
        
        for i, scenario in enumerate(self.scenarios):
            print(f"  Scenario {i}: {scenario['id']} "
                  f"(probability={scenario['probability']:.2f}, "
                  f"obstacles={len(scenario['obstacles'])}, "
                  f"enabled={scenario['enabled']})")
    
    def get_scenario_info(self) -> List[Dict[str, Any]]:
        """
        Get scenario information.
        
        Returns:
            List of scenario information dictionaries
        """
        return [
            {
                'id': scenario['id'],
                'obstacles': scenario['obstacles'],
                'probability': scenario['probability'],
                'enabled': scenario['enabled']
            }
            for scenario in self.scenarios
        ]
    
    def get_constraint_info(self) -> Dict[str, Any]:
        """
        Get constraint information.
        
        Returns:
            Constraint information dictionary
        """
        return {
            'constraint_name': self.constraint_name,
            'safety_margin': self.safety_margin,
            'max_scenarios': self.max_scenarios,
            'max_obstacles_per_scenario': self.max_obstacles_per_scenario,
            'scenario_count': len(self.scenarios),
            'enabled_scenarios': sum(1 for s in self.scenarios if s['enabled']),
            'total_obstacles': sum(len(s['obstacles']) for s in self.scenarios),
            'enabled': self.enabled
        }
    
    def set_safety_margin(self, margin: float) -> None:
        """
        Set safety margin.
        
        Args:
            margin: Safety margin
        """
        self.safety_margin = margin
        self.parameters['safety_margin'] = margin
    
    def set_max_scenarios(self, max_scenarios: int) -> None:
        """
        Set maximum number of scenarios.
        
        Args:
            max_scenarios: Maximum number of scenarios
        """
        self.max_scenarios = max_scenarios
        self.parameters['max_scenarios'] = max_scenarios
        
        # Remove excess scenarios if necessary
        while len(self.scenarios) > max_scenarios:
            self.scenarios.pop()
    
    def set_max_obstacles_per_scenario(self, max_obstacles: int) -> None:
        """
        Set maximum obstacles per scenario.
        
        Args:
            max_obstacles: Maximum obstacles per scenario
        """
        self.max_obstacles_per_scenario = max_obstacles
        self.parameters['max_obstacles_per_scenario'] = max_obstacles
    
    def check_collision_in_scenario(self, position: np.ndarray, scenario_id: str) -> Tuple[bool, List[str]]:
        """
        Check collision in specific scenario.
        
        Args:
            position: Position to check [x, y]
            scenario_id: Scenario ID
            
        Returns:
            (collision_detected, colliding_obstacle_ids)
        """
        colliding_obstacles = []
        
        # Find scenario
        scenario = None
        for s in self.scenarios:
            if s['id'] == scenario_id:
                scenario = s
                break
        
        if scenario is None or not scenario['enabled']:
            return False, []
        
        # Check collision with each obstacle in scenario
        for obstacle in scenario['obstacles']:
            if not obstacle.get('enabled', True):
                continue
            
            center = obstacle.get('center', [0.0, 0.0])
            radius = obstacle.get('radius', 1.0)
            
            # Compute distance to obstacle
            dx = position[0] - center[0]
            dy = position[1] - center[1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Check collision
            safe_radius = radius + self.safety_margin
            if distance < safe_radius:
                colliding_obstacles.append(obstacle.get('id', 'unknown'))
        
        return len(colliding_obstacles) > 0, colliding_obstacles
    
    def check_collision_all_scenarios(self, position: np.ndarray) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Check collision in all scenarios.
        
        Args:
            position: Position to check [x, y]
            
        Returns:
            (any_collision, scenario_collisions)
        """
        scenario_collisions = {}
        any_collision = False
        
        for scenario in self.scenarios:
            if not scenario['enabled']:
                continue
            
            collision, colliding_obstacles = self.check_collision_in_scenario(
                position, scenario['id']
            )
            
            if collision:
                any_collision = True
                scenario_collisions[scenario['id']] = colliding_obstacles
        
        return any_collision, scenario_collisions
    
    def get_safe_distance(self, position: np.ndarray) -> float:
        """
        Get minimum safe distance across all scenarios.
        
        Args:
            position: Position to check [x, y]
            
        Returns:
            Minimum safe distance
        """
        min_distance = float('inf')
        
        for scenario in self.scenarios:
            if not scenario['enabled']:
                continue
            
            for obstacle in scenario['obstacles']:
                if not obstacle.get('enabled', True):
                    continue
                
                center = obstacle.get('center', [0.0, 0.0])
                radius = obstacle.get('radius', 1.0)
                
                # Compute distance to obstacle
                dx = position[0] - center[0]
                dy = position[1] - center[1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Safe distance
                safe_radius = radius + self.safety_margin
                min_distance = min(min_distance, distance - safe_radius)
        
        return min_distance
    
    def get_scenario_statistics(self) -> Dict[str, Any]:
        """
        Get scenario statistics.
        
        Returns:
            Statistics dictionary
        """
        total_obstacles = sum(len(s['obstacles']) for s in self.scenarios)
        enabled_obstacles = sum(
            len([obs for obs in s['obstacles'] if obs.get('enabled', True)])
            for s in self.scenarios if s['enabled']
        )
        
        return {
            'total_scenarios': len(self.scenarios),
            'enabled_scenarios': sum(1 for s in self.scenarios if s['enabled']),
            'total_obstacles': total_obstacles,
            'enabled_obstacles': enabled_obstacles,
            'average_obstacles_per_scenario': total_obstacles / max(len(self.scenarios), 1),
            'scenario_probabilities': [s['probability'] for s in self.scenarios]
        }
