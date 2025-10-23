"""
Decomposition constraints for MPC planning.

This module implements decomposition-based constraints that break down
complex obstacle avoidance problems into simpler subproblems.
"""

import numpy as np
import casadi as cs
from typing import Dict, List, Optional, Any
from .base_constraint import BaseConstraint


class DecompositionConstraints(BaseConstraint):
    """
    Decomposition constraints for complex obstacle avoidance.
    
    These constraints use decomposition techniques to break down
    complex obstacle avoidance problems into simpler subproblems.
    """
    
    def __init__(self, safety_margin: float = 1.0, max_obstacles: int = 10,
                 decomposition_method: str = "sequential", enabled: bool = True):
        """
        Initialize decomposition constraints.
        
        Args:
            safety_margin: Safety margin around obstacles
            max_obstacles: Maximum number of obstacles
            decomposition_method: Decomposition method ("sequential", "parallel", "hierarchical")
            enabled: Whether constraints are enabled
        """
        super().__init__("decomposition_constraints", enabled)
        
        self.safety_margin = safety_margin
        self.max_obstacles = max_obstacles
        self.decomposition_method = decomposition_method
        
        # Obstacle data
        self.obstacles = []
        self.obstacle_groups = []
        self.constraint_data = {}
        
        # Constraint parameters
        self.parameters = {
            'safety_margin': safety_margin,
            'max_obstacles': max_obstacles,
            'decomposition_method': decomposition_method
        }
    
    def add_obstacle(self, center: np.ndarray, radius: float, 
                     priority: int = 1, obstacle_id: Optional[str] = None) -> str:
        """
        Add obstacle to decomposition.
        
        Args:
            center: Obstacle center [x, y]
            radius: Obstacle radius
            priority: Obstacle priority (higher = more important)
            obstacle_id: Optional obstacle ID
            
        Returns:
            Obstacle ID
        """
        if len(self.obstacles) >= self.max_obstacles:
            print(f"Warning: Maximum number of obstacles ({self.max_obstacles}) reached")
            return None
        
        if obstacle_id is None:
            obstacle_id = f"decomp_obstacle_{len(self.obstacles)}"
        
        obstacle = {
            'id': obstacle_id,
            'center': np.array(center),
            'radius': radius,
            'priority': priority,
            'enabled': True
        }
        
        self.obstacles.append(obstacle)
        self._update_obstacle_groups()
        
        return obstacle_id
    
    def _update_obstacle_groups(self) -> None:
        """Update obstacle groups based on decomposition method."""
        if self.decomposition_method == "sequential":
            # Group obstacles by priority
            self.obstacle_groups = []
            priorities = sorted(set(obs['priority'] for obs in self.obstacles))
            
            for priority in priorities:
                group = [obs for obs in self.obstacles if obs['priority'] == priority]
                self.obstacle_groups.append(group)
        
        elif self.decomposition_method == "parallel":
            # Group obstacles by spatial proximity
            self.obstacle_groups = self._group_by_proximity()
        
        elif self.decomposition_method == "hierarchical":
            # Group obstacles hierarchically
            self.obstacle_groups = self._group_hierarchically()
    
    def _group_by_proximity(self, max_distance: float = 10.0) -> List[List[Dict]]:
        """
        Group obstacles by spatial proximity.
        
        Args:
            max_distance: Maximum distance for grouping
            
        Returns:
            List of obstacle groups
        """
        groups = []
        used_obstacles = set()
        
        for obstacle in self.obstacles:
            if obstacle['id'] in used_obstacles:
                continue
            
            group = [obstacle]
            used_obstacles.add(obstacle['id'])
            
            # Find nearby obstacles
            for other_obs in self.obstacles:
                if other_obs['id'] in used_obstacles:
                    continue
                
                distance = np.linalg.norm(obstacle['center'] - other_obs['center'])
                if distance <= max_distance:
                    group.append(other_obs)
                    used_obstacles.add(other_obs['id'])
            
            groups.append(group)
        
        return groups
    
    def _group_hierarchically(self) -> List[List[Dict]]:
        """
        Group obstacles hierarchically.
        
        Returns:
            List of obstacle groups
        """
        # Simple hierarchical grouping based on priority and size
        groups = []
        
        # High priority obstacles (individual groups)
        high_priority = [obs for obs in self.obstacles if obs['priority'] >= 3]
        for obs in high_priority:
            groups.append([obs])
        
        # Medium priority obstacles (grouped by proximity)
        medium_priority = [obs for obs in self.obstacles if obs['priority'] == 2]
        if medium_priority:
            groups.append(medium_priority)
        
        # Low priority obstacles (single group)
        low_priority = [obs for obs in self.obstacles if obs['priority'] <= 1]
        if low_priority:
            groups.append(low_priority)
        
        return groups
    
    def add_constraints(self, x: cs.SX, u: cs.SX, k: int, **kwargs) -> List[cs.SX]:
        """
        Add decomposition constraints for time step k.
        
        Args:
            x: State variables [x, y, psi, v, ...]
            u: Input variables
            k: Time step
            **kwargs: Additional parameters
            
        Returns:
            List of constraint expressions
        """
        if not self.enabled or not self.obstacles:
            return []
        
        constraints = []
        
        # Get vehicle position
        vehicle_x = x[0]  # x position
        vehicle_y = x[1]  # y position
        
        # Apply decomposition method
        if self.decomposition_method == "sequential":
            constraints.extend(self._add_sequential_constraints(vehicle_x, vehicle_y))
        elif self.decomposition_method == "parallel":
            constraints.extend(self._add_parallel_constraints(vehicle_x, vehicle_y))
        elif self.decomposition_method == "hierarchical":
            constraints.extend(self._add_hierarchical_constraints(vehicle_x, vehicle_y))
        
        return constraints
    
    def _add_sequential_constraints(self, vehicle_x: cs.SX, vehicle_y: cs.SX) -> List[cs.SX]:
        """
        Add sequential decomposition constraints.
        
        Args:
            vehicle_x: Vehicle x position
            vehicle_y: Vehicle y position
            
        Returns:
            List of constraint expressions
        """
        constraints = []
        
        # Process obstacles in priority order
        for group in self.obstacle_groups:
            for obstacle in group:
                if not obstacle['enabled']:
                    continue
                
                # Distance constraint
                dx = vehicle_x - obstacle['center'][0]
                dy = vehicle_y - obstacle['center'][1]
                distance = cs.sqrt(dx*dx + dy*dy)
                
                safe_radius = obstacle['radius'] + self.safety_margin
                constraint = distance - safe_radius
                constraints.append(constraint)
        
        return constraints
    
    def _add_parallel_constraints(self, vehicle_x: cs.SX, vehicle_y: cs.SX) -> List[cs.SX]:
        """
        Add parallel decomposition constraints.
        
        Args:
            vehicle_x: Vehicle x position
            vehicle_y: Vehicle y position
            
        Returns:
            List of constraint expressions
        """
        constraints = []
        
        # Process each group independently
        for group in self.obstacle_groups:
            group_constraints = []
            
            for obstacle in group:
                if not obstacle['enabled']:
                    continue
                
                # Distance constraint
                dx = vehicle_x - obstacle['center'][0]
                dy = vehicle_y - obstacle['center'][1]
                distance = cs.sqrt(dx*dx + dy*dy)
                
                safe_radius = obstacle['radius'] + self.safety_margin
                constraint = distance - safe_radius
                group_constraints.append(constraint)
            
            # Combine group constraints (at least one must be satisfied)
            if group_constraints:
                # Use soft constraints for parallel processing
                for constraint in group_constraints:
                    constraints.append(constraint)
        
        return constraints
    
    def _add_hierarchical_constraints(self, vehicle_x: cs.SX, vehicle_y: cs.SX) -> List[cs.SX]:
        """
        Add hierarchical decomposition constraints.
        
        Args:
            vehicle_x: Vehicle x position
            vehicle_y: Vehicle y position
            
        Returns:
            List of constraint expressions
        """
        constraints = []
        
        # Process groups in hierarchical order
        for i, group in enumerate(self.obstacle_groups):
            group_constraints = []
            
            for obstacle in group:
                if not obstacle['enabled']:
                    continue
                
                # Distance constraint
                dx = vehicle_x - obstacle['center'][0]
                dy = vehicle_y - obstacle['center'][1]
                distance = cs.sqrt(dx*dx + dy*dy)
                
                # Adjust safety margin based on hierarchy level
                hierarchy_factor = 1.0 + 0.1 * i  # Higher levels get more margin
                safe_radius = obstacle['radius'] + self.safety_margin * hierarchy_factor
                constraint = distance - safe_radius
                group_constraints.append(constraint)
            
            # Add group constraints
            constraints.extend(group_constraints)
        
        return constraints
    
    def update(self, state: np.ndarray, data: Dict[str, Any], 
               module_data: Dict[str, Any]) -> None:
        """
        Update decomposition constraints.
        
        Args:
            state: Current state
            data: Real-time data
            module_data: Module-specific data
        """
        # Update obstacle data from real-time data
        if 'obstacles' in data:
            self._update_obstacles_from_data(data['obstacles'])
        
        # Update constraint data
        self.constraint_data.update({
            'safety_margin': self.safety_margin,
            'decomposition_method': self.decomposition_method,
            'obstacle_count': len(self.obstacles),
            'group_count': len(self.obstacle_groups),
            'enabled_obstacles': sum(1 for obs in self.obstacles if obs['enabled'])
        })
    
    def _update_obstacles_from_data(self, obstacles_data: List[Dict[str, Any]]) -> None:
        """
        Update obstacles from real-time data.
        
        Args:
            obstacles_data: List of obstacle data dictionaries
        """
        # Clear existing obstacles
        self.obstacles.clear()
        self.obstacle_groups.clear()
        
        # Add obstacles from data
        for obs_data in obstacles_data:
            self.add_obstacle(
                obs_data.get('center', [0.0, 0.0]),
                obs_data.get('radius', 1.0),
                obs_data.get('priority', 1),
                obs_data.get('id')
            )
    
    def visualize(self, data: Dict[str, Any], module_data: Dict[str, Any]) -> None:
        """
        Visualize decomposition constraints.
        
        Args:
            data: Real-time data
            module_data: Module-specific data
        """
        print("Decomposition Constraints:")
        print(f"  Safety margin: {self.safety_margin}")
        print(f"  Decomposition method: {self.decomposition_method}")
        print(f"  Obstacle count: {len(self.obstacles)}")
        print(f"  Group count: {len(self.obstacle_groups)}")
        print(f"  Enabled obstacles: {sum(1 for obs in self.obstacles if obs['enabled'])}")
        print(f"  Enabled: {self.enabled}")
        
        for i, group in enumerate(self.obstacle_groups):
            print(f"  Group {i}: {len(group)} obstacles")
            for obs in group:
                print(f"    Obstacle {obs['id']}: center={obs['center']}, "
                      f"radius={obs['radius']}, priority={obs['priority']}")
    
    def get_constraint_info(self) -> Dict[str, Any]:
        """
        Get constraint information.
        
        Returns:
            Constraint information dictionary
        """
        return {
            'constraint_name': self.constraint_name,
            'safety_margin': self.safety_margin,
            'decomposition_method': self.decomposition_method,
            'max_obstacles': self.max_obstacles,
            'obstacle_count': len(self.obstacles),
            'group_count': len(self.obstacle_groups),
            'enabled_obstacles': sum(1 for obs in self.obstacles if obs['enabled']),
            'enabled': self.enabled
        }
    
    def set_decomposition_method(self, method: str) -> None:
        """
        Set decomposition method.
        
        Args:
            method: Decomposition method
        """
        if method not in ["sequential", "parallel", "hierarchical"]:
            raise ValueError(f"Unknown decomposition method: {method}")
        
        self.decomposition_method = method
        self.parameters['decomposition_method'] = method
        self._update_obstacle_groups()
    
    def get_obstacle_groups(self) -> List[List[Dict[str, Any]]]:
        """
        Get obstacle groups.
        
        Returns:
            List of obstacle groups
        """
        return self.obstacle_groups.copy()
    
    def get_group_statistics(self) -> Dict[str, Any]:
        """
        Get group statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_groups': len(self.obstacle_groups),
            'group_sizes': [len(group) for group in self.obstacle_groups],
            'average_group_size': np.mean([len(group) for group in self.obstacle_groups]) if self.obstacle_groups else 0,
            'max_group_size': max([len(group) for group in self.obstacle_groups]) if self.obstacle_groups else 0,
            'min_group_size': min([len(group) for group in self.obstacle_groups]) if self.obstacle_groups else 0
        }
