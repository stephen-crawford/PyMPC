"""
Guidance constraints for MPC planning.

This module implements guidance constraints that provide high-level
navigation and path planning constraints for autonomous vehicles.
"""

import numpy as np
import casadi as cs
from typing import Dict, List, Optional, Any
from .base_constraint import BaseConstraint


class GuidanceConstraints(BaseConstraint):
    """
    Guidance constraints for high-level navigation.
    
    These constraints provide guidance for autonomous vehicles
    including lane keeping, intersection handling, and traffic rules.
    """
    
    def __init__(self, lane_width: float = 3.5, safety_margin: float = 0.5,
                 max_speed: float = 15.0, enabled: bool = True):
        """
        Initialize guidance constraints.
        
        Args:
            lane_width: Standard lane width
            safety_margin: Safety margin from lane boundaries
            max_speed: Maximum allowed speed
            enabled: Whether constraints are enabled
        """
        super().__init__("guidance_constraints", enabled)
        
        self.lane_width = lane_width
        self.safety_margin = safety_margin
        self.max_speed = max_speed
        
        # Guidance data
        self.lanes = []
        self.intersections = []
        self.traffic_rules = []
        self.guidance_data = {}
        
        # Constraint parameters
        self.parameters = {
            'lane_width': lane_width,
            'safety_margin': safety_margin,
            'max_speed': max_speed
        }
    
    def add_lane(self, start_point: np.ndarray, end_point: np.ndarray,
                 lane_id: Optional[str] = None, lane_type: str = "normal") -> str:
        """
        Add lane constraint.
        
        Args:
            start_point: Lane start point [x, y]
            end_point: Lane end point [x, y]
            lane_id: Optional lane ID
            lane_type: Type of lane ("normal", "highway", "residential")
            
        Returns:
            Lane ID
        """
        if lane_id is None:
            lane_id = f"lane_{len(self.lanes)}"
        
        # Calculate lane direction and width
        direction = end_point - start_point
        length = np.linalg.norm(direction)
        direction = direction / length if length > 0 else np.array([1.0, 0.0])
        
        # Perpendicular direction for lane boundaries
        perp_direction = np.array([-direction[1], direction[0]])
        
        lane = {
            'id': lane_id,
            'start_point': np.array(start_point),
            'end_point': np.array(end_point),
            'direction': direction,
            'perp_direction': perp_direction,
            'length': length,
            'width': self.lane_width,
            'type': lane_type,
            'enabled': True
        }
        
        self.lanes.append(lane)
        return lane_id
    
    def add_intersection(self, center: np.ndarray, radius: float,
                         intersection_id: Optional[str] = None) -> str:
        """
        Add intersection constraint.
        
        Args:
            center: Intersection center [x, y]
            radius: Intersection radius
            intersection_id: Optional intersection ID
            
        Returns:
            Intersection ID
        """
        if intersection_id is None:
            intersection_id = f"intersection_{len(self.intersections)}"
        
        intersection = {
            'id': intersection_id,
            'center': np.array(center),
            'radius': radius,
            'enabled': True
        }
        
        self.intersections.append(intersection)
        return intersection_id
    
    def add_traffic_rule(self, rule_type: str, parameters: Dict[str, Any],
                         rule_id: Optional[str] = None) -> str:
        """
        Add traffic rule constraint.
        
        Args:
            rule_type: Type of traffic rule
            parameters: Rule parameters
            rule_id: Optional rule ID
            
        Returns:
            Rule ID
        """
        if rule_id is None:
            rule_id = f"rule_{len(self.traffic_rules)}"
        
        rule = {
            'id': rule_id,
            'type': rule_type,
            'parameters': parameters,
            'enabled': True
        }
        
        self.traffic_rules.append(rule)
        return rule_id
    
    def add_constraints(self, x: cs.SX, u: cs.SX, k: int, **kwargs) -> List[cs.SX]:
        """
        Add guidance constraints for time step k.
        
        Args:
            x: State variables [x, y, psi, v, ...]
            u: Input variables
            k: Time step
            **kwargs: Additional parameters
            
        Returns:
            List of constraint expressions
        """
        if not self.enabled:
            return []
        
        constraints = []
        
        # Get vehicle state
        vehicle_x = x[0]  # x position
        vehicle_y = x[1]  # y position
        # vehicle_psi = x[2]  # heading angle
        vehicle_v = x[3]  # velocity
        
        # Add lane constraints
        constraints.extend(self._add_lane_constraints(vehicle_x, vehicle_y))
        
        # Add intersection constraints
        constraints.extend(self._add_intersection_constraints(vehicle_x, vehicle_y))
        
        # Add traffic rule constraints
        constraints.extend(self._add_traffic_rule_constraints(vehicle_x, vehicle_y, vehicle_v))
        
        return constraints
    
    def _add_lane_constraints(self, vehicle_x: cs.SX, vehicle_y: cs.SX) -> List[cs.SX]:
        """
        Add lane keeping constraints.
        
        Args:
            vehicle_x: Vehicle x position
            vehicle_y: Vehicle y position
            
        Returns:
            List of constraint expressions
        """
        constraints = []
        
        for lane in self.lanes:
            if not lane['enabled']:
                continue
            
            # Check if vehicle is within lane bounds
            start = lane['start_point']
            # end = lane['end_point']
            direction = lane['direction']
            perp_direction = lane['perp_direction']
            
            # Vector from lane start to vehicle
            to_vehicle = cs.vertcat(vehicle_x - start[0], vehicle_y - start[1])
            
            # Project onto lane direction to check if within lane length
            longitudinal_distance = cs.dot(to_vehicle, direction)
            
            # Project onto perpendicular direction for lateral distance
            lateral_distance = cs.dot(to_vehicle, perp_direction)
            
            # Lane width constraint
            half_width = lane['width'] / 2.0 - self.safety_margin
            constraints.append(lateral_distance + half_width)  # Left boundary
            constraints.append(half_width - lateral_distance)  # Right boundary
            
            # Lane length constraint (if within lane)
            if longitudinal_distance >= 0 and longitudinal_distance <= lane['length']:
                # Vehicle is within lane length, apply lateral constraints
                pass  # Already applied above
        
        return constraints
    
    def _add_intersection_constraints(self, vehicle_x: cs.SX, vehicle_y: cs.SX) -> List[cs.SX]:
        """
        Add intersection constraints.
        
        Args:
            vehicle_x: Vehicle x position
            vehicle_y: Vehicle y position
            
        Returns:
            List of constraint expressions
        """
        constraints = []
        
        for intersection in self.intersections:
            if not intersection['enabled']:
                continue
            
            # Distance to intersection center
            dx = vehicle_x - intersection['center'][0]
            dy = vehicle_y - intersection['center'][1]
            distance = cs.sqrt(dx*dx + dy*dy)
            
            # Speed limit in intersection
            if distance <= intersection['radius']:
                # Apply intersection-specific constraints
                # This could include speed limits, right-of-way rules, etc.
                pass
        
        return constraints
    
    def _add_traffic_rule_constraints(self, vehicle_x: cs.SX, vehicle_y: cs.SX, 
                                    vehicle_v: cs.SX) -> List[cs.SX]:
        """
        Add traffic rule constraints.
        
        Args:
            vehicle_x: Vehicle x position
            vehicle_y: Vehicle y position
            vehicle_v: Vehicle velocity
            
        Returns:
            List of constraint expressions
        """
        constraints = []
        
        for rule in self.traffic_rules:
            if not rule['enabled']:
                continue
            
            rule_type = rule['type']
            params = rule['parameters']
            
            if rule_type == "speed_limit":
                # Speed limit constraint
                max_speed = params.get('max_speed', self.max_speed)
                constraints.append(max_speed - vehicle_v)
            
            elif rule_type == "stop_sign":
                # Stop sign constraint
                stop_position = params.get('position', [0.0, 0.0])
                stop_radius = params.get('radius', 5.0)
                
                dx = vehicle_x - stop_position[0]
                dy = vehicle_y - stop_position[1]
                distance = cs.sqrt(dx*dx + dy*dy)
                
                if distance <= stop_radius:
                    # Vehicle must stop
                    constraints.append(vehicle_v)  # v <= 0
            
            elif rule_type == "yield":
                # Yield constraint
                yield_position = params.get('position', [0.0, 0.0])
                yield_radius = params.get('radius', 3.0)
                
                dx = vehicle_x - yield_position[0]
                dy = vehicle_y - yield_position[1]
                distance = cs.sqrt(dx*dx + dy*dy)
                
                if distance <= yield_radius:
                    # Reduce speed in yield zone
                    max_yield_speed = params.get('max_speed', 5.0)
                    constraints.append(max_yield_speed - vehicle_v)
        
        return constraints
    
    def update(self, state: np.ndarray, data: Dict[str, Any], 
               module_data: Dict[str, Any]) -> None:
        """
        Update guidance constraints.
        
        Args:
            state: Current state
            data: Real-time data
            module_data: Module-specific data
        """
        # Update guidance data from real-time data
        if 'lanes' in data:
            self._update_lanes_from_data(data['lanes'])
        
        if 'intersections' in data:
            self._update_intersections_from_data(data['intersections'])
        
        if 'traffic_rules' in data:
            self._update_traffic_rules_from_data(data['traffic_rules'])
        
        # Update constraint data
        self.guidance_data.update({
            'lane_count': len(self.lanes),
            'intersection_count': len(self.intersections),
            'traffic_rule_count': len(self.traffic_rules),
            'lane_width': self.lane_width,
            'safety_margin': self.safety_margin,
            'max_speed': self.max_speed
        })
    
    def _update_lanes_from_data(self, lanes_data: List[Dict[str, Any]]) -> None:
        """
        Update lanes from real-time data.
        
        Args:
            lanes_data: List of lane data dictionaries
        """
        # Clear existing lanes
        self.lanes.clear()
        
        # Add lanes from data
        for lane_data in lanes_data:
            self.add_lane(
                lane_data.get('start_point', [0.0, 0.0]),
                lane_data.get('end_point', [10.0, 0.0]),
                lane_data.get('id'),
                lane_data.get('type', 'normal')
            )
    
    def _update_intersections_from_data(self, intersections_data: List[Dict[str, Any]]) -> None:
        """
        Update intersections from real-time data.
        
        Args:
            intersections_data: List of intersection data dictionaries
        """
        # Clear existing intersections
        self.intersections.clear()
        
        # Add intersections from data
        for intersection_data in intersections_data:
            self.add_intersection(
                intersection_data.get('center', [0.0, 0.0]),
                intersection_data.get('radius', 5.0),
                intersection_data.get('id')
            )
    
    def _update_traffic_rules_from_data(self, rules_data: List[Dict[str, Any]]) -> None:
        """
        Update traffic rules from real-time data.
        
        Args:
            rules_data: List of traffic rule data dictionaries
        """
        # Clear existing traffic rules
        self.traffic_rules.clear()
        
        # Add traffic rules from data
        for rule_data in rules_data:
            self.add_traffic_rule(
                rule_data.get('type', 'speed_limit'),
                rule_data.get('parameters', {}),
                rule_data.get('id')
            )
    
    def visualize(self, data: Dict[str, Any], module_data: Dict[str, Any]) -> None:
        """
        Visualize guidance constraints.
        
        Args:
            data: Real-time data
            module_data: Module-specific data
        """
        print("Guidance Constraints:")
        print(f"  Lane width: {self.lane_width}")
        print(f"  Safety margin: {self.safety_margin}")
        print(f"  Max speed: {self.max_speed}")
        print(f"  Lane count: {len(self.lanes)}")
        print(f"  Intersection count: {len(self.intersections)}")
        print(f"  Traffic rule count: {len(self.traffic_rules)}")
        print(f"  Enabled: {self.enabled}")
        
        for lane in self.lanes:
            print(f"  Lane {lane['id']}: {lane['start_point']} -> {lane['end_point']} "
                  f"(type: {lane['type']})")
        
        for intersection in self.intersections:
            print(f"  Intersection {intersection['id']}: center={intersection['center']}, "
                  f"radius={intersection['radius']}")
        
        for rule in self.traffic_rules:
            print(f"  Traffic rule {rule['id']}: type={rule['type']}, "
                  f"parameters={rule['parameters']}")
    
    def get_constraint_info(self) -> Dict[str, Any]:
        """
        Get constraint information.
        
        Returns:
            Constraint information dictionary
        """
        return {
            'constraint_name': self.constraint_name,
            'lane_width': self.lane_width,
            'safety_margin': self.safety_margin,
            'max_speed': self.max_speed,
            'lane_count': len(self.lanes),
            'intersection_count': len(self.intersections),
            'traffic_rule_count': len(self.traffic_rules),
            'enabled': self.enabled
        }
    
    def set_lane_width(self, width: float) -> None:
        """
        Set lane width.
        
        Args:
            width: Lane width
        """
        self.lane_width = width
        self.parameters['lane_width'] = width
    
    def set_safety_margin(self, margin: float) -> None:
        """
        Set safety margin.
        
        Args:
            margin: Safety margin
        """
        self.safety_margin = margin
        self.parameters['safety_margin'] = margin
    
    def set_max_speed(self, speed: float) -> None:
        """
        Set maximum speed.
        
        Args:
            speed: Maximum speed
        """
        self.max_speed = speed
        self.parameters['max_speed'] = speed
    
    def get_lane_info(self) -> List[Dict[str, Any]]:
        """
        Get lane information.
        
        Returns:
            List of lane information dictionaries
        """
        return [
            {
                'id': lane['id'],
                'start_point': lane['start_point'].tolist(),
                'end_point': lane['end_point'].tolist(),
                'width': lane['width'],
                'type': lane['type'],
                'enabled': lane['enabled']
            }
            for lane in self.lanes
        ]
    
    def get_intersection_info(self) -> List[Dict[str, Any]]:
        """
        Get intersection information.
        
        Returns:
            List of intersection information dictionaries
        """
        return [
            {
                'id': intersection['id'],
                'center': intersection['center'].tolist(),
                'radius': intersection['radius'],
                'enabled': intersection['enabled']
            }
            for intersection in self.intersections
        ]
    
    def get_traffic_rule_info(self) -> List[Dict[str, Any]]:
        """
        Get traffic rule information.
        
        Returns:
            List of traffic rule information dictionaries
        """
        return [
            {
                'id': rule['id'],
                'type': rule['type'],
                'parameters': rule['parameters'],
                'enabled': rule['enabled']
            }
            for rule in self.traffic_rules
        ]
