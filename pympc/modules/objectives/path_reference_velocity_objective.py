"""
Path reference velocity objective for MPC planning.

This module implements the path reference velocity objective that
encourages the vehicle to follow a desired velocity profile along a path.
"""

import numpy as np
import casadi as cs
from typing import Dict, Optional, Any
from .base_objective import BaseObjective


class PathReferenceVelocityObjective(BaseObjective):
    """
    Path reference velocity objective for velocity tracking.
    
    This objective encourages the vehicle to follow a desired
    velocity profile along a reference path.
    """
    
    def __init__(self, velocity_weight: float = 1.0, acceleration_weight: float = 0.1,
                 jerk_weight: float = 0.01, enabled: bool = True):
        """
        Initialize path reference velocity objective.
        
        Args:
            velocity_weight: Weight for velocity tracking
            acceleration_weight: Weight for acceleration tracking
            jerk_weight: Weight for jerk minimization
            enabled: Whether objective is enabled
        """
        super().__init__("path_reference_velocity_objective", enabled)
        
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.jerk_weight = jerk_weight
        
        # Reference data
        self.reference_velocities = None
        self.reference_accelerations = None
        self.path_parameters = None
        self.velocity_profile = None
        
        # Objective parameters
        self.parameters = {
            'velocity_weight': velocity_weight,
            'acceleration_weight': acceleration_weight,
            'jerk_weight': jerk_weight
        }
    
    def set_reference_velocity_profile(self, velocities: np.ndarray, 
                                     path_parameters: Optional[np.ndarray] = None,
                                     accelerations: Optional[np.ndarray] = None) -> None:
        """
        Set reference velocity profile.
        
        Args:
            velocities: Reference velocities along path
            path_parameters: Path parameters (optional)
            accelerations: Reference accelerations (optional)
        """
        self.reference_velocities = velocities.copy()
        
        if path_parameters is not None:
            self.path_parameters = path_parameters.copy()
        else:
            # Default path parameters (uniform spacing)
            self.path_parameters = np.linspace(0, len(velocities)-1, len(velocities))
        
        if accelerations is not None:
            self.reference_accelerations = accelerations.copy()
        else:
            # Compute accelerations from velocity differences
            self.reference_accelerations = np.gradient(velocities)
    
    def set_velocity_profile(self, profile: Dict[str, Any]) -> None:
        """
        Set velocity profile from dictionary.
        
        Args:
            profile: Velocity profile dictionary
        """
        self.velocity_profile = profile.copy()
        
        # Extract velocities and parameters
        if 'velocities' in profile:
            self.reference_velocities = np.array(profile['velocities'])
        
        if 'parameters' in profile:
            self.path_parameters = np.array(profile['parameters'])
        
        if 'accelerations' in profile:
            self.reference_accelerations = np.array(profile['accelerations'])
    
    def _get_reference_velocity_at_s(self, s: float) -> float:
        """
        Get reference velocity at path parameter s.
        
        Args:
            s: Path parameter
            
        Returns:
            Reference velocity
        """
        if self.reference_velocities is None:
            return 5.0  # Default velocity
        
        # Clamp s to valid range
        s = max(0.0, min(s, len(self.reference_velocities) - 1))
        
        # Interpolate velocity
        if s == int(s):
            return self.reference_velocities[int(s)]
        else:
            # Linear interpolation
            s_low = int(s)
            s_high = min(s_low + 1, len(self.reference_velocities) - 1)
            alpha = s - s_low
            
            v_low = self.reference_velocities[s_low]
            v_high = self.reference_velocities[s_high]
            
            return (1 - alpha) * v_low + alpha * v_high
    
    def _get_reference_acceleration_at_s(self, s: float) -> float:
        """
        Get reference acceleration at path parameter s.
        
        Args:
            s: Path parameter
            
        Returns:
            Reference acceleration
        """
        if self.reference_accelerations is None:
            return 0.0  # Default acceleration
        
        # Clamp s to valid range
        s = max(0.0, min(s, len(self.reference_accelerations) - 1))
        
        # Interpolate acceleration
        if s == int(s):
            return self.reference_accelerations[int(s)]
        else:
            # Linear interpolation
            s_low = int(s)
            s_high = min(s_low + 1, len(self.reference_accelerations) - 1)
            alpha = s - s_low
            
            a_low = self.reference_accelerations[s_low]
            a_high = self.reference_accelerations[s_high]
            
            return (1 - alpha) * a_low + alpha * a_high
    
    def add_objective(self, x: cs.SX, u: cs.SX, k: int, **kwargs) -> cs.SX:
        """
        Add path reference velocity objective for time step k.
        
        Args:
            x: State variables [x, y, psi, v, s, ...]
            u: Input variables
            k: Time step
            **kwargs: Additional parameters
            
        Returns:
            Objective function expression
        """
        if not self.enabled or self.reference_velocities is None:
            return 0.0
        
        # Get vehicle state
        vehicle_v = x[3]  # velocity
        
        # Get path parameter (if available in state)
        if x.shape[0] > 4:  # Assuming spline parameter is at index 4
            s = x[4]  # Path parameter
        else:
            # Estimate path parameter from position
            s = self._estimate_path_parameter(x[0], x[1])
        
        # Get reference velocity and acceleration
        v_ref = self._get_reference_velocity_at_s(s)
        a_ref = self._get_reference_acceleration_at_s(s)
        
        # Velocity tracking objective
        velocity_error = vehicle_v - v_ref
        velocity_objective = self.velocity_weight * velocity_error**2
        
        # Acceleration tracking objective (if control input available)
        acceleration_objective = 0.0
        if u.shape[0] > 0:  # Assuming first control input is acceleration
            vehicle_a = u[0]  # acceleration
            acceleration_error = vehicle_a - a_ref
            acceleration_objective = self.acceleration_weight * acceleration_error**2
        
        # Jerk minimization objective (if second control input available)
        jerk_objective = 0.0
        if u.shape[0] > 1:  # Assuming second control input is jerk
            vehicle_jerk = u[1]  # jerk
            jerk_objective = self.jerk_weight * vehicle_jerk**2
        
        # Combine objectives
        objective = velocity_objective + acceleration_objective + jerk_objective
        
        return objective
    
    def _estimate_path_parameter(self, x: float, y: float) -> float:
        """
        Estimate path parameter from vehicle position.
        
        Args:
            x: Vehicle x position
            y: Vehicle y position
            
        Returns:
            Estimated path parameter
        """
        if self.path_parameters is None:
            return 0.0
        
        # Simple estimation - in practice, you'd use more sophisticated methods
        # For now, just return 0.0 as a placeholder
        return 0.0
    
    def update(self, state: np.ndarray, data: Dict[str, Any], 
               module_data: Dict[str, Any]) -> None:
        """
        Update path reference velocity objective.
        
        Args:
            state: Current state
            data: Real-time data
            module_data: Module-specific data
        """
        # Update objective data
        self.objective_data.update({
            'velocity_weight': self.velocity_weight,
            'acceleration_weight': self.acceleration_weight,
            'jerk_weight': self.jerk_weight,
            'reference_velocities': self.reference_velocities.tolist() if self.reference_velocities is not None else None,
            'path_parameters': self.path_parameters.tolist() if self.path_parameters is not None else None
        })
    
    def visualize(self, data: Dict[str, Any], module_data: Dict[str, Any]) -> None:
        """
        Visualize path reference velocity objective.
        
        Args:
            data: Real-time data
            module_data: Module-specific data
        """
        print("Path Reference Velocity Objective:")
        print(f"  Velocity weight: {self.velocity_weight}")
        print(f"  Acceleration weight: {self.acceleration_weight}")
        print(f"  Jerk weight: {self.jerk_weight}")
        print(f"  Reference velocities: {len(self.reference_velocities) if self.reference_velocities is not None else 0}")
        print(f"  Path parameters: {len(self.path_parameters) if self.path_parameters is not None else 0}")
        print(f"  Enabled: {self.enabled}")
    
    def get_objective_info(self) -> Dict[str, Any]:
        """
        Get objective information.
        
        Returns:
            Objective information dictionary
        """
        return {
            'objective_name': self.objective_name,
            'velocity_weight': self.velocity_weight,
            'acceleration_weight': self.acceleration_weight,
            'jerk_weight': self.jerk_weight,
            'reference_velocities_set': self.reference_velocities is not None,
            'path_parameters_set': self.path_parameters is not None,
            'enabled': self.enabled
        }
    
    def set_velocity_weight(self, weight: float) -> None:
        """
        Set velocity weight.
        
        Args:
            weight: Velocity weight
        """
        self.velocity_weight = weight
        self.parameters['velocity_weight'] = weight
    
    def set_acceleration_weight(self, weight: float) -> None:
        """
        Set acceleration weight.
        
        Args:
            weight: Acceleration weight
        """
        self.acceleration_weight = weight
        self.parameters['acceleration_weight'] = weight
    
    def set_jerk_weight(self, weight: float) -> None:
        """
        Set jerk weight.
        
        Args:
            weight: Jerk weight
        """
        self.jerk_weight = weight
        self.parameters['jerk_weight'] = weight
    
    def set_weights(self, velocity: float = None, acceleration: float = None,
                   jerk: float = None) -> None:
        """
        Set multiple weights at once.
        
        Args:
            velocity: Velocity weight
            acceleration: Acceleration weight
            jerk: Jerk weight
        """
        if velocity is not None:
            self.set_velocity_weight(velocity)
        if acceleration is not None:
            self.set_acceleration_weight(acceleration)
        if jerk is not None:
            self.set_jerk_weight(jerk)
    
    def get_velocity_profile(self) -> Dict[str, Any]:
        """
        Get velocity profile information.
        
        Returns:
            Velocity profile dictionary
        """
        return {
            'velocities': self.reference_velocities.tolist() if self.reference_velocities is not None else None,
            'parameters': self.path_parameters.tolist() if self.path_parameters is not None else None,
            'accelerations': self.reference_accelerations.tolist() if self.reference_accelerations is not None else None,
            'profile': self.velocity_profile
        }
    
    def create_constant_velocity_profile(self, velocity: float, num_points: int = 100) -> None:
        """
        Create constant velocity profile.
        
        Args:
            velocity: Constant velocity value
            num_points: Number of profile points
        """
        velocities = np.full(num_points, velocity)
        parameters = np.linspace(0, num_points-1, num_points)
        accelerations = np.zeros(num_points)
        
        self.set_reference_velocity_profile(velocities, parameters, accelerations)
    
    def create_ramp_velocity_profile(self, start_velocity: float, end_velocity: float, 
                                   num_points: int = 100) -> None:
        """
        Create ramp velocity profile.
        
        Args:
            start_velocity: Starting velocity
            end_velocity: Ending velocity
            num_points: Number of profile points
        """
        velocities = np.linspace(start_velocity, end_velocity, num_points)
        parameters = np.linspace(0, num_points-1, num_points)
        accelerations = np.full(num_points, (end_velocity - start_velocity) / (num_points - 1))
        
        self.set_reference_velocity_profile(velocities, parameters, accelerations)
    
    def create_sinusoidal_velocity_profile(self, base_velocity: float, amplitude: float,
                                         frequency: float, num_points: int = 100) -> None:
        """
        Create sinusoidal velocity profile.
        
        Args:
            base_velocity: Base velocity
            amplitude: Velocity amplitude
            frequency: Frequency of oscillation
            num_points: Number of profile points
        """
        parameters = np.linspace(0, 2*np.pi, num_points)
        velocities = base_velocity + amplitude * np.sin(frequency * parameters)
        accelerations = amplitude * frequency * np.cos(frequency * parameters)
        
        self.set_reference_velocity_profile(velocities, parameters, accelerations)
