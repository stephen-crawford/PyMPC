"""
Base constraint module for MPC planning.

This module provides the base class for all constraint implementations.
"""

import numpy as np
import casadi as cs
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod


class BaseConstraint(ABC):
    """Abstract base class for MPC constraints."""
    
    def __init__(self, constraint_name: str, enabled: bool = True):
        """
        Initialize constraint.
        
        Args:
            constraint_name: Name of the constraint
            enabled: Whether constraint is enabled
        """
        self.constraint_name = constraint_name
        self.enabled = enabled
        self.parameters = {}
        self.bounds = {}
        
        # Constraint data
        self.constraint_data = {}
        self.violation_count = 0
        self.total_violations = 0
    
    @abstractmethod
    def add_constraints(self, x: cs.SX, u: cs.SX, k: int, **kwargs) -> List[cs.SX]:
        """
        Add constraints for time step k.
        
        Args:
            x: State variables
            u: Input variables
            k: Time step
            **kwargs: Additional parameters
            
        Returns:
            List of constraint expressions
        """
        pass
    
    def set_parameters(self, data: Dict[str, Any], module_data: Dict[str, Any], 
                       k: int) -> Dict[str, float]:
        """
        Set parameters for time step k.
        
        Args:
            data: Real-time data
            module_data: Module-specific data
            k: Time step
            
        Returns:
            Parameter dictionary
        """
        return self.parameters
    
    def update(self, state: np.ndarray, data: Dict[str, Any], 
               module_data: Dict[str, Any]) -> None:
        """
        Update constraint with current state and data.
        
        Args:
            state: Current state
            data: Real-time data
            module_data: Module-specific data
        """
        pass
    
    def visualize(self, data: Dict[str, Any], module_data: Dict[str, Any]) -> None:
        """
        Visualize constraint.
        
        Args:
            data: Real-time data
            module_data: Module-specific data
        """
        print(f"Constraint '{self.constraint_name}': enabled={self.enabled}")
    
    def reset(self) -> None:
        """Reset constraint state."""
        self.constraint_data = {}
        self.violation_count = 0
        self.total_violations = 0
    
    def set_parameter(self, name: str, value: Any) -> None:
        """
        Set constraint parameter.
        
        Args:
            name: Parameter name
            value: Parameter value
        """
        self.parameters[name] = value
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get constraint parameter.
        
        Args:
            name: Parameter name
            default: Default value
            
        Returns:
            Parameter value
        """
        return self.parameters.get(name, default)
    
    def set_bounds(self, name: str, bounds: Tuple[float, float]) -> None:
        """
        Set constraint bounds.
        
        Args:
            name: Constraint name
            bounds: Lower and upper bounds
        """
        self.bounds[name] = bounds
    
    def get_bounds(self, name: str) -> Tuple[float, float]:
        """
        Get constraint bounds.
        
        Args:
            name: Constraint name
            
        Returns:
            Constraint bounds
        """
        return self.bounds.get(name, (0.0, 0.0))
    
    def check_violation(self, constraint_value: float, 
                       tolerance: float = 1e-6) -> bool:
        """
        Check if constraint is violated.
        
        Args:
            constraint_value: Constraint value
            tolerance: Tolerance for violation
            
        Returns:
            True if violated, False otherwise
        """
        return abs(constraint_value) > tolerance
    
    def record_violation(self) -> None:
        """Record constraint violation."""
        self.violation_count += 1
        self.total_violations += 1
    
    def get_violation_statistics(self) -> Dict[str, Any]:
        """
        Get violation statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'constraint_name': self.constraint_name,
            'violation_count': self.violation_count,
            'total_violations': self.total_violations,
            'enabled': self.enabled
        }
    
    def enable(self) -> None:
        """Enable constraint."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable constraint."""
        self.enabled = False
    
    def is_enabled(self) -> bool:
        """
        Check if constraint is enabled.
        
        Returns:
            True if enabled, False otherwise
        """
        return self.enabled


class LinearConstraint(BaseConstraint):
    """Linear constraint implementation."""
    
    def __init__(self, constraint_name: str, A: np.ndarray, b: np.ndarray,
                 enabled: bool = True):
        """
        Initialize linear constraint.
        
        Args:
            constraint_name: Name of the constraint
            A: Constraint matrix
            b: Constraint vector
            enabled: Whether constraint is enabled
        """
        super().__init__(constraint_name, enabled)
        self.A = A
        self.b = b
    
    def add_constraints(self, x: cs.SX, u: cs.SX, k: int, **kwargs) -> List[cs.SX]:
        """
        Add linear constraints.
        
        Args:
            x: State variables
            u: Input variables
            k: Time step
            **kwargs: Additional parameters
            
        Returns:
            List of constraint expressions
        """
        if not self.enabled:
            return []
        
        # Combine state and input variables
        z = cs.vertcat(x, u)
        
        # Linear constraint: A*z <= b
        constraints = []
        for i in range(self.A.shape[0]):
            constraint = cs.dot(self.A[i, :], z) - self.b[i]
            constraints.append(constraint)
        
        return constraints


class QuadraticConstraint(BaseConstraint):
    """Quadratic constraint implementation."""
    
    def __init__(self, constraint_name: str, Q: np.ndarray, q: np.ndarray, r: float,
                 enabled: bool = True):
        """
        Initialize quadratic constraint.
        
        Args:
            constraint_name: Name of the constraint
            Q: Quadratic matrix
            q: Linear vector
            r: Constant term
            enabled: Whether constraint is enabled
        """
        super().__init__(constraint_name, enabled)
        self.Q = Q
        self.q = q
        self.r = r
    
    def add_constraints(self, x: cs.SX, u: cs.SX, k: int, **kwargs) -> List[cs.SX]:
        """
        Add quadratic constraints.
        
        Args:
            x: State variables
            u: Input variables
            k: Time step
            **kwargs: Additional parameters
            
        Returns:
            List of constraint expressions
        """
        if not self.enabled:
            return []
        
        # Combine state and input variables
        z = cs.vertcat(x, u)
        
        # Quadratic constraint: z^T*Q*z + q^T*z + r <= 0
        constraint = cs.quad_form(z, self.Q) + cs.dot(self.q, z) + self.r
        return [constraint]


class BoundConstraint(BaseConstraint):
    """Bound constraint implementation."""
    
    def __init__(self, constraint_name: str, variable_names: List[str],
                 lower_bounds: List[float], upper_bounds: List[float],
                 enabled: bool = True):
        """
        Initialize bound constraint.
        
        Args:
            constraint_name: Name of the constraint
            variable_names: Names of variables to bound
            lower_bounds: Lower bounds
            upper_bounds: Upper bounds
            enabled: Whether constraint is enabled
        """
        super().__init__(constraint_name, enabled)
        self.variable_names = variable_names
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
    
    def add_constraints(self, x: cs.SX, u: cs.SX, k: int, **kwargs) -> List[cs.SX]:
        """
        Add bound constraints.
        
        Args:
            x: State variables
            u: Input variables
            k: Time step
            **kwargs: Additional parameters
            
        Returns:
            List of constraint expressions
        """
        if not self.enabled:
            return []
        
        constraints = []
        
        # Add state bounds
        for i, (lb, ub) in enumerate(zip(self.lower_bounds, self.upper_bounds)):
            if i < x.shape[0]:
                constraints.append(x[i] - lb)  # x[i] >= lb
                constraints.append(ub - x[i])  # x[i] <= ub
        
        # Add input bounds
        for i, (lb, ub) in enumerate(zip(self.lower_bounds, self.upper_bounds)):
            if i < u.shape[0]:
                constraints.append(u[i] - lb)  # u[i] >= lb
                constraints.append(ub - u[i])  # u[i] <= ub
        
        return constraints


class ConstraintManager:
    """Manager for multiple constraints."""
    
    def __init__(self):
        """Initialize constraint manager."""
        self.constraints: List[BaseConstraint] = []
        self.constraint_data = {}
    
    def add_constraint(self, constraint: BaseConstraint) -> None:
        """
        Add constraint to manager.
        
        Args:
            constraint: Constraint to add
        """
        self.constraints.append(constraint)
    
    def remove_constraint(self, constraint: BaseConstraint) -> None:
        """
        Remove constraint from manager.
        
        Args:
            constraint: Constraint to remove
        """
        if constraint in self.constraints:
            self.constraints.remove(constraint)
    
    def get_constraints_by_name(self, name: str) -> List[BaseConstraint]:
        """
        Get constraints by name.
        
        Args:
            name: Constraint name
            
        Returns:
            List of constraints with specified name
        """
        return [c for c in self.constraints if c.constraint_name == name]
    
    def get_enabled_constraints(self) -> List[BaseConstraint]:
        """
        Get enabled constraints.
        
        Returns:
            List of enabled constraints
        """
        return [c for c in self.constraints if c.enabled]
    
    def add_all_constraints(self, x: cs.SX, u: cs.SX, k: int, **kwargs) -> List[cs.SX]:
        """
        Add all constraints for time step k.
        
        Args:
            x: State variables
            u: Input variables
            k: Time step
            **kwargs: Additional parameters
            
        Returns:
            List of all constraint expressions
        """
        all_constraints = []
        
        for constraint in self.constraints:
            if constraint.enabled:
                constraints = constraint.add_constraints(x, u, k, **kwargs)
                all_constraints.extend(constraints)
        
        return all_constraints
    
    def update_constraints(self, state: np.ndarray, data: Dict[str, Any]) -> None:
        """
        Update all constraints.
        
        Args:
            state: Current state
            data: Real-time data
        """
        for constraint in self.constraints:
            constraint.update(state, data, self.constraint_data)
    
    def visualize_constraints(self, data: Dict[str, Any]) -> None:
        """
        Visualize all constraints.
        
        Args:
            data: Real-time data
        """
        for constraint in self.constraints:
            constraint.visualize(data, self.constraint_data)
    
    def reset_constraints(self) -> None:
        """Reset all constraints."""
        for constraint in self.constraints:
            constraint.reset()
        self.constraint_data = {}
    
    def get_constraint_statistics(self) -> Dict[str, Any]:
        """
        Get constraint statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_constraints': len(self.constraints),
            'enabled_constraints': len(self.get_enabled_constraints()),
            'constraint_names': [c.constraint_name for c in self.constraints],
            'violation_statistics': [c.get_violation_statistics() for c in self.constraints]
        }
