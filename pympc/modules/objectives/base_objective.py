"""
Base objective module for MPC planning.

This module provides the base class for all objective implementations.
"""

import numpy as np
import casadi as cs
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod


class BaseObjective(ABC):
    """Abstract base class for MPC objectives."""
    
    def __init__(self, objective_name: str, enabled: bool = True):
        """
        Initialize objective.
        
        Args:
            objective_name: Name of the objective
            enabled: Whether objective is enabled
        """
        self.objective_name = objective_name
        self.enabled = enabled
        self.parameters = {}
        self.weights = {}
        
        # Objective data
        self.objective_data = {}
        self.evaluation_count = 0
        self.total_evaluation = 0.0
    
    @abstractmethod
    def add_objective(self, x: cs.SX, u: cs.SX, k: int, **kwargs) -> cs.SX:
        """
        Add objective function for time step k.
        
        Args:
            x: State variables
            u: Input variables
            k: Time step
            **kwargs: Additional parameters
            
        Returns:
            Objective function expression
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
        Update objective with current state and data.
        
        Args:
            state: Current state
            data: Real-time data
            module_data: Module-specific data
        """
        pass
    
    def visualize(self, data: Dict[str, Any], module_data: Dict[str, Any]) -> None:
        """
        Visualize objective.
        
        Args:
            data: Real-time data
            module_data: Module-specific data
        """
        print(f"Objective '{self.objective_name}': enabled={self.enabled}")
    
    def reset(self) -> None:
        """Reset objective state."""
        self.objective_data = {}
        self.evaluation_count = 0
        self.total_evaluation = 0.0
    
    def set_parameter(self, name: str, value: Any) -> None:
        """
        Set objective parameter.
        
        Args:
            name: Parameter name
            value: Parameter value
        """
        self.parameters[name] = value
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get objective parameter.
        
        Args:
            name: Parameter name
            default: Default value
            
        Returns:
            Parameter value
        """
        return self.parameters.get(name, default)
    
    def set_weight(self, name: str, weight: float) -> None:
        """
        Set objective weight.
        
        Args:
            name: Weight name
            weight: Weight value
        """
        self.weights[name] = weight
    
    def get_weight(self, name: str, default: float = 1.0) -> float:
        """
        Get objective weight.
        
        Args:
            name: Weight name
            default: Default weight
            
        Returns:
            Weight value
        """
        return self.weights.get(name, default)
    
    def evaluate_objective(self, objective_value: float) -> None:
        """
        Evaluate objective function.
        
        Args:
            objective_value: Objective function value
        """
        self.evaluation_count += 1
        self.total_evaluation += objective_value
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """
        Get evaluation statistics.
        
        Returns:
            Statistics dictionary
        """
        avg_evaluation = self.total_evaluation / max(self.evaluation_count, 1)
        return {
            'objective_name': self.objective_name,
            'evaluation_count': self.evaluation_count,
            'total_evaluation': self.total_evaluation,
            'average_evaluation': avg_evaluation,
            'enabled': self.enabled
        }
    
    def enable(self) -> None:
        """Enable objective."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable objective."""
        self.enabled = False
    
    def is_enabled(self) -> bool:
        """
        Check if objective is enabled.
        
        Returns:
            True if enabled, False otherwise
        """
        return self.enabled


class QuadraticObjective(BaseObjective):
    """Quadratic objective implementation."""
    
    def __init__(self, objective_name: str, Q: np.ndarray, R: np.ndarray,
                 q: np.ndarray = None, r: float = 0.0, enabled: bool = True):
        """
        Initialize quadratic objective.
        
        Args:
            objective_name: Name of the objective
            Q: State weight matrix
            R: Input weight matrix
            q: Linear state weight vector
            r: Constant term
            enabled: Whether objective is enabled
        """
        super().__init__(objective_name, enabled)
        self.Q = Q
        self.R = R
        self.q = q if q is not None else np.zeros(Q.shape[0])
        self.r = r
    
    def add_objective(self, x: cs.SX, u: cs.SX, k: int, **kwargs) -> cs.SX:
        """
        Add quadratic objective.
        
        Args:
            x: State variables
            u: Input variables
            k: Time step
            **kwargs: Additional parameters
            
        Returns:
            Objective function expression
        """
        if not self.enabled:
            return 0.0
        
        # Quadratic objective: x^T*Q*x + u^T*R*u + q^T*x + r
        state_cost = cs.quad_form(x, self.Q) + cs.dot(self.q, x)
        input_cost = cs.quad_form(u, self.R)
        
        return state_cost + input_cost + self.r


class LinearObjective(BaseObjective):
    """Linear objective implementation."""
    
    def __init__(self, objective_name: str, c: np.ndarray, d: np.ndarray = None,
                 constant: float = 0.0, enabled: bool = True):
        """
        Initialize linear objective.
        
        Args:
            objective_name: Name of the objective
            c: State weight vector
            d: Input weight vector
            constant: Constant term
            enabled: Whether objective is enabled
        """
        super().__init__(objective_name, enabled)
        self.c = c
        self.d = d if d is not None else np.zeros(len(c))
        self.constant = constant
    
    def add_objective(self, x: cs.SX, u: cs.SX, k: int, **kwargs) -> cs.SX:
        """
        Add linear objective.
        
        Args:
            x: State variables
            u: Input variables
            k: Time step
            **kwargs: Additional parameters
            
        Returns:
            Objective function expression
        """
        if not self.enabled:
            return 0.0
        
        # Linear objective: c^T*x + d^T*u + constant
        state_cost = cs.dot(self.c, x)
        input_cost = cs.dot(self.d, u)
        
        return state_cost + input_cost + self.constant


class ObjectiveManager:
    """Manager for multiple objectives."""
    
    def __init__(self):
        """Initialize objective manager."""
        self.objectives: List[BaseObjective] = []
        self.objective_data = {}
    
    def add_objective(self, objective: BaseObjective) -> None:
        """
        Add objective to manager.
        
        Args:
            objective: Objective to add
        """
        self.objectives.append(objective)
    
    def remove_objective(self, objective: BaseObjective) -> None:
        """
        Remove objective from manager.
        
        Args:
            objective: Objective to remove
        """
        if objective in self.objectives:
            self.objectives.remove(objective)
    
    def get_objectives_by_name(self, name: str) -> List[BaseObjective]:
        """
        Get objectives by name.
        
        Args:
            name: Objective name
            
        Returns:
            List of objectives with specified name
        """
        return [o for o in self.objectives if o.objective_name == name]
    
    def get_enabled_objectives(self) -> List[BaseObjective]:
        """
        Get enabled objectives.
        
        Returns:
            List of enabled objectives
        """
        return [o for o in self.objectives if o.enabled]
    
    def add_all_objectives(self, x: cs.SX, u: cs.SX, k: int, **kwargs) -> cs.SX:
        """
        Add all objectives for time step k.
        
        Args:
            x: State variables
            u: Input variables
            k: Time step
            **kwargs: Additional parameters
            
        Returns:
            Combined objective function
        """
        total_objective = None
        
        for objective in self.objectives:
            if objective.enabled:
                obj = objective.add_objective(x, u, k, **kwargs)
                if total_objective is None:
                    total_objective = obj
                else:
                    total_objective += obj
        
        return total_objective if total_objective is not None else 0.0
    
    def update_objectives(self, state: np.ndarray, data: Dict[str, Any]) -> None:
        """
        Update all objectives.
        
        Args:
            state: Current state
            data: Real-time data
        """
        for objective in self.objectives:
            objective.update(state, data, self.objective_data)
    
    def visualize_objectives(self, data: Dict[str, Any]) -> None:
        """
        Visualize all objectives.
        
        Args:
            data: Real-time data
        """
        for objective in self.objectives:
            objective.visualize(data, self.objective_data)
    
    def reset_objectives(self) -> None:
        """Reset all objectives."""
        for objective in self.objectives:
            objective.reset()
        self.objective_data = {}
    
    def get_objective_statistics(self) -> Dict[str, Any]:
        """
        Get objective statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_objectives': len(self.objectives),
            'enabled_objectives': len(self.get_enabled_objectives()),
            'objective_names': [o.objective_name for o in self.objectives],
            'evaluation_statistics': [o.get_evaluation_statistics() for o in self.objectives]
        }
