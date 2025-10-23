"""
Parameters manager for MPC planning.

This module manages parameter passing between modules, solver, and planner.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod


class ParameterManager:
    """Manager for MPC parameters."""
    
    def __init__(self):
        """Initialize parameter manager."""
        self.parameters: Dict[str, Any] = {}
        self.parameter_history: List[Dict[str, Any]] = []
        self.parameter_bounds: Dict[str, Tuple[float, float]] = {}
        self.parameter_types: Dict[str, str] = {}
        
        # Parameter categories
        self.dynamics_params: Dict[str, Any] = {}
        self.solver_params: Dict[str, Any] = {}
        self.objective_params: Dict[str, Any] = {}
        self.constraint_params: Dict[str, Any] = {}
        self.visualization_params: Dict[str, Any] = {}
    
    def set_parameter(self, name: str, value: Any, 
                      bounds: Optional[Tuple[float, float]] = None,
                      param_type: str = "float") -> None:
        """
        Set a parameter value.
        
        Args:
            name: Parameter name
            value: Parameter value
            bounds: Parameter bounds (for numeric parameters)
            param_type: Parameter type
        """
        self.parameters[name] = value
        
        if bounds is not None:
            self.parameter_bounds[name] = bounds
        
        self.parameter_types[name] = param_type
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get a parameter value.
        
        Args:
            name: Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value
        """
        return self.parameters.get(name, default)
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set multiple parameters.
        
        Args:
            params: Parameter dictionary
        """
        self.parameters.update(params)
    
    def get_parameters(self, names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get multiple parameters.
        
        Args:
            names: List of parameter names (if None, return all)
            
        Returns:
            Parameter dictionary
        """
        if names is None:
            return self.parameters.copy()
        else:
            return {name: self.parameters.get(name) for name in names}
    
    def set_dynamics_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set dynamics parameters.
        
        Args:
            params: Dynamics parameters
        """
        self.dynamics_params.update(params)
        self.parameters.update(params)
    
    def get_dynamics_parameters(self) -> Dict[str, Any]:
        """
        Get dynamics parameters.
        
        Returns:
            Dynamics parameters
        """
        return self.dynamics_params.copy()
    
    def set_solver_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set solver parameters.
        
        Args:
            params: Solver parameters
        """
        self.solver_params.update(params)
        self.parameters.update(params)
    
    def get_solver_parameters(self) -> Dict[str, Any]:
        """
        Get solver parameters.
        
        Returns:
            Solver parameters
        """
        return self.solver_params.copy()
    
    def set_objective_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set objective parameters.
        
        Args:
            params: Objective parameters
        """
        self.objective_params.update(params)
        self.parameters.update(params)
    
    def get_objective_parameters(self) -> Dict[str, Any]:
        """
        Get objective parameters.
        
        Returns:
            Objective parameters
        """
        return self.objective_params.copy()
    
    def set_constraint_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set constraint parameters.
        
        Args:
            params: Constraint parameters
        """
        self.constraint_params.update(params)
        self.parameters.update(params)
    
    def get_constraint_parameters(self) -> Dict[str, Any]:
        """
        Get constraint parameters.
        
        Returns:
            Constraint parameters
        """
        return self.constraint_params.copy()
    
    def set_visualization_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set visualization parameters.
        
        Args:
            params: Visualization parameters
        """
        self.visualization_params.update(params)
        self.parameters.update(params)
    
    def get_visualization_parameters(self) -> Dict[str, Any]:
        """
        Get visualization parameters.
        
        Returns:
            Visualization parameters
        """
        return self.visualization_params.copy()
    
    def validate_parameter(self, name: str, value: Any) -> bool:
        """
        Validate a parameter value.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Returns:
            True if valid, False otherwise
        """
        if name not in self.parameters:
            return True  # New parameter, assume valid
        
        # Check bounds if they exist
        if name in self.parameter_bounds:
            lb, ub = self.parameter_bounds[name]
            if not (lb <= value <= ub):
                return False
        
        # Check type if specified
        if name in self.parameter_types:
            param_type = self.parameter_types[name]
            if param_type == "float" and not isinstance(value, (int, float)):
                return False
            elif param_type == "int" and not isinstance(value, int):
                return False
            elif param_type == "bool" and not isinstance(value, bool):
                return False
            elif param_type == "str" and not isinstance(value, str):
                return False
        
        return True
    
    def set_parameter_with_validation(self, name: str, value: Any,
                                    bounds: Optional[Tuple[float, float]] = None,
                                    param_type: str = "float") -> bool:
        """
        Set parameter with validation.
        
        Args:
            name: Parameter name
            value: Parameter value
            bounds: Parameter bounds
            param_type: Parameter type
            
        Returns:
            True if set successfully, False otherwise
        """
        if bounds is not None:
            self.parameter_bounds[name] = bounds
        
        self.parameter_types[name] = param_type
        
        if self.validate_parameter(name, value):
            self.parameters[name] = value
            return True
        else:
            return False
    
    def save_parameter_snapshot(self) -> None:
        """Save current parameter state."""
        self.parameter_history.append(self.parameters.copy())
    
    def restore_parameter_snapshot(self, index: int = -1) -> bool:
        """
        Restore parameter state from history.
        
        Args:
            index: History index (-1 for latest)
            
        Returns:
            True if restored successfully, False otherwise
        """
        if not self.parameter_history or abs(index) > len(self.parameter_history):
            return False
        
        self.parameters = self.parameter_history[index].copy()
        return True
    
    def get_parameter_history(self) -> List[Dict[str, Any]]:
        """
        Get parameter history.
        
        Returns:
            Parameter history
        """
        return self.parameter_history.copy()
    
    def clear_parameter_history(self) -> None:
        """Clear parameter history."""
        self.parameter_history.clear()
    
    def get_parameter_statistics(self) -> Dict[str, Any]:
        """
        Get parameter statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_parameters': len(self.parameters),
            'dynamics_parameters': len(self.dynamics_params),
            'solver_parameters': len(self.solver_params),
            'objective_parameters': len(self.objective_params),
            'constraint_parameters': len(self.constraint_params),
            'visualization_parameters': len(self.visualization_params),
            'parameter_types': dict(self.parameter_types),
            'parameter_bounds': dict(self.parameter_bounds),
            'history_length': len(self.parameter_history)
        }
    
    def export_parameters(self, filename: str) -> bool:
        """
        Export parameters to file.
        
        Args:
            filename: Output filename
            
        Returns:
            True if exported successfully, False otherwise
        """
        try:
            import json
            with open(filename, 'w') as f:
                json.dump({
                    'parameters': self.parameters,
                    'bounds': self.parameter_bounds,
                    'types': self.parameter_types,
                    'categories': {
                        'dynamics': self.dynamics_params,
                        'solver': self.solver_params,
                        'objective': self.objective_params,
                        'constraint': self.constraint_params,
                        'visualization': self.visualization_params
                    }
                }, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting parameters: {e}")
            return False
    
    def import_parameters(self, filename: str) -> bool:
        """
        Import parameters from file.
        
        Args:
            filename: Input filename
            
        Returns:
            True if imported successfully, False otherwise
        """
        try:
            import json
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.parameters = data.get('parameters', {})
            self.parameter_bounds = data.get('bounds', {})
            self.parameter_types = data.get('types', {})
            
            categories = data.get('categories', {})
            self.dynamics_params = categories.get('dynamics', {})
            self.solver_params = categories.get('solver', {})
            self.objective_params = categories.get('objective', {})
            self.constraint_params = categories.get('constraint', {})
            self.visualization_params = categories.get('visualization', {})
            
            return True
        except Exception as e:
            print(f"Error importing parameters: {e}")
            return False
    
    def reset(self) -> None:
        """Reset parameter manager."""
        self.parameters.clear()
        self.parameter_history.clear()
        self.parameter_bounds.clear()
        self.parameter_types.clear()
        self.dynamics_params.clear()
        self.solver_params.clear()
        self.objective_params.clear()
        self.constraint_params.clear()
        self.visualization_params.clear()


class ParameterValidator:
    """Validator for MPC parameters."""
    
    def __init__(self):
        """Initialize parameter validator."""
        self.validation_rules: Dict[str, callable] = {}
        self.required_parameters: List[str] = []
    
    def add_validation_rule(self, parameter_name: str, rule: callable) -> None:
        """
        Add validation rule for parameter.
        
        Args:
            parameter_name: Parameter name
            rule: Validation function
        """
        self.validation_rules[parameter_name] = rule
    
    def add_required_parameter(self, parameter_name: str) -> None:
        """
        Add required parameter.
        
        Args:
            parameter_name: Parameter name
        """
        if parameter_name not in self.required_parameters:
            self.required_parameters.append(parameter_name)
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate parameters.
        
        Args:
            parameters: Parameter dictionary
            
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check required parameters
        for param in self.required_parameters:
            if param not in parameters:
                errors.append(f"Required parameter '{param}' not found")
        
        # Check validation rules
        for param, rule in self.validation_rules.items():
            if param in parameters:
                try:
                    if not rule(parameters[param]):
                        errors.append(f"Parameter '{param}' failed validation")
                except Exception as e:
                    errors.append(f"Parameter '{param}' validation error: {e}")
        
        return len(errors) == 0, errors


def create_parameter_manager() -> ParameterManager:
    """
    Create a parameter manager.
    
    Returns:
        Parameter manager instance
    """
    return ParameterManager()


def create_parameter_validator() -> ParameterValidator:
    """
    Create a parameter validator.
    
    Returns:
        Parameter validator instance
    """
    return ParameterValidator()
