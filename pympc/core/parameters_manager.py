"""
Parameter Manager for MPC.

This module provides the ParameterManager class that handles parameter
management for MPC optimization, matching the original C++ implementation.
"""

import numpy as np
import casadi as cd
from typing import Dict, Any, Optional, Union, List, Tuple
from collections import OrderedDict

from pympc.utils.utils import LOG_DEBUG, LOG_INFO, LOG_WARN


class ParameterManager:
    """
    Parameter Manager handles parameter storage and retrieval for MPC optimization.
    
    This matches the original C++ ParameterManager implementation from
    tud-amr/mpc_planner with proper indexing and lookup mechanisms.
    """
    
    def __init__(self):
        """Initialize the parameter manager."""
        # Parameter storage and lookup
        self.parameter_lookup: Dict[str, Tuple[int, int]] = {}  # param -> (start_idx, length)
        self.parameter_count: int = 0
        self.parameter_values: Optional[np.ndarray] = None
        
        # Parameter metadata
        self.parameter_metadata: Dict[str, Dict[str, Any]] = {}
        self.parameter_order: List[str] = []  # Maintain insertion order
        
        # RQT support (for ROS parameter tuning)
        self.rqt_parameters: List[str] = []
        self.rqt_parameter_config_names: List[str] = []
        self.rqt_parameter_min_values: List[float] = []
        self.rqt_parameter_max_values: List[float] = []
        
        # Parameter validation
        self.parameter_validators: Dict[str, callable] = {}
        
    def add(self, parameter: str, length: int = 1, 
            add_to_rqt_reconfigure: bool = False,
            rqt_config_name: Optional[str] = None,
            rqt_min_value: float = 0.0,
            rqt_max_value: float = 100000000000.0,
            validator: Optional[callable] = None) -> None:
        """
        Add a parameter with given length and optional settings.
        
        Args:
            parameter: Parameter name
            length: Parameter length (for vector parameters)
            add_to_rqt_reconfigure: Whether to add to RQT parameter tuning
            rqt_config_name: RQT configuration name (defaults to parameter name)
            rqt_min_value: Minimum value for RQT
            rqt_max_value: Maximum value for RQT
            validator: Optional validation function
        """
        if parameter in self.parameter_lookup:
            LOG_DEBUG(f"Parameter '{parameter}' already exists, skipping...")
            return
        
        start_idx = self.parameter_count
        self.parameter_lookup[parameter] = (start_idx, length)
        self.parameter_count += length
        self.parameter_order.append(parameter)
        
        # Store metadata
        self.parameter_metadata[parameter] = {
            'length': length,
            'start_idx': start_idx,
            'rqt_enabled': add_to_rqt_reconfigure
        }
        
        # Add to RQT if requested
        if add_to_rqt_reconfigure:
            self.rqt_parameters.append(parameter)
            config_name = rqt_config_name or f'["weights"]["{parameter}"]'
            self.rqt_parameter_config_names.append(config_name)
            self.rqt_parameter_min_values.append(rqt_min_value)
            self.rqt_parameter_max_values.append(rqt_max_value)
        
        # Store validator
        if validator:
            self.parameter_validators[parameter] = validator
        
        LOG_DEBUG(f"Added parameter '{parameter}' with length {length}")
    
    def load(self, p: Union[np.ndarray, List[float]]) -> None:
        """
        Load a flat vector of parameter values.
        
        Args:
            p: Parameter vector
        """
        p = np.array(p, dtype=float).flatten()
        if p.shape[0] != self.parameter_count:
            raise ValueError(f"Expected {self.parameter_count} parameters, got {p.shape[0]}")
        
        self.parameter_values = p.copy()
        LOG_DEBUG(f"Loaded {len(p)} parameters")
    
    def set_parameter(self, key: str, value: Union[float, int, np.ndarray, List[float]]) -> None:
        """
        Set a parameter value (scalar or vector) in the flat array.
        
        Args:
            key: Parameter name
            value: Parameter value(s)
        """
        if key not in self.parameter_lookup:
            raise KeyError(f"Parameter '{key}' not found in parameter lookup")
        
        # Initialize parameter values if not done
        if self.parameter_values is None:
            self.parameter_values = np.zeros(self.parameter_count)
        
        start_idx, length = self.parameter_lookup[key]
        value = np.atleast_1d(value).astype(float)
        
        if value.shape[0] != length:
            raise ValueError(f"Expected length {length} for '{key}', got {value.shape[0]}")
        
        # Validate parameter if validator exists
        if key in self.parameter_validators:
            try:
                self.parameter_validators[key](value)
            except Exception as e:
                raise ValueError(f"Parameter validation failed for '{key}': {e}")
        
        # Apply RQT bounds if parameter is in RQT list
        if key in self.rqt_parameters:
            idx = self.rqt_parameters.index(key)
            min_val = self.rqt_parameter_min_values[idx]
            max_val = self.rqt_parameter_max_values[idx]
            value = np.clip(value, min_val, max_val)
        
        # Set the parameter values
        self.parameter_values[start_idx:start_idx + length] = value
        
        LOG_DEBUG(f"Set parameter '{key}' = {value}")
    
    def get(self, key: str) -> Union[float, np.ndarray]:
        """
        Retrieve a parameter value (scalar or vector).
        
        Args:
            key: Parameter name
            
        Returns:
            Parameter value(s)
        """
        if self.parameter_values is None:
            raise RuntimeError("Parameters not loaded. Call load() first.")
        
        if key not in self.parameter_lookup:
            raise KeyError(f"Parameter '{key}' not found in parameter lookup")
        
        start_idx, length = self.parameter_lookup[key]
        val = self.parameter_values[start_idx:start_idx + length]
        
        return val[0] if length == 1 else val
    
    def get_all(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Return all parameters as a dictionary.
        
        Returns:
            Dictionary of {parameter_name: value}
        """
        if self.parameter_values is None:
            raise RuntimeError("Parameters not loaded. Call load() first.")
        
        all_params = {}
        for key, (start_idx, length) in self.parameter_lookup.items():
            value = self.parameter_values[start_idx:start_idx + length]
            all_params[key] = value[0] if length == 1 else value.copy()
        
        return all_params
    
    def get_parameter_vector(self) -> np.ndarray:
        """
        Get the full parameter vector.
        
        Returns:
            Full parameter vector
        """
        if self.parameter_values is None:
            raise RuntimeError("Parameters not loaded. Call load() first.")
        
        return self.parameter_values.copy()
    
    def update_from_dict(self, param_dict: Dict[str, Any]) -> None:
        """
        Update parameters from a dictionary.
        
        Args:
            param_dict: Dictionary of parameter names and values
        """
        for key, value in param_dict.items():
            if key in self.parameter_lookup:
                self.set_parameter(key, value)
            else:
                LOG_WARN(f"Parameter '{key}' not found in lookup, skipping...")
    
    def get_parameter_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a parameter.
        
        Args:
            key: Parameter name
            
        Returns:
            Parameter information dictionary or None
        """
        if key not in self.parameter_lookup:
            return None
        
        start_idx, length = self.parameter_lookup[key]
        info = {
            'name': key,
            'start_idx': start_idx,
            'length': length,
            'end_idx': start_idx + length - 1,
            'rqt_enabled': key in self.rqt_parameters
        }
        
        if key in self.rqt_parameters:
            idx = self.rqt_parameters.index(key)
            info.update({
                'rqt_config_name': self.rqt_parameter_config_names[idx],
                'rqt_min_value': self.rqt_parameter_min_values[idx],
                'rqt_max_value': self.rqt_parameter_max_values[idx]
            })
        
        return info
    
    def get_rqt_parameters(self) -> List[Dict[str, Any]]:
        """
        Get RQT parameter information.
        
        Returns:
            List of RQT parameter dictionaries
        """
        rqt_info = []
        for i, param in enumerate(self.rqt_parameters):
            rqt_info.append({
                'parameter': param,
                'config_name': self.rqt_parameter_config_names[i],
                'min_value': self.rqt_parameter_min_values[i],
                'max_value': self.rqt_parameter_max_values[i]
            })
        
        return rqt_info
    
    def validate_all_parameters(self) -> bool:
        """
        Validate all parameters using their validators.
        
        Returns:
            True if all parameters are valid, False otherwise
        """
        if self.parameter_values is None:
            LOG_WARN("No parameters loaded for validation")
            return False
        
        for key, validator in self.parameter_validators.items():
            try:
                value = self.get(key)
                validator(value)
            except Exception as e:
                LOG_WARN(f"Parameter validation failed for '{key}': {e}")
                return False
        
        return True
    
    def reset(self) -> None:
        """Reset the parameter manager."""
        self.parameter_values = None
        LOG_DEBUG("Parameter manager reset")
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return self.parameter_count
    
    def get_parameter_names(self) -> List[str]:
        """Get list of all parameter names."""
        return self.parameter_order.copy()
    
    def has_parameter(self, key: str) -> bool:
        """
        Check if a parameter exists.
        
        Args:
            key: Parameter name
            
        Returns:
            True if parameter exists, False otherwise
        """
        return key in self.parameter_lookup
    
    def get_parameter_length(self, key: str) -> int:
        """
        Get the length of a parameter.
        
        Args:
            key: Parameter name
            
        Returns:
            Parameter length
        """
        if key not in self.parameter_lookup:
            raise KeyError(f"Parameter '{key}' not found")
        
        return self.parameter_lookup[key][1]
    
    def create_parameter_subset(self, parameter_names: List[str]) -> 'ParameterManager':
        """
        Create a subset parameter manager with only specified parameters.
        
        Args:
            parameter_names: List of parameter names to include
            
        Returns:
            New ParameterManager with subset of parameters
        """
        subset_manager = ParameterManager()
        
        for param_name in parameter_names:
            if param_name in self.parameter_lookup:
                start_idx, length = self.parameter_lookup[param_name]
                subset_manager.add(param_name, length)
                
                # Copy metadata
                if param_name in self.parameter_metadata:
                    subset_manager.parameter_metadata[param_name] = self.parameter_metadata[param_name].copy()
        
        return subset_manager
    
    def __str__(self) -> str:
        """String representation of the parameter manager."""
        result = f"ParameterManager with {self.parameter_count} parameters:\n"
        for param_name in self.parameter_order:
            start_idx, length = self.parameter_lookup[param_name]
            result += f"  {param_name}: [{start_idx}:{start_idx + length}] (length={length})\n"
        return result
    
    def __len__(self) -> int:
        """Return the number of parameters."""
        return self.parameter_count
    
    def __contains__(self, key: str) -> bool:
        """Check if a parameter exists."""
        return key in self.parameter_lookup
    
    def __getitem__(self, key: str) -> Union[float, np.ndarray]:
        """Get a parameter value."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Union[float, int, np.ndarray, List[float]]) -> None:
        """Set a parameter value."""
        self.set_parameter(key, value)
