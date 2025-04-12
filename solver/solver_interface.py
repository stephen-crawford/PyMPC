from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional, List


class ParameterManager:
    """Manages parameters across different solver backends"""

    def __init__(self):
        # Backend-specific storage
        self.all_parameters = []  # For CasADI-like indexed access
        self.parameter_dict = {}  # For generic dictionary-based access
        self.parameter_bundles = {}  # Maps parameter names to indices

    def register_parameter(self, key: str, default_value, indices: Union[int, List[int]] = None):
        """Registers a parameter in both storage systems"""
        # Store in generic dictionary
        self.parameter_dict[key] = default_value

        # Store in indexed system (CasADI compatible)
        if indices is not None:
            if isinstance(indices, int):
                indices = [indices]
            self.parameter_bundles[key] = indices

            # Ensure all_parameters list is large enough
            max_index = max(indices)
            while len(self.all_parameters) <= max_index:
                self.all_parameters.append(None)

            # Set value at all relevant indices
            for idx in indices:
                self.all_parameters[idx] = default_value

    def set_parameter(self, key: str, value, index: int = 0):
        """Set parameter value in both storage systems"""
        # Update generic dictionary
        if index == 0 or key not in self.parameter_bundles:
            self.parameter_dict[key] = value

        # Update indexed system if applicable
        if key in self.parameter_bundles:
            indices = self.parameter_bundles[key]
            if len(indices) == 1:
                self.all_parameters[indices[0]] = value
            else:
                if index < 0 or index >= len(indices):
                    raise IndexError(f"Index {index} out of bounds for parameter bundle '{key}'")
                self.all_parameters[indices[index]] = value

    def get_parameter(self, key: str, index: int = 0):
        """Get parameter value from either storage system"""
        # Try to get from indexed system first if applicable
        if key in self.parameter_bundles:
            indices = self.parameter_bundles[key]
            if len(indices) == 1:
                return self.all_parameters[indices[0]]
            else:
                if index < 0 or index >= len(indices):
                    raise IndexError(f"Index {index} out of bounds for parameter bundle '{key}'")
                return self.all_parameters[indices[index]]

        # Fall back to dictionary
        if key in self.parameter_dict:
            return self.parameter_dict[key]

        raise KeyError(f"Parameter '{key}' not found")

    def length(self):
        """Return the length of the all_parameters list"""
        return len(self.all_parameters)


def set_solver_parameter(params, key, value, k=0, index=0, settings=None):
    """Universal parameter setter that works with different parameter storage approaches"""
    if hasattr(params, 'set_parameter'):
        # Use the unified interface if available
        params.set_parameter(key, value, index)
    elif hasattr(params, 'all_parameters') and settings is not None:
        # Legacy CasADI-style approach
        bundles = settings["params"].parameter_bundles
        length = settings["params"].length()

        if key not in bundles:
            raise KeyError(f"Parameter '{key}' not found in parameter bundles.")

        indices = bundles[key]

        # Handling single index case
        if len(indices) == 1:
            param_index = k * length + indices[0]
            params.all_parameters[param_index] = value
        else:
            # Handling multiple indices case
            if index < 0 or index >= len(indices):
                raise IndexError(f"Index {index} out of bounds for parameter bundle '{key}'")

            param_index = k * length + indices[index]
            params.all_parameters[param_index] = value
    else:
        # Simple dictionary-based approach
        if isinstance(params, dict):
            params[key] = value
        else:
            setattr(params, key, value)


class BaseSolver(ABC):
    def __init__(self, dt, N):
        self.dt = dt
        self.N = N
        self.params = ParameterManager()  # Use the unified parameter manager

        # Register default parameters
        self.params.register_parameter("solver_timeout", 0.1)

    def set_parameter(self, key, value, index=0):
        """Unified parameter setting interface"""
        set_solver_parameter(self.params, key, value, index=index)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def set_xinit(self, state):
        pass

    @abstractmethod
    def get_output(self, k, var_name):
        pass

    @abstractmethod
    def explain_exit_flag(self, code):
        pass