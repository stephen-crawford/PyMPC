"""
Module Manager for MPC.

This module provides the ModuleManager class that handles collections of 
constraint and objective modules for MPC, matching the original C++ implementation.
"""

import copy
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

from pympc.utils.utils import LOG_DEBUG, LOG_INFO, LOG_WARN
from pympc.utils.const import CONSTRAINT, OBJECTIVE


class ModuleManager:
    """
    Module Manager handles collections of constraint and objective modules
    for MPC. It coordinates updates, parameter setting, and visualization.
    
    This matches the original C++ ModuleManager implementation from
    tud-amr/mpc_planner.
    """
    
    def __init__(self):
        """Initialize the module manager."""
        self.modules: List[BaseModule] = []
        self._module_lookup: Dict[str, BaseModule] = {}
        self._constraint_modules: List[BaseModule] = []
        self._objective_modules: List[BaseModule] = []
        
    def add_module(self, module: 'BaseModule') -> None:
        """
        Add a module instance to the manager.
        
        Args:
            module: Module instance to add
        """
        if not isinstance(module, BaseModule):
            raise TypeError("Module must inherit from BaseModule")
        
        # Check for duplicate names
        if module.name in self._module_lookup:
            LOG_WARN(f"Module '{module.name}' already exists, replacing...")
            self.remove_module(module.name)
        
        self.modules.append(module)
        self._module_lookup[module.name] = module
        
        # Categorize modules
        if module.module_type == CONSTRAINT:
            self._constraint_modules.append(module)
        elif module.module_type == OBJECTIVE:
            self._objective_modules.append(module)
        
        LOG_DEBUG(f"Added module: {module.name} (type: {module.module_type})")
    
    def remove_module(self, module_name: str) -> bool:
        """
        Remove a module by name.
        
        Args:
            module_name: Name of module to remove
            
        Returns:
            True if module was removed, False if not found
        """
        if module_name not in self._module_lookup:
            return False
        
        module = self._module_lookup[module_name]
        self.modules.remove(module)
        del self._module_lookup[module_name]
        
        # Remove from categorized lists
        if module in self._constraint_modules:
            self._constraint_modules.remove(module)
        if module in self._objective_modules:
            self._objective_modules.remove(module)
        
        LOG_DEBUG(f"Removed module: {module_name}")
        return True
    
    def get_module(self, module_name: str) -> Optional['BaseModule']:
        """
        Get a module by name.
        
        Args:
            module_name: Name of module to retrieve
            
        Returns:
            Module instance or None if not found
        """
        return self._module_lookup.get(module_name)
    
    def get_modules(self) -> List['BaseModule']:
        """Get all modules."""
        return self.modules.copy()
    
    def get_constraint_modules(self) -> List['BaseModule']:
        """Get all constraint modules."""
        return self._constraint_modules.copy()
    
    def get_objective_modules(self) -> List['BaseModule']:
        """Get all objective modules."""
        return self._objective_modules.copy()
    
    def is_data_ready(self, data) -> bool:
        """
        Check if all modules have required data.
        
        Args:
            data: Data object to check
            
        Returns:
            True if all modules are ready, False otherwise
        """
        for module in self.modules:
            if not module.is_data_ready(data):
                LOG_DEBUG(f"Module {module.name} not ready")
                return False
        return True
    
    def define_parameters(self, parameter_manager) -> None:
        """
        Define parameters for all modules.
        
        Args:
            parameter_manager: Parameter manager instance
        """
        LOG_DEBUG("Defining parameters for all modules")
        for module in self.modules:
            if hasattr(module, "define_parameters"):
                try:
                    module.define_parameters(parameter_manager)
                    LOG_DEBUG(f"Defined parameters for module: {module.name}")
                except Exception as e:
                    LOG_WARN(f"Error defining parameters for module {module.name}: {e}")
    
    def update_all(self, state, data) -> None:
        """
        Update all modules with current state and data.
        
        Args:
            state: Current state
            data: Current data
        """
        LOG_DEBUG("Updating all modules")
        for module in self.modules:
            try:
                if hasattr(module, "update"):
                    module.update(state, data)
                if hasattr(module, "on_data_received"):
                    module.on_data_received(data)
            except Exception as e:
                LOG_WARN(f"Error updating module {module.name}: {e}")
    
    def set_parameters_all(self, parameter_manager, data, horizon: int) -> None:
        """
        Set parameters for all modules across all stages.
        
        Args:
            parameter_manager: Parameter manager instance
            data: Current data
            horizon: Prediction horizon length
        """
        LOG_DEBUG(f"Setting parameters for all modules across horizon {horizon}")
        for k in range(horizon):
            for module in self.modules:
                if module.is_data_ready(data):
                    try:
                        if hasattr(module, "set_parameters"):
                            module.set_parameters(parameter_manager, data, k)
                    except Exception as e:
                        LOG_WARN(f"Error setting parameters for module {module.name} at stage {k}: {e}")
    
    def get_objectives(self, symbolic_state, parameter_manager, stage_idx: int) -> List[Any]:
        """
        Calculate objective values from all objective modules.
        
        Args:
            symbolic_state: Symbolic state variables
            parameter_manager: Parameter manager instance
            stage_idx: Current stage index
            
        Returns:
            List of objective values
        """
        objectives = []
        for module in self._objective_modules:
            try:
                if hasattr(module, "get_value"):
                    obj_value = module.get_value(symbolic_state, parameter_manager, stage_idx)
                    if obj_value is not None:
                        objectives.append(obj_value)
                elif hasattr(module, "get_stage_cost_symbolic"):
                    obj_value = module.get_stage_cost_symbolic(symbolic_state, parameter_manager, stage_idx)
                    if obj_value is not None:
                        objectives.append(obj_value)
            except Exception as e:
                LOG_WARN(f"Error getting objective from module {module.name}: {e}")
                continue
        
        return objectives
    
    def get_constraints(self, symbolic_state, parameter_manager, stage_idx: int) -> List[Any]:
        """
        Calculate constraint values from all constraint modules.
        
        Args:
            symbolic_state: Symbolic state variables
            parameter_manager: Parameter manager instance
            stage_idx: Current stage index
            
        Returns:
            List of constraint expressions
        """
        constraints = []
        for module in self._constraint_modules:
            try:
                if hasattr(module, "get_constraints"):
                    module_constraints = module.get_constraints(symbolic_state, parameter_manager, stage_idx)
                    if module_constraints:
                        constraints.extend(module_constraints)
            except Exception as e:
                LOG_WARN(f"Error getting constraints from module {module.name}: {e}")
                continue
        
        return constraints
    
    def get_constraint_bounds(self) -> tuple:
        """
        Get constraint bounds from all constraint modules.
        
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        lower_bounds = []
        upper_bounds = []
        
        for module in self._constraint_modules:
            try:
                if hasattr(module, "get_lower_bound"):
                    module_lower = module.get_lower_bound()
                    if module_lower:
                        lower_bounds.extend(module_lower)
                
                if hasattr(module, "get_upper_bound"):
                    module_upper = module.get_upper_bound()
                    if module_upper:
                        upper_bounds.extend(module_upper)
            except Exception as e:
                LOG_WARN(f"Error getting bounds from module {module.name}: {e}")
                continue
        
        return lower_bounds, upper_bounds
    
    def get_penalties(self, symbolic_state, parameter_manager, stage_idx: int) -> List[Any]:
        """
        Calculate penalty terms from all modules.
        
        Args:
            symbolic_state: Symbolic state variables
            parameter_manager: Parameter manager instance
            stage_idx: Current stage index
            
        Returns:
            List of penalty terms
        """
        penalties = []
        for module in self.modules:
            try:
                if hasattr(module, "get_penalty"):
                    penalty = module.get_penalty(symbolic_state, parameter_manager, stage_idx)
                    if penalty is not None:
                        penalties.append(penalty)
            except Exception as e:
                LOG_WARN(f"Error getting penalty from module {module.name}: {e}")
                continue
        
        return penalties
    
    def visualize_all(self, data) -> Dict[str, Any]:
        """
        Trigger visualization for all modules.
        
        Args:
            data: Current data
            
        Returns:
            Dictionary of visualization overlays
        """
        overlays = {}
        for module in self.modules:
            try:
                if hasattr(module, "get_visualization_overlay"):
                    overlay = module.get_visualization_overlay()
                    if overlay:
                        overlays[module.name] = overlay
            except Exception as e:
                LOG_WARN(f"Error getting visualization from module {module.name}: {e}")
                continue
        
        return overlays
    
    def check_objectives_reached(self, state, data) -> bool:
        """
        Check if all objectives have been reached.
        
        Args:
            state: Current state
            data: Current data
            
        Returns:
            True if all objectives reached, False otherwise
        """
        if not self._objective_modules:
            return False
        
        for module in self._objective_modules:
            try:
                if hasattr(module, "is_objective_reached"):
                    if not module.is_objective_reached(state, data):
                        return False
            except Exception as e:
                LOG_WARN(f"Error checking objective reached for module {module.name}: {e}")
                return False
        
        return True
    
    def reset_all(self) -> None:
        """Reset all modules."""
        LOG_DEBUG("Resetting all modules")
        for module in self.modules:
            try:
                if hasattr(module, "reset"):
                    module.reset()
            except Exception as e:
                LOG_WARN(f"Error resetting module {module.name}: {e}")
    
    def get_module_count(self) -> int:
        """Get total number of modules."""
        return len(self.modules)
    
    def get_constraint_count(self) -> int:
        """Get number of constraint modules."""
        return len(self._constraint_modules)
    
    def get_objective_count(self) -> int:
        """Get number of objective modules."""
        return len(self._objective_modules)
    
    def __str__(self) -> str:
        """String representation of the module manager."""
        result = "--- MPC Module Manager ---\n"
        result += f"Total modules: {self.get_module_count()}\n"
        result += f"Constraint modules: {self.get_constraint_count()}\n"
        result += f"Objective modules: {self.get_objective_count()}\n\n"
        
        for module in self.modules:
            result += f"{module.name}: {module.module_type}\n"
        
        return result
    
    def copy(self) -> 'ModuleManager':
        """Create a deep copy of the module manager."""
        return copy.deepcopy(self)


class BaseModule(ABC):
    """
    Abstract base class for all MPC modules.
    
    This matches the original C++ BaseModule implementation.
    """
    
    def __init__(self, solver=None):
        """
        Initialize the base module.
        
        Args:
            solver: Solver instance (optional)
        """
        self.name = self.__class__.__name__.lower()
        self.module_type = None  # To be set by subclasses
        self.solver = solver
        self.config = {}
        
    @abstractmethod
    def is_data_ready(self, data) -> bool:
        """
        Check if required data is available.
        
        Args:
            data: Data object to check
            
        Returns:
            True if data is ready, False otherwise
        """
        pass
    
    def update(self, state, data) -> None:
        """
        Update module with current state and data.
        
        Args:
            state: Current state
            data: Current data
        """
        pass
    
    def on_data_received(self, data) -> None:
        """
        Process incoming data.
        
        Args:
            data: Incoming data
        """
        pass
    
    def define_parameters(self, parameter_manager) -> None:
        """
        Define parameters for the module.
        
        Args:
            parameter_manager: Parameter manager instance
        """
        pass
    
    def set_parameters(self, parameter_manager, data, k: int) -> None:
        """
        Set parameter values for the module.
        
        Args:
            parameter_manager: Parameter manager instance
            data: Current data
            k: Stage index
        """
        pass
    
    def get_visualization_overlay(self) -> Optional[Dict[str, Any]]:
        """
        Get visualization overlay for the module.
        
        Returns:
            Visualization overlay dictionary or None
        """
        return None
    
    def reset(self) -> None:
        """Reset module state."""
        pass
    
    def get_config_value(self, key: str, default=None):
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set_config_value(self, key: str, value) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
