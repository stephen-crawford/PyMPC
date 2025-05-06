import copy
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional, List

from solver.src.modules_manager import ModuleManager
from utils.const import CONSTRAINT, OBJECTIVE
from utils.utils import parameter_map_path, write_to_yaml, print_header, print_value


class ParameterManager:

    def __init__(self):
        self._params = dict()

        self.parameter_bundles = dict()  # Used to generate function names in C++ with an integer parameter

        self.rqt_params = []
        self.rqt_param_config_names = []
        self.rqt_param_min_values = []
        self.rqt_param_max_values = []

        self._param_idx = 0
        self._p = None

    def add(
        self,
        parameter,
        add_to_rqt_reconfigure=False,
        rqt_config_name=lambda p: f'["weights"]["{p}"]',
        bundle_name=None,
        rqt_min_value=0.0,
        rqt_max_value=100.0,
    ):
        """
        Adds a parameter to the parameter dictionary.

        Args:
            parameter (Any): The parameter to be added.
            add_to_rqt_reconfigure (bool, optional): Whether to add the parameter to the RQT Reconfigure. Defaults to False.
            rqt_config_name (function, optional): A function that returns the name of the parameter in CONFIG for the parameter in RQT Reconfigure. Defaults to lambda p: f'["weights"]["{p}"]'.
        """

        if parameter in self._params.keys():
            return

        self._params[parameter] = copy.deepcopy(self._param_idx)
        if bundle_name is None:
            bundle_name = parameter

        if bundle_name not in self.parameter_bundles.keys():
            self.parameter_bundles[bundle_name] = [copy.deepcopy(self._param_idx)]
        else:
            self.parameter_bundles[bundle_name].append(copy.deepcopy(self._param_idx))

        self._param_idx += 1

        if add_to_rqt_reconfigure:
            self.rqt_params.append(parameter)
            self.rqt_param_config_names.append(rqt_config_name)
            self.rqt_param_min_values.append(rqt_min_value)
            self.rqt_param_max_values.append(rqt_max_value)

    def length(self):
        return self._param_idx

    def load(self, p):
        self._p = p

    def save_map(self):
        file_path = parameter_map_path()

        map = self._params
        map["num parameters"] = self._param_idx
        write_to_yaml(file_path, self._params)

    def get_p(self) -> float:
        return self._p

    def get(self, parameter):
        if self._p is None:
            print("Load parameters before requesting them!")

        return self._p[self._params[parameter]]

    def has_parameter(self, parameter):
        return parameter in self._params

    def print(self):
        print_header("Parameters")
        print("----------")
        for param, idx in self._params.items():
            if param in self.rqt_params:
                print_value(f"{idx}", f"{param} (in rqt_reconfigure)", tab=True)
            else:
                print_value(f"{idx}", f"{param}", tab=True)
        print("----------")


class BaseSolver(ABC):
    def __init__(self, dt, N):
        self.dt = dt
        self.N = N
        self.params = ParameterManager()  # Use the unified parameter manager
        self.module_manager = ModuleManager()
        # Register default parameters
        self.params.register_parameter("solver_timeout", 0.1)

    # TODO: Parameter setting needs to be fixed
    def set_solver_parameter(self, key, value, index=0, settings=None):
        """Universal parameter setter that works with different parameter storage approaches"""
        if hasattr(self.params, 'set_parameter'):
            # Use the unified interface if available
            self.params.set_parameter(key, value, index)
        elif hasattr(self.params, 'all_parameters') and settings is not None:
            # Legacy CasADI-style approach
            bundles = settings["params"].parameter_bundles
            length = settings["params"].length()

            if key not in bundles:
                raise KeyError(f"Parameter '{key}' not found in parameter bundles.")

            indices = bundles[key]

            # Handling single index case
            if len(indices) == 1:
                param_index = index * length + indices[0]
                self.params.all_parameters[param_index] = value
            else:
                # Handling multiple indices case
                if index < 0 or index >= len(indices):
                    raise IndexError(f"Index {index} out of bounds for parameter bundle '{key}'")

                param_index = index * length + indices[index]
                self.params.all_parameters[param_index] = value
        else:
            # Simple dictionary-based approach
            if isinstance(self.params, dict):
                self.params[key] = value
            else:
                setattr(self.params, key, value)

    def set_parameter(self, key, value, index=0):
        """Unified parameter setting interface"""
        self.set_solver_parameter(key, value, index)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def set_initial_state(self, state):
        pass

    @abstractmethod
    def get_output(self, k, var_name):
        pass

    @abstractmethod
    def explain_exit_flag(self, code):
        pass

    @staticmethod
    def define_parameters(module_manager, params, settings):

        # Define parameters for objectives and constraints (in order)
        for module in module_manager.modules:
            if module.module_type == OBJECTIVE:
                module.define_parameters(params)

        for module in module_manager.modules:
            if module.module_type == CONSTRAINT:
                module.define_parameters(params)

        return params
    @staticmethod
    def objective(module_manager, z, p, model, settings, stage_idx):
        cost = 0.0

        params = settings["params"]
        params.load(p)
        model.load(z)

        for module in module_manager.modules:
            if module.module_type == OBJECTIVE:
                cost += module.get_value(model, params, settings, stage_idx)

        # if stage_idx == 0:
        # print(cost)

        return cost


    @staticmethod
    def constraints(module_manager, z, p, model, settings, stage_idx):
        constraints = []

        params = settings["params"]
        params.load(p)
        model.load(z)

        for module in module_manager.modules:
            if module.module_type == CONSTRAINT:
                constraints += module.get_constraints(model, params, settings, stage_idx)

        return constraints

    @staticmethod
    def constraint_upper_bounds(module_manager):
        ub = []
        for module in module_manager.modules:
            if module.module_type == CONSTRAINT:
                ub += module.get_upper_bound()
        return ub

    @staticmethod
    def constraint_lower_bounds(module_manager):
        lb = []
        for module in module_manager.modules:
            if module.module_type == CONSTRAINT:
                lb += module.get_lower_bound()
        return lb

    @staticmethod
    def constraint_number(module_manager):
        nh = 0
        for module in module_manager.modules:
            if module.module_type == CONSTRAINT:
                nh += module.nh
        return nh