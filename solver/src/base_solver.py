from abc import ABC, abstractmethod

from solver.src.modules_manager import ModuleManager
from solver.src.parameter_manager import ParameterManager
from utils.const import CONSTRAINT, OBJECTIVE


class BaseSolver(ABC):
    def __init__(self, timestep=0.1, horizon=30):
        self.timestep = timestep
        self.horizon = horizon
        self.parameter_manager = ParameterManager()  # Use the unified parameter manager
        self.module_manager = ModuleManager()
        # Register default parameters
        self.parameter_manager.set_parameter("solver_timeout", 0.1)

    def get_module_manager(self):
        return self.module_manager

    def get_parameter_manager(self):
        return self.parameter_manager

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
    def define_parameters(module_manager, parameter_manager, settings):

        # Define parameters for objectives and constraints (in order)
        for module in module_manager.modules:
            if module.module_type == OBJECTIVE:
                module.define_parameters(parameter_manager)

        for module in module_manager.modules:
            if module.module_type == CONSTRAINT:
                module.define_parameters(parameter_manager)

        return parameter_manager
    @staticmethod
    def get_objective_cost(module_manager, z, p, model, settings, stage_idx):
        cost = 0.0

        parameter_manager = settings["parameter_manager"]
        parameter_manager.load(p)
        model.load(z)

        for module in module_manager.modules:
            if module.module_type == OBJECTIVE:
                cost += module.get_value(model, parameter_manager, settings, stage_idx)

        # if stage_idx == 0:
        # print(cost)

        return cost


    @staticmethod
    def get_constraint_list(module_manager, z, p, model, settings, stage_idx):
        constraints = []

        parameter_manager = settings["parameter_manager"]
        parameter_manager.load(p)
        model.load(z)

        for module in module_manager.modules:
            if module.module_type == CONSTRAINT:
                constraints += module.get_constraints(model, parameter_manager, settings, stage_idx)

        return constraints

    @staticmethod
    def get_constraint_upper_bounds_list(module_manager):
        ub = []
        for module in module_manager.modules:
            if module.module_type == CONSTRAINT:
                ub += module.get_upper_bound()
        return ub

    @staticmethod
    def get_constraint_lower_bounds_list(module_manager):
        lb = []
        for module in module_manager.modules:
            if module.module_type == CONSTRAINT:
                lb += module.get_lower_bound()
        return lb
