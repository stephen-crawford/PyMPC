import copy
from abc import ABC, abstractmethod

from planning.types import Data
from planning.modules_manager import ModuleManager
from planning.parameter_manager import ParameterManager
from utils.const import CONSTRAINT, OBJECTIVE
from utils.utils import LOG_DEBUG


class BaseSolver(ABC):
    def __init__(self, config):
        self.config = config
        self.module_manager = ModuleManager()
        self.parameter_manager = ParameterManager()
        self.data = None

    def initialize_solver(self, data):
        pass

    def initialize_rollout(self, state, data, shift_forward=True):
        pass

    def _initialize_base_rollout(self, state, data):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def get_output(self, k, var_name):
        pass

    @abstractmethod
    def explain_exit_flag(self, code):
        pass
    
    def define_parameters(self):
        """Define parameters for all modules."""
        for module in self.module_manager.get_modules():
            if hasattr(module, "define_parameters"):
                module.define_parameters(self.parameter_manager)

    # Standardized planner ↔ solver interface: fetch objectives/constraints via ModuleManager
    def get_objective_cost(self, state, stage_idx):
        """Return a list of objective term dicts for the given stage.
        This delegates to ModuleManager to aggregate module-provided costs.
        """
        return self.module_manager.get_objectives(state, self.data, stage_idx) or []

    def get_constraints(self, stage_idx):
        """Return a list of (constraint, lb, ub) tuples for the given stage.
        ModuleManager supplies constraints and bounds; solver subclasses convert them as needed.
        """
        state = None
        if hasattr(self.module_manager, 'get_constraints_with_bounds'):
            return self.module_manager.get_constraints_with_bounds(state, self.data, stage_idx)
        cons = self.module_manager.get_constraints(state, self.data, stage_idx) or []
        lbs = self.module_manager.get_lower_bounds(state, self.data, stage_idx) or []
        ubs = self.module_manager.get_upper_bounds(state, self.data, stage_idx) or []
        n = max(len(cons), len(lbs), len(ubs))
        result = []
        for i in range(n):
            c = cons[i] if i < len(cons) else None
            lb = lbs[i] if i < len(lbs) else None
            ub = ubs[i] if i < len(ubs) else None
            if c is not None:
                result.append((c, lb, ub))
        return result