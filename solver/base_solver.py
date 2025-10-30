import copy
from abc import ABC, abstractmethod

from planning.src.types import Data
from solver.src.modules_manager import ModuleManager
from solver.src.parameter_manager import ParameterManager
from utils.const import CONSTRAINT, OBJECTIVE
from utils.utils import LOG_DEBUG


class BaseSolver(ABC):
    def __init__(self, config):
        self.config = config

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