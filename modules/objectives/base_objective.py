from abc import abstractmethod

from planning.modules_manager import Module
from utils.const import OBJECTIVE
from utils.utils import read_config_file, LOG_DEBUG, LOG_INFO

class BaseObjective(Module):
    def __init__(self):
        super().__init__()
        self.name = "_objective".join(self.__class__.__name__.lower().split('objective'))
        self.module_type = OBJECTIVE
        self.controller = (self.module_type, None, self.name)
        self.solver = None
        # Config is already loaded in Module.__init__ as self.settings and self.config
        LOG_INFO(f"Initializing {self.name.title()} Objective")
        LOG_DEBUG(f"BaseObjective.__init__: name={self.name}, module_type={self.module_type}")

    def get_objective(self, state, data, stage_idx):
        # Default adapter: call legacy get_value if implemented
        LOG_DEBUG(f"BaseObjective.get_objective: {self.name}, stage_idx={stage_idx}")
        if hasattr(self, 'get_value'):
            LOG_DEBUG(f"  Using legacy get_value method")
            result = self.get_value(state, data, stage_idx)
            LOG_DEBUG(f"  get_value returned {len(result) if isinstance(result, (list, tuple)) else 1 if result else 0} term(s)")
            return result
        LOG_DEBUG(f"  No get_value method, returning empty list")
        return []

    @abstractmethod
    def get_stage_cost_symbolic(self, symbolic_state, stage_idx):
        pass
