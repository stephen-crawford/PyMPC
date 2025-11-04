from planning.modules_manager import Module
from utils.const import CONSTRAINT
from utils.utils import read_config_file, LOG_DEBUG, LOG_INFO
import casadi as cd


class BaseConstraint(Module):
    def __init__(self, settings=None):
        super().__init__()  # This loads config as self.settings and self.config
        self.name = self.__class__.__name__.lower()
        self.module_type = CONSTRAINT
        # Override config if settings provided, otherwise use from Module base class
        if settings is not None:
            self.config = settings
            self.settings = settings
        LOG_INFO(f"Initializing {self.name.title()} Constraints")
        LOG_DEBUG(f"BaseConstraint.__init__: name={self.name}, module_type={self.module_type}")

    def get_visualizer(self):
        """Return a visualizer for this constraint module. Override in subclasses."""
        return None

