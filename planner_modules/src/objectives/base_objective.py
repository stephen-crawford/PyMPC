from solver.src.modules_manager import Module
from utils.const import OBJECTIVE
from utils.utils import read_config_file, LOG_DEBUG

class BaseObjective(Module):
	def __init__(self, solver, settings=None):
		super().__init__()
		self.solver = solver
		self.name = self.__class__.__name__.lower()
		self.module_type = OBJECTIVE
		self.controller = (self.module_type, solver, self.name)
		if settings is None:
			self.config = read_config_file()
		else:
			self.config = settings
		LOG_DEBUG(f"Initializing {self.name.title()} Objective")

