from abc import abstractmethod

from solver.src.modules_manager import Module
from utils.const import OBJECTIVE
from utils.utils import read_config_file, LOG_DEBUG

class BaseObjective(Module):
	def __init__(self, solver):
		super().__init__()
		self.solver = solver
		self.name = "_objective".join(self.__class__.__name__.lower().split('objective'))
		self.module_type = OBJECTIVE
		self.controller = (self.module_type, solver, self.name)
		self.config = read_config_file()
		LOG_DEBUG(f"Initializing {self.name.title()} Objective")

	@abstractmethod
	def get_stage_cost_symbolic(self, symbolic_state, stage_idx):
		pass
