from solver.src.modules_manager import Module
from utils.const import CONSTRAINT
from utils.utils import read_config_file, LOG_DEBUG
import casadi as cd


class BaseConstraint(Module):
	def __init__(self, solver, settings=None):
		super().__init__()
		self.solver = solver
		self.name = self.__class__.__name__.lower()
		self.module_type = CONSTRAINT
		print("self.name is ", self.name)
		if settings is None:
			self.config = read_config_file()
		else:
			self.config = settings
		LOG_DEBUG(f"Initializing {self.name.title()} Constraints")

	def define_parameters(self, params):
		pass

	def get_penalty(self, symbolic_state, params, stage_idx):
		return cd.MX(0)