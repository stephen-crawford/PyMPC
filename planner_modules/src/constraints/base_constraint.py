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

	def get_visualization_overlay(self):
		"""Optional visualization overlay for this constraint.

		Return a dict compatible with StandardizedVisualizer._plot_constraint_projection or None.
		This method is intentionally a no-op by default and safe for all modules.
		"""
		return None

	def define_parameters(self, params):
		"""Define parameters for this constraint module."""
		pass

	def get_penalty(self, symbolic_state, params, stage_idx):
		"""Get penalty terms for this constraint (default: no penalty)."""
		return cd.MX(0)
	
	def get_constraints(self, symbolic_state, params, stage_idx):
		"""Get constraint expressions for this module."""
		return []
	
	def get_lower_bound(self):
		"""Get lower bounds for constraints."""
		return []
	
	def get_upper_bound(self):
		"""Get upper bounds for constraints."""
		return []
	
	def is_data_ready(self, data):
		"""Check if required data is available."""
		return True