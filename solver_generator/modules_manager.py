from solver_generator.util.logging import print_value, print_header
from utils.utils import LOG_DEBUG
from utils.const import CONSTRAINT, OBJECTIVE


class ModuleManager:
	"""
  Module Manager handles collections of constraint and objective modules
  for MPC solvers. It coordinates updates, parameter setting, and visualization.
  """

	def __init__(self, solver=None):
		self.solver = solver
		self.constraint_modules = []
		self.objective_modules = []
		self.modules = []
		self.constraint_manager = None

	def add_module(self, module_class_or_instance):
		"""Create and add a module of the given class or accept an already created instance"""
		if isinstance(module_class_or_instance, type):
			# It's a class, so instantiate it
			module = module_class_or_instance(self.solver)
		else:
			# It's already an instance
			module = module_class_or_instance

		self.modules.append(module)

		if hasattr(module, "module_type"):
			if module.module_type == CONSTRAINT:
				self.constraint_modules.append(module)
			elif module.module_type == OBJECTIVE:
				self.objective_modules.append(module)

		return module

	def define_parameters(self, params):
		"""Define all module parameters for the solver"""
		for module in self.modules:
			if hasattr(module, "define_parameters"):
				module.define_parameters(params)

	def objective(self, z, param, model, settings, stage_idx):
		"""Calculate objective value from all objective modules"""
		objective_value = 0.0
		for module in self.objective_modules:
			if hasattr(module, "get_value"):
				objective_value += module.get_value(model, param, settings, stage_idx)
		return objective_value

	def constraints(self, z, param, model, settings, stage_idx):
		"""Calculate constraint values from all constraint modules"""
		if self.constraint_manager:
			return self.constraint_manager.inequality(z, param, settings, model)
		return []

	def constraint_lower_bounds(self):
		"""Get lower bounds for all constraints"""
		bounds = []
		for module in self.constraint_modules:
			if hasattr(module, "lower_bounds"):
				bounds.extend(module.lower_bounds())
		return bounds

	def constraint_upper_bounds(self):
		"""Get upper bounds for all constraints"""
		bounds = []
		for module in self.constraint_modules:
			if hasattr(module, "upper_bounds"):
				bounds.extend(module.upper_bounds())
		return bounds

	def constraint_number(self):
		"""Get total number of constraints"""
		count = 0
		for module in self.constraint_modules:
			if hasattr(module, "n_constraints"):
				count += module.n_constraints
		return count

	def update_all(self, state, data, module_data):
		"""Update all modules with current state and data"""
		for module in self.modules:
			missing_data = ""
			if module.is_data_ready(data, missing_data):
				module.update(state, data, module_data)

	def set_parameters_all(self, data, module_data):
		"""Set parameters for all modules across all stages"""
		if not self.solver:
			LOG_DEBUG("Warning: No solver set in ModuleManager")
			return

		for k in range(self.solver.N):
			for module in self.modules:
				missing_data = ""
				if module.is_data_ready(data, missing_data):
					module.set_parameters(data, module_data, k)

	def visualize_all(self, data, module_data):
		"""Trigger visualization for all modules"""
		for module in self.modules:
			missing_data = ""
			if module.is_data_ready(data, missing_data):
				module.visualize(data, module_data)

	def check_objectives_reached(self, state, data):
		"""Check if all objectives have been reached"""
		if not self.objective_modules:
			return False

		for module in self.objective_modules:
			if hasattr(module, "is_objective_reached") and not module.is_objective_reached(state, data):
				return False
		return True

	def reset_all(self):
		"""Reset all modules"""
		for module in self.modules:
			if hasattr(module, "reset"):
				module.reset()

	def __str__(self):
		result = "--- MPC Modules ---\n"
		for module in self.modules:
			module_name = getattr(module, "name", "Unnamed Module")
			result += f"{module_name}: {str(module)}\n"
		return result

	def print(self):
		print_header("MPC Modules")
		for module in self.modules:
			module_name = getattr(module, "name", "Unnamed Module")
			print_value(module_name, str(module), tab=True)


class ConstraintManager:
	"""Manages and combines all constraints"""

	def __init__(self, module_manager):
		self.module_manager = module_manager
		self.nh = self.module_manager.constraint_number()  # Number of inequality constraints
		LOG_DEBUG(f"ConstraintManager initialized with {self.nh} constraints")

	def inequality(self, z, param, settings, model):
		"""Evaluate all inequality constraints"""
		constraints_list = []

		for module in self.module_manager.constraint_modules:
			if hasattr(module, "evaluate_constraints"):
				constraints_list.extend(module.evaluate_constraints(z, param, settings, model))

		return constraints_list


# Helper functions that interface with the solver generator
def define_parameters(modules, params, settings):
	"""Define parameters for all modules"""
	modules.define_parameters(params)


def objective(modules, z, p, model, settings, stage_idx):
	"""Calculate objective for all modules at given stage"""
	return modules.objective(z, p, model, settings, stage_idx)


def constraints(modules, z, p, model, settings, stage_idx):
	"""Calculate constraints for all modules at given stage"""
	return modules.constraints(z, p, model, settings, stage_idx)


def constraint_lower_bounds(modules):
	"""Get lower bounds for all constraints"""
	return modules.constraint_lower_bounds()


def constraint_upper_bounds(modules):
	"""Get upper bounds for all constraints"""
	return modules.constraint_upper_bounds()


def constraint_number(modules):
	"""Get total number of constraints"""
	return modules.constraint_number()


def initialize_module_manager(solver=None):
	"""Factory function to create and set up module manager"""
	module_manager = ModuleManager(solver)

	# Create constraint manager after modules are added
	module_manager.constraint_manager = ConstraintManager(module_manager)

	return module_manager