import copy

from utils.const import CONSTRAINT, OBJECTIVE
from utils.utils import LOG_DEBUG, read_config_file
from utils.utils import print_value, print_header, CONFIG, get_config_dotted
from utils.visualizer_compat import ROSLine


class ModuleManager:
    """
  Module Manager handles collections of constraint and objective modules
  for MPC. It coordinates updates, parameter setting, and visualization.
  """

    def __init__(self):
        self.modules = []

    def add_module(self, module):
        """Add a module instance and check dependencies"""
        # Check if module has dependencies
        if hasattr(module, 'dependencies') and module.dependencies:
            for dep_name in module.dependencies:
                # Check if dependency exists in modules
                dep_found = any(m.name == dep_name for m in self.modules)
                if not dep_found:
                    raise RuntimeError(
                        f"Module '{module.name}' requires dependency '{dep_name}' "
                        f"which is not present in the module manager. "
                        f"Please add '{dep_name}' before adding '{module.name}'."
                    )
        self.modules.append(module)

    def get_modules(self):
        return self.modules

    def is_data_ready(self, state, data):
        for module in self.modules:
            if not module.is_data_ready(state, data):
                return False
        return True

    def define_parameters(self, parameter_manager):
        """Define all module parameters for the solver"""
        for module in self.modules:
            if hasattr(module, "define_parameters"):
                module.define_parameters(self, parameter_manager)

    def objective(self, state, parameter_manager, stage_idx):
        """Calculate objective value from all objective modules"""
        objective_value = []
        for module in self.modules:
            if module.module_type == OBJECTIVE:
                if hasattr(module, "get_value"):
                    objective_value.append(module.get_value(state, parameter_manager, stage_idx))
        return objective_value

    def constraints(self, state, param, stage_idx):
        """Calculate constraint values from all constraint modules"""
        constraint_values = []
        for module in self.modules:
            if module.module_type == CONSTRAINT:
                if hasattr(module, "calculate_constraints"):
                    constraint_values.extend(module.calculate_constraints(state, parameter_manager, stage_idx))
        return constraint_values

    def constraint_lower_bounds(self):
        """Get lower bounds for all constraints"""
        bounds = []
        for module in self.modules:
            if module.module_type == CONSTRAINT:
                if hasattr(module, "lower_bounds"):
                    bounds.extend(module.lower_bounds())
        return bounds

    def constraint_upper_bounds(self):
        """Get upper bounds for all constraints"""
        bounds = []
        for module in self.modules:
            if module.module_type == CONSTRAINT:
                if hasattr(module, "upper_bounds"):
                    bounds.extend(module.upper_bounds())
        return bounds

    def constraint_number(self):
        """Get total number of constraints"""
        count = 0
        for module in self.modules:
            if module.module_type == CONSTRAINT:
                if hasattr(module, "n_constraints"):
                    count += module.n_constraints
        return count

    def update_parameters_all(self, parameter_manager):
        """Set parameters for all modules across all stages"""
        for module in self.modules:
            module.update_parameters(parameter_manager)

    def get_all_visualizers(self):
        """Trigger visualization for all modules"""
        visualizers = []
        for module in self.modules:
            module_visualizer = module.get_visualizer()
            if module_visualizer is not None:
                visualizers.append(module_visualizer)
        return visualizers

    def check_objectives_reached(self, state, data):
        """Check if all objectives have been reached"""
        objective_modules = []
        for module in self.modules:
            if module.module_type == OBJECTIVE:
                objective_modules.append(module)
        if len(objective_modules) == 0:
            return False

        for module in objective_modules:
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

    def copy(self, new_solver=None):
        """
        Create a copy of the module manager.
        
        Args:
            new_solver: Optional new solver instance to attach modules to.
                       If None, creates a deep copy with original solver references.
        """
        if new_solver is None:
            return copy.deepcopy(self)
        
        # Create new manager and copy modules, updating solver references
        new_manager = ModuleManager()
        for module in self.modules:
            # Create a shallow copy of the module and update its solver reference
            module_copy = copy.copy(module)
            module_copy.solver = new_solver
            new_manager.modules.append(module_copy)
        return new_manager

class Module:

    def __init__(self):
        self.name = None
        self.settings = read_config_file()
        self.module_type = None
        self.description = ""

        self.dependencies = []
        self.visualizer = None

    def __str__(self):
        result = self.description
        return result

    def update_parameters(self, parameter_manager):
        pass

    def define_parameters(self, parameter_manager):
        pass

    def build_visualizer(self):
        pass

    def is_data_ready(self, data, state):
        """Check if required data is available"""
        return True

    def on_data_received(self, data):
        """Process incoming data by type"""
        pass

    def reset(self):
        """Reset constraint state"""
        pass

    def get_name(self):
        return self.name
    
    def get_visualizer(self):
        return self.visualizer

    def copy(self):
        return copy.deepcopy(self)
