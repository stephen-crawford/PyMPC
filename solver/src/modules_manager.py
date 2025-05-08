from utils.const import CONSTRAINT, OBJECTIVE
from utils.utils import LOG_DEBUG
from utils.utils import print_value, print_header, CONFIG, get_config_dotted
from utils.visualizer import ROSLine


class ModuleManager:
    """
  Module Manager handles collections of constraint and objective modules
  for MPC. It coordinates updates, parameter setting, and visualization.
  """

    def __init__(self):
        self.modules = []

    def add_module(self, module):
        """Add a module instance"""
        self.modules.append(module)

    def get_modules(self):
        return self.modules

    def is_data_ready(self, data):
        for module in self.modules:
            if not module.is_data_ready(data):
                return False

    def define_parameters(self, params):
        """Define all module parameters for the solver"""
        for module in self.modules:
            if hasattr(module, "define_parameters"):
                module.define_parameters(params)

    def objective(self, param, model, settings, stage_idx):
        """Calculate objective value from all objective modules"""
        objective_value = 0.0
        for module in self.modules:
            if module.module_type == OBJECTIVE:
                if hasattr(module, "get_value"):
                    objective_value += module.get_value(model, param, settings, stage_idx)
        return objective_value

    def constraints(self, param, model, settings, stage_idx):
        """Calculate constraint values from all constraint modules"""
        constraint_values = []
        for module in self.modules:
            if module.module_type == CONSTRAINT:
                if hasattr(module, "calculate_constraints"):
                    constraint_values.extend(module.calculate_constraints(param, settings, stage_idx))
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

    def update_all(self, state, data, module_data):
        """Update all modules with current state and data"""
        for module in self.modules:
            missing_data = ""
            if module.is_data_ready(data, missing_data):
                module.update(state, data, module_data)

    def set_parameters_all(self, data, module_data, solver_N):
        """Set parameters for all modules across all stages"""
        for k in range(solver_N):
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

class Module:

    def __init__(self):
        self.name = None
        self.config = None
        self.module_type = None
        self.description = ""

        self.dependencies = []
        self.parameters_requests = []
        self.visualizer = None

    def write_to_solver(self, header_file):
        return

    def __str__(self):
        result = self.description
        return result

    def set_parameters(self, parameter_manager, data, module_data, k):
        pass

    def add_definitions(self, header_file):
        pass

    def update(self, state, data, module_data):
        """Update constraint with current state and data"""
        pass

    def get_module_parameter_requests(self):
        return self.parameters_requests

    def visualize(self, data, module_data):
        """Visualize constraint state"""
        if not self.config.get("debug_visuals", CONFIG.get("debug_visuals", False)):
            return
        LOG_DEBUG(f"{self.name.title()}::Visualize")

    def is_data_ready(self, state, missing_data):
        """Check if required data is available"""
        return ""

    def on_data_received(self, data, data_name):
        """Process incoming data by type"""
        pass

    def reset(self):
        """Reset constraint state"""
        pass

    def get_config_value(self, key, default=None):
        res = self.config.get(key, CONFIG.get(f"{self.name}.{key}", default))
        if res is None:
            res = get_config_dotted(self.config, key)
        return res

    def create_visualization_publisher(self, name_suffix, publisher_type=ROSLine):
        """Create standardized visualization publisher"""
        publisher_name = f"{self.name}/{name_suffix}"
        return publisher_type(publisher_name, "map")


