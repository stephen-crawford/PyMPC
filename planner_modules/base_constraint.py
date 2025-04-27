from solver.solver_interface import set_solver_parameter
from utils.const import CONSTRAINT
from utils.utils import read_config_file, LOG_DEBUG, CONFIG, get_config_dotted
from utils.visualizer import ROSLine

class BaseConstraint:
	def __init__(self, solver):
		self.solver = solver
		self.name = self.__class__.__name__.lower()
		self.module_type = CONSTRAINT
		print("self.name is ", self.name)
		self.config = read_config_file()
		LOG_DEBUG(f"Initializing {self.name.title()} Constraints")

	def update(self, state, data, module_data):
		"""Update constraint with current state and data"""
		pass

	def set_solver_parameter(self, param_name, value, k, index=None):
		"""Set solver parameter with consistent approach"""
		if index is not None:
			set_solver_parameter(self.solver.params, param_name, value, k, index=index, settings=CONFIG)
		else:
			set_solver_parameter(self.solver.params, param_name, value, k, settings=CONFIG)

	def visualize(self, data, module_data):
		"""Visualize constraint state"""
		if not self.config.get("debug_visuals", CONFIG.get("debug_visuals", False)):
			return
		LOG_DEBUG(f"{self.name.title()}::Visualize")

	def is_data_ready(self, data, missing_data):
		"""Check if required data is available"""
		return True

	def on_data_received(self, data, data_name):
		"""Process incoming data by type"""
		pass

	def reset(self):
		"""Reset constraint state"""
		pass

	def get_config_value(self, key, default=None):
		print("Searching config for value, " + str(key))
		print("Found value: " + str(self.config.get(key)))
		print("Config is: " + str(self.config))
		res = self.config.get(key, CONFIG.get(f"{self.name}.{key}", default))
		if res is None:
			res = get_config_dotted(self.config, key)
		return res

	def check_data_availability(self, data, required_fields):
		"""Check if all required data fields are available"""
		missing = []
		for field in required_fields:
			if not hasattr(data, field) or getattr(data, field) is None:
				missing.append(field)
		return missing

	def report_missing_data(self, missing_fields, missing_data_str):
		"""Update missing data string with missing fields"""
		if missing_fields:
			missing_data_str += f"{', '.join(missing_fields)} "
			return False
		return True

	def create_visualization_publisher(self, name_suffix, publisher_type=ROSLine):
		"""Create standardized visualization publisher"""
		publisher_name = f"{self.name}/{name_suffix}"
		return publisher_type(publisher_name)

	def visualize_trajectory(self, trajectory, name_suffix, scale=0.1, color_int=0):
		"""Visualize a trajectory with standard formatting"""
		publisher = self.create_visualization_publisher(name_suffix)
		line = publisher.add_new_line()
		line.set_scale(scale)
		line.set_color_int(color_int)

		for i in range(1, len(trajectory)):
			line.add_line(trajectory[i - 1], trajectory[i])

		publisher.publish()


