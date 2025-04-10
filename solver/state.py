import os
import yaml
import numpy as np
from utils.utils import read_config_file, LOG_DEBUG


class State:
	"""
  State class that manages the vehicle state and integrates with the SMPC constraint system.
  """

	def __init__(self):
		# Load configuration
		self._config = {}
		self._model_map = {}
		self._state = None
		self._nu = 0

		# Initialize empty state
		self.initialize()

		# Load configuration files
		file_dir = os.path.dirname(os.path.abspath(__file__))
		self._load_config_yaml(os.path.join(file_dir, "config/solver_settings.yaml"), self._config)
		self._load_config_yaml(os.path.join(file_dir, "config/model_map.yaml"), self._model_map)

		# Re-initialize with proper dimensions after loading config
		self.initialize()

	def _load_config_yaml(self, path, target_dict):
		"""Load YAML configuration file into the target dictionary."""
		try:
			with open(path, 'r') as f:
				data = yaml.safe_load(f)
				target_dict.update(data)
		except Exception as e:
			LOG_DEBUG(f"Error loading config from {path}: {e}")

	def initialize(self):
		"""Initialize the state vector."""
		if self._config and "nx" in self._config:
			nx = self._config["nx"]
			self._state = np.zeros(nx)
			if "nu" in self._config:
				self._nu = self._config["nu"]
		else:
			# Default initialization if config is not yet loaded
			self._state = np.zeros(10)  # Default state size
			self._nu = 2  # Default control input size

	def get(self, var_name):
		"""Get a state variable by name."""
		if var_name not in self._model_map:
			LOG_DEBUG(f"Variable {var_name} not found in model map")
			return 0.0

		var_index = self._model_map[var_name][1]
		state_index = var_index - self._nu  # States come after inputs

		if 0 <= state_index < len(self._state):
			return self._state[state_index]
		else:
			LOG_DEBUG(f"State index {state_index} out of bounds")
			return 0.0

	def getPos(self):
		"""Get the position as a tuple (x, y)."""
		return (self.get("x"), self.get("y"))

	def set(self, var_name, value):
		"""Set a state variable by name."""
		if var_name not in self._model_map:
			LOG_DEBUG(f"Variable {var_name} not found in model map")
			return

		var_index = self._model_map[var_name][1]
		state_index = var_index - self._nu

		if 0 <= state_index < len(self._state):
			self._state[state_index] = value
		else:
			LOG_DEBUG(f"State index {state_index} out of bounds")

	def print(self):
		"""Print the current state values."""
		LOG_DEBUG("Current state:")
		for var_name, info in self._model_map.items():
			if info[0] == "x":  # Assuming "x" indicates a state variable
				try:
					value = self.get(var_name)
					LOG_DEBUG(f"{var_name}: {value}")
				except Exception as e:
					LOG_DEBUG(f"Error printing {var_name}: {e}")

	def to_dict(self):
		"""Convert state to dictionary for debugging."""
		result = {}
		for var_name, info in self._model_map.items():
			if info[0] == "x":
				result[var_name] = self.get(var_name)
		return result

	def get_full_state(self):
		"""Return the full state vector."""
		return self._state.copy()

	def set_full_state(self, state_vector):
		"""Set the full state vector."""
		if len(state_vector) == len(self._state):
			self._state = np.array(state_vector)
		else:
			LOG_DEBUG(f"Error: State vector length mismatch. Expected {len(self._state)}, got {len(state_vector)}")