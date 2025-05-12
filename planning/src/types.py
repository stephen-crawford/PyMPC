import os
from enum import Enum

import numpy as np

from utils.utils import LOG_DEBUG, load_yaml


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
        load_yaml(os.path.join(file_dir, "config/solver_settings.yaml"), self._config)
        load_yaml(os.path.join(file_dir, "config/model_map.yaml"), self._model_map)

        # Re-initialize with proper dimensions after loading config
        self.initialize()

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

    def get_position(self):
        """Get the position as a tuple (x, y)."""
        return self.get("x"), self.get("y")

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

class Disc:
    def __init__(self, offset: float, radius: float):
        self.offset = offset
        self.radius = radius

    def get_position(self, robot_position: np.ndarray, angle: float) -> np.ndarray:
        """Returns the disc's position relative to the robot's position and orientation angle."""
        return robot_position + np.array([
            self.offset * np.cos(angle),
            self.offset * np.sin(angle)
        ])

    def to_robot_center(self, disc_position: np.ndarray, angle: float) -> np.ndarray:
        """Returns the robot's center position from the disc's position and orientation angle."""
        return disc_position - np.array([
            self.offset * np.cos(angle),
            self.offset * np.sin(angle)
        ])

class PredictionType(Enum):
    DETERMINISTIC = 0
    GAUSSIAN = 1
    NONGAUSSIAN = 2
    NONE = 3

class PredictionStep:
    def __init__(self, position: np.ndarray, angle: float, major_radius: float, minor_radius: float):
        # Mean
        self.position = position
        self.angle = angle
        # Covariance
        self.major_radius = major_radius
        self.minor_radius = minor_radius

# Type definition for Mode
# Mode = List[PredictionStep]

class Prediction:
    def __init__(self, type_=None):
        self.type = type_
        self.modes = []
        self.probabilities = []

    def empty(self) -> bool:
        return len(self.modes) == 0

class ObstacleType(Enum):
    STATIC = 0
    DYNAMIC = 1

class DynamicObstacle:
    def __init__(self, index: int, position: np.ndarray, angle: float, radius: float,
                 _type: 'ObstacleType' = ObstacleType.DYNAMIC):
        self.index = index
        self.position = position
        self.angle = angle
        self.radius = radius
        self.type = _type
        self.prediction = Prediction()

class ReferencePath:
    def __init__(self, length: int = 10):
        self.x = []
        self.y = []
        self.psi = []
        self.v = []
        self.s = []

    def clear(self):
        self.x = []
        self.y = []
        self.psi = []
        self.v = []
        self.s = []

    def point_in_path(self, point_num: int, other_x: float, other_y: float) -> bool:
        # Implementation would check if point is in path
        # This is a placeholder for the actual implementation
        return False  # Replace with actual implementation

    def empty(self) -> bool:
        return len(self.x) == 0

    def has_velocity(self) -> bool:
        return len(self.v) > 0

    def has_distance(self) -> bool:
        return len(self.s) > 0

# Type definition for Boundary
# Boundary = ReferencePath

class Trajectory:
    def __init__(self, dt: float = 0.0, length: int = 10):
        self.dt = dt
        self.positions = []

    def add(self, p_or_x, y=None):
        if y is None:
            # p is an ndarray
            p = p_or_x
            self.positions.append(p.copy())
        else:
            # p_or_x is x, y is y
            self.positions.append(np.array([p_or_x, y]))

class FixedSizeTrajectory:
    def __init__(self, size: int = 50):
        self._size = size
        self.positions = []

    def add(self, p: np.ndarray):
        self.positions.append(p.copy())
        if len(self.positions) > self._size:
            self.positions.pop(0)

class Data:
    def __init__(self):
        self._store = {}

    def empty(self):
        return len(self._store) == 0

    def add(self, key, value):
        if key in self._store:
            self._store[key].append(value)
        else:
            self._store[key] = [value]

    def has(self, key):
        return key in self._store

    def set(self, key, value):
        self._store[key] = value

    def get(self, key):
        return self._store.get(key, None)

    def remove(self, key):
        if key in self._store:
            del self._store[key]

    def reset(self):
        self._store.clear()

    def __getattr__(self, key):
        if key in self._store:
            return self._store[key]
        raise AttributeError(f"'Data' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self._store[key] = value

