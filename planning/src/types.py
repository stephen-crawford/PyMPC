import os
from enum import Enum

import numpy as np

from planning.src.dynamic_models import DynamicsModel
from utils.utils import LOG_DEBUG, load_yaml

import os
import numpy as np
from utils.utils import LOG_DEBUG, LOG_WARN


class State:
    """
    State class that manages the vehicle state and integrates with the SMPC constraint system.
    Supports different vehicle models and provides a clean interface for the solver.
    """

    def __init__(self, model_type=None):
        # Internal state storage
        self._state_dict = {}  # Dictionary for name-based access
        self._state_vector = None  # Vector for numeric operations

        # Model configuration
        self._model_type = model_type
        self._model_map = {}  # Maps variable names to indices
        self._config = {}

        # Load configuration files
        self._load_config()

        # Initialize state based on model type or default
        self.initialize(model_type)

    def _load_config(self):
        """Load configuration files for solver settings and model mappings."""
        try:
            file_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(file_dir, "config/solver_settings.yaml")
            model_map_path = os.path.join(file_dir, "config/model_map.yaml")

            # Implement your YAML loader here or use a library
            from utils.utils import load_yaml
            load_yaml(config_path, self._config)
            load_yaml(model_map_path, self._model_map)

            LOG_DEBUG(f"Loaded configuration with {len(self._config)} settings")
            LOG_DEBUG(f"Loaded model map with {len(self._model_map)} variables")
        except Exception as e:
            LOG_WARN(f"Error loading configuration: {e}")
            # Set default configuration
            self._config = {"nx": 6, "nu": 2}

    def initialize(self, model_type: DynamicsModel):
        """Initialize the state vector based on model type."""
        if model_type:
            self._model_type = model_type

        # Clear current state
        self._state_dict = {}
        state_vars = self._model_type.get_vars()
        self._state_vector = np.zeros(len(model_type.get_vars()))

        # Initialize state dictionary
        for i, var in enumerate(state_vars):
            self._state_dict[var] = 0.0

        LOG_DEBUG(f"Initialized state with variables: {state_vars}")


    def set_from_dynamics_model(self, dynamics_model):
        """Configure state based on a dynamics model."""
        if not dynamics_model:
            LOG_WARN("No dynamics model provided to set state from")
            return

        # Extract state variables from dynamics model
        state_vars = dynamics_model.states

        # Reset state vector and dictionary
        self._state_vector = np.zeros(len(state_vars))
        self._state_dict = {var: 0.0 for var in state_vars}

        # Update model type based on dynamics model class name
        self._model_type = dynamics_model.__class__.__name__

        LOG_DEBUG(f"Set state from dynamics model: {self._model_type} with {len(state_vars)} variables")

    def get(self, var_name):
        """Get a state variable by name."""
        # First check if it's in the state dictionary
        if var_name in self._state_dict:
            return self._state_dict[var_name]

        # Then check if it's in the model map
        if var_name in self._model_map:
            var_info = self._model_map[var_name]
            if isinstance(var_info, (list, tuple)) and len(var_info) >= 2:
                var_type, idx = var_info[0], var_info[1]
                if var_type == "x" and isinstance(idx, int):
                    if 0 <= idx < len(self._state_vector):
                        return self._state_vector[idx]

        # Not found
        LOG_DEBUG(f"Variable {var_name} not found in state")
        return 0.0

    def set(self, var_name, value):
        """Set a state variable by name."""
        # Always update the dictionary
        self._state_dict[var_name] = value

        # If it's a recognized state variable, update the vector too
        var_idx = self._get_index(var_name)
        if var_idx is not None:
            if 0 <= var_idx < len(self._state_vector):
                self._state_vector[var_idx] = value
            else:
                LOG_DEBUG(f"State index {var_idx} for {var_name} out of bounds")

    def _get_index(self, var_name):
        """Get the index of a variable in the state vector."""
        # First check if it's in the state dictionary (ordered)
        if var_name in self._state_dict:
            # Find its position in the ordered keys
            keys = list(self._state_dict.keys())
            if var_name in keys:
                return keys.index(var_name)

        # Then check if it's in the model map
        if var_name in self._model_map:
            var_info = self._model_map[var_name]
            if isinstance(var_info, (list, tuple)) and len(var_info) >= 2:
                if var_info[0] == "x":  # Check if it's a state variable
                    return var_info[1]

        # Not found
        return None

    def has(self, key):
        """Check if the state has a specific variable."""
        return key in self._state_dict or key in self._model_map

    def get_position(self):
        """Get the position as a tuple (x, y)."""
        return self.get("x"), self.get("y")

    def get_full_state(self):
        """Return the full state vector."""
        return self._state_vector.copy()

    def set_full_state(self, state_vector):
        """Set the full state vector."""
        if len(state_vector) == len(self._state_vector):
            self._state_vector = np.array(state_vector)

            # Also update the dictionary
            state_vars = list(self._state_dict.keys())
            for i, var in enumerate(state_vars):
                if i < len(state_vector):
                    self._state_dict[var] = state_vector[i]
        else:
            LOG_DEBUG(
                f"Error: State vector length mismatch. Expected {len(self._state_vector)}, got {len(state_vector)}")

    def update_from_dict(self, state_dict):
        """Update state from a dictionary."""
        for var_name, value in state_dict.items():
            self.set(var_name, value)

    def to_dict(self):
        """Convert state to dictionary for debugging."""
        return self._state_dict.copy()

    def reset(self):
        """Reset the state to zeros."""
        for key in self._state_dict:
            self._state_dict[key] = 0.0
        self._state_vector = np.zeros_like(self._state_vector)

    def print(self):
        """Print the current state values."""
        print(f"Current state ({self._model_type}):")
        for var, value in self._state_dict.items():
            print(f"  {var}: {value}")

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

