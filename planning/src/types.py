import os
import random
from enum import Enum

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from typing import List, Optional

from planning.src.dynamic_models import DynamicsModel
from utils.const import DETERMINISTIC, GAUSSIAN
from utils.math_utils import Halfspace
from utils.utils import LOG_DEBUG, LOG_WARN, CONFIG


class State:
    """
    State class that manages the vehicle state and integrates with the SMPC constraint system.
    Supports different vehicle models and provides a clean interface for the solver.
    """

    def __init__(self, model_type=None):
        # Internal state storage
        self._state_dict = {}  # Dictionary for name-based access--this includes both state values for the model and its control inputs
        self._state_vector = []  # Vector for numeric operations

        # Model configuration
        self._model_type = model_type
        self._config = {}

        self.initialize(model_type)

    def initialize(self, model_type: DynamicsModel):
        LOG_DEBUG("Initializing state")
        if model_type:
            self._model_type = model_type
        else:
            return
        self._state_dict = {}
        all_vars = self._model_type.get_all_vars()
        self._state_vector = np.zeros(len(all_vars))

        for i, var in enumerate(all_vars):
            self._state_dict[var] = 0.0

        LOG_DEBUG(f"Initialized state with variables: {all_vars}")

    def get(self, var_name):
        """Get a state variable by name."""
        # First check if it's in the state dictionary
        if var_name in self._state_dict:
            return self._state_dict[var_name]
        # Not found
        LOG_DEBUG(f"Variable {var_name} not found in State, returning 0.0")
        return 0.0

    def set(self, var_name, value):
        self._state_dict[var_name] = value

        # Only try to update the vector if value is numeric
        if isinstance(value, (float, int)):
            var_idx = self._get_index(var_name)
            if var_idx is not None and 0 <= var_idx < len(self._state_vector):
                self._state_vector[var_idx] = value

    def _get_index(self, var_name):
        """Get the index of a variable in the state vector."""
        # First check if it's in the state dictionary (ordered)
        if var_name in self._state_dict:
            # Find its position in the ordered keys
            keys = list(self._state_dict.keys())
            if var_name in keys:
                return keys.index(var_name)

        # Not found
        return None

    def has(self, key):
        """Check if the state has a specific variable."""
        return key in self._state_dict

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

    def get_state_dict(self):
        return self._state_dict.copy()

    def get_model_type(self):
        return self._model_type

    def __str__(self):
        """Return a readable string representation of the current state."""
        lines = [f"State: ({self._model_type}):"]
        for var, value in self._state_dict.items():
            lines.append(f" {var}: {value}")
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def copy(self):
        state = State()
        state.initialize(self._model_type)
        state._state_dict = self._state_dict.copy()
        state._config = self._config.copy()
        state._state_vector = self._state_vector.copy()
        return state


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

    def __str__(self):

        return f"PredictionStep({self.position}, {self.angle}, {self.major_radius}, {self.minor_radius})"

class Prediction:
    def __init__(self, type_=None):
        self.type = type_
        self.steps = []
        self.probabilities = []

    def empty(self) -> bool:
        return len(self.steps) == 0

class ObstacleType(Enum):
    STATIC = 0
    DYNAMIC = 1


class StaticObstacle:
    def __init__(self, position=None, angle=None, radius=None, _type=None):
        self.position = position
        self.angle = angle
        self.radius = radius
        self.type = _type if _type is not None else ObstacleType.STATIC
        self.prediction = Prediction()
        self.halfspaces = []  # List to store halfspace constraints

    def add_halfspace(self, A, b):
        """Add a halfspace constraint defined by Ax <= b"""
        self.halfspaces.append(Halfspace(A, b))

    def set(self, attribute, value):
        self.__setattr__(attribute, value)

class DynamicObstacle:
    def __init__(self, index: int, position: np.ndarray, angle: float, radius: float,
                 _type: 'ObstacleType' = ObstacleType.DYNAMIC):
        self.index = index
        self.position = position
        self.angle = angle
        self.radius = radius
        self.type = _type
        self.s = 0.0
        self.prediction = Prediction()


def generate_dynamic_obstacles(
    number: int,
    prediction_type: str,
    size: float,
    distribution: str = "random_paths",
    area: tuple = ((0, 20), (0, 20), (0, 0)),  # x, y, z ranges
    path_types=("straight", "curved", "s-turn", "circle"),
    num_points=11,
    dim: int = 2
) -> List[DynamicObstacle]:
    """
    Generate multiple dynamic obstacles, optionally in 3D, each with its own independent path.

    Args:
        number: Number of obstacles.
        prediction_type: Behavior of prediction, DETERMINISTIC, GAUSSIAN, NONGAUSSIAN
        size: Radius of each obstacle.
        distribution: 'random_paths' or 'grid'.
        area: ((xmin,xmax), (ymin,ymax), (zmin,zmax)) for randomization.
        path_types: Allowed path types (randomly chosen per obstacle).
        num_points: Points per reference path.
        dim: 2 or 3 (controls whether paths and positions include z).

    Returns:
        List[DynamicObstacle]
    """
    obstacles = []

    for i in range(number):
        # Generate start and goal positions
        if distribution == "random_paths":
            start = [
                np.random.uniform(area[0][0], area[0][1]),
                np.random.uniform(area[1][0], area[1][1]),
            ]
            goal = [
                np.random.uniform(area[0][0], area[0][1]),
                np.random.uniform(area[1][0], area[1][1]),
            ]
            if dim == 3:
                start.append(np.random.uniform(area[2][0], area[2][1]))
                goal.append(np.random.uniform(area[2][0], area[2][1]))
            else:
                start.append(0.0)
                goal.append(0.0)

        elif distribution == "grid":
            # Simple grid for 2D or 3D
            grid_size = int(np.ceil(np.sqrt(number)))
            row, col = divmod(i, grid_size)
            step_x = (area[0][1] - area[0][0]) / grid_size
            step_y = (area[1][1] - area[1][0]) / grid_size
            start = [area[0][0] + col * step_x, area[1][0] + row * step_y]
            goal = [
                np.random.uniform(area[0][0], area[0][1]),
                np.random.uniform(area[1][0], area[1][1]),
            ]
            if dim == 3:
                start.append(np.random.uniform(area[2][0], area[2][1]))
                goal.append(np.random.uniform(area[2][0], area[2][1]))
            else:
                start.append(0.0)
                goal.append(0.0)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Generate unique path for this obstacle
        path_type = np.random.choice(path_types)
        ref_path = generate_reference_path(start, goal, path_type, num_points=num_points)

        # Initial position (3D if dim == 3)
        pos = np.array([ref_path.x[0], ref_path.y[0], ref_path.z[0]]) if dim == 3 else np.array([ref_path.x[0], ref_path.y[0]])
        # Compute heading angle in XY plane
        angle = np.arctan2(ref_path.y[1] - ref_path.y[0], ref_path.x[1] - ref_path.x[0])

        # Create obstacle
        obst = DynamicObstacle(index=i, position=pos, angle=angle, radius=size)

        # Prediction behavior
        if prediction_type == DETERMINISTIC:
            obst.prediction.path = ref_path
            obst.prediction.type = PredictionType.DETERMINISTIC
            obst.prediction.steps = []
        elif prediction_type == GAUSSIAN:
            obst.prediction.path = ref_path
            obst.prediction.type = PredictionType.GAUSSIAN
            obst.prediction.steps = []

        obstacles.append(obst)

    return obstacles


class ReferencePath:
    def __init__(self):
        self.x = [] # list of x coordinates
        self.y = [] # list of y coordinates
        self.z = [] # list of z coordinates
        self.psi = [] # list of headings
        self.v = [] # list of velocities
        self.s = [] # list of arc length progress for aligning with other attributes

        self.x_spline = None # a CubicSpline with x as the dependent val
        self.y_spline = None # a CubicSpline with Y as the dependent val
        self.z_spline = None # a CubicSpline with Z as the dependent val
        self.v_spline = None # a CubicSpline with v as the dependent val

    def clear(self):
        self.x = []
        self.y = []
        self.psi = []
        self.v = []
        self.s = []
        self.x_spline = None # a CubicSpline with x as the dependent val
        self.y_spline = None # a CubicSpline with Y as the dependent val
        self.z_spline = None # a CubicSpline with Z as the dependent val
        self.v_spline = None # a CubicSpline with v as the dependent val

    def empty(self) -> bool:
        return len(self.x) == 0

    def get_velocity(self):
        return self.v

    def has_velocity(self) -> bool:
        return len(self.v) > 0

    def has_distance(self) -> bool:
        return len(self.s) > 0

    def set(self, attribute, value):
        self.__setattr__(attribute, value)

    def get_arc_length(self):
        return self.s[-1] - self.s[0]

    def get(self, attribute):
        return self.__getattribute__(attribute)

    def format_val(self, val):
        return f"{val:.3f}" if isinstance(val, (float, int)) else "   ---"

    def __str__(self):
        if self.empty():
            return "ReferencePath: [empty]"

        header = f"{'Index':>5} | {'x':>8} | {'y':>8} | {'psi':>8} | {'v':>8} | {'s':>8}"
        lines = [header, "-" * len(header)]

        length = len(self.x)
        for i in range(length):
            x = self.x[i] if i < len(self.x) else None
            y = self.y[i] if i < len(self.y) else None
            psi = self.psi[i] if i < len(self.psi) else None
            v = self.v[i] if i < len(self.v) else None
            s = self.s[i] if i < len(self.s) else None
            lines.append(
                f"{i:5d} | {self.format_val(x):>8} | {self.format_val(y):>8} | "
                f"{self.format_val(psi):>8} | {self.format_val(v):>8} | {self.format_val(s):>8}"
            )

        return "\n".join(lines)

def generate_reference_path(start, goal, path_type="curved", num_points=11) -> ReferencePath:
    t = np.linspace(0, 1, num_points)

    if path_type == "straight":
        x = np.linspace(start[0], goal[0], num_points)
        y = np.linspace(start[1], goal[1], num_points)
        z = np.linspace(start[2], goal[2], num_points) if len(start) > 2 else np.zeros(num_points)

    elif path_type == "curved":
        start_np = np.array(start[:2])
        goal_np = np.array(goal[:2])
        mid = 0.5 * (start_np + goal_np) + np.array([0.0, goal_np[1]])
        x_spline = CubicSpline([0, 0.5, 1.0], [start_np[0], mid[0], goal_np[0]])
        y_spline = CubicSpline([0, 0.5, 1.0], [start_np[1], mid[1], goal_np[1]])
        x = x_spline(t)
        y = y_spline(t)
        z = np.linspace(start[2], goal[2], num_points) if len(start) > 2 else np.zeros(num_points)


    elif path_type == "s-turn":
        x = np.linspace(start[0], goal[0], num_points)
        amplitude = 2.0
        y = np.linspace(start[1], goal[1], num_points) + amplitude * np.sin(2 * np.pi * t)
        z = np.linspace(start[2], goal[2], num_points) if len(start) > 2 else np.zeros(num_points)


    elif path_type == "circle":
        radius = 5.0
        theta = np.linspace(0, np.pi, num_points)
        x = start[0] + radius * np.cos(theta)
        y = start[1] + radius * np.sin(theta)
        z = np.linspace(start[2], goal[2], num_points) if len(start) > 2 else np.zeros(num_points)

    else:
        raise ValueError(f"Unknown path type: {path_type}")

    # Compute arc length
    s = np.zeros(len(x))
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        dz = z[i] - z[i - 1]
        s[i] = s[i - 1] + np.sqrt(dx**2 + dy**2 + dz**2)

    # Build splines
    spline_x = CubicSpline(s, x)
    spline_y = CubicSpline(s, y)
    spline_z = CubicSpline(s, z)

    # Store in ReferencePath object
    path = ReferencePath()
    path.x = x
    path.y = y
    path.z = z
    path.s = s
    path.x_spline = spline_x
    path.y_spline = spline_y
    path.z_spline = spline_z
    path.length = float(s[-1])  # ✅ Set total path length here

    return path


def calculate_path_normals_improved(reference_path):
    """
    Calculate improved normal vectors for the reference path.
    This version handles curvature better and ensures consistent orientation.
    """
    normals = []

    for i in range(len(reference_path.x)):
        if i == 0:
            # First point: use forward difference
            dx = reference_path.x[i + 1] - reference_path.x[i]
            dy = reference_path.y[i + 1] - reference_path.y[i]
        elif i == len(reference_path.x) - 1:
            # Last point: use backward difference
            dx = reference_path.x[i] - reference_path.x[i - 1]
            dy = reference_path.y[i] - reference_path.y[i - 1]
        else:
            # Middle points: use central difference for smoother results
            dx = reference_path.x[i + 1] - reference_path.x[i - 1]
            dy = reference_path.y[i + 1] - reference_path.y[i - 1]

        # Normalize the tangent vector
        norm = np.sqrt(dx ** 2 + dy ** 2)
        if norm > 1e-6:
            dx_norm = dx / norm
            dy_norm = dy / norm
        else:
            dx_norm = 1.0
            dy_norm = 0.0

        # Normal vector (perpendicular to tangent, pointing left)
        # Rotate tangent 90 degrees counterclockwise
        normal_x = -dy_norm
        normal_y = dx_norm

        normals.append((normal_x, normal_y))

    return normals


def generate_road_boundaries_improved(reference_path, road_width):
    """
    Generate road boundaries with improved normal calculation and
    curvature compensation.
    """
    # Calculate improved normals
    normals = calculate_path_normals_improved(reference_path)

    # Use a smaller offset for curved sections
    half_width = road_width / 2.0

    # Adjust width based on curvature to prevent crossing
    adjusted_half_width = []

    for i in range(len(reference_path.x)):
        # Calculate local curvature (simplified)
        if i > 0 and i < len(reference_path.x) - 1:
            # Use three points to estimate curvature
            p1 = np.array([reference_path.x[i - 1], reference_path.y[i - 1]])
            p2 = np.array([reference_path.x[i], reference_path.y[i]])
            p3 = np.array([reference_path.x[i + 1], reference_path.y[i + 1]])

            # Vectors
            v1 = p2 - p1
            v2 = p3 - p2

            # Estimate curvature using cross product
            cross_prod = np.cross(v1, v2)
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 1e-6 and v2_norm > 1e-6:
                curvature = abs(cross_prod) / (v1_norm * v2_norm)
                # Reduce width in high curvature areas
                width_factor = max(0.3, 1.0 - curvature * 2.0)
                adjusted_half_width.append(half_width * width_factor)
            else:
                adjusted_half_width.append(half_width)
        else:
            adjusted_half_width.append(half_width)

    # Generate boundaries
    left_x = []
    left_y = []
    right_x = []
    right_y = []

    for i in range(len(reference_path.x)):
        nx, ny = normals[i]
        width = adjusted_half_width[i]

        # Left boundary (offset in the positive normal direction)
        left_x.append(reference_path.x[i] + nx * width)
        left_y.append(reference_path.y[i] + ny * width)

        # Right boundary (offset in the negative normal direction)
        right_x.append(reference_path.x[i] - nx * width)
        right_y.append(reference_path.y[i] - ny * width)

    return left_x, left_y, right_x, right_y


def smooth_boundaries(x_coords, y_coords, s_coords, smoothing_factor=0.1):
    """
    Apply smoothing to boundary coordinates to reduce sharp edges.
    """
    from scipy.ndimage import gaussian_filter1d

    # Apply Gaussian smoothing
    x_smooth = gaussian_filter1d(x_coords, sigma=smoothing_factor * len(x_coords))
    y_smooth = gaussian_filter1d(y_coords, sigma=smoothing_factor * len(y_coords))

    return x_smooth, y_smooth


# Modified test code section
def create_improved_boundaries(data, reference_path):
    """
    Replace the boundary generation section in your test code with this.
    """
    # Road width
    road_width = data.road_width if hasattr(data, 'road_width') and data.road_width is not None else 4.0

    # Generate improved boundaries
    left_x, left_y, right_x, right_y = generate_road_boundaries_improved(reference_path, road_width)

    # Optional: Apply smoothing
    left_x, left_y = smooth_boundaries(left_x, left_y, reference_path.s)
    right_x, right_y = smooth_boundaries(right_x, right_y, reference_path.s)

    # Create splines for the boundaries
    left_boundary_spline_x = CubicSpline(reference_path.s, np.array(left_x))
    left_boundary_spline_y = CubicSpline(reference_path.s, np.array(left_y))
    right_boundary_spline_x = CubicSpline(reference_path.s, np.array(right_x))
    right_boundary_spline_y = CubicSpline(reference_path.s, np.array(right_y))

    # Store boundary data
    data.left_boundary_x = left_x
    data.left_boundary_y = left_y
    data.right_boundary_x = right_x
    data.right_boundary_y = right_y

    # Store the spline functions
    data.left_spline_x = left_boundary_spline_x
    data.left_spline_y = left_boundary_spline_y
    data.right_spline_x = right_boundary_spline_x
    data.right_spline_y = right_boundary_spline_y

    # Create Bound objects
    data.left_bound = Bound(left_x, left_y, reference_path.s)
    data.right_bound = Bound(right_x, right_y, reference_path.s)

    return data


# Alternative: Use spline-based normal calculation
def calculate_spline_normals(reference_path):
    """
    Calculate normals using spline derivatives for smoother results.
    """
    if not hasattr(reference_path, 'x_spline') or not hasattr(reference_path, 'y_spline'):
        # Create splines if they don't exist
        reference_path.x_spline = CubicSpline(reference_path.s, reference_path.x)
        reference_path.y_spline = CubicSpline(reference_path.s, reference_path.y)

    normals = []

    for i, s_val in enumerate(reference_path.s):
        # Get derivatives from splines
        dx = reference_path.x_spline.derivative()(s_val)
        dy = reference_path.y_spline.derivative()(s_val)

        # Normalize
        norm = np.sqrt(dx ** 2 + dy ** 2)
        if norm > 1e-6:
            dx_norm = dx / norm
            dy_norm = dy / norm
        else:
            dx_norm = 1.0
            dy_norm = 0.0

        # Normal vector (perpendicular to tangent, pointing left)
        normal_x = -dy_norm
        normal_y = dx_norm

        normals.append((normal_x, normal_y))

    return normals

def calculate_path_normals(reference_path: ReferencePath):
    normals = []
    for i in range(len(reference_path.x)):
        if i == 0:
            # For the first point, use the direction to the next point
            dx = reference_path.x[1] - reference_path.x[0]
            dy = reference_path.y[1] - reference_path.y[0]
        elif i == len(reference_path.x) - 1:
            # For the last point, use the direction from the previous point
            dx = reference_path.x[-1] - reference_path.x[-2]
            dy = reference_path.y[-1] - reference_path.y[-2]
        else:
            # For middle points, use the average of adjacent segments
            dx1 = reference_path.x[i] - reference_path.x[i - 1]
            dy1 = reference_path.y[i] - reference_path.y[i - 1]
            dx2 = reference_path.x[i + 1] - reference_path.x[i]
            dy2 = reference_path.y[i + 1] - reference_path.y[i]
            dx = dx1 + dx2
            dy = dy1 + dy2

        # Calculate the normal vector (perpendicular to the path direction)
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length > 0:
            nx = -dy / length
            ny = dx / length
            normals.append((nx, ny))
        else:
            # Avoid division by zero
            normals.append((0, 0))

    return normals


class Bound:
    def __init__(self, x, y, s):
        self.x = x
        self.y = y
        self.s = s


def generate_static_obstacles(
    number: int,
    size: float,
    reference_path: Optional[ReferencePath] = None,
    area: tuple = ((0, 20), (0, 20), (0, 0)),  # (x_range, y_range, z_range)
    lateral_offset_range=(-2.0, 2.0),
    along_path_ratio=1.0
) -> List[StaticObstacle]:
    """
    Generate static obstacles either randomly or along a provided reference path.

    Args:
        number: Number of obstacles to generate.
        size: Radius of each obstacle.
        reference_path: If provided, obstacles will be placed near this path.
        area: ((xmin,xmax), (ymin,ymax), (zmin,zmax)) for randomization.
        lateral_offset_range: Range for lateral deviation from the path.
        along_path_ratio: Fraction of obstacles to place along the path (0.0 - 1.0).

    Returns:
        List[StaticObstacle]
    """
    obstacles = []

    for i in range(number):
        if reference_path and np.random.rand() < along_path_ratio:
            # Place obstacle near reference path
            s_offset = np.random.uniform(0, reference_path.s[-1])  # choose random point along path
            x_center = reference_path.x_spline(s_offset)
            y_center = reference_path.y_spline(s_offset)

            # Compute tangent and normal for lateral offset
            ds = 0.1
            s_next = min(s_offset + ds, reference_path.s[-1])
            dx = reference_path.x_spline(s_next) - x_center
            dy = reference_path.y_spline(s_next) - y_center
            tangent = np.array([dx, dy]) / (np.linalg.norm([dx, dy]) + 1e-6)
            normal = np.array([-tangent[1], tangent[0]])

            lateral_offset = np.random.uniform(*lateral_offset_range)
            position = np.array([
                x_center + lateral_offset * normal[0],
                y_center + lateral_offset * normal[1],
                0.0  # z fixed to 0 for 2D case
            ])
        else:
            # Place randomly in the given area
            x = np.random.uniform(area[0][0], area[0][1])
            y = np.random.uniform(area[1][0], area[1][1])
            z = np.random.uniform(area[2][0], area[2][1])
            position = np.array([x, y, z])

        # Orientation for static obstacles can be random or zero
        angle = np.random.uniform(-np.pi, np.pi)

        # Create the obstacle
        obst = StaticObstacle(position=position, angle=angle, radius=size)
        obstacles.append(obst)

    return obstacles

class Trajectory:
    def __init__(self, timestep: float = 0.1, length: int = 30): # defaults match timestep and horizon for solver
        self.timestep = timestep
        self.states = []

    def add_state(self, state: State):
        if state is None:
            return
        else:
            self.states.append(state)

    def get_states(self):
        return self.states

    def get_element_history_by_name(self, name):
        history = []
        for state in self.states:
            history.append(state.get(name))
        return history

    def reset(self):
        self.states = []

    def __str__(self):
        """Return a readable string representation of the current state."""
        lines = [f"Trajectory object contents: ({self.states}) --> "]
        k = 0
        for state in self.states:
            lines.append(f"  {state}: at {k:.3f}")
            k += 1
        return "\n".join(lines)


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
        else:
            self.__setattr__(key, None)

    def __setattr__(self, key, value):
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self._store[key] = value

    def __str__(self):
        """Return a readable string representation of the current state."""
        lines = [f"Data object contents: ({self._store}) --> "]
        for var, value in self._store.items():
            if isinstance(value, (float, int)):
                lines.append(f"  {var}: {value:.3f}")
            elif isinstance(value, np.ndarray):
                lines.append(f"  {var}: {np.array2string(value, precision=3, separator=', ')}")
            else:
                lines.append(f"  {var}: {value}")
        return "\n".join(lines)

class Costmap:
    def __init__(self, width, height, resolution, origin, obstacle_threshold=100):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = np.array(origin)
        self.obstacle_threshold = obstacle_threshold
        self.data = np.zeros((height, width), dtype=np.uint8)

    def set_obstacles(self, obstacle_coords):
        for x, y in obstacle_coords:
            i, j = self.world_to_map(x, y)
            LOG_DEBUG(f"For obstacle with coordinates x: {x}, y: {y}, calculated i: {i}, j: {j}")
            if 0 <= i < self.width and 0 <= j < self.height:
                self.data[j, i] = 255  # Mark as obstacle

    def inflate_obstacles(self, inflation_radius):
        inflation_cells = int(inflation_radius / self.resolution)
        inflated = self.data.copy()
        obstacle_indices = np.argwhere(self.data >= self.obstacle_threshold)
        for j, i in obstacle_indices:
            min_i = max(0, i - inflation_cells)
            max_i = min(self.width, i + inflation_cells)
            min_j = max(0, j - inflation_cells)
            max_j = min(self.height, j + inflation_cells)
            inflated[min_j:max_j, min_i:max_i] = 255
        self.data = inflated

    def get_size_in_cells_x(self):
        return self.width

    def get_size_in_cells_y(self):
        return self.height

    def getCost(self, i, j):
        return self.data[j, i]

    def map_to_world(self, i, j):
        x = self.origin[0] + i * self.resolution
        y = self.origin[1] + j * self.resolution
        return x, y

    def world_to_map(self, x, y):
        i = int((x - self.origin[0]) / self.resolution)
        j = int((y - self.origin[1]) / self.resolution)
        return i, j



