import numpy as np
import casadi as cd
from scipy.interpolate import CubicSpline

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planning.src.types import Data
from utils.math_utils import Spline2DAdapter, SplineAdapter, haar_difference_without_abs, CasadiSpline2D, CasadiSpline
from utils.utils import LOG_DEBUG


class ContouringConstraints(BaseConstraint):
    def __init__(self, solver):
        super().__init__(solver)
        self.width_left = None
        self.width_right = None
        self.nh = 2
        self.name = "contouring_constraints"  # Override default name if needed
        self.num_segments = self.get_config_value("contouring.num_segments")
        LOG_DEBUG(f"{self.name.title()} Constraints successfully initialized")

    def update(self, state, data: Data):
        if data.get("path_width_left") is None and self.width_left is not None:
            data.set("path_width_left", self.width_left)

        if data.get("path_width_right") is None and self.width_right is not None:
            data.set("path_width_right", self.width_right)

    def on_data_received(self, data, data_name):
        if data_name == "reference_path":
            self.process_reference_path(data)

    def process_reference_path(self, data):
        LOG_DEBUG("Reference Path Received")

        if not data.left_bound is None and not data.right_bound is None:
            LOG_DEBUG("Received Road Boundaries")
            self.calculate_road_widths(data)

    def calculate_road_widths(self, data):
        widths_left = np.zeros(len(data.left_bound.x))
        widths_right = np.zeros(len(data.right_bound.x))

        for i in range(len(widths_left)):
            center = np.array([data.reference_path.x[i], data.reference_path.y[i]])
            left = np.array([data.left_bound.x[i], data.left_bound.y[i]])
            right = np.array([data.right_bound.x[i], data.right_bound.y[i]])

            widths_left[i] = np.linalg.norm(center - left)
            widths_right[i] = np.linalg.norm(center - right)

        # Initialize splines
        self.width_left = CubicSpline(data.reference_path.s, widths_left)
        self.width_right = CubicSpline(data.reference_path.s, widths_right)

        # Add custom methods to match C++ implementation
        self.width_left.m_x_ = data.reference_path.s
        self.width_right.m_x_ = data.reference_path.s

        # Add method to get spline parameters
        self._add_parameter_methods_to_splines()

    def _add_parameter_methods_to_splines(self):
        def get_parameters(spline, index, a, b, c, d):
            # Extract cubic spline parameters for the segment
            if index < len(spline.c[0]) - 1:
                a = spline.c[0][index]
                b = spline.c[1][index]
                c = spline.c[2][index]
                d = spline.c[3][index]
            return a, b, c, d

        # Attach method to the splines
        self.width_left.get_parameters = get_parameters.__get__(self.width_left)
        self.width_right.get_parameters = get_parameters.__get__(self.width_right)

    def set_parameters(self, parameter_manager, data, k):
        if k == 1:
            LOG_DEBUG(f"{self.name}::set_parameters")

        for segment_index in range(self.num_segments):
            self.set_boundary_parameters(parameter_manager, segment_index)

    def define_parameters(self, params):
        print("Defining contouring parameters")
        for segment_index in range(self.num_segments):
            params.add(f"width_right_{segment_index}_a")
            params.add(f"width_right_{segment_index}_b")
            params.add(f"width_right_{segment_index}_c")
            params.add(f"width_right_{segment_index}_d")

            params.add(f"width_left_{segment_index}_a")
            params.add(f"width_left_{segment_index}_b")
            params.add(f"width_left_{segment_index}_c")
            params.add(f"width_left_{segment_index}_d")

    def get_lower_bound(self):
        lower_bound = [-np.inf, -np.inf]
        return lower_bound

    def get_upper_bound(self):
        upper_bound = [0., 0.]
        return upper_bound

    def get_constraints(self, model, params, settings, stage_idx):
        print("Trying to get constraints in contouring constraints")
        constraints = []
        pos_x = model.get("x")
        pos_y = model.get("y")
        s = model.get("spline")

        # Create adapters
        spline = CasadiSpline2D(params, self.num_segments)
        path_x, path_y = spline.at(s)

        # Debug: Sample a few points to verify spline correctness
        if stage_idx == 0:  # Only log for first stage to avoid flooding logs
            LOG_DEBUG("Sampling spline points:")
            for test_s in [0.0, 0.5, 1.0]:  # Sample at different s values
                test_xy = spline.at(test_s)
                LOG_DEBUG(f"  spline={test_s}: x={test_xy[0]}, y={test_xy[1]}")

        try:
            slack = model.get("slack")
        except:
            slack = 0.0

        try:
            psi = model.get("psi")
        except:
            psi = 0.0

        path_dx_normalized, path_dy_normalized = spline.deriv_normalized(s)

        contour_error = path_dy_normalized * (pos_x - path_x) - path_dx_normalized * (pos_y - path_y)

        width_left = CasadiSpline(params, "width_left", self.num_segments)
        width_right = CasadiSpline(params, "width_right", self.num_segments)

        # Accurate width of the vehicle incorporating its orientation w.r.t. the path
        delta_psi = haar_difference_without_abs(psi, cd.atan2(path_dy_normalized, path_dx_normalized)) # Angle w.r.t. the path
        w_cur = model.width / 2. * cd.cos(delta_psi) + model.lr * cd.sin(cd.fabs(delta_psi))

        # Simpler
        # w_cur = model.width / 2.

        # Forces does not support bounds that depend on the parameters. Two constraints are needed.
        constraints.append(contour_error + w_cur - width_right.at(s) - slack)
        constraints.append(-contour_error + w_cur - width_left.at(s) - slack)  # -width_left because widths are positive

        # constraints.append(cd.fabs(delta_psi) - 0.35 * np.pi)

        return constraints

    def set_boundary_parameters(self, parameter_manager, segment_index):
        # Initialize parameter values
        ra, rb, rc, rd = 0., 0., 0., 0.
        la, lb, lc, ld = 0., 0., 0., 0.

        # Get appropriate parameter values
        if segment_index < len(self.width_right.m_x_) - 1:
            ra, rb, rc, rd = self.width_right.get_parameters(segment_index, ra, rb, rc, rd)
            la, lb, lc, ld = self.width_left.get_parameters(segment_index, la, lb, lc, ld)
        else:
            # Handle edge case for last segment
            ra, rb, rc, rd = self.width_right.get_parameters(len(self.width_right.m_x_) - 1, ra, rb, rc, rd)
            la, lb, lc, ld = self.width_left.get_parameters(len(self.width_left.m_x_) - 1, la, lb, lc, ld)

            # Zero out certain parameters for last segment
            ra = rb = rc = 0.
            la = lb = lc = 0.

        # Set right boundary parameters
        parameter_manager.set_parameter(f"width_right_{segment_index}_a", ra)
        parameter_manager.set_parameter(f"width_right_{segment_index}_b", rb)
        parameter_manager.set_parameter(f"width_right_{segment_index}_c", rc)
        parameter_manager.set_parameter(f"width_right_{segment_index}_d", rd)

        # Set left boundary parameters
        parameter_manager.set_parameter(f"width_left_{segment_index}_a", la)
        parameter_manager.set_parameter(f"width_left_{segment_index}_b", lb)
        parameter_manager.set_parameter(f"width_left_{segment_index}_c", lc)
        parameter_manager.set_parameter(f"width_left_{segment_index}_d", ld)

    def is_data_ready(self, data):
        required_fields = ["left_bound", "right_bound"]
        missing_data = ""
        for field in required_fields:
            if not data.has(field):
                missing_data += f"{field.replace('_', ' ').title()} "
        return len(missing_data) < 1

    def visualize(self, data):
        # Use the parent class method to check debug_visuals setting
        super().visualize(data)

        # Additional check for required data
        if self.width_right is None or self.width_left is None or data.path is None:
            return
