import numpy as np
from scipy.interpolate import CubicSpline

from planner.src.types import TwoDimensionalSpline, Spline, Spline2DAdapter, SplineAdapter
from planner_modules.base_constraint import BaseConstraint
from utils.visualizer import ROSLine, ROSPointMarker
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

    def update(self, state, data, module_data):
        if module_data.path_width_left is None and self.width_left is not None:
            module_data.path_width_left = self.width_left

        if module_data.path_width_right is None and self.width_right is not None:
            module_data.path_width_right = self.width_right

    def on_data_received(self, data, data_name):
        if data_name == "reference_path":
            self._process_reference_path(data)

    def _process_reference_path(self, data):
        LOG_DEBUG("Reference Path Received")

        if not data.left_bound.empty() and not data.right_bound.empty():
            LOG_DEBUG("Received Road Boundaries")
            self._calculate_road_widths(data)

    def _calculate_road_widths(self, data):
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

    def set_parameters(self, data, module_data, k):
        if k == 1:
            LOG_DEBUG(f"{self.name}::set_parameters")

        for i in range(self.num_segments):
            index = module_data.current_path_segment + i
            self._set_boundary_parameters(k, index, i)

        if k == 1:
            LOG_DEBUG(f"{self.name}.set_parameters Done")

    def define_parameters(self, params):
        for i in range(self.num_segments):
            params.add(f"width_right{i}_a", bundle_name="width_right_a")
            params.add(f"width_right{i}_b", bundle_name="width_right_b")
            params.add(f"width_right{i}_c", bundle_name="width_right_c")
            params.add(f"width_right{i}_d", bundle_name="width_right_d")

            params.add(f"width_left{i}_a", bundle_name="width_left_a")
            params.add(f"width_left{i}_b", bundle_name="width_left_b")
            params.add(f"width_left{i}_c", bundle_name="width_left_c")
            params.add(f"width_left{i}_d", bundle_name="width_left_d")

    def get_lower_bound(self):
        lower_bound = []
        lower_bound.append(-np.inf)
        lower_bound.append(-np.inf)
        # lower_bound.append(-np.inf)
        return lower_bound

    def get_upper_bound(self):
        upper_bound = []
        upper_bound.append(0.)
        upper_bound.append(0.)
        # upper_bound.append(0.)
        return upper_bound

    def get_constraints(self, model, params, settings, stage_idx):
        constraints = []
        pos_x = model.get("x")
        pos_y = model.get("y")
        s = model.get("spline")

        try:
            slack = model.get("slack")
        except:
            slack = 0.0

        try:
            psi = model.get("psi")
        except:
            psi = 0.0

        spline = Spline2DAdapter(params, self.num_segments, s)
        path_x, path_y = spline.at(s)
        path_dx_normalized, path_dy_normalized = spline.deriv_normalized(s)

        contour_error = path_dy_normalized * (pos_x - path_x) - path_dx_normalized * (pos_y - path_y)

        width_left = SplineAdapter(params, "width_left", self.num_segments, s)
        width_right = SplineAdapter(params, "width_right", self.num_segments, s)

        # Accurate width of the vehicle incorporating its orientation w.r.t. the path
        # delta_psi = haar_difference_without_abs(psi, cd.atan2(path_dy_normalized, path_dx_normalized)) # Angle w.r.t. the path
        # w_cur = model.width / 2. * cd.cos(delta_psi) + model.lr * cd.sin(cd.fabs(delta_psi))

        # Simpler
        w_cur = model.width / 2.

        # Forces does not support bounds that depend on the parameters. Two constraints are needed.
        constraints.append(contour_error + w_cur - width_right.at(s) - slack)
        constraints.append(-contour_error + w_cur - width_left.at(s) - slack)  # -width_left because widths are positive

        # constraints.append(cd.fabs(delta_psi) - 0.35 * np.pi)

        return constraints

    def _set_boundary_parameters(self, k, index, i):
        # Initialize parameter values
        ra, rb, rc, rd = 0., 0., 0., 0.
        la, lb, lc, ld = 0., 0., 0., 0.

        # Get appropriate parameter values
        if index < len(self.width_right.m_x_) - 1:
            ra, rb, rc, rd = self.width_right.get_parameters(index, ra, rb, rc, rd)
            la, lb, lc, ld = self.width_left.get_parameters(index, la, lb, lc, ld)
        else:
            # Handle edge case for last segment
            ra, rb, rc, rd = self.width_right.get_parameters(len(self.width_right.m_x_) - 1, ra, rb, rc, rd)
            la, lb, lc, ld = self.width_left.get_parameters(len(self.width_left.m_x_) - 1, la, lb, lc, ld)

            # Zero out certain parameters for last segment
            ra = rb = rc = 0.
            la = lb = lc = 0.

        # Set right boundary parameters
        self.set_solver_parameter("width_right_a", ra, k, index=i)
        self.set_solver_parameter("width_right_b", rb, k, index=i)
        self.set_solver_parameter("width_right_c", rc, k, index=i)
        self.set_solver_parameter("width_right_d", rd, k, index=i)

        # Set left boundary parameters
        self.set_solver_parameter("width_left_a", la, k, index=i)
        self.set_solver_parameter("width_left_b", lb, k, index=i)
        self.set_solver_parameter("width_left_c", lc, k, index=i)
        self.set_solver_parameter("width_left_d", ld, k, index=i)

    def is_data_ready(self, data, missing_data):
        required_fields = ["left_bound", "right_bound"]
        for field in required_fields:
            if not hasattr(data, field) or getattr(data, field).empty():
                missing_data += f"{field.replace('_', ' ').title()} "
                return False
        return True

    def visualize(self, data, module_data):
        # Use the parent class method to check debug_visuals setting
        super().visualize(data, module_data)

        # Additional check for required data
        if self.width_right is None or self.width_left is None or module_data.path is None:
            return

        self._visualize_road_boundaries(module_data)
        self._visualize_contour_errors(module_data)

    def _visualize_road_boundaries(self, module_data):
        # Create publisher for road boundaries
        line_publisher = self.create_visualization_publisher("road_boundary")
        line = line_publisher.add_new_line()
        line.set_scale(0.1)
        line.set_color_int(0)

        prev_right, prev_left = None, None

        # Sample points along the path
        cur_s = 0.
        while cur_s < self.width_right.m_x_[-1]:
            right = self.width_right(cur_s)
            left = self.width_left(cur_s)

            path_point = module_data.path.get_point(cur_s)
            dpath = module_data.path.get_orthogonal(cur_s)

            if cur_s > 0:
                line.add_line(prev_left, path_point - dpath * left)
                line.add_line(prev_right, path_point + dpath * right)

            prev_left = path_point - dpath * left
            prev_right = path_point + dpath * right
            cur_s += 0.5

        line_publisher.publish()

    def _visualize_contour_errors(self, module_data):
        # Visualization of contour error and road boundaries
        publisher = ROSPointMarker(f"{self.name}/road_boundary_points")
        points = publisher.get_new_point_marker("CUBE")
        contour_line = publisher.get_new_line()
        contour_line.set_scale(0.15)
        points.set_scale(0.15, 0.15, 0.15)

        robot_half_width = self.get_config_value("robot.width", 0.5) / 2.0

        for k in range(1, self.solver.N):
            cur_s = self.solver.get_output(k, "spline")
            path_point = module_data.path.get_point(cur_s)

            # Visualize path point
            points.set_color_int(5, 10)
            points.add_point_marker(path_point)

            dpath = module_data.path.get_orthogonal(cur_s)

            # Calculate boundary points
            boundary_left = path_point - dpath * self.width_left(cur_s)
            boundary_right = path_point + dpath * self.width_right(cur_s)

            # Visualize planned position
            pos = np.array([self.solver.get_output(k, "x"), self.solver.get_output(k, "y")])
            points.set_color(0., 0., 0.)
            points.add_point_marker(pos, 0.2)

            # Calculate and visualize contour error
            contour_error = dpath.T @ (pos - path_point)
            w_right = self.width_right(cur_s)
            w_left = self.width_left(cur_s)

            # Red contour error line
            contour_line.set_color(1., 0., 0.)
            contour_line.add_line(path_point, path_point + dpath * contour_error, 0.1)

            # Green left boundary line
            contour_line.set_color(0., 1., 0.)
            contour_line.add_line(path_point, path_point + dpath * (-w_left + robot_half_width))

            # Blue right boundary line
            contour_line.set_color(0., 0., 1.)
            contour_line.add_line(path_point, path_point + dpath * (w_right - robot_half_width))

            # Visualize boundary points
            points.set_color(0., 0., 0.)
            points.add_point_marker(boundary_left)
            points.add_point_marker(boundary_right)

        publisher.publish()