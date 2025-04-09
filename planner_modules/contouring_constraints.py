import logging
import numpy as np
from scipy.interpolate import CubicSpline

from solver.solver_interface import set_solver_parameter
from utils.const import OBJECTIVE, CONSTRAINT
from utils.utils import read_config_file, LOG_DEBUG, distance
from utils.visualizer import ROSLine, ROSPointMarker

CONFIG = read_config_file()


class ContouringConstraints:
    def __init__(self, solver):
        self.solver = solver
        self.module_type = CONSTRAINT
        self.name = "contouring_constraints"
        self.width_left = None
        self.width_right = None

        LOG_DEBUG("Initializing Contouring Constraints")
        self.num_segments = CONFIG["contouring"]["num_segments"]
        LOG_DEBUG("Contouring Constraints successfully initialized")

    def update(self, state, data, module_data):
        if module_data.path_width_left is None and self.width_left is not None:
            module_data.path_width_left = self.width_left

        if module_data.path_width_right is None and self.width_right is not None:
            module_data.path_width_right = self.width_right

    def on_data_received(self, data, data_name):
        if data_name == "reference_path":
            LOG_DEBUG("Reference Path Received")

            if not data.left_bound.empty() and not data.right_bound.empty():
                LOG_DEBUG("Received Road Boundaries")

                widths_left = np.zeros(len(data.left_bound.x))
                widths_right = np.zeros(len(data.right_bound.x))

                for i in range(len(widths_left)):
                    center = np.array([data.reference_path.x[i], data.reference_path.y[i]])
                    left = np.array([data.left_bound.x[i], data.left_bound.y[i]])
                    right = np.array([data.right_bound.x[i], data.right_bound.y[i]])

                    widths_left[i] = np.linalg.norm(center - left)
                    widths_right[i] = np.linalg.norm(center - right)

                # Initialize splines if they don't exist
                self.width_left = CubicSpline(data.reference_path.s, widths_left)
                self.width_right = CubicSpline(data.reference_path.s, widths_right)

                # Add custom methods to match C++ implementation
                self.width_left.m_x_ = data.reference_path.s
                self.width_right.m_x_ = data.reference_path.s

                # Add method to get spline parameters
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
            logging.log(10, "ContouringConstraints::set_parameters")

        for i in range(self.num_segments):
            index = module_data.current_path_segment + i

            # Boundaries
            ra, rb, rc, rd = 0., 0., 0., 0.
            la, lb, lc, ld = 0., 0., 0., 0.

            if index < len(self.width_right.m_x_) - 1:
                ra, rb, rc, rd = self.width_right.get_parameters(index, ra, rb, rc, rd)
                la, lb, lc, ld = self.width_left.get_parameters(index, la, lb, lc, ld)
            else:
                ra, rb, rc, rd = self.width_right.get_parameters(len(self.width_right.m_x_) - 1, ra, rb, rc, rd)
                la, lb, lc, ld = self.width_left.get_parameters(len(self.width_left.m_x_) - 1, la, lb, lc, ld)

                ra = 0.
                rb = 0.
                rc = 0.
                la = 0.
                lb = 0.
                lc = 0.

            # Boundary - match the C++ parameter names exactly
            self.set_solver_parameter_width_right_a(k, self.solver.params, ra, i)
            self.set_solver_parameter_width_right_b(k, self.solver.params, rb, i)
            self.set_solver_parameter_width_right_c(k, self.solver.params, rc, i)
            self.set_solver_parameter_width_right_d(k, self.solver.params, rd, i)

            self.set_solver_parameter_width_left_a(k, self.solver.params, la, i)
            self.set_solver_parameter_width_left_b(k, self.solver.params, lb, i)
            self.set_solver_parameter_width_left_c(k, self.solver.params, lc, i)
            self.set_solver_parameter_width_left_d(k, self.solver.params, ld, i)

        if k == 1:
            LOG_DEBUG("ContouringConstraints.set_parameters Done")

    # Parameter setter methods to match C++ implementation
    def set_solver_parameter_width_right_a(self, k, params, value, i):
        set_solver_parameter(params, "width_right_a", value, k, index=i, settings=CONFIG)

    def set_solver_parameter_width_right_b(self, k, params, value, i):
        set_solver_parameter(params, "width_right_b", value, k, index=i, settings=CONFIG)

    def set_solver_parameter_width_right_c(self, k, params, value, i):
        set_solver_parameter(params, "width_right_c", value, k, index=i, settings=CONFIG)

    def set_solver_parameter_width_right_d(self, k, params, value, i):
        set_solver_parameter(params, "width_right_d", value, k, index=i, settings=CONFIG)

    def set_solver_parameter_width_left_a(self, k, params, value, i):
        set_solver_parameter(params, "width_left_a", value, k, index=i, settings=CONFIG)

    def set_solver_parameter_width_left_b(self, k, params, value, i):
        set_solver_parameter(params, "width_left_b", value, k, index=i, settings=CONFIG)

    def set_solver_parameter_width_left_c(self, k, params, value, i):
        set_solver_parameter(params, "width_left_c", value, k, index=i, settings=CONFIG)

    def set_solver_parameter_width_left_d(self, k, params, value, i):
        set_solver_parameter(params, "width_left_d", value, k, index=i, settings=CONFIG)

    def is_data_ready(self, data, missing_data):
        if data.left_bound.empty() or data.right_bound.empty():
            missing_data += "Road Bounds "
            return False

        return True

    def visualize(self, data, module_data):
        if not CONFIG["debug_visuals"]:
            return

        LOG_DEBUG("ContouringConstraints::Visualize")

        if self.width_right is None or self.width_left is None or module_data.path is None:
            return

        line_publisher = ROSLine(self.name + "/road_boundary")
        line = line_publisher.add_new_line()
        line.set_scale(0.1)
        line.set_color_int(0)

        prev_right, prev_left = None, None

        cur_s = 0.
        while cur_s < self.width_right.m_x_[-1]:  # Use [-1] instead of .back()
            right = self.width_right(cur_s)  # Using the callable feature of CubicSpline
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

        # Visualization of contour error and road boundaries
        publisher = ROSPointMarker(self.name + "/road_boundary_points")
        points = publisher.get_new_point_marker("CUBE")
        contour_line = publisher.get_new_line()
        contour_line.set_scale(0.15)
        points.set_scale(0.15, 0.15, 0.15)

        for k in range(1, self.solver.N):  # Start from 1 to match C++ loop
            cur_s = self.solver.get_output(k, "spline")
            path_point = module_data.path.get_point(cur_s)

            points.set_color_int(5, 10)
            points.add_point_marker(path_point)

            dpath = module_data.path.get_orthogonal(cur_s)

            # Line is parallel to the spline
            boundary_left = path_point - dpath * self.width_left(cur_s)
            boundary_right = path_point + dpath * self.width_right(cur_s)

            # Visualize the contouring error
            w_cur = CONFIG["robot"]["width"] / 2.0
            pos = np.array([self.solver.get_output(k, "x"), self.solver.get_output(k, "y")])

            points.set_color(0., 0., 0.)
            points.add_point_marker(pos, 0.2)  # Planned positions and black dots

            contour_error = dpath.T @ (pos - path_point)  # Matrix multiplication
            w_right = self.width_right(cur_s)
            w_left = self.width_left(cur_s)

            contour_line.set_color(1., 0., 0.)  # Red contour error
            contour_line.add_line(path_point, path_point + dpath * contour_error, 0.1)

            contour_line.set_color(0., 1., 0.)  # Width left and right in green and blue
            contour_line.add_line(path_point, path_point + dpath * (-w_left + w_cur))

            contour_line.set_color(0., 0., 1.)
            contour_line.add_line(path_point, path_point + dpath * (w_right - w_cur))

            points.set_color(0., 0., 0.)
            points.add_point_marker(boundary_left)
            points.add_point_marker(boundary_right)

        publisher.publish()