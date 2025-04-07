from functools import partial
from math import exp
from venv import logger

from solver.solver_interface import set_solver_parameter
from utils.const import OBJECTIVE
from utils.utils import read_config_file, LOG_DEBUG, PROFILE_SCOPE, distance
from planner.src.types import *

CONFIG = read_config_file()


class Contouring:

    def __init__(self, solver):
        self.module_type = OBJECTIVE
        self.solver = solver
        self.name = "contouring"
        self.controller = (self.module_type, solver, self.name)
        LOG_DEBUG("Initializing contouring module")

        # Configuration options from CONFIG
        self.get_num_segments = CONFIG["contouring"]["get_num_segments"]
        self.add_road_constraints = CONFIG["contouring"]["add_road_constraints"]
        self.two_way_road = CONFIG["road"]["two_way"]
        self.dynamic_velocity_reference = CONFIG["contouring"]["dynamic_velocity_reference"]

        # Initialize class attributes
        self.spline = None  # Will be set by on_data_received
        self.closest_point = 0
        self.closest_segment = 0
        self.bound_left = None
        self.bound_right = None
        self.n_segments = self.get_num_segments  # This was missing

        self.set_solver_param = partial(set_solver_parameter, settings=CONFIG)

        LOG_DEBUG("Contouring module successfully initialized")

    def update(self, state, real_time_data, module_data):
        PROFILE_SCOPE("Contouring update")
        LOG_DEBUG("Updating contouring module")

        if self.spline is None:
            LOG_DEBUG("No spline available yet")
            return

        # Update the closest point
        closest_s, self.closest_segment = self.spline.find_closest_point(state.getPos(), self.closest_segment)

        if module_data.path is None and self.spline is not None:
            module_data.path = self.spline

        state.set("spline", closest_s)  # Initialize the spline state
        module_data.current_path_segment = self.closest_segment

        if self.add_road_constraints:
            self.construct_road_constraints(real_time_data, module_data)

    def set_parameters(self, data, module_data, k):
        # Retrieve weights once
        if k == 0:
            contouring_weight = CONFIG["weights"]["contour"]
            lag_weight = CONFIG["weights"]["lag"]

            terminal_angle_weight = CONFIG["weights"]["terminal_angle"]
            terminal_contouring_weight = CONFIG["weights"]["terminal_contouring"]

            if self.dynamic_velocity_reference:
                reference_velocity = CONFIG["weights"]["reference_velocity"]
                velocity_weight = CONFIG["weights"]["velocity"]
            else:
                velocity_weight = None

            self.set_solver_param(self.solver.params, "contour", contouring_weight, k)
            self.set_solver_param(self.solver.params, "lag", lag_weight, k)
            self.set_solver_param(self.solver.params, "terminal_angle", terminal_angle_weight, k)
            self.set_solver_param(self.solver.params, "terminal_contouring", terminal_contouring_weight, k)

            if self.dynamic_velocity_reference:
                self.set_solver_param(self.solver.params, "reference_velocity", velocity_weight, k)
                self.set_solver_param(self.solver.params, "velocity", velocity_weight, k)

        self.set_spline_parameters(k)

    def set_spline_parameters(self, k):
        if self.spline is None:
            return

        # Set spline parameters for each segment
        for i in range(self.n_segments):
            # Calculate the index for the spline segment
            index = self.closest_segment + i

            if index >= self.spline.get_num_segments():
                # Use the last segment if we're past the end
                index = self.spline.get_num_segments() - 1

            # Retrieve spline parameters
            ax, bx, cx, dx, ay, by, cy, dy = self.spline.get_parameters(index)

            # Get the start position of the spline segment
            start = self.spline.get_segment_start(index)

            # Set solver parameters for each spline coefficient
            self.set_solver_param(self.solver.params, "spline_a", ax, k, index=i)
            self.set_solver_param(self.solver.params, "spline_b", bx, k, index=i)
            self.set_solver_param(self.solver.params, "spline_c", cx, k, index=i)
            self.set_solver_param(self.solver.params, "spline_d", dx, k, index=i)

            self.set_solver_param(self.solver.params, "spline_ya", ay, k, index=i)
            self.set_solver_param(self.solver.params, "spline_yb", by, k, index=i)
            self.set_solver_param(self.solver.params, "spline_yc", cy, k, index=i)
            self.set_solver_param(self.solver.params, "spline_yd", dy, k, index=i)

            # Set solver parameter for spline segment start
            self.set_solver_param(self.solver.params, "spline_start", start, k, index=i)

    def on_data_received(self, data, data_name):
        if data_name == "reference_path":
            LOG_DEBUG( "Received Reference Path")

            # Construct a spline from the given points
            if data.reference_path.s.empty():
                self.spline = TwoDimensionalSpline(data.reference_path.x, data.reference_path.y)
            else:
                self.spline = TwoDimensionalSpline(data.reference_path.x, data.reference_path.y, data.reference_path.s)

            if self.add_road_constraints and (not data.left_bound.empty() and not data.right_bound.empty()):
                # Add bounds
                self.bound_left = TwoDimensionalSpline(
                    data.left_bound.x,
                    data.left_bound.y,
                    self.spline.getTVector())

                self.bound_right = TwoDimensionalSpline(
                    data.right_bound.x,
                    data.right_bound.y,
                    self.spline.getTVector())

                # update the road width
                CONFIG["road"]["width"] = distance(self.bound_left.get_point(0), self.bound_right.get_point(0))

            self.closest_segment = 0

    def is_data_ready(self, data, missing_data):
        if data.reference_path.x.empty():
            missing_data += "Reference Path "

        return not data.reference_path.x.empty()

    def is_objective_reached(self, state, data):
        if self.spline is None:
            return False

        # Check if we reached the end of the spline
        return distance(state.getPos(), self.spline.get_point(self.spline.parameter_length())) < 1.0

    def construct_road_constraints(self, data, module_data):
        LOG_DEBUG( "Constructing road constraints.")

        if self.bound_left is None or self.bound_right is None:
            self.construct_road_constraints_from_centerline(data, module_data)
        else:
            self.construct_road_constraints_from_bounds(data, module_data)

    def construct_road_constraints_from_centerline(self, data, module_data):
        # If bounds are not supplied construct road constraints based on a set width
        if module_data.static_obstacles.empty():
            module_data.static_obstacles.resize(self.solver.N)
            for k in range(module_data.static_obstacles.size()):
                module_data.static_obstacles[k].reserve(2)

        # Get road width
        road_width_half = CONFIG["road"]["width"] / 2.0

        for k in range(self.solver.N):
            module_data.static_obstacles[k].clear()

            cur_s = self.solver.get_ego_prediction(k, "spline")

            # This is the final point and the normal vector of the path
            path_point = self.spline.get_point(cur_s)
            dpath = self.spline.get_orthogonal(cur_s)

            # left HALFSPACE
            A = self.spline.get_orthogonal(cur_s)
            if self.two_way_road:
                width_times = 3.0
            else:
                width_times = 1.0

            # line is parallel to the spline
            boundary_left = path_point + dpath * (width_times * road_width_half - data.robot_area[0].radius)

            b = A.transpose() * boundary_left

            module_data.static_obstacles[k].emplace_back(A, b)

            # right HALFSPACE
            A = self.spline.get_orthogonal(cur_s)

            boundary_right = path_point - dpath * (road_width_half - data.robot_area[0].radius)
            b = A.transpose() * boundary_right

            module_data.static_obstacles[k].emplace_back(-A, -b)

    def construct_road_constraints_from_bounds(self, data, module_data):
        if module_data.static_obstacles.empty():
            module_data.static_obstacles.resize(self.solver.N)
            for k in range(module_data.static_obstacles.size()):
                module_data.static_obstacles[k].reserve(2)

        for k in range(self.solver.N):
            module_data.static_obstacles[k].clear()
            cur_s = self.solver.get_ego_prediction(k, "spline")

            # left
            Al = self.bound_left.get_orthogonal(cur_s)
            bl = Al.transpose() * (self.bound_left.get_point(cur_s) + Al * data.robot_area[0].radius)
            module_data.static_obstacles[k].emplace_back(-Al, -bl)

            # right HALFSPACE
            Ar = self.bound_right.get_orthogonal(cur_s)
            br = Ar.transpose() * (self.bound_right.get_point(cur_s) - Ar * data.robot_area[0].radius)
            module_data.static_obstacles[k].emplace_back(Ar, br)

    # Visualization methods would need to be adapted to your visualization system
    # I've commented them out as they rely on undefined components

    def reset(self):
        if hasattr(self, 'spline') and self.spline is not None:
            self.spline.reset()
        self.closest_segment = 0