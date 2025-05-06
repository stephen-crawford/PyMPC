from planner.src.types import *
from planner_modules.base_objective import BaseObjective
from utils.const import OBJECTIVE
from utils.utils import read_config_file, LOG_DEBUG, PROFILE_SCOPE, distance, haar_difference_without_abs

from planner_modules.base_constraint import BaseConstraint

CONFIG = read_config_file()


class Contouring(BaseObjective):
	def __init__(self, solver):
		# Override module_type since Contouring is an OBJECTIVE not a CONSTRAINT
		super().__init__(solver)

		# Configuration options from CONFIG
		self.num_segments = self.get_config_value("num_segments")
		self.add_road_constraints = self.get_config_value("add_road_constraints")
		self.two_way_road = self.get_config_value("two_way")
		self.dynamic_velocity_reference = self.get_config_value("dynamic_velocity_reference")

		# Initialize class attributes
		self.spline = None  # Will be set by on_data_received
		self.closest_point = 0
		self.closest_segment = 0
		self.bound_left = None
		self.bound_right = None

		LOG_DEBUG("Contouring module successfully initialized")

	def update(self, state, real_time_data, module_data):
		PROFILE_SCOPE("Contouring update")
		LOG_DEBUG(f"Updating {self.name.title()} Objective")

		if self.spline is None:
			LOG_DEBUG("No spline available")
			return

		# Update the closest point
		last_segment, self.closest_segment = self.spline.find_closest_point(state.get_position(), self.closest_segment)

		if module_data.path is None and self.spline is not None:
			module_data.path = self.spline

		state.set("spline", last_segment)  # Initialize the spline state
		module_data.current_path_segment = self.closest_segment

		if self.add_road_constraints:
			self.construct_road_constraints(real_time_data, module_data)

	def define_parameters(self, params):
		"""Define all parameters used by this module"""
		# Core parameters
		params.add("contour", add_to_rqt_reconfigure=True)
		params.add("lag", add_to_rqt_reconfigure=True)
		params.add("terminal_angle", add_to_rqt_reconfigure=True)
		params.add("terminal_contouring", add_to_rqt_reconfigure=True)

		# Velocity reference parameters if needed
		if self.dynamic_velocity_reference:
			params.add("reference_velocity", add_to_rqt_reconfigure=True)
			params.add("velocity", add_to_rqt_reconfigure=True)

		# Spline parameters
		for i in range(self.num_segments):
			# X-coordinates
			params.add(f"spline_a", bundle_name="spline_a")
			params.add(f"spline_b", bundle_name="spline_b")
			params.add(f"spline_c", bundle_name="spline_c")
			params.add(f"spline_d", bundle_name="spline_d")

			# Y-coordinates
			params.add(f"spline_ya", bundle_name="spline_ya")
			params.add(f"spline_yb", bundle_name="spline_yb")
			params.add(f"spline_yc", bundle_name="spline_yc")
			params.add(f"spline_yd", bundle_name="spline_yd")

			# Segment start
			params.add(f"spline_start", bundle_name="spline_start")

		return params

	def set_parameters(self, data, module_data, k):
		print(f"set_parameters called with k={k}")
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

			self.set_solver_parameter("contour", contouring_weight, k)
			self.set_solver_parameter("lag", lag_weight, k)
			self.set_solver_parameter("terminal_angle", terminal_angle_weight, k)
			self.set_solver_parameter("terminal_contouring", terminal_contouring_weight, k)

			if self.dynamic_velocity_reference:
				self.set_solver_parameter("reference_velocity", velocity_weight, k)
				self.set_solver_parameter("velocity", velocity_weight, k)
		print("Setting spline parameters")
		self.set_spline_parameters(k)

	def set_spline_parameters(self, k):
		print("Now setting spline params")
		if self.spline is None:
			return

		# Set spline parameters for each segment
		for i in range(self.num_segments):
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
			self.set_solver_parameter("spline_a", ax, k, i)
			self.set_solver_parameter("spline_b", bx, k, i)
			self.set_solver_parameter("spline_c", cx, k, i)
			self.set_solver_parameter("spline_d", dx, k, i)

			self.set_solver_parameter("spline_ya", ay, k, i)
			self.set_solver_parameter("spline_yb", by, k, i)
			self.set_solver_parameter("spline_yc", cy, k, i)
			self.set_solver_parameter("spline_yd", dy, k, i)

			# Set solver parameter for spline segment start
			self.set_solver_parameter("spline_start", start, k, i)

	def get_value(self, model, params, settings, stage_idx):
		cost = 0

		pos_x = model.get("x")
		pos_y = model.get("y")
		psi = model.get("psi")
		v = model.get("v")
		s = model.get("spline")

		contour_weight = params.get("contour")
		lag_weight = params.get("lag")

		# From path
		if self.dynamic_velocity_reference:
			if not params.has_parameter("spline_v0_a"):
				raise IOError(
					"contouring/dynamic_velocity_reference is enabled, but there is no PathReferenceVelocity module.")

			path_velocity = SplineAdapter(params, "spline_v", self.get_num_segments, s)
			reference_velocity = path_velocity.at(s)
			velocity_weight = params.get("velocity")

		path = Spline2DAdapter(params, self.get_num_segments, s)
		path_x, path_y = path.at(s)
		path_dx_normalized, path_dy_normalized = path.deriv_normalized(s)

		# MPCC
		contour_error = path_dy_normalized * (pos_x - path_x) - path_dx_normalized * (pos_y - path_y)
		lag_error = path_dx_normalized * (pos_x - path_x) + path_dy_normalized * (pos_y - path_y)

		cost += lag_weight * lag_error ** 2
		cost += contour_weight * contour_error ** 2

		if self.dynamic_velocity_reference:
			cost += velocity_weight * (v - reference_velocity) ** 2

		# Terminal cost
		if True and stage_idx == settings["N"] - 1:
			terminal_angle_weight = params.get("terminal_angle")
			terminal_contouring_mp = params.get("terminal_contouring")

			# Compute the angle w.r.t. the path
			path_angle = cd.atan2(path_dy_normalized, path_dx_normalized)
			angle_error = haar_difference_without_abs(psi, path_angle)

			# Penalize the angle error
			cost += terminal_angle_weight * angle_error ** 2
			cost += terminal_contouring_mp * lag_weight * lag_error ** 2
			cost += terminal_contouring_mp * contour_weight * contour_error ** 2

		return cost

	def on_data_received(self, data, data_name):
		if data_name == "reference_path":
			LOG_DEBUG("Received Reference Path")

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
					self.spline.get_t_vector())

				self.bound_right = TwoDimensionalSpline(
					data.right_bound.x,
					data.right_bound.y,
					self.spline.get_t_vector())

				# update the road width
				CONFIG["road"]["width"] = distance(self.bound_left.get_point(0), self.bound_right.get_point(0))

			self.closest_segment = 0

	def is_data_ready(self, data, missing_data):
		required_fields = ["reference_path"]
		missing_fields = self.check_data_availability(data, required_fields)

		if missing_fields or data.reference_path.x.empty():
			missing_data += "Reference Path "
			return False

		return True

	def is_objective_reached(self, state, data):
		if self.spline is None:
			return False

		# Check if we reached the end of the spline
		return distance(state.getPos(), self.spline.get_point(self.spline.parameter_length())) < 1.0

	def construct_road_constraints(self, data, module_data):
		LOG_DEBUG("Constructing road constraints.")

		if self.bound_left is None or self.bound_right is None:
			self.construct_road_constraints_from_centerline(data, module_data)
		else:
			self.construct_road_constraints_from_bounds(data, module_data)

	def construct_road_constraints_from_centerline(self, data, module_data):
		# If bounds are not supplied construct road constraints based on a set width
		print("Constructing road constraints from centerline")

		if module_data.static_obstacles.empty():
			print("Static obstacles empty")
			module_data.static_obstacles.resize(self.solver.N)
			for k in range(module_data.static_obstacles.size()):
				module_data.static_obstacles[k].reserve(2)

		# Get road width
		road_width_half = CONFIG["road"]["width"] / 2.0
		print("Static obstacles: " + str(module_data.static_obstacles))
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

	def reset(self):
		if hasattr(self, 'spline') and self.spline is not None:
			self.spline.reset()
		self.closest_segment = 0