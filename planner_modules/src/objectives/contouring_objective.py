import casadi as cd

from planner_modules.src.objectives.base_objective import BaseObjective
from planning.src.types import StaticObstacle
from utils.math_utils import TwoDimensionalSpline, haar_difference_without_abs, \
	distance, CasadiSpline, CasadiSpline2D
from utils.utils import LOG_DEBUG, PROFILE_SCOPE


class ContouringObjective(BaseObjective):
	def __init__(self, solver):
		super().__init__(solver)
		LOG_DEBUG("Contouring Objective initializing")
		# Configuration options from CONFIGs
		self.num_segments = self.get_config_value("contouring.num_segments")
		self.add_road_constraints = self.get_config_value("contouring.add_road_constraints")
		self.two_way_road = self.get_config_value("road.two_way")
		self.dynamic_velocity_reference = self.get_config_value("contouring.dynamic_velocity_reference")

		# Initialize class attributes
		self.spline = None
		self.closest_point = 0
		self.closest_segment = 0
		self.bound_left = None
		self.bound_right = None

		LOG_DEBUG("Contouring module successfully initialized")

	def update(self, state, data):
		PROFILE_SCOPE("Contouring update")
		LOG_DEBUG(f"Updating {self.name.title()} Objective")

		if self.spline is None:
			LOG_DEBUG("No spline available")
			return

		# Update the closest point
		segment_index, parameter_value = self.spline.find_closest_point(state.get_position(), self.closest_segment)
		self.closest_segment = segment_index

		if  self.spline is not None:
			data.path = self.spline

		state.set("spline", self.closest_segment)  # Initialize the spline state
		data.current_path_segment = self.closest_segment

		if self.add_road_constraints:
			self.construct_road_constraints(data)

	def define_parameters(self, params):
		"""Define all parameters used by this module"""
		LOG_DEBUG("Defining contouring objective parameters")
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
			params.add(f"spline_{i}_ax")
			params.add(f"spline_{i}_bx")
			params.add(f"spline_{i}_cx")
			params.add(f"spline_{i}_dx")

			# Y-coordinates
			params.add(f"spline_{i}_ay")
			params.add(f"spline_{i}_by")
			params.add(f"spline_{i}_cy")
			params.add(f"spline_{i}_dy")

			# Segment start
			params.add(f"spline_{i}_start")

		return params

	def set_parameters(self, parameter_manager, data, k):
		print(f"set_parameters called with k={k}")
		# Retrieve weights once
		if k == 0:
			contouring_weight = self.get_config_value("weights.contour")
			lag_weight = self.get_config_value("weights.lag")

			terminal_angle_weight = self.get_config_value("weights.terminal_angle")
			terminal_contouring_weight = self.get_config_value("weights.terminal_contouring")

			if self.dynamic_velocity_reference:
				reference_velocity_weight = self.get_config_value("weights.reference_velocity")
				velocity_weight = self.get_config_value("weights.velocity")
				parameter_manager.set_parameter("reference_velocity", reference_velocity_weight)
				parameter_manager.set_parameter("velocity", velocity_weight)

			parameter_manager.set_parameter("contour", contouring_weight)
			parameter_manager.set_parameter("lag", lag_weight)
			parameter_manager.set_parameter("terminal_angle", terminal_angle_weight)
			parameter_manager.set_parameter("terminal_contouring", terminal_contouring_weight)

		self.set_spline_parameters(parameter_manager, k)

	def set_spline_parameters(self, parameter_manager, k):
		if self.spline is None:
			return

		# Set spline parameters for each segment
		for i in range(self.num_segments):
			# Calculate the index for the spline segment
			index = self.closest_segment + i

			if index >= self.spline.get_num_segments():
				# Use the last segment if we're past the end
				index = self.spline.get_num_segments() - 1
			LOG_DEBUG("Contouring objective trying to set spline parameters for index, " + str(index) + " given numb segments is " + str(self.spline.get_num_segments()))
			# Retrieve spline parameters
			ax, bx, cx, dx, ay, by, cy, dy = self.spline.get_parameters(index)

			# Get the start position of the spline segment
			start = self.spline.get_segment_start(index)

			# Set solver parameters for each spline coefficient
			parameter_manager.set_parameter(f"spline_{i}_ax", ax)
			parameter_manager.set_parameter(f"spline_{i}_bx", bx)
			parameter_manager.set_parameter(f"spline_{i}_cx", cx)
			parameter_manager.set_parameter(f"spline_{i}_dx", dx)

			parameter_manager.set_parameter(f"spline_{i}_ay", ay)
			parameter_manager.set_parameter(f"spline_{i}_by", by)
			parameter_manager.set_parameter(f"spline_{i}_cy", cy)
			parameter_manager.set_parameter(f"spline_{i}_dy", dy)

			# Set solver parameter for spline segment start
			parameter_manager.set_parameter(f"spline_{i}_start", start)

	def get_value(self, model, params, settings, stage_idx):
		cost = 0

		pos_x = model.get("x")
		pos_y = model.get("y")
		psi = model.get("psi")
		v = model.get("v")
		s = model.get("spline")

		contour_weight = params.get("contour")
		lag_weight = params.get("lag")


		path = CasadiSpline2D(params, self.num_segments)
		path_x, path_y = path.at(s)
		path_dx_normalized, path_dy_normalized = path.deriv_normalized(s)

		contour_error = path_dy_normalized * (pos_x - path_x) - path_dx_normalized * (pos_y - path_y)
		lag_error = path_dx_normalized * (pos_x - path_x) + path_dy_normalized * (pos_y - path_y)

		cost += lag_weight * lag_error ** 2
		cost += contour_weight * contour_error ** 2

		if self.dynamic_velocity_reference:
			if not params.has_parameter("spline_0_va"):
				raise IOError(
					"contouring/dynamic_velocity_reference is enabled, but there is no PathReferenceVelocity module.")

			path_velocity = CasadiSpline(params, "spline_v", self.num_segments)
			reference_velocity = path_velocity.at(s)
			velocity_weight = params.get("velocity")

			cost += velocity_weight * (v - reference_velocity) ** 2

		# Terminal cost
		if True and stage_idx == settings["horizon"] - 1:
			terminal_angle_weight = params.get("terminal_angle")
			terminal_contouring_mp = params.get("terminal_contouring")
			LOG_DEBUG(f"DEBUG: terminal_angle_weight = {terminal_angle_weight}")
			LOG_DEBUG(f"DEBUG: terminal_contouring = {terminal_contouring_mp}")
			# Compute the angle w.r.t. the path
			LOG_DEBUG(f"path_dx_normalized = {path_dx_normalized}")
			LOG_DEBUG(f"path_dy_normalized = {path_dy_normalized}")
			path_angle = cd.atan2(path_dy_normalized, path_dx_normalized)
			angle_error = haar_difference_without_abs(psi, path_angle)
			LOG_DEBUG(f"DEBUG: angle_error = {angle_error}")
			LOG_DEBUG(f"DEBUG: path_angle = {path_angle}")
			# Penalize the angle error
			cost += terminal_angle_weight * angle_error ** 2
			cost += terminal_contouring_mp * lag_weight * lag_error ** 2
			cost += terminal_contouring_mp * contour_weight * contour_error ** 2
			LOG_DEBUG(f"DEBUG: cost = {cost}")

		return cost

	def on_data_received(self, data, data_name):
		LOG_DEBUG("RECEIVED DATA FOR CONTOURING OBJ")
		if data_name == "reference_path":
			LOG_DEBUG("Received Reference Path")

			# Construct a spline from the given points
			if data.reference_path.s is None:
				LOG_DEBUG("data.reference.s is None")
				self.spline = TwoDimensionalSpline(data.reference_path.x, data.reference_path.y)
			else:
				LOG_DEBUG("data.reference.s is not None")
				self.spline = TwoDimensionalSpline(data.reference_path.x, data.reference_path.y, data.reference_path.s)

			LOG_DEBUG("self.spline now" + str(self.spline))
			if self.add_road_constraints and (not data.left_bound is None and not data.right_bound is None):
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
				#write_to_config("road.width", distance(self.bound_left.get_point(0), self.bound_right.get_point(0)))

			self.closest_segment = 0

	def process_reference_path(self, data):
		LOG_DEBUG("Processing reference path for objective")

		self.on_data_received(data, "reference_path")

	def is_data_ready(self, data):
		missing_data = ""
		if not data.has("reference_path") or data.reference_path.x is None:
			missing_data += "reference_path"

		return len(missing_data) < 1

	def is_objective_reached(self, state, data):
		is_ready = self.is_data_ready(data)
		if not is_ready:
			LOG_DEBUG("Data not ready yet")
			return False
		if self.spline is None:
			LOG_DEBUG("Spline not found")
			return False

		return distance(state.get_position(), self.spline.get_point(self.spline.parameter_length())) < 1.0

	def construct_road_constraints(self, data):
		LOG_DEBUG("Constructing road constraints.")

		if self.bound_left is None or self.bound_right is None:
			self.construct_road_constraints_from_centerline(data)
		else:
			self.construct_road_constraints_from_bounds(data)

	def construct_road_constraints_from_centerline(self, data):
		# If bounds are not supplied construct road constraints based on a set width

		if data.static_obstacles.empty():
			data.static_obstacles.resize(self.solver.horizon)
			for k in range(data.static_obstacles.size()):
				data.static_obstacles[k].reserve(2)

		# Get road width
		road_width_half = self.get_config_value("road.width") / 2.0
		for k in range(self.solver.horizon):
			data.static_obstacles[k].clear()

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

			data.static_obstacles[k].emplace_back(A, b)

			# right HALFSPACE
			A = self.spline.get_orthogonal(cur_s)

			boundary_right = path_point - dpath * (road_width_half - data.robot_area[0].radius)
			b = A.transpose() * boundary_right

			data.static_obstacles[k].emplace_back(-A, -b)

	def construct_road_constraints_from_bounds(self, data):
		if data.static_obstacles is None:
			data.set("static_obstacles", [None] * self.solver.horizon)

		LOG_DEBUG("Forecasting road from bounds")
		for k in range(self.solver.horizon):
			# Create a static obstacle for this time step
			data.static_obstacles[k] = StaticObstacle()

			# Get the current position on the spline
			cur_s = self.solver.get_ego_prediction(k, "spline")

			# Left bound halfspace constraint
			Al = self.bound_left.get_orthogonal(cur_s)
			bl = Al.transpose() @ (self.bound_left.get_point(cur_s) + Al * data.robot_area[0].radius)
			data.static_obstacles[k].add_halfspace(-Al, -bl)

			# Right bound halfspace constraint
			Ar = self.bound_right.get_orthogonal(cur_s)
			br = Ar.transpose() @ (self.bound_right.get_point(cur_s) - Ar * data.robot_area[0].radius)
			data.static_obstacles[k].add_halfspace(Ar, br)

	def reset(self):
		if hasattr(self, 'spline') and self.spline is not None:
			self.spline.reset()
		self.closest_segment = 0