import numpy as np
import casadi as cd
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from planner_modules.src.objectives.base_objective import BaseObjective
from planning.src.types import StaticObstacle, ReferencePath, compute_arc_length
from utils.math_utils import distance, haar_difference_without_abs, safe_norm
from utils.utils import LOG_DEBUG, LOG_INFO, LOG_WARN


class ContouringObjective(BaseObjective):
	def __init__(self, solver):
		super().__init__(solver)
		LOG_DEBUG("Contouring Objective initializing")
		# Configuration options from CONFIGs
		self.num_segments = self.get_config_value("contouring.num_segments")
		self.add_road_constraints = self.get_config_value("contouring.add_road_constraints")
		self.two_way_road = self.get_config_value("road.two_way")
		self.dynamic_velocity_reference = self.get_config_value("contouring.dynamic_velocity_reference")

		self.closest_point_idx = 0
		self.closest_segment = 0

		# Road bounds
		self.reference_path = None
		self.bound_left_spline = None
		self.bound_right_spline = None

		LOG_DEBUG("Contouring module successfully initialized")

	def update(self, state, data):
		LOG_INFO("ContouringObjective.update")

		if self.reference_path is None:
			LOG_WARN("No reference path available")
			return

		# Update the closest point
		self.closest_point_idx, self.closest_segment = self._find_closest_point(
			state.get_position(),
			self.reference_path)

		# Pass path data to data object
		if self.reference_path is not None:
			data.reference_path = ReferencePath()
			data.reference_path.set('x', self.reference_path.x)
			data.reference_path.set('y', self.reference_path.y)
			data.reference_path.set('x_spline', self.reference_path.x_spline)
			data.reference_path.set('y_spline', self.reference_path.y_spline)
			data.reference_path.set('s', self.reference_path.s)

		data.current_path_segment = self.closest_segment

		if self.add_road_constraints:
			self.construct_road_constraints(data)

		# #Visualization for debug
		# plot_debug = self.get_config_value("plot.debug", True)
		# if plot_debug and self.reference_path is not None:
		# 	path_x = self.reference_path.x
		# 	path_y = self.reference_path.y
		#
		# 	fig, ax = plt.subplots()
		# 	ax.plot(path_x, path_y, label="Reference Path", linewidth=2)
		#
		# 	# Current vehicle state
		# 	pos_x, pos_y = state.get_position()
		# 	ax.plot(pos_x, pos_y, 'ro', label="Current Position")
		#
		# 	# Closest point
		# 	closest_idx = self.closest_point_idx
		# 	ax.plot(path_x[closest_idx], path_y[closest_idx], 'gx', markersize=10, label="Closest Point")
		#
		# 	# Optionally add road boundaries if available
		# 	if self.bound_left_spline is not None and self.bound_right_spline is not None:
		# 		left_pts = self.bound_left_spline(self.reference_path.s)
		# 		right_pts = self.bound_right_spline(self.reference_path.s)
		# 		ax.plot(left_pts[:, 0], left_pts[:, 1], '--', color='blue', label='Left Bound')
		# 		ax.plot(right_pts[:, 0], right_pts[:, 1], '--', color='green', label='Right Bound')
		#
		# 	ax.set_aspect("equal")
		# 	ax.legend()
		# 	ax.set_title("Contouring Update Debug Plot")
		# 	ax.set_xlabel("X")
		# 	ax.set_ylabel("Y")
		# 	plt.grid(True)
		# 	plt.show()

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

		# Parameters for path interpolation (pre-evaluated at segment boundaries)
		for i in range(self.num_segments + 1):
			# Path coordinates at segment boundaries
			params.add(f"path_x_{i}")
			params.add(f"path_y_{i}")

			# Derivatives at segment boundaries (needed for normal vectors)
			params.add(f"path_dx_{i}")
			params.add(f"path_dy_{i}")

			# Velocity reference if needed
			if self.dynamic_velocity_reference:
				params.add(f"path_vel_{i}")

		return params

	def set_parameters(self, parameter_manager, data, k):
		LOG_DEBUG(f"set_parameters called with k={k}")
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

		self.set_path_parameters(parameter_manager)

	def set_path_parameters(self, parameter_manager):
		LOG_INFO("ContouringObjective.set_path_parameters")
		if self.reference_path is None:
			return

		path_x = self.reference_path.x
		path_y = self.reference_path.y

		if len(path_x) < self.num_segments + 1:
			# Interpolate to get the required number of points
			s_original = np.linspace(0, 1, len(path_x))
			s_new = np.linspace(0, 1, self.num_segments + 1)

			path_x_interp = np.interp(s_new, s_original, path_x)
			path_y_interp = np.interp(s_new, s_original, path_y)
		else:
			# Sample at segment boundaries
			indices = np.linspace(0, len(path_x) - 1, self.num_segments + 1).astype(int)
			path_x_interp = path_x[indices]
			path_y_interp = path_y[indices]

		# Calculate derivatives using finite differences
		path_dx = np.gradient(path_x_interp)
		path_dy = np.gradient(path_y_interp)

		# Set parameters
		for i in range(self.num_segments + 1):
			parameter_manager.set_parameter(f"path_x_{i}", float(path_x_interp[i]))
			parameter_manager.set_parameter(f"path_y_{i}", float(path_y_interp[i]))
			parameter_manager.set_parameter(f"path_dx_{i}", float(path_dx[i]))
			parameter_manager.set_parameter(f"path_dy_{i}", float(path_dy[i]))

			# Set velocity reference if using dynamic velocity
			if self.dynamic_velocity_reference and hasattr(self.reference_path,
														   'v') and self.reference_path.v is not None:
				if hasattr(self.reference_path.v, '__call__'):  # It's a function/spline
					s_val = i / self.num_segments if self.num_segments > 0 else 0
					if hasattr(self.reference_path, 's') and self.reference_path.s is not None:
						actual_s = np.interp(s_val, [0, 1], [self.reference_path.s[0], self.reference_path.s[-1]])
						vel_val = self.reference_path.v(actual_s)
					else:
						vel_val = self.reference_path.v[min(i, len(self.reference_path.v) - 1)]
				else:  # It's an array
					vel_val = self.reference_path.v[min(i, len(self.reference_path.v) - 1)]

				parameter_manager.set_parameter(f"path_vel_{i}", float(vel_val))

	def get_value(self, model, params, stage_idx):
		"""
		Create a symbolic objective function for trajectory optimization based on contouring error.

		This function returns a Casadi symbolic expression that represents the contouring objective,
		which will be used by the optimizer as part of the cost function.

		Parameters:
			model: The vehicle model containing symbolic state variables
			params: Parameter manager with symbolic weights and path data
			stage_idx: Current stage in the optimization horizon

		Returns:
			cost: Casadi symbolic expression for the objective function
		"""
		LOG_DEBUG(f"Building symbolic objective for stage {stage_idx}")

		# Get symbolic state variables from the model
		pos_x = model.get("x")  # Symbolic x position
		pos_y = model.get("y")  # Symbolic y position
		psi = model.get("psi")  # Symbolic heading angle
		v = model.get("v")  # Symbolic velocity
		s = model.get("spline")  # Symbolic progress parameter (0-1)

		# Get symbolic weights from parameters
		contour_weight = params.get("contour")
		lag_weight = params.get("lag")

		# Evaluate reference path at current progress (returns symbolic expressions)
		path_x = self._evaluate_spline_casadi(s, "path_x", params)
		path_y = self._evaluate_spline_casadi(s, "path_y", params)

		# Get path tangent vectors (symbolic)
		path_dx = self._evaluate_spline_casadi(s, "path_dx", params)
		path_dy = self._evaluate_spline_casadi(s, "path_dy", params)

		# Calculate normalized tangent vector with safe division
		norm = cd.sqrt(path_dx ** 2 + path_dy ** 2)
		norm_safe = cd.if_else(norm > 1e-6, norm, 1e-6)  # Prevent division by zero

		path_dx_normalized = path_dx / norm_safe
		path_dy_normalized = path_dy / norm_safe

		# Symbolic contouring error (lateral deviation - perpendicular to path)
		contour_error = path_dy_normalized * (pos_x - path_x) - path_dx_normalized * (pos_y - path_y)

		# Symbolic lag error (longitudinal deviation - along path direction)
		lag_error = path_dx_normalized * (pos_x - path_x) + path_dy_normalized * (pos_y - path_y)

		# Build symbolic cost expression
		cost = lag_weight * lag_error ** 2 + contour_weight * contour_error ** 2

		# Add velocity reference tracking to symbolic cost if enabled
		if self.dynamic_velocity_reference:
			reference_velocity = self._evaluate_spline_casadi(s, "path_vel", params)
			velocity_weight = params.get("reference_velocity_weight")
			velocity_cost = velocity_weight * (v - reference_velocity) ** 2
			cost += velocity_cost

		# Add terminal costs at the end of the horizon
		if stage_idx == self.get_config_value("horizon") - 1:
			terminal_angle_weight = params.get("terminal_angle")
			terminal_contouring_multiplier = params.get("terminal_contouring")

			# Symbolic heading error relative to path direction
			path_angle = cd.atan2(path_dy_normalized, path_dx_normalized)
			angle_error = haar_difference_without_abs(psi, path_angle)

			# Add terminal costs to symbolic expression
			terminal_cost = terminal_angle_weight * angle_error ** 2
			terminal_cost += terminal_contouring_multiplier * lag_weight * lag_error ** 2
			terminal_cost += terminal_contouring_multiplier * contour_weight * contour_error ** 2

			cost += terminal_cost

		LOG_DEBUG(f"Successfully created symbolic objective function for stage {stage_idx}")
		return cost

	def on_data_received(self, data, data_name):
		LOG_DEBUG("RECEIVED DATA FOR CONTOURING OBJ")
		if data_name == "reference_path":
			LOG_DEBUG("Received Reference Path")
			self.process_reference_path(data)

	def process_reference_path(self, data):
		LOG_DEBUG("Processing reference path for Contouring Objective")
		# Store the original path data

		self.reference_path = data.reference_path

		# Compute arc length if not provided
		if data.reference_path.s is None:
			self.reference_path.s = compute_arc_length(data.reference_path)

		# Create velocity reference spline if available
		if self.dynamic_velocity_reference and len(data.reference_path.v) > 0:
			self.reference_path.v_spline = CubicSpline(self.reference_path.s, data.reference_path.v)

		# Process road bounds if available
		if self.add_road_constraints and data.left_bound is not None and data.right_bound is not None:
			self.bound_left_spline = CubicSpline(self.reference_path.s,
												 np.column_stack((data.left_bound.x, data.left_bound.y)))
			self.bound_right_spline = CubicSpline(self.reference_path.s,
												  np.column_stack((data.right_bound.x, data.right_bound.y)))

	def _find_closest_point(self, position, reference_path: ReferencePath):
		"""Find the closest point on the path to the given position"""
		if reference_path.empty():
			return 0, 0

		# Extract position coordinates
		pos_x, pos_y = position

		# Compute squared distances to all path points
		dx = reference_path.x - pos_x
		dy =  reference_path.y  - pos_y
		distances_squared = dx ** 2 + dy ** 2

		# Find the index of the closest point
		closest_idx = np.argmin(distances_squared)

		# Determine the segment index based on the closest point
		segment_idx = 0
		if len(reference_path.s) > 1:
			segment_length = (reference_path.s[-1] - reference_path.s[0]) / self.num_segments
			if segment_length > 0:
				segment_idx = min(int((reference_path.s[closest_idx] - reference_path.s[0]) / segment_length), self.num_segments - 1)

		return closest_idx, segment_idx

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

		if self.reference_path is None:
			return False

		# Get the final point on the path
		final_x = self.reference_path.x[-1]
		final_y = self.reference_path.y[-1]
		final_point = np.array([final_x, final_y])

		# Check if we're close enough to the final point

		reached = distance(state.get_position(), final_point) < 1.0
		LOG_DEBUG(f"Objective reached: {reached}")
		return reached

	def construct_road_constraints(self, data):
		LOG_DEBUG("Constructing road constraints.")

		if (self.bound_left_spline is None or
				self.bound_right_spline is None):
			self.construct_road_constraints_from_centerline(data)
		else:
			self.construct_road_constraints_from_bounds(data)

	def construct_road_constraints_from_centerline(self, data):
		"""Construct road constraints based on centerline and width"""
		# If bounds are not supplied construct road constraints based on a set width
		if data.static_obstacles is None:
			data.static_obstacles = []

		# Get road width
		road_width_half = self.get_config_value("road.width") / 2.0

		for k in range(self.solver.horizon):
			data.static_obstacles[k] = StaticObstacle()

			# Get predicted spline parameter for this timestep
			norm_s = self.solver.get_ego_prediction(k, "spline")

			# Convert to actual arc length
			if len(self.reference_path.s) >= 2:
				s_min = self.reference_path.s[0]
				s_max = self.reference_path.s[-1]
				cur_s = s_min + norm_s * (s_max - s_min)
			else:
				continue

			# Get path point and derivatives
			path_point_x = self.reference_path.x(cur_s)
			path_point_y = self.reference_path.y(cur_s)
			path_point = np.array([path_point_x, path_point_y])

			# Get orthogonal vector (normal to path)
			path_dx = self.reference_path.x_spline.derivative()(cur_s)
			path_dy = self.reference_path.y_spline.derivative()(cur_s)

			# Normalize derivatives
			norm = safe_norm(path_dx, path_dy)

			path_dx_norm = path_dx / norm
			path_dy_norm = path_dy / norm

			# Create orthogonal vector (rotate 90 degrees)
			dpath = np.array([-path_dy_norm, path_dx_norm])  # Orthogonal to path direction

			# Adjust width based on road type
			if self.two_way_road:
				width_times = 3.0
			else:
				width_times = 1.0

			# Left halfspace constraint
			A = dpath
			boundary_left = path_point + A * (width_times * road_width_half - data.robot_area[0].radius)
			b = np.dot(A, boundary_left)
			data.static_obstacles[k].add_halfspace(A, b)

			# Right halfspace constraint
			boundary_right = path_point - A * (road_width_half - data.robot_area[0].radius)
			b = np.dot(-A, -boundary_right)

			data.static_obstacles[k].add_halfspace(-A, -b)

	def construct_road_constraints_from_bounds(self, data):
		"""Construct road constraints using actual road bounds"""
		if data.static_obstacles is None:
			data.set("static_obstacles", [None] * self.solver.horizon)

		LOG_DEBUG("Forecasting road from bounds")
		for k in range(self.solver.horizon):
			# Create a static obstacle for this time step
			data.static_obstacles[k] = StaticObstacle()

			# Get the current normalized position on the spline
			norm_s = self.solver.get_ego_prediction(k, "spline")

			# Convert to actual arc length
			if len(self.reference_path.s) >= 2:
				s_min = self.reference_path.s[0]
				s_max = self.reference_path.s[-1]
				cur_s = s_min + norm_s * (s_max - s_min)
				cur_s = float(cur_s)
			else:
				continue

			# Left bound point and orthogonal vector

			LOG_WARN(f"When forecasting from bounds, the left spline is {self.bound_left_spline(cur_s)} based on cur_s {cur_s}")
			left_x = self.bound_left_spline(cur_s)[0]
			left_y = self.bound_left_spline(cur_s)[1]
			left_point = np.array([left_x, left_y])

			left_dx = self.bound_left_spline.derivative()(cur_s)[0]
			left_dy = self.bound_left_spline.derivative()(cur_s)[1]

			# Normalize and create orthogonal vector
			norm_left = safe_norm(left_dx, left_dy)

			left_dx_norm = left_dx / norm_left
			left_dy_norm = left_dy / norm_left

			# Create orthogonal vector (rotate 90 degrees)
			left_ortho = np.array([-left_dy_norm, left_dx_norm])

			# Same for right bound
			right_x = self.bound_right_spline(cur_s)[0]
			right_y = self.bound_right_spline(cur_s)[1]
			right_point = np.array([right_x, right_y])

			right_dx = self.bound_right_spline.derivative()(cur_s)[0]
			right_dy = self.bound_right_spline.derivative()(cur_s)[1]

			norm_right = safe_norm(right_dx, right_dy)

			right_dx_norm = right_dx / norm_right
			right_dy_norm = right_dy / norm_right

			right_ortho = np.array([-right_dy_norm, right_dx_norm])
			# Left bound halfspace constraint
			Al = left_ortho
			bl = np.dot(Al, left_point + Al * data.robot_area[0].radius)

			data.static_obstacles[k].add_halfspace(-Al, -bl)

			# Right bound halfspace constraint
			Ar = right_ortho
			br = np.dot(Ar, right_point - Ar * data.robot_area[0].radius)
			LOG_DEBUG(f"Result of constructing road constraints from bounds: {left_ortho}, {right_ortho}, {Al}, {bl}, {Ar}, {br}")
			data.static_obstacles[k].add_halfspace(Ar, br)

	def _evaluate_spline_casadi(self, arc_progress, param_prefix, params):
		"""
		Evaluates a spline at the current arc length progress using CasADi.

		Parameters:
		-----------
		arc_progress : CasADi symbolic expression
			The normalized arc length parameter (0 to 1)
		param_prefix : str
			The prefix of the parameter name (e.g., "path_x", "path_y", "path_dx", "path_dy", "path_vel")
		params : ParameterManager
			The parameter manager containing the parameter values

		Returns:
		--------
		CasADi symbolic expression
			The interpolated value at the given arc length
		"""
		result = 0

		# Construct cubic spline interpolation in CasADi
		for i in range(self.num_segments):
			# Get segment boundaries
			t0 = i / self.num_segments
			t1 = (i + 1) / self.num_segments

			# Get parameter values at segment boundaries
			p0 = params.get(f"{param_prefix}_{i}")
			p1 = params.get(f"{param_prefix}_{i + 1}")

			# Linear interpolation within segment
			# Use conditional expression to check if arc_progress is in this segment

			if i == self.num_segments - 1:
				in_segment = cd.logic_and(arc_progress >= t0, arc_progress <= t1)
			else:
				in_segment = cd.logic_and(arc_progress >= t0, arc_progress < t1)

			# Normalize progress within segment
			t_norm = (arc_progress - t0) / (t1 - t0)

			# Linear interpolation formula: p0 * (1 - t_norm) + p1 * t_norm
			segment_value = p0 * (1 - t_norm) + p1 * t_norm

			# Add contribution only if in this segment
			result = result + in_segment * segment_value
		LOG_DEBUG(f"For arc progress {arc_progress}, param_prefix {param_prefix}, result is {result}")
		return result

	def reset(self):
		"""Reset the state of the contouring objective"""
		self.closest_segment = 0