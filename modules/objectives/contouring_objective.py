import numpy as np
import casadi as cd
from docutils.nodes import reference
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline

from planner_modules.src.objectives.base_objective import BaseObjective
from planning.src.types import StaticObstacle, ReferencePath
from utils.const import CONSTRAINT
from utils.math_utils import distance, haar_difference_without_abs, safe_norm, Spline, Spline2D
from utils.utils import LOG_DEBUG, LOG_INFO, LOG_WARN

class ContouringObjective(BaseObjective):
	def __init__(self, solver):

		super().__init__(solver)
		LOG_DEBUG("Contouring Objective initializing")
		
		# Enforce dependency: contouring objective requires contouring constraints
		self.dependencies = ["contouring_constraints"]

		# Configuration options from CONFIGs
		self.num_segments = self.get_config_value("contouring.num_segments")
		self.add_road_constraints = self.get_config_value("contouring.add_road_constraints")
		self.two_way_road = self.get_config_value("road.two_way")
		self.dynamic_velocity_reference = self.get_config_value("contouring.dynamic_velocity_reference")
		self.goal_reaching_contouring = self.get_config_value("contouring.goal_reaching_contouring")
		self.three_dimensional_contouring = self.get_config_value("contouring.three_dimensional_contouring")
		self.closest_point_idx = 0
		self.closest_segment = 0

		# Road bounds
		self.reference_path = None
		self.bound_left_spline = None
		self.bound_right_spline = None
		self.bound_upper_spline = None
		self.bound_lower_spline = None
		
		# Reference to contouring constraints module (for shared state)
		self._contouring_constraints = None

		LOG_DEBUG("Contouring module successfully initialized")
		
	def _get_arc_length_bounds(self):
		"""Safely get minimum and maximum arc length values from reference path."""
		if self.reference_path is None:
			return None, None
		if hasattr(self.reference_path, 'get_arc_length'):
			s_max = float(self.reference_path.get_arc_length())
			s_min = 0.0
			return s_min, s_max
		if not hasattr(self.reference_path, 's') or self.reference_path.s is None:
			return None, None
		if isinstance(self.reference_path.s, (list, np.ndarray)) and len(self.reference_path.s) > 0:
			s_min = float(self.reference_path.s[0])
			s_max = float(self.reference_path.s[-1])
			return s_min, s_max
		return None, None
		
	def _check_dependencies(self):
		"""Check that required dependencies are present in module manager."""
		if self.solver and hasattr(self.solver, 'module_manager'):
			constraint_modules = [m for m in self.solver.module_manager.get_modules() 
								if m.module_type == CONSTRAINT and m.name == "contouring_constraints"]
			if not constraint_modules:
				raise RuntimeError(
					"ContouringObjective requires ContouringConstraints to be present. "
					"Please add ContouringConstraints to the solver's module manager before adding ContouringObjective."
				)
			self._contouring_constraints = constraint_modules[0]
			LOG_DEBUG("ContouringObjective: Found required ContouringConstraints dependency")
		else:
			LOG_WARN("ContouringObjective: Cannot check dependencies - solver or module_manager not available")

	def update(self, state, data):
		LOG_INFO("ContouringObjective.update")

		if self.reference_path is None:
			LOG_WARN(f"{id(self)} No reference path available")
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
		closest_pt = np.array(
			[self.reference_path.x[self.closest_point_idx], self.reference_path.y[self.closest_point_idx]])
		vehicle_pos = np.array(state.get_position())
		LOG_DEBUG(f"Distance to closest point: {np.linalg.norm(closest_pt - vehicle_pos)}")

		if self.add_road_constraints:
			self.construct_road_constraints(data)

		# if self.get_config_value("plot.debug", True):
		# 	self.animate_forecasted_bounds(state, data)

	def define_parameters(self, params):
		"""Define all parameters used by this module"""
		# Check dependencies before defining parameters
		self._check_dependencies()
		LOG_DEBUG("Defining contouring objective parameters")
		# Core parameters
		params.add("contour_weight", add_to_rqt_reconfigure=True)
		params.add("contouring_lag_weight", add_to_rqt_reconfigure=True)

		# Velocity reference parameters if needed
		if self.dynamic_velocity_reference:
			params.add("contouring_reference_velocity_weight", add_to_rqt_reconfigure=True)

		if self.goal_reaching_contouring:
			params.add("contouring_goal_weight", add_to_rqt_reconfigure=True)
			params.add("contouring_goal_x")
			params.add("contouring_goal_y")
			if self.three_dimensional_contouring:
				params.add("contouring_goal_z")

		for i in range(self.num_segments + 1):
			params.add(f"path_{i}_start")

			params.add(f"path_x_{i}_a")
			params.add(f"path_x_{i}_b")
			params.add(f"path_x_{i}_c")
			params.add(f"path_x_{i}_d")
			params.add(f"path_y_{i}_a")
			params.add(f"path_y_{i}_b")
			params.add(f"path_y_{i}_c")
			params.add(f"path_y_{i}_d")


			if self.three_dimensional_contouring:
				params.add(f"path_z_{i}_a")
				params.add(f"path_z_{i}_b")
				params.add(f"path_z_{i}_c")
				params.add(f"path_z_{i}_d")

				params.add(f"path_dz_{i}_a")
				params.add(f"path_dz_{i}_b")
				params.add(f"path_dz_{i}_c")
				params.add(f"path_dz_{i}_d")


			# Derivatives at segment boundaries (needed for normal vectors)
			params.add(f"path_dx_{i}_a")
			params.add(f"path_dx_{i}_b")
			params.add(f"path_dx_{i}_c")
			params.add(f"path_dx_{i}_d")
			params.add(f"path_dy_{i}_a")
			params.add(f"path_dy_{i}_b")
			params.add(f"path_dy_{i}_c")
			params.add(f"path_dy_{i}_d")

			# Velocity reference if needed
			if self.dynamic_velocity_reference:
				params.add(f"path_vel_{i}_a")
				params.add(f"path_vel_{i}_b")
				params.add(f"path_vel_{i}_c")
				params.add(f"path_vel_{i}_d")



	def set_parameters(self, parameter_manager, data, k):
		#LOG_DEBUG(f"set_parameters called with k={k}")

		# Retrieve weights once
		if k == 0:

			contouring_weight = self.get_config_value("weights.contour_weight")
			lag_weight = self.get_config_value("weights.contouring_lag_weight")

			if self.goal_reaching_contouring:
				contouring_goal_weight = self.get_config_value("weights.contouring_goal_weight")
				parameter_manager.set_parameter("contouring_goal_weight", contouring_goal_weight)

			if self.dynamic_velocity_reference:
				velocity_weight = self.get_config_value("weights.contour_velocity_weight")
				parameter_manager.set_parameter("contouring_reference_velocity_weight", velocity_weight)

			parameter_manager.set_parameter("contour_weight", contouring_weight)
			parameter_manager.set_parameter("contouring_lag_weight", lag_weight)

		self.process_reference_path(data)
		self.set_path_parameters(parameter_manager)

	def _fit_cubic_spline_coefficients(self, x_data, y_data):
		"""
		Fit cubic spline coefficients for the given data points.
		Returns coefficients for each segment: [a, b, c, d] where:
		y = a*t^3 + b*t^2 + c*t + d, where t is normalized within each segment
		"""
		n = len(x_data)
		if n < 2:
			raise ValueError("Need at least 2 points for spline fitting")

		# Normalize x_data to [0, 1] range for each segment
		segments = []
		segment_starts = []

		for i in range(n - 1):
			# Create normalized parameter t within segment [0, 1]
			x_start = x_data[i]
			x_end = x_data[i + 1]
			y_start = y_data[i]
			y_end = y_data[i + 1]

			segment_starts.append(x_start)

			# For cubic spline, we need to estimate derivatives
			# Use finite differences for derivative estimation
			if i == 0:  # First segment
				if n > 2:
					dy_start = (y_data[i + 1] - y_data[i]) / (x_data[i + 1] - x_data[i])
				else:
					dy_start = 0
			else:
				dy_start = (y_data[i + 1] - y_data[i - 1]) / (x_data[i + 1] - x_data[i - 1])

			if i == n - 2:  # Last segment
				if n > 2:
					dy_end = (y_data[i + 1] - y_data[i]) / (x_data[i + 1] - x_data[i])
				else:
					dy_end = 0
			else:
				dy_end = (y_data[i + 2] - y_data[i]) / (x_data[i + 2] - x_data[i])

			# Scale derivatives by segment length
			dx = x_end - x_start
			dy_start_scaled = dy_start * dx
			dy_end_scaled = dy_end * dx

			# Solve for cubic coefficients: y = a*t^3 + b*t^2 + c*t + d
			# Conditions: y(0) = y_start, y(1) = y_end, y'(0) = dy_start_scaled, y'(1) = dy_end_scaled
			d = y_start
			c = dy_start_scaled
			a = 2 * y_start - 2 * y_end + dy_start_scaled + dy_end_scaled
			b = -3 * y_start + 3 * y_end - 2 * dy_start_scaled - dy_end_scaled

			segments.append([float(a), float(b), float(c), float(d)])

		return segments, segment_starts

	def set_path_parameters(self, parameter_manager):
		LOG_DEBUG(f"{self.name.title()}::set_path_parameters")
		if self.reference_path is None:
			LOG_WARN("No reference path available when trying to set path params so returning")
			return

		path_x = self.reference_path.x
		path_y = self.reference_path.y

		# Optional variables for three-dimensional contouring
		path_z = None
		path_z_interp = None
		path_z_coeffs = None

		if self.three_dimensional_contouring:
			path_z = self.reference_path.z


		# Prepare interpolated data for the number of segments
		if len(path_x) < self.num_segments + 1:
			# Get lists of evenly spaced partitions over the provided interval
			s_original = np.linspace(0, 1, len(path_x))
			s_new = np.linspace(0, 1, self.num_segments + 1)

			# Interpolate path coordinates
			path_x_interp = np.interp(s_new, s_original, path_x)
			path_y_interp = np.interp(s_new, s_original, path_y)


			if self.three_dimensional_contouring:
				path_z_interp = np.interp(s_new, s_original, path_z)
		else:
			# Compute cumulative arc length
			dx = np.diff(path_x)
			dy = np.diff(path_y)

			arc_lengths = np.sqrt(dx ** 2 + dy ** 2)
			if self.three_dimensional_contouring:
				dz = np.diff(path_z)
				arc_lengths = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

			s = np.concatenate([[0], np.cumsum(arc_lengths)])
			s /= s[-1]  # Normalize to [0, 1]

			# Resample at evenly spaced points in [0, 1]
			s_new = np.linspace(0, 1, self.num_segments + 1)

			path_x_interp = np.interp(s_new, s, path_x)
			path_y_interp = np.interp(s_new, s, path_y)

			if self.three_dimensional_contouring:
				path_z_interp = np.interp(s_new, s, path_z)

		# Fit cubic spline coefficients for each variable
		try:
			path_x_coeffs, x_starts = self._fit_cubic_spline_coefficients(s_new, path_x_interp)
			path_y_coeffs, y_starts = self._fit_cubic_spline_coefficients(s_new, path_y_interp)

			if self.three_dimensional_contouring:
				path_z_coeffs, _ = self._fit_cubic_spline_coefficients(s_new, path_z_interp)

		except ValueError as e:
			LOG_WARN(f"Error fitting spline coefficients: {e}")
			return

		# Set parameters for each segment
		for i in range(self.num_segments):
			# Set segment start parameter
			parameter_manager.set_parameter(f"path_{i}_start", float(x_starts[i]))

			# Set path coordinate coefficients
			parameter_manager.set_parameter(f"path_x_{i}_a", path_x_coeffs[i][0])
			parameter_manager.set_parameter(f"path_x_{i}_b", path_x_coeffs[i][1])
			parameter_manager.set_parameter(f"path_x_{i}_c", path_x_coeffs[i][2])
			parameter_manager.set_parameter(f"path_x_{i}_d", path_x_coeffs[i][3])

			parameter_manager.set_parameter(f"path_y_{i}_a", path_y_coeffs[i][0])
			parameter_manager.set_parameter(f"path_y_{i}_b", path_y_coeffs[i][1])
			parameter_manager.set_parameter(f"path_y_{i}_c", path_y_coeffs[i][2])
			parameter_manager.set_parameter(f"path_y_{i}_d", path_y_coeffs[i][3])


			if self.three_dimensional_contouring:
				parameter_manager.set_parameter(f"path_z_{i}_a", path_z_coeffs[i][0])
				parameter_manager.set_parameter(f"path_z_{i}_b", path_z_coeffs[i][1])
				parameter_manager.set_parameter(f"path_z_{i}_c", path_z_coeffs[i][2])
				parameter_manager.set_parameter(f"path_z_{i}_d", path_z_coeffs[i][3])

			# Set velocity reference if using dynamic velocity
			if self.dynamic_velocity_reference and hasattr(self.reference_path, 'v') and self.reference_path.v is not None:
				if hasattr(self.reference_path.v, '__call__'):  # It's a function/spline
					s_val = i / self.num_segments if self.num_segments > 0 else 0
					s_min, s_max = self._get_arc_length_bounds()
					if s_min is not None and s_max is not None:
						actual_s = np.interp(s_val, [0, 1], [s_min, s_max])
						vel_val = self.reference_path.v(actual_s)
					else:
						vel_val = self.reference_path.v[min(i, len(self.reference_path.v) - 1)] if isinstance(self.reference_path.v, (list, np.ndarray)) else 1.0
				else:  # It's an array
					vel_val = self.reference_path.v[min(i, len(self.reference_path.v) - 1)] if isinstance(self.reference_path.v, (list, np.ndarray)) else 1.0

				# For now, set velocity as constant (could be extended to fit spline)
				parameter_manager.set_parameter(f"path_vel_{i}_a", 0.0)
				parameter_manager.set_parameter(f"path_vel_{i}_b", 0.0)
				parameter_manager.set_parameter(f"path_vel_{i}_c", 0.0)
				parameter_manager.set_parameter(f"path_vel_{i}_d", float(vel_val))
			#LOG_INFO("Finished setting contouring objective parameters")

	def get_value(self, symbolic_state, params, stage_idx):
		if params is None:
			params = self.solver.parameter_manager

		# Validate reference path is available
		if self.reference_path is None:
			# Try to get reference path from contouring constraints module
			if self._contouring_constraints and self._contouring_constraints.reference_path:
				self.reference_path = self._contouring_constraints.reference_path
			else:
				raise ValueError("ContouringObjective: reference_path is not available. Ensure on_data_received() was called with a valid reference_path.")
		
		# Ensure reference_path.s is properly initialized
		if not hasattr(self.reference_path, 's') or self.reference_path.s is None or len(self.reference_path.s) == 0:
			raise ValueError("ContouringObjective: reference_path.s is not properly initialized.")
		
		# Get arc length (ensure it's a scalar)
		s_min, s_max = self._get_arc_length_bounds()
		if s_max is None:
			raise ValueError("ContouringObjective: Cannot determine arc length from reference_path.")
		
		# Get symbolic state variables
		pos_x = symbolic_state.get("x")
		pos_y = symbolic_state.get("y")
		psi = symbolic_state.get("psi")
		v = symbolic_state.get("v")
		spline_val = symbolic_state.get("spline")
		
		if spline_val is None:
			raise ValueError("ContouringObjective: spline state variable is None. Ensure vehicle dynamics model includes 'spline' state.")
		
		s = spline_val / s_max


		# Validate that we have all required variables
		if any(var is None for var in [pos_x, pos_y, psi, v, s]):
			missing_vars = [name for name, var in
							[("x", pos_x), ("y", pos_y), ("psi", psi), ("v", v), ("spline", s)] if var is None]
			raise ValueError(f"Missing required symbolic variables: {missing_vars}")

		# Get symbolic weights from parameters
		contour_weight = params.get("contour_weight")
		lag_weight = params.get("contouring_lag_weight")
		velocity_weight = None

		goal_weight = 0
		remaining_distance = 0
		if self.goal_reaching_contouring:
			goal_weight = params.get("contouring_goal_weight")
			remaining_distance = s_max - s * s_max
			LOG_DEBUG("Remaining distance: " + str(remaining_distance))

		# From path
		reference_velocity = None
		if self.dynamic_velocity_reference:
			if not params.has_parameter("spline_v0_a"):
				raise IOError(
					"contouring/dynamic_velocity_reference is enabled, but there is no PathReferenceVelocity module.")

			path_velocity = Spline(params, "spline_v", self.num_segments, s)
			reference_velocity = path_velocity.at(s)
			velocity_weight = params.get("velocity")

		path = Spline2D(params, self.num_segments, s)
		path_x, path_y = path.at(s)
		path_dx_normalized, path_dy_normalized = path.deriv_normalized(s)

		# MPCC
		contour_error = path_dy_normalized * (pos_x - path_x) - path_dx_normalized * (pos_y - path_y)
		lag_error = path_dx_normalized * (pos_x - path_x) + path_dy_normalized * (pos_y - path_y)

		# Cost components
		lag_cost = lag_weight * lag_error ** 2
		contour_cost = contour_weight * contour_error ** 2
		goal_cost = goal_weight * remaining_distance ** 2

		# Velocity cost (if enabled)
		velocity_cost = 0
		if self.dynamic_velocity_reference:

			velocity_cost = velocity_weight * (v - reference_velocity) ** 2

		terminal_cost = 0
		if self.goal_reaching_contouring and stage_idx == self.solver.horizon - 1:

			terminal_angle_weight = self.get_config_value("contouring.terminal_angle")
			terminal_contouring_mp = self.get_config_value("contouring.terminal_contouring")

			# Compute the angle w.r.t. the path
			path_angle = cd.atan2(path_dy_normalized, path_dx_normalized)
			angle_error = haar_difference_without_abs(psi, path_angle)
			terminal_cost = 0
			# Penalize the angle error
			terminal_cost += terminal_angle_weight * angle_error ** 2
			terminal_cost += terminal_contouring_mp * lag_weight * lag_error ** 2
			terminal_cost += terminal_contouring_mp * contour_weight * contour_error ** 2

		return {
			"contouring_lag_cost": lag_cost,
			"contouring_contour_cost": contour_cost,
			"contouring_velocity_cost": velocity_cost,
			"contouring_goal_cost": goal_cost,
			"contouring_terminal_cost": terminal_cost
		}

	def on_data_received(self, data):
		LOG_DEBUG("RECEIVED DATA FOR CONTOURING OBJ")
		if data.has("reference_path") and data.reference_path is not None:
			LOG_DEBUG("Received Reference Path")
			self.process_reference_path(data)
			# Share reference path with constraint module if available
			if self._contouring_constraints and not self._contouring_constraints.reference_path:
				self._contouring_constraints.reference_path = self.reference_path
				self._contouring_constraints.process_reference_path(data)

	def process_reference_path(self, data):
		LOG_DEBUG("Processing reference path for Contouring Objective")
		# Store the original path data

		self.reference_path = data.reference_path

		# Create velocity reference spline if available
		if self.dynamic_velocity_reference and len(data.reference_path.v) > 0:
			self.reference_path.v_spline = CubicSpline(self.reference_path.s, data.reference_path.v)

		# Process road bounds if available
		if data.left_bound is not None and data.right_bound is not None:
			LOG_DEBUG("Processing provided left and right bounds for Contouring Objective")
			self.bound_left_spline = CubicSpline(self.reference_path.s,
												 np.column_stack((data.left_bound.x, data.left_bound.y)))
			self.bound_right_spline = CubicSpline(self.reference_path.s,
												  np.column_stack((data.right_bound.x, data.right_bound.y)))
		if self.three_dimensional_contouring:
			if data.upper_bound is not None and data.lower_bound is not None:
				LOG_DEBUG("Processing provided lower and upper bounds for Contouring Objective")
				self.bound_lower_spline = CubicSpline(self.reference_path.s,
													  np.column_stack((data.lower_bound.x, data.lower_bound.y)))
				self.bound_upper_spline = CubicSpline(self.reference_path.s,
													  np.column_stack((data.upper_bound.x, data.upper_bound.y)))


	def _find_closest_point(self, position, reference_path: ReferencePath):
		"""Find the closest point on the path to the given position"""
		if reference_path.empty():
			return 0, 0

		pos_x, pos_y = position

		# Compute squared distances to all path points
		dx = reference_path.x - pos_x
		dy = reference_path.y - pos_y
		distances_squared = dx ** 2 + dy ** 2

		closest_idx = np.argmin(distances_squared)

		# FIX: Better segment calculation based on actual arc length
		segment_idx = 0
		if len(reference_path.s) > 1 and self.num_segments > 0:
			# Get the arc length at the closest point
			closest_s = reference_path.s[closest_idx]
			s_min = reference_path.s[0]
			s_max = reference_path.s[-1]

			# Normalize to [0, 1] and find segment
			norm_s = (closest_s - s_min) / (s_max - s_min)
			segment_idx = min(int(norm_s * self.num_segments), self.num_segments - 1)

		LOG_DEBUG(f"Found closest point at {closest_idx} and closest segment: {segment_idx}")
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
		final_x = self.reference_path.x[-1] if isinstance(self.reference_path.x, (list, np.ndarray)) else self.reference_path.x
		final_y = self.reference_path.y[-1] if isinstance(self.reference_path.y, (list, np.ndarray)) else self.reference_path.y
		final_point = np.array([float(final_x), float(final_y)])

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

		data.set("static_obstacles", [None] * self.solver.horizon + 1)

		# Get road width
		road_width_half = self.get_config_value("road.width") / 2.0

		for k in range(self.solver.horizon + 1):
			data.static_obstacles[k] = StaticObstacle()

			# Get predicted spline parameter for this timestep
			norm_s = self.solver.get_initial_state().get("spline") / self.reference_path.get_arc_length()

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
		# Ensure static_obstacles exists
		if not hasattr(data, "static_obstacles") or data.static_obstacles is None:
			data.set("static_obstacles", [])

		# Ensure we have enough slots for all horizon steps
		required_length = self.solver.horizon + 1
		current_length = len(data.static_obstacles)

		if current_length < required_length:
			data.static_obstacles.extend([None] * (required_length - current_length))

		# Get current vehicle progress
		current_norm_s = self.solver.get_initial_state().get("spline") / self.reference_path.get_arc_length()

		# Estimate vehicle velocity to predict future positions
		vehicle_velocity = self.solver.get_initial_state().get("v")  # Default to 1.0 if not available
		dt = self.solver.dt if hasattr(self.solver, 'dt') else 0.1  # Time step
		LOG_DEBUG("Static obstacles: {}".format(data.static_obstacles))

		# Create obstacles for all horizon steps (not just the new ones)
		for k in range(self.solver.horizon + 1):
			# Create a static obstacle for this time step
			data.static_obstacles[k] = StaticObstacle()

			# Initialize prediction structure for static obstacles
			if not hasattr(data.static_obstacles[k], 'prediction'):
				# Create a basic prediction object
				data.static_obstacles[k].prediction = type('Prediction', (), {
					'steps': [],
					'type': None,  # or whatever default prediction type you use
					'path': None
				})()

			# Since this is a road boundary (static), set a dummy position
			# You might want to use the center of the road or some representative point
			if not hasattr(data.static_obstacles[k], 'position'):
				# Use path center as representative position
				future_distance = vehicle_velocity * dt * k
				future_norm_s = current_norm_s + (future_distance / self.reference_path.get_arc_length())
				future_norm_s = max(0.0, min(1.0, future_norm_s))

				if len(self.reference_path.s) >= 2:
					s_min = self.reference_path.s[0]
					s_max = self.reference_path.s[-1]
					cur_s = s_min + future_norm_s * (s_max - s_min)
					path_point_x = float(self.reference_path.x_spline(cur_s))
					path_point_y = float(self.reference_path.y_spline(cur_s))
					data.static_obstacles[k].position = np.array([path_point_x, path_point_y, 0.0])
				else:
					data.static_obstacles[k].position = np.array([0.0, 0.0, 0.0])

			# Rest of your existing code for creating halfspace constraints...
			# Project future position along path
			future_distance = vehicle_velocity * dt * k
			future_norm_s = current_norm_s + (future_distance / self.reference_path.get_arc_length())

			# Clamp to valid range [0, 1]
			future_norm_s = max(0.0, min(1.0, future_norm_s))

			# Convert to actual arc length
			if len(self.reference_path.s) >= 2:
				s_min = self.reference_path.s[0]
				s_max = self.reference_path.s[-1]
				cur_s = s_min + future_norm_s * (s_max - s_min)
				cur_s = float(cur_s)
			else:
				continue

			# Get centerline path point and its tangent
			path_point_x = float(self.reference_path.x_spline(cur_s))
			path_point_y = float(self.reference_path.y_spline(cur_s))

			path_dx = float(self.reference_path.x_spline.derivative()(cur_s))
			path_dy = float(self.reference_path.y_spline.derivative()(cur_s))

			# Normalize path tangent
			path_norm = safe_norm(path_dx, path_dy)
			path_dx_norm = path_dx / path_norm
			path_dy_norm = path_dy / path_norm

			# Create consistent normal vector (pointing left from path direction)
			path_normal = np.array([-path_dy_norm, path_dx_norm])

			# Get left and right bound points
			left_bounds = self.bound_left_spline(cur_s)
			right_bounds = self.bound_right_spline(cur_s)

			left_point = np.array([float(left_bounds[0]), float(left_bounds[1])])
			right_point = np.array([float(right_bounds[0]), float(right_bounds[1])])

			# Determine which side is actually left/right by checking cross product
			center_point = np.array([path_point_x, path_point_y])
			center_to_left = left_point - center_point
			center_to_right = right_point - center_point

			# Cross product to determine orientation
			left_cross = np.cross(np.array([path_dx_norm, path_dy_norm]), center_to_left)
			right_cross = np.cross(np.array([path_dx_norm, path_dy_norm]), center_to_right)

			# Ensure correct assignment (left should have positive cross product)
			if left_cross < 0:
				left_point, right_point = right_point, left_point
				LOG_DEBUG("Swapped left and right bounds based on cross product")

			# Get robot radius for proper constraint placement
			robot_radius = data.robot_area[0].radius if hasattr(data, 'robot_area') and len(
				data.robot_area) > 0 else 0.5

			# Create halfspace constraints with proper robot radius consideration
			# Left bound: normal points inward (toward the road)
			left_normal = -path_normal  # Points right (inward from left bound)
			# Move constraint inward by robot radius
			left_constraint_point = left_point + left_normal * .5 * robot_radius
			bl = float(np.dot(left_normal, left_constraint_point))
			data.static_obstacles[k].add_halfspace(left_normal, bl)

			# Right bound: normal points inward (toward the road)
			right_normal = path_normal  # Points left (inward from right bound)
			# Move constraint inward by robot radius
			right_constraint_point = right_point + right_normal * .5 * robot_radius
			br = float(np.dot(right_normal, right_constraint_point))
			data.static_obstacles[k].add_halfspace(right_normal, br)

			LOG_DEBUG(f"Time step {k}: Future norm_s={float(future_norm_s):.3f}, cur_s={float(cur_s):.3f}")
			LOG_DEBUG(f"Left constraint: normal={left_normal}, b={float(bl)}")
			LOG_DEBUG(f"Right constraint: normal={right_normal}, b={float(br)}")
			LOG_DEBUG(f"Left point: {left_point}, Right point: {right_point}")

			# Additional safety check: ensure constraints are not too restrictive
			# Check that the centerline is feasible
			center_feasible_left = np.dot(left_normal, center_point) <= bl + 1e-6
			center_feasible_right = np.dot(right_normal, center_point) <= br + 1e-6

			if not center_feasible_left or not center_feasible_right:
				LOG_WARN(
					f"Centerline infeasible at time step {k}! Left: {center_feasible_left}, Right: {center_feasible_right}")

	def reset(self):
		"""Reset the state of the contouring objective"""
		self.closest_segment = 0

	def animate_forecasted_bounds(self, state, data):
		if self.reference_path is None:
			print("No reference path - cannot animate")
			return

		fig, ax = plt.subplots()
		ax.plot(self.reference_path.x, self.reference_path.y, label="Reference Path", linewidth=2)

		# Plot static bounds if available
		if self.bound_left_spline is not None and self.bound_right_spline is not None:
			s_vals = np.linspace(self.reference_path.s[0], self.reference_path.s[-1], 200)
			left_pts = self.bound_left_spline(s_vals)
			right_pts = self.bound_right_spline(s_vals)
			ax.plot(left_pts[:, 0], left_pts[:, 1], '--', color='blue', label='Left Bound')
			ax.plot(right_pts[:, 0], right_pts[:, 1], '--', color='green', label='Right Bound')

		ax.set_aspect('equal')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.grid(True)
		ax.legend()

		dynamic_artists = []

		def animate(k):
			# Clear previous dynamic artists
			nonlocal dynamic_artists
			for artist in dynamic_artists:
				artist.remove()
			dynamic_artists = []

			# Vehicle position at step k
			x = self.solver.get_initial_state().get("x")
			y = self.solver.get_initial_state().get("y")
			pos = [x, y]

			pos_x, pos_y = float(pos[0]), float(pos[1])

			# Plot vehicle position
			vehicle_dot, = ax.plot(pos_x, pos_y, 'ro', label="Vehicle")
			dynamic_artists.append(vehicle_dot)

			# Closest point on path for vehicle at this frame
			closest_idx, closest_segment = self._find_closest_point((pos_x, pos_y), self.reference_path)

			path_x = self.reference_path.x
			path_y = self.reference_path.y

			# Plot closest point
			closest_pt_plot, = ax.plot(path_x[closest_idx], path_y[closest_idx], 'bx', markersize=10,
									   label="Closest Point")
			dynamic_artists.append(closest_pt_plot)

			segment_text = ax.text(path_x[closest_idx], path_y[closest_idx], f"seg {closest_segment}", fontsize=9,
								   color="purple")
			dynamic_artists.append(segment_text)

			# Plot forecasted bounds from static obstacles
			if hasattr(data, "static_obstacles") and k < len(data.static_obstacles):
				obstacle = data.static_obstacles[k]
				if obstacle is not None and hasattr(obstacle, "halfspaces"):
					for halfspace in obstacle.halfspaces:
						A = np.array(halfspace.A).flatten()
						b = float(halfspace.b)

						norm = np.linalg.norm(A)
						if norm < 1e-8:
							continue
						A = A / norm
						b = b / norm

						s = self.solver.get_initial_state().get("spline")  # Usually in [0, 1]
						norm_s = s / self.reference_path.get_arc_length()
						cur_s = self.reference_path.s[0] + norm_s * (
									self.reference_path.s[-1] - self.reference_path.s[0])
						path_x_halfspace = float(self.reference_path.x_spline(cur_s))
						path_y_halfspace = float(self.reference_path.y_spline(cur_s))
						path_point = np.array([path_x_halfspace, path_y_halfspace])

						point_on_line = path_point + A * (b - np.dot(A, path_point))
						dir_vector = np.array([-A[1], A[0]])
						t = 5.0
						line_start = point_on_line - t * dir_vector
						line_end = point_on_line + t * dir_vector

						# Plot halfspace line
						line, = ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], color='orange',
										alpha=0.5)
						dynamic_artists.append(line)

						# Plot normal arrow into allowed halfspace
						arrow = ax.annotate('', xy=point_on_line + A * 1.0, xytext=point_on_line,
											arrowprops=dict(facecolor='green', edgecolor='green', width=1.5,
															headwidth=6))
						dynamic_artists.append(arrow)

						# Compute lag and contour errors arrows
						dx = pos_x - float(path_x[closest_idx])
						dy = pos_y - float(path_y[closest_idx])

						s_val = float(self.reference_path.s[closest_idx])
						path_dx = float(self.reference_path.x_spline.derivative()(s_val))
						path_dy = float(self.reference_path.y_spline.derivative()(s_val))
						tangent_norm = np.linalg.norm([path_dx, path_dy]) + 1e-6
						path_dx /= tangent_norm
						path_dy /= tangent_norm

						normal_x = -path_dy
						normal_y = path_dx

						lag_error = dx * path_dx + dy * path_dy
						contour_error = dx * normal_x + dy * normal_y

						arrow_start = np.array([float(path_x[closest_idx]), float(path_y[closest_idx])])
						arrow_end_lag = arrow_start + lag_error * np.array([path_dx, path_dy])
						lag_arrow = ax.annotate('Lag', xy=arrow_end_lag, xytext=arrow_start,
												arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=6))
						dynamic_artists.append(lag_arrow)

						arrow_end_contour = arrow_start + contour_error * np.array([normal_x, normal_y])
						contour_arrow = ax.annotate('Contour', xy=arrow_end_contour, xytext=arrow_start,
													arrowprops=dict(facecolor='red', shrink=0.05, width=1.5,
																	headwidth=6))
						dynamic_artists.append(contour_arrow)

						# Display lag and contour errors near vehicle
						error_text = ax.text(pos_x + 1, pos_y - 1,
											 f"Lag: {lag_error:.2f}\nContour: {contour_error:.2f}",
											 fontsize=8, color='black', bbox=dict(facecolor='white', alpha=0.7))
						dynamic_artists.append(error_text)

			return dynamic_artists

		ani = FuncAnimation(fig, animate, frames=range(self.solver.horizon), interval=100, blit=False)
		plt.show()



