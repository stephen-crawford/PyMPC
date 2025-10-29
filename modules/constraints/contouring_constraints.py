import numpy as np
import casadi as cd
from scipy.interpolate import CubicSpline

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planning.src.types import Data, ReferencePath
from utils.math_utils import haar_difference_without_abs, Spline2D, Spline
from utils.utils import LOG_DEBUG, LOG_INFO, LOG_WARN


class ContouringConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)

		self.name = "contouring_constraints"
		self.num_segments = self.get_config_value("contouring.num_segments")
		self.slack = self.get_config_value("contouring.slack")
		self.dynamic_velocity_reference = self.get_config_value("contouring.dynamic_velocity_reference")
		self.three_dimensional_contouring = self.get_config_value("contouring.three_dimensional_contouring")
		self.penalty_weight = self.get_config_value("contouring.penalty_weight")
		self.num_constraints = 2
		if self.three_dimensional_contouring:
			self.num_constraints += 2
		if self.dynamic_velocity_reference:
			self.num_constraints += 1

		# Road bounds
		self.reference_path = None
		self.bound_left_spline = None
		self.bound_right_spline = None
		self.bound_upper_spline = None
		self.bound_lower_spline = None
		self.bound_velocity_spline = None

		LOG_DEBUG(f"{self.name.title()} successfully initialized")
	
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

	def update(self, state, data: Data):
		LOG_DEBUG(f"{self.name.title()}::update")

		if self.reference_path is None:
			LOG_WARN("No reference path available")
			return

		# Pass path data to data object
		if self.reference_path is not None:
			data.reference_path = ReferencePath()
			data.reference_path.set('x', self.reference_path.x)
			data.reference_path.set('y', self.reference_path.y)
			data.reference_path.set('x_spline', self.reference_path.x_spline)
			data.reference_path.set('y_spline', self.reference_path.y_spline)
			data.reference_path.set('s', self.reference_path.s)

	def on_data_received(self, data):
		if data.has("reference_path") and data.reference_path is not None:
			LOG_DEBUG("Received Reference Path")
			self.process_reference_path(data)

	def process_reference_path(self, data):
		self.reference_path = data.reference_path
		# Create velocity reference spline if available
		if self.dynamic_velocity_reference and len(data.reference_path.v) > 0:
			self.reference_path.v_spline = CubicSpline(self.reference_path.s, data.reference_path.v)

		# Process road bounds if available
		if data.left_bound is not None and data.right_bound is not None:
			LOG_DEBUG("Processing provided left and right bounds for Contouring Constraints")
			self.bound_left_spline = CubicSpline(self.reference_path.s,
												 np.column_stack((data.left_bound.x, data.left_bound.y)))
			self.bound_right_spline = CubicSpline(self.reference_path.s,
												  np.column_stack((data.right_bound.x, data.right_bound.y)))
		if self.three_dimensional_contouring:
			if data.upper_bound is not None and data.lower_bound is not None:
				LOG_DEBUG("Processing provided lower and upper bounds for Contouring Constraints")
				self.bound_lower_spline = CubicSpline(self.reference_path.s,
													  np.column_stack((data.lower_bound.x, data.lower_bound.y)))
				self.bound_upper_spline = CubicSpline(self.reference_path.s,
													  np.column_stack((data.upper_bound.x, data.upper_bound.y)))

	def define_parameters(self, params):
		LOG_DEBUG(f"{self.name.title()}::define_parameters")


		for i in range(self.num_segments):
			params.add(f"path_{i}_start")
			# Path coordinates polynomial coefficients
			params.add(f"path_x_{i}_a")
			params.add(f"path_x_{i}_b")
			params.add(f"path_x_{i}_c")
			params.add(f"path_x_{i}_d")
			params.add(f"path_y_{i}_a")
			params.add(f"path_y_{i}_b")
			params.add(f"path_y_{i}_c")
			params.add(f"path_y_{i}_d")

			# Width polynomial coefficients
			params.add(f"width_left_{i}_a")
			params.add(f"width_left_{i}_b")
			params.add(f"width_left_{i}_c")
			params.add(f"width_left_{i}_d")
			params.add(f"width_right_{i}_a")
			params.add(f"width_right_{i}_b")
			params.add(f"width_right_{i}_c")
			params.add(f"width_right_{i}_d")

			if self.three_dimensional_contouring:
				params.add(f"path_z_{i}_a")
				params.add(f"path_z_{i}_b")
				params.add(f"path_z_{i}_c")
				params.add(f"path_z_{i}_d")
				params.add(f"path_dz_{i}_a")
				params.add(f"path_dz_{i}_b")
				params.add(f"path_dz_{i}_c")
				params.add(f"path_dz_{i}_d")
				params.add(f"width_lower_{i}_a")
				params.add(f"width_lower_{i}_b")
				params.add(f"width_lower_{i}_c")
				params.add(f"width_lower_{i}_d")
				params.add(f"width_upper_{i}_a")
				params.add(f"width_upper_{i}_b")
				params.add(f"width_upper_{i}_c")
				params.add(f"width_upper_{i}_d")

			# Velocity reference if needed
			if self.dynamic_velocity_reference:
				params.add(f"path_vel_{i}_a")
				params.add(f"path_vel_{i}_b")
				params.add(f"path_vel_{i}_c")
				params.add(f"path_vel_{i}_d")

	def set_parameters(self, parameter_manager, data, k):
		LOG_DEBUG(f"{self.name.title()}::set_parameters with k: {k}")

		if k == 0:
			self.process_reference_path(data)
			self.set_path_parameters(parameter_manager)

	def _compute_width_from_bounds(self, data):
		"""
		Compute the available width (safety margin) from the reference path to the bounds.
		Returns arrays of left and right widths at each reference path point.
		"""
		if self.reference_path is None or self.bound_left_spline is None or self.bound_right_spline is None:
			return None, None

		# Get reference path points
		path_x = self.reference_path.x
		path_y = self.reference_path.y

		# Evaluate bound splines at reference path s values
		left_bound_points = self.bound_left_spline(self.reference_path.s)
		right_bound_points = self.bound_right_spline(self.reference_path.s)

		# Calculate widths (distances from path to bounds)
		width_left = []
		width_right = []

		for i in range(len(path_x)):
			# Calculate distance from path point to left bound
			left_dist = np.sqrt((path_x[i] - left_bound_points[i, 0]) ** 2 +
								(path_y[i] - left_bound_points[i, 1]) ** 2)
			width_left.append(left_dist)

			# Calculate distance from path point to right bound
			right_dist = np.sqrt((path_x[i] - right_bound_points[i, 0]) ** 2 +
								 (path_y[i] - right_bound_points[i, 1]) ** 2)
			width_right.append(right_dist)
		return np.array(width_left), np.array(width_right)

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

		if self.three_dimensional_contouring:
			path_z = self.reference_path.z

		# Compute width parameters from bounds
		width_left_orig, width_right_orig = self._compute_width_from_bounds(None)

		# Prepare interpolated data for the number of segments
		if len(path_x) < self.num_segments + 1:
			# Get lists of evenly spaced partitions over the provided interval
			s_original = np.linspace(0, 1, len(path_x))
			s_new = np.linspace(0, 1, self.num_segments + 1)

			# Interpolate path coordinates
			path_x_interp = np.interp(s_new, s_original, path_x)
			path_y_interp = np.interp(s_new, s_original, path_y)

			# Interpolate width parameters
			if width_left_orig is not None and width_right_orig is not None:
				width_left_interp = np.interp(s_new, s_original, width_left_orig)
				width_right_interp = np.interp(s_new, s_original, width_right_orig)
			else:
				# Default widths if bounds not available
				width_left_interp = np.full(len(s_new), 3.0)
				width_right_interp = np.full(len(s_new), 3.0)

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

			# Interpolate width parameters
			if width_left_orig is not None and width_right_orig is not None:
				width_left_interp = np.interp(s_new, s, width_left_orig)
				width_right_interp = np.interp(s_new, s, width_right_orig)
			else:
				# Default widths if bounds not available
				width_left_interp = np.full(len(s_new), 3.0)
				width_right_interp = np.full(len(s_new), 3.0)

			if self.three_dimensional_contouring:
				path_z_interp = np.interp(s_new, s, path_z)

		# Fit cubic spline coefficients for each variable
		try:
			path_x_coeffs, x_starts = self._fit_cubic_spline_coefficients(s_new, path_x_interp)
			path_y_coeffs, y_starts = self._fit_cubic_spline_coefficients(s_new, path_y_interp)
			width_left_coeffs, _ = self._fit_cubic_spline_coefficients(s_new, width_left_interp)
			width_right_coeffs, _ = self._fit_cubic_spline_coefficients(s_new, width_right_interp)

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

			# Set width coefficients
			parameter_manager.set_parameter(f"width_left_{i}_a", width_left_coeffs[i][0])
			parameter_manager.set_parameter(f"width_left_{i}_b", width_left_coeffs[i][1])
			parameter_manager.set_parameter(f"width_left_{i}_c", width_left_coeffs[i][2])
			parameter_manager.set_parameter(f"width_left_{i}_d", width_left_coeffs[i][3])

			parameter_manager.set_parameter(f"width_right_{i}_a", width_right_coeffs[i][0])
			parameter_manager.set_parameter(f"width_right_{i}_b", width_right_coeffs[i][1])
			parameter_manager.set_parameter(f"width_right_{i}_c", width_right_coeffs[i][2])
			parameter_manager.set_parameter(f"width_right_{i}_d", width_right_coeffs[i][3])

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

	def get_lower_bound(self):
		lower_bound = [-np.inf, -np.inf]
		return lower_bound

	def get_upper_bound(self):
		upper_bound = [0., 0.]
		return upper_bound

	def get_constraints(self, symbolic_state, params, stage_idx):
		LOG_INFO(f"{self.name}::get_constraints")

		# Validate reference path is available
		if self.reference_path is None:
			LOG_WARN("ContouringConstraints: reference_path is not available. Returning empty constraints.")
			return []
		
		# Ensure reference_path.s is properly initialized
		if not hasattr(self.reference_path, 's') or self.reference_path.s is None or len(self.reference_path.s) == 0:
			LOG_WARN("ContouringConstraints: reference_path.s is not properly initialized. Returning empty constraints.")
			return []
		
		# Get arc length (ensure it's a scalar)
		s_min, s_max = self._get_arc_length_bounds()
		if s_max is None:
			LOG_WARN("ContouringConstraints: Cannot determine arc length. Returning empty constraints.")
			return []
		
		pos_x = symbolic_state.get("x")
		pos_y = symbolic_state.get("y")
		psi = symbolic_state.get("psi")
		spline_val = symbolic_state.get("spline")
		
		if spline_val is None:
			LOG_WARN("ContouringConstraints: spline state variable is None. Returning empty constraints.")
			return []
		
		s = spline_val / s_max

		# Create spline objects using the new spline classes
		path_spline = Spline2D(params, self.num_segments, s)
		width_left_spline = Spline(params, "width_left", self.num_segments, s)
		width_right_spline = Spline(params, "width_right", self.num_segments, s)

		# Get path coordinates and derivatives using spline classes
		path_x, path_y = path_spline.at(s)
		path_dx_normalized, path_dy_normalized = path_spline.deriv_normalized(s)

		# Calculate contouring error (lateral deviation from path)
		dx = pos_x - path_x
		dy = pos_y - path_y

		# Normal vector pointing left from path direction
		contour_error = path_dy_normalized * dx - path_dx_normalized * dy

		# Evaluate width splines at current position
		width_right = width_right_spline.at(s)
		width_left = width_left_spline.at(s)

		# Get vehicle parameters
		vehicle_width = symbolic_state.get("width")
		if vehicle_width is None:
			vehicle_width = 2.0

		lr = symbolic_state.get("lr")
		if lr is None:
			lr = 1.0

		# Calculate effective vehicle width considering orientation
		path_heading = cd.atan2(path_dy_normalized, path_dx_normalized)
		delta_psi = haar_difference_without_abs(psi, path_heading)
		w_cur = vehicle_width / 2. * cd.cos(delta_psi) + lr * cd.sin(cd.fabs(delta_psi))

		# Get slack parameter
		horizon_factor = 1.0 + (stage_idx * 0.1)  # Looser constraints further ahead
		adaptive_slack = self.slack * horizon_factor

		# Constraints to ensure vehicle stays within road boundaries
		c1 = contour_error + w_cur - width_right - adaptive_slack
		c2 = -contour_error + w_cur - width_left - adaptive_slack

		LOG_DEBUG(f"Contouring constraints - contour_error: {contour_error}, w_cur: {w_cur}")
		LOG_DEBUG(f"Width constraints - left: {width_left}, right: {width_right}")
		LOG_DEBUG(f"returning constraints: {[c1, c2]}")
		return [c1, c2]

	def get_penalty(self, symbolic_state, params, stage_idx):
		LOG_INFO(f"{self.name}::get_penalty")

		# Handle both State (from types.py) and DynamicsModel (from dynamic_models.py)
		if hasattr(symbolic_state, '_state_dict'):
			# State object from types.py
			pos_x = symbolic_state.get("x")
			pos_y = symbolic_state.get("y")
			spline_val = symbolic_state.get("spline")
		else:
			# DynamicsModel object - try to access via get() method
			try:
				pos_x = symbolic_state.get("x")
				pos_y = symbolic_state.get("y")
				spline_val = symbolic_state.get("spline")
			except (IOError, TypeError, AttributeError):
				# Fallback: return None if variables not available
				LOG_WARN(f"{self.name}::get_penalty: Could not access state variables, skipping penalty")
				return None
		
		# Validate reference path is available
		if self.reference_path is None:
			LOG_WARN(f"{self.name}::get_penalty: reference_path is not available")
			return None
		
		# If spline is not available, calculate from position
		if spline_val is None or (hasattr(spline_val, '__len__') and len(spline_val) == 0):
			LOG_WARN(f"{self.name}::get_penalty: Spline variable not available, using default")
			s = 0.0  # Use start of path as fallback
		else:
			# Get arc length (ensure it's a scalar)
			if isinstance(self.reference_path.s, (list, np.ndarray)) and len(self.reference_path.s) > 0:
				s_max = float(self.reference_path.s[-1])
			elif hasattr(self.reference_path, 'get_arc_length'):
				s_max = float(self.reference_path.get_arc_length())
			else:
				LOG_WARN(f"{self.name}::get_penalty: Cannot determine s_max, using default")
				s_max = 1.0
			s = spline_val / s_max

		# Create spline objects using the new spline classes
		path_spline = Spline2D(params, self.num_segments, s)

		# Get path coordinates and derivatives using spline classes
		path_x, path_y = path_spline.at(s)
		path_dx_normalized, path_dy_normalized = path_spline.deriv_normalized(s)

		# Calculate contouring error (lateral deviation from path)
		dx = pos_x - path_x
		dy = pos_y - path_y

		contour_error = path_dy_normalized * dx - path_dx_normalized * dy

		penalty = self.penalty_weight * cd.fabs(contour_error)
		return penalty

	def is_data_ready(self, data):
		required_fields = ["left_bound", "right_bound", "reference_path"]
		missing_data = ""
		for field in required_fields:
			if not data.has(field) or getattr(data, field) is None:
				missing_data += f"{field.replace('_', ' ').title()} "
		return len(missing_data) < 1

	def visualize(self, data):
		# Use the parent class method to check debug_visuals setting
		super().visualize(data)