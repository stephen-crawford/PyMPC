import numpy as np
import casadi as cd
from scipy.interpolate import CubicSpline

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planning.src.types import Data, ReferencePath
from utils.math_utils import haar_difference_without_abs
from utils.utils import LOG_DEBUG, LOG_INFO, LOG_WARN




class ContouringConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)

		self.name = "contouring_constraints"
		self.num_segments = self.get_config_value("contouring.num_segments")
		self.dynamic_velocity_reference = self.get_config_value("contouring.dynamic_velocity_reference")
		self.three_dimensional_contouring = self.get_config_value("contouring.three_dimensional_contouring")

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

	def on_data_received(self, data, data_name):
		LOG_DEBUG(f"{self.name.title()} on data received: {data_name}")
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
		for i in range(self.num_segments + 1):
			# Path coordinates at segment boundaries
			params.add(f"path_x_{i}")
			params.add(f"path_y_{i}")

			if self.three_dimensional_contouring:
				params.add(f"path_z_{i}")
				params.add(f"path_dz_{i}")

			# Derivatives at segment boundaries (needed for normal vectors)
			params.add(f"path_dx_{i}")
			params.add(f"path_dy_{i}")

			# Width parameters for safety margins
			params.add(f"width_left_{i}")
			params.add(f"width_right_{i}")

			# Velocity reference if needed
			if self.dynamic_velocity_reference:
				params.add(f"path_vel_{i}")

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

	def set_path_parameters(self, parameter_manager):
		LOG_DEBUG(f"{self.name.title()}::set_path_parameters")
		if self.reference_path is None:
			LOG_WARN("No reference path available when trying to set path params so returning ")
			return

		path_x = self.reference_path.x
		path_y = self.reference_path.y

		# Optional variables for three-dimensional contouring
		path_z = None
		path_z_interp = None
		path_dz = None

		if self.three_dimensional_contouring:
			path_z = self.reference_path.z

		# Compute width parameters from bounds
		width_left_orig, width_right_orig = self._compute_width_from_bounds(None)

		if len(path_x) < self.num_segments + 1:
			# Get lists of evenly spaced partitions over the provided interval
			s_original = np.linspace(0, 1, len(path_x))
			s_new = np.linspace(0, 1, self.num_segments + 1)

			# Interpolates at each s_new point based on the coordinates (s_new, path_x) for each i in S_new
			path_x_interp = np.interp(s_new, s_original, path_x)
			# Same type of interpolation but for points (s_new, path_y)
			path_y_interp = np.interp(s_new, s_original, path_y)

			# Interpolate width parameters
			if width_left_orig is not None and width_right_orig is not None:
				width_left_interp = np.interp(s_new, s_original, width_left_orig)
				width_right_interp = np.interp(s_new, s_original, width_right_orig)
			else:
				# Default widths if bounds not available
				width_left_interp = np.full(len(s_new), 3.0)  # Default 3m width
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
				width_left_interp = np.full(len(s_new), 3.0)  # Default 3m width
				width_right_interp = np.full(len(s_new), 3.0)

			if self.three_dimensional_contouring:
				path_z_interp = np.interp(s_new, s, path_z)

		# Calculate derivatives using finite differences
		path_dx = np.gradient(path_x_interp)
		path_dy = np.gradient(path_y_interp)

		if self.three_dimensional_contouring:
			path_dz = np.gradient(path_z_interp)

		# Set parameters
		for i in range(self.num_segments + 1):
			parameter_manager.set_parameter(f"path_x_{i}", float(path_x_interp[i]))
			parameter_manager.set_parameter(f"path_y_{i}", float(path_y_interp[i]))
			parameter_manager.set_parameter(f"path_dx_{i}", float(path_dx[i]))
			parameter_manager.set_parameter(f"path_dy_{i}", float(path_dy[i]))

			# Set width parameters
			parameter_manager.set_parameter(f"width_left_{i}", float(width_left_interp[i]))
			parameter_manager.set_parameter(f"width_right_{i}", float(width_right_interp[i]))

			if self.three_dimensional_contouring:
				parameter_manager.set_parameter(f"path_z_{i}", float(path_z_interp[i]))
				parameter_manager.set_parameter(f"path_dz_{i}", float(path_dz[i]))

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

	def get_lower_bound(self):
		lower_bound = [-np.inf, -np.inf]
		return lower_bound

	def get_upper_bound(self):
		upper_bound = [0., 0.]
		return upper_bound

	def get_constraints(self, symbolic_state, params, stage_idx):
		LOG_INFO(f"{self.name}::get_constraints")

		pos_x = symbolic_state.get("x")
		pos_y = symbolic_state.get("y")
		psi = symbolic_state.get("psi")
		s = symbolic_state.get("spline") / self.reference_path.s[-1]

		# Get path coordinates using the same method as ContouringObjective
		path_x = self._evaluate_spline_casadi(s, "path_x", params)
		path_y = self._evaluate_spline_casadi(s, "path_y", params)

		# Get path derivatives using the same method as ContouringObjective
		path_dx = self._evaluate_spline_casadi(s, "path_dx", params)
		path_dy = self._evaluate_spline_casadi(s, "path_dy", params)

		# Normalize the derivatives to get direction vector (same as ContouringObjective)
		norm = cd.sqrt(path_dx ** 2 + path_dy ** 2)
		norm_safe = cd.if_else(norm > 1e-6, norm, 1e-6)  # Prevent division by zero
		path_dx_normalized = path_dx / norm_safe
		path_dy_normalized = path_dy / norm_safe

		# Calculate contouring error (lateral deviation from path) - same as ContouringObjective
		# Position error vector
		dx = pos_x - path_x
		dy = pos_y - path_y

		# Normal vector pointing left from path direction: (-path_dy_normalized, path_dx_normalized)
		contour_error = path_dy_normalized * dx - path_dx_normalized * dy

		# Evaluate width splines at current position - NOW THESE WILL WORK!
		width_right = self._evaluate_spline_casadi(s, "width_right", params)
		width_left = self._evaluate_spline_casadi(s, "width_left", params)

		# Get vehicle width - fix symbolic state access
		vehicle_width = symbolic_state.get("width")  # Default width if not available
		if vehicle_width is None:
			vehicle_width = 2.0

		# Accurate width of the vehicle incorporating its orientation w.r.t. the path
		path_heading = cd.atan2(path_dy_normalized, path_dx_normalized)
		delta_psi = haar_difference_without_abs(psi, path_heading)  # Angle w.r.t. the path

		# Get vehicle parameters - fix symbolic state access
		lr = symbolic_state.get("lr")  # Default rear axle distance if not available
		if lr is None:
			lr = 1.0

		w_cur = vehicle_width / 2. * cd.cos(delta_psi) + lr * cd.sin(cd.fabs(delta_psi))

		# Get slack parameter
		slack = self.get_config_value("contouring.slack", 0.1)  # Default slack if not configured

		# Add constraints that ensure the vehicle stays within road boundaries
		# Constraint 1: Right boundary (contour_error + w_cur <= width_right + slack)
		# Rearranged: contour_error + w_cur - width_right - slack <= 0
		c1 = contour_error + w_cur - width_right - slack

		# Constraint 2: Left boundary (-contour_error + w_cur <= width_left + slack)
		# Rearranged: -contour_error + w_cur - width_left - slack <= 0
		c2 = -contour_error + w_cur - width_left - slack


		LOG_DEBUG(f"Contouring constraints - contour_error: {contour_error}, w_cur: {w_cur}")
		LOG_DEBUG(f"Width constraints - left: {width_left}, right: {width_right}")
		LOG_DEBUG(f"returning constraints: {[c1, c2]}")
		return [c1, c2]

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
