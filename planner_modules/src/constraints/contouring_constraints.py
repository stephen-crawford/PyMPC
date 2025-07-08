import numpy as np
import casadi as cd
from scipy.interpolate import CubicSpline

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planning.src.types import Data
from utils.math_utils import haar_difference_without_abs
from utils.utils import LOG_DEBUG


class ContouringConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)
		self.width_left_spline = None
		self.width_right_spline = None
		self.nh = 2
		self.name = "contouring_constraints"  # Override default name if needed
		self.num_segments = self.get_config_value("contouring.num_segments")
		# Store reference path data for future use
		self.reference_path_s = None
		self.reference_path_x = None
		self.reference_path_y = None
		LOG_DEBUG(f"{self.name.title()} Constraints successfully initialized")

	def update(self, state, data: Data):
		LOG_DEBUG(f"{self.name.title()} Update")
		# Use existing splines if they're available in data
		if data.get("path_width_left") is None and self.width_left_spline is not None:
			data.set("path_width_left", self.width_left_spline)

		if data.get("path_width_right") is None and self.width_right_spline is not None:
			data.set("path_width_right", self.width_right_spline)

	def on_data_received(self, data, data_name):
		LOG_DEBUG(f"{self.name.title()} Data received: {data_name}")
		if data_name == "reference_path":
			self.process_reference_path(data)

	# Used for testing
	def process_reference_path(self, data):
		if not data.left_bound is None and not data.right_bound is None:
			self.calculate_road_widths(data)
			# Store reference path data for spline creation
			self.reference_path_s = data.reference_path.s
			self.reference_path_x = data.reference_path.x
			self.reference_path_y = data.reference_path.y

	def calculate_road_widths(self, data: Data):
		widths_left = np.zeros(len(data.left_bound.x))
		widths_right = np.zeros(len(data.right_bound.x))
		LOG_DEBUG("DATA WHEN CALCULATING ROAD WIDTHS" + str(data))
		for i in range(len(widths_left)):
			center = np.array([data.reference_path.x[i], data.reference_path.y[i]])
			left = np.array([data.left_bound.x[i], data.left_bound.y[i]])
			right = np.array([data.right_bound.x[i], data.right_bound.y[i]])

			widths_left[i] = np.linalg.norm(center - left)
			widths_right[i] = np.linalg.norm(center - right)

		# Initialize splines using SciPy's CubicSpline
		self.width_left_spline = CubicSpline(data.reference_path.s, widths_left)
		self.width_right_spline = CubicSpline(data.reference_path.s, widths_right)

	def set_parameters(self, parameter_manager, data, k):
		if k == 1:
			LOG_DEBUG(f"{self.name}::set_parameters")

		# For SciPy CubicSpline, we only need to set parameters once when the splines are created
		# No need to set segment-by-segment parameters as they're internally managed by SciPy

	def define_parameters(self, params):
		LOG_DEBUG("Defining contouring constraint parameters")
		# With SciPy's CubicSpline, we don't need to manually define coefficients
		# Instead, we'll store pre-computed spline values at segment boundaries
		for segment_index in range(self.num_segments + 1):
			params.add(f"reference_path_x_{segment_index}")
			params.add(f"reference_path_y_{segment_index}")
			params.add(f"width_left_{segment_index}")
			params.add(f"width_right_{segment_index}")
		LOG_DEBUG("Params now, " + str(params))

	def get_lower_bound(self):
		lower_bound = [-np.inf, -np.inf]
		return lower_bound

	def get_upper_bound(self):
		upper_bound = [0., 0.]
		return upper_bound

	def get_constraints(self, model, params, settings, stage_idx):
		LOG_DEBUG(f"{self.name}::get_constraints")
		constraints = []
		pos_x = model.get("x")
		pos_y = model.get("y")
		s = model.get("path")
		self.precompute_parameter_values(params)

		# Get path coordinates using CasADi-compatible operations
		# We need to evaluate the path at the current arc length s
		# This would ideally use the parameter values we've set
		path_x = self._evaluate_spline_casadi(s, "reference_path_x", params)
		path_y = self._evaluate_spline_casadi(s, "reference_path_y", params)

		# Get path derivatives for normal vector calculation
		path_dx, path_dy = self._evaluate_spline_derivative_casadi(s, "reference_path_x", "reference_path_y", params)

		# Normalize the derivatives to get direction vector
		norm = cd.sqrt(path_dx ** 2 + path_dy ** 2)
		path_dx_normalized = path_dx / norm
		path_dy_normalized = path_dy / norm

		try:
			slack = model.get("slack")
		except:
			slack = 0.0

		try:
			psi = model.get("psi")
		except:
			psi = 0.0

		# Calculate contouring error (lateral deviation from path)
		contour_error = path_dy_normalized * (pos_x - path_x) - path_dx_normalized * (pos_y - path_y)

		# Evaluate width splines at current position
		width_right = self._evaluate_spline_casadi(s, "width_right", params)
		width_left = self._evaluate_spline_casadi(s, "width_left", params)

		# Accurate width of the vehicle incorporating its orientation w.r.t. the path
		delta_psi = haar_difference_without_abs(psi, cd.atan2(path_dy_normalized,
															  path_dx_normalized))  # Angle w.r.t. the path
		w_cur = model.width / 2. * cd.cos(delta_psi) + model.lr * cd.sin(cd.fabs(delta_psi))

		# Simpler alternative
		# w_cur = model.width / 2.

		# Add constraints that ensure the vehicle stays within road boundaries
		constraints.append(contour_error + w_cur - width_right - slack)
		constraints.append(-contour_error + w_cur - width_left - slack)  # -width_left because widths are positive

		# constraints.append(cd.fabs(delta_psi) - 0.35 * np.pi)
		LOG_DEBUG(f"returning constraints: {constraints}")
		return constraints
	#
	# def _evaluate_spline_casadi(self, s, param_name, params):
	# 	"""
    #     Evaluate spline at point s using CasADi-compatible operations
    #     This approximates the CubicSpline evaluation using linear interpolation between knot points
    #     """
	# 	# Create simple linear interpolation using the segment values
	# 	LOG_DEBUG(f"Trying to evaluate casadi spline for {param_name}, {params}")
	# 	result = 0
	# 	LOG_DEBUG("Total segments: " + str(self.num_segments))
	# 	for i in range(self.num_segments):
	# 		LOG_DEBUG("At segment: " + str(i))
	# 		# Get parameter values at segment boundaries
	# 		p0 = params.get(f"{param_name}_{i}")
	# 		LOG_DEBUG(f"Trying to get {param_name}_{i}")
	# 		LOG_DEBUG("p0 = " + str(p0))
	# 		p1 = params.get(f"{param_name}_{i + 1}")
	# 		LOG_DEBUG("p1 = " + str(p1))
	#
	# 		# Define segment boundaries (assuming uniform distribution)
	# 		s0 = i / self.num_segments
	# 		s1 = (i + 1) / self.num_segments
	#
	# 		# Linear interpolation within each segment using a smooth transition function
	# 		alpha = (s - s0) / (s1 - s0)
	# 		indicator = cd.logic_and(s >= s0, s < s1)
	# 		result += indicator * (p0 * (1 - alpha) + p1 * alpha)
	# 	LOG_DEBUG(f"Getting spline for casadi for parameter {param_name}_{self.num_segments}")
	# 	# Handle the case for s == 1.0
	# 	result += (s == 1.0) * params.get(f"{param_name}_{self.num_segments}")
	#
	# 	return result

	def _evaluate_spline_casadi(self, arc_progress, param_prefix, params):
		"""
		Evaluates a spline at the current arc length progress using CasADi.
		Uses piecewise linear interpolation between segment points.
		"""

		# Clamp arc_progress to [0, 1] to avoid extrapolation issues
		arc_progress = cd.fmax(0.0, cd.fmin(1.0, arc_progress))

		# Initialize result
		result = params.get(f"{param_prefix}_0")  # Default to first value

		# Piecewise linear interpolation
		for i in range(self.num_segments):
			# Segment boundaries in normalized coordinates
			t0 = i / self.num_segments
			t1 = (i + 1) / self.num_segments

			# Values at boundaries
			p0 = params.get(f"{param_prefix}_{i}")
			p1 = params.get(f"{param_prefix}_{i + 1}")

			# Check if arc_progress is in this segment
			in_segment = cd.logic_and(arc_progress >= t0, arc_progress < t1)

			# Handle last segment specially (include right endpoint)
			if i == self.num_segments - 1:
				in_segment = cd.logic_and(arc_progress >= t0, arc_progress <= t1)

			# Normalize progress within segment

			segment_width = t1 - t0
			t_norm = cd.if_else(segment_width > 1e-12, (arc_progress - t0) / segment_width, 0.0)

			# Linear interpolation
			segment_value = p0 + t_norm * (p1 - p0)

			# Update result if in this segment
			result = cd.if_else(in_segment, segment_value, result)

		return result
	def _evaluate_spline_derivative_casadi(self, s, param_x_name, param_y_name, params):
		"""
        Evaluate spline derivative at point s using CasADi-compatible operations
        Using simplified finite difference method for derivatives
        """
		# Create simple derivative approximation
		dx_result = 0
		dy_result = 0

		for i in range(self.num_segments):
			# Get parameter values at segment boundaries
			x0 = params.get(f"{param_x_name}_{i}")
			x1 = params.get(f"{param_x_name}_{i + 1}")
			y0 = params.get(f"{param_y_name}_{i}")
			y1 = params.get(f"{param_y_name}_{i + 1}")

			# Define segment boundaries (assuming uniform distribution)
			s0 = i / self.num_segments
			s1 = (i + 1) / self.num_segments

			# Simple derivative approximation (constant within each segment)
			dx = (x1 - x0) / (s1 - s0)
			dy = (y1 - y0) / (s1 - s0)

			# Use indicator function to select the right segment
			indicator = cd.logic_and(s >= s0, s < s1)
			dx_result += indicator * dx
			dy_result += indicator * dy

		# Handle the case for s == 1.0 (use the derivative from the last segment)
		last_dx = (params.get(f"{param_x_name}_{self.num_segments}") - params.get(
			f"{param_x_name}_{self.num_segments - 1}")) * self.num_segments
		last_dy = (params.get(f"{param_y_name}_{self.num_segments}") - params.get(
			f"{param_y_name}_{self.num_segments - 1}")) * self.num_segments

		dx_result += (s == 1.0) * last_dx
		dy_result += (s == 1.0) * last_dy

		return dx_result, dy_result

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
		if self.width_right_spline is None or self.width_left_spline is None or data.path is None:
			return

	def precompute_parameter_values(self, parameter_manager):
		"""
        Precompute all parameters needed for the optimization
        This should be called after splines are created but before optimization
        """
		LOG_DEBUG("Precomputing parameter values")
		if (self.reference_path_s is None or
				self.reference_path_x is None or
				self.reference_path_y is None):
			LOG_DEBUG("No reference path given")
			return
		if  (self.width_left_spline is None or
				self.width_right_spline is None):
			LOG_DEBUG("Cannot precompute parameters: missing width data")
			return

		# Calculate normalized arc length points for the segments
		s_values = np.linspace(0, 1, self.num_segments + 1)

		# Map normalized values to actual arc length values
		actual_s = self.reference_path_s[0] + s_values * (self.reference_path_s[-1] - self.reference_path_s[0])

		# Evaluate reference path at segment points
		reference_path_x_spline = CubicSpline(self.reference_path_s, self.reference_path_x)
		reference_path_y_spline = CubicSpline(self.reference_path_s, self.reference_path_y)

		ref_x_values = reference_path_x_spline(actual_s)
		ref_y_values = reference_path_y_spline(actual_s)
		width_left_values = self.width_left_spline(actual_s)
		width_right_values = self.width_right_spline(actual_s)

		# Set all parameters
		for i, s_val in enumerate(s_values):
			parameter_manager.set_parameter(f"reference_path_x_{i}", ref_x_values[i])
			parameter_manager.set_parameter(f"reference_path_y_{i}", ref_y_values[i])
			parameter_manager.set_parameter(f"width_left_{i}", width_left_values[i])
			parameter_manager.set_parameter(f"width_right_{i}", width_right_values[i])