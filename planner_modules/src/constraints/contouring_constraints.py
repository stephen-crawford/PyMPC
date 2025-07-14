import numpy as np
import casadi as cd
from scipy.interpolate import CubicSpline

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planning.src.types import Data
from utils.math_utils import haar_difference_without_abs
from utils.utils import LOG_DEBUG, LOG_INFO


class ContouringConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)
		self.width_left_spline = None
		self.width_right_spline = None
		self.nh = 2
		self.name = "contouring_constraints"
		self.num_segments = self.get_config_value("contouring.num_segments")
		# Store reference path data for future use
		self.reference_path = None
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

	def process_reference_path(self, data):
		if not data.left_bound is None and not data.right_bound is None:
			self.calculate_road_widths(data)
			# Store reference path data for spline creation
			self.reference_path = data.reference_path

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
		if k == 0:
			self.reference_path = data.reference_path
			if not data.left_bound is None and not data.right_bound is None:
				self.calculate_road_widths(data)
			LOG_DEBUG(f"{self.name}::set_parameters")
			# Precompute parameter values once when splines are ready
			self.precompute_parameter_values(parameter_manager)

	def define_parameters(self, params):
		LOG_DEBUG("Defining contouring constraint parameters")
		# Define parameters for pre-computed spline values at segment boundaries
		for segment_index in range(self.num_segments + 1):
			params.add(f"reference_path_x_{segment_index}")
			params.add(f"reference_path_y_{segment_index}")
			params.add(f"reference_path_dx_{segment_index}")
			params.add(f"reference_path_dy_{segment_index}")
			params.add(f"width_left_{segment_index}")
			params.add(f"width_right_{segment_index}")

	def get_lower_bound(self):
		lower_bound = [-np.inf, -np.inf]
		return lower_bound

	def get_upper_bound(self):
		upper_bound = [0., 0.]
		return upper_bound

	def get_constraints(self, symbolic_state, params, stage_idx):
		LOG_INFO(f"{self.name}::get_constraints")
		constraints = []

		pos_x = symbolic_state.get("x")
		pos_y = symbolic_state.get("y")
		psi = symbolic_state.get("psi")
		s = symbolic_state.get("spline") / self.reference_path.s[-1]

		# Get path coordinates using the same method as ContouringObjective
		path_x = self._evaluate_spline_casadi(s, "reference_path_x", params)
		path_y = self._evaluate_spline_casadi(s, "reference_path_y", params)

		# Get path derivatives using the same method as ContouringObjective
		path_dx = self._evaluate_spline_casadi(s, "reference_path_dx", params)
		path_dy = self._evaluate_spline_casadi(s, "reference_path_dy", params)

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

		# Evaluate width splines at current position
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
		constraints.append(contour_error + w_cur - width_right - slack)

		# Constraint 2: Left boundary (-contour_error + w_cur <= width_left + slack)
		# Rearranged: -contour_error + w_cur - width_left - slack <= 0
		constraints.append(-contour_error + w_cur - width_left - slack)

		LOG_DEBUG(f"Contouring constraints - contour_error: {contour_error}, w_cur: {w_cur}")
		LOG_DEBUG(f"Width constraints - left: {width_left}, right: {width_right}")
		LOG_DEBUG(f"returning constraints: {constraints}")
		return constraints

	def _evaluate_spline_casadi(self, arc_progress, param_prefix, params):
		"""
		Evaluates a spline at the current arc length progress using CasADi.
		Uses the same method as ContouringObjective for consistency.
		"""
		result = 0

		# Construct linear interpolation between segment points in CasADi
		for i in range(self.num_segments):
			# Get segment boundaries
			t0 = i / self.num_segments
			t1 = (i + 1) / self.num_segments

			# Get parameter values at segment boundaries
			p0 = params.get(f"{param_prefix}_{i}")
			p1 = params.get(f"{param_prefix}_{i + 1}")

			# Check if arc_progress is in this segment
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

		# Additional check for required data
		if self.width_right_spline is None or self.width_left_spline is None or data.reference_path is None:
			return

	def precompute_parameter_values(self, parameter_manager):
		"""
		Precompute all parameters needed for the optimization
		This should be called after splines are created but before optimization
		"""
		LOG_DEBUG("Precomputing parameter values")
		if (self.reference_path is None or
				self.reference_path.s is None or
				self.reference_path.x is None or
				self.reference_path.y is None):
			LOG_DEBUG("No reference path given")
			return
		if (self.width_left_spline is None or
				self.width_right_spline is None):
			LOG_DEBUG("Cannot precompute parameters: missing width data")
			return

		# Calculate normalized arc length points for the segments
		s_values = np.linspace(0, 1, self.num_segments + 1)

		# Map normalized values to actual arc length values
		actual_s = self.reference_path.s[0] + s_values * (self.reference_path.s[-1] - self.reference_path.s[0])

		# Evaluate reference path at segment points
		reference_path_x_spline = CubicSpline(self.reference_path.s, self.reference_path.x)
		reference_path_y_spline = CubicSpline(self.reference_path.s, self.reference_path.y)

		ref_x_values = reference_path_x_spline(actual_s)
		ref_y_values = reference_path_y_spline(actual_s)

		# Calculate derivatives using finite differences (same as ContouringObjective)
		ref_dx_values = np.gradient(ref_x_values)
		ref_dy_values = np.gradient(ref_y_values)

		width_left_values = self.width_left_spline(actual_s)
		width_right_values = self.width_right_spline(actual_s)

		# Set all parameters
		for i, s_val in enumerate(s_values):
			parameter_manager.set_parameter(f"reference_path_x_{i}", float(ref_x_values[i]))
			parameter_manager.set_parameter(f"reference_path_y_{i}", float(ref_y_values[i]))
			parameter_manager.set_parameter(f"reference_path_dx_{i}", float(ref_dx_values[i]))
			parameter_manager.set_parameter(f"reference_path_dy_{i}", float(ref_dy_values[i]))
			parameter_manager.set_parameter(f"width_left_{i}", float(width_left_values[i]))
			parameter_manager.set_parameter(f"width_right_{i}", float(width_right_values[i]))