import numpy as np
import casadi as cd
from docutils.nodes import reference
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline

from modules.objectives.base_objective import BaseObjective
from planning.types import StaticObstacle, ReferencePath
from utils.const import CONSTRAINT
from utils.math_tools import distance, haar_difference_without_abs, safe_norm, Spline, Spline2D
from utils.utils import LOG_DEBUG, LOG_INFO, LOG_WARN

class ContouringObjective(BaseObjective):
	def __init__(self):

		super().__init__()
		LOG_DEBUG("Contouring Objective initializing")
		
		# No solver dependency; rely on data/config at runtime
		
		# Optional dependency: contouring constraints (if present, used for road boundaries)
		# Can work without it but will use default road width
		self.dependencies = []  # Make optional - module will check at runtime

		# Configuration options from CONFIGs (with safe defaults)
		_num_segments = self.get_config_value("contouring.num_segments", 10)
		try:
			self.num_segments = int(_num_segments)
		except Exception:
			self.num_segments = 10
		self.add_road_constraints = bool(self.get_config_value("contouring.add_road_constraints", True))
		self.two_way_road = bool(self.get_config_value("road.two_way", False))
		self.dynamic_velocity_reference = bool(self.get_config_value("contouring.dynamic_velocity_reference", False))
		self.goal_reaching_contouring = bool(self.get_config_value("contouring.goal_reaching_contouring", True))
		self.three_dimensional_contouring = bool(self.get_config_value("contouring.three_dimensional_contouring", False))
		# Weights with safe defaults
		self._w_contour = float(self.get_config_value("weights.contour_weight", 1.0))
		self._w_lag = float(self.get_config_value("weights.contouring_lag_weight", 0.1))
		self._w_goal = float(self.get_config_value("contouring.terminal_contouring", 10.0))
		self._w_angle = float(self.get_config_value("contouring.terminal_angle", 1.0))
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
		"""Check if contouring constraints are present (optional dependency)."""
		self._contouring_constraints = None
		if self.solver and hasattr(self.solver, 'module_manager'):
			constraint_modules = [m for m in self.solver.module_manager.get_modules() 
								if m.module_type == CONSTRAINT and m.name == "contouring_constraints"]
			if constraint_modules:
				self._contouring_constraints = constraint_modules[0]
				LOG_DEBUG("ContouringObjective: Found ContouringConstraints dependency")
			else:
				LOG_WARN("ContouringObjective: ContouringConstraints not found - will use default road width")
		else:
			LOG_DEBUG("ContouringObjective: Cannot check dependencies - solver or module_manager not available")

	def update(self, state, data):
		LOG_INFO("ContouringObjective.update")
		# Log key weights and options once at first call
		try:
			w_contour = self.get_config_value("weights.contour_weight")
			w_lag = self.get_config_value("weights.contouring_lag_weight")
			road_width = float(self.get_config_value("road.width", 7.0))
			LOG_INFO(f"  Weights: contour={w_contour}, lag={w_lag}; road.width={road_width}")
		except Exception:
			pass
		# Store current state for use in construct_road_constraints
		self._last_state = state
		
		# If self.reference_path is not set, try to get it from data
		if self.reference_path is None and hasattr(data, 'reference_path') and data.reference_path is not None:
			LOG_INFO("ContouringObjective: Setting reference_path from data")
			self.reference_path = data.reference_path
			# Process it if needed
			if hasattr(self, 'process_reference_path'):
				self.process_reference_path(data)
		
		# CRITICAL REQUIREMENT: Ensure reference path starts at current vehicle position
		# This must be done at EVERY planning step, not just once, because the vehicle moves
		# Road constraints are generated from the reference path, so it must be aligned with the vehicle's current position
		LOG_DEBUG(f"Checking path alignment: reference_path={'present' if self.reference_path is not None else 'None'}, state={'present' if state is not None else 'None'}")
		if self.reference_path is not None and state is not None:
			try:
				vehicle_pos = state.get_position()
				LOG_DEBUG(f"  vehicle_pos: {vehicle_pos}")
				if vehicle_pos is not None and len(vehicle_pos) >= 2:
					ref_path_start = (float(self.reference_path.x[0]), float(self.reference_path.y[0]))
					vehicle_pos_tuple = (float(vehicle_pos[0]), float(vehicle_pos[1]))
					
					dist = np.sqrt((ref_path_start[0] - vehicle_pos_tuple[0])**2 + 
								 (ref_path_start[1] - vehicle_pos_tuple[1])**2)
					
					LOG_DEBUG(f"  Path alignment check: vehicle=({vehicle_pos_tuple[0]:.3f}, {vehicle_pos_tuple[1]:.3f}), path_start=({ref_path_start[0]:.3f}, {ref_path_start[1]:.3f}), dist={dist:.3f}m")
			except Exception:
				pass
		
		# Sanity-check a few normals and left/right points from the reference path for diagnostics
		try:
			ref = self.reference_path
			if ref is not None and hasattr(ref, 'x_spline') and hasattr(ref, 'y_spline') and hasattr(ref, 's') and len(ref.s) >= 3:
				road_width = float(self.get_config_value("road.width", 7.0))
				half_w = 0.5 * road_width
				s_min = float(ref.s[0]); s_max = float(ref.s[-1])
				s_samples = np.linspace(s_min, s_min + max(1e-6, (s_max - s_min)) * 0.1, 3)
				for idx, s in enumerate(s_samples):
					x = float(ref.x_spline(s)); y = float(ref.y_spline(s))
					dx = float(ref.x_spline.derivative()(s)); dy = float(ref.y_spline.derivative()(s))
					norm = np.hypot(dx, dy)
					if norm < 1e-9:
						continue
					nx = -dy / norm; ny = dx / norm
					xl = x + half_w * nx; yl = y + half_w * ny
					xr = x - half_w * nx; yr = y - half_w * ny
					LOG_INFO(f"  Path sample[{idx}] s={s:.3f}: center=({x:.3f},{y:.3f}) normal=({nx:.3f},{ny:.3f}) L=({xl:.3f},{yl:.3f}) R=({xr:.3f},{yr:.3f})")
		except Exception:
			pass
		
		# Check if contouring constraints module is present - if so, it will handle road constraints
		# So we should disable our own road constraint creation to avoid duplicates
		has_contouring_constraints = False
		if self.solver and hasattr(self.solver, 'module_manager'):
			constraint_modules = [m for m in self.solver.module_manager.get_modules() 
								if m.module_type == CONSTRAINT and m.name == "contouring_constraints"]
			if constraint_modules:
				has_contouring_constraints = True
				LOG_INFO("ContouringObjective: ContouringConstraints module found - it will handle road constraints")
			else:
				LOG_DEBUG("ContouringObjective: ContouringConstraints module not found")
		else:
			if not self.solver:
				LOG_WARN("ContouringObjective: self.solver is None - cannot check for ContouringConstraints")
			elif not hasattr(self.solver, 'module_manager'):
				LOG_WARN("ContouringObjective: solver.module_manager not available - cannot check for ContouringConstraints")
		
		# Log reference path status
		has_ref_path = self.reference_path is not None
		# Only add road constraints if enabled AND contouring constraints module is not present
		should_add_road_constraints = self.add_road_constraints and not has_contouring_constraints
		LOG_INFO(f"ContouringObjective.update: reference_path={'present' if has_ref_path else 'missing'}, add_road_constraints={self.add_road_constraints}, has_contouring_constraints={has_contouring_constraints}, will_add={should_add_road_constraints}")
		
		if has_ref_path:
			# Log reference path details
			if hasattr(self.reference_path, 's') and self.reference_path.s is not None:
				s_arr = np.asarray(self.reference_path.s, dtype=float)
				if s_arr.size > 0:
					LOG_DEBUG(f"  Reference path arc length: {s_arr[0]:.2f} to {s_arr[-1]:.2f} (total: {s_arr[-1] - s_arr[0]:.2f})")
			if hasattr(self.reference_path, 'x') and self.reference_path.x is not None:
				x_arr = np.asarray(self.reference_path.x, dtype=float)
				if x_arr.size > 0:
					LOG_DEBUG(f"  Reference path x range: {x_arr[0]:.2f} to {x_arr[-1]:.2f}")
			if hasattr(self.reference_path, 'y') and self.reference_path.y is not None:
				y_arr = np.asarray(self.reference_path.y, dtype=float)
				if y_arr.size > 0:
					LOG_DEBUG(f"  Reference path y range: {y_arr[0]:.2f} to {y_arr[-1]:.2f}")
		
		try:
			import logging as _logging
			_logger = _logging.getLogger("integration_test")
			_logger.info(f"ContouringObjective.update: reference_path={has_ref_path}, add_road_constraints={self.add_road_constraints}")
		except Exception:
			pass
		
		# CRITICAL: After path realignment, we need to update path parameters
		# This ensures the solver uses the realigned path
		# We'll do this in set_parameters() which is called after update()
		
		# Construct road constraints if enabled AND contouring constraints module is not present
		# (ContouringConstraints module will handle road constraints if present)
		should_add_road_constraints = self.add_road_constraints and not has_contouring_constraints
		if should_add_road_constraints and self.reference_path is not None:
			LOG_INFO("ContouringObjective: Constructing road constraints from reference path")
			try:
				self.construct_road_constraints(data)
				# Log static obstacles status
				if hasattr(data, 'static_obstacles') and data.static_obstacles is not None:
					non_none = sum(1 for obs in data.static_obstacles if obs is not None)
					LOG_INFO(f"ContouringObjective: Created {non_none} static obstacles for road constraints (out of {len(data.static_obstacles)} slots)")
					if non_none > 0:
						# Log first obstacle details
						first_obs = next((obs for obs in data.static_obstacles if obs is not None), None)
						if first_obs and hasattr(first_obs, 'halfspaces'):
							LOG_DEBUG(f"  First static obstacle has {len(first_obs.halfspaces)} halfspace(s)")
				else:
					LOG_WARN("ContouringObjective: construct_road_constraints did not create static_obstacles!")
			except Exception as e:
				LOG_WARN(f"ContouringObjective: Error constructing road constraints: {e}")
				import traceback
				LOG_DEBUG(f"Traceback: {traceback.format_exc()}")

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

		# NOTE: Road constraints are already constructed above if needed (line 198-216)
		# This duplicate call is removed to avoid creating constraints twice
		# if self.add_road_constraints:
		#	self.construct_road_constraints(data)

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
		LOG_DEBUG(f"ContouringObjective.set_parameters called with k={k}")
		
		# Retrieve weights once (only for first stage)
		if k == 0:
			LOG_INFO("ContouringObjective.set_parameters: Setting weights for stage 0")
			contouring_weight = self.get_config_value("weights.contour_weight")
			lag_weight = self.get_config_value("weights.contouring_lag_weight")

			if self.goal_reaching_contouring:
				contouring_goal_weight = self.get_config_value("weights.contouring_goal_weight")
				parameter_manager.set_parameter("contouring_goal_weight", contouring_goal_weight, stage_index=0)

			if self.dynamic_velocity_reference:
				velocity_weight = self.get_config_value("weights.contour_velocity_weight")
				parameter_manager.set_parameter("contouring_reference_velocity_weight", velocity_weight, stage_index=0)

			parameter_manager.set_parameter("contour_weight", contouring_weight, stage_index=0)
			parameter_manager.set_parameter("contouring_lag_weight", lag_weight, stage_index=0)
			LOG_DEBUG(f"  Set weights: contour={contouring_weight}, lag={lag_weight}")

		# Process reference path and set path parameters (only once, typically at k=0)
		# CRITICAL: This is called AFTER update(), so if the path was realigned in update(),
		# we need to ensure the path parameters reflect the realigned path
		if k == 0:
			LOG_INFO("ContouringObjective.set_parameters: Processing reference path and setting path parameters")
			has_ref_path = hasattr(data, 'reference_path') and data.reference_path is not None
			LOG_DEBUG(f"  data.reference_path available: {has_ref_path}")
			
			# Ensure reference_path is set in self
			if self.reference_path is None and has_ref_path:
				LOG_INFO("  Setting self.reference_path from data")
				self.reference_path = data.reference_path
			
			# Process reference path if needed (this will rebuild splines if path was realigned)
			# Note: If path was realigned in update(), splines should already be rebuilt
			# But we still call process_reference_path to ensure consistency
			path_was_realigned = hasattr(self, '_path_realigned') and self._path_realigned
			if path_was_realigned:
				LOG_INFO("  Path was realigned in update(), ensuring parameters reflect realigned path")
				self._path_realigned = False  # Reset flag
			
			self.process_reference_path(data)
			
			# CRITICAL: After path realignment in update(), we need to ensure path parameters are updated
			# The path realignment rebuilds the splines, so we must update the parameters
			# Set path parameters (this sets parameters for ALL stages)
			if self.reference_path is not None:
				LOG_INFO("  Calling set_path_parameters (will use realigned path if realignment occurred)...")
				self.set_path_parameters(parameter_manager)
			else:
				LOG_WARN("  Cannot set path parameters: reference_path is None")
		else:
			LOG_DEBUG(f"  Skipping path parameter setup for stage {k} (already done at stage 0)")

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
		LOG_INFO("ContouringObjective.set_path_parameters: Setting path spline parameters")
		LOG_DEBUG(f"{self.name.title()}::set_path_parameters")
		if self.reference_path is None:
			LOG_WARN("No reference path available when trying to set path params so returning")
			return
		
		LOG_INFO(f"  reference_path is available: {self.reference_path is not None}")

		path_x = self.reference_path.x
		path_y = self.reference_path.y
		
		# HIGH-LEVEL DEBUG: Log reference path details
		LOG_INFO(f"  Reference path: len(x)={len(path_x) if hasattr(path_x, '__len__') else 'N/A'}, len(y)={len(path_y) if hasattr(path_y, '__len__') else 'N/A'}")
		if hasattr(self.reference_path, 's') and self.reference_path.s is not None:
			s_arr = np.asarray(self.reference_path.s, dtype=float)
			if s_arr.size > 0:
				LOG_INFO(f"  Arc length range: s_min={s_arr[0]:.4f}, s_max={s_arr[-1]:.4f}, total={s_arr[-1] - s_arr[0]:.4f}")
		LOG_INFO(f"  num_segments={self.num_segments}")

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
		# CRITICAL: Parameters must be set for ALL stages (not just stage 0)
		# Since set_path_parameters is called once at k=0, we set parameters for all stages
		params_set = 0
		LOG_INFO(f"  Setting parameters for {self.num_segments} segments...")
		for i in range(self.num_segments):
			# Set segment start parameter (for all stages - pass None to set for all)
			start_val = float(x_starts[i])
			parameter_manager.set_parameter(f"path_{i}_start", start_val, stage_index=None)

			# Set path coordinate coefficients (for all stages)
			parameter_manager.set_parameter(f"path_x_{i}_a", path_x_coeffs[i][0], stage_index=None)
			parameter_manager.set_parameter(f"path_x_{i}_b", path_x_coeffs[i][1], stage_index=None)
			parameter_manager.set_parameter(f"path_x_{i}_c", path_x_coeffs[i][2], stage_index=None)
			parameter_manager.set_parameter(f"path_x_{i}_d", path_x_coeffs[i][3], stage_index=None)

			parameter_manager.set_parameter(f"path_y_{i}_a", path_y_coeffs[i][0], stage_index=None)
			parameter_manager.set_parameter(f"path_y_{i}_b", path_y_coeffs[i][1], stage_index=None)
			parameter_manager.set_parameter(f"path_y_{i}_c", path_y_coeffs[i][2], stage_index=None)
			parameter_manager.set_parameter(f"path_y_{i}_d", path_y_coeffs[i][3], stage_index=None)
			params_set += 1
			
			# Log first segment details
			if i == 0:
				LOG_INFO(f"  Segment 0: start={start_val:.4f}, x_coeffs=[{path_x_coeffs[0][0]:.4f}, {path_x_coeffs[0][1]:.4f}, {path_x_coeffs[0][2]:.4f}, {path_x_coeffs[0][3]:.4f}]")
		
		LOG_INFO(f"  Set {params_set} segment parameters (path_x and path_y for each) for ALL stages")
		
		# VERIFY: Check that parameters were actually set
		try:
			test_stage = 0
			test_params = parameter_manager.get_all(test_stage)
			has_path_0_start = "path_0_start" in test_params
			has_path_x_0_a = "path_x_0_a" in test_params
			LOG_INFO(f"  VERIFICATION: Stage {test_stage} has path_0_start={has_path_0_start}, path_x_0_a={has_path_x_0_a}")
			if has_path_0_start:
				LOG_INFO(f"    path_0_start value: {test_params['path_0_start']}")
		except Exception as e:
			LOG_WARN(f"  Could not verify parameters: {e}")

		# Set 3D and velocity parameters (if needed)
		for i in range(self.num_segments):
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

	def get_value(self, state, data, stage_idx):
		"""Compute contouring objective cost symbolically.
		
		CRITICAL: This method receives symbolic_state for future stages (stage_idx > 0),
		ensuring objectives are computed symbolically using predicted states.
		This matches the reference codebase pattern.
		
		Reference: https://github.com/tud-amr/mpc_planner - objectives are evaluated symbolically.
		"""
		# Get parameters from parameter_manager for this stage
		symbolic_state = state  # state is already symbolic for future stages
		if self.solver and hasattr(self.solver, 'parameter_manager'):
			params_dict = self.solver.parameter_manager.get_all(stage_idx)
			
			# HIGH-LEVEL DEBUG: Log parameter availability for first stage
			if stage_idx == 0:
				path_params = [k for k in params_dict.keys() if 'path' in k]
				LOG_DEBUG(f"ContouringObjective.get_value stage {stage_idx}: Retrieved {len(params_dict)} total parameters, {len(path_params)} path-related")
				if not path_params:
					LOG_WARN(f"  NO PATH PARAMETERS FOUND in stage {stage_idx}!")
					LOG_DEBUG(f"  All param keys: {list(params_dict.keys())[:10]}...")  # Show first 10
			
			# Create a wrapper dict that has .get() method for compatibility with Spline2D
			class ParamDictWrapper:
				def __init__(self, d):
					self._dict = d
				def get(self, key, default=None):
					return self._dict.get(key, default)
				def has_parameter(self, key):
					return key in self._dict
			params = ParamDictWrapper(params_dict)
		else:
			LOG_WARN(f"ContouringObjective.get_value stage {stage_idx}: No solver or parameter_manager available!")
			# Create empty wrapper
			class ParamDictWrapper:
				def __init__(self):
					self._dict = {}
				def get(self, key, default=None):
					return default
				def has_parameter(self, key):
					return False
			params = ParamDictWrapper()

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
		
		# Get symbolic state variables (state is already symbolic for future stages)
		pos_x = state.get("x")
		pos_y = state.get("y")
		psi = state.get("psi")
		v = state.get("v")
		spline_val = state.get("spline")
		
		if spline_val is None:
			raise ValueError("ContouringObjective: spline state variable is None. Ensure vehicle dynamics model includes 'spline' state.")
		
		# Normalize spline to [0,1] for Spline2D evaluation
		# spline_val should represent arc length, s_max is total path arc length
		s = spline_val / s_max
		
		# HIGH-LEVEL DEBUG: Log spline math for first few stages
		if stage_idx <= 2:
			try:
				import casadi as cd
				# Try to get numeric values if possible
				if not isinstance(spline_val, (cd.MX, cd.SX)):
					LOG_INFO(f"ContouringObjective.get_value stage {stage_idx}: spline_val={float(spline_val):.4f}, s_max={float(s_max):.4f}, normalized_s={float(s):.4f}")
				else:
					LOG_INFO(f"ContouringObjective.get_value stage {stage_idx}: spline_val=symbolic, s_max={float(s_max):.4f}, normalizing to [0,1]")
			except Exception:
				LOG_DEBUG(f"ContouringObjective.get_value stage {stage_idx}: computing with symbolic spline")


		# Validate that we have all required variables
		if any(var is None for var in [pos_x, pos_y, psi, v, s]):
			missing_vars = [name for name, var in
							[("x", pos_x), ("y", pos_y), ("psi", psi), ("v", v), ("spline", s)] if var is None]
			raise ValueError(f"Missing required symbolic variables: {missing_vars}")

		# Get symbolic weights from parameters
		# Safe weight fallbacks from config defaults
		contour_weight = params.get("contour_weight") or self._w_contour
		lag_weight = params.get("contouring_lag_weight") or self._w_lag
		velocity_weight = None

		goal_weight = 0
		remaining_distance = 0
		if self.goal_reaching_contouring:
			goal_weight = params.get("contouring_goal_weight") or self._w_goal
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
			velocity_weight = params.get("velocity") or 0.0

		# Preferred: parametric spline path; fallback to reference_path splines
		try:
			# HIGH-LEVEL DEBUG: Check if parameters are available
			if stage_idx <= 2:
				has_path_x = any(params.has_parameter(f"path_x_{i}_a") for i in range(self.num_segments))
				has_path_y = any(params.has_parameter(f"path_y_{i}_a") for i in range(self.num_segments))
				has_path_start = params.has_parameter("path_0_start")
				LOG_INFO(f"ContouringObjective stage {stage_idx}: Spline2D params - path_x={has_path_x}, path_y={has_path_y}, path_0_start={has_path_start}, num_segments={self.num_segments}")
				if not has_path_x or not has_path_y or not has_path_start:
					LOG_WARN(f"  Missing parameters! path_x={has_path_x}, path_y={has_path_y}, path_0_start={has_path_start}")
					# Debug: show what parameters we DO have
					if hasattr(params, '_dict'):
						param_keys = list(params._dict.keys())
						path_params = [k for k in param_keys if 'path' in k]
						LOG_DEBUG(f"  Available path-related params ({len(path_params)}): {path_params[:10]}...")  # Show first 10
			
			path = Spline2D(params, self.num_segments, s)
			path_x, path_y = path.at(s)
			path_dx_normalized, path_dy_normalized = path.deriv_normalized(s)
			
			# HIGH-LEVEL DEBUG: Log path evaluation for first stages
			if stage_idx <= 2:
				try:
					import casadi as cd
					if not isinstance(path_x, (cd.MX, cd.SX)):
						LOG_INFO(f"  Path evaluation: path_x={float(path_x):.4f}, path_y={float(path_y):.4f}, s={float(s):.4f}")
				except Exception:
					LOG_DEBUG(f"  Path evaluation: symbolic")
		except Exception as e:
			# HIGH-LEVEL DEBUG: Log fallback usage
			if stage_idx <= 2:
				LOG_WARN(f"ContouringObjective stage {stage_idx}: Spline2D failed ({type(e).__name__}: {e}), using fallback reference_path splines")
			# Fallback: use reference_path splines; support both numeric and CasADi symbolic s
			import casadi as cd
			s_min, s_max = self._get_arc_length_bounds()
			# Denormalize: s is [0,1], convert back to actual arc length
			cur_s = s * (s_max - s_min) + s_min
			
			# HIGH-LEVEL DEBUG: Log denormalization
			if stage_idx <= 2:
				try:
					if not isinstance(cur_s, (cd.MX, cd.SX)):
						LOG_INFO(f"  Fallback: normalized_s={float(s):.4f} -> arc_length={float(cur_s):.4f} (s_min={float(s_min):.4f}, s_max={float(s_max):.4f})")
				except Exception:
					LOG_DEBUG(f"  Fallback: denormalizing symbolic s")
			# If cur_s is symbolic, use CasADi interpolants; else use scipy splines directly
			if isinstance(cur_s, (cd.MX, cd.SX)):
				# Sample along s for interpolants
				s_vals = np.asarray(self.reference_path.s, dtype=float)
				if s_vals is None or s_vals.size == 0:
					s_vals = np.linspace(float(s_min), float(s_max), 100)
				x_vals = np.array([float(self.reference_path.x_spline(si)) for si in s_vals])
				y_vals = np.array([float(self.reference_path.y_spline(si)) for si in s_vals])
				# CasADi interpolants
				x_interp = cd.interpolant('x_interp', 'linear', [s_vals], x_vals)
				y_interp = cd.interpolant('y_interp', 'linear', [s_vals], y_vals)
				path_x = x_interp(cur_s)
				path_y = y_interp(cur_s)
				# Finite-difference derivatives
				eps = 1e-3
				dx = (x_interp(cur_s + eps) - x_interp(cur_s - eps)) / (2 * eps)
				dy = (y_interp(cur_s + eps) - y_interp(cur_s - eps)) / (2 * eps)
				nrm = cd.sqrt(dx*dx + dy*dy)
				nrm = cd.fmax(nrm, 1e-6)
				path_dx_normalized = dx / nrm
				path_dy_normalized = dy / nrm
			else:
				path_x = self.reference_path.x_spline(float(cur_s))
				path_y = self.reference_path.y_spline(float(cur_s))
				dx = self.reference_path.x_spline.derivative()(float(cur_s))
				dy = self.reference_path.y_spline.derivative()(float(cur_s))
				nrm = np.sqrt(float(dx*dx + dy*dy))
				nrm = max(float(nrm), 1e-6)
				path_dx_normalized = dx / nrm
				path_dy_normalized = dy / nrm

		# MPCC
		contour_error = path_dy_normalized * (pos_x - path_x) - path_dx_normalized * (pos_y - path_y)
		lag_error = path_dx_normalized * (pos_x - path_x) + path_dy_normalized * (pos_y - path_y)

		# Log detailed information for diagnosis
		try:
			# Try to get numeric values for logging (if symbolic, will be None)
			import casadi as cd
			if isinstance(pos_x, (cd.MX, cd.SX)):
				# Symbolic state - log that we're computing costs
				LOG_DEBUG(f"ContouringObjective.get_value: stage={stage_idx}, computing symbolic costs (x,y,psi,v,spline)")
			else:
				# Numeric state - log actual values
				LOG_INFO(f"ContouringObjective.get_value: stage={stage_idx}, pos=({float(pos_x):.2f},{float(pos_y):.2f}), path=({float(path_x):.2f},{float(path_y):.2f}), spline={float(s):.3f}")
				LOG_DEBUG(f"  contour_error={float(contour_error):.4f}, lag_error={float(lag_error):.4f}")
		except Exception:
			LOG_DEBUG(f"ContouringObjective.get_value: stage={stage_idx}, computing costs")

		# Cost components
		lag_cost = lag_weight * lag_error ** 2
		contour_cost = contour_weight * contour_error ** 2
		goal_cost = goal_weight * remaining_distance ** 2
		
		# Log cost values for all stages (try to extract numeric if possible)
		try:
			import casadi as cd
			if not isinstance(lag_cost, (cd.MX, cd.SX)):
				LOG_INFO(f"ContouringObjective.get_value: stage={stage_idx}, lag_cost={float(lag_cost):.4f}, "
				        f"contour_cost={float(contour_cost):.4f}, goal_cost={float(goal_cost):.4f}, "
				        f"contour_error={float(contour_error):.4f}, lag_error={float(lag_error):.4f}")
			else:
				LOG_INFO(f"ContouringObjective.get_value: stage={stage_idx}, computing symbolic costs (lag, contour, goal)")
		except Exception:
			LOG_DEBUG(f"ContouringObjective.get_value: stage={stage_idx}, computing costs (unable to extract numeric values)")

		# Velocity cost (if enabled)
		velocity_cost = 0
		if self.dynamic_velocity_reference:

			velocity_cost = velocity_weight * (v - reference_velocity) ** 2

		terminal_cost = 0
		horizon_val = self.solver.horizon if (hasattr(self.solver, 'horizon') and self.solver.horizon is not None) else 10
		if self.goal_reaching_contouring and stage_idx == horizon_val - 1:

			terminal_angle_weight = float(self.get_config_value("contouring.terminal_angle", 1.0))
			terminal_contouring_mp = float(self.get_config_value("contouring.terminal_contouring", 10.0))

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
		# Convert to numpy arrays for arithmetic
		x_arr = np.asarray(reference_path.x, dtype=float)
		y_arr = np.asarray(reference_path.y, dtype=float)
		s_arr = np.asarray(reference_path.s, dtype=float) if hasattr(reference_path, 's') else None
		if x_arr.size == 0 or y_arr.size == 0:
			return 0, 0

		pos_x, pos_y = float(position[0]), float(position[1])

		# Compute squared distances to all path points
		dx = x_arr - pos_x
		dy = y_arr - pos_y
		distances_squared = dx ** 2 + dy ** 2

		closest_idx = np.argmin(distances_squared)

		# Better segment calculation based on actual arc length
		segment_idx = 0
		if s_arr is not None and s_arr.size > 1 and self.num_segments > 0:
			# Get the arc length at the closest point
			closest_s = s_arr[closest_idx]
			s_min = s_arr[0]
			s_max = s_arr[-1]

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
		LOG_INFO("ContouringObjective.construct_road_constraints_from_centerline: Creating road constraints from centerline")

		horizon_val = self.get_horizon(data, default=10)
		LOG_DEBUG(f"  Horizon: {horizon_val}, creating {horizon_val + 1} static obstacle slots")
		data.set("static_obstacles", [None] * (horizon_val + 1))

		# Get road width with safe default (overridden by config if present)
		road_width_half = float(self.get_config_value("planner.road.width", 7.0)) / 2.0

		# Determine starting arc length and per-step interpolation (solver-agnostic)
		s_arr = np.asarray(self.reference_path.s, dtype=float)
		if s_arr.size < 2:
			return
		
		# CRITICAL FIX: Find the closest point on the reference path to the current vehicle position
		# This ensures the road constraints start from where the vehicle actually is
		# After path realignment, the path should start at the vehicle position, so idx0 should be 0
		idx0 = 0
		# Try to get current state from data (passed during update)
		current_pos = None
		if hasattr(data, 'current_state') and data.current_state is not None:
			try:
				current_pos = data.current_state.get_position()
			except Exception:
				pass
		# Also try to get from state stored in self (if update was called)
		if current_pos is None and hasattr(self, '_last_state') and self._last_state is not None:
			try:
				current_pos = self._last_state.get_position()
			except Exception:
				pass
		
		if current_pos is not None:
			# Find closest point on reference path
			try:
				x_arr = np.asarray(self.reference_path.x, dtype=float)
				y_arr = np.asarray(self.reference_path.y, dtype=float)
				dx = x_arr - current_pos[0]
				dy = y_arr - current_pos[1]
				distances_squared = dx ** 2 + dy ** 2
				idx0 = np.argmin(distances_squared)
				closest_dist = np.sqrt(distances_squared[idx0])
				LOG_INFO(f"  Found closest point on path: idx={idx0}, path_point=({x_arr[idx0]:.2f}, {y_arr[idx0]:.2f}), vehicle=({current_pos[0]:.2f}, {current_pos[1]:.2f}), dist={closest_dist:.3f}m")
				
				# CRITICAL: If path was realigned, idx0 should be 0 and distance should be very small
				# If not, there's a problem with the realignment
				if idx0 != 0 and closest_dist > 0.1:
					LOG_WARN(f"  WARNING: Closest point is not at path start (idx={idx0}, dist={closest_dist:.3f}m). Path may not be properly realigned!")
			except Exception as e:
				LOG_WARN(f"  Could not find closest point: {e}, using idx0=0")
		elif hasattr(data, 'current_path_segment') and data.current_path_segment is not None:
			idx0 = int(max(0, min(int(data.current_path_segment), s_arr.size - 1)))
		elif hasattr(self, 'closest_point_idx'):
			idx0 = int(max(0, min(int(self.closest_point_idx), s_arr.size - 1)))
		
		s0 = float(s_arr[idx0])
		s_end = float(s_arr[-1])
		step_s = (s_end - s0) / float(max(1, horizon_val))

		# Get robot radius safely
		robot_radius = 0.5
		if hasattr(data, 'robot_area') and data.robot_area and len(data.robot_area) > 0:
			robot_radius = float(data.robot_area[0].radius)

		constraints_created = 0
		for k in range(horizon_val + 1):
			from planning.types import StaticObstacle
			data.static_obstacles[k] = StaticObstacle()
			cur_s = min(s_end, s0 + k * step_s)

			# Get path point and derivatives from splines
			path_point_x = self.reference_path.x_spline(cur_s)
			path_point_y = self.reference_path.y_spline(cur_s)
			path_point = np.array([float(path_point_x), float(path_point_y)])

			# Get orthogonal vector (normal to path)
			path_dx = float(self.reference_path.x_spline.derivative()(cur_s))
			path_dy = float(self.reference_path.y_spline.derivative()(cur_s))

			# Normalize derivatives
			norm = safe_norm(path_dx, path_dy)
			if norm < 1e-6:
				LOG_DEBUG(f"  Stage {k}: Skipping (norm too small: {norm})")
				continue
			path_dx_norm = path_dx / norm
			path_dy_norm = path_dy / norm

			# Create orthogonal vector (rotate 90 degrees)
			dpath = np.array([-path_dy_norm, path_dx_norm])  # Orthogonal to path direction

			# Adjust width based on road type
			if self.two_way_road:
				width_times = 3.0
			else:
				width_times = 1.0

			# CRITICAL FIX: For stage 0, ensure the constraints accommodate the current vehicle position
			# This prevents infeasibility when the vehicle starts away from the path
			# Since we've ensured the path starts at vehicle position, this should mainly be a safety check
			if k == 0:
				current_pos = None
				# Try multiple ways to get current state
				if hasattr(self, '_last_state') and self._last_state is not None:
					try:
						current_pos = self._last_state.get_position()
					except Exception:
						pass
				# Also try from data
				if current_pos is None and hasattr(data, 'current_state') and data.current_state is not None:
					try:
						current_pos = data.current_state.get_position()
					except Exception:
						pass
				
				if current_pos is not None:
					try:
						# Check if current position would violate constraints
						# Calculate distance from vehicle to path point
						vehicle_pos = np.array([float(current_pos[0]), float(current_pos[1])])
						dist_to_path = np.linalg.norm(vehicle_pos - path_point)
						
						# Calculate distance from path center to road edge (accounting for robot radius)
						road_edge_dist = width_times * road_width_half - robot_radius
						
						# If vehicle is outside the road, expand it
						if dist_to_path > road_edge_dist:
							# Expand road width to include vehicle with some margin
							required_edge_dist = dist_to_path + robot_radius + 1.0  # 1.0m margin
							required_width_half = required_edge_dist / width_times
							LOG_INFO(f"  Stage 0: Vehicle at ({current_pos[0]:.2f}, {current_pos[1]:.2f}) is {dist_to_path:.2f}m from path point ({path_point[0]:.2f}, {path_point[1]:.2f})")
							LOG_INFO(f"  Stage 0: Expanding road width from {road_width_half * 2:.2f}m to {required_width_half * 2:.2f}m to include vehicle")
							road_width_half = required_width_half
						else:
							LOG_DEBUG(f"  Stage 0: Vehicle at ({current_pos[0]:.2f}, {current_pos[1]:.2f}) is within road bounds (dist_to_path={dist_to_path:.2f}m, road_edge_dist={road_edge_dist:.2f}m)")
					except Exception as e:
						LOG_DEBUG(f"  Could not adjust road width for stage 0: {e}")
						import traceback
						LOG_DEBUG(f"  Traceback: {traceback.format_exc()}")

			# Left halfspace constraint: A·p <= b where A is normal pointing left
			# The constraint A·p <= b means the point p must be on the "left" side of the boundary
			# Boundary is at: path_point + A * (width_times * road_width_half - robot_radius)
			# This accounts for road width and subtracts robot_radius so the disc center can be closer to the edge
			A = dpath.copy()
			boundary_left = path_point + A * (width_times * road_width_half - robot_radius)
			b_left = np.dot(A, boundary_left)
			data.static_obstacles[k].add_halfspace(A, b_left)

			# Right halfspace constraint: -A·p <= b_right where -A points right
			# The constraint -A·p <= b_right means A·p >= -b_right, so the point must be on the "right" side
			# FIXED: Use width_times consistently for both left and right boundaries
			boundary_right = path_point - A * (width_times * road_width_half - robot_radius)
			b_right = np.dot(-A, boundary_right)
			data.static_obstacles[k].add_halfspace(-A, b_right)
			
			# Log constraint details for stage 0
			if k == 0:
				LOG_DEBUG(f"  Stage 0: path_point=({path_point[0]:.2f}, {path_point[1]:.2f}), "
				         f"road_width_half={road_width_half:.2f}, width_times={width_times}, robot_radius={robot_radius:.2f}")
				LOG_DEBUG(f"  Stage 0: Left boundary b={b_left:.2f}, Right boundary b_right={b_right:.2f}")
			
			constraints_created += 1
			# Log details for first few stages
			if k <= 2:
				LOG_DEBUG(f"  Stage {k}: s={cur_s:.2f}, path_point=({path_point[0]:.2f}, {path_point[1]:.2f}), "
				         f"halfspaces={len(data.static_obstacles[k].halfspaces)}")
		
		LOG_INFO(f"ContouringObjective: Created {constraints_created} road constraint obstacles (out of {horizon_val + 1} stages)")

	def construct_road_constraints_from_bounds(self, data):
		"""Construct road constraints using actual road bounds"""
		# Ensure static_obstacles exists
		if not hasattr(data, "static_obstacles") or data.static_obstacles is None:
			data.set("static_obstacles", [])

		# Ensure we have enough slots for all horizon steps
		horizon_val = self.solver.horizon if (hasattr(self.solver, 'horizon') and self.solver.horizon is not None) else 10
		required_length = horizon_val + 1
		current_length = len(data.static_obstacles)

		if current_length < required_length:
			data.static_obstacles.extend([None] * (required_length - current_length))

		# Get current vehicle progress from data/closest index (solver-agnostic)
		s_arr = np.asarray(self.reference_path.s, dtype=float)
		if s_arr.size == 0:
			return
		idx = 0
		if hasattr(data, 'current_path_segment') and data.current_path_segment is not None:
			idx = int(max(0, min(int(data.current_path_segment), s_arr.size - 1)))
		elif hasattr(self, 'closest_point_idx'):
			idx = int(max(0, min(int(self.closest_point_idx), s_arr.size - 1)))
		current_norm_s = float((s_arr[idx] - s_arr[0]) / (s_arr[-1] - s_arr[0])) if (s_arr[-1] - s_arr[0]) > 1e-9 else 0.0

		# Estimate vehicle velocity to predict future positions
		vehicle_velocity = self.solver.get_initial_state().get("v")  # Default to 1.0 if not available
		dt = self.solver.dt if hasattr(self.solver, 'dt') else 0.1  # Time step
		LOG_DEBUG("Static obstacles: {}".format(data.static_obstacles))

		# Create obstacles for all horizon steps (not just the new ones)
		horizon_val = self.solver.horizon if (hasattr(self.solver, 'horizon') and self.solver.horizon is not None) else 10
		for k in range(horizon_val + 1):
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

		horizon_val = self.solver.horizon if (hasattr(self.solver, 'horizon') and self.solver.horizon is not None) else 10
		ani = FuncAnimation(fig, animate, frames=range(horizon_val), interval=100, blit=False)
		plt.show()



