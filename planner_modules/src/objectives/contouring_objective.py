import numpy as np
import casadi as cd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
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
		closest_pt = np.array(
			[self.reference_path.x[self.closest_point_idx], self.reference_path.y[self.closest_point_idx]])
		vehicle_pos = np.array(state.get_position())
		LOG_DEBUG(f"Distance to closest point: {np.linalg.norm(closest_pt - vehicle_pos)}")

		if self.add_road_constraints:
			self.construct_road_constraints(data)

		if self.get_config_value("plot.debug", True):
			self.animate_forecasted_bounds(state, data)

	def define_parameters(self, params):
		"""Define all parameters used by this module"""
		LOG_DEBUG("Defining contouring objective parameters")
		# Core parameters
		params.add("contour_weight", add_to_rqt_reconfigure=True)
		params.add("lag_weight", add_to_rqt_reconfigure=True)
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
			contouring_weight = self.get_config_value("weights.contour_weight")
			lag_weight = self.get_config_value("weights.lag_weight")

			terminal_angle_weight = self.get_config_value("weights.terminal_angle")
			terminal_contouring_weight = self.get_config_value("weights.terminal_contouring")

			if self.dynamic_velocity_reference:
				reference_velocity_weight = self.get_config_value("weights.reference_velocity")
				velocity_weight = self.get_config_value("weights.velocity")
				parameter_manager.set_parameter("reference_velocity", reference_velocity_weight)
				parameter_manager.set_parameter("velocity", velocity_weight)

			parameter_manager.set_parameter("contour_weight", contouring_weight)
			parameter_manager.set_parameter("lag_weight", lag_weight)
			parameter_manager.set_parameter("terminal_angle", terminal_angle_weight)
			parameter_manager.set_parameter("terminal_contouring", terminal_contouring_weight)

		self.set_path_parameters(parameter_manager)

	def set_path_parameters(self, parameter_manager):
		LOG_INFO("ContouringObjective.set_path_parameters")
		if self.reference_path is None:
			return

		# TODO: Will need to be updated to allow for 3D
		path_x = self.reference_path.x
		path_y = self.reference_path.y

		if len(path_x) < self.num_segments + 1:
			# Get lists of evenly spaced partitions over the provided interval
			s_original = np.linspace(0, 1, len(path_x))
			s_new = np.linspace(0, 1, self.num_segments + 1)

			# Interpolates at each s_new point based on the coordinates (s_new, path_x) for each i in S_new
			path_x_interp = np.interp(s_new, s_original, path_x)
			# Same type of interpolation but for points (s_new, path_y)
			path_y_interp = np.interp(s_new, s_original, path_y)
		else:

			# Compute cumulative arc length
			dx = np.diff(path_x)
			dy = np.diff(path_y)
			arc_lengths = np.sqrt(dx ** 2 + dy ** 2)
			s = np.concatenate([[0], np.cumsum(arc_lengths)])
			s /= s[-1]  # Normalize to [0, 1]

			# Resample at evenly spaced points in [0, 1]
			s_new = np.linspace(0, 1, self.num_segments + 1)
			path_x_interp = np.interp(s_new, s, path_x)
			path_y_interp = np.interp(s_new, s, path_y)


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

	def get_value(self, params, stage_idx):
		"""
		Create a symbolic objective function for trajectory optimization based on contouring error.

		Fixed version with correct contour and lag error calculations.
		"""

		# Get symbolic state variables from the model
		pos_x = self.solver.get("x", stage_idx)
		pos_y = self.solver.get("y", stage_idx)
		psi = self.solver.get("psi", stage_idx)
		v = self.solver.get("v", stage_idx)
		s = self.solver.get("spline", stage_idx)

		# Get symbolic weights from parameters
		contour_weight = params.get("contour_weight")
		lag_weight = params.get("lag_weight")

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

		# Position error vector (FIXED: Remove absolute values)
		dx = pos_x - path_x
		dy = pos_y - path_y

		# FIXED: Correct contour error (lateral deviation - perpendicular to path)
		# Normal vector pointing left from path direction: (-path_dy_normalized, path_dx_normalized)
		contour_error = path_dy_normalized * dx - path_dx_normalized * dy

		# FIXED: Correct lag error (longitudinal deviation - along path direction)
		lag_error = path_dx_normalized * dx + path_dy_normalized * dy
		lag_cost = lag_weight * lag_error ** 2
		contour_cost = contour_weight * contour_error ** 2


		velocity_cost = 0
		if self.dynamic_velocity_reference:
			reference_velocity = self._evaluate_spline_casadi(s, "path_vel", params)
			velocity_weight = params.get("reference_velocity")  # Fixed parameter name

			velocity_cost = velocity_weight * (v - reference_velocity) ** 2

		return {
			"contouring_lag_cost": lag_cost,
			"contouring_contour_cost": contour_cost,
			"contouring_velocity_cost": velocity_cost
		}

	def on_data_received(self, data, data_name):
		LOG_DEBUG("RECEIVED DATA FOR CONTOURING OBJ")
		if data.has("reference_path") and data.reference_path is not None:
			LOG_DEBUG("Received Reference Path")
			self.process_reference_path(data)

	def process_reference_path(self, data):
		LOG_DEBUG("Processing reference path for Contouring Objective")
		# Store the original path data

		self.reference_path = data.reference_path

		# Compute arc progress list if not provided
		if data.reference_path.s is None:
			self.reference_path.s = compute_arc_length(data.reference_path)

		# Create velocity reference spline if available
		if self.dynamic_velocity_reference and len(data.reference_path.v) > 0:
			self.reference_path.v_spline = CubicSpline(self.reference_path.s, data.reference_path.v)

		# Process road bounds if available
		if self.add_road_constraints and data.left_bound is not None and data.right_bound is not None:
			LOG_DEBUG("Processing provided left and right bounds for Contouring Objective")
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

		# Use the arc progress list to determine the index of the segment which stores the closest point
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

		data.set("static_obstacles", [None] * self.solver.horizon)

		# Get road width
		road_width_half = self.get_config_value("road.width") / 2.0

		for k in range(self.solver.horizon):
			data.static_obstacles[k] = StaticObstacle()

			# Get predicted spline parameter for this timestep
			norm_s = self.solver.get_ego_prediction(k, "spline") / self.reference_path.get_length()

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

		"""Construct road constraints using actual road bounds"""
		if data.static_obstacles is None:
			data.set("static_obstacles", [None] * self.solver.horizon)

		LOG_DEBUG("Forecasting road from bounds")
		for k in range(self.solver.horizon):
			# Create a static obstacle for this time step
			data.static_obstacles[k] = StaticObstacle()

			# Get the current normalized position on the spline
			norm_s = self.solver.get_ego_prediction(k, "spline") / self.reference_path.get_arc_length()
			LOG_DEBUG("When constructing road constraints norm s is {}".format(norm_s))

			# Convert to actual arc length
			if len(self.reference_path.s) >= 2:
				s_min = self.reference_path.s[0]
				s_max = self.reference_path.s[-1]
				cur_s = s_min + norm_s * (s_max - s_min)
				cur_s = float(cur_s)
			else:
				continue

			# Get centerline path point and its tangent (for consistent normal direction)
			path_point_x = self.reference_path.x_spline(cur_s)
			path_point_y = self.reference_path.y_spline(cur_s)

			path_dx = self.reference_path.x_spline.derivative()(cur_s)
			path_dy = self.reference_path.y_spline.derivative()(cur_s)

			# Normalize path tangent
			path_norm = safe_norm(path_dx, path_dy)
			path_dx_norm = path_dx / path_norm
			path_dy_norm = path_dy / path_norm

			# Create consistent normal vector (pointing left from path direction)
			path_normal = np.array([-path_dy_norm, path_dx_norm])

			# Get left and right bound points
			left_x = self.bound_left_spline(cur_s)[0]
			left_y = self.bound_left_spline(cur_s)[1]
			left_point = np.array([left_x, left_y])

			right_x = self.bound_right_spline(cur_s)[0]
			right_y = self.bound_right_spline(cur_s)[1]
			right_point = np.array([right_x, right_y])

			# Determine which side is actually left/right by checking cross product
			center_to_left = left_point - np.array([path_point_x, path_point_y])
			center_to_right = right_point - np.array([path_point_x, path_point_y])

			# Cross product to determine orientation
			left_cross = np.cross(np.array([path_dx_norm, path_dy_norm]), center_to_left)
			right_cross = np.cross(np.array([path_dx_norm, path_dy_norm]), center_to_right)

			# Ensure correct assignment (left should have positive cross product)
			if left_cross < 0:
				left_point, right_point = right_point, left_point
				LOG_DEBUG("Swapped left and right bounds based on cross product")

			# Create halfspace constraints
			# Left bound: normal points inward (toward the road)
			# Remove robot radius subtraction or handle it properly
			Al = -path_normal  # Points right (inward from left bound)
			bl = np.dot(Al, left_point)  # Remove the radius subtraction
			data.static_obstacles[k].add_halfspace(Al, bl)

			Ar = path_normal  # Points left (inward from right bound)
			br = np.dot(Ar, right_point)  # Remove the radius subtraction
			data.static_obstacles[k].add_halfspace(Ar, br)

			LOG_DEBUG(f"Constructed road constraints: Left({Al}, {bl}), Right({Ar}, {br})")
			LOG_DEBUG(f"Left point: {left_point}, Right point: {right_point}")
			LOG_DEBUG(f"Path normal: {path_normal}, Center: ({path_point_x}, {path_point_y})")


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
			x = self.solver.get_ego_prediction(k, "x")
			y = self.solver.get_ego_prediction(k, "y")
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

						s = self.solver.get_ego_prediction(k, "spline")  # Usually in [0, 1]
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



