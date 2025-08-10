import casadi as ca
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Polygon as MPLPolygon
from scipy.interpolate import CubicSpline

from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
from planner_modules.src.constraints.decomp_constraints import DecompConstraints
from planner_modules.src.objectives.contouring_objective import ContouringObjective
from planning.src.data_prep import define_robot_area
from planning.src.dynamic_models import ContouringSecondOrderUnicycleModel, numeric_rk4
from planning.src.planner import Planner
from planning.src.types import Data, Bound, generate_reference_path, calculate_path_normals, State, \
	generate_dynamic_obstacles, Costmap
from solver.src.casadi_solver import CasADiSolver
from utils.const import GAUSSIAN
from utils.utils import LOG_DEBUG


import casadi as ca
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Polygon as MPLPolygon
from scipy.interpolate import CubicSpline

from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
from planner_modules.src.constraints.decomp_constraints import DecompConstraints
from planner_modules.src.objectives.contouring_objective import ContouringObjective
from planning.src.data_prep import define_robot_area
from planning.src.dynamic_models import ContouringSecondOrderUnicycleModel, numeric_rk4
from planning.src.planner import Planner
from planning.src.types import Data, Bound, generate_reference_path, calculate_path_normals, State, \
	generate_dynamic_obstacles, Costmap
from solver.src.casadi_solver import CasADiSolver
from utils.const import GAUSSIAN
from utils.utils import LOG_DEBUG


class MPCVisualizer:
	def __init__(self, ax, data, horizon):
		self.ax = ax
		self.data = data
		self.horizon = horizon

		# Storage for ALL dynamic elements - organized by type
		self.dynamic_elements = {
			'corridor_patches': [],
			'uncertainty_ellipses': [],
			'constraint_lines': [],
			'obstacle_ellipses': [],
			'debug_points': [],
			'debug_lines': [],
			'feasible_regions': []
		}

		# MPC trajectory elements
		self.mpc_trajectory_line = None
		self.mpc_points = None

		# Vehicle and trajectory elements
		self.vehicle_dot = None
		self.trajectory_line = None

		# Dynamic obstacle patches (separate from static)
		self.obstacle_patches = []

		# Colors
		self.stage_colors = cm.get_cmap('Blues', horizon)

	def clear_all_dynamic_elements(self):
		"""Clear ALL dynamic visualization elements"""
		# Clear all dynamic patches and lines
		for element_type, elements in self.dynamic_elements.items():
			for element in elements:
				try:
					if hasattr(element, 'remove'):
						element.remove()
				except Exception as e:
					# Element might already be removed
					pass
			elements.clear()

		# Clear MPC prediction elements
		if self.mpc_trajectory_line:
			self.mpc_trajectory_line.set_data([], [])
		if self.mpc_points:
			self.mpc_points.set_data([], [])

	def setup_static_elements(self):
		"""Plot all static elements that don't change during MPC"""
		# Boundaries and reference path
		self.ax.plot(self.data.left_boundary_x, self.data.left_boundary_y, 'r--',
					 label='Left boundary', linewidth=2)
		self.ax.plot(self.data.right_boundary_x, self.data.right_boundary_y, 'm--',
					 label='Right boundary', linewidth=2)
		self.ax.plot(self.data.reference_path.x, self.data.reference_path.y, 'k:',
					 label='Reference path', linewidth=1)

		# Start and goal
		self.ax.plot(self.data.goal[0], self.data.goal[1], 'r*', markersize=15, label='Goal')
		self.ax.plot(self.data.start[0], self.data.start[1], 'go', markersize=10, label='Start')

		# Static obstacles
		if hasattr(self.data, "static_obstacles") and self.data.static_obstacles:
			for obs in self.data.static_obstacles:
				circle = plt.Circle((float(obs.position[0]), float(obs.position[1])),
									getattr(obs, "radius", 0.5), color='green',
									alpha=0.6, label='Static Obstacle')
				self.ax.add_patch(circle)

		# Dynamic obstacles (these will be updated)
		self.obstacle_patches = []
		if hasattr(self.data, "dynamic_obstacles") and self.data.dynamic_obstacles:
			for obs in self.data.dynamic_obstacles:
				circle = plt.Circle((float(obs.position[0]), float(obs.position[1])),
									getattr(obs, "radius", 0.5), color='red', alpha=0.6)
				self.ax.add_patch(circle)
				self.obstacle_patches.append(circle)

		# Initialize dynamic trajectory elements
		self.vehicle_dot, = self.ax.plot([], [], 'bo', markersize=10, label="Vehicle")
		self.trajectory_line, = self.ax.plot([], [], 'b-', linewidth=3, label="Actual Trajectory")

		# MPC prediction trajectory
		self.mpc_trajectory_line, = self.ax.plot([], [], 'g--', linewidth=2,
												 alpha=0.7, label="MPC Prediction")
		self.mpc_points, = self.ax.plot([], [], 'go', markersize=4, alpha=0.7)

		self.ax.legend(loc='upper right')

	def plot_halfspace_line(self, a1, a2, b, x_limits, y_limits, color='red', alpha=0.7):
		"""Plot a single halfspace constraint line: a1*x + a2*y = b"""
		x_min, x_max = x_limits
		y_min, y_max = y_limits

		# Find two points on the line a1*x + a2*y = b
		points = []

		if abs(a2) > 1e-8:  # Not vertical line
			# Check intersections with x boundaries
			y_at_xmin = (b - a1 * x_min) / a2
			y_at_xmax = (b - a1 * x_max) / a2

			if y_min <= y_at_xmin <= y_max:
				points.append([x_min, y_at_xmin])
			if y_min <= y_at_xmax <= y_max:
				points.append([x_max, y_at_xmax])

		if abs(a1) > 1e-8:  # Not horizontal line
			# Check intersections with y boundaries
			x_at_ymin = (b - a2 * y_min) / a1
			x_at_ymax = (b - a2 * y_max) / a1

			if x_min <= x_at_ymin <= x_max:
				points.append([x_at_ymin, y_min])
			if x_min <= x_at_ymax <= x_max:
				points.append([x_at_ymax, y_max])

		# Remove duplicate points
		unique_points = []
		for p in points:
			is_duplicate = False
			for up in unique_points:
				if abs(p[0] - up[0]) < 1e-6 and abs(p[1] - up[1]) < 1e-6:
					is_duplicate = True
					break
			if not is_duplicate:
				unique_points.append(p)

		if len(unique_points) >= 2:
			line_x = [p[0] for p in unique_points[:2]]
			line_y = [p[1] for p in unique_points[:2]]
			line, = self.ax.plot(line_x, line_y, color=color, alpha=alpha, linewidth=2)
			return line
		return None

	def update_decomp_visualization(self, decomp_constraints, x_limits, y_limits, solver, current_pos):
		"""Update decomposition visualization - completely redraws constraints"""
		print(f"\n=== UPDATING DECOMP VISUALIZATION AT ({current_pos[0]:.2f}, {current_pos[1]:.2f}) ===")

		try:
			# Method 1: Try to get constraints from current parameter values
			success = self._visualize_from_parameter_manager(decomp_constraints, solver, x_limits, y_limits)

			if not success:
				# Method 2: Try to access stored constraint values
				success = self._visualize_from_stored_constraints(decomp_constraints, x_limits, y_limits)

			# Method 3: Debug visualization (if enabled)
			if not success:
				self._create_debug_decomp_visualization(decomp_constraints, current_pos, x_limits, y_limits)

			# Method 4: Visualize occupied positions
			self._visualize_occupied_positions(decomp_constraints)

		except Exception as e:
			print(f"Error in update_decomp_visualization: {e}")
			import traceback
			traceback.print_exc()

	def _visualize_from_parameter_manager(self, decomp_constraints, solver, x_limits, y_limits):
		"""Try to plot constraints from the parameter manager"""
		try:
			if not hasattr(solver, 'parameter_manager'):
				return False

			param_manager = solver.parameter_manager
			plotted_any = False

			# Try to get constraint parameters for each disc, step, and constraint
			for disc_id in range(decomp_constraints.num_discs):
				for step in range(1, min(6, decomp_constraints.solver.horizon + 1)):  # Limit visualization
					for cons_id in range(decomp_constraints.max_constraints):
						try:
							base_name = decomp_constraints.constraint_name(cons_id, step, disc_id)

							a1_name = f"{base_name}_a1"
							a2_name = f"{base_name}_a2"
							b_name = f"{base_name}_b"

							if (param_manager.has_parameter(a1_name) and
									param_manager.has_parameter(a2_name) and
									param_manager.has_parameter(b_name)):

								a1 = float(param_manager.get_parameter(a1_name))
								a2 = float(param_manager.get_parameter(a2_name))
								b = float(param_manager.get_parameter(b_name))

								# Skip dummy constraints
								if abs(a1) > 1e-6 or abs(a2) > 1e-6:
									color = plt.cm.Set1((step * disc_id + cons_id) % 9)
									line = self.plot_halfspace_line(a1, a2, b, x_limits, y_limits,
																	color=color, alpha=0.6)
									if line:
										self.dynamic_elements['constraint_lines'].append(line)
										plotted_any = True

						except Exception as e:
							continue

			return plotted_any

		except Exception as e:
			print(f"Error in _visualize_from_parameter_manager: {e}")
			return False

	def _visualize_from_stored_constraints(self, decomp_constraints, x_limits, y_limits):
		"""Try to plot from stored constraint values in decomp_constraints"""
		try:
			plotted_any = False

			# Access the stored constraint coefficients
			for disc_id in range(decomp_constraints.num_discs):
				for step in range(1, min(6, decomp_constraints.solver.horizon + 1)):
					for cons_id in range(decomp_constraints.max_constraints):
						try:
							# Get stored values
							a1_val = decomp_constraints._a1[disc_id][step][cons_id]
							a2_val = decomp_constraints._a2[disc_id][step][cons_id]
							b_val = decomp_constraints._b[disc_id][step][cons_id]

							if a1_val is None or a2_val is None or b_val is None:
								continue

							# Convert to float if needed
							a1 = float(a1_val)
							a2 = float(a2_val)
							b = float(b_val)

							# Skip dummy constraints
							if abs(a1) > 1e-6 or abs(a2) > 1e-6:
								color = plt.cm.Set1((step * disc_id + cons_id) % 9)
								line = self.plot_halfspace_line(a1, a2, b, x_limits, y_limits,
																color=color, alpha=0.6)
								if line:
									self.dynamic_elements['constraint_lines'].append(line)
									plotted_any = True
									print(f"Plotted constraint: step={step}, disc={disc_id}, cons={cons_id}")

						except Exception as e:
							continue

			return plotted_any

		except Exception as e:
			print(f"Error in _visualize_from_stored_constraints: {e}")
			return False

	def _visualize_occupied_positions(self, decomp_constraints):
		"""Visualize the occupied positions (obstacles) used by decomp"""
		try:
			if hasattr(decomp_constraints, 'occ_pos') and decomp_constraints.occ_pos:
				occupied_positions = decomp_constraints.occ_pos.get_occupied_positions()
				print(f"Found {len(occupied_positions)} occupied positions")

				# Sample positions for visualization (don't plot all for performance)
				sample_size = min(400, len(occupied_positions))
				if sample_size > 0:
					sample_indices = np.linspace(0, len(occupied_positions) - 1, sample_size, dtype=int)

					for idx in sample_indices:
						pos = occupied_positions[idx]
						point, = self.ax.plot(pos[0], pos[1], 'rx', markersize=3, alpha=0.4)
						self.dynamic_elements['debug_points'].append(point)

		except Exception as e:
			print(f"Error in _visualize_occupied_positions: {e}")

	def _create_debug_decomp_visualization(self, decomp_constraints, current_pos, x_limits, y_limits):
		"""Create a comprehensive debug visualization of the decomposition process"""
		try:
			# 1. Visualize the current position prominently
			current_marker, = self.ax.plot(current_pos[0], current_pos[1], 'co', markersize=12,
										   markeredgecolor='black', markeredgewidth=2, alpha=0.8)
			self.dynamic_elements['debug_points'].append(current_marker)

			# 2. Visualize decomp range if available
			if hasattr(decomp_constraints, 'range'):
				range_val = decomp_constraints.range
				from matplotlib.patches import Rectangle
				rect = Rectangle((current_pos[0] - range_val, current_pos[1] - range_val),
								 2 * range_val, 2 * range_val,
								 facecolor='none', edgecolor='purple',
								 linewidth=2, linestyle='--', alpha=0.7)
				self.ax.add_patch(rect)
				self.dynamic_elements['feasible_regions'].append(rect)

			# 3. Try to visualize ellipsoids and polyhedra if available
			self._debug_visualize_ellipsoids(decomp_constraints)
			self._debug_visualize_polyhedra(decomp_constraints, x_limits, y_limits)

		except Exception as e:
			print(f"Error in _create_debug_decomp_visualization: {e}")

	def _debug_visualize_ellipsoids(self, decomp_constraints):
		"""Debug visualization of ellipsoids"""
		try:
			if hasattr(decomp_constraints, 'decomp_util'):
				decomp_util = decomp_constraints.decomp_util

				# Try different methods to access ellipsoids
				ellipsoid_sources = [
					('get_ellipsoids',
					 lambda: decomp_util.get_ellipsoids() if hasattr(decomp_util, 'get_ellipsoids') else None),
					('ellipsoids_', lambda: getattr(decomp_util, 'ellipsoids_', None)),
					('ellipsoids', lambda: getattr(decomp_util, 'ellipsoids', None)),
				]

				for source_name, source_func in ellipsoid_sources:
					try:
						ellipsoids = source_func()
						if ellipsoids is not None and len(ellipsoids) > 0:
							print(f"Found {len(ellipsoids)} ellipsoids from {source_name}")

							for i, ellipsoid in enumerate(ellipsoids[:5]):  # Limit to 5
								ell = self._plot_single_ellipsoid(ellipsoid, i, source_name)
								if ell:
									self.dynamic_elements['obstacle_ellipses'].append(ell)
							return  # Stop after first successful source

					except Exception as e:
						print(f"Error accessing ellipsoids from {source_name}: {e}")

		except Exception as e:
			print(f"Error in _debug_visualize_ellipsoids: {e}")

	def _plot_single_ellipsoid(self, ellipsoid, index, source):
		"""Plot a single ellipsoid"""
		try:
			# Try different ellipsoid formats
			center = None
			axes = None
			angle = 0

			if hasattr(ellipsoid, 'center') and hasattr(ellipsoid, 'axes'):
				center = ellipsoid.center
				axes = ellipsoid.axes
				angle = getattr(ellipsoid, 'angle', 0)
			elif hasattr(ellipsoid, 'c') and hasattr(ellipsoid, 'E'):
				# Alternative format: center c, shape matrix E
				center = ellipsoid.c
				E = ellipsoid.E
				try:
					# Compute axes from shape matrix
					eigenvals, eigenvecs = np.linalg.eigh(E)
					axes = 1.0 / np.sqrt(eigenvals)  # For ellipsoid (x-c)^T E (x-c) <= 1
					angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
				except:
					axes = [1.0, 1.0]

			if center is not None and axes is not None and len(center) >= 2 and len(axes) >= 2:
				width = 2 * axes[0]
				height = 2 * axes[1]

				color = plt.cm.Set1(index % 9)
				ell = Ellipse(xy=(center[0], center[1]),
							  width=width, height=height, angle=angle,
							  edgecolor=color, facecolor=color,
							  alpha=0.3, linewidth=2)
				self.ax.add_patch(ell)

				print(f"Plotted ellipsoid {index}: center=({center[0]:.2f}, {center[1]:.2f}), "
					  f"axes=({axes[0]:.2f}, {axes[1]:.2f})")
				return ell

		except Exception as e:
			print(f"Error plotting ellipsoid {index}: {e}")
		return None

	def _debug_visualize_polyhedra(self, decomp_constraints, x_limits, y_limits):
		"""Debug visualization of polyhedra"""
		try:
			if hasattr(decomp_constraints, 'decomp_util'):
				decomp_util = decomp_constraints.decomp_util

				# Try to get polyhedra
				polyhedra_sources = [
					('get_polyhedrons',
					 lambda: decomp_util.get_polyhedrons() if hasattr(decomp_util, 'get_polyhedrons') else None),
					('polyhedra_', lambda: getattr(decomp_util, 'polyhedra_', None)),
					('polyhedra', lambda: getattr(decomp_util, 'polyhedra', None)),
				]

				for source_name, source_func in polyhedra_sources:
					try:
						polyhedra = source_func()
						if polyhedra is not None and len(polyhedra) > 0:
							print(f"Found {len(polyhedra)} polyhedra from {source_name}")

							for i, poly in enumerate(polyhedra[:3]):  # Limit to 3
								self._plot_single_polyhedron(poly, i, x_limits, y_limits, source_name)
							return  # Stop after first successful source

					except Exception as e:
						print(f"Error accessing polyhedra from {source_name}: {e}")

		except Exception as e:
			print(f"Error in _debug_visualize_polyhedra: {e}")

	def _plot_single_polyhedron(self, poly, index, x_limits, y_limits, source):
		"""Plot a single polyhedron"""
		try:
			A, b = None, None

			# Try different polyhedron formats
			if hasattr(poly, 'A_') and hasattr(poly, 'b_'):
				A, b = poly.A_, poly.b_
			elif hasattr(poly, 'A') and hasattr(poly, 'b'):
				A, b = poly.A, poly.b
			elif hasattr(poly, 'constraints'):
				# Try to extract from constraints
				constraints = poly.constraints
				if hasattr(constraints, 'A') and hasattr(constraints, 'b'):
					A, b = constraints.A, constraints.b

			if A is not None and b is not None:
				A = np.array(A)
				b = np.array(b)

				print(f"Polyhedron {index} from {source}: A shape {A.shape}, b shape {b.shape}")

				# Plot constraint lines
				for i in range(A.shape[0]):
					if A.shape[1] >= 2:
						a1, a2 = float(A[i, 0]), float(A[i, 1])
						b_val = float(b[i])

						if abs(a1) > 1e-8 or abs(a2) > 1e-8:
							color = plt.cm.Set1((index * 3 + i) % 9)
							line = self.plot_halfspace_line(a1, a2, b_val, x_limits, y_limits,
															color=color, alpha=0.8)
							if line:
								self.dynamic_elements['constraint_lines'].append(line)

				# Try to create feasible region
				feasible_region = self._create_polyhedron_visualization(A, b, x_limits, y_limits, index)
				if feasible_region:
					self.dynamic_elements['feasible_regions'].append(feasible_region)

		except Exception as e:
			print(f"Error plotting polyhedron {index}: {e}")

	def _clip_polygon_by_halfspace(self, polygon, a1, a2, b):
		"""Clip polygon by halfspace a1*x + a2*y <= b using Sutherland-Hodgman algorithm"""
		if not polygon:
			return []

		clipped = []

		for i in range(len(polygon)):
			current_vertex = polygon[i]
			previous_vertex = polygon[i - 1]

			# Distance from vertices to the line (positive = outside halfspace)
			current_dist = a1 * current_vertex[0] + a2 * current_vertex[1] - b
			previous_dist = a1 * previous_vertex[0] + a2 * previous_vertex[1] - b

			if current_dist <= 1e-9:  # Current vertex is inside
				if previous_dist > 1e-9:  # Previous was outside, add intersection
					t = previous_dist / (previous_dist - current_dist)
					intersection = [
						previous_vertex[0] + t * (current_vertex[0] - previous_vertex[0]),
						previous_vertex[1] + t * (current_vertex[1] - previous_vertex[1])
					]
					clipped.append(intersection)
				clipped.append(current_vertex)
			elif previous_dist <= 1e-9:  # Previous was inside, current outside
				t = previous_dist / (previous_dist - current_dist)
				intersection = [
					previous_vertex[0] + t * (current_vertex[0] - previous_vertex[0]),
					previous_vertex[1] + t * (current_vertex[1] - previous_vertex[1])
				]
				clipped.append(intersection)

		return clipped

	def _calculate_polygon_area(self, vertices):
		"""Calculate area of polygon using shoelace formula"""
		if len(vertices) < 3:
			return 0

		area = 0
		n = len(vertices)
		for i in range(n):
			j = (i + 1) % n
			area += vertices[i][0] * vertices[j][1]
			area -= vertices[j][0] * vertices[i][1]
		return abs(area) / 2

	def _create_polyhedron_visualization(self, A, b, x_limits, y_limits, step_idx):
		"""Create polygon visualization from polyhedron Ax <= b"""
		try:
			# Start with bounding box
			margin = 0.5
			x_min, x_max = x_limits[0] + margin, x_limits[1] - margin
			y_min, y_max = y_limits[0] + margin, y_limits[1] - margin

			subject_polygon = [
				[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]
			]

			# Apply each constraint
			for i in range(A.shape[0]):
				a1, a2, b_val = float(A[i, 0]), float(A[i, 1]), float(b[i])
				subject_polygon = self._clip_polygon_by_halfspace(subject_polygon, a1, a2, b_val)

				if len(subject_polygon) < 3:
					break

			# Create polygon patch if feasible
			if len(subject_polygon) >= 3:
				area = self._calculate_polygon_area(subject_polygon)
				if area > 0.01:  # Minimum area threshold
					alpha = max(0.1, 0.4 - step_idx * 0.05)
					color = self.stage_colors(step_idx / max(1, self.horizon))

					poly_patch = MPLPolygon(subject_polygon, closed=True,
											facecolor=color, alpha=alpha,
											edgecolor='darkblue', linewidth=1,
											zorder=2, label=f'Feasible Region {step_idx}' if step_idx <= 2 else "")
					self.ax.add_patch(poly_patch)

					LOG_DEBUG(f"Created feasible region for step {step_idx}, area={area:.3f}")
					return poly_patch
		except Exception as e:
			print(f"Error creating polyhedron visualization: {e}")
		return None

	def update_uncertainty_ellipses(self):
		"""Update uncertainty ellipses for dynamic obstacles"""
		chi_sq_val = 5.991  # 95% confidence for 2D

		if hasattr(self.data, "dynamic_obstacles") and self.data.dynamic_obstacles:
			for obs in self.data.dynamic_obstacles:
				if hasattr(obs, 'prediction') and obs.prediction and obs.prediction.type == "GAUSSIAN":
					for step in obs.prediction.steps:
						if hasattr(step, 'covariance') and step.covariance is not None:
							cov = step.covariance
							vals, vecs = np.linalg.eigh(cov)
							order = vals.argsort()[::-1]
							vals, vecs = vals[order], vecs[:, order]

							angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
							width = 2 * np.sqrt(chi_sq_val * vals[0])
							height = 2 * np.sqrt(chi_sq_val * vals[1])

							ell = Ellipse(xy=(step.position[0], step.position[1]),
										  width=width, height=height, angle=angle,
										  edgecolor='blue', facecolor='none',
										  alpha=0.3, linewidth=2)
							self.ax.add_patch(ell)
							self.dynamic_elements['uncertainty_ellipses'].append(ell)

	def update_mpc_prediction(self, planner_output):
		"""Update MPC prediction trajectory"""
		if planner_output.success and hasattr(planner_output, 'trajectory_history'):
			trajectory = planner_output.trajectory_history[-1]  # Latest trajectory
			states = trajectory.get_states()

			if len(states) > 1:
				pred_x = [state.get("x") for state in states[1:]]  # Skip current state
				pred_y = [state.get("y") for state in states[1:]]

				self.mpc_trajectory_line.set_data(pred_x, pred_y)
				self.mpc_points.set_data(pred_x, pred_y)
			else:
				self.mpc_trajectory_line.set_data([], [])
				self.mpc_points.set_data([], [])
		else:
			self.mpc_trajectory_line.set_data([], [])
			self.mpc_points.set_data([], [])

	def update_dynamic_elements(self, current_pos, states_x, states_y,
								decomp_constraints, planner_output, x_limits, y_limits,
								iteration, solver):
		"""Update all dynamic visualization elements with proper clear/redraw"""
		# STEP 1: Clear all previous dynamic elements
		self.clear_all_dynamic_elements()

		# STEP 2: Update vehicle position and trajectory (these are line objects, not patches)
		self.vehicle_dot.set_data([current_pos[0]], [current_pos[1]])
		self.trajectory_line.set_data(states_x, states_y)

		# STEP 3: Update MPC prediction
		self.update_mpc_prediction(planner_output)

		# STEP 4: Update obstacle positions if they move
		if hasattr(self.data, "dynamic_obstacles") and self.data.dynamic_obstacles:
			for idx, obs in enumerate(self.data.dynamic_obstacles):
				if idx < len(self.obstacle_patches):
					if hasattr(obs, 'update_position'):
						obs.update_position(self.data.planning_start_time)
					self.obstacle_patches[idx].center = (float(obs.position[0]),
														 float(obs.position[1]))

		# STEP 5: Update decomposition visualization (this creates new elements)
		self.update_decomp_visualization(decomp_constraints, x_limits, y_limits, solver, current_pos)

		# STEP 6: Update uncertainty ellipses (this creates new elements)
		self.update_uncertainty_ellipses()

		# STEP 7: Force matplotlib to redraw
		self.ax.figure.canvas.draw_idle()


def run(dt=0.1, horizon=10, model=ContouringSecondOrderUnicycleModel, start=(0.0, 0.0), goal=(20.0, 20.0),
		max_iterations=200):

	casadi_solver = CasADiSolver(dt, horizon)
	vehicle = model()
	casadi_solver.set_dynamics_model(vehicle)

	# Create the planner
	planner = Planner(casadi_solver, vehicle)

	contouring_objective = ContouringObjective(casadi_solver)
	casadi_solver.module_manager.add_module(contouring_objective)
	contouring_constraints = ContouringConstraints(casadi_solver)
	casadi_solver.module_manager.add_module(contouring_constraints)
	decomp_constraints = DecompConstraints(casadi_solver)
	casadi_solver.module_manager.add_module(decomp_constraints)

	data = Data()
	data.start = np.array(start)
	data.goal = np.array(goal)
	data.goal_received = True
	data.planning_start_time = 0.0

	reference_path = generate_reference_path(data.start, data.goal, path_type="curved")
	data.reference_path = reference_path

	dynamic_obstacles = generate_dynamic_obstacles(10, GAUSSIAN, 1)
	data.dynamic_obstacles = dynamic_obstacles

	costmap = Costmap(width=400, height=400, resolution=0.05, origin=[0, 0])
	obst_pos = [obs.position for obs in data.dynamic_obstacles]
	LOG_DEBUG("Obstacle positions: " + str(obst_pos))
	costmap.set_obstacles(obst_pos)
	costmap.inflate_obstacles(1)
	data.costmap = costmap

	normals = calculate_path_normals(data.reference_path)

	# Road width setup
	road_width = data.road_width if hasattr(data, 'road_width') and data.road_width is not None else 8.0
	half_width = road_width / 2
	quarter_width = half_width / 2

	# Generate boundaries
	left_x, left_y, right_x, right_y = [], [], [], []

	for i in range(len(data.reference_path.x)):
		nx, ny = normals[i]
		left_x.append(data.reference_path.x[i] + nx * quarter_width)
		left_y.append(data.reference_path.y[i] + ny * quarter_width)
		right_x.append(data.reference_path.x[i] - nx * quarter_width)
		right_y.append(data.reference_path.y[i] - ny * quarter_width)

	# Store boundary data
	data.left_boundary_x = left_x
	data.left_boundary_y = left_y
	data.right_boundary_x = right_x
	data.right_boundary_y = right_y

	# Create splines
	left_boundary_spline_x = CubicSpline(data.reference_path.s, np.array(left_x))
	left_boundary_spline_y = CubicSpline(data.reference_path.s, np.array(left_y))
	right_boundary_spline_x = CubicSpline(data.reference_path.s, np.array(right_x))
	right_boundary_spline_y = CubicSpline(data.reference_path.s, np.array(right_y))

	data.left_spline_x = left_boundary_spline_x
	data.left_spline_y = left_boundary_spline_y
	data.right_spline_x = right_boundary_spline_x
	data.right_spline_y = right_boundary_spline_y

	data.left_bound = Bound(left_x, left_y, data.reference_path.s)
	data.right_bound = Bound(right_x, right_y, data.reference_path.s)

	data.robot_area = define_robot_area(vehicle.length, vehicle.width, 1)

	# Add solver timeout parameter
	casadi_solver.parameter_manager.add("solver_timeout", 10.0)

	# Create initial state
	state = State(model())
	state.set("x", start[0])
	state.set("y", start[1])
	state.set("psi", 0.1)
	state.set("v", 0.0)
	state.set("spline", 0.0)
	state.set("a", 0.0)
	state.set("w", 0.0)
	planner.set_state(state)

	planner.initialize(data)

	# Initialize plot
	matplotlib.use('TkAgg')
	plt.ion()
	fig, ax = plt.subplots(figsize=(14, 10))
	ax.set_title("MPC with Contouring Objective & Decomposition Constraints")
	ax.set_xlabel("X [m]")
	ax.set_ylabel("Y [m]")
	ax.grid(True, alpha=0.3)
	ax.set_aspect('equal')

	# Calculate plot limits
	all_x = left_x + right_x + [start[0], goal[0]]
	all_y = left_y + right_y + [start[1], goal[1]]

	if hasattr(data, "dynamic_obstacles") and data.dynamic_obstacles:
		for obs in data.dynamic_obstacles:
			all_x.append(float(obs.position[0]))
			all_y.append(float(obs.position[1]))

	margin = 3.0
	x_min, x_max = min(all_x) - margin, max(all_x) + margin
	y_min, y_max = min(all_y) - margin, max(all_y) + margin

	ax.set_xlim(x_min, x_max)
	ax.set_ylim(y_min, y_max)

	# Initialize visualizer
	visualizer = MPCVisualizer(ax, data, horizon)
	visualizer.setup_static_elements()

	plt.draw()
	plt.pause(0.1)

	# Main simulation loop
	success_flags = []
	control_inputs = []
	states_x, states_y = [start[0]], [start[1]]

	print("Starting MPC simulation...")

	for i in range(max_iterations):
		data.planning_start_time = i * dt
		output = planner.solve_mpc(data)
		success_flags.append(output.success)

		if output.success:
			# Get control inputs
			next_a = output.trajectory_history[-1].get_states()[1].get("a")
			next_w = output.trajectory_history[-1].get_states()[1].get("w")

			# Simulate one step forward
			current_state = planner.get_state()
			z_k = [next_a, next_w, current_state.get("x"), current_state.get("y"),
				   current_state.get("psi"), current_state.get("v"), current_state.get("spline")]
			z_k = ca.vertcat(*z_k)
			vehicle.load(z_k)

			next_state_symbolic = vehicle.discrete_dynamics(z_k, casadi_solver.parameter_manager,
															casadi_solver.timestep)
			next_state = numeric_rk4(next_state_symbolic, vehicle, casadi_solver.parameter_manager,
									 casadi_solver.timestep)

			next_x, next_y = float(next_state[0]), float(next_state[1])
			next_psi, next_v, next_spline = float(next_state[2]), float(next_state[3]), float(next_state[4])

			# Update planner state
			new_state = planner.get_state().copy()
			new_state.set("x", next_x)
			new_state.set("y", next_y)
			new_state.set("psi", next_psi)
			new_state.set("v", next_v)
			new_state.set("w", next_w)
			new_state.set("a", next_a)
			new_state.set("spline", next_spline)
			planner.set_state(new_state)

			# Update trajectory history
			states_x.append(next_x)
			states_y.append(next_y)
			control_inputs.append([next_a, next_w])

			# Update visualization
			current_pos = [next_x, next_y]
			visualizer.update_dynamic_elements(current_pos, states_x, states_y,
											   decomp_constraints, output,
											   (x_min, x_max), (y_min, y_max), i, casadi_solver)

			# Force plot update
			fig.canvas.draw()
			fig.canvas.flush_events()
			plt.pause(0.02)  # Smaller pause for smoother animation

			# Check if goal reached
			if planner.is_objective_reached(data):
				print(f"Goal reached at iteration {i}!")
				break

			# Progress indicator
			if i % 10 == 0:
				print(f"Iteration {i}/{max_iterations}, Position: ({next_x:.2f}, {next_y:.2f})")

		else:
			print(f"Iteration {i}: MPC failed!")

		casadi_solver.reset()

	# Keep plot open at the end
	plt.ioff()
	print(f"Simulation completed. Success rate: {sum(success_flags)}/{len(success_flags)}")
	plt.show()

	LOG_DEBUG("Control input history: " + str(control_inputs))
	LOG_DEBUG("Forecasts: " + str(planner.solver.get_forecasts()))

	return data, planner.output.realized_trajectory, planner.solver.get_forecasts(), success_flags


def test():
	import logging

	# Configure logger
	logger = logging.getLogger("root")
	logger.setLevel(logging.DEBUG)

	data, realized_trajectory, trajectory_history, success_flags = run(dt=0.1, horizon=10, max_iterations=100)


if __name__ == "__main__":
	test()