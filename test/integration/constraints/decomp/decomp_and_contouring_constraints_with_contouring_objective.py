import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.interpolate import CubicSpline

from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
from planner_modules.src.constraints.decomp_constraints import DecompConstraints
from planner_modules.src.objectives.contouring_objective import ContouringObjective
from planning.src.data_prep import define_robot_area
from planning.src.dynamic_models import ContouringSecondOrderUnicycleModel, numeric_rk4
from planning.src.planner import Planner
from planning.src.types import Data, Bound, generate_reference_path, calculate_path_normals, State, \
	generate_dynamic_obstacles, generate_static_obstacles, Costmap
from solver.src.casadi_solver import CasADiSolver
from utils.const import GAUSSIAN
from utils.utils import LOG_DEBUG


def run(dt=0.1, horizon=10, model=ContouringSecondOrderUnicycleModel, start=(0.0, 0.0), goal=(20.0, 20.0),
		max_iterations=200):


	dt = dt
	horizon = horizon

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
	# Store path
	data.reference_path = reference_path

	dynamic_obstacles = generate_dynamic_obstacles(10, GAUSSIAN, 1)

	data.dynamic_obstacles = dynamic_obstacles

	data.static_obstacles = generate_static_obstacles(1, 1, reference_path)


	costmap = Costmap(width=400, height=400, resolution=0.05, origin=[-10, -10])
	obst_pos = [obs.position for obs in data.dynamic_obstacles]
	costmap.set_obstacles(obst_pos)
	costmap.inflate_obstacles(0.5)

	# Pass to DecompConstraints
	data.costmap = costmap


	normals = calculate_path_normals(data.reference_path)

	# Road width (adjust based on your actual data structure)
	road_width = data.road_width if hasattr(data, 'road_width') and data.road_width is not None else 8.0
	half_width = road_width / 2
	quarter_width = half_width / 2

	# Generate left and right boundaries
	left_x = []
	left_y = []
	right_x = []
	right_y = []

	for i in range(len(data.reference_path.x)):
		nx, ny = normals[i]

		# Left boundary (offset in the positive normal direction)
		left_x.append(data.reference_path.x[i] + nx * quarter_width)
		left_y.append(data.reference_path.y[i] + ny * quarter_width)

		# Right boundary (offset in the negative normal direction)
		right_x.append(data.reference_path.x[i] - nx * quarter_width)
		right_y.append(data.reference_path.y[i] - ny * quarter_width)

	# Create splines for the boundaries if needed
	left_boundary_spline_x = CubicSpline(data.reference_path.s, np.array(left_x))
	left_boundary_spline_y = CubicSpline(data.reference_path.s, np.array(left_y))
	right_boundary_spline_x = CubicSpline(data.reference_path.s, np.array(right_x))
	right_boundary_spline_y = CubicSpline(data.reference_path.s, np.array(right_y))

	# Store boundary data
	data.left_boundary_x = left_x
	data.left_boundary_y = left_y
	data.right_boundary_x = right_x
	data.right_boundary_y = right_y

	# Store the spline functions themselves if needed for later evaluation
	data.left_spline_x = left_boundary_spline_x
	data.left_spline_y = left_boundary_spline_y
	data.right_spline_x = right_boundary_spline_x
	data.right_spline_y = right_boundary_spline_y

	# Add boundaries to data
	data.left_bound = Bound(left_x, left_y, data.reference_path.s)
	data.right_bound = Bound(right_x, right_y, data.reference_path.s)

	data.robot_area = define_robot_area(vehicle.length, vehicle.width, 1)

	# Add solver timeout parameter
	casadi_solver.parameter_manager.add("solver_timeout", 10.0)

	# Create initial state - make sure to match the model's state variables
	state = State(model())
	state.set("x", start[0])
	state.set("y", start[1])
	state.set("psi", 0.1)
	state.set("v", 0.0)
	state.set("spline", 0.0)
	state.set("a", 0.0)
	state.set("w", 0.0)
	planner.set_state(state)

	# Define parameters for the goal objective
	planner.initialize(data)

	# ✅ Initialize real-time plot
	matplotlib.use('TkAgg')
	plt.ion()
	fig, ax = plt.subplots(figsize=(12, 10))
	ax.set_title("MPC with Contouring Objective & Ellipsoidal Obstacle Constraints")
	ax.set_xlabel("X [m]")
	ax.set_ylabel("Y [m]")
	ax.grid(True)
	ax.set_aspect('equal')

	# ✅ CRITICAL: Set fixed axis limits based on your problem domain
	# Calculate appropriate limits based on path and obstacles
	all_x = left_x + right_x + [start[0], goal[0]]
	all_y = left_y + right_y + [start[1], goal[1]]

	if hasattr(data, "dynamic_obstacles") and data.dynamic_obstacles:
		for obs in data.dynamic_obstacles:
			all_x.append(float(obs.position[0]))
			all_y.append(float(obs.position[1]))

	if hasattr(data, "dynamic_obstacles") and data.dynamic_obstacles:
		for obs in data.dynamic_obstacles:
			all_x.append(float(obs.position[0]))
			all_y.append(float(obs.position[1]))

	margin = 2.0  # Add some margin
	x_min, x_max = min(all_x) - margin, max(all_x) + margin
	y_min, y_max = min(all_y) - margin, max(all_y) + margin

	ax.set_xlim(x_min, x_max)
	ax.set_ylim(y_min, y_max)

	# Static elements
	ax.plot(data.left_boundary_x, data.left_boundary_y, 'r--', label='Left boundary', linewidth=2)
	ax.plot(data.right_boundary_x, data.right_boundary_y, 'm--', label='Right boundary', linewidth=2)
	ax.plot(data.reference_path.x, data.reference_path.y, 'k:', label='Reference path', linewidth=1)
	ax.plot(data.goal[0], data.goal[1], 'r*', markersize=15, label='Goal')
	ax.plot(data.start[0], data.start[1], 'go', markersize=10, label='Start')

	# ✅ Dynamic artists
	vehicle_dot, = ax.plot([], [], 'bo', markersize=10, label="Vehicle")
	trajectory_line, = ax.plot([], [], 'b-', linewidth=3, label="Trajectory")

	# ✅ Obstacles as patches
	obstacle_patches = []
	if hasattr(data, "dynamic_obstacles") and data.dynamic_obstacles:
		for obs in data.dynamic_obstacles:
			circle = plt.Circle((float(obs.position[0]), float(obs.position[1])),
								getattr(obs, "radius", 0.5), color='red', alpha=0.6)
			ax.add_patch(circle)
			obstacle_patches.append(circle)

	static_obstacle_patches = []
	if hasattr(data, "static_obstacles") and data.static_obstacles:
		for obs in data.static_obstacles:
			circle = plt.Circle((float(obs.position[0]), float(obs.position[1])),
								getattr(obs, "radius", 0.5), color='green', alpha=0.6, label='Static Obstacle')
			ax.add_patch(circle)
			static_obstacle_patches.append(circle)

	ax.legend(loc='upper right')

	# ✅ Ellipsoid patches
	ellipsoid_patches = []
	for _ in data.dynamic_obstacles:
		ell = Ellipse((0, 0), 0, 0, edgecolor='purple', facecolor='none', linewidth=2, alpha=0.6)
		ax.add_patch(ell)
		ellipsoid_patches.append(ell)

	ax.legend()
	plt.draw()
	plt.pause(0.1)

	prediction_lines = []
	prediction_markers = []
	uncertainty_ellipses = []

	for _ in data.dynamic_obstacles:
		# Dashed red line for predicted path
		line, = ax.plot([], [], 'r--', linewidth=1.2, alpha=0.6)
		marker, = ax.plot([], [], 'ro', markersize=3, alpha=0.5)
		prediction_lines.append(line)
		prediction_markers.append(marker)

		# Ellipse for uncertainty (will only show for GAUSSIAN type)
		ell = Ellipse((0, 0), 0, 0, edgecolor='blue', facecolor='none', alpha=0.3)
		ax.add_patch(ell)
		uncertainty_ellipses.append(ell)

	# ✅ Initial draw
	plt.draw()
	plt.pause(0.1)

	success_flags = []
	control_inputs = []
	states_x, states_y = [start[0]], [start[1]]

	# ✅ Real-time update inside loop
	for i in range(max_iterations):
		data.planning_start_time = i * dt
		output = planner.solve_mpc(data)
		success_flags.append(output.success)

		if output.success:
			next_a = output.trajectory_history[-1].get_states()[1].get("a")
			next_w = output.trajectory_history[-1].get_states()[1].get("w")

			current_state = planner.get_state()
			z_k = [next_a, next_w, current_state.get("x"), current_state.get("y"),
				   current_state.get("psi"), current_state.get("v"), current_state.get("spline")]
			z_k = ca.vertcat(*z_k)
			vehicle.load(z_k)

			next_state_symbolic = vehicle.discrete_dynamics(z_k, casadi_solver.parameter_manager,
															casadi_solver.timestep)
			next_state = numeric_rk4(next_state_symbolic, vehicle, casadi_solver.parameter_manager,
									 casadi_solver.timestep)

			next_x, next_y, next_psi, next_v, next_spline = float(next_state[0]), float(next_state[1]), float(
				next_state[2]), float(next_state[3]), float(next_state[4])

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

			# ✅ Update lists
			states_x.append(next_x)
			states_y.append(next_y)

			# ✅ Update plot elements with explicit data
			vehicle_dot.set_data([next_x], [next_y])
			trajectory_line.set_data(states_x, states_y)
			LOG_DEBUG("Vehicle now set to : " + str([next_x, next_y]))
			# ✅ Update obstacles if they move (only if they actually have dynamic positions)
			if hasattr(data, "dynamic_obstacles") and data.dynamic_obstacles:
				for idx, obs in enumerate(data.dynamic_obstacles):
					if idx < len(obstacle_patches):
						# Only update if obstacle has a position update method
						if hasattr(obs, 'update_position'):
							obs.update_position(data.planning_start_time)
						obstacle_patches[idx].center = (float(obs.position[0]), float(obs.position[1]))

			# ✅ Force plot update
			from shapely.geometry import Polygon, box
			from shapely.ops import unary_union
			import matplotlib.cm as cm
			import math

			# Define chi-square quantile for 95% confidence
			chi_sq_val = 5.991  # df=2, alpha=0.95

			def plot_corridors_and_ellipses(ax, decomp_constraints, horizon, x_min, x_max, y_min, y_max, stage_colors):
				# Remove old corridors
				for artist in getattr(ax, "_corridor_polygons", []):
					artist.remove()
				ax._corridor_polygons = []

				# Base rectangle
				base_rect = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])

				for k in range(horizon):
					halfspaces = []
					for idx in range(decomp_constraints.num_constraints):
						a1 = decomp_constraints.a1[0][k][idx]
						a2 = decomp_constraints.a2[0][k][idx]
						b = decomp_constraints.b[0][k][idx]
						if abs(a1) < 1e-8 and abs(a2) < 1e-8:
							continue
						halfspaces.append((a1, a2, b))

					# Start with the bounding box
					region = base_rect
					for (a1, a2, b) in halfspaces:
						# Create a large polygon representing the half-space a1*x + a2*y <= b
						points = []
						# Four corners of an extended bounding box
						for px, py in [(x_min - 100, y_min - 100), (x_max + 100, y_min - 100),
									   (x_max + 100, y_max + 100), (x_min - 100, y_max + 100)]:
							if a1 * px + a2 * py <= b + 1e-9:
								points.append((px, py))
						# Approximate half-space by cutting the rectangle with its boundary line
						line_points = []
						if abs(a2) > 1e-8:
							# y = (b - a1*x)/a2
							line_points.append((x_min - 100, (b - a1 * (x_min - 100)) / a2))
							line_points.append((x_max + 100, (b - a1 * (x_max + 100)) / a2))
						if abs(a1) > 1e-8:
							# x = (b - a2*y)/a1
							line_points.append(((b - a2 * (y_min - 100)) / a1, y_min - 100))
							line_points.append(((b - a2 * (y_max + 100)) / a1, y_max + 100))
						points += line_points

						# Sort points and make a polygon
						halfspace_poly = Polygon(points).convex_hull
						region = region.intersection(halfspace_poly)

					if not region.is_empty:
						color = stage_colors(k)
						poly_patch = plt.Polygon(list(region.exterior.coords), closed=True, color=color, alpha=0.15,
												 zorder=1)
						ax.add_patch(poly_patch)
						ax._corridor_polygons.append(poly_patch)

			def plot_uncertainty_ellipses(ax, dynamic_obstacles):
				for ell in getattr(ax, "_uncertainty_ellipses", []):
					ell.remove()
				ax._uncertainty_ellipses = []

				for obs in dynamic_obstacles:
					if obs.prediction and obs.prediction.type == "GAUSSIAN":
						for step in obs.prediction.steps:
							cov = step.covariance
							if cov is None: continue
							vals, vecs = np.linalg.eigh(cov)
							order = vals.argsort()[::-1]
							vals, vecs = vals[order], vecs[:, order]
							angle = math.degrees(math.atan2(*vecs[:, 0][::-1]))
							width, height = 2 * np.sqrt(chi_sq_val * vals)
							ell = Ellipse(xy=(step.position[0], step.position[1]),
										  width=width, height=height, angle=angle,
										  edgecolor='blue', facecolor='none', alpha=0.3, lw=2)
							ax.add_patch(ell)
							ax._uncertainty_ellipses.append(ell)

			stage_colors = cm.get_cmap('Greens', horizon)
			plot_corridors_and_ellipses(ax, decomp_constraints, horizon, x_min, x_max, y_min, y_max, stage_colors)
			plot_uncertainty_ellipses(ax, data.dynamic_obstacles)

			# chi = chi_square_quantile(dof=2, alpha=1.0 - ellipsoid_constraints.risk)
			# ego_radius = data.robot_area[0].radius if hasattr(data, 'robot_area') else 0.0
			#
			# for idx, obs in enumerate(data.dynamic_obstacles):
			# 	obstacle_patches[idx].center = (obs.position[0], obs.position[1])
			# 	pred = obs.prediction
			# 	if pred and len(pred.steps) > 0 and pred.type in [PredictionType.GAUSSIAN, PredictionType.NONGAUSSIAN]:
			# 		step = pred.steps[0]
			# 		major = (step.major_radius + obs.radius + ego_radius) * np.sqrt(chi)
			# 		minor = (step.minor_radius + obs.radius + ego_radius) * np.sqrt(chi)
			# 		ellipsoid_patches[idx].center = (step.position[0], step.position[1])
			# 		ellipsoid_patches[idx].width = 2 * major
			# 		ellipsoid_patches[idx].height = 2 * minor
			# 		ellipsoid_patches[idx].angle = np.degrees(step.angle)
			# 		ellipsoid_patches[idx].set_visible(True)
			# 	else:
			# 		ellipsoid_patches[idx].set_visible(False)
			# ✅ Force refresh
			fig.canvas.draw()
			fig.canvas.flush_events()
			plt.pause(0.05)

			if planner.is_objective_reached(data):
				print(f"Objective reached at iteration {i}!")
				break
		else:
			print(f"Iteration {i}: MPC failed!")

		casadi_solver.reset()

	# ✅ Keep plot open at the end
	plt.ioff()
	plt.show()

	LOG_DEBUG("Control input history: " + str(control_inputs))
	LOG_DEBUG("Forecasts: " + str(planner.solver.get_forecasts()))

	return data, planner.output.realized_trajectory, planner.solver.get_forecasts(), success_flags

def test():
	import logging

	# Configuring the logger
	logger = logging.getLogger("root")
	logger.setLevel(logging.DEBUG)

	data, realized_trajectory, trajectory_history, success_flags = run()