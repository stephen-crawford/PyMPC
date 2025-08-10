import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.patches import Ellipse
from scipy.interpolate import CubicSpline

from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
from planner_modules.src.constraints.gaussian_constraints import GaussianConstraints
from planner_modules.src.constraints.linearized_constraints import LinearizedConstraints
from planner_modules.src.objectives.contouring_objective import ContouringObjective
from planning.src.data_prep import define_robot_area
from planning.src.dynamic_models import ContouringSecondOrderUnicycleModel, numeric_rk4
from planning.src.planner import Planner
from planning.src.types import Data, Bound, generate_reference_path, calculate_path_normals, State, \
	generate_dynamic_obstacles, PredictionType
from solver.src.casadi_solver import CasADiSolver
from utils.const import GAUSSIAN
from utils.utils import LOG_DEBUG


def run(dt=0.1, horizon=15, model=ContouringSecondOrderUnicycleModel, start=(0.0, 0.0), goal=(20.0, 20.0),
		max_iterations=300):
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
	gaussian_constraints = GaussianConstraints(casadi_solver)
	casadi_solver.module_manager.add_module(gaussian_constraints)

	data = Data()
	data.start = np.array(start)
	data.goal = np.array(goal)
	data.goal_received = True
	data.planning_start_time = 0.0

	reference_path = generate_reference_path(data.start, data.goal, path_type="curved")
	# Store path
	data.reference_path = reference_path

	dynamic_obstacles = generate_dynamic_obstacles(10, GAUSSIAN, .5)

	data.dynamic_obstacles = dynamic_obstacles

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
	ax.set_title("MPC with Contouring Objective & Gaussian Constraints")
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
	gaussian_ellipses = []
	gaussian_annotations = []  # NEW: keep track of labels

	if hasattr(data, "dynamic_obstacles") and data.dynamic_obstacles:
		for obs in data.dynamic_obstacles:
			circle = plt.Circle((float(obs.position[0]), float(obs.position[1])),
								getattr(obs, "radius", 0.5), color='black', alpha=0.6)
			ax.add_patch(circle)
			obstacle_patches.append(circle)

	ax.legend(loc='upper right')

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

			# ✅ Update obstacles if they move
			if hasattr(data, "dynamic_obstacles") and data.dynamic_obstacles:
				for idx, obs in enumerate(data.dynamic_obstacles):
					if idx < len(obstacle_patches):
						# Only update if obstacle has a position update method
						if hasattr(obs, 'update_position'):
							obs.update_position(data.planning_start_time)
						obstacle_patches[idx].center = (float(obs.position[0]), float(obs.position[1]))

			# ✅ ENHANCED GAUSSIAN CONSTRAINT VISUALIZATION
				# ✅ UPDATED CONSTRAINT VISUALIZATION
				current_k = 1  # Use k=1 for constraint visualization
				disc_id = 0

				print(f"\n{'=' * 60}")
				print(f"ITERATION {i}: Enhanced Gaussian Constraint Visualization")
				print(f"{'=' * 60}")

				# Clear previous Gaussian ellipses
				for ellipse in gaussian_ellipses:
					if hasattr(ellipse, 'remove'):
						ellipse.remove()
					else:
						# Handle matplotlib artists that don't have remove method
						try:
							ellipse.set_visible(False)
						except:
							pass
				gaussian_ellipses.clear()

				for ann in gaussian_annotations:
					try:
						ann.remove()
					except:
						pass
				gaussian_annotations.clear()

				# Use the enhanced visualization
				constraint_info = visualize_gaussian_constraints_enhanced(
					ax, gaussian_constraints, data, new_state, casadi_solver,
					disc_id, current_k, gaussian_ellipses, gaussian_annotations)

				# Update title with detailed constraint information
				ax.set_title(f"MPC with Gaussian Constraints - Iteration {i}\n{constraint_info}")

				# ✅ Force refresh
				fig.canvas.draw()
				fig.canvas.flush_events()
				plt.pause(0.15)  # Slightly longer pause for better visualization

				if planner.is_objective_reached(data):
					print(f"Objective reached at iteration {i}!")
					break
		else:
			print(f"Iteration {i}: MPC failed!")

		casadi_solver.reset()

	plt.ioff()
	plt.show()

	LOG_DEBUG("Control input history: " + str(control_inputs))
	LOG_DEBUG("Forecasts: " + str(planner.solver.get_forecasts()))

	return data, planner.output.realized_trajectory, planner.solver.get_forecasts(), success_flags


def visualize_gaussian_constraints_correct(ax, gaussian_constraints, data, current_state,
										   casadi_solver, disc_id=0, current_k=1, gaussian_ellipses=None, gaussian_annotations=None):
	"""
	Correctly visualize the actual Gaussian constraint boundaries that the solver sees
	"""
	if gaussian_ellipses is None:
		gaussian_ellipses = []

	if gaussian_annotations is None:
		gaussian_annotations = []


	vehicle_pos = np.array([current_state.get("x"), current_state.get("y")])
	vehicle_psi = current_state.get("psi")

	print(f"\n=== CORRECT GAUSSIAN CONSTRAINT VISUALIZATION ===")
	print(f"Vehicle position: ({vehicle_pos[0]:.3f}, {vehicle_pos[1]:.3f})")
	print(f"Vehicle orientation: {vehicle_psi:.3f} rad")

	constraint_count = 0
	violated_constraints = 0
	solver_constraint_values = []

	# Get actual constraint values from solver if available
	if hasattr(casadi_solver, 'solution') and casadi_solver.solution is not None:
		try:
			constraints_expr = gaussian_constraints.get_constraints(
				casadi_solver.dynamics_model,
				casadi_solver.parameter_manager,
				current_k
			)

			# Extract constraint values
			for i, constraint in enumerate(constraints_expr[0]):
				try:
					actual_value = float(casadi_solver.solution.value(constraint))
					solver_constraint_values.append(actual_value)
					print(f"Solver constraint {i}: {actual_value:.6f} {'VIOLATED' if actual_value < 0 else 'OK'}")
				except Exception as e:
					print(f"Could not evaluate constraint {i}: {e}")
		except Exception as e:
			print(f"Could not extract constraint values: {e}")

	# Visualize constraint boundaries
	if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles:
		for obs_idx in range(min(len(data.dynamic_obstacles), gaussian_constraints.max_obstacles)):
			obs = data.dynamic_obstacles[obs_idx]
			obs_pos = np.array([float(obs.position[0]), float(obs.position[1])])

			# Get parameters exactly as the solver sees them
			try:
				# Get parameters from the solver's parameter manager
				param_mgr = casadi_solver.parameter_manager

				obs_x = param_mgr.get_value(f"gaussian_obstacle_{obs_idx}_x_step_{current_k}")
				obs_y = param_mgr.get_value(f"gaussian_obstacle_{obs_idx}_y_step_{current_k}")
				sigma_major = param_mgr.get_value(f"gaussian_obstacle_{obs_idx}_major_step_{current_k}")
				sigma_minor = param_mgr.get_value(f"gaussian_obstacle_{obs_idx}_minor_step_{current_k}")
				risk = param_mgr.get_value(f"gaussian_obstacle_{obs_idx}_risk_step_{current_k}")
				r_obstacle = param_mgr.get_value(f"gaussian_obstacle_{obs_idx}_r_step_{current_k}")
				r_vehicle = param_mgr.get_value("ego_disc_radius")

				combined_radius = r_vehicle + r_obstacle

				print(f"\nObstacle {obs_idx} parameters:")
				print(f"  Position: ({obs_x:.3f}, {obs_y:.3f})")
				print(f"  Sigma: major={sigma_major:.3f}, minor={sigma_minor:.3f}")
				print(f"  Risk: {risk:.3f}, Combined radius: {combined_radius:.3f}")

			except Exception as e:
				print(f"Could not get parameters for obstacle {obs_idx}: {e}")
				# Fallback to default values
				obs_x, obs_y = obs_pos[0], obs_pos[1]
				sigma_major = sigma_minor = 1.0
				risk = 0.05
				r_obstacle = 0.5
				r_vehicle = gaussian_constraints.robot_radius
				combined_radius = r_vehicle + r_obstacle

			# Calculate inverse error function (same as solver)
			x_erfinv = 1.0 - 2.0 * risk
			x_erfinv = max(-0.999, min(0.999, x_erfinv))  # Clamp to valid range

			# Use same approximation as solver
			z = np.sqrt(-2.0 * np.log((1.0 - x_erfinv) / 2.0))

			# Beasley-Springer-Moro coefficients (same as solver)
			c0, c1, c2 = 2.515517, 0.802853, 0.010328
			d1, d2, d3 = 1.432788, 0.189269, 0.001308

			y_erfinv = z - (c0 + c1 * z + c2 * z * z) / (1.0 + d1 * z + d2 * z * z + d3 * z * z * z)
			if x_erfinv < 0:
				y_erfinv = -y_erfinv

			# Newton-Raphson refinement (2 iterations like solver)
			for _ in range(2):
				from scipy.special import erf
				erf_val = erf(y_erfinv)
				error = erf_val - x_erfinv
				derivative = 2.0 / np.sqrt(np.pi) * np.exp(-y_erfinv * y_erfinv)
				if abs(derivative) > 1e-10:
					y_erfinv = y_erfinv - error / derivative

			print(f"  Inverse erf value: {y_erfinv:.6f}")

			# For each disc
			for disc_idx in range(gaussian_constraints.num_discs):
				try:
					disc_offset = param_mgr.get_value(f"ego_disc_{disc_idx}_offset")
				except:
					disc_offset = 0.0  # Fallback

				# Calculate disc position (same as solver)
				rotation_matrix = np.array([[np.cos(vehicle_psi), -np.sin(vehicle_psi)],
											[np.sin(vehicle_psi), np.cos(vehicle_psi)]])
				disc_relative_pos = np.array([disc_offset, 0])
				disc_pos = vehicle_pos + rotation_matrix @ disc_relative_pos

				diff_pos = disc_pos - np.array([obs_x, obs_y])
				diff_norm = np.linalg.norm(diff_pos)

				if diff_norm > 1e-10:  # Avoid division by zero
					a_ij = diff_pos / diff_norm

					# Calculate variance term (same as solver)
					Sigma = np.diag([sigma_major ** 2 + 1e-6, sigma_minor ** 2 + 1e-6])  # Add regularization
					variance_term = a_ij.T @ Sigma @ a_ij
					variance_term = max(variance_term, 1e-12)  # Ensure positive

					# Calculate constraint boundary
					safety_margin = y_erfinv * np.sqrt(variance_term)
					total_safety_distance = combined_radius + safety_margin

					# The actual constraint value (same formula as solver)
					constraint_value = np.dot(a_ij, diff_pos) - combined_radius - safety_margin

					print(f"  Disc {disc_idx}:")
					print(f"    Position: ({disc_pos[0]:.3f}, {disc_pos[1]:.3f})")
					print(f"    Direction vector a_ij: ({a_ij[0]:.3f}, {a_ij[1]:.3f})")
					print(f"    Variance term: {variance_term:.6f}")
					print(f"    Safety margin: {safety_margin:.3f}")
					print(f"    Total safety distance: {total_safety_distance:.3f}")
					print(f"    Constraint value: {constraint_value:.6f}")

					# Visualize the constraint boundary
					# Since this is a halfspace constraint, we'll show it as a line
					# perpendicular to a_ij at the constraint boundary

					# Point on the constraint boundary
					boundary_point = np.array([obs_x, obs_y]) + a_ij * total_safety_distance

					# Create a line perpendicular to a_ij
					perp_vector = np.array([-a_ij[1], a_ij[0]])  # Perpendicular to a_ij
					line_length = 2.0  # Length of the boundary line to draw

					line_start = boundary_point - perp_vector * line_length
					line_end = boundary_point + perp_vector * line_length

					# Draw the constraint boundary line
					color = 'red' if constraint_value < 0 else 'green'
					alpha = 0.8 if constraint_value < 0 else 0.5

					line = ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]],
								   color=color, linewidth=3, alpha=alpha,
								   label=f'Constraint {obs_idx}-{disc_idx}' if obs_idx == 0 and disc_idx == 0 else "")[
						0]
					gaussian_ellipses.append(line)

					# Draw arrow showing constraint direction (feasible region)
					arrow_start = boundary_point
					arrow_end = boundary_point + a_ij * 0.5
					arrow = ax.annotate('', xy=arrow_end, xytext=arrow_start,
										arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=alpha))
					gaussian_ellipses.append(arrow)

					# Also draw a circular approximation for reference
					approx_circle = plt.Circle(np.array([obs_x, obs_y]), total_safety_distance,
											   fill=False, edgecolor='blue', linewidth=1,
											   linestyle=':', alpha=0.3,
											   label=f'Approx {obs_idx}' if disc_idx == 0 else "")
					ax.add_patch(approx_circle)
					gaussian_ellipses.append(approx_circle)

					# Add constraint value annotation
					annotation = ax.annotate(f'{constraint_value:.3f}',
								xy=boundary_point, xytext=(5, 5), textcoords='offset points',
								bbox=dict(boxstyle='round,pad=0.3',
										  facecolor=color, alpha=0.3),
								fontsize=8, ha='left')

					gaussian_annotations.append(annotation)

					constraint_count += 1
					if constraint_value < 0:
						violated_constraints += 1

	# Summary
	constraint_info = f"Constraints: {constraint_count}, Violated: {violated_constraints}"
	if solver_constraint_values:
		solver_violations = sum(1 for val in solver_constraint_values if val < 0)
		constraint_info += f", Solver Violations: {solver_violations}"

	print(f"\nConstraint Summary: {constraint_info}")
	return constraint_info


def visualize_gaussian_constraints_enhanced(ax, gaussian_constraints, data, current_state,
											casadi_solver, disc_id=0, current_k=1, gaussian_ellipses=None, gaussian_annotations=None):
	"""
	Enhanced visualization showing both constraint boundaries and uncertainty ellipses
	"""
	if gaussian_ellipses is None:
		gaussian_ellipses = []
	if gaussian_annotations is None:
		gaussian_annotations = []

	# First, show the correct constraint boundaries
	constraint_info = visualize_gaussian_constraints_correct(
		ax, gaussian_constraints, data, current_state, casadi_solver,
		disc_id, current_k, gaussian_ellipses, gaussian_annotations
	)

	# Add uncertainty ellipses for reference
	if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles:
		for obs_idx, obs in enumerate(data.dynamic_obstacles):
			obs_pos = np.array([float(obs.position[0]), float(obs.position[1])])

			# Get uncertainty parameters
			try:
				param_mgr = casadi_solver.parameter_manager
				sigma_major = param_mgr.get_value(f"gaussian_obstacle_{obs_idx}_major_step_{current_k}")
				sigma_minor = param_mgr.get_value(f"gaussian_obstacle_{obs_idx}_minor_step_{current_k}")
			except:
				sigma_major = sigma_minor = 1.0

			# Show 95% confidence ellipse
			from matplotlib.patches import Ellipse
			confidence_level = 0.95
			chi2_val = -2 * np.log(1 - confidence_level)

			width = 2 * np.sqrt(chi2_val * sigma_major ** 2)
			height = 2 * np.sqrt(chi2_val * sigma_minor ** 2)

			uncertainty_ellipse = Ellipse(obs_pos, width, height,
										  fill=False, edgecolor='orange', linewidth=2,
										  linestyle='--', alpha=0.6,
										  label=f'95% Confidence' if obs_idx == 0 else "")
			ax.add_patch(uncertainty_ellipse)
			gaussian_ellipses.append(uncertainty_ellipse)

	return constraint_info


def test():
	import logging

	# Configuring the logger
	logger = logging.getLogger("root")
	logger.setLevel(logging.DEBUG)

	data, realized_trajectory, trajectory_history, success_flags = run()