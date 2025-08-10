import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.patches import Ellipse
from scipy.interpolate import CubicSpline

from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
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
	linear_constraints = LinearizedConstraints(casadi_solver)
	casadi_solver.module_manager.add_module(linear_constraints)


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
	ax.set_title("MPC with Contouring Objective & Obstacles")
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

			# ✅ ENHANCED CONSTRAINT VISUALIZATION
			current_k = 1  # Use k=1 instead of k=0 for more interesting constraints
			disc_id = 0

			print(f"\n{'=' * 50}")
			print(f"ITERATION {i}: Enhanced Constraint Visualization")
			print(f"{'=' * 50}")

			# Add debugging info
			add_constraint_debugging_info(ax, linear_constraints, data, new_state, disc_id, current_k)

			# Enhanced visualization with feasible region highlighting
			active_constraints, is_feasible = visualize_halfspaces_enhanced(
				ax, linear_constraints, data, new_state, disc_id, current_k)

			# Add title with constraint status
			constraint_status = f"Feasible: {is_feasible}, Active: {len(active_constraints)}"
			ax.set_title(f"MPC with Contouring Objective & Obstacles\nIteration {i}: {constraint_status}")

			# ✅ Force refresh
			fig.canvas.draw()
			fig.canvas.flush_events()
			plt.pause(0.1)  # Slightly longer pause to see the visualization better

			if planner.is_objective_reached(data):
				print(f"Objective reached at iteration {i}!")
				break
		else:
			print(f"Iteration {i}: MPC failed!")

		LOG_DEBUG("Active constraints BEFORE reset, a1:" + str(linear_constraints._a1) + " and a2:" + str(
			linear_constraints._a2) + " and b: " + str(linear_constraints._b))
		casadi_solver.reset()
		LOG_DEBUG("Active constraints after reset, a1:" + str(linear_constraints._a1) + " and a2:" + str(
			linear_constraints._a2))

	# ✅ Keep plot open at the end
	plt.ioff()
	plt.show()

	LOG_DEBUG("Control input history: " + str(control_inputs))
	LOG_DEBUG("Forecasts: " + str(planner.solver.get_forecasts()))

	return data, planner.output.realized_trajectory, planner.solver.get_forecasts(), success_flags


def visualize_halfspaces_enhanced(ax, linear_constraints, data, current_state, disc_id=0, current_k=1):
	"""
	Enhanced visualization of linearized constraints and feasible regions

	Args:
		ax: matplotlib axis
		linear_constraints: LinearizedConstraints object
		data: Data object with obstacles
		current_state: current vehicle state
		disc_id: disc index to visualize
		current_k: time step to visualize
	"""

	# Clear previous constraint visualizations
	for artist in getattr(ax, "_constraint_lines", []):
		try:
			artist.remove()
		except:
			pass
	ax._constraint_lines = []

	# Get axis limits for plotting
	x_min, x_max = ax.get_xlim()
	y_min, y_max = ax.get_ylim()

	# Vehicle position
	vehicle_pos = np.array([current_state.get("x"), current_state.get("y")])

	print(f"=== Visualizing Halfspaces at k={current_k}, disc={disc_id} ===")
	print(f"Vehicle position: {vehicle_pos}")

	# Get number of constraints
	num_constraints = len(linear_constraints._a1[disc_id][current_k])
	print(f"Number of constraints: {num_constraints}")

	# Collect all active constraints
	active_constraints = []
	constraint_lines = []

	for obs_id in range(num_constraints):
		a1 = linear_constraints._a1[disc_id][current_k][obs_id]
		a2 = linear_constraints._a2[disc_id][current_k][obs_id]
		b = linear_constraints._b[disc_id][current_k][obs_id]

		print(f"Constraint {obs_id}: a1={a1:.3f}, a2={a2:.3f}, b={b:.3f}")

		# Skip inactive constraints (dummy values)
		if abs(a1) < 1e-6 and abs(a2) < 1e-6:
			print(f"  -> Skipping inactive constraint {obs_id}")
			continue

		# Skip constraints with dummy b values
		if abs(b - (-1000.0 + vehicle_pos[0])) < 1e-3:
			print(f"  -> Skipping dummy constraint {obs_id}")
			continue

		active_constraints.append((a1, a2, b, obs_id))

		# Plot constraint line: a1*x + a2*y = b
		if abs(a2) > 1e-6:
			# Solve for y: y = (b - a1*x) / a2
			xx = np.linspace(x_min, x_max, 200)
			yy = (b - a1 * xx) / a2
			# Filter to stay within y bounds
			valid_mask = (yy >= y_min) & (yy <= y_max)
			xx_valid = xx[valid_mask]
			yy_valid = yy[valid_mask]
		elif abs(a1) > 1e-6:
			# Vertical line: x = b / a1
			x_line = b / a1
			if x_min <= x_line <= x_max:
				xx_valid = np.full(200, x_line)
				yy_valid = np.linspace(y_min, y_max, 200)
			else:
				xx_valid, yy_valid = [], []
		else:
			xx_valid, yy_valid = [], []

		if len(xx_valid) > 0:
			# Color constraints differently for obstacles vs boundaries
			if obs_id < len(data.dynamic_obstacles):
				color = 'red'
				alpha = 0.8
				label = f'Obstacle {obs_id}'
			else:
				color = 'blue'
				alpha = 0.6
				label = f'Boundary {obs_id}'

			line, = ax.plot(xx_valid, yy_valid, '--', color=color, linewidth=2,
							alpha=alpha, label=label, zorder=4)
			ax._constraint_lines.append(line)
			constraint_lines.append((xx_valid, yy_valid, color))

	print(f"Active constraints: {len(active_constraints)}")

	# Visualize infeasible regions (one constraint at a time)
	X, Y = np.meshgrid(np.linspace(x_min, x_max, 100),
					   np.linspace(y_min, y_max, 100))

	for i, (a1, a2, b, obs_id) in enumerate(active_constraints):
		# Infeasible region: a1*x + a2*y > b (constraint violation)
		infeasible = (a1 * X + a2 * Y - b) > 0

		# Color based on constraint type
		if obs_id < len(data.dynamic_obstacles):
			color = '#FFCCCC'
		else:
			color = '#CCCCFF'  # Light blue for boundaries

		contour = ax.contourf(X, Y, infeasible, levels=[0.5, 1],
							  colors=[color], alpha=0.3, zorder=1)
		ax._constraint_lines.extend(contour.collections)

	# Visualize FEASIBLE region (intersection of all halfspaces)
	if len(active_constraints) > 0:
		feasible_region = np.ones_like(X, dtype=bool)

		for a1, a2, b, obs_id in active_constraints:
			# Feasible region: a1*x + a2*y <= b
			constraint_satisfied = (a1 * X + a2 * Y - b) <= 0
			feasible_region = feasible_region & constraint_satisfied

		# Highlight feasible region
		feasible_contour = ax.contourf(X, Y, feasible_region.astype(int),
									   levels=[0.5, 1], colors=['#CCFFCC'],
									   alpha=0.4, zorder=2)
		ax._constraint_lines.extend(feasible_contour.collections)

		# Add feasible region to legend
		feasible_patch = patches.Patch(color='#CCFFCC', alpha=0.4, label='Feasible Region')
		ax.legend(handles=ax.get_legend_handles_labels()[0] + [feasible_patch],
				  loc='upper right')

	# Highlight vehicle position and check if it's feasible
	vehicle_circle = plt.Circle(vehicle_pos, 0.3, color='blue', alpha=0.8, zorder=7)
	ax.add_patch(vehicle_circle)
	ax._constraint_lines.append(vehicle_circle)

	# Check feasibility at vehicle position
	is_feasible = True
	violated_constraints = []

	for a1, a2, b, obs_id in active_constraints:
		constraint_value = a1 * vehicle_pos[0] + a2 * vehicle_pos[1] - b
		if constraint_value > 1e-6:  # Constraint violated
			is_feasible = False
			violated_constraints.append(obs_id)

	# Add feasibility text
	feasibility_text = "FEASIBLE" if is_feasible else f"INFEASIBLE (violates: {violated_constraints})"
	feasibility_color = 'green' if is_feasible else 'red'

	ax.text(0.02, 0.98, f"Vehicle: {feasibility_text}",
			transform=ax.transAxes, fontsize=12, weight='bold',
			color=feasibility_color, va='top',
			bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

	# Add constraint summary
	summary_text = f"Active Constraints: {len(active_constraints)}\n"
	summary_text += f"Obstacles: {sum(1 for _, _, _, obs_id in active_constraints if obs_id < len(data.dynamic_obstacles))}\n"
	summary_text += f"Boundaries: {sum(1 for _, _, _, obs_id in active_constraints if obs_id >= len(data.dynamic_obstacles))}"

	ax.text(0.02, 0.02, summary_text, transform=ax.transAxes, fontsize=10,
			va='bottom', ha='left',
			bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

	print(f"Vehicle feasibility: {feasibility_text}")
	return active_constraints, is_feasible


def add_constraint_debugging_info(ax, linear_constraints, data, current_state, disc_id=0, current_k=1):
	"""
	Add detailed debugging information about constraints
	"""
	print(f"\n=== CONSTRAINT DEBUGGING INFO ===")
	print(f"Time step k={current_k}, Disc={disc_id}")

	vehicle_pos = np.array([current_state.get("x"), current_state.get("y")])
	print(f"Vehicle position: ({vehicle_pos[0]:.3f}, {vehicle_pos[1]:.3f})")

	if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles:
		print(f"Number of dynamic obstacles: {len(data.dynamic_obstacles)}")

		for i, obs in enumerate(data.dynamic_obstacles):
			obs_pos = np.array([float(obs.position[0]), float(obs.position[1])])
			distance = np.linalg.norm(obs_pos - vehicle_pos)
			print(f"  Obstacle {i}: pos=({obs_pos[0]:.3f}, {obs_pos[1]:.3f}), "
				  f"radius={getattr(obs, 'radius', 'N/A')}, distance={distance:.3f}")

	# Check constraint arrays
	if (disc_id < len(linear_constraints._a1) and
			current_k < len(linear_constraints._a1[disc_id])):

		constraint_array = linear_constraints._a1[disc_id][current_k]
		print(f"Constraint array length: {len(constraint_array)}")

		for i in range(len(constraint_array)):
			a1 = linear_constraints._a1[disc_id][current_k][i]
			a2 = linear_constraints._a2[disc_id][current_k][i]
			b = linear_constraints._b[disc_id][current_k][i]

			print(f"  Constraint {i}: a1={a1:.6f}, a2={a2:.6f}, b={b:.6f}")

			# Check if it's a dummy constraint
			is_dummy = (abs(a1) < 1e-6 and abs(a2) < 1e-6) or abs(b - (-1000.0 + vehicle_pos[0])) < 1e-3

			if not is_dummy:
				# Evaluate constraint at vehicle position
				constraint_value = a1 * vehicle_pos[0] + a2 * vehicle_pos[1] - b
				status = "VIOLATED" if constraint_value > 1e-6 else "SATISFIED"
				print(f"    -> ACTIVE: value at vehicle = {constraint_value:.6f} ({status})")
			else:
				print(f"    -> DUMMY/INACTIVE")
	else:
		print(f"ERROR: Invalid indices disc_id={disc_id}, k={current_k}")


# Integrate this into your main loop by replacing the constraint visualization section:
"""
# Replace this section in your main loop:
# ✅ Plot linearized constraints
current_k = 0  # Using first prediction step
disc_id = 0
# ... existing constraint plotting code ...

# With this enhanced version:
current_k = 1  # Use k=1 instead of k=0 for more interesting constraints
disc_id = 0

# Add debugging info
add_constraint_debugging_info(ax, linear_constraints, data, new_state, disc_id, current_k)

# Enhanced visualization
active_constraints, is_feasible = visualize_halfspaces_enhanced(
    ax, linear_constraints, data, new_state, disc_id, current_k)
"""

def test():
	import logging

	# Configuring the logger
	logger = logging.getLogger("root")
	logger.setLevel(logging.DEBUG)

	data, realized_trajectory, trajectory_history, success_flags = run()