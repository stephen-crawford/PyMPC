import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from scipy.interpolate import CubicSpline

from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
from planner_modules.src.constraints.linearized_constraints import LinearizedConstraints
from planner_modules.src.objectives.contouring_objective import ContouringObjective
from planner_modules.src.objectives.goal_objective import GoalObjective
from planning.src.data_prep import define_robot_area
from planning.src.dynamic_models import ContouringSecondOrderUnicycleModel, numeric_rk4
from planning.src.planner import Planner
from planning.src.types import Data, Bound, generate_reference_path, calculate_path_normals, State, \
	generate_dynamic_obstacles
from solver.src.casadi_solver import CasADiSolver
from utils.const import DETERMINISTIC
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

	dynamic_obstacles = generate_dynamic_obstacles(10, DETERMINISTIC, 1)

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
								getattr(obs, "radius", 0.5), color='red', alpha=0.6)
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
			# ✅ Update obstacles if they move (only if they actually have dynamic positions)
			if hasattr(data, "dynamic_obstacles") and data.dynamic_obstacles:
				for idx, obs in enumerate(data.dynamic_obstacles):
					if idx < len(obstacle_patches):
						# Only update if obstacle has a position update method
						if hasattr(obs, 'update_position'):
							obs.update_position(data.planning_start_time)
						obstacle_patches[idx].center = (float(obs.position[0]), float(obs.position[1]))

			# ✅ Force plot update

			for artist in getattr(ax, "_constraint_lines", []):
				try:
					artist.remove()
				except:
					pass
			ax._constraint_lines = []

			# ✅ Plot linearized constraints
			current_k = 0  # Using first prediction step
			disc_id = 0
			LOG_DEBUG(f"Num linear constraints: {len(linear_constraints._a1[disc_id][current_k])}")
			num_constraints = len(linear_constraints._a1[disc_id][current_k])

			for obs_id in range(num_constraints):
				a1 = linear_constraints._a1[disc_id][current_k][obs_id]
				a2 = linear_constraints._a2[disc_id][current_k][obs_id]
				b = linear_constraints._b[disc_id][current_k][obs_id]

				if abs(a1) < 1e-6 and abs(a2) < 1e-6:
					continue

				# ✅ Constraint line
				xx = np.linspace(x_min, x_max, 200)
				if abs(a2) > 1e-6:
					yy = (b - a1 * xx) / a2
				else:
					xx = np.full(200, b / a1)
					yy = np.linspace(y_min, y_max, 200)

				line, = ax.plot(xx, yy, 'g--', linewidth=1.5, alpha=0.8, zorder=4)
				ax._constraint_lines.append(line)

				# ✅ Shading for half-space
				X, Y = np.meshgrid(np.linspace(x_min, x_max, 300),
								   np.linspace(y_min, y_max, 300))
				infeasible = (a1 * X + a2 * Y - b) > 0
				shade = ax.contourf(X, Y, infeasible, levels=[0.5, 1], colors=['#FFCCCC'], alpha=0.25, zorder=1)
				ax._constraint_lines.extend(shade.collections)

				# ✅ Draw normal vector
				if obs_id < len(data.dynamic_obstacles):
					obs = data.dynamic_obstacles[obs_id]
					obs_center = (float(obs.position[0]), float(obs.position[1]))
					normal_vec = np.array([a1, a2])
					normal_vec = normal_vec / np.linalg.norm(normal_vec) * 1.0
					arrow = ax.arrow(obs_center[0], obs_center[1], normal_vec[0], normal_vec[1],
									 head_width=0.2, color='orange', alpha=0.8, zorder=5)
					ax._constraint_lines.append(arrow)

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