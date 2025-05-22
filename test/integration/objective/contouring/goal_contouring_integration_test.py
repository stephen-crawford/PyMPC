from venv import logger

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from scipy.interpolate import CubicSpline
import casadi as ca

from planner_modules.src.objectives.contouring_objective import ContouringObjective
from planner_modules.src.objectives.goal_objective import GoalObjective
from planning.src.data_prep import define_robot_area
from planning.src.dynamic_models import ContouringSecondOrderUnicycleModel, numeric_rk4
from planning.src.planner import Planner
from planning.src.types import Data, Bound, ReferencePath, generate_reference_path, calculate_path_normals
from solver.src.casadi_solver import CasADiSolver
from utils.utils import CONFIG, LOG_DEBUG, LOG_INFO, LOG_WARN


def run(dt=0.5, horizon=10, model=ContouringSecondOrderUnicycleModel, start=(0.0, 0.0), goal=(5.0, 5.0), max_iterations=100):

	dt = dt
	horizon = horizon

	casadi_solver = CasADiSolver(dt, horizon)

	vehicle = model()

	casadi_solver.set_dynamics_model(vehicle)

	# Create the planner
	planner = Planner(casadi_solver, vehicle)

	goal_objective = GoalObjective(casadi_solver)
	casadi_solver.module_manager.add_module(goal_objective)
	contouring_objective = ContouringObjective(casadi_solver)
	casadi_solver.module_manager.add_module(contouring_objective)

	data = Data()
	data.start = np.array(start)
	data.goal = np.array(goal)
	data.goal_received = True
	data.planning_start_time = 0.0

	# Add solver timeout parameter
	casadi_solver.parameter_manager.add("solver_timeout", 10.0)

	planner.initialize()

	state = planner.get_state()
	state.set("x", data.start[0])
	state.set("y", data.start[1])
	state.set("psi", 0.1)
	state.set("v", 0.5)
	state.set("spline", 0)

	# Add solver timeout parameter
	casadi_solver.parameter_manager.add("solver_timeout", 10.0)  # Increased timeout

	# Define parameters for the goal objective
	planner.initialize()

	# Store path
	data.reference_path = generate_reference_path(data.start, data.goal, path_type="curved")
	# Calculate normal vectors for left and right boundaries

	normals = calculate_path_normals(data.reference_path)

	# Road width (adjust based on your actual data structure)
	road_width = data.road_width if data.road_width is not None else 4.0
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
	# Create Bound objects from earlier calculated values
	data.left_bound = Bound(left_x, left_y, data.reference_path.s)
	data.right_bound = Bound(right_x, right_y, data.reference_path.s)

	data.robot_area = define_robot_area(vehicle.length, vehicle.width, 1)
	LOG_DEBUG(f"Robot area set to {data.robot_area[0].radius}")

	# Mimics behavior of a robot receiving data
	planner.on_data_received(data, "reference_path")

	# Create initial state - make sure to match the model's state variables
	state = planner.get_state()
	state.set("x", 0.0)
	state.set("y", 0.0)
	state.set("psi", 0.1)
	state.set("v", 0.5)
	state.set("spline", 0.0)
	state.set("a", 0.0)
	state.set("w", 0.0)

	# Print the state to verify values
	print("INITIAL STATE:", {var: state.get(var) for var in vehicle.get_vars()})
	# Run MPC loop
	states_x = [state.get("x")]
	states_y = [state.get("y")]
	success_flags = []

	# Add arrays to store trajectories for visualization
	trajectories_found = []
	all_trajectories_x = []
	all_trajectories_y = []
	states = []


	iter = 0
	for i in range(max_iterations):
		iter += 1
		LOG_WARN(f"Running mpc from state_x {planner.get_state().get('x')}")
		states.append(planner.get_state().copy())
		if i % (max_iterations // 10) == 0:
			LOG_INFO(f"Starting MPC simulation loop for iteration {i} with state {planner.get_state()}")
		data.planning_start_time = i * dt

		# Solve MPC
		output = planner.solve_mpc(data)
		success_flags.append(output.success)


		if output.success:
			# Store the whole predicted trajectory for visualization
			traj_x = []
			traj_y = []
			for k in range(horizon + 1):
				x_k = casadi_solver.get_output(k, "x")
				y_k = casadi_solver.get_output(k, "y")
				if x_k is not None and y_k is not None:
					traj_x.append(float(x_k))  # Convert to float
					traj_y.append(float(y_k))  # Convert to float

			all_trajectories_x.append(traj_x)
			all_trajectories_y.append(traj_y)

			# Extract next state controls and convert to numeric values
			next_a = float(casadi_solver.get_output(0, "a"))
			next_w = float(casadi_solver.get_output(0, "w"))

			# Get current state values as floats
			current_x = float(state.get("x"))
			current_y = float(state.get("y"))
			current_psi = float(state.get("psi"))
			current_v = float(state.get("v"))
			current_spline = float(state.get("spline"))

			# Create the z vector for integration
			z_k = [next_a, next_w, current_x, current_y, current_psi, current_v, current_spline]
			z_k = ca.vertcat(*z_k)
			vehicle.load(z_k)

			# Get the next state through integration
			next_state_symbolic = vehicle.discrete_dynamics(z_k, casadi_solver.parameter_manager,
															casadi_solver.timestep)

			# Convert symbolic result to numeric values
			if isinstance(next_state_symbolic, (ca.MX, ca.SX)):
				next_state_numeric = numeric_rk4(next_state_symbolic, vehicle, casadi_solver.parameter_manager,
												 casadi_solver.timestep)
			else:
				next_state_numeric = next_state_symbolic

			# Extract numeric values from the integrated state
			next_x = float(next_state_numeric[0])
			next_y = float(next_state_numeric[1])
			next_psi = float(next_state_numeric[2])
			next_v = float(next_state_numeric[3])
			next_spline = float(next_state_numeric[4])
			# Update the planner state with numeric values
			planner.get_state().set("a", next_a)
			planner.get_state().set("w", next_w)
			planner.get_state().set("x", next_x)
			planner.get_state().set("y", next_y)
			planner.get_state().set("psi", next_psi)
			planner.get_state().set("v", next_v)
			planner.get_state().set("spline", next_spline)

			casadi_solver.reset()

			# Check if goal reached
			if planner.is_objective_reached(data):
				LOG_DEBUG("Objective reached so ending.")
				success_flags.append(output.success)
				break

		else:
			LOG_DEBUG(f"Iteration {i}: MPC failed!")
			if hasattr(casadi_solver, 'info') and 'error' in casadi_solver.info:
				LOG_DEBUG(f"Error: {casadi_solver.info['error']}")
			# Print more debug info when solver fails
			casadi_solver.print_if_bound_limited()
			LOG_DEBUG(casadi_solver.explain_exit_flag())

	# Print statistics

	LOG_DEBUG(f"After running {iter} all trajectories_x are: {all_trajectories_x}")
	xs = []
	for state in states:
		xs.append(state.get("x"))
	LOG_DEBUG(f"After running all states_x are: {xs}")
	return data, states, all_trajectories_x, all_trajectories_y, success_flags


def plot_vehicle(x, y, psi, length, width, ax, color='blue', alpha=0.7):
	# Center vehicle on (x, y)
	rear_axle_to_center = length / 2
	vehicle_patch = Rectangle(
		(-rear_axle_to_center, -width / 2),  # Centered at origin, will transform
		length,
		width,
		color=color,
		alpha=alpha,
		zorder=3
	)
	# Apply transformation: rotate around origin, then translate to (x, y)
	transform = Affine2D().rotate(psi).translate(x, y) + ax.transData
	vehicle_patch.set_transform(transform)
	ax.add_patch(vehicle_patch)


def plot_trajectory(data, states, states_x, states_y, all_trajectories_x, all_trajectories_y, success_flags):
	plt.figure(figsize=(12, 8))

	# Plot the goal and start points
	plt.plot(data.goal[0], data.goal[1], 'r*', markersize=12, label='Goal')
	plt.plot(0, 0, 'go', markersize=8, label='Start')
	goal_circle = plt.Circle(data.goal, 1, color='r', fill=False, linestyle='--', label='Goal region')
	plt.gca().add_patch(goal_circle)

	for i in range(1, len(states_x), 5):
		dx = states_x[i] - states_x[i - 1]
		dy = states_y[i] - states_y[i - 1]
		plt.arrow(states_x[i - 1], states_y[i - 1], dx, dy, head_width=0.1, color='blue', alpha=0.5)
	LOG_DEBUG(f"States x {states_x}")
	# Plot the actual vehicle
	plt.plot(states_x, states_y, 'b-', linewidth=2, markersize=data.robot_area[0].radius,label ='Vehicle trajectory')

	# Plot all predicted trajectories (MPC horizon predictions at each step)
	for i, (traj_x, traj_y) in enumerate(zip(all_trajectories_x, all_trajectories_y)):
		if i % 5 == 0:  # Plot every 5th trajectory to avoid cluttering
			# Use a very faint line for earlier trajectories
			alpha = 0.3
			plt.plot(traj_x, traj_y, 'g--', alpha=alpha, linewidth=1)

	# Show vehicle shapes at intervals
	vehicle_length = data.robot_area[0].radius * 2  # Substitute with vehicle.length if available
	vehicle_width = data.robot_area[0].radius * 2

	for i in range(0, len(states_x), 10):
		x = states_x[i]
		y = states_y[i]
		psi = states[i].get("psi")
		plot_vehicle(x, y, psi, vehicle_length, vehicle_width, plt.gca(), color='cyan', alpha=0.6)

	# Plot road constraints
	plt.plot(data.left_boundary_x, data.left_boundary_y, 'r--', label='Left boundary')
	plt.plot(data.right_boundary_x, data.right_boundary_y, 'm--', label='Right boundary')

	plt.xlabel('X position [m]')
	plt.ylabel('Y position [m]')
	plt.title('MPC Goal Objective Test with Trajectory Predictions')
	plt.legend()
	plt.grid(True)
	plt.axis('equal')

	# Add info text
	iterations = len(states_x) - 1
	success_rate = sum(success_flags) / len(success_flags) * 100
	info_text = f"Iterations: {iterations}\nSuccess rate: {success_rate:.1f}%\nFinal pos: ({states_x[-1]:.2f}, {states_y[-1]:.2f})"
	plt.figtext(0.02, 0.02, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

	plt.tight_layout()
	plt.show()

def test():
	import logging

	# Configuring the logger
	logger = logging.getLogger("root")
	logger.setLevel(logging.DEBUG)


	data, states, all_trajectories_x, all_trajectories_y, success_flags = run()
	LOG_DEBUG("Returned states: {}".format(states))
	states_x = []
	states_y = []

	for state in states:
		states_x.append(state.get("x"))
		states_y.append(state.get("y"))

	plot_trajectory(data, states, states_x, states_y, all_trajectories_x, all_trajectories_y, success_flags)
