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


def run(dt=0.5, horizon=5, model=ContouringSecondOrderUnicycleModel, start=(0.0, 0.0), goal=(5.0, 5.0), max_iterations=100):

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
	print("INITIAL STATE:", {var: state.get(var) for var in vehicle.get_dependent_vars()})
	# Run MPC loop
	states_x = [state.get("x")]
	states_y = [state.get("y")]
	success_flags = []

	# Add arrays to store trajectories for visualization

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
			next_a = output.trajectory_history[-1].get_states()[1].get("a")
			next_w = output.trajectory_history[-1].get_states()[1].get("w")
			LOG_DEBUG("OUTPUT TRAJ HISTORY: " + str(output.trajectory_history[-1].get_states()[1]))
			z_k = [next_a, next_w, state.get("x"), state.get("y"), state.get("psi"), state.get("v"), state.get("spline")]
			LOG_DEBUG("Vec for calc next state through prop: " + str(z_k))
			# Convert to CasADi vector
			z_k = ca.vertcat(*z_k)
			vehicle.load(z_k)

			next_state_symbolic = vehicle.discrete_dynamics(z_k, casadi_solver.parameter_manager,
															casadi_solver.timestep)
			next_state = numeric_rk4(next_state_symbolic, vehicle, casadi_solver.parameter_manager,
									 casadi_solver.timestep)

			next_x = next_state[0]
			next_y = next_state[1]
			next_psi = next_state[2]
			next_v = next_state[3]

			LOG_DEBUG(f"Next state: {next_state}")
			states_x.append(float(next_x))
			states_y.append(float(next_y))
			LOG_DEBUG("Going to set the next state based on integrated dynamics. x: " + str(next_x) + " y:" + str(
				next_y) + " psi: " + str(next_psi) + " v: " + str(next_v))
			new_state = planner.get_state().copy()
			new_state.set("x", next_x)
			new_state.set("y", next_y)
			new_state.set("psi", next_psi)
			new_state.set("v", next_v)
			new_state.set("w", next_w)
			new_state.set("a", next_a)
			LOG_DEBUG("Next state is: " + str(new_state))
			output.control_history.append((next_a, next_w))
			output.realized_trajectory.add_state(new_state)
			planner.set_state(new_state)
			state = planner.get_state()

			LOG_DEBUG(f"Next state: {planner.get_state()}")
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
			casadi_solver.reset()

	# Print statistics
	return data, planner.output.realized_trajectory, planner.output.trajectory_history, success_flags


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


def plot_trajectory(data, realized_trajectory, trajectory_history,  success_flags):
	plt.figure(figsize=(12, 8))

	states_x = [float(s.get("x")) for s in realized_trajectory.get_states()]
	states_y = [float(s.get("y")) for s in realized_trajectory.get_states()]
	states_psi = [float(s.get("psi")) for s in realized_trajectory.get_states()]
	LOG_DEBUG("States x: " + str(states_x))
	all_trajectories_x = []
	all_trajectories_y = []

	for traj in trajectory_history:
		traj_x = [float(s.get("x")) for s in traj.get_states()]
		traj_y = [float(s.get("y")) for s in traj.get_states()]
		all_trajectories_x.append(traj_x)
		all_trajectories_y.append(traj_y)

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
		psi = states_psi[i]
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


	data, realized_trajectory, trajectory_history, success_flags = run()


	plot_trajectory(data, realized_trajectory, trajectory_history, success_flags)
