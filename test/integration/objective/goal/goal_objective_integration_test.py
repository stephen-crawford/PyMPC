import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

from planner_modules.src.objectives.goal_objective import GoalObjective
from planning.src.dynamic_models import SecondOrderUnicycleModel, numeric_rk4
from planning.src.planner import Planner
from planning.src.types import Data, generate_reference_path
from solver.src.casadi_solver import CasADiSolver
from utils.utils import CONFIG, LOG_DEBUG, LOG_INFO


def run(dt=0.1, horizon=10, model=SecondOrderUnicycleModel, start=(0.0, 0.0), goal=(5.0, 5.0), max_iterations=100):

	dt = dt
	horizon = horizon

	casadi_solver = CasADiSolver(dt, horizon)

	vehicle = model()

	casadi_solver.set_dynamics_model(vehicle)

	# Create the planner
	planner = Planner(casadi_solver, vehicle)

	goal_objective = GoalObjective(casadi_solver)
	casadi_solver.module_manager.add_module(goal_objective)

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

	if "a" in vehicle.inputs:
		state.set("a", 0.0)
	if "w" in vehicle.inputs:
		state.set("w", 0.0)


	states_x = [state.get("x")]
	states_y = [state.get("y")]
	success_flags = []

	LOG_INFO("Starting MPC simulation loop...")

	# Add arrays to store trajectories for visualization
	all_trajectories_x = []
	all_trajectories_y = []

	for i in range(max_iterations):
		if i % (max_iterations//10) == 0:
			LOG_INFO(f"Starting MPC simulation loop for iteration {i} with state {planner.state}")
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
					traj_x.append(x_k)
					traj_y.append(y_k)

			all_trajectories_x.append(traj_x)
			all_trajectories_y.append(traj_y)

			# Extract next state - use the correct state variable names
			next_a = casadi_solver.get_output(0, "a")
			next_w = casadi_solver.get_output(0, "w")


			z_k = [next_a, next_w, state.get("x"), state.get("y"), state.get("psi"), state.get("v")]
			# Convert to CasADi vector
			z_k = ca.vertcat(*z_k)
			vehicle.load(z_k)

			next_state_symbolic = vehicle.discrete_dynamics(z_k, casadi_solver.parameter_manager, casadi_solver.timestep)
			next_state = numeric_rk4(next_state_symbolic, vehicle, casadi_solver.parameter_manager, casadi_solver.timestep)

			next_x = next_state[0]
			next_y = next_state[1]
			next_psi = next_state[2]
			next_v = next_state[3]

			LOG_DEBUG(f"Next state: {next_state}")
			states_x.append(float(next_x))
			states_y.append(float(next_y))

			planner.get_state().set("a", next_a)
			planner.get_state().set("w", next_w)
			planner.get_state().set("x", next_x)
			planner.get_state().set("y", next_y)
			planner.get_state().set("psi", next_psi)
			planner.get_state().set("v", next_v)

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

	# Print statistics
	return data, states_x, states_y, all_trajectories_x, all_trajectories_y, success_flags


def plot_trajectory(data, states_x, states_y, all_trajectories_x, all_trajectories_y, success_flags):
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

	# Plot the actual vehicle
	plt.plot(states_x, states_y, 'b-', linewidth=2, label='Vehicle trajectory')

	# Plot all predicted trajectories (MPC horizon predictions at each step)
	for i, (traj_x, traj_y) in enumerate(zip(all_trajectories_x, all_trajectories_y)):
		if i % 5 == 0:  # Plot every 5th trajectory to avoid cluttering
			# Use a very faint line for earlier trajectories
			alpha = 0.3
			plt.plot(traj_x, traj_y, 'g--', alpha=alpha, linewidth=1)

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

	logger = logging.getLogger("root")
	logger.setLevel(logging.DEBUG)

	data, states_x, states_y, all_trajectories_x, all_trajectories_y, success_flags = run()

	plot_trajectory(data, states_x, states_y, all_trajectories_x, all_trajectories_y, success_flags)
