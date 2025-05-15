import matplotlib.pyplot as plt
import numpy as np

from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
from planner_modules.src.objectives.contouring_objective import ContouringObjective
from planner_modules.src.objectives.goal_objective import GoalObjective
from planning.src.dynamic_models import SecondOrderUnicycleModel, ContouringSecondOrderUnicycleModel
from planning.src.planner import Planner
from planning.src.types import State, Data
from solver.src.casadi_solver import CasADiSolver
from utils.math_utils import Spline
from utils.utils import CONFIG


def test_objective():
	# Set up basic CONFIG
	CONFIG["control_frequency"] = 10
	CONFIG["shift_previous_solution_forward"] = True
	CONFIG["enable_output"] = True
	CONFIG["debug_limits"] = True
	CONFIG["debug_visuals"] = True

	# Default weights for the goal objective - INCREASE WEIGHT SUBSTANTIALLY
	if "weights" not in CONFIG:
		CONFIG["weights"] = {}
	CONFIG["weights"]["goal_weight"] = 50.0  # Increased from 0.5 to make goal more important

	# Create simulation objects
	dt = 0.1
	horizon = 10  # Increased horizon for better planning

	# Print the full CONFIG for debugging
	print("CONFIG:", CONFIG)

	# Create the solver with appropriate dimensions
	casadi_solver = CasADiSolver(dt, horizon)

	# Set up the dynamics model first
	bike = ContouringSecondOrderUnicycleModel()
	casadi_solver.set_dynamics_model(bike)

	# Print the state and control variables to ensure we're using the right names
	print("MODEL STATE VARS:", bike.get_vars())
	print("MODEL CONTROL VARS:", bike.inputs)

	# Create the planner
	planner = Planner(casadi_solver, bike)

	# Add goal objective module
	goal_objective = GoalObjective(casadi_solver)
	contouring_constraint = ContouringConstraints(casadi_solver)
	contouring_objective = ContouringObjective(casadi_solver)
	casadi_solver.module_manager.add_module(contouring_objective)
	casadi_solver.module_manager.add_module(goal_objective)
	casadi_solver.module_manager.add_module(contouring_constraint)

	# Create data object with goal - use a much closer goal for testing
	data = Data()
	data.goal = np.array([5.0, 5.0])  # Set a much closer goal
	data.goal_received = True
	data.planning_start_time = 0.0

	# Add solver timeout parameter
	casadi_solver.parameter_manager.add("solver_timeout", 10.0)  # Increased timeout

	# Define parameters for the goal objective
	planner.initialize()

	# Create reference path for visualization

	path = Spline()
	x_vals = np.linspace(0, 5, 50)
	y_vals = np.linspace(0, 5, 50)
	s_vals = np.linspace(0, np.sqrt(50), 50)

	path.x = x_vals.tolist()
	path.y = y_vals.tolist()
	path.s = s_vals.tolist()

	data.reference_path = path

	# Create left and right boundaries
	# Left boundary (wider than the path)
	left_bound = Spline()
	left_bound.x = (x_vals + 1.0).tolist()  # 1m to the left (offset perpendicular to path)
	left_bound.y = (y_vals + 1.0).tolist()
	left_bound.s = s_vals.tolist()

	# Right boundary (wider than the path)
	right_bound = Spline()
	right_bound.x = (x_vals - 1.0).tolist()  # 1m to the right
	right_bound.y = (y_vals - 1.0).tolist()
	right_bound.s = s_vals.tolist()

	# Add boundaries to data
	data.left_bound = left_bound
	data.right_bound = right_bound

	# Process the reference path to calculate width parameters
	# This is needed to initialize the contouring constraints
	contouring_constraint.process_reference_path(data)

	# Create initial state - make sure to match the model's state variables
	state = planner.get_state()
	state.set("x", 0.0)
	state.set("y", 0.0)
	state.set("psi", 0.1)
	state.set("v", 0.5)

	# Check if model uses vx/vy or not
	if "vx" in bike.get_vars():
		state.set("vx", 0.5 * np.cos(0.1))
		state.set("vy", 0.5 * np.sin(0.1))

	# Add acceleration and steering if needed for the model
	if "a" in bike.inputs:
		state.set("a", 0.0)
	if "w" in bike.inputs:
		state.set("w", 0.0)

	# Print the state to verify values
	print("INITIAL STATE:", {var: state.get(var) for var in bike.get_vars()})


	# Set initial state constraint
	casadi_solver.set_initial_state(state)

	# Initialize with a simple trajectory
	casadi_solver.initialize_base_rollout(state)

	# Run MPC loop
	states_x = [state.get("x")]
	states_y = [state.get("y")]
	success_flags = []

	print("Starting MPC simulation loop...")

	max_iterations = 100  # Increased max iterations

	# Add arrays to store trajectories for visualization
	all_trajectories_x = []
	all_trajectories_y = []

	for i in range(max_iterations):
		data.planning_start_time = i * dt

		# Initialize warmstart
		if i > 0:
			casadi_solver.initialize_warmstart(state, shift_forward=True)

		# Set initial state constraint for this iteration
		casadi_solver.set_initial_state(state)

		# Load warmstart values
		casadi_solver.load_warmstart()

		# Solve MPC
		output = planner.solve_mpc(state, data)
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
			next_x = casadi_solver.get_output(1, "x")
			next_y = casadi_solver.get_output(1, "y")
			next_v = casadi_solver.get_output(1, "v")
			next_psi = casadi_solver.get_output(1, "psi")

			# Print debug info
			print(f"Raw next outputs: x={next_x}, y={next_y}, v={next_v}, psi={next_psi}")

			# Check for unrealistic jumps in position
			curr_x = state.get("x")
			curr_y = state.get("y")
			curr_v = state.get("v")

			# Calculate maximum possible distance traveled based on velocity
			max_dist = curr_v * dt * 1.5  # Allow for some acceleration
			actual_dist = np.sqrt((next_x - curr_x) ** 2 + (next_y - curr_y) ** 2)

			if actual_dist > max_dist:
				print(
					f"WARNING: Unrealistic jump detected! Distance: {actual_dist:.2f}m, Max possible: {max_dist:.2f}m")
				print("Limiting movement to physically possible distance")
				# Limit the movement to a physically possible distance
				direction = np.array([next_x - curr_x, next_y - curr_y])
				direction = direction / np.linalg.norm(direction)
				next_x = curr_x + direction[0] * max_dist
				next_y = curr_y + direction[1] * max_dist

			# Update state with solution values
			state.set("x", next_x)
			state.set("y", next_y)
			state.set("v", next_v)
			state.set("psi", next_psi)

			# Update vx/vy if they're part of the state
			if "vx" in bike.get_vars():
				next_vx = casadi_solver.get_output(1, "vx")
				next_vy = casadi_solver.get_output(1, "vy")
				if next_vx is not None and next_vy is not None:
					state.set("vx", next_vx)
					state.set("vy", next_vy)
				else:
					# Recompute from v and psi if not directly available
					state.set("vx", next_v * np.cos(next_psi))
					state.set("vy", next_v * np.sin(next_psi))

			# Store for plotting
			states_x.append(state.get("x"))
			states_y.append(state.get("y"))

			iter_x = state.get("x")
			iter_y = state.get("y")
			print(f"Iteration {i}: Position = ({iter_x:.2f}, {iter_y:.2f}), Success = {output.success}")

			# Check if goal reached
			goal_distance = np.sqrt((state.get("x") - data.goal[0]) ** 2 + (state.get("y") - data.goal[1]) ** 2)
			print(f"Distance to goal: {goal_distance:.2f}")

			if goal_distance < 0.5:  # Use a smaller threshold (0.5m) for goal reaching
				print(f"Goal reached at iteration {i}!")
				break
		else:
			print(f"Iteration {i}: MPC failed!")
			if hasattr(casadi_solver, 'info') and 'error' in casadi_solver.info:
				print(f"Error: {casadi_solver.info['error']}")
			# Print more debug info when solver fails
			casadi_solver.print_if_bound_limited()
			print(casadi_solver.explain_exit_flag())

	# Plot both actual path and MPC predictions
	plt.figure(figsize=(12, 8))

	# Plot the goal and start points
	plt.plot(data.goal[0], data.goal[1], 'r*', markersize=12, label='Goal')
	plt.plot(0, 0, 'go', markersize=8, label='Start')

	# Plot the actual vehicle trajectory
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

	# Print statistics
	print(f"Success rate: {sum(success_flags) / len(success_flags) * 100:.1f}%")
	print(f"Final position: ({states_x[-1]:.2f}, {states_y[-1]:.2f})")
	print(f"Distance to goal: {np.sqrt((states_x[-1] - data.goal[0]) ** 2 + (states_y[-1] - data.goal[1]) ** 2):.2f}")


if __name__ == "__main__":
	from utils.utils import LOG_DEBUG

	LOG_DEBUG("This is a test log message")

	test_objective()