import numpy as np
import matplotlib.pyplot as plt

from planner_modules.src.objectives.goal_objective import GoalObjective
from utils.utils import CONFIG
from planning.src.types import Trajectory, State

# Import your existing classes
from planning.src.planner import Planner
from solver.src.casadi_solver import CasADiSolver
from planner_modules.src.constraints.base_constraint import BaseConstraint
from planner_modules.src.objectives.base_objective import BaseObjective
from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
from utils.const import OBJECTIVE
from utils.utils import LOG_DEBUG
from utils.math_utils import distance
from utils.utils import ExperimentManager, Benchmarker
from solver.src.modules_manager import ModuleManager
from solver.src.parameter_manager import ParameterManager

# Test setup and execution
def test_mpc_planner():
	# Set up basic CONFIG
	CONFIG["control_frequency"] = 10
	CONFIG["shift_previous_solution_forward"] = True
	CONFIG["enable_output"] = True
	CONFIG["debug_limits"] = False
	CONFIG["debug_visuals"] = False

	# Create simulation objects
	dt = 0.1
	horizon = 20
	casadi_solver = CasADiSolver(dt, horizon)
	planner = Planner(casadi_solver)

	# Add goal objective module
	goal_objective = GoalObjective(casadi_solver)
	casadi_solver.module_manager.add_module(goal_objective)

	# Define parameters
	goal_objective.define_parameters(casadi_solver.parameter_manager)

	# Initialize the solver problem
	casadi_solver.finalize_problem()

	# Create initial state and data
	state = State(0.0, 0.0, 1.0, 0.1)  # Start at origin with small velocity
	data = Data()
	data.goal = np.array([5.0, 5.0])  # Goal at (5,5)
	data.planning_start_time = 0.0
	module_data = ModuleData()

	# Create reference path for visualization
	path = Path()
	x_vals = np.linspace(0, 5, 50)
	y_vals = np.linspace(0, 5, 50)
	s_vals = np.linspace(0, np.sqrt(50), 50)

	path.x = x_vals.tolist()
	path.y = y_vals.tolist()
	path.s = s_vals.tolist()

	data.reference_path = path
	module_data.path = path

	# Create simple boundaries (straight corridor)
	left_path = Path()
	left_path.x = (x_vals + 1).tolist()
	left_path.y = (y_vals + 1).tolist()
	left_path.s = s_vals.tolist()

	right_path = Path()
	right_path.x = (x_vals - 1).tolist()
	right_path.y = (y_vals - 1).tolist()
	right_path.s = s_vals.tolist()

	data.left_bound = left_path
	data.right_bound = right_path

	# Run MPC loop
	states_x = [state.x]
	states_y = [state.y]
	success_flags = []

	print("Starting MPC simulation loop...")

	num_iterations = 50
	for i in range(num_iterations):
		data.planning_start_time = i * dt

		# Solve MPC
		output = planner.solve_mpc(state, data)
		success_flags.append(output.success)

		if output.success:
			# Extract first control action
			next_x = test_solver.get_output(1, "x")
			next_y = test_solver.get_output(1, "y")
			next_v = test_solver.get_output(1, "v")
			next_heading = test_solver.get_output(1, "heading")

			# Update state (simple integrator model)
			state.x = next_x
			state.y = next_y
			state.v = next_v
			state.heading = next_heading

			states_x.append(state.x)
			states_y.append(state.y)

			print(f"Iteration {i}: Position = ({state.x:.2f}, {state.y:.2f}), Success = {output.success}")

			# Check if goal reached
			if goal_objective.is_objective_reached(state, data):
				print(f"Goal reached at iteration {i}!")
				break
		else:
			print(f"Iteration {i}: MPC failed!")

	# Plot results
	plt.figure(figsize=(10, 6))
	plt.plot(states_x, states_y, 'b-', label='Vehicle trajectory')
	plt.plot(data.goal[0], data.goal[1], 'r*', markersize=10, label='Goal')
	plt.plot(0, 0, 'go', markersize=8, label='Start')
	plt.xlabel('X position [m]')
	plt.ylabel('Y position [m]')
	plt.title('MPC Vehicle Trajectory')
	plt.legend()
	plt.grid(True)
	plt.axis('equal')
	plt.show()

	# Print statistics
	print(f"Success rate: {sum(success_flags) / len(success_flags) * 100:.1f}%")
	print(f"Final position: ({states_x[-1]:.2f}, {states_y[-1]:.2f})")
	print(f"Distance to goal: {np.sqrt((states_x[-1] - data.goal[0]) ** 2 + (states_y[-1] - data.goal[1]) ** 2):.2f}")


if __name__ == "__main__":
	test_mpc_planner()