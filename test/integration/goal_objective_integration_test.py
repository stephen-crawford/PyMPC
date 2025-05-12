import numpy as np
import matplotlib.pyplot as plt
from utils.utils import CONFIG
from planning.src.types import Trajectory

# Import your existing classes
from planning.src.planner import Planner
from solver.src.casadi_solver import CasADiSolver
from planner_modules.src.constraints.base_constraint import BaseConstraint
from planner_modules.src.objectives.base_objective import BaseObjective
from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
from utils.const import OBJECTIVE
from utils.utils import LOG_DEBUG, distance
from utils.utils import ExperimentManager, Benchmarker
from solver.src.modules_manager import ModuleManager
from solver.src.parameter_manager import ParameterManager


# Your Goal Objective
class GoalObjective(BaseObjective):
	def __init__(self, solver):
		super().__init__(solver)
		self.module_type = OBJECTIVE
		self.name = "goal_objective"
		LOG_DEBUG("Initializing Goal Module")
		LOG_DEBUG("Goal Module successfully initialized")
		self.goal_weight = 10.0  # Default value for testing

	def update(self, state, data, module_data):
		return

	def define_parameters(self, parameter_manager):
		parameter_manager.add("goal_weight")
		parameter_manager.add("goal_x")
		parameter_manager.add("goal_y")

	def get_value(self, model, params, settings, stage_idx):
		cost = 0

		pos_x = model.get("x")
		pos_y = model.get("y")

		goal_weight = params.get("goal_weight")
		goal_x = params.get("goal_x")
		goal_y = params.get("goal_y")

		cost += goal_weight * ((pos_x - goal_x) ** 2 + (pos_y - goal_y) ** 2) / (goal_x ** 2 + goal_y ** 2 + 0.01)
		return cost

	def set_parameters(self, parameter_manager, data, module_data, k):
		if k == 0:
			LOG_DEBUG("Goal Module.set_parameters()")

		parameter_manager.set_parameter("goal_x", data.goal[0])
		parameter_manager.set_parameter("goal_y", data.goal[1])
		parameter_manager.set_parameter("goal_weight", self.goal_weight)

	def is_data_ready(self, data):
		missing_data = ""
		if not hasattr(data, 'goal') or data.goal is None:
			missing_data += "Goal "
		return len(missing_data) < 1

	def is_objective_reached(self, state, data):
		if not hasattr(data, 'goal') or data.goal is None:
			return False
		# Check if we reached the goal
		return distance(state.getPos(), data.goal) < 1.0


# Classes for simulation
class State:
	def __init__(self, x=0, y=0, v=0, heading=0):
		self.x = x
		self.y = y
		self.v = v
		self.heading = heading
		self.spline = 0.0

	def getPos(self):
		return np.array([self.x, self.y])

	def getVel(self):
		return np.array([self.v * np.cos(self.heading), self.v * np.sin(self.heading)])

	def get(self, name, default=None):
		if name == "x":
			return self.x
		elif name == "y":
			return self.y
		elif name == "v":
			return self.v
		elif name == "heading":
			return self.heading
		elif name == "spline":
			return self.spline
		return default

	def reset(self):
		self.x = 0
		self.y = 0
		self.v = 0
		self.heading = 0
		self.spline = 0.0


class Data:
	def __init__(self):
		self.goal = np.array([10.0, 10.0])  # Default goal
		self.planning_start_time = 0
		self.goal_received = True  # For testing
		self.left_bound = Path()
		self.right_bound = Path()
		self.reference_path = Path()

	def reset(self):
		self.planning_start_time = 0


class ModuleData:
	def __init__(self):
		self.path = None
		self.path_width_left = None
		self.path_width_right = None


class Path:
	def __init__(self):
		self.x = []
		self.y = []
		self.s = []

	def empty(self):
		return len(self.x) == 0


# Simple solver simulator for testing
class TestSolver:
	def __init__(self, dt=0.1, N=20):
		self.dt = dt
		self.horizon = N
		self.timestep = dt
		# Initialize state mapping
		self.state_map = {"x": 0, "y": 1, "v": 2, "heading": 3, "spline": 4}
		self.nx = 5  # State dimensionality
		self.nu = 2  # Control dimensionality (acceleration, steering)

		# Create actual solver
		self.solver = CasADiSolver(dt, N)
		self.solver.nx = self.nx
		self.solver.nu = self.nu
		self.solver.set_state_map(self.state_map)

		# Parameter manager
		self.params = ParameterManager()

		# Module manager
		self.module_manager = ModuleManager()

	def initialize_warmstart(self, state, shift_forward=True):
		self.solver.initialize_warmstart(state, shift_forward)

	def initialize_with_braking(self, state):
		self.solver.initialize_with_braking(state)

	def set_xinit(self, state):
		self.solver.set_initial_state(state)

	def get_ego_prediction(self, k, var_name):
		return self.solver.get_ego_prediction(k, var_name)

	def get_output(self, k, var_name):
		return self.solver.get_output(k, var_name)

	def get_module_manager(self):
		return self.module_manager

	def load_warmstart(self):
		self.solver.load_warmstart()

	def solve(self):
		return self.solver.solve()

	def reset(self):
		self.solver.reset()

	def explain_exit_flag(self, flag):
		return self.solver.explain_exit_flag(flag)


# Test setup and execution
def test_mpc_planner():
	# Set up basic CONFIG
	CONFIG["control_frequency"] = 10
	CONFIG["shift_previous_solution_forward"] = True
	CONFIG["enable_output"] = True
	CONFIG["debug_limits"] = False
	CONFIG["debug_visuals"] = False
	CONFIG["recording"] = {"enable": False}

	# Create simulation objects
	dt = 0.1
	horizon = 20
	test_solver = TestSolver(dt, horizon)
	planner = Planner(test_solver)

	# Add goal objective module
	goal_objective = GoalObjective(test_solver)
	test_solver.module_manager.add_module(goal_objective)

	# Define parameters
	goal_objective.define_parameters(test_solver.params)

	# Initialize the solver problem
	test_solver.solver.finalize_problem()

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