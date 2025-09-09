import time
import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from planner_modules.src.objectives.contouring_objective import ContouringObjective
from planner_modules.src.constraints.scenario_constraints import ScenarioConstraints
from planning.src.data_prep import define_robot_area
from planning.src.dynamic_models import ContouringSecondOrderUnicycleModel
from planning.src.planner import Planner
from planning.src.types import Data, Bound, generate_reference_path, calculate_path_normals, State, \
	generate_dynamic_obstacles, PredictionType, ScenarioSolveStatus
from solver.src.casadi_solver import CasADiSolver
from utils.const import GAUSSIAN
from utils.utils import LOG_INFO, LOG_WARN


class MPCVisualizer:
	"""A dedicated visualizer class for the MPC simulation."""

	def __init__(self, ax, data, vehicle_model, horizon):
		self.ax = ax
		self.data = data
		self.vehicle = vehicle_model
		self.horizon = horizon
		self.dynamic_elements = []

	def setup_static_elements(self):
		"""Plots static environment elements."""
		self.ax.plot(self.data.left_bound.x, self.data.left_bound.y, 'k--', label='Road Boundary')
		self.ax.plot(self.data.right_bound.x, self.data.right_bound.y, 'k--')
		self.ax.plot(self.data.reference_path.x, self.data.reference_path.y, 'g:', label='Reference Path')
		self.ax.plot(self.data.goal[0], self.data.goal[1], 'r*', markersize=20, label='Goal')
		self.ax.plot(self.data.start[0], self.data.start[1], 'go', markersize=10, label='Start')
		self.ax.legend(loc='upper left')

	def clear_dynamic_elements(self):
		"""Removes artists that are updated each frame."""
		for element in self.dynamic_elements:
			element.remove()
		self.dynamic_elements.clear()

	def update_frame(self, current_state, realized_x, realized_y, planner, scenario_constraints):
		"""Master update function called every simulation step."""
		self.clear_dynamic_elements()

		# ** THE FIX IS HERE **
		# Corrected syntax for creating a rotated rectangle
		vehicle_patch = patches.Rectangle(
			(current_state.get("x"), current_state.get("y")),  # Set anchor point at the center
			self.vehicle.length,
			self.vehicle.width,
			angle=np.rad2deg(current_state.get("psi")),
			fc='blue',
			label="Vehicle",
			alpha=0.8,
			transform=matplotlib.transforms.Affine2D().rotate_deg_around(
				current_state.get("x"), current_state.get("y"), np.rad2deg(current_state.get("psi"))
			) + self.ax.transData
		)

		self.ax.add_patch(vehicle_patch)
		self.dynamic_elements.append(vehicle_patch)

		realized_line, = self.ax.plot(realized_x, realized_y, 'b-', linewidth=2, label="Realized Trajectory")
		self.dynamic_elements.append(realized_line)

		# MPC prediction
		forecast = planner.solver.get_forecasts()
		if forecast:
			pred_x = [s.get("x") for s in forecast.get_states()]
			pred_y = [s.get("y") for s in forecast.get_states()]
			pred_line, = self.ax.plot(pred_x, pred_y, 'c--', linewidth=2, label="Predicted Trajectory")
			self.dynamic_elements.append(pred_line)

		# Dynamic obstacles
		for obs in self.data.dynamic_obstacles:
			obs_patch = plt.Circle(obs.position, obs.radius, color='red', alpha=0.7)
			self.ax.add_patch(obs_patch)
			self.dynamic_elements.append(obs_patch)

		# Scenario-specific visualizations
		self.visualize_scenarios(scenario_constraints)

		# Adjust view and redraw
		self.ax.set_xlim(current_state.get("x") - 20, current_state.get("x") + 20)
		self.ax.set_ylim(current_state.get("y") - 20, current_state.get("y") + 20)
		self.ax.figure.canvas.draw()
		self.ax.figure.canvas.flush_events()

	def visualize_scenarios(self, scenario_constraints):
		best_solver = getattr(scenario_constraints, 'best_solver', None)
		if not best_solver or not hasattr(best_solver, 'scenario_module'):
			return

		scenario_module = best_solver.scenario_module
		sampler = scenario_module.get_sampler()

		# Visualize sampled obstacle positions
		if sampler and hasattr(sampler, 'samples') and sampler.samples_ready():
			samples = sampler.samples
			num_steps_to_plot = min(len(samples), self.horizon)
			num_obs_to_plot = len(self.data.dynamic_obstacles)

			for k in range(num_steps_to_plot):
				for obs_id in range(num_obs_to_plot):
					if k < len(samples) and obs_id < len(samples[k]):
						x_samples, y_samples = samples[k][obs_id][0], samples[k][obs_id][1]
						alpha = 0.3 * (1 - k / num_steps_to_plot)
						color = plt.cm.autumn(obs_id / max(1, num_obs_to_plot))
						scatter = self.ax.scatter(x_samples, y_samples, s=5, color=color, alpha=alpha,
												  edgecolors='none', zorder=0)
						self.dynamic_elements.append(scatter)

		# Display solver status
		solve_status = getattr(scenario_module, 'solve_status', ScenarioSolveStatus.INFEASIBLE)
		status_text = f"Scenario Solver: {solve_status.name}"
		status_color = 'green' if solve_status == ScenarioSolveStatus.SUCCESS else 'red'
		status_artist = self.ax.text(0.98, 0.98, status_text, transform=self.ax.transAxes,
									 va='top', ha='right',
									 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
									 color=status_color)
		self.dynamic_elements.append(status_artist)


def run(dt=0.1, horizon=15, model=ContouringSecondOrderUnicycleModel, start=(0.0, 0.0), goal=(50.0, 10.0),
		max_iterations=300):
	casadi_solver = CasADiSolver(dt, horizon)
	vehicle = model()
	casadi_solver.set_dynamics_model(vehicle)

	planner = Planner(casadi_solver, vehicle)

	# Add objectives and constraints
	contouring_objective = ContouringObjective(casadi_solver)
	casadi_solver.module_manager.add_module(contouring_objective)
	scenario_constraints = ScenarioConstraints(casadi_solver)
	casadi_solver.module_manager.add_module(scenario_constraints)

	# Initialize all modules
	casadi_solver.define_parameters()

	# Setup environment data
	data = Data()
	data.start, data.goal = np.array(start), np.array(goal)
	data.reference_path = generate_reference_path(data.start, data.goal, path_type="curved")
	data.dynamic_obstacles = generate_dynamic_obstacles(3, GAUSSIAN, 0.5)

	normals = np.array(calculate_path_normals(data.reference_path))
	road_width = 8.0
	# ** THE FIX IS HERE **: Use half_width for proper lane boundaries
	half_width = road_width / 2
	left_x = data.reference_path.x + normals[:, 0] * half_width
	left_y = data.reference_path.y + normals[:, 1] * half_width
	right_x = data.reference_path.x - normals[:, 0] * half_width
	right_y = data.reference_path.y - normals[:, 1] * half_width

	data.left_bound = Bound(left_x, left_y, data.reference_path.s)
	data.right_bound = Bound(right_x, right_y, data.reference_path.s)
	data.robot_area = define_robot_area(vehicle.length, vehicle.width, 1)

	# Initialize planner state
	initial_state = State(vehicle)
	initial_state.set("x", start[0])
	initial_state.set("y", start[1])
	initial_state.set("psi", np.arctan2(goal[1] - start[1], goal[0] - start[0]))
	planner.set_state(initial_state)

	# Setup Visualization
	matplotlib.use('TkAgg')
	plt.ion()
	fig, ax = plt.subplots(figsize=(12, 12))
	ax.set_title("MPCC with Scenario Constraints"), ax.set_xlabel("X [m]"), ax.set_ylabel("Y [m]")
	ax.grid(True), ax.set_aspect('equal')
	visualizer = MPCVisualizer(ax, data, vehicle, horizon)
	visualizer.setup_static_elements()
	plt.show()

	# Main Simulation Loop
	states_x, states_y = [start[0]], [start[1]]
	for i in range(max_iterations):
		LOG_INFO(f"\n--- Iteration {i} ---")
		data.planning_start_time = time.time()

		# ** THE FIX IS HERE **: Process data and generate scenarios BEFORE solving
		scenario_constraints.on_data_received(data)

		# Solve the MPC problem
		output = planner.solve_mpc(data)

		if output.success:
			# Use the first control action to propagate the state
			next_state = planner.solver.dynamics_model.numeric_rk4(
				planner.get_state(),
				output.forecasts.get_states()[0],  # a, w
				dt
			)
			planner.set_state(next_state)

			states_x.append(next_state.get("x"))
			states_y.append(next_state.get("y"))

			for obs in data.dynamic_obstacles:
				if hasattr(obs, 'update_position'):
					obs.update_position(dt)

			visualizer.update_frame(next_state, states_x, states_y, planner, scenario_constraints)
			plt.pause(0.05)

			if planner.is_objective_reached(data):
				LOG_INFO(f"Objective reached at iteration {i}!")
				break
		else:
			LOG_WARN(f"Iteration {i}: MPC failed. Stopping.")
			visualizer.update_frame(planner.get_state(), states_x, states_y, planner, scenario_constraints)
			break

	LOG_INFO("Simulation finished.")
	plt.ioff()
	ax.set_title("MPCC Simulation Finished")
	plt.show()


# Main execution block
if __name__ == '__main__':
	import logging

	logging.basicConfig(level=logging.DEBUG)  # Use INFO for cleaner output, DEBUG for verbose
	run()