import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.patches import Ellipse
from scipy.interpolate import CubicSpline

from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
from planner_modules.src.constraints.linearized_constraints import LinearizedConstraints
from planner_modules.src.constraints.scenario_constraints import ScenarioConstraints
from planner_modules.src.objectives.contouring_objective import ContouringObjective
from planning.src.data_prep import define_robot_area
from planning.src.dynamic_models import ContouringSecondOrderUnicycleModel, numeric_rk4
from planning.src.planner import Planner
from planning.src.types import Data, Bound, generate_reference_path, calculate_path_normals, State, \
	generate_dynamic_obstacles, PredictionType
from solver.src.casadi_solver import CasADiSolver
from utils.const import GAUSSIAN
from utils.utils import LOG_DEBUG

from scipy.interpolate import CubicSpline
from utils.utils import LOG_DEBUG

def run(dt=0.1, horizon=15, model=ContouringSecondOrderUnicycleModel, start=(0.0, 0.0), goal=(20.0, 20.0),
		max_iterations=300):
	"""
	Updated test function that works with the corrected scenario module
	"""

	# Initialize solver and vehicle
	casadi_solver = CasADiSolver(dt, horizon)
	vehicle = model()
	casadi_solver.set_dynamics_model(vehicle)

	# Create the planner with corrected module initialization
	planner = Planner(casadi_solver, vehicle)

	# Add modules with proper error handling
	try:
		contouring_objective = ContouringObjective(casadi_solver)
		casadi_solver.module_manager.add_module(contouring_objective)

		contouring_constraints = ContouringConstraints(casadi_solver)
		casadi_solver.module_manager.add_module(contouring_constraints)

		# Initialize scenario constraints with the corrected implementation
		scenario_constraints = ScenarioConstraints(casadi_solver)
		casadi_solver.module_manager.add_module(scenario_constraints)

	except Exception as e:
		LOG_DEBUG(f"Error initializing modules: {e}")
		raise

	# Initialize data
	data = Data()
	data.start = np.array(start)
	data.goal = np.array(goal)
	data.goal_received = True
	data.planning_start_time = 0.0

	# Generate reference path
	reference_path = generate_reference_path(data.start, data.goal, path_type="curved")
	data.reference_path = reference_path

	# Generate dynamic obstacles with proper validation
	try:
		dynamic_obstacles = generate_dynamic_obstacles(10, GAUSSIAN, 0.5)
		# Validate obstacle structure for scenario constraints
		for i, obs in enumerate(dynamic_obstacles):
			if not hasattr(obs, 'prediction') or obs.prediction is None:
				LOG_DEBUG(f"Warning: Obstacle {i} missing prediction data")
			elif hasattr(obs.prediction, 'type') and obs.prediction.type != PredictionType.GAUSSIAN:
				LOG_DEBUG(f"Warning: Obstacle {i} prediction type is not GAUSSIAN")

		data.dynamic_obstacles = dynamic_obstacles

	except Exception as e:
		LOG_DEBUG(f"Error generating dynamic obstacles: {e}")
		# Create minimal obstacles for testing
		data.dynamic_obstacles = []

	# Calculate path normals and boundaries
	normals = calculate_path_normals(data.reference_path)

	# Road width configuration
	road_width = getattr(data, 'road_width', 8.0)
	half_width = road_width / 2
	quarter_width = half_width / 2

	# Generate boundaries
	left_x, left_y, right_x, right_y = [], [], [], []

	for i in range(len(data.reference_path.x)):
		nx, ny = normals[i]

		# Left boundary
		left_x.append(data.reference_path.x[i] + nx * quarter_width)
		left_y.append(data.reference_path.y[i] + ny * quarter_width)

		# Right boundary
		right_x.append(data.reference_path.x[i] - nx * quarter_width)
		right_y.append(data.reference_path.y[i] - ny * quarter_width)

	# Create boundary splines
	left_boundary_spline_x = CubicSpline(data.reference_path.s, np.array(left_x))
	left_boundary_spline_y = CubicSpline(data.reference_path.s, np.array(left_y))
	right_boundary_spline_x = CubicSpline(data.reference_path.s, np.array(right_x))
	right_boundary_spline_y = CubicSpline(data.reference_path.s, np.array(right_y))

	# Store boundary data
	data.left_boundary_x = left_x
	data.left_boundary_y = left_y
	data.right_boundary_x = right_x
	data.right_boundary_y = right_y
	data.left_spline_x = left_boundary_spline_x
	data.left_spline_y = left_boundary_spline_y
	data.right_spline_x = right_boundary_spline_x
	data.right_spline_y = right_boundary_spline_y

	# Add boundaries to data
	data.left_bound = Bound(left_x, left_y, data.reference_path.s)
	data.right_bound = Bound(right_x, right_y, data.reference_path.s)

	# Define robot area
	data.robot_area = define_robot_area(vehicle.length, vehicle.width, 1)

	# Add solver parameters
	casadi_solver.parameter_manager.add("solver_timeout", 10.0)

	# Create initial state
	state = State(model())
	state.set("x", start[0])
	state.set("y", start[1])
	state.set("psi", 0.1)
	state.set("v", 0.0)
	state.set("spline", 0.0)
	state.set("a", 0.0)
	state.set("w", 0.0)
	planner.set_state(state)

	# Initialize planner
	try:
		planner.initialize(data)
	except Exception as e:
		LOG_DEBUG(f"Error initializing planner: {e}")
		raise

	# Check if scenario constraints are ready
	if hasattr(scenario_constraints, 'is_data_ready'):
		if not scenario_constraints.is_data_ready(data):
			LOG_DEBUG("Scenario constraints report data not ready")
		else:
			LOG_DEBUG("Scenario constraints data ready")

	# Initialize plotting
	matplotlib.use('TkAgg')
	plt.ion()
	fig, ax = plt.subplots(figsize=(12, 10))
	ax.set_title("MPC with Contouring Objective & Scenario Constraints")
	ax.set_xlabel("X [m]")
	ax.set_ylabel("Y [m]")
	ax.grid(True)
	ax.set_aspect('equal')

	# Calculate plot limits
	all_x = left_x + right_x + [start[0], goal[0]]
	all_y = left_y + right_y + [start[1], goal[1]]

	if data.dynamic_obstacles:
		for obs in data.dynamic_obstacles:
			all_x.append(float(obs.position[0]))
			all_y.append(float(obs.position[1]))

	margin = 2.0
	x_min, x_max = min(all_x) - margin, max(all_x) + margin
	y_min, y_max = min(all_y) - margin, max(all_y) + margin

	ax.set_xlim(x_min, x_max)
	ax.set_ylim(y_min, y_max)

	# Plot static elements
	ax.plot(data.left_boundary_x, data.left_boundary_y, 'r--', label='Left boundary', linewidth=2)
	ax.plot(data.right_boundary_x, data.right_boundary_y, 'm--', label='Right boundary', linewidth=2)
	ax.plot(data.reference_path.x, data.reference_path.y, 'k:', label='Reference path', linewidth=1)
	ax.plot(data.goal[0], data.goal[1], 'r*', markersize=15, label='Goal')
	ax.plot(data.start[0], data.start[1], 'go', markersize=10, label='Start')

	# Dynamic plot elements
	vehicle_dot, = ax.plot([], [], 'bo', markersize=10, label="Vehicle")
	trajectory_line, = ax.plot([], [], 'b-', linewidth=3, label="Trajectory")

	# Obstacle patches
	obstacle_patches = []
	if data.dynamic_obstacles:
		for obs in data.dynamic_obstacles:
			circle = plt.Circle((float(obs.position[0]), float(obs.position[1])),
								getattr(obs, "radius", 0.5), color='black', alpha=0.6)
			ax.add_patch(circle)
			obstacle_patches.append(circle)

	ax.legend(loc='upper right')
	plt.draw()
	plt.pause(0.1)

	# Simulation variables
	success_flags = []
	control_inputs = []
	states_x, states_y = [start[0]], [start[1]]

	# Main simulation loop with enhanced error handling
	for i in range(max_iterations):
		try:
			data.planning_start_time = i * dt

			# Update obstacle data if needed
			if hasattr(scenario_constraints, 'on_data_received'):
				scenario_constraints.on_data_received(data)

			# Solve MPC
			output = planner.solve_mpc(data)
			success_flags.append(output.success)

			if output.success:
				# Extract control inputs
				try:
					next_a = output.trajectory_history[-1].get_states()[1].get("a")
					next_w = output.trajectory_history[-1].get_states()[1].get("w")
				except (IndexError, AttributeError) as e:
					LOG_DEBUG(f"Error extracting control inputs: {e}")
					next_a, next_w = 0.0, 0.0

				# Update state
				current_state = planner.get_state()
				z_k = [next_a, next_w, current_state.get("x"), current_state.get("y"),
					   current_state.get("psi"), current_state.get("v"), current_state.get("spline")]
				z_k = ca.vertcat(*z_k)
				vehicle.load(z_k)

				# Compute next state
				next_state_symbolic = vehicle.discrete_dynamics(z_k, casadi_solver.parameter_manager,
																casadi_solver.timestep)
				next_state = numeric_rk4(next_state_symbolic, vehicle, casadi_solver.parameter_manager,
										 casadi_solver.timestep)

				next_x = float(next_state[0])
				next_y = float(next_state[1])
				next_psi = float(next_state[2])
				next_v = float(next_state[3])
				next_spline = float(next_state[4])

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

				# Update trajectory history
				states_x.append(next_x)
				states_y.append(next_y)
				control_inputs.append([next_a, next_w])

				# Update plot
				vehicle_dot.set_data([next_x], [next_y])
				trajectory_line.set_data(states_x, states_y)

				# Update obstacles if they move
				if data.dynamic_obstacles:
					for idx, obs in enumerate(data.dynamic_obstacles):
						if idx < len(obstacle_patches):
							if hasattr(obs, 'update_position'):
								obs.update_position(data.planning_start_time)
							obstacle_patches[idx].center = (float(obs.position[0]), float(obs.position[1]))

				# Enhanced constraint visualization
				try:
					visualize_scenario_constraints(ax, scenario_constraints, data, new_state, i)
				except Exception as e:
					LOG_DEBUG(f"Error visualizing constraints: {e}")

				# Update plot
				fig.canvas.draw()
				fig.canvas.flush_events()
				plt.pause(0.1)

				# Check if goal reached
				if planner.is_objective_reached(data):
					print(f"Objective reached at iteration {i}!")
					break

			else:
				print(f"Iteration {i}: MPC failed!")
				LOG_DEBUG(f"MPC failure at iteration {i}")

		except Exception as e:
			LOG_DEBUG(f"Error in main loop iteration {i}: {e}")
			success_flags.append(False)

		finally:
			# Always reset solver
			try:
				casadi_solver.reset()
			except Exception as e:
				LOG_DEBUG(f"Error resetting solver: {e}")

	# Keep plot open
	plt.ioff()
	plt.show()

	# Return results
	return data, planner.output.realized_trajectory, planner.solver.get_forecasts(), success_flags


def visualize_scenario_constraints(ax, scenario_constraints, data, current_state, iteration):
	"""
	Visualize scenario constraints and their status

	Args:
		ax: matplotlib axis
		scenario_constraints: ScenarioConstraints object
		data: Data object
		current_state: current vehicle state
		iteration: current iteration number
	"""

	# Clear previous constraint visualizations
	for artist in getattr(ax, "_scenario_lines", []):
		try:
			artist.remove()
		except:
			pass
	ax._scenario_lines = []

	# Get vehicle position
	vehicle_pos = np.array([current_state.get("x"), current_state.get("y")])

	print(f"\n=== Scenario Constraint Visualization - Iteration {iteration} ===")
	print(f"Vehicle position: ({vehicle_pos[0]:.3f}, {vehicle_pos[1]:.3f})")

	# Check if scenario solvers are available
	if not hasattr(scenario_constraints, 'scenario_solvers') or not scenario_constraints.scenario_solvers:
		print("No scenario solvers available")
		return

	# Get constraint information from the best solver
	best_solver = getattr(scenario_constraints, 'best_solver', None)
	if best_solver is None and scenario_constraints.scenario_solvers:
		best_solver = scenario_constraints.scenario_solvers[0]

	if best_solver is None:
		print("No best solver available")
		return

	# Check solver status
	solver_status = getattr(best_solver, 'exit_code', -1)
	solve_status = getattr(best_solver.scenario_module, 'solve_status', 'UNKNOWN')

	print(f"Best solver: {best_solver.solver_id}")
	print(f"Exit code: {solver_status}")
	print(f"Solve status: {solve_status}")

	# Visualize constraint information
	constraint_info_text = f"Scenario Constraints Status:\\n"
	constraint_info_text += f"Solver ID: {getattr(best_solver, 'solver_id', 'N/A')}\\n"
	constraint_info_text += f"Exit Code: {solver_status}\\n"
	constraint_info_text += f"Status: {solve_status}"

	# Add status color
	if solver_status == 1:  # Success
		status_color = 'green'
	else:
		status_color = 'red'

	# Display constraint status
	ax.text(0.02, 0.85, constraint_info_text, transform=ax.transAxes,
			fontsize=10, va='top', ha='left',
			bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
			color=status_color)

	# Visualize sample-based constraints if available
	try:
		visualize_sample_constraints(ax, best_solver, data, vehicle_pos)
	except Exception as e:
		print(f"Error visualizing sample constraints: {e}")

	# Update title with constraint status
	feasible_text = "FEASIBLE" if solver_status == 1 else "INFEASIBLE"
	ax.set_title(f"MPC with Scenario Constraints - Iteration {iteration}: {feasible_text}")


def visualize_sample_constraints(ax, solver, data, vehicle_pos):
	"""
	Visualize sample-based constraints from scenario sampling

	Args:
		ax: matplotlib axis
		solver: scenario solver
		data: data object
		vehicle_pos: current vehicle position
	"""

	if not hasattr(solver, 'scenario_module') or solver.scenario_module is None:
		return

	scenario_module = solver.scenario_module
	sampler = scenario_module.get_sampler()

	if sampler is None or not sampler.samples_ready():
		print("Sampler not ready or unavailable")
		return

	# Get samples if available
	if hasattr(sampler, 'samples') and sampler.samples:
		samples = sampler.samples

		print(f"Visualizing samples: {len(samples)} time steps")

		# Visualize samples for first few time steps
		for k in range(min(3, len(samples))):  # Show first 3 time steps
			if k >= len(samples):
				continue

			step_samples = samples[k]

			for obs_id in range(min(len(step_samples), len(data.dynamic_obstacles))):
				if obs_id >= len(step_samples):
					continue

				obs_samples = step_samples[obs_id]

				if len(obs_samples) >= 2:  # x and y coordinates
					x_samples = obs_samples[0]
					y_samples = obs_samples[1]

					if len(x_samples) > 0 and len(y_samples) > 0:
						# Plot sample points
						alpha = max(0.1, 0.5 - k * 0.15)  # Fade with time
						color = plt.cm.viridis(k / 3.0)  # Different color for each time step

						scatter = ax.scatter(x_samples, y_samples,
											 s=10, alpha=alpha, color=color,
											 label=f'Samples k={k}, obs={obs_id}' if k == 0 and obs_id == 0 else "")
						ax._scenario_lines.append(scatter)

		print(f"Displayed samples for obstacles: {min(len(step_samples), len(data.dynamic_obstacles))}")

	# Visualize constraint boundaries if available
	try:
		visualize_constraint_boundaries(ax, scenario_module, data, vehicle_pos)
	except Exception as e:
		print(f"Error visualizing constraint boundaries: {e}")


def visualize_constraint_boundaries(ax, scenario_module, data, vehicle_pos):
	"""
	Visualize constraint boundaries from scenario module

	Args:
		ax: matplotlib axis
		scenario_module: scenario module
		data: data object
		vehicle_pos: vehicle position
	"""

	# Get axis limits for plotting constraint boundaries
	x_min, x_max = ax.get_xlim()
	y_min, y_max = ax.get_ylim()

	# Check if disc managers have constraint information
	if not hasattr(scenario_module, 'disc_manager') or not scenario_module.disc_manager:
		return

	constraint_count = 0

	for disc_id, disc in enumerate(scenario_module.disc_manager):
		if not hasattr(disc, 'constraint_matrices'):
			continue

		# This would depend on your specific constraint structure
		# For now, we'll add a placeholder visualization

		# Draw a circle around vehicle showing constraint region
		constraint_circle = plt.Circle(vehicle_pos, 2.0,
									   fill=False, color='orange',
									   linestyle='--', linewidth=2,
									   alpha=0.7, label=f'Constraint Region {disc_id}' if disc_id == 0 else "")
		ax.add_patch(constraint_circle)
		ax._scenario_lines.append(constraint_circle)

		constraint_count += 1

	if constraint_count > 0:
		print(f"Visualized {constraint_count} constraint regions")


def add_constraint_debugging_info(ax, scenario_constraints, data, current_state, iteration):
	"""
	Add detailed debugging information about scenario constraints

	Args:
		ax: matplotlib axis
		scenario_constraints: ScenarioConstraints object
		data: data object
		current_state: current vehicle state
		iteration: current iteration
	"""

	print(f"\\n=== SCENARIO CONSTRAINT DEBUGGING - Iteration {iteration} ===")

	vehicle_pos = np.array([current_state.get("x"), current_state.get("y")])
	print(f"Vehicle position: ({vehicle_pos[0]:.3f}, {vehicle_pos[1]:.3f})")

	# Check data readiness
	if hasattr(scenario_constraints, 'is_data_ready'):
		data_ready = scenario_constraints.is_data_ready(data)
		print(f"Data ready: {data_ready}")

	# Check obstacle information
	if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles:
		print(f"Number of dynamic obstacles: {len(data.dynamic_obstacles)}")

		for i, obs in enumerate(data.dynamic_obstacles):
			obs_pos = np.array([float(obs.position[0]), float(obs.position[1])])
			distance = np.linalg.norm(obs_pos - vehicle_pos)

			prediction_type = "N/A"
			if hasattr(obs, 'prediction') and obs.prediction is not None:
				prediction_type = getattr(obs.prediction, 'type', 'Unknown')

			print(f"  Obstacle {i}: pos=({obs_pos[0]:.3f}, {obs_pos[1]:.3f}), "
				  f"radius={getattr(obs, 'radius', 'N/A')}, "
				  f"distance={distance:.3f}, prediction_type={prediction_type}")

	# Check solver status
	if hasattr(scenario_constraints, 'scenario_solvers'):
		print(f"Number of scenario solvers: {len(scenario_constraints.scenario_solvers)}")

		for i, solver in enumerate(scenario_constraints.scenario_solvers):
			exit_code = getattr(solver, 'exit_code', 'N/A')
			print(f"  Solver {i}: exit_code={exit_code}")

	# Check best solver
	best_solver = getattr(scenario_constraints, 'best_solver', None)
	if best_solver is not None:
		print(f"Best solver ID: {getattr(best_solver, 'solver_id', 'N/A')}")
	else:
		print("No best solver selected")


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