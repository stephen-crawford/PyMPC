import pygame
import numpy as np
import time
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Optional

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
GRAY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)

# Configuration
CONFIG = {
	"N": 30,  # Planning horizon
	"dt": 0.1,  # Time step
	"control_frequency": 10.0,  # Hz
	"max_obstacles": 5,
	"scenario_constraints": {
		"parallel_solvers": 3,
		"enable_safe_horizon": True
	},
	"shift_previous_solution_forward": True,
	"enable_output": True,
}


# Utility classes to match your framework
class Timer:
	def __init__(self, duration):
		self.duration = duration
		self.start_time = time.time()

	def has_finished(self):
		return (time.time() - self.start_time) >= self.duration

	def start(self):
		self.start_time = time.time()


class Benchmarker:
	def __init__(self, name):
		self.name = name
		self.start_time = None
		self.last_duration = 0

	def start(self):
		self.start_time = time.time()

	def stop(self):
		if self.start_time:
			self.last_duration = time.time() - self.start_time
			self.start_time = None

	def is_running(self):
		return self.start_time is not None

	def cancel(self):
		self.start_time = None

	def get_last(self):
		return self.last_duration


class PredictionType(Enum):
	DETERMINISTIC = 0
	UNCERTAIN = 1


# Types to model your framework
@dataclass
class Vector2D:
	x: float
	y: float

	def norm(self):
		return math.sqrt(self.x ** 2 + self.y ** 2)


@dataclass
class State:
	x: float = 0.0
	y: float = 0.0
	v: float = 0.0
	heading: float = 0.0

	def getPos(self):
		return np.array([self.x, self.y])

	def getVel(self):
		return np.array([self.v * math.cos(self.heading), self.v * math.sin(self.heading)])

	def get(self, key, default=0.0):
		return getattr(self, key, default)

	def reset(self):
		self.x = 0.0
		self.y = 0.0
		self.v = 0.0
		self.heading = 0.0


@dataclass
class Prediction:
	mean: List[Vector2D] = None
	covariance: List[np.ndarray] = None
	timestamps: List[float] = None
	type: PredictionType = PredictionType.DETERMINISTIC

	def __post_init__(self):
		if self.mean is None:
			self.mean = []
		if self.covariance is None:
			self.covariance = []
		if self.timestamps is None:
			self.timestamps = []

	def empty(self):
		return len(self.mean) == 0


@dataclass
class DynamicObstacle:
	id: int
	position: Vector2D
	velocity: Vector2D
	radius: float
	prediction: Prediction


@dataclass
class PlannerData:
	dynamic_obstacles: List[DynamicObstacle]
	planning_start_time: float

	def __init__(self):
		self.dynamic_obstacles = []
		self.planning_start_time = time.time()

	def reset(self):
		self.dynamic_obstacles = []
		self.planning_start_time = time.time()


class Trajectory:
	def __init__(self, dt=None, N=None):
		self.points = []
		self.dt = dt
		self.N = N

	def add(self, x, y):
		self.points.append((x, y))

	def clear(self):
		self.points = []

	def size(self):
		return len(self.points)


class PlannerOutput:
	def __init__(self, dt=None, N=None):
		self.trajectory = Trajectory(dt, N)
		self.success = False


# Simplified CasADi solver for demo
class SimplifiedSolver:
	def __init__(self, dt=0.1, N=20):
		self.dt = dt
		self.N = N
		self.nx = 4  # x, y, v, heading
		self.nu = 2  # acceleration, steering
		self.state_map = {"x": 0, "y": 1, "v": 2, "heading": 3}
		self.params = {"solver_timeout": 0.1}
		self._info = {"pobj": float('inf')}
		self.output = None
		self.solver_id = 0

		# For storing the current solution
		self.current_X = np.zeros((N + 1, self.nx))
		self.current_U = np.zeros((N, self.nu))

	def reset(self):
		self.current_X = np.zeros((self.N + 1, self.nx))
		self.current_U = np.zeros((self.N, self.nu))
		self._info = {"pobj": float('inf')}

	def set_xinit(self, state):
		self.current_X[0, 0] = state.x
		self.current_X[0, 1] = state.y
		self.current_X[0, 2] = state.v
		self.current_X[0, 3] = state.heading

	def get_ego_prediction(self, k, var_name):
		if var_name in self.state_map:
			idx = self.state_map[var_name]
			return self.current_X[k, idx]
		return 0.0

	def initialize_warmstart(self, state, shift_forward=True):
		if shift_forward and self.current_X.sum() != 0:
			# Shift solution forward
			self.current_X[:-1] = self.current_X[1:]
			self.current_U[:-1] = self.current_U[1:]

	def initialize_with_braking(self, state):
		# Create a simple braking trajectory
		x0, y0 = state.x, state.y
		v0 = state.v
		heading = state.heading
		decel = 2.0  # m/s^2

		for k in range(self.N + 1):
			t = k * self.dt
			v = max(0, v0 - decel * t)
			s = v0 * t - 0.5 * decel * t * t

			self.current_X[k, 0] = x0 + s * math.cos(heading)
			self.current_X[k, 1] = y0 + s * math.sin(heading)
			self.current_X[k, 2] = v
			self.current_X[k, 3] = heading

			if k < self.N:
				self.current_U[k, 0] = -decel  # Acceleration
				self.current_U[k, 1] = 0.0  # Steering

	def load_warmstart(self):
		# In a real implementation, this would load values into the solver
		pass

	def get_output(self, k, var_name):
		if var_name in self.state_map:
			idx = self.state_map[var_name]
			return self.current_X[k, idx]
		return 0.0

	def solve(self):
		# In a real implementation, this would solve the optimization problem
		# Here we'll just generate a dummy solution to visualize
		return 1  # Success

	def explain_exit_flag(self, code):
		if code == 1:
			return "Solved successfully"
		else:
			return "Solver failed"


# Scenario-based optimization module
class ScenarioModule:
	def __init__(self, solver_id):
		self.solver_id = solver_id
		self.sampler = ScenarioSampler(solver_id)
		self.trajectory = Trajectory()
		self.N = CONFIG["N"]
		self.dt = CONFIG["dt"]
		self.cost = float('inf')

	def update(self, data, module_data):
		# In a real implementation, this would update the scenario module
		pass

	def set_parameters(self, data, k):
		# In a real implementation, this would set parameters for stage k
		pass

	def optimize(self, data):
		# Generate a different trajectory for each solver
		# This simulates finding different solutions based on different sampled scenarios
		self.trajectory.clear()

		# Get ego vehicle position
		if data.dynamic_obstacles:
			ego_x, ego_y = data.dynamic_obstacles[0].position.x, data.dynamic_obstacles[0].position.y
		else:
			ego_x, ego_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

		# Generate a trajectory with some randomness based on solver_id
		seed = self.solver_id + time.time()
		random.seed(seed)

		# Different costs for different solvers
		self.cost = 1000 - self.solver_id * 200 + random.random() * 400

		for k in range(self.N):
			# Add some randomness to make trajectories different
			noise_x = (random.random() - 0.5) * 50 * (k / self.N) * (self.solver_id + 1)
			noise_y = (random.random() - 0.5) * 30 * (k / self.N) * (self.solver_id + 1)

			# Generate a curved path
			t = k * self.dt
			x = ego_x + 200 * math.sin(t * 0.5) + noise_x
			y = ego_y - 150 * t + noise_y

			self.trajectory.add(x, y)

		return 1  # Success

	def get_sampler(self):
		return self.sampler

	def is_data_ready(self, data, missing_data=""):
		return True

	def visualize(self, data):
		# Visualization will be handled by pygame
		pass


class ScenarioSampler:
	def __init__(self, solver_id):
		self.solver_id = solver_id

	def integrate_and_translate_to_mean_and_variance(self, dynamic_obstacles, dt):
		# In a real implementation, this would sample different trajectories
		# For each dynamic obstacle based on their uncertainty
		pass


class ScenarioSolver:
	def __init__(self, solver_id):
		self.solver = SimplifiedSolver(CONFIG["dt"], CONFIG["N"])
		self.solver.solver_id = solver_id
		self.scenario_module = ScenarioModule(solver_id)
		self.exit_code = 0
		self.N = CONFIG["N"]
		self.dt = CONFIG["dt"]
		self._solver_id = solver_id

	def get(self):
		return self


class ScenarioConstraints:
	def __init__(self, solver):
		self.solver = solver
		self.name = "scenario_constraints"
		self._planning_time = 1.0 / CONFIG["control_frequency"]
		self._scenario_solvers = []
		self._best_solver = None

		# Create parallel solvers
		for i in range(CONFIG["scenario_constraints"]["parallel_solvers"]):
			self._scenario_solvers.append(ScenarioSolver(i))

	def update(self, state, data, module_data):
		for solver in self._scenario_solvers:
			# Copy the main solver, including its initial guess
			solver.solver = self.solver
			solver.scenario_module.update(data, module_data)

	def set_parameters(self, data, module_data, k):
		# Not implemented for the demo
		pass

	def optimize(self, state, data, module_data):
		for solver in self._scenario_solvers:
			# Set the planning timeout
			used_time = time.time() - data.planning_start_time
			solver.solver.params = self.solver.params
			solver.solver.params["solver_timeout"] = self._planning_time - used_time - 0.008

			# Copy solver parameters and initial guess
			solver.solver = self.solver

			# Set the scenario constraint parameters for each solver
			for k in range(solver.N):
				solver.scenario_module.set_parameters(data, k)

			# Load the previous solution
			solver.solver.load_warmstart()

			# Run optimization (Safe Horizon MPC)
			solver.exit_code = solver.scenario_module.optimize(data)

			# Set a cost for this solver (would come from actual optimization)
			solver.solver._info = {"pobj": solver.scenario_module.cost}

		# Retrieve the lowest cost solution
		lowest_cost = float('inf')
		self._best_solver = None

		for solver in self._scenario_solvers:
			if solver.exit_code == 1 and solver.solver._info["pobj"] < lowest_cost:
				lowest_cost = solver.solver._info["pobj"]
				self._best_solver = solver

		if self._best_solver is None:  # No feasible solution
			return self._scenario_solvers[0].exit_code

		# Load the solution into the main solver
		self.solver.output = self._best_solver.solver.output
		self.solver._info = self._best_solver.solver._info

		return self._best_solver.exit_code

	def on_data_received(self, data, data_name):
		if data_name == "dynamic obstacles":
			# Check if uncertainty was provided
			for obs in data.dynamic_obstacles:
				assert obs.prediction.type != PredictionType.DETERMINISTIC, (
					"When using Scenario Constraints, the predictions should have a non-zero "
					"uncertainty."
				)

	def is_data_ready(self, data, missing_data=""):
		# Simplified check for demo
		return len(data.dynamic_obstacles) > 0

	def visualize(self, screen):
		# Draw trajectories from all solvers
		for i, solver in enumerate(self._scenario_solvers):
			# Different colors for different solvers
			colors = [BLUE, GREEN, MAGENTA, CYAN, YELLOW]
			color = colors[i % len(colors)]

			# Draw the trajectory
			points = solver.scenario_module.trajectory.points
			if len(points) >= 2:
				pygame.draw.lines(screen, color, False, points, 2)

			# Draw points along the trajectory
			for point in points:
				pygame.draw.circle(screen, color, (int(point[0]), int(point[1])), 3)

			# Show cost near the end of the trajectory
			if points:
				font = pygame.font.SysFont(None, 24)
				cost_text = f"Cost: {solver.solver._info['pobj']:.1f}"
				text_surface = font.render(cost_text, True, color)
				screen.blit(text_surface, (points[-1][0] + 10, points[-1][1] - 10))

		# Highlight the best trajectory
		if self._best_solver:
			points = self._best_solver.scenario_module.trajectory.points
			if len(points) >= 2:
				# Draw a thicker line for the best trajectory
				pygame.draw.lines(screen, RED, False, points, 4)

				# Draw a star at the end of the best trajectory
				if points:
					end_point = points[-1]
					star_size = 10
					pygame.draw.polygon(screen, RED, [
						(end_point[0], end_point[1] - star_size),
						(end_point[0] + star_size // 2, end_point[1] - star_size // 3),
						(end_point[0] + star_size, end_point[1]),
						(end_point[0] + star_size // 2, end_point[1] + star_size // 3),
						(end_point[0], end_point[1] + star_size),
						(end_point[0] - star_size // 2, end_point[1] + star_size // 3),
						(end_point[0] - star_size, end_point[1]),
						(end_point[0] - star_size // 2, end_point[1] - star_size // 3),
					])

					# Indicate that this is the best trajectory
					font = pygame.font.SysFont(None, 24)
					text = font.render("BEST", True, RED)
					screen.blit(text, (end_point[0] + 15, end_point[1] + 15))


class Planner:
	def __init__(self, solver, modules):
		self.solver = solver
		self.modules = modules
		self.startup_timer = Timer(1.0)
		self.was_reset = False
		self.output = None
		self.warmstart = None
		self.benchmarkers = []

		# Initialize modules
		for module in self.modules:
			module.solver = solver

	def solve_mpc(self, state, data):
		was_feasible = self.output.success if self.output else False
		self.output = PlannerOutput(self.solver.dt, self.solver.N)
		module_data = {}

		# Check if data is ready
		is_data_ready = True
		for module in self.modules:
			missing_data = ""
			if not module.is_data_ready(data, missing_data):
				is_data_ready = False
				break

		if not is_data_ready:
			if self.startup_timer.has_finished():
				print("Data is not ready")
			self.output.success = False
			return self.output

		# Create benchmarkers
		planning_benchmarker = Benchmarker("planning")
		planning_benchmarker.start()

		# Initialize solver with previous solution or braking trajectory
		if was_feasible:
			self.solver.initialize_warmstart(state, CONFIG["shift_previous_solution_forward"])
		else:
			self.solver.initialize_with_braking(state)

		# Set initial state
		self.solver.set_xinit(state)

		# Update modules
		for module in self.modules:
			module.update(state, data, module_data)

		# Set parameters for each stage
		for k in range(self.solver.N):
			for module in self.modules:
				module.set_parameters(data, module_data, k)

		# Load warmstart
		self.warmstart = Trajectory()
		for k in range(self.solver.N):
			self.warmstart.add(
				self.solver.get_ego_prediction(k, "x"),
				self.solver.get_ego_prediction(k, "y")
			)
		self.solver.load_warmstart()

		# Set solver timeout
		used_time = time.time() - data.planning_start_time
		self.solver.params["solver_timeout"] = 1.0 / CONFIG["control_frequency"] - used_time - 0.006

		# Run optimization through modules
		exit_flag = -1
		for module in self.modules:
			exit_flag = module.optimize(state, data, module_data)
			if exit_flag != -1:
				break

		if exit_flag == -1:
			exit_flag = self.solver.solve()

		planning_benchmarker.stop()

		if exit_flag != 1:
			self.output.success = False
			print(f"MPC failed: {self.solver.explain_exit_flag(exit_flag)}")
			return self.output

		# Set output trajectory
		self.output.success = True
		for k in range(1, self.solver.N):
			self.output.trajectory.add(
				self.solver.get_output(k, "x"),
				self.solver.get_output(k, "y")
			)

		return self.output

	def visualize(self, screen, state, data):
		# Let modules visualize
		for module in self.modules:
			module.visualize(screen)

		# Draw the final trajectory
		if self.output and self.output.success:
			points = self.output.trajectory.points
			if len(points) >= 2:
				pygame.draw.lines(screen, RED, False, points, 3)


# Game class for handling the demo
class ScenarioOptimizerDemo:
	def __init__(self):
		pygame.init()
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
		pygame.display.set_caption("Scenario-Based Trajectory Optimization Demo")
		self.clock = pygame.time.Clock()
		self.running = True

		# Create solver
		self.solver = SimplifiedSolver(CONFIG["dt"], CONFIG["N"])

		# Create modules
		self.scenario_constraints = ScenarioConstraints(self.solver)
		self.modules = [self.scenario_constraints]

		# Create planner
		self.planner = Planner(self.solver, self.modules)

		# Create state and data
		self.state = State(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100, 20.0, -math.pi / 2)
		self.data = PlannerData()
		self.planner.output = PlannerOutput()

		# Create obstacles
		self.create_obstacles()

		# Add the ego vehicle as the first obstacle to help visualize
		ego_obstacle = DynamicObstacle(
			id=0,
			position=Vector2D(self.state.x, self.state.y),
			velocity=Vector2D(self.state.v * math.cos(self.state.heading),
			                  self.state.v * math.sin(self.state.heading)),
			radius=15,
			prediction=Prediction(type=PredictionType.UNCERTAIN)
		)
		self.data.dynamic_obstacles.insert(0, ego_obstacle)

		# Font for text
		self.font = pygame.font.SysFont(None, 24)

		# Last planning time
		self.last_planning_time = 0
		self.next_planning_time = time.time()

	def create_obstacles(self):
		# Create some dynamic obstacles with uncertain predictions
		obstacles = []

		# Add some random obstacles
		for i in range(1, CONFIG["max_obstacles"]):
			# Random position in the top half of the screen
			x = random.randint(100, SCREEN_WIDTH - 100)
			y = random.randint(100, SCREEN_HEIGHT // 2)

			# Random velocity
			vx = random.uniform(-10, 10)
			vy = random.uniform(-5, 5)

			# Create prediction
			prediction = Prediction(type=PredictionType.UNCERTAIN)
			for t in range(20):
				mean_pos = Vector2D(x + vx * t * CONFIG["dt"], y + vy * t * CONFIG["dt"])
				prediction.mean.append(mean_pos)

				# Add growing uncertainty over time
				cov = np.array([[t * 5, 0], [0, t * 5]])
				prediction.covariance.append(cov)
				prediction.timestamps.append(t * CONFIG["dt"])

			obstacle = DynamicObstacle(
				id=i,
				position=Vector2D(x, y),
				velocity=Vector2D(vx, vy),
				radius=10 + random.randint(5, 15),
				prediction=prediction
			)
			obstacles.append(obstacle)

		self.data.dynamic_obstacles = obstacles

	def update_state_and_obstacles(self):
		# Update ego vehicle state
		ego_speed = 20.0  # pixels per second

		# Move ego vehicle along the best path if available
		if self.planner.output and self.planner.output.success and self.planner.output.trajectory.size() > 0:
			target = self.planner.output.trajectory.points[0]
			dx = target[0] - self.state.x
			dy = target[1] - self.state.y
			distance = math.sqrt(dx * dx + dy * dy)

			if distance > 0:
				# Move towards the next point on the trajectory
				step = min(ego_speed * CONFIG["dt"], distance)
				self.state.x += step * dx / distance
				self.state.y += step * dy / distance
				self.state.heading = math.atan2(dy, dx)

		# Update dynamic obstacles
		for obstacle in self.data.dynamic_obstacles[1:]:  # Skip ego vehicle
			obstacle.position.x += obstacle.velocity.x * CONFIG["dt"]
			obstacle.position.y += obstacle.velocity.y * CONFIG["dt"]

			# Bounce off edges
			if obstacle.position.x < 0 or obstacle.position.x > SCREEN_WIDTH:
				obstacle.velocity.x *= -1
			if obstacle.position.y < 0 or obstacle.position.y > SCREEN_HEIGHT:
				obstacle.velocity.y *= -1

			# Update predictions
			for i, mean_pos in enumerate(obstacle.prediction.mean):
				mean_pos.x = obstacle.position.x + obstacle.velocity.x * obstacle.prediction.timestamps[i]
				mean_pos.y = obstacle.position.y + obstacle.velocity.y * obstacle.prediction.timestamps[i]

		# Update ego obstacle position to match state
		if self.data.dynamic_obstacles:
			self.data.dynamic_obstacles[0].position.x = self.state.x
			self.data.dynamic_obstacles[0].position.y = self.state.y
			self.data.dynamic_obstacles[0].velocity.x = self.state.v * math.cos(self.state.heading)
			self.data.dynamic_obstacles[0].velocity.y = self.state.v * math.sin(self.state.heading)

	def run(self):
		while self.running:
			# Handle events
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					self.running = False
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_ESCAPE:
						self.running = False
					elif event.key == pygame.K_r:
						# Reset the demo
						self.create_obstacles()
						self.state = State(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100, 20.0, -math.pi / 2)
						self.data.dynamic_obstacles[0].position.x = self.state.x
						self.data.dynamic_obstacles[0].position.y = self.state.y

			# Clear the screen
			self.screen.fill(WHITE)

			# Run planner periodically
			current_time = time.time()
			if current_time >= self.next_planning_time:
				# Set planning start time
				self.data.planning_start_time = current_time

				# Solve MPC
				start_time = time.time()
				self.planner.solve_mpc(self.state, self.data)
				self.last_planning_time = time.time() - start_time

				# Schedule next planning
				self.next_planning_time = current_time + 1.0 / CONFIG["control_frequency"]

			# Update state and obstacles
			self.update_state_and_obstacles()

			# Draw the environment grid
			self.draw_grid()

			# Visualize obstacles with their predictions
			self.visualize_obstacles()

			# Visualize trajectories
			self.planner.visualize(self.screen, self.state, self.data)

			# Draw ego vehicle
			self.draw_ego_vehicle()

			# Draw status information
			self.draw_status()

			# Update the display
			pygame.display.flip()

			# Cap the frame rate
			self.clock.tick(FPS)

		pygame.quit()

	def draw_grid(self):
		# Draw vertical lines
		for x in range(0, SCREEN_WIDTH, 50):
			pygame.draw.line(self.screen, GRAY, (x, 0), (x, SCREEN_HEIGHT), 1)

		# Draw horizontal lines
		for y in range(0, SCREEN_HEIGHT, 50):
			pygame.draw.line(self.screen, GRAY, (0, y), (SCREEN_WIDTH, y), 1)

	def visualize_obstacles(self):
		# Draw each obstacle
		for i, obstacle in enumerate(self.data.dynamic_obstacles[1:], 1):  # Skip ego vehicle
			# Draw obstacle
			pygame.draw.circle(
				self.screen,
				BLACK if i < len(self.data.dynamic_obstacles) else RED,
				(int(obstacle.position.x), int(obstacle.position.y)),
				int(obstacle.radius)
			)

			# Draw prediction mean path
			points = [(mean.x, mean.y) for mean in obstacle.prediction.mean]
			if len(points) >= 2:
				pygame.draw.lines(self.screen, BLACK, False, points, 1)

			# Draw prediction uncertainty ellipses for selected points
			for j in range(0, len(obstacle.prediction.mean), 5):  # Every 5th point
				mean = obstacle.prediction.mean[j]
				cov = obstacle.prediction.covariance[j]

				# Draw simplified uncertainty ellipse
				rx = math.sqrt(cov[0, 0])
				ry = math.sqrt(cov[1, 1])
				pygame.draw.ellipse(
					self.screen,
					LIGHT_BLUE,
					(int(mean.x - rx), int(mean.y - ry), int(2 * rx), int(2 * ry)),
					1
				)

	def draw_ego_vehicle(self):
		# Draw the ego vehicle as a triangle pointing in the direction of heading
		x, y = self.state.x, self.state.y
		r = 15  # size of triangle

		# Calculate triangle points based on heading
		head_x = x + r * math.cos(self.state.heading)
		head_y = y + r * math.sin(self.state.heading)

		left_x = x + r * math.cos(self.state.heading + 2.5)
		left_y = y + r * math.sin(self.state.heading + 2.5)

		right_x = x + r * math.cos(self.state.heading - 2.5)
		right_y = y + r * math.sin(self.state.heading - 2.5)

		# Draw the triangle
		pygame.draw.polygon(self.screen, RED, [(head_x, head_y), (left_x, left_y), (right_x, right_y)])

		# Draw a circle at the center
		pygame.draw.circle(self.screen, RED, (int(x), int(y)), 5)

	def draw_status(self):
		# Display planning information
		info_text = [
			f"FPS: {int(self.clock.get_fps())}",
			f"Planning Time: {self.last_planning_time * 1000:.1f} ms",
			f"Control Frequency: {CONFIG['control_frequency']:.1f} Hz",
			f"Ego Speed: {self.state.v:.1f} px/s",
			f"Obstacles: {len(self.data.dynamic_obstacles) - 1}",
			f"Planning Horizon: {CONFIG['N']} steps",
			f"Status: {'Success' if self.planner.output.success else 'Failure'}"
		]

		# Add instructions
		instructions = [
			"Press R to reset simulation",
			"Press ESC to quit"
		]

		# Render information
		y_offset = 20
		for text in info_text:
			text_surface = self.font.render(text, True, BLACK)
			self.screen.blit(text_surface, (20, y_offset))
			y_offset += 25

		# Add a separator
		pygame.draw.line(self.screen, BLACK, (20, y_offset), (250, y_offset), 1)
		y_offset += 10

		# Render instructions
		for text in instructions:
			text_surface = self.font.render(text, True, BLACK)
			self.screen.blit(text_surface, (20, y_offset))
			y_offset += 25

# Main execution
if __name__ == "__main__":
	# Create and run the demo
	demo = ScenarioOptimizerDemo()
	demo.run()