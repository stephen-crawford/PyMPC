import pygame
import numpy as np
import time
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Optional

from planner.src.planner import Planner
from planner.src.types import PredictionType, Prediction, DynamicObstacle, TwoDimensionalSpline
from planner_modules.scenario_constraints import ScenarioConstraints
from solver.casadi_solver import CasADiSolver
from solver.state import State
from utils.utils import Timer, Benchmarker

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
		self.solver = CasADiSolver(CONFIG["dt"], CONFIG["N"])
		self.solver.solver_id = solver_id
		self.scenario_module = ScenarioModule(solver_id)
		self.exit_code = 0
		self.N = CONFIG["N"]
		self.dt = CONFIG["dt"]
		self._solver_id = solver_id

	def get(self):
		return self

	def visualize(self, screen, state, data):
		# Let modules visualize
		for module in self.modules:
			module.visualize(screen)

		# Draw the final trajectory
		if self.output and self.output.success:
			points = self.output.trajectory.points
			if len(points) >= 2:
				pygame.draw.lines(screen, RED, False, points, 3)


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

		# Create your State class instead of the simplified one
		self.state = State()

		# Initialize state with the ego vehicle's position and attributes
		self.state.set("x", SCREEN_WIDTH // 2)
		self.state.set("y", SCREEN_HEIGHT - 100)
		self.state.set("v", 20.0)
		self.state.set("heading", -math.pi / 2)

		self.planner.output = PlannerOutput()

		# Create obstacles
		self.data = self.create_data_object()  # You need to implement this method
		self.create_obstacles()

		# Add the ego vehicle as the first obstacle to help visualize
		ego_obstacle = DynamicObstacle(
			id=0,
			position=TwoDimensionalSpline(self.state.get("x"), self.state.get("y")),
			velocity=TwoDimensionalSpline(
				self.state.get("v") * math.cos(self.state.get("heading")),
				self.state.get("v") * math.sin(self.state.get("heading"))
			),
			radius=15,
			prediction=Prediction(type=PredictionType.UNCERTAIN)
		)
		self.data.dynamic_obstacles.insert(0, ego_obstacle)

		# Font for text
		self.font = pygame.font.SysFont(None, 24)

		# Last planning time
		self.last_planning_time = 0
		self.next_planning_time = time.time()

	def create_data_object(self):
		# Create an empty data object structure
		class Data:
			def __init__(self):
				self.dynamic_obstacles = []
				self.planning_start_time = 0
				# Add other fields your planner expects
				self.reference_path = None
				self.left_bound = self.create_empty_spline()
				self.right_bound = self.create_empty_spline()

			def create_empty_spline(self):
				class EmptySpline:
					def __init__(self):
						self.x = []
						self.y = []
						self.s = []

					def empty(self):
						return len(self.x) == 0

				return EmptySpline()

		return Data()

	def update_state_and_obstacles(self):
		# Update ego vehicle state
		ego_speed = 20.0  # pixels per second

		# Move ego vehicle along the best path if available
		if self.planner.output and self.planner.output.success and self.planner.output.trajectory.size() > 0:
			target = self.planner.output.trajectory.points[0]
			dx = target[0] - self.state.get("x")
			dy = target[1] - self.state.get("y")
			distance = math.sqrt(dx * dx + dy * dy)

			if distance > 0:
				# Move towards the next point on the trajectory
				step = min(ego_speed * CONFIG["dt"], distance)
				new_x = self.state.get("x") + step * dx / distance
				new_y = self.state.get("y") + step * dy / distance
				new_heading = math.atan2(dy, dx)

				# Update the state with new values
				self.state.set("x", new_x)
				self.state.set("y", new_y)
				self.state.set("heading", new_heading)

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
			self.data.dynamic_obstacles[0].position.x = self.state.get("x")
			self.data.dynamic_obstacles[0].position.y = self.state.get("y")
			self.data.dynamic_obstacles[0].velocity.x = self.state.get("v") * math.cos(self.state.get("heading"))
			self.data.dynamic_obstacles[0].velocity.y = self.state.get("v") * math.sin(self.state.get("heading"))

	def draw_ego_vehicle(self):
		# Draw the ego vehicle as a triangle pointing in the direction of heading
		x, y = self.state.get("x"), self.state.get("y")
		heading = self.state.get("heading")
		r = 15  # size of triangle

		# Calculate triangle points based on heading
		head_x = x + r * math.cos(heading)
		head_y = y + r * math.sin(heading)

		left_x = x + r * math.cos(heading + 2.5)
		left_y = y + r * math.sin(heading + 2.5)

		right_x = x + r * math.cos(heading - 2.5)
		right_y = y + r * math.sin(heading - 2.5)

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
			f"Ego Speed: {self.state.get('v'):.1f} px/s",
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

	# Handle reset keypress to reset the demo
	def handle_reset(self):
		self.create_obstacles()

		# Reset the state using your State class methods
		self.state = State()
		self.state.set("x", SCREEN_WIDTH // 2)
		self.state.set("y", SCREEN_HEIGHT - 100)
		self.state.set("v", 20.0)
		self.state.set("heading", -math.pi / 2)

		self.data.dynamic_obstacles[0].position.x = self.state.get("x")
		self.data.dynamic_obstacles[0].position.y = self.state.get("y")

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
						# Reset using our custom reset method
						self.handle_reset()

			# Clear the screen
			self.screen.fill(WHITE)

			# Run planner periodically
			current_time = time.time()
			# but make sure to update the solver's initial state
			if current_time >= self.next_planning_time:
				# Update the solver with our state
				self.solver.set_xinit(self.state)

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