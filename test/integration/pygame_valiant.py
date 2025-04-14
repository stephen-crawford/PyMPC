import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import sys
import random
import math
from pygame.locals import *

# Initialize pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dynamic Sensing Gated Scenario-based Stochastic MPC")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
DARK_RED = (139, 0, 0)

# Fonts
font_small = pygame.font.SysFont("Arial", 16)
font_medium = pygame.font.SysFont("Arial", 20)
font_large = pygame.font.SysFont("Arial", 24)
font_xl = pygame.font.SysFont("Arial", 32)


class DynamicObstacleAgent:
	def __init__(self, position=None):
		if position is None:
			self.position = np.array([WIDTH // 2, HEIGHT // 2], dtype=np.float64)
		else:
			self.position = position.astype(np.float64)  # Ensure position is float

		# History and trajectory
		self.position_history = [self.position.copy()]

		# Dynamic modes
		self.modes = ['stationary', 'random_walk', 'patrol', 'chase', 'evade']
		self.current_mode = 'stationary'
		self.mode_change_probability = 0.02  # Probability to change mode each step
		self.mode_change_counter = 0
		self.mode_duration = random.randint(50, 100)  # Steps before mode reconsideration

		# Movement parameters
		self.max_speed = 8
		self.radius = 25  # Size of the obstacle agent

		# Patrol parameters
		self.patrol_points = [
			np.array([WIDTH // 5, HEIGHT // 5], dtype=np.float64),
			np.array([WIDTH * 3 // 5, HEIGHT // 5], dtype=np.float64),
			np.array([WIDTH * 3 // 5, HEIGHT * 3 // 5], dtype=np.float64),
			np.array([WIDTH // 5, HEIGHT * 3 // 5], dtype=np.float64)
		]
		self.current_patrol_point = 0

		# State variables
		self.velocity = np.zeros(2, dtype=np.float64)
		self.target_position = None

	def update_mode(self, player_position=None):
		self.mode_change_counter += 1

		# Consider changing mode
		if self.mode_change_counter >= self.mode_duration or random.random() < self.mode_change_probability:
			# Reset counter and choose new duration
			self.mode_change_counter = 0
			self.mode_duration = random.randint(50, 100)

			# Choose new mode with weights depending on player position
			if player_position is not None:
				distance_to_player = np.linalg.norm(self.position - player_position)

				# Mode probabilities change based on distance to player
				if distance_to_player < 200:
					# More likely to chase or evade when player is close
					weights = [0.05, 0.15, 0.2, 0.4, 0.2]  # [stationary, random, patrol, chase, evade]
				else:
					# More likely to patrol or be stationary when player is far
					weights = [0.2, 0.2, 0.4, 0.1, 0.1]  # [stationary, random, patrol, chase, evade]

				self.current_mode = random.choices(self.modes, weights=weights)[0]
			else:
				# Without player info, just choose randomly
				self.current_mode = random.choice(self.modes)

	def move(self, player_position=None):
		# Update movement based on current mode
		if self.current_mode == 'stationary':
			self.velocity = np.zeros(2, dtype=np.float64)

		elif self.current_mode == 'random_walk':
			# Add random acceleration
			acceleration = np.array([random.uniform(-1, 1), random.uniform(-1, 1)], dtype=np.float64)
			self.velocity = self.velocity * 0.9 + acceleration

			# Limit speed
			speed = np.linalg.norm(self.velocity)
			if speed > self.max_speed * 0.7:
				self.velocity = self.velocity * (self.max_speed * 0.7 / speed)

		elif self.current_mode == 'patrol':
			# Move toward current patrol point
			target = self.patrol_points[self.current_patrol_point]
			direction = target - self.position
			distance = np.linalg.norm(direction)

			if distance < 20:  # Reached the patrol point
				self.current_patrol_point = (self.current_patrol_point + 1) % len(self.patrol_points)
				target = self.patrol_points[self.current_patrol_point]
				direction = target - self.position
				distance = np.linalg.norm(direction)

			if distance > 0:
				self.velocity = direction / distance * self.max_speed * 0.8
			else:
				self.velocity = np.zeros(2, dtype=np.float64)

		elif self.current_mode == 'chase' and player_position is not None:
			# Chase the player
			direction = player_position - self.position
			distance = np.linalg.norm(direction)

			if distance > 0:
				self.velocity = direction / distance * self.max_speed
			else:
				self.velocity = np.zeros(2, dtype=np.float64)

		elif self.current_mode == 'evade' and player_position is not None:
			# Move away from the player
			direction = self.position - player_position
			distance = np.linalg.norm(direction)

			if distance > 0:
				self.velocity = direction / distance * self.max_speed
			else:
				# If exactly at player position (unlikely), move randomly
				angle = random.uniform(0, 2 * math.pi)
				self.velocity = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * self.max_speed

		# Update position - explicit type conversion to ensure no type mismatch
		self.position = self.position + self.velocity

		# Ensure the agent stays within bounds
		self.position[0] = max(self.radius, min(WIDTH - self.radius, self.position[0]))
		self.position[1] = max(self.radius, min(HEIGHT - self.radius, self.position[1]))

		# Record position history
		self.position_history.append(self.position.copy())

		# Trim history if it gets too long
		if len(self.position_history) > 100:
			self.position_history.pop(0)

	def render(self, screen):
		# Draw position history as a trail
		if len(self.position_history) > 1:
			for i in range(len(self.position_history) - 1):
				alpha = int(255 * (i / len(self.position_history)))
				color = (min(255, DARK_RED[0] + alpha // 3),
				         min(255, DARK_RED[1] + alpha // 3),
				         min(255, DARK_RED[2] + alpha // 3))
				pygame.draw.line(screen, color,
				                 self.position_history[i].astype(int),
				                 self.position_history[i + 1].astype(int), 3)

		# Draw the obstacle agent with different colors based on mode
		if self.current_mode == 'stationary':
			color = DARK_RED
		elif self.current_mode == 'random_walk':
			color = ORANGE
		elif self.current_mode == 'patrol':
			color = YELLOW
		elif self.current_mode == 'chase':
			color = RED
		elif self.current_mode == 'evade':
			color = PURPLE
		else:
			color = DARK_RED

		pygame.draw.circle(screen, color, self.position.astype(int), self.radius)

		# Draw mode text above agent
		mode_text = font_small.render(self.current_mode.upper(), True, BLACK)
		text_pos = (int(self.position[0] - mode_text.get_width() // 2),
		            int(self.position[1] - self.radius - 20))
		screen.blit(mode_text, text_pos)

		# If in patrol mode, draw patrol points and connections
		if self.current_mode == 'patrol':
			for i, point in enumerate(self.patrol_points):
				pygame.draw.circle(screen, YELLOW, point.astype(int), 5)

				# Draw line to next patrol point
				next_idx = (i + 1) % len(self.patrol_points)
				pygame.draw.line(screen, YELLOW,
				                 point.astype(int),
				                 self.patrol_points[next_idx].astype(int), 1)


class DynamicMPC:
	def __init__(self):
		# Algorithm parameters
		self.T = 20  # Time horizon
		self.current_t = 0
		self.epsilon = 0.01  # Confidence bound (probability of missing a mode)
		self.x = np.array([WIDTH // 4, HEIGHT // 2], dtype=np.float64)  # Initial state as float
		self.x_history = [self.x.copy()]
		self.u_history = []
		self.obstacles = []
		self.generate_obstacles(3)  # Reduced static obstacles since we have dynamic obstacle
		self.target = np.array([WIDTH * 3 // 4, HEIGHT // 2], dtype=np.float64)

		# Create dynamic obstacle agent
		self.dynamic_obstacle = DynamicObstacleAgent(np.array([WIDTH // 2, HEIGHT // 2], dtype=np.float64))

		# Control bounds
		self.u_max = 20
		self.scenario_count = 40
		self.scenarios = []
		self.generate_scenarios()

		# Valiant & Valiant support estimation parameters
		self.all_observed_modes = set()
		self.observation_counts = {mode: 0 for mode in self.dynamic_obstacle.modes}
		self.min_observations = 20  # Minimum observations needed before estimation
		self.current_confidence = 0.0
		self.delta = 0.05

		# VV support estimation parameters
		self.vv_particles = []
		self.generate_vv_particles()

		# Support estimation phase
		self.in_estimation_phase = True
		self.target_confidence = 0.95  # Target confidence level

		# Visualization parameters
		self.show_forecast = True
		self.show_constraints = True
		self.show_scenarios = True
		self.show_vv = True
		self.paused = False
		self.step_mode = False
		self.speed = 1
		self.frame_count = 0

		# Mode history
		self.modes = ['normal', 'aggressive', 'conservative', 'evasive']
		self.current_mode = 'normal'
		self.mode_history = [self.current_mode]

		# Generate initial optimization
		self.forecast_trajectories = []
		self.optimal_trajectory = []
		self.violated_stationary_constraints = []
		self.violated_dynamic_constraints = []


		# Dynamic obstacle predictions
		self.obstacle_predictions = []

	def generate_obstacles(self, num_obstacles):
		# Generate random obstacles
		self.obstacles = []
		for _ in range(num_obstacles):
			pos = np.array([
				random.randint(WIDTH // 3, WIDTH * 2 // 3),
				random.randint(HEIGHT // 4, HEIGHT * 3 // 4)
			], dtype=np.float64)
			radius = random.randint(20, 50)
			self.obstacles.append((pos, radius))

	def generate_scenarios(self):
		self.scenarios = []
		for _ in range(self.scenario_count):
			# Create disturbance trajectory for each scenario
			disturbance = []
			for _ in range(self.T):
				# Random disturbance vectors for each time step
				dist = np.array([random.uniform(-10, 10), random.uniform(-10, 10)], dtype=np.float64)
				disturbance.append(dist)
			self.scenarios.append(disturbance)

	def generate_vv_particles(self):
		# Valiant and Valiant support estimation technique visualization
		# Generating particles representing the support estimation
		self.vv_particles = []
		num_particles = 30
		for _ in range(num_particles):
			angle = random.uniform(0, 2 * math.pi)
			distance = random.uniform(5, 15)
			particle = np.array([math.cos(angle) * distance, math.sin(angle) * distance], dtype=np.float64)
			self.vv_particles.append(particle)

	def record_observation(self, mode):
		# Record an observation of a specific mode
		self.all_observed_modes.add(mode)
		self.observation_counts[mode] += 1

		# Update confidence calculation using Valiant & Valiant
		self.update_confidence()

		# Update VV particles visualization based on observations
		self.update_vv_visualization()

	def update_confidence(self):
		n = sum(self.observation_counts.values())
		k = len(self.all_observed_modes)
		m = len(self.dynamic_obstacle.modes)
		delta = self.delta  # confidence level, e.g., 0.05

		f1 = sum(1 for count in self.observation_counts.values() if count == 1)
		if f1 == 0:
			f1 = 0.1  # to avoid divide-by-zero

		missing_mass = f1 / n

		# Conservative upper bound on number of unseen modes (simplified V&V)
		unseen_bound = (n * missing_mass) + math.sqrt(n * math.log(1 / delta))  # simplified

		if k + unseen_bound >= m:
			self.current_confidence = 1.0
		else:
			self.current_confidence = 1.0 - (m - k - unseen_bound) / m

		self.current_confidence = max(0.0, min(1.0, self.current_confidence))

		if self.in_estimation_phase and self.current_confidence >= self.target_confidence:
			self.in_estimation_phase = False
			print(f"Estimation phase complete! Confidence level: {self.current_confidence:.4f}")

	def update_vv_visualization(self):
		# Update the VV particle visualization based on observations
		# Particles are distributed based on observed modes and their frequencies

		# Reset particles
		self.vv_particles = []

		# Create particles based on observation counts
		total_obs = sum(self.observation_counts.values())
		if total_obs == 0:
			return

		for mode, count in self.observation_counts.items():
			if count > 0:
				# Calculate proportion of particles for this mode
				mode_proportion = count / total_obs
				num_particles = max(1, int(30 * mode_proportion))

				# Generate particles for this mode
				for _ in range(num_particles):
					angle = random.uniform(0, 2 * math.pi)

					# Vary distance based on observation frequency
					distance = 15 * (0.5 + mode_proportion)

					# Color-coded by mode (just for visualization)
					particle = np.array([math.cos(angle) * distance, math.sin(angle) * distance], dtype=np.float64)
					self.vv_particles.append(particle)

	def predict_dynamic_obstacle(self):
		# Predict future positions of dynamic obstacle based on observed modes
		predictions = []
		current_pos = self.dynamic_obstacle.position.copy()
		current_vel = self.dynamic_obstacle.velocity.copy()

		# Only use observed modes for prediction
		prediction_modes = ['stationary']  # Always include stationary as a fallback
		for mode in self.all_observed_modes:
			if mode not in prediction_modes:
				prediction_modes.append(mode)

		# If no modes observed yet, use generic predictions
		if len(prediction_modes) <= 1:
			prediction_modes = ['stationary', 'continue', 'towards_player', 'away_from_player']

		for mode in prediction_modes:
			trajectory = [current_pos.copy()]

			# Current velocity is used for 'continue' prediction
			vel = current_vel.copy()

			for _ in range(self.T):
				next_pos = trajectory[-1].copy()

				if mode == 'stationary':
					# No movement
					pass

				elif mode == 'continue' or mode == 'random_walk':
					# Continue with current velocity plus some noise for random walk
					if mode == 'random_walk':
						noise = np.array([random.uniform(-1, 1), random.uniform(-1, 1)], dtype=np.float64)
						vel = vel * 0.9 + noise
						speed = np.linalg.norm(vel)
						if speed > self.dynamic_obstacle.max_speed * 0.7:
							vel = vel * (self.dynamic_obstacle.max_speed * 0.7 / speed)

					next_pos += vel

				elif mode == 'patrol':
					# Simplified patrol prediction
					patrol_center = np.mean(self.dynamic_obstacle.patrol_points, axis=0)
					direction = patrol_center - trajectory[-1]
					distance = np.linalg.norm(direction)
					if distance > 0:
						next_pos += direction / distance * self.dynamic_obstacle.max_speed * 0.5

				elif mode == 'chase' or mode == 'towards_player':
					# Move towards player's current position
					direction = self.x - trajectory[-1]
					distance = np.linalg.norm(direction)
					if distance > 0:
						next_pos += direction / distance * self.dynamic_obstacle.max_speed

				elif mode == 'evade' or mode == 'away_from_player':
					# Move away from player's current position
					direction = trajectory[-1] - self.x
					distance = np.linalg.norm(direction)
					if distance > 0:
						next_pos += direction / distance * self.dynamic_obstacle.max_speed

				# Keep within bounds
				next_pos[0] = max(0, min(WIDTH, next_pos[0]))
				next_pos[1] = max(0, min(HEIGHT, next_pos[1]))

				trajectory.append(next_pos)

			predictions.append(trajectory)

		self.obstacle_predictions = predictions

	def forecast(self):
		# Generate forecast trajectories for each scenario
		self.forecast_trajectories = []
		self.violated_stationary_constraints = []
		self.violated_dynamic_constraints = []

		# Predict dynamic obstacle movement
		self.predict_dynamic_obstacle()

		for scenario in self.scenarios:
			trajectory = [self.x.copy()]
			violated_dyn_consts = []
			violated_stationary_consts = []

			for t in range(self.T):
				# Apply random control and add disturbance
				u = np.array([random.uniform(-self.u_max, self.u_max),
				              random.uniform(-self.u_max, self.u_max)], dtype=np.float64)

				# Bias control toward target
				direction_to_target = self.target - trajectory[-1]
				direction_norm = np.linalg.norm(direction_to_target)
				if direction_norm > 1e-6:  # Avoid division by zero
					direction_to_target = direction_to_target / direction_norm
					bias = direction_to_target * self.u_max * 0.7
					u = u * 0.1 + bias

				# Apply constraints
				u_norm = np.linalg.norm(u)
				if u_norm > self.u_max:
					u = u * self.u_max / u_norm

				# Simple dynamics model: next_state = current_state + control + disturbance
				next_state = trajectory[-1] + u + scenario[t]

				# Check constraints (collision with static obstacles)
				violated_stat = False
				for obs_pos, obs_radius in self.obstacles:
					distance = np.linalg.norm(next_state - obs_pos)
					if distance < obs_radius + 10:  # 10 = agent radius
						violated_stat = True
						break

				violated_dyn = False
				# Check constraints (collision with dynamic obstacle predictions)
				# Sample from different prediction models of the dynamic obstacle
				for prediction_idx in range(len(self.obstacle_predictions)):
					if t < len(self.obstacle_predictions[prediction_idx]) - 1:
						dynamic_obs_pos = self.obstacle_predictions[prediction_idx][t + 1]
						distance = np.linalg.norm(next_state - dynamic_obs_pos)
						if distance < self.dynamic_obstacle.radius + 10:  # 10 = agent radius
							violated_dyn = True
							break

				violated_stationary_consts.append(violated_stat)
				violated_dyn_consts.append(violated_dyn)
				trajectory.append(next_state)

			self.forecast_trajectories.append(trajectory)
			self.violated_stationary_constraints.append(violated_stationary_consts)
			self.violated_dynamic_constraints.append(violated_dyn_consts)

	def solve_optimization(self):
		# Solve the stochastic MPC optimization
		best_cost = float('inf')
		best_trajectory = None

		# Adjust optimization based on current phase
		risk_factor = 2.0 if self.in_estimation_phase else 1.0

		for i, trajectory in enumerate(self.forecast_trajectories):
			# Evaluate cost for this trajectory
			cost = 0

			# In estimation phase, prioritize exploring environment
			if self.in_estimation_phase:
				# Reward staying at a distance to observe obstacle behavior
				optimal_distance = 150  # Good distance to observe from
				for j in range(len(trajectory)):
					# Distance to obstacle
					distance_to_obstacle = np.linalg.norm(trajectory[j] - self.dynamic_obstacle.position)
					# Penalty for being too close or too far
					distance_cost = abs(distance_to_obstacle - optimal_distance) / optimal_distance
					cost += distance_cost

					# Small penalty for distance to target
					target_cost = np.linalg.norm(trajectory[j] - self.target) / WIDTH * 0.1
					cost += target_cost

					# High penalty for constraint violations

					if j < len(self.violated_dynamic_constraints[i]) and self.violated_dynamic_constraints[i][j]:
						cost += 100000 * risk_factor
					if j < len(self.violated_stationary_constraints[i]) and self.violated_stationary_constraints[i][j]:
						cost += 100000
			else:
				# In MPC phase, prioritize reaching target safely
				for j in range(len(trajectory) - 1):
					# Cost: distance to target + control effort
					distance_to_target = np.linalg.norm(trajectory[j] - self.target)
					control_effort = np.linalg.norm(trajectory[j + 1] - trajectory[j])
					step_cost = distance_to_target + 0.3 * control_effort

					# Add penalty for constraint violations
					if j < len(self.violated_dynamic_constraints[i]) and self.violated_dynamic_constraints[i][j]:
						step_cost += 100000
					if j < len(self.violated_stationary_constraints[i]) and self.violated_stationary_constraints[i][j]:
						step_cost += 1000000
					cost += step_cost

			# Check if this trajectory has lower cost
			if cost < best_cost:
				best_cost = cost
				best_trajectory = trajectory

		self.optimal_trajectory = best_trajectory
		return best_trajectory

	def step(self):
		if self.paused:
			return

		# Only update simulation every few frames for visualization purposes
		self.frame_count += 1
		if self.frame_count % (6 - self.speed) != 0:
			return

		self.frame_count = 0

		if self.current_t >= self.T:
			# Reset simulation when done
			self.current_t = 0
			self.x = np.array([WIDTH // 4, HEIGHT // 2], dtype=np.float64)
			self.x_history = [self.x.copy()]
			self.u_history = []
			self.generate_obstacles(3)
			self.generate_scenarios()
			return

		# Update the dynamic obstacle first
		self.dynamic_obstacle.update_mode(self.x)
		self.dynamic_obstacle.move(self.x)

		# Record observation of the current obstacle mode
		self.record_observation(self.dynamic_obstacle.current_mode)

		# Generate forecast trajectories
		self.forecast()

		# Solve MPC optimization problem
		self.solve_optimization()

		if len(self.optimal_trajectory) > 1:
			# Apply the first control action
			u = self.optimal_trajectory[1] - self.optimal_trajectory[0]

			# Add noise to represent stochasticity
			noise = np.array([random.uniform(-5, 5), random.uniform(-5, 5)], dtype=np.float64)

			# Update state
			self.x = self.x + u + noise
			self.x_history.append(self.x.copy())
			self.u_history.append(u)

		self.current_t += 1

	def render(self, screen):
		# Clear screen
		screen.fill(WHITE)

		# Draw grid
		self.draw_grid(screen)

		# Draw obstacles
		for obs_pos, obs_radius in self.obstacles:
			pygame.draw.circle(screen, RED, obs_pos.astype(int), obs_radius)

		# Draw target
		pygame.draw.circle(screen, GREEN, self.target.astype(int), 15)

		# Draw dynamic obstacle predictions
		if self.show_forecast and self.obstacle_predictions:
			colors = [(200, 150, 150), (220, 100, 100), (200, 100, 150), (150, 100, 100)]
			for i, prediction in enumerate(self.obstacle_predictions):
				for j in range(len(prediction) - 1):
					pygame.draw.line(screen, colors[i % len(colors)],
					                 prediction[j].astype(int),
					                 prediction[j + 1].astype(int), 1)

		# Draw VV support estimation visualization
		if self.show_vv:
			# Draw confidence boundary
			confidence_radius = 100 * (1.0 - self.current_confidence)
			pygame.draw.circle(screen, PURPLE, self.x.astype(int), int(confidence_radius), 1)

			# Draw particles representing observed modes
			for particle in self.vv_particles:
				particle_pos = self.x + particle
				pygame.draw.circle(screen, PURPLE, particle_pos.astype(int), 3)

		# Draw forecast trajectories
		if self.show_forecast and self.forecast_trajectories:
			for i, trajectory in enumerate(self.forecast_trajectories):
				# Skip optimal trajectory as it will be drawn differently
				if trajectory is self.optimal_trajectory:
					continue

				color = GRAY
				for j in range(len(trajectory) - 1):
					if self.violated_stationary_constraints and len(self.violated_stationary_constraints[i]) > j and \
							self.violated_stationary_constraints[i][j]:
						pygame.draw.line(screen, RED,
						                 trajectory[j].astype(int),
						                 trajectory[j + 1].astype(int), 1)
					else:
						pygame.draw.line(screen, color,
						                 trajectory[j].astype(int),
						                 trajectory[j + 1].astype(int), 1)

		# Draw optimal trajectory
		if self.optimal_trajectory and len(self.optimal_trajectory) > 1:
			for j in range(len(self.optimal_trajectory) - 1):
				pygame.draw.line(screen, BLUE,
				                 self.optimal_trajectory[j].astype(int),
				                 self.optimal_trajectory[j + 1].astype(int), 2)

		# Draw state history
		if len(self.x_history) > 1:
			for j in range(len(self.x_history) - 1):
				pygame.draw.line(screen, GREEN,
				                 self.x_history[j].astype(int),
				                 self.x_history[j + 1].astype(int), 3)

		# Draw current state
		pygame.draw.circle(screen, BLUE, self.x.astype(int), 10)

		# Draw dynamic obstacle
		self.dynamic_obstacle.render(screen)

		# Draw UI elements
		self.draw_ui(screen)

	def draw_grid(self, screen):
		# Draw a grid for reference
		for x in range(0, WIDTH, 50):
			pygame.draw.line(screen, LIGHT_BLUE, (x, 0), (x, HEIGHT), 1)
		for y in range(0, HEIGHT, 50):
			pygame.draw.line(screen, LIGHT_BLUE, (0, y), (WIDTH, y), 1)

	def draw_ui(self, screen):
		# Draw algorithm information
		title = font_xl.render("Dynamic Sensing Gated Scenario-based Stochastic MPC", True, BLACK)
		screen.blit(title, (20, 20))

		# Phase indicator
		phase_text = font_large.render(
			f"Phase: {'Observation & Estimation' if self.in_estimation_phase else 'MPC Control'}",
			True, BLUE if self.in_estimation_phase else GREEN)
		screen.blit(phase_text, (20, 60))

		# Confidence level
		confidence_text = font_large.render(f"Confidence: {self.current_confidence:.2f} / {self.target_confidence}",
		                                    True, PURPLE)
		screen.blit(confidence_text, (20, 90))

		# Draw dynamic obstacle mode
		obstacle_mode_text = font_large.render(f"Obstacle Mode: {self.dynamic_obstacle.current_mode}", True, DARK_RED)
		screen.blit(obstacle_mode_text, (20, 120))

		# Draw time step
		step_text = font_medium.render(f"Time Step: {self.current_t}/{self.T}", True, BLACK)
		screen.blit(step_text, (20, 150))

		# Draw confidence bound
		epsilon_text = font_medium.render(f"ε-Confidence: {self.epsilon}", True, BLACK)
		screen.blit(epsilon_text, (20, 170))

		# Draw controls
		controls_text = [
			"Controls:",
			"R: Reset Simulation",
			"S: Toggle Scenarios",
			"F: Toggle Forecast",
			"V: Toggle VV Support",
			"Space: Pause/Resume",
			"N: Next Step (when paused)",
			"1-3: Adjust Speed"
		]

		for i, text in enumerate(controls_text):
			control_text = font_small.render(text, True, BLACK)
			screen.blit(control_text, (WIDTH - 200, 20 + i * 25))

		# Draw status indicators
		status_y = HEIGHT - 80

		# Simulation status
		status_text = font_medium.render("Status: " + ("Paused" if self.paused else "Running"), True, BLACK)
		screen.blit(status_text, (20, status_y))

		# Speed indicator
		speed_text = font_medium.render(f"Speed: {self.speed}", True, BLACK)
		screen.blit(speed_text, (20, status_y + 25))

		# Draw the algorithm pseudocode in a box
		self.draw_algorithm_box(screen)

		# Draw dynamic obstacle modes info
		self.draw_obstacle_modes_box(screen)

	def draw_algorithm_box(self, screen):
		# Draw a box with the algorithm pseudocode
		box_x, box_y = WIDTH - 360, HEIGHT - 240
		box_width, box_height = 340, 220

		# Draw box background
		pygame.draw.rect(screen, LIGHT_BLUE, (box_x, box_y, box_width, box_height))
		pygame.draw.rect(screen, BLACK, (box_x, box_y, box_width, box_height), 2)

		# Algorithm pseudocode
		title = font_medium.render("Algorithm Steps:", True, BLACK)
		screen.blit(title, (box_x + 10, box_y + 10))

		steps = [
			"1. Initialize mode history H",
			"2. Form candidate distribution D with VV",
			"3. Update H ← D",
			"4. For t = 0 to T:",
			"   a. Forecast across each mode in H",
			"   b. Solve stochastic optimization",
			"   c. Apply optimal control u_t",
			"   d. Get next state observation"
		]

		# Highlight current step
		current_algo_step = min(4, self.current_t + 3)

		for i, step in enumerate(steps):
			if i == current_algo_step and not self.paused:
				color = BLUE
			else:
				color = BLACK

			step_text = font_small.render(step, True, color)
			screen.blit(step_text, (box_x + 20, box_y + 40 + i * 22))

	def draw_obstacle_modes_box(self, screen):
		# Draw a box with obstacle modes information
		box_x, box_y = 20, HEIGHT - 240
		box_width, box_height = 320, 150

		# Draw box background
		pygame.draw.rect(screen, LIGHT_BLUE, (box_x, box_y, box_width, box_height))
		pygame.draw.rect(screen, BLACK, (box_x, box_y, box_width, box_height), 2)

		# Box title
		title = font_medium.render("Dynamic Obstacle Modes:", True, BLACK)
		screen.blit(title, (box_x + 10, box_y + 10))

		# Obstacle modes description
		modes = [
			"Stationary: Stays in place",
			"Random Walk: Moves randomly",
			"Patrol: Follows patrol points",
			"Chase: Pursues the agent",
			"Evade: Moves away from agent"
		]

		for i, mode in enumerate(modes):
			# Highlight current mode
			if self.dynamic_obstacle.modes[i] == self.dynamic_obstacle.current_mode:
				color = DARK_RED
			else:
				color = BLACK

			mode_text = font_small.render(mode, True, color)
			screen.blit(mode_text, (box_x + 20, box_y + 40 + i * 22))


def main():
	clock = pygame.time.Clock()
	mpc = DynamicMPC()
	running = True
	print("Starting demo")

	while running:
		# Handle events
		for event in pygame.event.get():
			if event.type == QUIT:
				running = False
			elif event.type == KEYDOWN:
				if event.key == K_r:
					# Reset simulation
					mpc = DynamicMPC()
				elif event.key == K_s:
					# Toggle scenario visibility
					mpc.show_scenarios = not mpc.show_scenarios
				elif event.key == K_f:
					# Toggle forecast visibility
					mpc.show_forecast = not mpc.show_forecast
				elif event.key == K_v:
					# Toggle VV support visibility
					mpc.show_vv = not mpc.show_vv
				elif event.key == K_SPACE:
					# Pause/resume simulation
					mpc.paused = not mpc.paused
				elif event.key == K_n and mpc.paused:
					# Step simulation when paused
					mpc.step()
				elif event.key == K_1:
					mpc.speed = 1
				elif event.key == K_2:
					mpc.speed = 2
				elif event.key == K_3:
					mpc.speed = 3

		# Update simulation state
		mpc.step()

		# Render everything to the screen
		mpc.render(screen)

		# Update the display
		pygame.display.flip()

		# Control the frame rate
		clock.tick(60)

	pygame.quit()
	sys.exit()


if __name__ == '__main__':
	main()