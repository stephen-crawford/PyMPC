"""
Safe Horizon Constraint module for scenario-based MPC with support tracking.
"""
import time
import numpy as np
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

from modules.constraints.base_constraint import BaseConstraint
from utils.utils import LOG_DEBUG

class SafeHorizonConstraint(BaseConstraint):
	def __init__(self):
		super().__init__()
		self.name = "safe_horizon_constraint"
		self.num_discs = int(self.get_config_value("num_discs", 1))
		self.robot_radius = float(self.get_config_value("robot.radius", 0.5))
		self.obstacle_radius = float(self.get_config_value("obstacle_radius", 0.1))
		LOG_DEBUG("SafeHorizonConstraint initialized")

	def update(self, state, data):
		return

	def calculate_constraints(self, state, data, stage_idx):
		constraints = []
		# Treat each scenario obstacle at this stage as a circular keep-out
		obstacles = []
		if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles is not None:
			obstacles.extend(list(data.dynamic_obstacles))
		if hasattr(data, 'static_obstacles') and isinstance(data.static_obstacles, list) and stage_idx < len(data.static_obstacles):
			obs_k = data.static_obstacles[stage_idx]
			if obs_k is not None:
				obstacles.append(obs_k)
		for obs in obstacles:
			pos = getattr(obs, 'position', None)
			pred = getattr(obs, 'prediction', None)
			if pred and len(getattr(pred, 'steps', [])) > stage_idx:
				pos = pred.steps[stage_idx].position
			if pos is None:
				continue
			for disc_id in range(self.num_discs):
				disc_offset = 0.0
				if hasattr(data, 'robot_area') and data.robot_area is not None and disc_id < len(data.robot_area):
					disc_offset = float(data.robot_area[disc_id].offset)
				total_radius = self.robot_radius + float(getattr(obs, 'radius', self.obstacle_radius))
				constraints.append({
					"type": "distance",
					"obs_x": float(pos[0]),
					"obs_y": float(pos[1]),
					"total_radius": float(total_radius),
					"disc_offset": disc_offset,
				})
		return constraints

	def lower_bounds(self, state=None, data=None, stage_idx=None):
		# ||p - c|| - R >= 0
		count = 0
		if data is not None and stage_idx is not None:
			if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles:
				count += len(data.dynamic_obstacles) * self.num_discs
			if hasattr(data, 'static_obstacles') and isinstance(data.static_obstacles, list) and stage_idx < len(data.static_obstacles) and data.static_obstacles[stage_idx] is not None:
				count += self.num_discs
		return [0.0] * count

	def upper_bounds(self, state=None, data=None, stage_idx=None):
		count = 0
		if data is not None and stage_idx is not None:
			if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles:
				count += len(data.dynamic_obstacles) * self.num_discs
			if hasattr(data, 'static_obstacles') and isinstance(data.static_obstacles, list) and stage_idx < len(data.static_obstacles) and data.static_obstacles[stage_idx] is not None:
				count += self.num_discs
		return [np.inf] * count
