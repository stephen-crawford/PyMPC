import logging

import numpy as np

from modules.constraints.base_constraint import BaseConstraint
from utils.utils import LOG_DEBUG

class DecompConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)
		self.name = "decomp_constraints"
		self.num_discs = int(self.get_config_value("num_discs", 1))
		self.robot_radius = float(self.get_config_value("robot.radius", 0.5))
		self.obstacle_radius = float(self.get_config_value("obstacle_radius", 0.1))
		LOG_DEBUG("DecompConstraints initialized")

	def update(self, state, data):
		return

	def calculate_constraints(self, state, data, stage_idx):
		constraints = []
		if not hasattr(data, 'dynamic_obstacles') or data.dynamic_obstacles is None:
			return constraints
		for obs in data.dynamic_obstacles:
			pos = getattr(obs, 'position', None)
			pred = getattr(obs, 'prediction', None)
			if pred and len(pred.steps) > stage_idx:
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
		count = 0
		if data is not None and stage_idx is not None and hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles:
			for _ in data.dynamic_obstacles:
				count += self.num_discs
		return [0.0] * count

	def upper_bounds(self, state=None, data=None, stage_idx=None):
		count = 0
		if data is not None and stage_idx is not None and hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles:
			for _ in data.dynamic_obstacles:
				count += self.num_discs
		return [np.inf] * count

