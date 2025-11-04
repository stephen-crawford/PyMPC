import numpy as np

from modules.constraints.base_constraint import BaseConstraint
from utils.utils import LOG_DEBUG


class GuidanceConstraints(BaseConstraint):
    def __init__(self):
        super().__init__()
        self.name = "guidance_constraints"
        LOG_DEBUG("GuidanceConstraints initialized")

    def update(self, state, data):
            return

    def calculate_constraints(self, state, data, stage_idx):
        return []

    def lower_bounds(self):
        return []

    def upper_bounds(self):
        return []