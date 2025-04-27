import numpy as np
import casadi as cd

from utils.const import CONSTRAINT, OBJECTIVE


def define_parameters(module_manager, params, settings):

  # Define parameters for objectives and constraints (in order)
  for module in module_manager.modules:
    if module.module_type == OBJECTIVE:
      module.define_parameters(params)

  for module in module_manager.modules:
    if module.module_type == CONSTRAINT:
      module.define_parameters(params)

  return params


def objective(module_manager, z, p, model, settings, stage_idx):
  cost = 0.0

  params = settings["params"]
  params.load(p)
  model.load(z)

  for module in module_manager.modules:
    if module.type == OBJECTIVE:
      cost += module.get_value(model, params, settings, stage_idx)

  # if stage_idx == 0:
  # print(cost)

  return cost


# lb <= constraints <= ub
def constraints(module_manager, z, p, model, settings, stage_idx):
  constraints = []

  params = settings["params"]
  params.load(p)
  model.load(z)

  for module in module_manager.modules:
    if module.type == CONSTRAINT:
      for constraint in module.constraints:
        constraints += constraint.get_constraints(model, params, settings, stage_idx)

  return constraints


def constraint_upper_bounds(module_manager):
  ub = []
  for module in module_manager.modules:
    if module.type == CONSTRAINT:
      for constraint in module.constraints:
        ub += constraint.get_upper_bound()
  return ub


def constraint_lower_bounds(module_manager):
  lb = []
  for module in module_manager.modules:
    if module.module_type == CONSTRAINT:
      for constraint in module.constraints:
        lb += constraint.get_lower_bound()
  return lb


def constraint_number(module_manager):
  nh = 0
  for module in module_manager.modules:
    if module.module_type == CONSTRAINT:
      for constraint in module.constraints:
        nh += constraint.nh
  return nh