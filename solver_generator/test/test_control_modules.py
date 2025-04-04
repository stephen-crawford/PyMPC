import sys, os

sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(sys.path[0], "..", "..", "mpc_planner_modules", "scripts"))

import numpy as np

from util.parameters import Parameters
from solver_model import ContouringSecondOrderUnicycleModel

from control_modules import ModuleManager, ObjectiveModule, ConstraintModule
from solver_definition import define_parameters, objective, constraints, constraint_lower_bounds, constraint_upper_bounds, constraint_number

from contouring import ContouringModule
from path_reference_velocity import PathReferenceVelocityModule
from ellipsoid_constraints import EllipsoidConstraintModule

from mpc_base import MPCBaseModule
from curvature_aware_contouring import CurvatureAwareContouringModule
from goal_module import GoalModule
from gaussian_constraints import GaussianConstraintModule
from guidance_constraints import GuidanceConstraintModule
from linearized_constraints import LinearizedConstraintModule
from scenario_constraints import ScenarioConstraintModule


def test_module_manager_objective():

  settings = dict()
  settings["contouring"] = dict()
  settings["contouring"]["get_num_segments"] = 10
  settings["contouring"]["dynamic_velocity_reference"] = False
  settings["N"] = 20

  modules = ModuleManager()
  modules.add_module(
    ContouringModule(settings)
  ) # Adds weights to the overall weight list

  assert len(modules.modules) == 1
  assert modules.modules[0].module_name == "Contouring"

  modules.add_module(PathReferenceVelocityModule(settings))

  assert len(modules.modules) == 2

  params = Parameters()
  params = define_parameters(modules, params, settings)
  settings["params"] = params

  model = ContouringSecondOrderUnicycleModel()

  npar = 10 * 9 + 2 + 4 + 10 * 4
  assert params.length() == npar

  z = np.zeros((model.get_nvar()))
  z[3] = 5.0
  p = np.ones((npar))
  obj = objective(modules, z, p, model, settings, 0)

  params.load(p)
  model.load(z)
  obj2 = modules.modules[0].get_value(model, params, settings, 0)
  assert obj == obj2

  assert obj > 0


def test_module_manager_constraints():
  settings = dict()
  settings["n_discs"] = 1
  settings["max_obstacles"] = 1


  modules = ModuleManager()
  modules.add_module(EllipsoidConstraintModule(settings))

  assert len(modules.modules) == 1

  params = Parameters()
  params = define_parameters(modules, params, settings)
  settings["params"] = params

  model = ContouringSecondOrderUnicycleModel()

  npar = 7 + 2
  assert params.length() == npar

  z = np.zeros((model.get_nvar()))
  p = np.zeros((npar))
  p[2] = 5.0
  p[3] = 10.0
  p[-1] = 1.0

  constr = constraints(modules, z, p, model, settings, 0)
  lb = constraint_lower_bounds(modules)
  ub = constraint_upper_bounds(modules)
  num = constraint_number(modules)

  assert num == len(lb) == len(ub) == len(constr)
  for i, c in enumerate(constr):
    assert lb[i] < c
    assert c < ub[i]


def test_all_modules():
  settings = dict()
  settings["n_discs"] = 1
  settings["max_obstacles"] = 1
  settings["contouring"] = dict()
  settings["contouring"]["get_num_segments"] = 10
  settings["contouring"]["dynamic_velocity_reference"] = False

  settings["linearized_constraints"] = dict()
  settings["linearized_constraints"]["add_halfspaces"] = 2
  # settings["contouring"]["dynamic_velocity_reference"] = False
  settings["N"] = 20

  modules = ModuleManager()
  base_module = modules.add_module(MPCBaseModule(settings))
  base_module.weigh_variable(var_name="a", weight_names="acceleration")
  base_module.weigh_variable(var_name="w", weight_names="angular_velocity") 
  base_module.weigh_variable(var_name="v",  
              weight_names=["velocity", "reference_velocity"], 
              cost_function=lambda x, w: w[0] * (x-w[1])**2) 

  modules.add_module(GoalModule(settings))
  modules.add_module(ContouringModule(settings))
  modules.add_module(CurvatureAwareContouringModule(settings))
  modules.add_module(PathReferenceVelocityModule(settings))

  modules.add_module(GaussianConstraintModule(settings))
  modules.add_module(EllipsoidConstraintModule(settings))
  modules.add_module(GuidanceConstraintModule(settings))
  modules.add_module(LinearizedConstraintModule(settings))
  modules.add_module(ScenarioConstraintModule(settings))
