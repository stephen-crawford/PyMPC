from util.logging import print_value, print_header


class ModuleManager:
  """
  Modules can bundle objectives and constraints
  In addition, they are directly linked to the c++ code module
  """

  def __init__(self):
    self.modules = []

  def add_module(self, module):
    self.modules.append(module)
    return module

  def inequality_constraints(self, z, param, settings, model):
    return self.constraint_manager.inequality(z, param, settings, model)

  def number_of_constraints(self):
    return self.constraint_manager.nh

  def get_last_added_module(self):
    return self.modules[-1]

  def __str__(self):
    result = "--- MPC Modules ---\n"
    for module in self.modules:
      result += str(module) + "\n"
    return result

  def print(self):
    print_header("MPC Modules")
    for module in self.modules:
      print_value(module.module_name, str(module), tab=True)


class Module:

  def __init__(self):
    self.module_name = "UNDEFINED"
    self.description = ""

    self.submodules = []
    self.dependencies = []
    self.sources = []

  def write_to_solver_interface(self, header_file):
    return

  def __str__(self):
    result = self.description
    return result

  def add_definitions(self, header_file):
    pass

  # def add_submodule(self, submodule):

  #   if not hasattr(self, "submodules"):
  #     self.submodules = []

  #   self.submodules.append(submodule)
  #   if hasattr(submodule, "constraints"):
  #     self.constraints += submodule.constraints

  #   if hasattr(submodule, "objectives"):
  #     self.objectives += submodule.objectives

  #   self.dependencies += submodule.dependencies # Add submodule dependencies
  #   self.sources += submodule.sources
  #   self.sources.append(str("src/" + submodule.import_name.replace(".h", ".cpp"))) # Add the base file of the submodule


class ConstraintModule(Module):

  def __init__(self):
    super(ConstraintModule, self).__init__()
    self.type = "constraint"

    self.constraints = []

  def define_parameters(self, params):
    for constraint in self.constraints:
      constraint.define_parameters(params)


class ObjectiveModule(Module):

  def __init__(self):
    super(ObjectiveModule, self).__init__()
    self.type = "objective"
    self.objectives = []

  def define_parameters(self, params):
    for objective in self.objectives:
      objective.define_parameters(params)

  def get_value(self, model, params, settings, stage_idx):
    cost = 0.0
    for objective in self.objectives:
      cost += objective.get_value(model, params, settings, stage_idx)

    return cost


class Objective:

  def __init__(self) -> None:
    pass

  def define_parameters(self, params):
    raise IOError("Objective did not specify parameters")

  def get_value(self, model, params, settings, stage_idx) -> float:
    raise IOError("Objective did not return a cost")


""" OBJECTIVE MODULES """

# class PreviewContouringModule(Module):

#   """
#   Track a reference path with contouring control
#   """

#   def __init__(self, params, weight_list, n_segments, T):
#     super().__init__()
#     self.module_name = "PreviewContouring" # Needs to correspond to the c++ name of the module
#     # self.import_name = "modules_objectives/reference_path.h"
#     self.depends = [ContouringModule]
#     self.type = "objective"
#     self.description = "Terminal cost for tracking a 2D reference path after the horizon"

#     self.n_segments = n_segments
#     self.T = T # How much seconds ahead?

#     self.objectives = []
#     self.objectives.append(objective.PreviewObjective(params, weight_list, self.n_segments, T))

# class HomotopyGuidanceObjectiveModule(Module):

#   """
#   Homotopic path search in the state space for generating guidance trajectories
#   """

#   def __init__(self, params, weight_list, n_segments, n_discs, constraint_submodule):
#     super().__init__()

#     if constraint_submodule is None:
#       constraint_submodule = EllipsoidalConstraintModule

#     self.module_name = "GuidanceObjective" # Needs to correspond to the c++ name of the module
#     self.import_name = "modules_objectives/guidance_objective.h"
#     self.type = "objective"

#     self.n_segments = n_segments

#     self.objectives = [] # No explicit objective for now
#     self.objectives.append(objective.ContouringObjective(params, weight_list, n_segments)) # We have a contouring objective for tracking the path
#     self.objectives.append(objective.ReferenceVelocityObjective(params, weight_list)) # Velocity tracking for trajectory tracking

#     self.constraints = []
#     self.add_submodule(constraint_submodule(params, n_discs))

#     self.description = "Tracks multiple guidance trajectories in parallel (with multiple MPC)\n" +\
#       "\t\t- Underlying Constraint: " + self.submodules[0].description

#   def write_to_solver_interface(self, header_file):
#     assert len(self.submodules) == 1, "GuidanceObjective can use one type of submodule for collision avoidance constraints"
#     header_file.write("#define GUIDANCE_CONSTRAINTS_TYPE " + self.submodules[0].module_name + "\n")

# """ CONSTRAINT MODULES """
# # For now keep this as parameters in the settings itself
# # class CollisionRegionModule(Module):
# #
# #   def __init__(self, params, n_discs):
# #     params.add_parameter("disc_r")
# #     params.add_multiple_parameters("disc_self.offset", n_discs)


# class ScenarioConstraintModule(Module):

#   """
#   Linear constraints for scenario-based motion planning
#   """

#   def __init__(self, params, n_discs, use_slack=False):
#     super().__init__()
#     self.module_name = "ScenarioConstraints" # Needs to correspond to the c++ name of the module
#     self.import_name = "modules_constraints/scenario_constraints.h"
#     self.dependencies.append("lmpcc_scenario_module")
#     self.type = "constraint"
#     self.description = "Avoid dynamic obstacles under motion uncertainty using scenario optimization."

#     self.constraints = []
#     self.constraints.append(inequality.LinearConstraints(params, n_discs, 24, use_slack, "scenario_constraint"))


# class LinearizedEllipsoidalConstraintModule(Module):

#   """
#   Linearized Ellipsoidal Constraints for collision avoidance
#   Data: Obstacle position, obstacle and vehicle radius
#   """

#   def __init__(self, params, n_discs, max_obstacles):
#     super().__init__()
#     self.module_name = "EllipsoidalConstraints" # Needs to correspond to the c++ name of the module
#     self.import_name = "modules_constraints/ellipsoidal_constraints.h"
#     self.type = "constraint"
#     self.description = "Avoid dynamic obstacles described as ellipsoids, but where the constraints are linearized."

#     self.constraints = []
#     self.constraints.append(inequality.EllipsoidLinearizedConstraints(n_discs, max_obstacles, params))


# class LinearizedConstraintsModule(Module):
#   """
#   Linear constraints for scenario-based motion planning
#   Data: A, b in A^Tx <= b
#   """

#   def __init__(self, params, n_discs, max_obstacles, static_obstacles):
#     self.module_name = "LinearizedConstraints" # Needs to correspond to the c++ name of the module
#     self.import_name = "modules_constraints/linearized_constraints.h"
#     self.type = "constraint"
#     self.description = "Avoid dynamic obstacles by computing a linear constraint for each obstacle that divides them from the robot."

#     self.constraints = []
#     self.constraints.append(inequality.LinearConstraints(params, n_discs, max_obstacles + static_obstacles))


# class BoundaryYModule(Module):
#   """
#   Linear constraints for scenario-based motion planning
#   Data: A, b in A^Tx <= b
#   """

#   def __init__(self, params, n_discs, width):
#     super().__init__()
#     self.module_name = "BoundaryYModule" # Needs to correspond to the c++ name of the module
#     # self.import_name = "modules_constraints/linearized_constraints.h"
#     self.type = "constraint"
#     self.description = "Simple road boundaries in the y-direction"

#     self.constraints = []
#     self.constraints.append(inequality.LinearConstraintsY(n_discs, -width/2., width/2.))

# class RoomBoundaryModule(Module):
#   """
#   Linear constraints for scenario-based motion planning
#   Data: A, b in A^Tx <= b
#   """

#   def __init__(self, params, n_discs, x_size, y_size, x_self.offset=0., y_self.offset=0.):
#     super().__init__()
#     self.module_name = "RoomBoundary" # Needs to correspond to the c++ name of the module
#     # self.import_name = "modules_constraints/linearized_constraints.h"
#     self.type = "constraint"
#     self.description = "Simple road boundaries for a rectanglular room."

#     self.constraints = []
#     self.constraints.append(inequality.RoomConstraints(n_discs, -x_size/2., x_size/2., -y_size/2., y_size/2., x_self.offset, y_self.offset))
