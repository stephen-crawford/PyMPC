from math import exp
from venv import logger

from utils.const import OBJECTIVE
from utils.utils import read_config_file, profile_scope, LOG_DEBUG, PROFILE_SCOPE

CONFIG = read_config_file()


class Contouring:

 def __init__(self, solver):
  self.module_type = OBJECTIVE
  self.solver = solver
  self.name = "contouring"
  self.controller = (self.module_type, solver, self.name)
  LOG_DEBUG("Initializing contouring module")
  self.get_num_segments = CONFIG["contouring"]["get_num_segments"]

  # boolean configuration options
  self.add_road_constraints = CONFIG["contouring"]["add_road_constraints"]
  self.two_way_road = CONFIG["road"]["two_way"]
  self.dynamic_velocity_reference = CONFIG["contouring"]["dynamic_velocity_reference"]


  self.spline = None
  self.closest_segment = 0
  self.bound_left = None
  self.bound_right = None

  logger.log(10, "Contouring module successfully initialized")


 def update(self, state, data, module_data):


  PROFILE_SCOPE("Contouring update")
  LOG_DEBUG("Updating contouring module")

  # update the closest point
  closest_s = None
  spline = find_closest_point(self, state.getPos(), self.closest_segment, closest_s)

  if (module_data.path.get() == None and spline.get() != None):
    module_data.path = spline

  state.set("spline", closest_s) # We need to initialize the spline state here

  module_data.current_path_segment = closest_segment

  if (_add_road_constraints):
   construct_road_constraints(data, module_data)



 def set_parameters(self, data, module_data, k):

  # Retrieve weights once

  if k == 0:
   contouring_weight = CONFIG["weights"]["contour"]
   lag_weight = CONFIG["weights"]["lag"]

   terminal_angle_weight = CONFIG["weights"]["terminal_angle"]
   terminal_contouring_weight = CONFIG["weights"]["terminal_contouring"]

   if self._dynamic_velocity_reference:
    reference_velocity = CONFIG["weights"]["reference_velocity"]
    velocity_weight = CONFIG["weights"]["velocity"]
   else:
    velocity_weight = None

  
   set_solver_parameter_contour(k, self._solver._params, contouring_weight)
   set_solver_parameter_lag(k, self._solver._params, lag_weight)

   set_solver_parameter_terminal_angle(k, self._solver._params, terminal_angle_weight)
   set_solver_parameter_terminal_contouring(k, self._solver._params, terminal_contouring_weight)

   if (_dynamic_velocity_reference):
    set_solver_parameter_velocity(k, self._solver._params, velocity_weight)
    set_solver_parameter_reference_velocity(k, self._solver._params, reference_velocity)

  setspline_parameters(k)

 def setspline_parameters(self, k):

  for i in range(self.get_num_segments):

   index = closest_segment + i

   spline.get_parameters(index,
               ax, bx, cx, dx,
               ay, by, cy, dy)

   start = spline.get_segment_start(index)


   set_solver_parameterspline_xa(k, self._solver._params, ax, i)
   set_solver_parameterspline_xb(k, self._solver._params, bx, i)
   set_solver_parameterspline_xc(k, self._solver._params, cx, i)
   set_solver_parameterspline_xd(k, self._solver._params, dx, i)

   set_solver_parameterspline_ya(k, self._solver._params, ay, i)
   set_solver_parameterspline_yb(k, self._solver._params, by, i)
   set_solver_parameterspline_yc(k, self._solver._params, cy, i)
   set_solver_parameterspline_yd(k, self._solver._params, dy, i)

   # Distance where this spline starts
   set_solver_parameterspline_start(k, self._solver._params, start, i)


 def on_data_received(self, data, data_name):
  if data_name == "reference_path":

   logger.log(10, "Received Reference Path")

   # Construct a spline from the given points
   if (data.reference_path.s.empty()):
    spline = make_shared(RosTools.Spline2D(data.reference_path.x, data.reference_path.y))
   else:
    spline = make_shared(RosTools.Spline2D(data.reference_path.x, data.reference_path.y, data.reference_path.s))

   if self._add_road_constraints and (not data.left_bound.empty() and not data.right_bound.empty()):

    # Add bounds
    bound_left = make_unique(RosTools.Spline2D(
      data.left_bound.x,
      data.left_bound.y,
      spline.getTVector()))
    bound_right = make_unique(RosTools.Spline2D>(
      data.right_bound.x,
      data.right_bound.y,
      spline.getTVector()))

    # update the road width
    CONFIG["road"]["width"] = RosTools.distance(bound_left.get_point(0), bound_right.get_point(0))

   closest_segment = -1

 def is_data_ready(self, data, missing_data):

  if (data.reference_path.x.empty()):
   missing_data += "Reference Path "

  return data.reference_path.x.empty()

 def is_objective_reached(self, state, data):

  if not spline:
   return False

  # Check if we reached the end of the spline
  return RosTools.distance(self, state.getPos(), spline.get_point(spline.parameter_length())) < 1.0

 def construct_road_constraints(self, data, module_data):

  logger.log(10, "Constructing road constraints.")

  if data.left_bound.empty() or data.right_bound.empty():
   construct_road_constraints_from_centerline(data, module_data)
  else:
   construct_road_constraints_from_bounds(data, module_data)


 def construct_road_constraints_from_centerline(self, data, module_data):

  # If bounds are not supplied construct road constraints based on a set width
  if module_data.static_obstacles.empty():
   module_data.static_obstacles.resize(_solver.N)
   for k in range(module_data.static_obstacles.size()):
    module_data.static_obstacles[k].reserve(2)

  # OLD VERSION:
  road_width_half = CONFIG["road"]["width"] / 2.
  for k in range(_solver.N):

   module_data.static_obstacles[k].clear()

   cur_s = _solver.get_ego_prediction(k, "spline")

   # This is the final point and the normal vector of the path
   vector_2d path_point = spline.get_point(cur_s)
   vector_2d dpath = spline.get_orthogonal(cur_s)

   # left HALFSPACE
   A = spline.get_orthogonal(cur_s)
   if self._two_way_road:
    width_times = 3.0
   else:
    width_times = 1.0

   # line is parallel to the spline
   boundary_left = path_point + dpath * (width_times * road_width_half - data.robot_area[0].radius)

   b = A.transpose() * boundary_left

   module_data.static_obstacles[k].emplace_back(A, b)

   # right HALFSPACE
   A = spline.get_orthogonal(cur_s) # Vector2d(-path_dy, path_dx) # line is parallel to the spline

   boundary_right = path_point - dpath * (road_width_half - data.robot_area[0].radius)
   b = A.transpose() * boundary_right # And lies on the boundary point

   module_data.static_obstacles[k].emplace_back(-A, -b)

 def construct_road_constraints_from_bounds(self, data, module_data):

  if module_data.static_obstacles.empty():
   module_data.static_obstacles.resize(_solver.N)
   for k in range(module_data.static_obstacles.size()):
    module_data.static_obstacles[k].reserve(2)

  for k in range(_solver.N):
   module_data.static_obstacles[k].clear()
   cur_s = _solver.get_ego_prediction(k, "spline")

   # left
   Al = bound_left.get_orthogonal(cur_s)
   bl = Al.transpose() * (bound_left.get_point(cur_s) + Al * data.robot_area[0].radius)
   module_data.static_obstacles[k].emplace_back(-Al, -bl)

   # right HALFSPACE
   Ar = bound_right.get_orthogonal(cur_s)
   br = Ar.transpose() * (bound_right.get_point(cur_s) - Ar * data.robot_area[0].radius)
   module_data.static_obstacles[k].emplace_back(Ar, br)

 def visualize(self, data, module_data):

  if spline.get() == None:
   return

  if data.reference_path.empty():
   return

  logger.log(10, "Contouring::Visualize")

  visualize_reference_path(data, module_data)
  visualize_road_constraints(data, module_data)

  if (CONFIG["debug_visuals"]):
   visualize_current_segment(data, module_data)
   visualize_debug_road_boundary(data, module_data)
   visualize_debug_gluedsplines(data, module_data)
   visualize_allspline_indices(data, module_data)
   visualize_tracked_section(data, module_data)


 def visualize_current_segment(self, data, module_data):

  # Visualize the current points
  publisher_current = VISUALS.missing_data(_name + "/current")
  cur_point = publisher_current.get_new_point_marker("CUBE")
  cur_point.set_color_int(10)
  cur_point.set_scale(0.3, 0.3, 0.3)
  cur_point.add_point_marker(spline.get_point(spline.get_segment_start(closest_segment)), 0.0)
  publisher_current.publish()

 def visualize_tracked_section(self, data, module_data):

  # Visualize the current points
  publisher = VISUALS.missing_data(_name + "/tracked_path")
  line = publisher.get_new_line()

  line.set_solor_nt(10)
  line.set_scale(0.3, 0.3, 0.3)

  i = closest_segment
  while i < closest_segment + _n_segments:

   s_start = spline.get_segment_start(i)
   s = s_start + 1.0
   while s < spline.get_segment_start(i + 1):

    if (s > 0):
     line.add_line(spline.get_point(s - 1.0), spline.get_point(s))
    s += 1.0

   i+=1

  publisher.publish()

 # Move to data visualization
 def visualize_reference_path(self, data, module_data):

  visualize_path_points(data.reference_path, _name + "/path", False)
  visualizespline(*spline, _name + "/path", True)

 def visualize_road_constraints(self, data, module_data):

  if module_data.static_obstacles.empty() or not self._add_road_constraints:
   return

  for k in range(_solver.N):

   for h in range(module_data.static_obstacles[k].size()):
    visualize_linear_constraint(module_data.static_obstacles[k][h], k, _solver.N, _name + "/road_boundary_constraints", _solver.N - 1, h == module_data.static_obstacles[k].size() - 1, 0.5, 0.1)


 def visualize_debug_road_boundary(self, data, module_data):

  publisher = VISUALS.missing_data(_name + "/road_boundary_points")
  points = publisher.get_new_point_marker("CUBE")
  points.set_scale(0.15, 0.15, 0.15)

  # OLD VERSION:
  two_way = self._two_way_road
  road_width_half = CONFIG["road"]["width"] / 2.
  for k in range(_solver.N):
   cur_s = _solver.get_ego_prediction(k, "spline")
   path_point = spline.get_point(cur_s)

   points.set_color_int(5, 10)
   points.add_point_marker(path_point)

   dpath = spline.get_orthogonal(cur_s)

   if two_way:
    width_times = 3.0 # 3w for lane
   else:
    width_times = 1.0

   # line is parallel to the spline
   boundary_left = path_point + dpath * (width_times * road_width_half - data.robot_area[0].radius)

   boundary_right = path_point - dpath * (road_width_half - data.robot_area[0].radius)

   points.set_color(0., 0., 0.)
   points.add_point_marker(boundary_left)
   points.add_point_marker(boundary_right)

  publisher.publish()

 def visualize_debug_gluedsplines(self, data,module_data):

  # Plot how the optimization joins the splines together to debug its internal contouring error computation
  publisher = VISUALS.missing_data(_name + "/gluedspline_points")
  points = publisher.get_new_point_marker("CUBE")
  points.set_scale(0.15, 0.15, 0.15)

  for k in range(_solver.N):

   for i in range (self._n_segments):
    index = closest_segment + i


    if (index < spline.get_num_segments()):
     spline.get_parameters(index, ax, bx, cx, dx, ay, by, cy, dy)

     start = spline.get_segment_start(index)
    else:
     # If we are beyond the spline, we should use the last spline
     spline.get_parameters(spline.get_num_segments() - 1, ax, bx, cx, dx, ay, by, cy, dy)

     start = spline.get_segment_start(spline.get_num_segments() - 1)

     # We should use very small splines at the end location
     # x = d_x
     # y = d_y
     ax = 0.
     bx = 0.
     cx = 0.
     ay = 0.
     by = 0.
     cy = 0.
     start = spline.parameter_length()

    s = _solver.get_ego_prediction(k, "spline") - start
    path_x.push_back(ax * s * s * s + bx * s * s + cx * s + dx)
    path_y.push_back(ay * s * s * s + by * s * s + cy * s + dy)

    # No lambda for the first segment (it is not glued to anything prior)
    if (i > 0):
     lambdas.push_back(1.0 / (1.0 + exp((s + 0.02) / 0.1))) # Sigmoid


   cur_path_x = path_x.back()
   cur_path_y = path_y.back()
   p = path_x.size() - 1
   while p > 0:
    # Glue with the previous path
    cur_path_x = lambdas[p - 1] * path_x[p - 1] + (1.0 - lambdas[p - 1]) * cur_path_x
    cur_path_y = lambdas[p - 1] * path_y[p - 1] + (1.0 - lambdas[p - 1]) * cur_path_y
    p -= 1

   points.add_point_marker(Vector2d(cur_path_x, cur_path_y))
  publisher.publish()


 def visualize_allspline_indices(self, data, module_data):

  # Plot how the optimization joins the splines together to debug its internal contouring error computation
  publisher = VISUALS.missing_data(_name + "/spline_variables")
  points = publisher.get_new_point_marker("CUBE")
  points.set_scale(0.15, 0.15, 0.15)

  for k in range(_solver.N):

   cur_s = _solver.get_ego_prediction(k, "spline")
   path_point = spline.get_point(cur_s)
   points.add_point_marker(path_point)

  publisher.publish()

 def reset(self):
  spline.reset()
  closest_segment = 0
