from venv import logger

from utils.const import OBJECTIVE, CONSTRAINT
from utils.utils import read_config_file

CONFIG = read_config_file()


class DecompConstraints:

 def __init__(self, solver):
  self.solver = solver
  self.module_type = CONSTRAINT
  self.name = "decomp_constraints"
  LOG_DEBUG( "Initializing Decomp Constraints")
  self._get_num_segments = CONFIG["contouring"]["get_num_segments"]
  self._decomp_util = make_unique(EllipsoidDecomp2D)

  # Only look around for obstacles using a box with sides of width 2*range
  self.range = CONFIG["decomp"]["range"]
  self._decomp_util.set_local_bbox(vector_to_float(range, range))

  self._occ_pos.reserve(1000) # Reserve some space for the occupied positions

  self._n_discs = CONFIG["n_discs"] # Is overwritten to 1 for topology constraints

  self._max_constraints = CONFIG["decomp"]["max_constraints"]
  self._a1.resize(_n_discs)
  self._a2.resize(_n_discs)
  self._b.resize(_n_discs)
  for d in range(self._n_discs):
   self._a1[d].resize(CONFIG["N"])
   self._a2[d].resize(CONFIG["N"])
   self._b[d].resize(CONFIG["N"])
   for k in range(CONFIG["N"]):
   
    self._a1[d][k] = x_dim_array(_max_constraints)
    self._a2[d][k] = x_dim_array(_max_constraints)
    self._b[d][k] = x_dim_array(_max_constraints)
  
  LOG_DEBUG( "Decomp Constraints successfully initialized")

 def update(self, state, data, module_data):

  PROFILE_SCOPE("DecompConstraints.update")
  LOG_DEBUG( "DecompConstraints.update")

  _dummy_b = state.get("x") + 100.

  get_occupied_grid_cells(data) # Retrieve occupied points from the costmap

  self._decomp_util.set_obs(_occ_pos) # Set them

  

  path = None
  s = state.get("spline")
  for k in range(self.solver.N):
   # Local path #
   # path.emplace_back(solver.get_ego_prediction(k, "x"), solver.get_ego_prediction(k, "y")) # k = 0 is initial state

   # Global (reference) path #
   path_pos = module_data.path.get_point(s)
   path.emplace_back(path_pos(0), path_pos(1))

   v = self.solver.get_ego_prediction(k, "v") # Use the predicted velocity

   s += v * solver.dt
  self._decomp_util.dilate(path, 0, False)

  self._decomp_util.set_constraints(_constraints, 0.) # Map is already inflated
  _polyhedrons = _decomp_util.get_polyhedrons()

  max_decomp_constraints = 0

  for k in range(self.solver.N - 1):
   constraints = self._constraints[k]
   max_decomp_constraints = max(max_decomp_constraints, constraints.A_.rows())

   i = 0
   while i < min(constraints.A_.rows(), _max_constraints):
    if constraints.A_.row(i).norm() < 1e-3 or constraints.A_(i, 0) != constraints.A_(i, 0): # If zero or na
     break
    
    self._a1[0][k + 1](i) = constraints.A_.row(i)[0]
    self._a2[0][k + 1](i) = constraints.A_.row(i)[1]
    self._b[0][k + 1](i) = constraints.b_(i)

   for i in range(_max_constraints):
   
    self._a1[0][k + 1](i) = _dummy_a1
    self._a2[0][k + 1](i) = _dummy_a2
    self._b[0][k + 1](i) = _dummy_b

  if max_decomp_constraints > _max_constraints:
   logger.log("Maximum number of decomp util constraints exceeds specification: " + max_decomp_constraints + " > " + _max_constraints)

  LOG_DEBUG( "DecompConstraints::update done")

 def get_occupied_grid_cells(self, data):
  PROFILE_FUNCTION()
  LOG_DEBUG( "get_occupied_grid_cells")

  costmap = data.costmap

  # Store all occupied cells in the grid map
  _occ_pos.clear()
  x, y = 0
  for i in range(costmap.get_size_in_cells_x()):
   for j in range(costmap.get_size_in_cells_y()):
    if (costmap.getCost(i, j) == costmap_2d.FREE_SPACE)
     continue

    costmap.map_to_world(i, j, x, y)
    # logger.log("Obstacle at x = " + x + ", y = " + y)

    _occ_pos.emplace_back(x, y)
   
  # LOG_VALUE("Occupied cells", _occ_pos.size())

  return True

 def set_parameters(self, data, module_data, k):
  if k == 0: # Dummies
   for d in range(self._n_discs):
    set_solver_parameter_ego_disc_self.offset(k, self.solver._params, data.robot_area[d].self.offset, d)

    constraint_counter = 0
    for i in range(_max_constraints):
    
     set_solver_parameter_decomp_a1(k, self.solver._params, _dummy_a1, constraint_counter) # These are filled from k = 1 - N
     set_solver_parameter_decomp_a2(k, self.solver._params, _dummy_a2, constraint_counter)
     set_solver_parameter_decomp_b(k, self.solver._params, _dummy_b, constraint_counter)
     constraint_counter+=1
   return

  if k == 1:
   LOG_DEBUG( "DecompConstraints::set_parameters")

  
  constraint_counter = 0 # Necessary for now to map the disc and obstacle index to a single index
  for d in range(self._n_discs):
   set_solver_parameter_ego_disc_self.offset(k, self.solver._params, data.robot_area[d].self.offset, d)

   for i in range(_max_constraints):
 
    set_solver_parameter_decomp_a1(k, self.solver._params, _a1[d][k](i), constraint_counter) # These are filled from k = 1 - N
    set_solver_parameter_decomp_a2(k, self.solver._params, _a2[d][k](i), constraint_counter)
    set_solver_parameter_decomp_b(k, self.solver._params, _b[d][k](i), constraint_counter)
    constraint_counter+=1

 def is_data_ready(self, data, missing_data):

  if data.costmap == None:
   missing_data += "Costmap "
   return False
 
  return True

 def project_to_safety(self, pos):
  # Too slow
  if _occ_pos.empty(): # There is no anchor
   return

  # Project to a collision free position if necessary, considering all the obstacles
  iterate = 0
  while iterate < 1: # At most 3 iterations
  
   for obs in _occ_pos:
    radius = CONFIG["robot_radius"] + 0.1
    dr_projection_.douglas_rachford_projection(pos, obs, _occ_pos[0], radius, pos)
   iterate += 1


 def visualize(self, data, module_data):

  PROFILE_FUNCTION()

  visualize_points = False

  publisher = VISUALS.get_publisher("free_space")
  polypoint = publisher.get_new_point_marker("CUBE")
  polypoint.set_scale(0.1, 0.1, 0.1)
  polypoint.set_color(1, 0, 0, 1)

  polyline = publisher.get_new_line()
  polyline.set_scale(0.1, 0.1)
  k = 0
  while k < self.solver.N:
  
   poly = _polyhedrons[k]
   polyline.set_color_int(k, self.solver.N)

   vertices = cal_vertices(poly)
   if (vertices.size() < 2)
    continue

   for i in range(vertices.size()):
 
    if (visualize_points):
     polypoint.add_point_marker((vertices[i](0), vertices[i](1), 0))

    if i > 0:
     polyline.add_line((vertices[i - 1](0), vertices[i - 1](1), 0), (vertices[i](0), vertices[i](1), 0))
   
   polyline.add_line((vertices.back()(0), vertices.back()(1), 0), (vertices[0](0), vertices[0](1), 0))
   k += CONFIG["visualization"]["draw_every"]

  publisher.publish()

  if not CONFIG["debug_visuals"]:
   return

  LOG_DEBUG( "DecompConstraints.Visualize")

  map_publisher = VISUALS.get_publisher("map")
  point = map_publisher.get_new_point_marker("CUBE")
  point.set_scale(0.1, 0.1, 0.1)
  point.set_color(0, 0, 0, 1)

  for vec in _occ_pos:
   point.add_point_marker((vec.x(), vec.y(), 0))
  map_publisher.publish()
