from planner.src.data_prep import logger
from utils.const import OBJECTIVE


from utils.utils import read_config_file

CONFIG = read_config_file()


class PathReferenceVelocity:

  def __init__(self, solver):
    self.solver = solver
    self.module_type = OBJECTIVE
    self.name = "path_reference_velocity"
    logger.log(10, "Initializing Path Reference Velocity")
    logger.log(10, "Path Reference Velocity successfully initialized")
    self._n_segments = CONFIG["contouring"]["get_num_segments"]

  def update(self, state, data, module_data):
   if module_data.path_velocity == None and _velocityspline != None:
    module_data.path_velocity = _velocityspline

  def on_data_received(self, data, data_name):
   if data_name == "reference_path":

    logger.log("Received Reference Path")

    if data.reference_path.hasVelocity():
     _velocityspline = make_shared()
     _velocityspline.set_points(data.reference_path.s, data.reference_path.v)


  def set_parameters(self, data, module_data, k):

   # Retrieve once
   if k == 0:
    # velocity_weight = CONFIG["weights"]["velocity"]
    reference_velocity = CONFIG["weights"]["reference_velocity"]
  
   if data.reference_path.hasVelocity(): # Use a spline-based velocity reference
  
    logger.log(10, "Using spline-based reference velocity")
    for i in range(self._n_segments):
     index = module_data.current_path_segment + i

     if index < velocityspline.m_x_.size() - 1:
      _velocityspline.get_parameters(index, a, b, c, d)
     else:
      # Brake at the end
      a = 0.
      b = 0.
      c = 0.
      d = 0.

     set_solver_parameterspline_va(k, self.solver._params, a, i)
     set_solver_parameterspline_vb(k, self.solver._params, b, i)
     set_solver_parameterspline_vc(k, self.solver._params, c, i)
     set_solver_parameterspline_vd(k, self.solver._params, d, i)
   else: # Use a constant velocity reference
    for i in range(self._n_segments):

     set_solver_parameterspline_va(k, self.solver._params, 0., i)
     set_solver_parameterspline_vb(k, self.solver._params, 0., i)
     set_solver_parameterspline_vc(k, self.solver._params, 0., i)
     set_solver_parameterspline_vd(k, self.solver._params, reference_velocity, i) # v = d

 def visualize(self, data, module_data):

  if data.reference_path.empty() or data.reference_path.s.empty():
   return

  if not CONFIG["debug_visuals"]:
   return

  logger.log(10, "PathReferenceVelocity.Visualize")

  # Only for debugging
  publisher = VISUALS.get_publisher("path_velocity")
  line = publisher.get_new_line()

  line.set_scale(0.25, 0.25, 0.1)
  spline_xy = make_unique(data.reference_path.x, data.reference_path.y, data.reference_path.s)

  prev = []
  prev_v = 0.
  for s in range(_velocityspline.m_x_.back()):

   cur = spline_xy.getPoint(s)
   v = _velocityspline.operator()(s)

   if s > 0.:
    line.set_color(0, (v + prev_v) / (2. * 3. * 2.), 0.)
    line.add_line(prev, cur)

   prev = cur
   prev_v = v

  publisher.publish()
