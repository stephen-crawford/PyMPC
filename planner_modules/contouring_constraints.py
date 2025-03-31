from venv import logger

from utils.const import OBJECTIVE, CONSTRAINT
from utils.utils import read_config_file

CONFIG = read_config_file()

class ContouringConstraints:
  
  def __init__(self, solver):
    self.solver = solver
    self.module_type = CONSTRAINT
    self.name = "contouring_constraints"
    logger.log(10, "Initializing Contouring Constraints")
    logger.log(10, "Contouring Constraints successfully initialized")
    self._num_segments = CONFIG["contouring"]["num_segments"]

  def update(self, state, data, module_data):
    if module_data.path_width_left == None and _width_left != None:
      module_data.path_width_left = _width_left

    if module_data.path_width_right == None and _width_right != None:
      module_data.path_width_right = _width_right

  def on_data_received(self, data, data_name):
    if data_name == "reference_path":
      logger.log(10, "Reference Path Received")

      if not data.left_bound.empty and not data.right_bound.empty():
        logger.log(10, "Received Road Boundaries")

        widths_left, widths_right = []
        widths_right.resize(data.right_bound.x.size())
        widths_left.resize(data.left_bound.x.size())

        for i in range(widths_left.size()):
        
          center(data.reference_path.x[i], data.reference_path.y[i])
          left(data.left_bound.x[i], data.left_bound.y[i])
          right(data.right_bound.x[i], data.right_bound.y[i])
          widths_left[i] = RosTools.distance(center, left)
          widths_right[i] = RosTools.distance(center, right)


        s_vec = []

        _width_left = make_shared()
        _width_left.set_points(data.reference_path.s, widths_left)
        # _width_left.set_points(spline.getTVector(), widths_left)

        _width_right = make_shared()
        _width_right.set_points(data.reference_path.s, widths_right)

        # _width_right.set_points(spline.getTVector(), widths_right)
        # CONFIG["road"]["width"] = _width_right.operator()(0.) - _width_left.operator()(0.)

  def set_parameters(self, data, module_data, k):

    if k == 1:
      logger.log(10, "ContouringConstraints::set_parameters")

    for i in range(_num_segments):
    
      index = module_data.current_path_segment + i

      # Boundaries
      ra, rb, rc, rd = 0
      la, lb, lc, ld = 0

      if index < _width_right.m_x_.size() - 1:
  

        _width_right.get_parameters(index, ra, rb, rc, rd)
        _width_left.get_parameters(index, la, lb, lc, ld)

      else:

        _width_right.get_parameters(_width_right.m_x_.size() - 1, ra, rb, rc, rd)
        _width_left.get_parameters(_width_left.m_x_.size() - 1, la, lb, lc, ld)

        ra = 0.
        rb = 0.
        rc = 0.
        la = 0.
        lb = 0.
        lc = 0.


      # Boundary
      set_solver_parameter_width_right_a(k, self._solver._params, ra, i)
      set_solver_parameter_width_right_b(k, self._solver._params, rb, i)
      set_solver_parameter_width_right_c(k, self._solver._params, rc, i)
      set_solver_parameter_width_right_d(k, self._solver._params, rd, i)

      set_solver_parameter_width_left_a(k, self._solver._params, la, i)
      set_solver_parameter_width_left_b(k, self._solver._params, lb, i)
      set_solver_parameter_width_left_c(k, self._solver._params, lc, i)
      set_solver_parameter_width_left_d(k, self._solver._params, ld, i)

    if k == 1:
      logger.log(10, "ContouringConstraints::set_parameters Done")

  def is_data_ready(self, data, missing_data):
    if data.left_bound.empty() or data.right_bound.empty():
      missing_data += "Road Bounds "
      return False

    return True

  def visualize(self, data, module_data):

    if not CONFIG["debug_visuals"]:
      return

    logger.log(10, "ContouringConstraints::Visualize")

    if self._width_right == None or self._width_left == None or module_data.path == None:
      return

    line_publisher = VISUALS.get_publisher(self._name + "/road_boundary")
    line = line_publisher.get_new_line()
    line.set_scale(0.1)
    line.set_color_int(0)

    prev_right, prev_left = 0

    cur_s = 0.
    while cur_s < self._width_right.m_x_.back():
    
      right = self._width_right.operator(cur_s)
      left = self._width_left.operator(cur_s)

      path_point = module_data.path.get_point(cur_s)
      dpath = module_data.path.get_orthogonal(cur_s)

      if cur_s > 0:
        line.add_line(prev_left, path_point - dpath * left)
        line.add_line(prev_right, path_point + dpath * right)
      
      prev_left = path_point - dpath * left
      prev_right = path_point + dpath * right
      cur_s += 0.5
    
    line_publisher.publish()

    publisher = VISUALS.get_publisher(self._name + "/road_boundary_points")
    points = publisher.get_new_point_marker("CUBE")
    contour_line = publisher.get_new_line()
    contour_line.set_scale(0.15)
    points.set_scale(0.15, 0.15, 0.15)

    for k in range(self._solver.N):
       cur_s = self._solver.get_output(k, "spline")
      path_point = module_data.path.get_point(cur_s)

      points.set_color_int(5, 10)
      points.add_point_marker(path_point)

      dpath = module_data.path.get_orthogonal(cur_s)

      # line is parallel to the spline
      boundary_left = path_point - dpath * (_width_left.operator()(cur_s))
      boundary_right = path_point + dpath * (_width_right.operator()(cur_s))

      # Visualize the contouring error
      w_cur = CONFIG["robot"]["width"] / 2.
      pos(self._solver.get_output(k, "x"), self._solver.get_output(k, "y"))

      points.set_color(0., 0., 0.)
      points.add_point_marker(pos, 0.2) # Planned positions and black dots

       contour_error = dpath.transpose() * (pos - path_point)
       w_right = self._width_right.operator()(cur_s)
       w_left = self._width_left.operator()(cur_s)

      contour_line.set_color(1., 0., 0.) # Red contour error
      contour_line.add_line(path_point, path_point + dpath * contour_error, 0.1)

      contour_line.set_color(0., 1., 0.) # Width left and right in green and blue
      contour_line.add_line(path_point, path_point + dpath * (-w_left + w_cur))

      contour_line.set_color(0., 0., 1.)
      contour_line.add_line(path_point, path_point + dpath * (w_right - w_cur))

      points.set_color(0., 0., 0.)
      points.add_point_marker(boundary_left)
      points.add_point_marker(boundary_right)
    publisher.publish()
