from functools import partial

from planner.src.types import TwoDimensionalSpline
from solver.solver_interface import set_solver_parameter
from utils.const import OBJECTIVE
from utils.utils import read_config_file, LOG_DEBUG
from utils.visualizer import VISUALS

CONFIG = read_config_file()




class PathReferenceVelocity:

    def __init__(self, solver):
        self.solver = solver
        self.module_type = OBJECTIVE
        self.name = 'path_reference_velocity'
        self.config = read_config_file().get(self.name, {})
        LOG_DEBUG("Initializing Path Reference Velocity")
        LOG_DEBUG("Path Reference Velocity successfully initialized")
        self.n_segments = self.get_config_value("contouring.get_num_segments")
        self.velocity_spline = None
        self.set_solver_param = partial(set_solver_parameter, settings=CONFIG)

    def update(self, state, data, module_data):
        if module_data.path_velocity is None and self.velocity_spline is not None:
            module_data.path_velocity = self.velocity_spline

    def get_config_value(self, key, default=None):
        """Get configuration value with fallback to default"""
        return self.config.get(key)

    def on_data_received(self, data, data_name):
        if data_name == "reference_path":
            LOG_DEBUG("Received Reference Path")
            if data.reference_path.hasVelocity():
                self.velocity_spline.set_points(data.reference_path.s, data.reference_path.v)

    def set_parameters(self, data, module_data, k):
        print("Trying to set parameters")
        reference_velocity = 0.0
        if k == 0:
            reference_velocity = self.get_config_value("weights.reference_velocity")

        if data.reference_path.hasVelocity():  # Use a spline-based velocity reference
            LOG_DEBUG("Using spline-based reference velocity")
            for i in range(self.n_segments):
                index = module_data.current_path_segment + i

                if index < self.velocity_spline.m_x_.size() - 1:
                    a, b, c, d = self.velocity_spline.get_parameters(index)
                else:
                    # Brake at the end
                    a = b = c = 0.0
                    d = 0.0

                self.set_solver_param(self.solver.params, "spline_va", a, k, i)
                self.set_solver_param(self.solver.params, "spline_vb", b, k, i)
                self.set_solver_param(self.solver.params, "spline_vc", c, k, i)
                self.set_solver_param(self.solver.params, "spline_vd", d, k, i)
        else:
            for i in range(self.n_segments):
                self.set_solver_param(self.solver.params, "spline_va", 0.0, k, i)
                self.set_solver_param(self.solver.params, "spline_vb", 0.0, k, i)
                self.set_solver_param(self.solver.params, "spline_vc", 0.0, k, i)
                self.set_solver_param(self.solver.params, "spline_vd", reference_velocity, k, i)

    def visualize(self, data, module_data):
        if data.reference_path.empty() or data.reference_path.s.empty():
            return

        if not self.get_config_value("debug_visuals"):
            return

        LOG_DEBUG("PathReferenceVelocity.Visualize")

        publisher = VISUALS.get_publisher("path_velocity")
        line = publisher.get_new_line()
        line.set_scale(0.25, 0.25, 0.1)

        spline_xy = TwoDimensionalSpline(data.reference_path.x, data.reference_path.y, data.reference_path.s)

        prev = []
        prev_v = 0.0
        for s in range(self.velocity_spline.m_x_.back()):
            cur = spline_xy.get_point(s)
            v = self.velocity_spline.operator()(s)

            if s > 0.0:
                line.set_color(0, (v + prev_v) / (2.0 * 3.0 * 2.0), 0.0)
                line.add_line(prev, cur)

            prev = cur
            prev_v = v

        publisher.publish()
