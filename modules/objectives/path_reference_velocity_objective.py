from scipy.interpolate import CubicSpline

from modules.objectives.base_objective import BaseObjective
from utils.utils import LOG_DEBUG

class PathReferenceVelocityObjective(BaseObjective):

	def __init__(self):
		super().__init__()
        self.name = 'path_reference_velocity'
        self.num_segments = self.get_config_value("contouring.get_num_segments")
        self.velocity_spline = None

    def update(self, state, data, module_data):
        if module_data.path_velocity is None and self.velocity_spline is not None:
            module_data.path_velocity = self.velocity_spline

    def on_data_received(self, data):
        LOG_DEBUG("Received Reference Path")
        if data.reference_path.has_velocity():
            self.velocity_spline = CubicSpline()
            self.velocity_spline.set_points(data.reference_path.s, data.reference_path.v)

    def define_parameters(self, params):

        for i in range(self.num_segments):
            params.add(f"spline_{i}_va")
            params.add(f"spline_{i}_vb")
            params.add(f"spline_{i}_vc")
            params.add(f"spline_{i}_vd")

        return params

    def get_stage_cost_symbolic(self, symbolic_state, stage_idx):
        """
        Return symbolic objective cost expressions for path reference velocity.
        
        CRITICAL: This method returns symbolic CasADi expressions for MPC rollouts.
        The symbolic_state contains CasADi variables for the predicted state at this stage.
        
        Note: The velocity cost is typically computed in the contouring objective,
        so this returns zero cost here.
        
        Reference: https://github.com/tud-amr/mpc_planner - objectives are evaluated symbolically.
        """
        # The cost is computed in the contouring cost
        import casadi as cd
        return {"path_reference_velocity_cost": cd.MX(0.0)}
    
    def get_value(self, state, params, stage_idx):
        # The cost is computed in the contouring cost
        # Return dict format for compatibility
        return {"path_reference_velocity_cost": 0.0}

    def set_parameters(self, parameter_manager, data, k):
        print("Trying to set parameters")
        reference_velocity = 0.0
        if k == 0:
            reference_velocity = self.get_config_value("weights.reference_velocity")

        if data.reference_path.has_velocity():  # Use a spline-based velocity reference
            LOG_DEBUG("Using spline-based reference velocity")
            for i in range(self.num_segments):
                index = data.current_path_segment + i

                if index < self.velocity_spline.m_x_.size() - 1:
                    a, b, c, d = self.velocity_spline.get_parameters(index)
                else:
                    # Brake at the end
                    a = b = c = d = 0.0

                parameter_manager.set_parameter(f"spline_{i}_va", a)
                parameter_manager.set_parameter(f"spline_{i}_vb", b)
                parameter_manager.set_parameter(f"spline_{i}_vc", c)
                parameter_manager.set_parameter(f"spline_{i}_vd", d)

        else:
            a = b = c = 0.0
            d = reference_velocity
            for i in range(self.num_segments):
                parameter_manager.set_parameter(f"spline_{i}_va", a)
                parameter_manager.set_parameter(f"spline_{i}_vb", b)
                parameter_manager.set_parameter(f"spline_{i}_vc", c)
                parameter_manager.set_parameter(f"spline_{i}_vd", d)
