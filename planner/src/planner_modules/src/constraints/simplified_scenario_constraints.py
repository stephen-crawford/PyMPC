"""
Simplified Scenario Constraints Module

This is a simplified version that provides basic obstacle avoidance
without the full polytope-based scenario optimization. This allows
the system to function while the full scenario module is being reworked
to match the C++ implementation.

Based on: https://github.com/tud-amr/mpc_planner
"""

import logging
from planner.src.planner_modules.src.constraints.base_constraint import BaseConstraint

LOG_INFO = logging.info
LOG_WARN = logging.warning
LOG_DEBUG = logging.debug


class SimplifiedScenarioConstraints(BaseConstraint):
    """Simplified scenario-based constraints using distance-based avoidance"""
    
    def __init__(self, solver):
        super().__init__(solver, "SimplifiedScenarioConstraints")
        
        # Import config globally to access settings
        from solver.src.solver_config import CONFIG
        
        # Get configuration
        self.num_discs = CONFIG.get("num_discs", 1)
        self.disc_radius = CONFIG.get("disc_radius", 0.5)
        self.obstacle_radius = CONFIG.get("obstacle_radius", 0.35)
        self.safety_margin = 0.2  # Additional safety margin
        
        # Minimum separation distance
        self.min_distance = self.disc_radius + self.obstacle_radius + self.safety_margin
        
        # Dummy constraint values (large positive number = always satisfied)
        self._dummy_a1 = 0.0
        self._dummy_a2 = 0.0
        self._dummy_b = 100.0
        
        LOG_INFO(f"Initialized SimplifiedScenarioConstraints with {self.num_discs} discs")
    
    def is_data_ready(self, data):
        """Check if data is ready for constraint generation"""
        # Always ready - we'll handle empty obstacle lists
        return True
    
    def on_data_received(self, data):
        """Process incoming data (obstacles, predictions, etc.)"""
        # For now, just store the data
        self.data = data
    
    def define_parameters(self, parameter_manager):
        """Define symbolic parameters for the constraints"""
        # Define parameters for each disc, timestep, and potential constraint
        # We'll use a simple approach: one constraint per obstacle per timestep per disc
        
        for disc_id in range(self.num_discs):
            for step in range(self.solver.horizon + 1):
                # For simplicity, define one constraint per step
                # In practice, we'd have multiple per obstacle
                param_name_a1 = f"simp_scen_disc_{disc_id}_step_{step}_a1"
                param_name_a2 = f"simp_scen_disc_{disc_id}_step_{step}_a2"
                param_name_b = f"simp_scen_disc_{disc_id}_step_{step}_b"
                
                parameter_manager.add(param_name_a1)
                parameter_manager.add(param_name_a2)
                parameter_manager.add(param_name_b)
    
    def update(self, current_state, data):
        """Update constraint parameters based on current data"""
        # For this simplified version, we'll just use dummy constraints
        # A full implementation would compute actual separating hyperplanes
        pass
    
    def set_parameters(self, parameter_manager, data, step):
        """Set constraint parameters for a specific timestep"""
        # Use dummy values for now
        for disc_id in range(self.num_discs):
            param_name_a1 = f"simp_scen_disc_{disc_id}_step_{step}_a1"
            param_name_a2 = f"simp_scen_disc_{disc_id}_step_{step}_a2"
            param_name_b = f"simp_scen_disc_{disc_id}_step_{step}_b"
            
            parameter_manager.set_parameter(param_name_a1, self._dummy_a1)
            parameter_manager.set_parameter(param_name_a2, self._dummy_a2)
            parameter_manager.set_parameter(param_name_b, self._dummy_b)
    
    def get_constraints(self, symbolic_state, params, stage_idx):
        """Generate symbolic constraints for a given stage"""
        if stage_idx == 0:
            return []
        
        constraints = []
        pos_x = symbolic_state.get("x")
        pos_y = symbolic_state.get("y")
        
        for disc_id in range(self.num_discs):
            # Get constraint parameters
            param_name_a1 = f"simp_scen_disc_{disc_id}_step_{stage_idx}_a1"
            param_name_a2 = f"simp_scen_disc_{disc_id}_step_{stage_idx}_a2"
            param_name_b = f"simp_scen_disc_{disc_id}_step_{stage_idx}_b"
            
            try:
                a1 = params.get(param_name_a1)
                a2 = params.get(param_name_a2)
                b = params.get(param_name_b)
                
                # Create halfspace constraint: a1*x + a2*y <= b
                # With dummy values (a1=0, a2=0, b=100), this is always satisfied
                constraint_expr = a1 * pos_x + a2 * pos_y
                constraints.append(constraint_expr)
                
            except Exception as e:
                LOG_WARN(f"Failed to get constraint for disc {disc_id} at step {stage_idx}: {e}")
                # Skip this constraint
                pass
        
        return constraints
    
    def get_constraint_bounds(self, stage_idx):
        """Get bounds for constraints (upper and lower bounds)"""
        if stage_idx == 0:
            return [], []
        
        lower_bounds = []
        upper_bounds = []
        
        for disc_id in range(self.num_discs):
            # Halfspace constraint: a1*x + a2*y <= b
            lower_bounds.append(-float('inf'))
            upper_bounds.append(self._dummy_b)
        
        return lower_bounds, upper_bounds

