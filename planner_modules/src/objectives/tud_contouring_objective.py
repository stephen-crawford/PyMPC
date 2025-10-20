#!/usr/bin/env python3
"""
TUD-AMR Style Contouring Objective

This implements contouring objective following the TUD-AMR MPC planner approach.
Key features:
- Minimizes lateral deviation from reference path
- Minimizes lag error (progress along path)
- Proper objective formulation for MPC
"""

import numpy as np
import casadi as cd
from scipy.interpolate import CubicSpline

from planner_modules.src.objectives.base_objective import BaseObjective
from planning.src.types import ReferencePath
from utils.math_utils import distance, haar_difference_without_abs, safe_norm, Spline, Spline2D
from utils.utils import LOG_DEBUG, LOG_INFO, LOG_WARN


class TUDContouringObjective(BaseObjective):
    """TUD-AMR style contouring objective for path following."""
    
    def __init__(self, solver):
        super().__init__(solver)
        
        LOG_DEBUG("TUD Contouring Objective initializing")
        
        # Configuration options
        self.num_segments = self.get_config_value("contouring.num_segments", 10)
        self.dynamic_velocity_reference = self.get_config_value("contouring.dynamic_velocity_reference", False)
        self.goal_reaching_contouring = self.get_config_value("contouring.goal_reaching_contouring", False)
        self.three_dimensional_contouring = self.get_config_value("contouring.three_dimensional_contouring", False)
        
        # State tracking
        self.closest_point_idx = 0
        self.closest_segment = 0
        
        # Road bounds
        self.reference_path = None
        self.bound_left_spline = None
        self.bound_right_spline = None
        
        LOG_DEBUG("TUD Contouring module successfully initialized")
    
    def update(self, state, data):
        """Update objective state."""
        LOG_INFO("TUDContouringObjective.update")
        
        if self.reference_path is None:
            LOG_WARN("No reference path available")
            return
        
        # Update closest point
        self.closest_point_idx, self.closest_segment = self._find_closest_point(
            state.get_position(), self.reference_path)
        
        # Pass path data to data object
        if self.reference_path is not None:
            data.reference_path = ReferencePath()
            data.reference_path.set('x', self.reference_path.x)
            data.reference_path.set('y', self.reference_path.y)
            data.reference_path.set('s', self.reference_path.s)
        
        data.current_path_segment = self.closest_segment
        
        # Log distance to closest point
        closest_pt = np.array([
            self.reference_path.x[self.closest_point_idx], 
            self.reference_path.y[self.closest_point_idx]
        ])
        vehicle_pos = np.array(state.get_position())
        LOG_DEBUG(f"Distance to closest point: {np.linalg.norm(closest_pt - vehicle_pos)}")
    
    def define_parameters(self, params):
        """Define parameters for the objective."""
        LOG_DEBUG("Defining TUD contouring objective parameters")
        
        # Core parameters
        params.add("contour_weight", add_to_rqt_reconfigure=True)
        params.add("contouring_lag_weight", add_to_rqt_reconfigure=True)
        
        # Velocity reference parameters if needed
        if self.dynamic_velocity_reference:
            params.add("contouring_reference_velocity_weight", add_to_rqt_reconfigure=True)
        
        if self.goal_reaching_contouring:
            params.add("contouring_goal_weight", add_to_rqt_reconfigure=True)
            params.add("contouring_goal_x")
            params.add("contouring_goal_y")
            if self.three_dimensional_contouring:
                params.add("contouring_goal_z")
        
        # Path parameters
        for i in range(self.num_segments + 1):
            params.add(f"path_{i}_start")
            params.add(f"path_x_{i}_a")
            params.add(f"path_x_{i}_b")
            params.add(f"path_x_{i}_c")
            params.add(f"path_x_{i}_d")
            params.add(f"path_y_{i}_a")
            params.add(f"path_y_{i}_b")
            params.add(f"path_y_{i}_c")
            params.add(f"path_y_{i}_d")
            
            if self.three_dimensional_contouring:
                params.add(f"path_z_{i}_a")
                params.add(f"path_z_{i}_b")
                params.add(f"path_z_{i}_c")
                params.add(f"path_z_{i}_d")
            
            # Velocity reference if needed
            if self.dynamic_velocity_reference:
                params.add(f"path_vel_{i}_a")
                params.add(f"path_vel_{i}_b")
                params.add(f"path_vel_{i}_c")
                params.add(f"path_vel_{i}_d")
    
    def set_parameters(self, parameter_manager, data, k):
        """Set parameters for the objective."""
        if k == 0:
            # Retrieve weights
            contouring_weight = self.get_config_value("weights.contour_weight", 1.0)
            lag_weight = self.get_config_value("weights.contouring_lag_weight", 1.0)
            
            if self.goal_reaching_contouring:
                contouring_goal_weight = self.get_config_value("weights.contouring_goal_weight", 1.0)
                parameter_manager.set_parameter("contouring_goal_weight", contouring_goal_weight)
            
            if self.dynamic_velocity_reference:
                velocity_weight = self.get_config_value("weights.contour_velocity_weight", 1.0)
                parameter_manager.set_parameter("contouring_reference_velocity_weight", velocity_weight)
            
            parameter_manager.set_parameter("contour_weight", contouring_weight)
            parameter_manager.set_parameter("contouring_lag_weight", lag_weight)
        
        self.process_reference_path(data)
        self.set_path_parameters(parameter_manager)
    
    def _fit_cubic_spline_coefficients(self, x_data, y_data):
        """Fit cubic spline coefficients for the given data points."""
        n = len(x_data)
        if n < 2:
            raise ValueError("Need at least 2 points for spline fitting")
        
        segments = []
        segment_starts = []
        
        for i in range(n - 1):
            x_start = x_data[i]
            x_end = x_data[i + 1]
            y_start = y_data[i]
            y_end = y_data[i + 1]
            
            segment_starts.append(x_start)
            
            # Estimate derivatives
            if i == 0:
                if n > 2:
                    dy_start = (y_data[i + 1] - y_data[i]) / (x_data[i + 1] - x_data[i])
                else:
                    dy_start = 0
            else:
                dy_start = (y_data[i + 1] - y_data[i - 1]) / (x_data[i + 1] - x_data[i - 1])
            
            if i == n - 2:
                if n > 2:
                    dy_end = (y_data[i + 1] - y_data[i]) / (x_data[i + 1] - x_data[i])
                else:
                    dy_end = 0
            else:
                dy_end = (y_data[i + 2] - y_data[i]) / (x_data[i + 2] - x_data[i])
            
            # Scale derivatives by segment length
            dx = x_end - x_start
            dy_start_scaled = dy_start * dx
            dy_end_scaled = dy_end * dx
            
            # Solve for cubic coefficients
            d = y_start
            c = dy_start_scaled
            a = 2 * y_start - 2 * y_end + dy_start_scaled + dy_end_scaled
            b = -3 * y_start + 3 * y_end - 2 * dy_start_scaled - dy_end_scaled
            
            segments.append([float(a), float(b), float(c), float(d)])
        
        return segments, segment_starts
    
    def set_path_parameters(self, parameter_manager):
        """Set path parameters for the objective."""
        LOG_DEBUG("TUD Contouring Objective::set_path_parameters")
        
        if self.reference_path is None:
            LOG_WARN("No reference path available")
            return
        
        path_x = self.reference_path.x
        path_y = self.reference_path.y
        
        # Prepare interpolated data
        if len(path_x) < self.num_segments + 1:
            s_original = np.linspace(0, 1, len(path_x))
            s_new = np.linspace(0, 1, self.num_segments + 1)
            
            path_x_interp = np.interp(s_new, s_original, path_x)
            path_y_interp = np.interp(s_new, s_original, path_y)
        else:
            # Compute cumulative arc length
            dx = np.diff(path_x)
            dy = np.diff(path_y)
            arc_lengths = np.sqrt(dx ** 2 + dy ** 2)
            s = np.concatenate([[0], np.cumsum(arc_lengths)])
            s /= s[-1]
            
            s_new = np.linspace(0, 1, self.num_segments + 1)
            path_x_interp = np.interp(s_new, s, path_x)
            path_y_interp = np.interp(s_new, s, path_y)
        
        # Fit cubic spline coefficients
        try:
            path_x_coeffs, x_starts = self._fit_cubic_spline_coefficients(s_new, path_x_interp)
            path_y_coeffs, y_starts = self._fit_cubic_spline_coefficients(s_new, path_y_interp)
        except ValueError as e:
            LOG_WARN(f"Error fitting spline coefficients: {e}")
            return
        
        # Set parameters for each segment
        for i in range(self.num_segments):
            # Set segment start parameter
            parameter_manager.set_parameter(f"path_{i}_start", float(x_starts[i]))
            
            # Set path coordinate coefficients
            parameter_manager.set_parameter(f"path_x_{i}_a", path_x_coeffs[i][0])
            parameter_manager.set_parameter(f"path_x_{i}_b", path_x_coeffs[i][1])
            parameter_manager.set_parameter(f"path_x_{i}_c", path_x_coeffs[i][2])
            parameter_manager.set_parameter(f"path_x_{i}_d", path_x_coeffs[i][3])
            
            parameter_manager.set_parameter(f"path_y_{i}_a", path_y_coeffs[i][0])
            parameter_manager.set_parameter(f"path_y_{i}_b", path_y_coeffs[i][1])
            parameter_manager.set_parameter(f"path_y_{i}_c", path_y_coeffs[i][2])
            parameter_manager.set_parameter(f"path_y_{i}_d", path_y_coeffs[i][3])
    
    def get_value(self, symbolic_state, params, stage_idx):
        """Get objective value."""
        if params is None:
            params = self.solver.parameter_manager
        
        # Get symbolic state variables
        pos_x = symbolic_state.get("x")
        pos_y = symbolic_state.get("y")
        psi = symbolic_state.get("psi")
        v = symbolic_state.get("v")
        s = symbolic_state.get("spline") / self.reference_path.s[-1]
        
        # Validate required variables
        if any(var is None for var in [pos_x, pos_y, psi, v, s]):
            missing_vars = [name for name, var in
                           [("x", pos_x), ("y", pos_y), ("psi", psi), ("v", v), ("spline", s)] if var is None]
            raise ValueError(f"Missing required symbolic variables: {missing_vars}")
        
        # Get symbolic weights
        contour_weight = params.get("contour_weight")
        lag_weight = params.get("contouring_lag_weight")
        
        # Create path spline
        path = Spline2D(params, self.num_segments, s)
        path_x, path_y = path.at(s)
        path_dx_normalized, path_dy_normalized = path.deriv_normalized(s)
        
        # TUD-AMR style MPCC (Model Predictive Contouring Control)
        # Calculate contouring error (lateral deviation from path)
        contour_error = path_dy_normalized * (pos_x - path_x) - path_dx_normalized * (pos_y - path_y)
        
        # Calculate lag error (progress along path)
        lag_error = path_dx_normalized * (pos_x - path_x) + path_dy_normalized * (pos_y - path_y)
        
        # Cost components
        lag_cost = lag_weight * lag_error ** 2
        contour_cost = contour_weight * contour_error ** 2
        
        # Additional costs
        velocity_cost = 0
        goal_cost = 0
        terminal_cost = 0
        
        # Velocity cost (if enabled)
        if self.dynamic_velocity_reference:
            if params.has_parameter("spline_v0_a"):
                path_velocity = Spline(params, "spline_v", self.num_segments, s)
                reference_velocity = path_velocity.at(s)
                velocity_weight = params.get("contouring_reference_velocity_weight")
                velocity_cost = velocity_weight * (v - reference_velocity) ** 2
        
        # Goal cost (if enabled)
        if self.goal_reaching_contouring:
            goal_weight = params.get("contouring_goal_weight")
            remaining_distance = self.reference_path.s[-1] - s * self.reference_path.s[-1]
            goal_cost = goal_weight * remaining_distance ** 2
        
        # Terminal cost (if enabled)
        if self.goal_reaching_contouring and stage_idx == self.solver.horizon - 1:
            terminal_angle_weight = self.get_config_value("contouring.terminal_angle", 1.0)
            terminal_contouring_mp = self.get_config_value("contouring.terminal_contouring", 1.0)
            
            # Compute angle error
            path_angle = cd.atan2(path_dy_normalized, path_dx_normalized)
            angle_error = haar_difference_without_abs(psi, path_angle)
            
            terminal_cost = (terminal_angle_weight * angle_error ** 2 +
                            terminal_contouring_mp * lag_weight * lag_error ** 2 +
                            terminal_contouring_mp * contour_weight * contour_error ** 2)
        
        return {
            "contouring_lag_cost": lag_cost,
            "contouring_contour_cost": contour_cost,
            "contouring_velocity_cost": velocity_cost,
            "contouring_goal_cost": goal_cost,
            "contouring_terminal_cost": terminal_cost
        }
    
    def on_data_received(self, data):
        """Process received data."""
        LOG_DEBUG("Received data for TUD Contouring Objective")
        if data.has("reference_path") and data.reference_path is not None:
            LOG_DEBUG("Received Reference Path")
            self.process_reference_path(data)
    
    def process_reference_path(self, data):
        """Process reference path."""
        LOG_DEBUG("Processing reference path for TUD Contouring Objective")
        
        self.reference_path = data.reference_path
        
        # Process road bounds if available
        if data.left_bound is not None and data.right_bound is not None:
            LOG_DEBUG("Processing road boundaries for TUD Contouring Objective")
            self.bound_left_spline = CubicSpline(self.reference_path.s,
                                                np.column_stack((data.left_bound.x, data.left_bound.y)))
            self.bound_right_spline = CubicSpline(self.reference_path.s,
                                                 np.column_stack((data.right_bound.x, data.right_bound.y)))
    
    def _find_closest_point(self, position, reference_path: ReferencePath):
        """Find the closest point on the path to the given position."""
        if reference_path.empty():
            return 0, 0
        
        pos_x, pos_y = position
        
        # Compute squared distances to all path points
        dx = reference_path.x - pos_x
        dy = reference_path.y - pos_y
        distances_squared = dx ** 2 + dy ** 2
        
        closest_idx = np.argmin(distances_squared)
        
        # Calculate segment index
        segment_idx = 0
        if len(reference_path.s) > 1 and self.num_segments > 0:
            closest_s = reference_path.s[closest_idx]
            s_min = reference_path.s[0]
            s_max = reference_path.s[-1]
            
            norm_s = (closest_s - s_min) / (s_max - s_min)
            segment_idx = min(int(norm_s * self.num_segments), self.num_segments - 1)
        
        LOG_DEBUG(f"Found closest point at {closest_idx} and segment: {segment_idx}")
        return closest_idx, segment_idx
    
    def is_data_ready(self, data):
        """Check if required data is available."""
        missing_data = ""
        if not data.has("reference_path") or data.reference_path.x is None:
            missing_data += "reference_path"
        
        return len(missing_data) < 1
    
    def is_objective_reached(self, state, data):
        """Check if objective is reached."""
        is_ready = self.is_data_ready(data)
        if not is_ready:
            LOG_DEBUG("Data not ready yet")
            return False
        
        if self.reference_path is None:
            return False
        
        # Get the final point on the path
        final_x = self.reference_path.x[-1]
        final_y = self.reference_path.y[-1]
        final_point = np.array([final_x, final_y])
        
        # Check if we're close enough to the final point
        reached = distance(state.get_position(), final_point) < 1.0
        LOG_DEBUG(f"TUD Contouring objective reached: {reached}")
        return reached
    
    def reset(self):
        """Reset the state of the contouring objective."""
        self.closest_segment = 0
