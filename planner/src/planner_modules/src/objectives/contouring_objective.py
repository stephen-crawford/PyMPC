"""
Contouring Objective Implementation for MPC

This module implements Model Predictive Contouring Control (MPCC) objectives
that minimize contouring and lag errors while maximizing progress along a path.

Based on the C++ mpc_planner implementation.
"""

import numpy as np
import casadi as cd
import logging

from planner.src.planner_modules.src.objectives.base_objective import BaseObjective
from planning.src.types import Data, State
from utils.math_utils import Spline, Spline2D
from utils.utils import LOG_DEBUG, LOG_INFO, LOG_WARN

LOG = logging.getLogger(__name__)


class ContouringObjective(BaseObjective):
    """
    Contouring objective implementation for MPCC.
    
    This objective minimizes:
    1. Contouring error (lateral deviation from path)
    2. Lag error (longitudinal deviation from path)
    3. Progress maximization
    4. Optional velocity tracking
    """
    
    def __init__(self, solver):
        super().__init__(solver)
        self.name = "contouring_objective"
        
        LOG_DEBUG("Initializing Contouring Objective")
        
        # Configuration
        self.num_segments = self.get_config_value("contouring.num_segments", 5)
        self.contour_weight = self.get_config_value("weights.contour_weight", 1.0)
        self.lag_weight = self.get_config_value("weights.lag_weight", 1.0)
        self.velocity_weight = self.get_config_value("weights.velocity_weight", 0.1)
        self.progress_weight = self.get_config_value("weights.progress_weight", 1.0)
        
        # Optional features
        self.dynamic_velocity_reference = self.get_config_value("contouring.dynamic_velocity_reference", False)
        self.goal_reaching_contouring = self.get_config_value("contouring.goal_reaching_contouring", False)
        self.three_dimensional_contouring = self.get_config_value("contouring.three_dimensional_contouring", False)
        
        # Path data
        self.reference_path = None
        self.path_spline = None
        
        # Closest point tracking
        self.closest_point_idx = 0
        self.closest_segment = 0
        
        LOG_INFO("Contouring Objective initialized successfully")
    
    def is_data_ready(self, data: Data) -> bool:
        """Check if required data is available."""
        return (hasattr(data, 'reference_path') and 
                data.reference_path is not None and 
                not data.reference_path.empty())
    
    def on_data_received(self, data: Data):
        """Process incoming reference path data."""
        if data.has("reference_path") and data.reference_path is not None:
            LOG_DEBUG("Received reference path for contouring objective")
            self._process_reference_path(data)
    
    def _process_reference_path(self, data: Data):
        """Process and store reference path data."""
        self.reference_path = data.reference_path
        
        # Create path spline for smooth interpolation
        if len(self.reference_path.x) > 1:
            # Create cumulative arc length parameter
            dx = np.diff(self.reference_path.x)
            dy = np.diff(self.reference_path.y)
            ds = np.sqrt(dx**2 + dy**2)
            s = np.concatenate([[0], np.cumsum(ds)])
            s = s / s[-1]  # Normalize to [0, 1]
            
            # Create splines for x and y coordinates
            from scipy.interpolate import CubicSpline
            self.path_spline = {
                'x': CubicSpline(s, self.reference_path.x),
                'y': CubicSpline(s, self.reference_path.y),
                's': s
            }
    
    def define_parameters(self, parameter_manager):
        """Define parameters for contouring objective."""
        LOG_DEBUG("Defining parameters for contouring objective")
        
        # Core weights
        parameter_manager.add("contour_weight", add_to_rqt_reconfigure=True)
        parameter_manager.add("lag_weight", add_to_rqt_reconfigure=True)
        parameter_manager.add("velocity_weight", add_to_rqt_reconfigure=True)
        parameter_manager.add("progress_weight", add_to_rqt_reconfigure=True)
        
        # Path parameters for each segment
        for i in range(self.num_segments + 1):
            parameter_manager.add(f"path_{i}_start")
            parameter_manager.add(f"path_x_{i}_a")
            parameter_manager.add(f"path_x_{i}_b")
            parameter_manager.add(f"path_x_{i}_c")
            parameter_manager.add(f"path_x_{i}_d")
            parameter_manager.add(f"path_y_{i}_a")
            parameter_manager.add(f"path_y_{i}_b")
            parameter_manager.add(f"path_y_{i}_c")
            parameter_manager.add(f"path_y_{i}_d")
            
            # Derivatives for normal vectors
            parameter_manager.add(f"path_dx_{i}_a")
            parameter_manager.add(f"path_dx_{i}_b")
            parameter_manager.add(f"path_dx_{i}_c")
            parameter_manager.add(f"path_dx_{i}_d")
            parameter_manager.add(f"path_dy_{i}_a")
            parameter_manager.add(f"path_dy_{i}_b")
            parameter_manager.add(f"path_dy_{i}_c")
            parameter_manager.add(f"path_dy_{i}_d")
            
            # Velocity reference if enabled
            if self.dynamic_velocity_reference:
                parameter_manager.add(f"path_vel_{i}_a")
                parameter_manager.add(f"path_vel_{i}_b")
                parameter_manager.add(f"path_vel_{i}_c")
                parameter_manager.add(f"path_vel_{i}_d")
    
    def set_parameters(self, parameter_manager, data: Data, k: int):
        """Set parameter values for current step."""
        if k == 0:
            # Set weights
            parameter_manager.set_parameter("contour_weight", self.contour_weight)
            parameter_manager.set_parameter("lag_weight", self.lag_weight)
            parameter_manager.set_parameter("velocity_weight", self.velocity_weight)
            parameter_manager.set_parameter("progress_weight", self.progress_weight)
            
            # Process reference path and set path parameters
            if self.reference_path is not None:
                self._set_path_parameters(parameter_manager)
    
    def _set_path_parameters(self, parameter_manager):
        """Set path spline parameters."""
        if self.reference_path is None or len(self.reference_path.x) < 2:
            LOG_WARN("No reference path available for parameter setting")
            return
        
        try:
            # Fit cubic splines to path data
            path_x = np.array(self.reference_path.x)
            path_y = np.array(self.reference_path.y)
            
            # Create arc length parameter
            dx = np.diff(path_x)
            dy = np.diff(path_y)
            ds = np.sqrt(dx**2 + dy**2)
            s = np.concatenate([[0], np.cumsum(ds)])
            s = s / s[-1]  # Normalize to [0, 1]
            
            # Resample path for segments
            s_new = np.linspace(0, 1, self.num_segments + 1)
            x_new = np.interp(s_new, s, path_x)
            y_new = np.interp(s_new, s, path_y)
            
            # Fit cubic spline coefficients
            for i in range(self.num_segments):
                # Get segment data
                s_start = s_new[i]
                s_end = s_new[i + 1]
                x_start = x_new[i]
                x_end = x_new[i + 1]
                y_start = y_new[i]
                y_end = y_new[i + 1]
                
                # Estimate derivatives
                if i == 0:
                    dx_start = (x_new[i + 1] - x_new[i]) / (s_new[i + 1] - s_new[i]) if len(x_new) > 1 else 0
                    dy_start = (y_new[i + 1] - y_new[i]) / (s_new[i + 1] - s_new[i]) if len(y_new) > 1 else 0
                else:
                    dx_start = (x_new[i + 1] - x_new[i - 1]) / (s_new[i + 1] - s_new[i - 1])
                    dy_start = (y_new[i + 1] - y_new[i - 1]) / (s_new[i + 1] - s_new[i - 1])
                
                if i == self.num_segments - 1:
                    dx_end = (x_new[i + 1] - x_new[i]) / (s_new[i + 1] - s_new[i]) if len(x_new) > 1 else 0
                    dy_end = (y_new[i + 1] - y_new[i]) / (s_new[i + 1] - s_new[i]) if len(y_new) > 1 else 0
                else:
                    dx_end = (x_new[i + 2] - x_new[i]) / (s_new[i + 2] - s_new[i])
                    dy_end = (y_new[i + 2] - y_new[i]) / (s_new[i + 2] - s_new[i])
                
                # Scale derivatives by segment length
                segment_length = s_end - s_start
                dx_start *= segment_length
                dy_start *= segment_length
                dx_end *= segment_length
                dy_end *= segment_length
                
                # Solve for cubic coefficients: y = a*t^3 + b*t^2 + c*t + d
                # Conditions: y(0) = y_start, y(1) = y_end, y'(0) = dy_start, y'(1) = dy_end
                d_x = x_start
                c_x = dx_start
                a_x = 2 * x_start - 2 * x_end + dx_start + dx_end
                b_x = -3 * x_start + 3 * x_end - 2 * dx_start - dx_end
                
                d_y = y_start
                c_y = dy_start
                a_y = 2 * y_start - 2 * y_end + dy_start + dy_end
                b_y = -3 * y_start + 3 * y_end - 2 * dy_start - dy_end
                
                # Set parameters
                parameter_manager.set_parameter(f"path_{i}_start", float(s_start))
                parameter_manager.set_parameter(f"path_x_{i}_a", float(a_x))
                parameter_manager.set_parameter(f"path_x_{i}_b", float(b_x))
                parameter_manager.set_parameter(f"path_x_{i}_c", float(c_x))
                parameter_manager.set_parameter(f"path_x_{i}_d", float(d_x))
                parameter_manager.set_parameter(f"path_y_{i}_a", float(a_y))
                parameter_manager.set_parameter(f"path_y_{i}_b", float(b_y))
                parameter_manager.set_parameter(f"path_y_{i}_c", float(c_y))
                parameter_manager.set_parameter(f"path_y_{i}_d", float(d_y))
                
                # Set derivative parameters
                parameter_manager.set_parameter(f"path_dx_{i}_a", float(3 * a_x))
                parameter_manager.set_parameter(f"path_dx_{i}_b", float(2 * b_x))
                parameter_manager.set_parameter(f"path_dx_{i}_c", float(c_x))
                parameter_manager.set_parameter(f"path_dx_{i}_d", float(0))
                parameter_manager.set_parameter(f"path_dy_{i}_a", float(3 * a_y))
                parameter_manager.set_parameter(f"path_dy_{i}_b", float(2 * b_y))
                parameter_manager.set_parameter(f"path_dy_{i}_c", float(c_y))
                parameter_manager.set_parameter(f"path_dy_{i}_d", float(0))
                
                # Set velocity reference if enabled
                if self.dynamic_velocity_reference:
                    # Default velocity (could be improved with actual velocity profile)
                    default_vel = 1.0
                    parameter_manager.set_parameter(f"path_vel_{i}_a", 0.0)
                    parameter_manager.set_parameter(f"path_vel_{i}_b", 0.0)
                    parameter_manager.set_parameter(f"path_vel_{i}_c", 0.0)
                    parameter_manager.set_parameter(f"path_vel_{i}_d", float(default_vel))
        
        except Exception as e:
            LOG_WARN(f"Error setting path parameters: {e}")
    
    def get_value(self, symbolic_state, params, stage_idx):
        """Get objective value for a given stage."""
        if self.reference_path is None:
            return {"contouring_cost": cd.MX(0)}
        
        try:
            # Get vehicle state
            pos_x = symbolic_state.get("x")
            pos_y = symbolic_state.get("y")
            psi = symbolic_state.get("psi")
            v = symbolic_state.get("v")
            s = symbolic_state.get("spline")
            
            # Normalize s to [0, 1] range
            if hasattr(self.reference_path, 's') and len(self.reference_path.s) > 0:
                s_normalized = s / self.reference_path.s[-1]
            else:
                s_normalized = s  # Assume s is already normalized
            
            # Clamp s to valid range
            s_normalized = cd.fmax(0.0, cd.fmin(1.0, s_normalized))
            
            # Get weights
            contour_weight = params.get("contour_weight")
            lag_weight = params.get("lag_weight")
            velocity_weight = params.get("velocity_weight")
            progress_weight = params.get("progress_weight")
            
            # Create path spline
            path_spline = Spline2D(params, self.num_segments, s_normalized)
            
            # Get path coordinates and derivatives
            path_x, path_y = path_spline.at(s_normalized)
            path_dx, path_dy = path_spline.deriv_normalized(s_normalized)
            
            # Calculate contouring and lag errors
            dx = pos_x - path_x
            dy = pos_y - path_y
            
            # Contouring error (lateral deviation)
            contour_error = path_dy * dx - path_dx * dy
            
            # Lag error (longitudinal deviation)
            lag_error = path_dx * dx + path_dy * dy
            
            # Progress cost (maximize progress along path)
            progress_cost = -progress_weight * s_normalized
            
            # Velocity cost (if enabled)
            velocity_cost = cd.MX(0)
            if self.dynamic_velocity_reference:
                try:
                    velocity_spline = Spline(params, "path_vel", self.num_segments, s_normalized)
                    reference_velocity = velocity_spline.at(s_normalized)
                    velocity_error = v - reference_velocity
                    velocity_cost = velocity_weight * velocity_error * velocity_error
                except:
                    pass  # Skip velocity cost if not available
            
            # Terminal cost (if at end of horizon)
            terminal_cost = cd.MX(0)
            if stage_idx == self.solver.horizon - 1:
                # Add terminal cost to encourage reaching the end
                remaining_progress = 1.0 - s_normalized
                terminal_cost = 10.0 * remaining_progress * remaining_progress
            
            # Combine costs
            total_cost = (contour_weight * contour_error * contour_error +
                         lag_weight * lag_error * lag_error +
                         progress_cost +
                         velocity_cost +
                         terminal_cost)
            
            return {
                "contouring_cost": total_cost,
                "contour_error": contour_error,
                "lag_error": lag_error,
                "progress": s_normalized
            }
            
        except Exception as e:
            LOG_WARN(f"Error calculating contouring objective: {e}")
            return {"contouring_cost": cd.MX(0)}
    
    def is_objective_reached(self, state: State, data: Data) -> bool:
        """Check if contouring objective is reached (end of path)."""
        if self.reference_path is None:
            return False
        
        # Check if we're close to the end of the path
        s = state.get("spline", 0)
        if hasattr(self.reference_path, 's') and len(self.reference_path.s) > 0:
            s_normalized = s / self.reference_path.s[-1]
        else:
            s_normalized = s
        
        # Consider objective reached if we're within 5% of the end
        return s_normalized >= 0.95
    
    def get_visualization_overlay(self):
        """Return path visualization overlay."""
        if self.reference_path is None:
            return None
        
        try:
            return {
                'lines': [{
                    'x': self.reference_path.x,
                    'y': self.reference_path.y,
                    'color': 'blue',
                    'linewidth': 2,
                    'alpha': 0.8,
                    'label': 'Reference Path'
                }]
            }
        except Exception as e:
            LOG_WARN(f"Error creating visualization overlay: {e}")
            return None