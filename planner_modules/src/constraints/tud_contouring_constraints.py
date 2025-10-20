#!/usr/bin/env python3
"""
TUD-AMR Style Contouring Constraints

This implements contouring constraints following the TUD-AMR MPC planner approach.
Key features:
- Path following with lateral deviation constraints
- Road boundary enforcement
- Proper constraint formulation for MPC
"""

import numpy as np
import casadi as cd
from scipy.interpolate import CubicSpline

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planning.src.types import Data, ReferencePath
from utils.math_utils import haar_difference_without_abs, Spline2D, Spline
from utils.utils import LOG_DEBUG, LOG_INFO, LOG_WARN


class TUDContouringConstraints(BaseConstraint):
    """TUD-AMR style contouring constraints for path following."""
    
    def __init__(self, solver):
        super().__init__(solver)
        
        self.name = "tud_contouring_constraints"
        self.num_segments = self.get_config_value("contouring.num_segments", 10)
        self.slack = self.get_config_value("contouring.slack", 0.1)
        self.num_constraints = 2  # Left and right boundary constraints
        
        # Road bounds
        self.reference_path = None
        self.bound_left_spline = None
        self.bound_right_spline = None
        
        LOG_DEBUG(f"{self.name.title()} successfully initialized")
    
    def get_visualization_overlay(self):
        """Return road corridor visualization data."""
        try:
            if self.reference_path is None:
                return None
            
            # Build road corridor visualization
            path_x = np.array(self.reference_path.x)
            path_y = np.array(self.reference_path.y)
            
            if path_x.size < 2:
                return None
            
            # Create road corridor polygon
            if self.bound_left_spline is not None and self.bound_right_spline is not None:
                # Use actual road boundaries
                s_vals = np.linspace(0, 1, 100)
                left_pts = self.bound_left_spline(s_vals)
                right_pts = self.bound_right_spline(s_vals)
                
                # Create corridor polygon
                corridor_x = np.concatenate([left_pts[:, 0], right_pts[::-1, 0]])
                corridor_y = np.concatenate([left_pts[:, 1], right_pts[::-1, 1]])
                
                return {
                    'polygons': [{
                        'x': corridor_x.tolist(),
                        'y': corridor_y.tolist(),
                        'color': '#99cc99',
                        'alpha': 0.2,
                        'label': 'Road Corridor'
                    }]
                }
            else:
                # Use default width
                road_width = 6.0
                dx = np.gradient(path_x)
                dy = np.gradient(path_y)
                normals_x = -dy / (np.sqrt(dx**2 + dy**2) + 1e-9)
                normals_y = dx / (np.sqrt(dx**2 + dy**2) + 1e-9)
                
                left_x = path_x + normals_x * road_width / 2
                left_y = path_y + normals_y * road_width / 2
                right_x = path_x - normals_x * road_width / 2
                right_y = path_y - normals_y * road_width / 2
                
                corridor_x = np.concatenate([left_x, right_x[::-1]])
                corridor_y = np.concatenate([left_y, right_y[::-1]])
                
                return {
                    'polygons': [{
                        'x': corridor_x.tolist(),
                        'y': corridor_y.tolist(),
                        'color': '#99cc99',
                        'alpha': 0.2,
                        'label': 'Road Corridor'
                    }]
                }
        except Exception:
            return None
    
    def update(self, state, data: Data):
        """Update constraint state."""
        LOG_DEBUG(f"{self.name.title()}::update")
        
        if self.reference_path is None:
            LOG_WARN("No reference path available")
            return
        
        # Pass path data to data object
        if self.reference_path is not None:
            data.reference_path = ReferencePath()
            data.reference_path.set('x', self.reference_path.x)
            data.reference_path.set('y', self.reference_path.y)
            data.reference_path.set('s', self.reference_path.s)
    
    def on_data_received(self, data):
        """Process received data."""
        if data.has("reference_path") and data.reference_path is not None:
            LOG_DEBUG("Received Reference Path")
            self.process_reference_path(data)
    
    def process_reference_path(self, data):
        """Process reference path and road boundaries."""
        self.reference_path = data.reference_path
        
        # Process road bounds if available
        if data.left_bound is not None and data.right_bound is not None:
            LOG_DEBUG("Processing road boundaries for TUD contouring constraints")
            self.bound_left_spline = CubicSpline(self.reference_path.s,
                                                np.column_stack((data.left_bound.x, data.left_bound.y)))
            self.bound_right_spline = CubicSpline(self.reference_path.s,
                                                 np.column_stack((data.right_bound.x, data.right_bound.y)))
    
    def define_parameters(self, params):
        """Define parameters for the constraint."""
        LOG_DEBUG(f"{self.name.title()}::define_parameters")
        
        for i in range(self.num_segments):
            # Path coordinates
            params.add(f"path_{i}_start")
            params.add(f"path_x_{i}_a")
            params.add(f"path_x_{i}_b")
            params.add(f"path_x_{i}_c")
            params.add(f"path_x_{i}_d")
            params.add(f"path_y_{i}_a")
            params.add(f"path_y_{i}_b")
            params.add(f"path_y_{i}_c")
            params.add(f"path_y_{i}_d")
            
            # Width parameters
            params.add(f"width_left_{i}_a")
            params.add(f"width_left_{i}_b")
            params.add(f"width_left_{i}_c")
            params.add(f"width_left_{i}_d")
            params.add(f"width_right_{i}_a")
            params.add(f"width_right_{i}_b")
            params.add(f"width_right_{i}_c")
            params.add(f"width_right_{i}_d")
    
    def set_parameters(self, parameter_manager, data, k):
        """Set parameters for the constraint."""
        LOG_DEBUG(f"{self.name.title()}::set_parameters with k: {k}")
        
        if k == 0:
            self.process_reference_path(data)
            self.set_path_parameters(parameter_manager, data)
    
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
    
    def set_path_parameters(self, parameter_manager, data):
        """Set path parameters for the constraint."""
        LOG_DEBUG(f"{self.name.title()}::set_path_parameters")
        
        if self.reference_path is None:
            LOG_WARN("No reference path available")
            return
        
        path_x = self.reference_path.x
        path_y = self.reference_path.y
        
        # Compute width parameters from bounds
        width_left_orig, width_right_orig = self._compute_width_from_bounds(data)
        
        # Prepare interpolated data
        if len(path_x) < self.num_segments + 1:
            s_original = np.linspace(0, 1, len(path_x))
            s_new = np.linspace(0, 1, self.num_segments + 1)
            
            path_x_interp = np.interp(s_new, s_original, path_x)
            path_y_interp = np.interp(s_new, s_original, path_y)
            
            if width_left_orig is not None and width_right_orig is not None:
                width_left_interp = np.interp(s_new, s_original, width_left_orig)
                width_right_interp = np.interp(s_new, s_original, width_right_orig)
            else:
                width_left_interp = np.full(len(s_new), 3.0)
                width_right_interp = np.full(len(s_new), 3.0)
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
            
            if width_left_orig is not None and width_right_orig is not None:
                width_left_interp = np.interp(s_new, s, width_left_orig)
                width_right_interp = np.interp(s_new, s, width_right_orig)
            else:
                width_left_interp = np.full(len(s_new), 3.0)
                width_right_interp = np.full(len(s_new), 3.0)
        
        # Fit cubic spline coefficients
        try:
            path_x_coeffs, x_starts = self._fit_cubic_spline_coefficients(s_new, path_x_interp)
            path_y_coeffs, y_starts = self._fit_cubic_spline_coefficients(s_new, path_y_interp)
            width_left_coeffs, _ = self._fit_cubic_spline_coefficients(s_new, width_left_interp)
            width_right_coeffs, _ = self._fit_cubic_spline_coefficients(s_new, width_right_interp)
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
            
            # Set width coefficients
            parameter_manager.set_parameter(f"width_left_{i}_a", width_left_coeffs[i][0])
            parameter_manager.set_parameter(f"width_left_{i}_b", width_left_coeffs[i][1])
            parameter_manager.set_parameter(f"width_left_{i}_c", width_left_coeffs[i][2])
            parameter_manager.set_parameter(f"width_left_{i}_d", width_left_coeffs[i][3])
            
            parameter_manager.set_parameter(f"width_right_{i}_a", width_right_coeffs[i][0])
            parameter_manager.set_parameter(f"width_right_{i}_b", width_right_coeffs[i][1])
            parameter_manager.set_parameter(f"width_right_{i}_c", width_right_coeffs[i][2])
            parameter_manager.set_parameter(f"width_right_{i}_d", width_right_coeffs[i][3])
    
    def _compute_width_from_bounds(self, data):
        """Compute width from road boundaries."""
        if self.reference_path is None or self.bound_left_spline is None or self.bound_right_spline is None:
            LOG_WARN("Missing reference path or boundary splines")
            return None, None
        
        # Get reference path points
        path_x = self.reference_path.x
        path_y = self.reference_path.y
        
        # Evaluate bound splines
        left_bound_points = self.bound_left_spline(self.reference_path.s)
        right_bound_points = self.bound_right_spline(self.reference_path.s)
        
        # Calculate widths
        width_left = []
        width_right = []
        
        for i in range(len(path_x)):
            left_dist = np.sqrt((path_x[i] - left_bound_points[i, 0]) ** 2 +
                              (path_y[i] - left_bound_points[i, 1]) ** 2)
            width_left.append(left_dist)
            
            right_dist = np.sqrt((path_x[i] - right_bound_points[i, 0]) ** 2 +
                               (path_y[i] - right_bound_points[i, 1]) ** 2)
            width_right.append(right_dist)
        
        return np.array(width_left), np.array(width_right)
    
    def get_lower_bound(self):
        """Get lower bounds for constraints."""
        return [-np.inf, -np.inf]
    
    def get_upper_bound(self):
        """Get upper bounds for constraints."""
        return [0.0, 0.0]
    
    def get_constraints(self, symbolic_state, params, stage_idx):
        """Get constraint expressions."""
        LOG_INFO(f"{self.name}::get_constraints")
        
        pos_x = symbolic_state.get("x")
        pos_y = symbolic_state.get("y")
        psi = symbolic_state.get("psi")
        s = symbolic_state.get("spline") / self.reference_path.s[-1]
        
        # Create spline objects
        path_spline = Spline2D(params, self.num_segments, s)
        width_left_spline = Spline(params, "width_left", self.num_segments, s)
        width_right_spline = Spline(params, "width_right", self.num_segments, s)
        
        # Get path coordinates and derivatives
        path_x, path_y = path_spline.at(s)
        path_dx_normalized, path_dy_normalized = path_spline.deriv_normalized(s)
        
        # Calculate contouring error (lateral deviation)
        dx = pos_x - path_x
        dy = pos_y - path_y
        
        # Normal vector pointing left from path direction
        contour_error = path_dy_normalized * dx - path_dx_normalized * dy
        
        # Evaluate width splines
        width_right = width_right_spline.at(s)
        width_left = width_left_spline.at(s)
        
        # Get vehicle parameters
        vehicle_width = symbolic_state.get("width", 2.0)
        lr = symbolic_state.get("lr", 1.0)
        
        # Calculate effective vehicle width
        path_heading = cd.atan2(path_dy_normalized, path_dx_normalized)
        delta_psi = haar_difference_without_abs(psi, path_heading)
        w_cur = vehicle_width / 2. * cd.cos(delta_psi) + lr * cd.sin(cd.fabs(delta_psi))
        
        # Get slack parameter
        horizon_factor = 1.0 + (stage_idx * 0.1)
        adaptive_slack = self.slack * horizon_factor
        
        # TUD-AMR style constraints: vehicle must stay within road boundaries
        c1 = contour_error + w_cur - width_right - adaptive_slack  # Right boundary
        c2 = -contour_error + w_cur - width_left - adaptive_slack  # Left boundary
        
        LOG_DEBUG(f"TUD contouring constraints - contour_error: {contour_error}, w_cur: {w_cur}")
        LOG_DEBUG(f"Width constraints - left: {width_left}, right: {width_right}")
        
        return [c1, c2]
    
    def is_data_ready(self, data):
        """Check if required data is available."""
        required_fields = ["reference_path"]
        missing_data = ""
        for field in required_fields:
            if not data.has(field) or getattr(data, field) is None:
                missing_data += f"{field.replace('_', ' ').title()} "
        return len(missing_data) < 1
