"""
Scenario Constraints Implementation for MPC

This module implements scenario-based constraints that account for multi-modal
uncertainty in obstacle motion, similar to the C++ scenario_module library.

Based on Oscar de Groot's Safe Horizon MPC approach.
"""

import numpy as np
import casadi as cd
from typing import List, Tuple, Optional, Dict
import logging

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planning.src.types import Data, State, DynamicObstacle, PredictionType
from utils.utils import LOG_DEBUG, LOG_WARN, LOG_INFO

LOG = logging.getLogger(__name__)


class ScenarioConstraints(BaseConstraint):
    """
    Scenario constraints implementation for multi-modal uncertainty handling.
    
    This implementation:
    1. Handles multiple scenarios for obstacle motion
    2. Implements Safe Horizon MPC approach
    3. Provides robust constraint generation
    4. Supports both deterministic and probabilistic obstacles
    """
    
    def __init__(self, solver):
        super().__init__(solver)
        self.name = "scenario_constraints"
        
        LOG_DEBUG("Initializing Scenario Constraints")
        
        # Configuration
        self.num_scenarios = self.get_config_value("scenario_constraints.num_scenarios", 3)
        self.max_obstacles_per_scenario = self.get_config_value("scenario_constraints.max_obstacles", 5)
        self.safety_margin = self.get_config_value("scenario_constraints.safety_margin", 1.0)
        self.use_slack = self.get_config_value("scenario_constraints.use_slack", True)
        self.slack_penalty_weight = self.get_config_value("scenario_constraints.slack_penalty_weight", 100.0)
        
        # Storage for scenario data
        self.scenario_data = {}
        self.active_scenarios = 0
        
        # Calculate total constraints per stage
        self.constraints_per_scenario = self.max_obstacles_per_scenario * 2  # 2 constraints per obstacle
        self.total_constraints = self.num_scenarios * self.constraints_per_scenario
        
        LOG_INFO(f"Initialized Scenario Constraints with {self.num_scenarios} scenarios")
    
    def is_data_ready(self, data: Data) -> bool:
        """Check if data is ready for scenario constraint generation."""
        # Always ready - we'll handle empty obstacle lists gracefully
        return True
    
    def on_data_received(self, data: Data):
        """Process incoming data and generate scenarios."""
        LOG_DEBUG("Processing data for scenario constraints")
        
        # Store the data
        self.data = data
        
        # Generate scenarios from obstacle data
        self._generate_scenarios(data)
    
    def _generate_scenarios(self, data: Data):
        """Generate multiple scenarios from obstacle predictions."""
        self.scenario_data = {}
        self.active_scenarios = 0
        
        if not hasattr(data, 'dynamic_obstacles') or not data.dynamic_obstacles:
            LOG_DEBUG("No dynamic obstacles available for scenario generation")
            return
        
        # Generate scenarios based on obstacle predictions
        for scenario_idx in range(self.num_scenarios):
            scenario_obstacles = []
            
            for obs in data.dynamic_obstacles:
                if len(scenario_obstacles) >= self.max_obstacles_per_scenario:
                    break
                
                # Create scenario-specific obstacle prediction
                scenario_obs = self._create_scenario_obstacle(obs, scenario_idx)
                if scenario_obs:
                    scenario_obstacles.append(scenario_obs)
            
            if scenario_obstacles:
                self.scenario_data[scenario_idx] = scenario_obstacles
                self.active_scenarios += 1
        
        LOG_DEBUG(f"Generated {self.active_scenarios} active scenarios")
    
    def _create_scenario_obstacle(self, obstacle: DynamicObstacle, scenario_idx: int) -> Optional[Dict]:
        """Create a scenario-specific obstacle prediction."""
        try:
            if not hasattr(obstacle, 'prediction') or obstacle.prediction is None:
                return None
            
            # Get base prediction
            if obstacle.prediction.type == PredictionType.GAUSSIAN:
                # For Gaussian predictions, create different scenarios by sampling
                return self._create_gaussian_scenario(obstacle, scenario_idx)
            elif obstacle.prediction.type == PredictionType.DETERMINISTIC:
                # For deterministic predictions, add small variations
                return self._create_deterministic_scenario(obstacle, scenario_idx)
            else:
                return None
                
        except Exception as e:
            LOG_WARN(f"Error creating scenario obstacle: {e}")
            return None
    
    def _create_gaussian_scenario(self, obstacle: DynamicObstacle, scenario_idx: int) -> Dict:
        """Create scenario from Gaussian prediction."""
        scenario = {
            'position': obstacle.position.copy(),
            'velocity': getattr(obstacle, 'velocity', [0.0, 0.0]).copy(),
            'radius': getattr(obstacle, 'radius', 0.5),
            'uncertainty': [0.1, 0.1],  # Default uncertainty
            'scenario_id': scenario_idx
        }
        
        # Add scenario-specific variations
        if hasattr(obstacle, 'prediction') and obstacle.prediction.steps:
            # Use different steps for different scenarios
            step_idx = min(scenario_idx, len(obstacle.prediction.steps) - 1)
            step = obstacle.prediction.steps[step_idx]
            
            scenario['position'] = step.position.copy()
            if hasattr(step, 'velocity'):
                scenario['velocity'] = step.velocity.copy()
            if hasattr(step, 'major_radius'):
                scenario['uncertainty'][0] = step.major_radius
            if hasattr(step, 'minor_radius'):
                scenario['uncertainty'][1] = step.minor_radius
        
        return scenario
    
    def _create_deterministic_scenario(self, obstacle: DynamicObstacle, scenario_idx: int) -> Dict:
        """Create scenario from deterministic prediction with variations."""
        scenario = {
            'position': obstacle.position.copy(),
            'velocity': getattr(obstacle, 'velocity', [0.0, 0.0]).copy(),
            'radius': getattr(obstacle, 'radius', 0.5),
            'uncertainty': [0.05, 0.05],  # Small uncertainty for deterministic
            'scenario_id': scenario_idx
        }
        
        # Add small variations based on scenario index
        variation_factor = 0.1 * (scenario_idx + 1)
        scenario['position'][0] += variation_factor * np.sin(scenario_idx)
        scenario['position'][1] += variation_factor * np.cos(scenario_idx)
        
        return scenario
    
    def define_parameters(self, parameter_manager):
        """Define symbolic parameters for scenario constraints."""
        LOG_DEBUG(f"Defining parameters for {self.name}")
        
        # Define parameters for each scenario and obstacle
        for scenario_idx in range(self.num_scenarios):
            for obs_idx in range(self.max_obstacles_per_scenario):
                for stage_idx in range(self.solver.horizon + 1):
                    # Position parameters
                    parameter_manager.add(f"scenario_{scenario_idx}_obs_{obs_idx}_x_step_{stage_idx}")
                    parameter_manager.add(f"scenario_{scenario_idx}_obs_{obs_idx}_y_step_{stage_idx}")
                    
                    # Uncertainty parameters
                    parameter_manager.add(f"scenario_{scenario_idx}_obs_{obs_idx}_major_step_{stage_idx}")
                    parameter_manager.add(f"scenario_{scenario_idx}_obs_{obs_idx}_minor_step_{stage_idx}")
                    
                    # Radius parameter
                    parameter_manager.add(f"scenario_{scenario_idx}_obs_{obs_idx}_radius_step_{stage_idx}")
                    
                    # Slack variables if enabled
                    if self.use_slack:
                        parameter_manager.add(f"scenario_{scenario_idx}_obs_{obs_idx}_slack_step_{stage_idx}")
    
    def set_parameters(self, parameter_manager, data: Data, step: int):
        """Set parameter values for current step."""
        LOG_DEBUG(f"Setting parameters for step {step}")
        
        # Set parameters for all scenarios
        for scenario_idx in range(self.num_scenarios):
            if scenario_idx in self.scenario_data:
                scenario_obstacles = self.scenario_data[scenario_idx]
            else:
                scenario_obstacles = []
            
            for obs_idx in range(self.max_obstacles_per_scenario):
                if obs_idx < len(scenario_obstacles):
                    # Set real obstacle data
                    obs = scenario_obstacles[obs_idx]
                    
                    # Project obstacle position forward in time
                    dt = self.solver.timestep if hasattr(self.solver, 'timestep') else 0.1
                    future_time = step * dt
                    
                    # Simple linear projection
                    future_pos = [
                        obs['position'][0] + obs['velocity'][0] * future_time,
                        obs['position'][1] + obs['velocity'][1] * future_time
                    ]
                    
                    parameter_manager.set_parameter(f"scenario_{scenario_idx}_obs_{obs_idx}_x_step_{step}", future_pos[0])
                    parameter_manager.set_parameter(f"scenario_{scenario_idx}_obs_{obs_idx}_y_step_{step}", future_pos[1])
                    parameter_manager.set_parameter(f"scenario_{scenario_idx}_obs_{obs_idx}_major_step_{step}", obs['uncertainty'][0])
                    parameter_manager.set_parameter(f"scenario_{scenario_idx}_obs_{obs_idx}_minor_step_{step}", obs['uncertainty'][1])
                    parameter_manager.set_parameter(f"scenario_{scenario_idx}_obs_{obs_idx}_radius_step_{step}", obs['radius'])
                    
                    # Set slack parameters if enabled
                    if self.use_slack:
                        parameter_manager.set_parameter(f"scenario_{scenario_idx}_obs_{obs_idx}_slack_step_{step}", 0.0)
                else:
                    # Set dummy values for inactive obstacles
                    parameter_manager.set_parameter(f"scenario_{scenario_idx}_obs_{obs_idx}_x_step_{step}", 1000.0)
                    parameter_manager.set_parameter(f"scenario_{scenario_idx}_obs_{obs_idx}_y_step_{step}", 1000.0)
                    parameter_manager.set_parameter(f"scenario_{scenario_idx}_obs_{obs_idx}_major_step_{step}", 0.1)
                    parameter_manager.set_parameter(f"scenario_{scenario_idx}_obs_{obs_idx}_minor_step_{step}", 0.1)
                    parameter_manager.set_parameter(f"scenario_{scenario_idx}_obs_{obs_idx}_radius_step_{step}", 0.1)
                    
                    if self.use_slack:
                        parameter_manager.set_parameter(f"scenario_{scenario_idx}_obs_{obs_idx}_slack_step_{step}", 0.0)
    
    def get_constraints(self, symbolic_state, params, stage_idx):
        """Generate symbolic constraints for a given stage."""
        if stage_idx == 0:
            return []
        
        constraints = []
        
        # Get vehicle position
        pos_x = symbolic_state.get("x")
        pos_y = symbolic_state.get("y")
        
        # Add constraints for each scenario
        for scenario_idx in range(self.num_scenarios):
            for obs_idx in range(self.max_obstacles_per_scenario):
                try:
                    # Get obstacle parameters
                    obs_x = params.get(f"scenario_{scenario_idx}_obs_{obs_idx}_x_step_{stage_idx}")
                    obs_y = params.get(f"scenario_{scenario_idx}_obs_{obs_idx}_y_step_{stage_idx}")
                    major_radius = params.get(f"scenario_{scenario_idx}_obs_{obs_idx}_major_step_{stage_idx}")
                    minor_radius = params.get(f"scenario_{scenario_idx}_obs_{obs_idx}_minor_step_{stage_idx}")
                    obs_radius = params.get(f"scenario_{scenario_idx}_obs_{obs_idx}_radius_step_{stage_idx}")
                    
                    # Skip dummy obstacles (far away)
                    if obs_x > 500.0 or obs_y > 500.0:
                        continue
                    
                    # Calculate distance to obstacle
                    dx = pos_x - obs_x
                    dy = pos_y - obs_y
                    distance = cd.sqrt(dx**2 + dy**2)
                    
                    # Calculate required separation (obstacle radius + safety margin)
                    required_separation = obs_radius + self.safety_margin
                    
                    # Create distance constraint: distance >= required_separation
                    constraint = distance - required_separation
                    
                    # Add slack if enabled
                    if self.use_slack:
                        slack = params.get(f"scenario_{scenario_idx}_obs_{obs_idx}_slack_step_{stage_idx}")
                        constraint += slack
                    
                    constraints.append(constraint)
                    
                except Exception as e:
                    LOG_WARN(f"Error creating constraint for scenario {scenario_idx}, obstacle {obs_idx}: {e}")
                    continue
        
        return constraints
    
    def get_lower_bound(self):
        """Get lower bounds for constraints."""
        lower_bounds = []
        
        for scenario_idx in range(self.num_scenarios):
            for obs_idx in range(self.max_obstacles_per_scenario):
                # Distance constraint: distance >= required_separation
                # Lower bound is 0 (distance can't be negative)
                lower_bounds.append(0.0)
        
        return lower_bounds
    
    def get_upper_bound(self):
        """Get upper bounds for constraints."""
        upper_bounds = []
        
        for scenario_idx in range(self.num_scenarios):
            for obs_idx in range(self.max_obstacles_per_scenario):
                # Distance constraint: distance >= required_separation
                # Upper bound is infinity (no upper limit on distance)
                upper_bounds.append(cd.inf)
        
        return upper_bounds
    
    def get_penalty(self, symbolic_state, params, stage_idx):
        """Get penalty terms for slack variables."""
        if not self.use_slack:
            return cd.MX(0)
        
        penalty = cd.MX(0)
        
        for scenario_idx in range(self.num_scenarios):
            for obs_idx in range(self.max_obstacles_per_scenario):
                try:
                    slack = params.get(f"scenario_{scenario_idx}_obs_{obs_idx}_slack_step_{stage_idx}")
                    penalty += self.slack_penalty_weight * slack * slack
                except Exception as e:
                    LOG_WARN(f"Error creating penalty for slack variable: {e}")
                    continue
        
        return penalty
    
    def get_visualization_overlay(self):
        """Return scenario obstacles as visualization overlay."""
        if not self.scenario_data:
            return None
        
        overlays = {
            'points': [],
            'polygons': []
        }
        
        try:
            for scenario_idx, scenario_obstacles in self.scenario_data.items():
                for obs in scenario_obstacles:
                    # Add obstacle as point
                    overlays['points'].append({
                        'x': obs['position'][0],
                        'y': obs['position'][1],
                        'color': f'C{scenario_idx}',
                        'marker': 'o',
                        'size': obs['radius'] * 20,
                        'alpha': 0.7
                    })
                    
                    # Add uncertainty ellipse as polygon
                    if obs['uncertainty'][0] > 0.1 or obs['uncertainty'][1] > 0.1:
                        ellipse_points = self._create_ellipse_points(
                            obs['position'], 
                            obs['uncertainty'][0], 
                            obs['uncertainty'][1]
                        )
                        overlays['polygons'].append({
                            'x': ellipse_points[:, 0].tolist(),
                            'y': ellipse_points[:, 1].tolist(),
                            'color': f'C{scenario_idx}',
                            'alpha': 0.3
                        })
        except Exception as e:
            LOG_WARN(f"Error creating visualization overlay: {e}")
            return None
        
        return overlays
    
    def _create_ellipse_points(self, center, major_radius, minor_radius, num_points=32):
        """Create points for uncertainty ellipse visualization."""
        angles = np.linspace(0, 2*np.pi, num_points)
        x = center[0] + major_radius * np.cos(angles)
        y = center[1] + minor_radius * np.sin(angles)
        return np.column_stack([x, y])