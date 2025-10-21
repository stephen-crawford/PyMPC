"""
Test configuration system for easy constraint setup and MPC testing.

This module provides a flexible configuration system for setting up MPC tests
with different constraint combinations and objectives.
"""

import json
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class ConstraintType(Enum):
    """Enumeration of constraint types."""
    LINEAR = "linear"
    ELLIPSOID = "ellipsoid"
    GAUSSIAN = "gaussian"
    SCENARIO = "scenario"
    POLYTOPE = "polytope"


class ObjectiveType(Enum):
    """Enumeration of objective types."""
    CONTOURING = "contouring"
    GOAL = "goal"
    TRACKING = "tracking"
    CUSTOM = "custom"


@dataclass
class TestConfig:
    """
    Configuration for MPC tests.
    """
    # Test identification
    test_name: str
    description: str = ""
    
    # MPC parameters
    horizon_length: int = 20
    dt: float = 0.1
    state_dim: int = 5
    control_dim: int = 2
    
    # Dynamics
    dynamics_type: str = "bicycle"
    dynamics_params: Dict[str, Any] = None
    
    # Objectives
    objectives: List[Dict[str, Any]] = None
    
    # Constraints
    constraints: List[Dict[str, Any]] = None
    
    # Initial conditions
    initial_state: List[float] = None
    
    # Reference path
    reference_path: Optional[Dict[str, Any]] = None
    
    # Solver options
    solver_options: Dict[str, Any] = None
    
    # Visualization options
    visualization: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.dynamics_params is None:
            self.dynamics_params = {"wheelbase": 2.5}
        
        if self.objectives is None:
            self.objectives = []
        
        if self.constraints is None:
            self.constraints = []
        
        if self.initial_state is None:
            self.initial_state = [0.0, 0.0, 0.0, 1.0, 0.0]
        
        if self.solver_options is None:
            self.solver_options = {
                "ipopt": {
                    "print_level": 0,
                    "sb": "yes",
                    "max_iter": 100
                },
                "print_time": False
            }
        
        if self.visualization is None:
            self.visualization = {
                "save_plots": True,
                "show_plots": False,
                "create_animation": False
            }


class TestConfigBuilder:
    """
    Builder class for creating test configurations.
    """
    
    def __init__(self, test_name: str):
        """
        Initialize test configuration builder.
        
        Args:
            test_name: Name of the test
        """
        self.config = TestConfig(test_name=test_name)
    
    def set_mpc_params(self, 
                      horizon_length: int = 20,
                      dt: float = 0.1,
                      state_dim: int = 5,
                      control_dim: int = 2) -> 'TestConfigBuilder':
        """Set MPC parameters."""
        self.config.horizon_length = horizon_length
        self.config.dt = dt
        self.config.state_dim = state_dim
        self.config.control_dim = control_dim
        return self
    
    def set_dynamics(self, 
                    dynamics_type: str = "bicycle",
                    **params) -> 'TestConfigBuilder':
        """Set dynamics parameters."""
        self.config.dynamics_type = dynamics_type
        self.config.dynamics_params = params
        return self
    
    def add_contouring_objective(self, 
                               reference_path: np.ndarray,
                               progress_weight: float = 1.0,
                               contouring_weight: float = 10.0,
                               control_weight: float = 0.1) -> 'TestConfigBuilder':
        """Add contouring objective."""
        objective = {
            "type": "contouring",
            "reference_path": reference_path.tolist(),
            "progress_weight": progress_weight,
            "contouring_weight": contouring_weight,
            "control_weight": control_weight
        }
        self.config.objectives.append(objective)
        return self
    
    def add_goal_objective(self, 
                         goal_state: np.ndarray,
                         goal_weight: float = 1.0,
                         control_weight: float = 0.1,
                         terminal_weight: float = 10.0) -> 'TestConfigBuilder':
        """Add goal objective."""
        objective = {
            "type": "goal",
            "goal_state": goal_state.tolist(),
            "goal_weight": goal_weight,
            "control_weight": control_weight,
            "terminal_weight": terminal_weight
        }
        self.config.objectives.append(objective)
        return self
    
    def add_linear_constraints(self, 
                             state_bounds: Optional[Dict[str, List[float]]] = None,
                             control_bounds: Optional[Dict[str, List[float]]] = None,
                             custom_constraints: Optional[List[Dict]] = None) -> 'TestConfigBuilder':
        """Add linear constraints."""
        constraint = {
            "type": "linear",
            "state_bounds": state_bounds,
            "control_bounds": control_bounds,
            "custom_constraints": custom_constraints
        }
        self.config.constraints.append(constraint)
        return self
    
    def add_ellipsoid_constraints(self, 
                                obstacles: List[Dict[str, Any]],
                                safety_margin: float = 0.3) -> 'TestConfigBuilder':
        """Add ellipsoid constraints."""
        constraint = {
            "type": "ellipsoid",
            "obstacles": obstacles,
            "safety_margin": safety_margin
        }
        self.config.constraints.append(constraint)
        return self
    
    def add_gaussian_constraints(self, 
                               uncertain_obstacles: List[Dict[str, Any]],
                               confidence_level: float = 0.95,
                               safety_margin: float = 0.2) -> 'TestConfigBuilder':
        """Add Gaussian constraints."""
        constraint = {
            "type": "gaussian",
            "uncertain_obstacles": uncertain_obstacles,
            "confidence_level": confidence_level,
            "safety_margin": safety_margin
        }
        self.config.constraints.append(constraint)
        return self
    
    def add_scenario_constraints(self, 
                               scenarios: List[Dict[str, Any]],
                               scenario_weights: Optional[List[float]] = None) -> 'TestConfigBuilder':
        """Add scenario constraints."""
        constraint = {
            "type": "scenario",
            "scenarios": scenarios,
            "scenario_weights": scenario_weights
        }
        self.config.constraints.append(constraint)
        return self
    
    def set_initial_state(self, initial_state: List[float]) -> 'TestConfigBuilder':
        """Set initial state."""
        self.config.initial_state = initial_state
        return self
    
    def set_reference_path(self, 
                          path_type: str = "curved",
                          **params) -> 'TestConfigBuilder':
        """Set reference path."""
        self.config.reference_path = {
            "type": path_type,
            "params": params
        }
        return self
    
    def set_solver_options(self, **options) -> 'TestConfigBuilder':
        """Set solver options."""
        self.config.solver_options.update(options)
        return self
    
    def set_visualization_options(self, **options) -> 'TestConfigBuilder':
        """Set visualization options."""
        self.config.visualization.update(options)
        return self
    
    def build(self) -> TestConfig:
        """Build the test configuration."""
        return self.config


class TestConfigManager:
    """
    Manager for test configurations.
    """
    
    def __init__(self, config_dir: str = "test_configs"):
        """
        Initialize test configuration manager.
        
        Args:
            config_dir: Directory for configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def save_config(self, config: TestConfig, filename: Optional[str] = None) -> Path:
        """
        Save test configuration to file.
        
        Args:
            config: Test configuration
            filename: Optional filename (defaults to test_name.json)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"{config.test_name}.json"
        
        filepath = self.config_dir / filename
        
        # Convert to dictionary and save
        config_dict = asdict(config)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        return filepath
    
    def load_config(self, filename: str) -> TestConfig:
        """
        Load test configuration from file.
        
        Args:
            filename: Configuration filename
            
        Returns:
            Test configuration
        """
        filepath = self.config_dir / filename
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return TestConfig(**config_dict)
    
    def list_configs(self) -> List[str]:
        """List available configuration files."""
        return [f.name for f in self.config_dir.glob("*.json")]
    
    def delete_config(self, filename: str) -> bool:
        """
        Delete configuration file.
        
        Args:
            filename: Configuration filename
            
        Returns:
            True if deleted, False if not found
        """
        filepath = self.config_dir / filename
        if filepath.exists():
            filepath.unlink()
            return True
        return False


class PredefinedTestConfigs:
    """
    Predefined test configurations for common scenarios.
    """
    
    @staticmethod
    def curving_road_ellipsoid() -> TestConfig:
        """Curving road with ellipsoid obstacles."""
        builder = TestConfigBuilder("curving_road_ellipsoid")
        
        # Create curving road
        t = np.linspace(0, 6*np.pi, 100)
        x = 0.2 * t
        y = 2 * np.sin(0.2 * t) + 0.5 * np.sin(0.6 * t)
        reference_path = np.column_stack([x, y])
        
        # Create obstacles
        obstacles = []
        for i in range(25):
            x_obs = 1.0 + 0.3 * i
            y_obs = 1.0 + 0.8 * np.sin(0.1 * i)
            obstacles.append({
                'center': [x_obs, y_obs],
                'shape': [0.6, 0.3],
                'rotation': 0.0
            })
        
        return (builder
                .set_mpc_params(horizon_length=25, dt=0.1)
                .add_contouring_objective(reference_path)
                .add_linear_constraints(
                    state_bounds={
                        'min': [-30, -15, -np.pi, 0, -0.6],
                        'max': [30, 15, np.pi, 5, 0.6]
                    }
                )
                .add_ellipsoid_constraints(obstacles, safety_margin=0.4)
                .set_initial_state([0.0, 0.0, 0.0, 2.0, 0.0])
                .build())
    
    @staticmethod
    def curving_road_gaussian() -> TestConfig:
        """Curving road with Gaussian obstacles."""
        builder = TestConfigBuilder("curving_road_gaussian")
        
        # Create curving road
        t = np.linspace(0, 6*np.pi, 100)
        x = 0.2 * t
        y = 2 * np.sin(0.2 * t) + 0.5 * np.sin(0.6 * t)
        reference_path = np.column_stack([x, y])
        
        # Create uncertain obstacles
        uncertain_obstacles = []
        for i in range(25):
            x_mean = 2.0 + 0.2 * i
            y_mean = 1.5 + 0.5 * np.sin(0.12 * i)
            uncertainty = 0.1 + 0.02 * i
            uncertain_obstacles.append({
                'mean': [x_mean, y_mean],
                'covariance': [[uncertainty, 0.0], [0.0, uncertainty]],
                'shape': [0.5, 0.3]
            })
        
        return (builder
                .set_mpc_params(horizon_length=25, dt=0.1)
                .add_contouring_objective(reference_path)
                .add_linear_constraints(
                    state_bounds={
                        'min': [-30, -15, -np.pi, 0, -0.6],
                        'max': [30, 15, np.pi, 5, 0.6]
                    }
                )
                .add_gaussian_constraints(uncertain_obstacles, confidence_level=0.95)
                .set_initial_state([0.0, 0.0, 0.0, 2.0, 0.0])
                .build())
    
    @staticmethod
    def curving_road_scenario() -> TestConfig:
        """Curving road with scenario constraints."""
        builder = TestConfigBuilder("curving_road_scenario")
        
        # Create curving road
        t = np.linspace(0, 6*np.pi, 100)
        x = 0.2 * t
        y = 2 * np.sin(0.2 * t) + 0.5 * np.sin(0.6 * t)
        reference_path = np.column_stack([x, y])
        
        # Create scenarios
        scenarios = []
        for scenario_idx in range(2):
            scenario_obstacles = []
            for i in range(25):
                if scenario_idx == 0:  # Aggressive scenario
                    x_obs = 0.5 + 0.35 * i
                    y_obs = 1.5 + 0.6 * np.sin(0.15 * i)
                else:  # Conservative scenario
                    x_obs = 1.5 + 0.2 * i
                    y_obs = 2.5 + 0.3 * np.sin(0.08 * i)
                
                scenario_obstacles.append({
                    'center': [x_obs, y_obs],
                    'shape': [0.7, 0.4]
                })
            
            scenarios.append({
                'obstacles': scenario_obstacles,
                'constraints': [],
                'probability': 0.5
            })
        
        return (builder
                .set_mpc_params(horizon_length=25, dt=0.1)
                .add_contouring_objective(reference_path)
                .add_linear_constraints(
                    state_bounds={
                        'min': [-30, -15, -np.pi, 0, -0.6],
                        'max': [30, 15, np.pi, 5, 0.6]
                    }
                )
                .add_scenario_constraints(scenarios, scenario_weights=[0.5, 0.5])
                .set_initial_state([0.0, 0.0, 0.0, 2.0, 0.0])
                .build())
    
    @staticmethod
    def goal_reaching() -> TestConfig:
        """Goal reaching with obstacles."""
        builder = TestConfigBuilder("goal_reaching")
        
        # Create obstacles
        obstacles = []
        for i in range(25):
            x_obs = 1.0 + 0.3 * i
            y_obs = 1.0 + 0.8 * np.sin(0.1 * i)
            obstacles.append({
                'center': [x_obs, y_obs],
                'shape': [0.6, 0.3],
                'rotation': 0.0
            })
        
        # Goal state
        goal_state = np.array([10.0, 5.0, 0.0, 1.0, 0.0])
        
        return (builder
                .set_mpc_params(horizon_length=25, dt=0.1)
                .add_goal_objective(goal_state, goal_weight=1.0, terminal_weight=10.0)
                .add_linear_constraints(
                    state_bounds={
                        'min': [-30, -15, -np.pi, 0, -0.6],
                        'max': [30, 15, np.pi, 5, 0.6]
                    }
                )
                .add_ellipsoid_constraints(obstacles, safety_margin=0.3)
                .set_initial_state([0.0, 0.0, 0.0, 2.0, 0.0])
                .build())
    
    @staticmethod
    def combined_constraints() -> TestConfig:
        """Combined constraint types."""
        builder = TestConfigBuilder("combined_constraints")
        
        # Create curving road
        t = np.linspace(0, 6*np.pi, 100)
        x = 0.2 * t
        y = 2 * np.sin(0.2 * t) + 0.5 * np.sin(0.6 * t)
        reference_path = np.column_stack([x, y])
        
        # Create ellipsoid obstacles
        obstacles = []
        for i in range(25):
            x_obs = 1.0 + 0.3 * i
            y_obs = 1.0 + 0.8 * np.sin(0.1 * i)
            obstacles.append({
                'center': [x_obs, y_obs],
                'shape': [0.6, 0.3],
                'rotation': 0.0
            })
        
        # Create uncertain obstacles
        uncertain_obstacles = []
        for i in range(25):
            x_mean = 2.0 + 0.2 * i
            y_mean = 1.5 + 0.5 * np.sin(0.12 * i)
            uncertainty = 0.1 + 0.02 * i
            uncertain_obstacles.append({
                'mean': [x_mean, y_mean],
                'covariance': [[uncertainty, 0.0], [0.0, uncertainty]],
                'shape': [0.5, 0.3]
            })
        
        # Create scenarios
        scenarios = []
        for scenario_idx in range(2):
            scenario_obstacles = []
            for i in range(25):
                if scenario_idx == 0:  # Aggressive scenario
                    x_obs = 0.5 + 0.35 * i
                    y_obs = 1.5 + 0.6 * np.sin(0.15 * i)
                else:  # Conservative scenario
                    x_obs = 1.5 + 0.2 * i
                    y_obs = 2.5 + 0.3 * np.sin(0.08 * i)
                
                scenario_obstacles.append({
                    'center': [x_obs, y_obs],
                    'shape': [0.7, 0.4]
                })
            
            scenarios.append({
                'obstacles': scenario_obstacles,
                'constraints': [],
                'probability': 0.5
            })
        
        return (builder
                .set_mpc_params(horizon_length=25, dt=0.1)
                .add_contouring_objective(reference_path)
                .add_linear_constraints(
                    state_bounds={
                        'min': [-30, -15, -np.pi, 0, -0.6],
                        'max': [30, 15, np.pi, 5, 0.6]
                    }
                )
                .add_ellipsoid_constraints(obstacles, safety_margin=0.3)
                .add_gaussian_constraints(uncertain_obstacles, confidence_level=0.95)
                .add_scenario_constraints(scenarios, scenario_weights=[0.5, 0.5])
                .set_initial_state([0.0, 0.0, 0.0, 2.0, 0.0])
                .build())


def create_test_config(test_name: str) -> TestConfigBuilder:
    """
    Create a new test configuration builder.
    
    Args:
        test_name: Name of the test
        
    Returns:
        Test configuration builder
    """
    return TestConfigBuilder(test_name)
