"""
Test Configuration for MPC Framework

This module provides configuration options for running different types of MPC tests
with various constraint types and scenarios.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class ConstraintType(Enum):
    """Available constraint types for testing."""
    CONTOURING = "contouring"
    SCENARIO = "scenario"
    LINEARIZED = "linearized"
    ELLIPSOID = "ellipsoid"
    GAUSSIAN = "gaussian"
    DECOMPOSITION = "decomposition"
    ALL_COMBINED = "all_combined"


class RoadType(Enum):
    """Available road types for testing."""
    CURVED = "curved"
    STRAIGHT = "straight"
    COMPLEX = "complex"
    HIGHWAY = "highway"


class TestScenario(Enum):
    """Available test scenarios."""
    SIMPLE_DEMO = "simple_demo"
    CONSTRAINT_DEMO = "constraint_demo"
    END_TO_END = "end_to_end"
    CUSTOM = "custom"


@dataclass
class TestConfig:
    """Configuration for MPC tests."""
    
    # Test identification
    test_name: str = "mpc_test"
    test_type: TestScenario = TestScenario.SIMPLE_DEMO
    
    # Road configuration
    road_type: RoadType = RoadType.CURVED
    road_length: float = 120.0
    road_width: float = 6.0
    num_road_points: int = 80
    
    # Vehicle configuration
    vehicle_length: float = 4.0
    vehicle_width: float = 1.8
    max_velocity: float = 12.0
    max_acceleration: float = 3.0
    max_steering_angle: float = 0.5
    
    # MPC configuration
    horizon: int = 15
    timestep: float = 0.1
    max_steps: int = 80
    solver_tolerance: float = 1e-6
    
    # Constraint configuration
    constraint_types: List[ConstraintType] = None
    use_obstacles: bool = True
    num_obstacles: int = 3
    obstacle_radius_range: tuple = (0.8, 1.5)
    
    # Visualization configuration
    create_animation: bool = True
    animation_fps: int = 6
    animation_interval: int = 150
    figure_size: tuple = (16, 10)
    
    # Output configuration
    output_dir: str = "test_outputs"
    save_results: bool = True
    generate_report: bool = True
    
    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.constraint_types is None:
            self.constraint_types = [ConstraintType.CONTOURING, ConstraintType.SCENARIO]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'test_name': self.test_name,
            'test_type': self.test_type.value,
            'road_type': self.road_type.value,
            'road_length': self.road_length,
            'road_width': self.road_width,
            'num_road_points': self.num_road_points,
            'vehicle_length': self.vehicle_length,
            'vehicle_width': self.vehicle_width,
            'max_velocity': self.max_velocity,
            'max_acceleration': self.max_acceleration,
            'max_steering_angle': self.max_steering_angle,
            'horizon': self.horizon,
            'timestep': self.timestep,
            'max_steps': self.max_steps,
            'solver_tolerance': self.solver_tolerance,
            'constraint_types': [ct.value for ct in self.constraint_types],
            'use_obstacles': self.use_obstacles,
            'num_obstacles': self.num_obstacles,
            'obstacle_radius_range': self.obstacle_radius_range,
            'create_animation': self.create_animation,
            'animation_fps': self.animation_fps,
            'animation_interval': self.animation_interval,
            'figure_size': self.figure_size,
            'output_dir': self.output_dir,
            'save_results': self.save_results,
            'generate_report': self.generate_report
        }


class TestConfigBuilder:
    """Builder class for creating test configurations."""
    
    def __init__(self):
        self.config = TestConfig()
    
    def set_test_name(self, name: str) -> 'TestConfigBuilder':
        """Set the test name."""
        self.config.test_name = name
        return self
    
    def set_test_type(self, test_type: TestScenario) -> 'TestConfigBuilder':
        """Set the test type."""
        self.config.test_type = test_type
        return self
    
    def set_road_type(self, road_type: RoadType) -> 'TestConfigBuilder':
        """Set the road type."""
        self.config.road_type = road_type
        return self
    
    def set_road_length(self, length: float) -> 'TestConfigBuilder':
        """Set the road length."""
        self.config.road_length = length
        return self
    
    def set_road_width(self, width: float) -> 'TestConfigBuilder':
        """Set the road width."""
        self.config.road_width = width
        return self
    
    def set_vehicle_params(self, length: float, width: float, max_velocity: float) -> 'TestConfigBuilder':
        """Set vehicle parameters."""
        self.config.vehicle_length = length
        self.config.vehicle_width = width
        self.config.max_velocity = max_velocity
        return self
    
    def set_mpc_params(self, horizon: int, timestep: float, max_steps: int) -> 'TestConfigBuilder':
        """Set MPC parameters."""
        self.config.horizon = horizon
        self.config.timestep = timestep
        self.config.max_steps = max_steps
        return self
    
    def set_constraint_types(self, constraint_types: List[ConstraintType]) -> 'TestConfigBuilder':
        """Set constraint types."""
        self.config.constraint_types = constraint_types
        return self
    
    def set_obstacles(self, use_obstacles: bool, num_obstacles: int = 3) -> 'TestConfigBuilder':
        """Set obstacle configuration."""
        self.config.use_obstacles = use_obstacles
        self.config.num_obstacles = num_obstacles
        return self
    
    def set_visualization(self, create_animation: bool, fps: int = 6, figure_size: tuple = (16, 10)) -> 'TestConfigBuilder':
        """Set visualization parameters."""
        self.config.create_animation = create_animation
        self.config.animation_fps = fps
        self.config.figure_size = figure_size
        return self
    
    def set_output(self, output_dir: str, save_results: bool = True, generate_report: bool = True) -> 'TestConfigBuilder':
        """Set output configuration."""
        self.config.output_dir = output_dir
        self.config.save_results = save_results
        self.config.generate_report = generate_report
        return self
    
    def build(self) -> TestConfig:
        """Build and return the configuration."""
        return self.config


# Predefined configurations for common test scenarios
def get_simple_demo_config() -> TestConfig:
    """Get configuration for simple demo test."""
    return TestConfigBuilder() \
        .set_test_name("simple_demo") \
        .set_test_type(TestScenario.SIMPLE_DEMO) \
        .set_road_type(RoadType.CURVED) \
        .set_road_length(120.0) \
        .set_vehicle_params(4.0, 1.8, 12.0) \
        .set_mpc_params(15, 0.1, 80) \
        .set_constraint_types([ConstraintType.CONTOURING, ConstraintType.SCENARIO]) \
        .set_obstacles(True, 3) \
        .set_visualization(True, 6, (16, 10)) \
        .set_output("demo_outputs") \
        .build()


def get_contouring_test_config() -> TestConfig:
    """Get configuration for contouring constraints test."""
    return TestConfigBuilder() \
        .set_test_name("contouring_test") \
        .set_test_type(TestScenario.CONSTRAINT_DEMO) \
        .set_road_type(RoadType.CURVED) \
        .set_road_length(120.0) \
        .set_vehicle_params(4.0, 1.8, 12.0) \
        .set_mpc_params(15, 0.1, 80) \
        .set_constraint_types([ConstraintType.CONTOURING]) \
        .set_obstacles(False, 0) \
        .set_visualization(True, 6, (16, 10)) \
        .set_output("contouring_outputs") \
        .build()


def get_scenario_test_config() -> TestConfig:
    """Get configuration for scenario constraints test."""
    return TestConfigBuilder() \
        .set_test_name("scenario_test") \
        .set_test_type(TestScenario.CONSTRAINT_DEMO) \
        .set_road_type(RoadType.CURVED) \
        .set_road_length(120.0) \
        .set_vehicle_params(4.0, 1.8, 12.0) \
        .set_mpc_params(15, 0.1, 80) \
        .set_constraint_types([ConstraintType.CONTOURING, ConstraintType.SCENARIO]) \
        .set_obstacles(True, 3) \
        .set_visualization(True, 6, (16, 10)) \
        .set_output("scenario_outputs") \
        .build()


def get_linearized_test_config() -> TestConfig:
    """Get configuration for linearized constraints test."""
    return TestConfigBuilder() \
        .set_test_name("linearized_test") \
        .set_test_type(TestScenario.CONSTRAINT_DEMO) \
        .set_road_type(RoadType.STRAIGHT) \
        .set_road_length(100.0) \
        .set_vehicle_params(4.0, 1.8, 12.0) \
        .set_mpc_params(15, 0.1, 80) \
        .set_constraint_types([ConstraintType.CONTOURING, ConstraintType.LINEARIZED]) \
        .set_obstacles(True, 2) \
        .set_visualization(True, 6, (16, 10)) \
        .set_output("linearized_outputs") \
        .build()


def get_ellipsoid_test_config() -> TestConfig:
    """Get configuration for ellipsoid constraints test."""
    return TestConfigBuilder() \
        .set_test_name("ellipsoid_test") \
        .set_test_type(TestScenario.CONSTRAINT_DEMO) \
        .set_road_type(RoadType.CURVED) \
        .set_road_length(120.0) \
        .set_vehicle_params(4.0, 1.8, 12.0) \
        .set_mpc_params(15, 0.1, 80) \
        .set_constraint_types([ConstraintType.CONTOURING, ConstraintType.ELLIPSOID]) \
        .set_obstacles(True, 3) \
        .set_visualization(True, 6, (16, 10)) \
        .set_output("ellipsoid_outputs") \
        .build()


def get_gaussian_test_config() -> TestConfig:
    """Get configuration for Gaussian constraints test."""
    return TestConfigBuilder() \
        .set_test_name("gaussian_test") \
        .set_test_type(TestScenario.CONSTRAINT_DEMO) \
        .set_road_type(RoadType.COMPLEX) \
        .set_road_length(150.0) \
        .set_vehicle_params(4.0, 1.8, 12.0) \
        .set_mpc_params(15, 0.1, 100) \
        .set_constraint_types([ConstraintType.CONTOURING, ConstraintType.GAUSSIAN]) \
        .set_obstacles(True, 4) \
        .set_visualization(True, 6, (16, 10)) \
        .set_output("gaussian_outputs") \
        .build()


def get_decomposition_test_config() -> TestConfig:
    """Get configuration for decomposition constraints test."""
    return TestConfigBuilder() \
        .set_test_name("decomposition_test") \
        .set_test_type(TestScenario.CONSTRAINT_DEMO) \
        .set_road_type(RoadType.CURVED) \
        .set_road_length(120.0) \
        .set_vehicle_params(4.0, 1.8, 12.0) \
        .set_mpc_params(15, 0.1, 80) \
        .set_constraint_types([ConstraintType.CONTOURING, ConstraintType.DECOMPOSITION]) \
        .set_obstacles(True, 3) \
        .set_visualization(True, 6, (16, 10)) \
        .set_output("decomposition_outputs") \
        .build()


def get_all_combined_test_config() -> TestConfig:
    """Get configuration for all constraints combined test."""
    return TestConfigBuilder() \
        .set_test_name("all_combined_test") \
        .set_test_type(TestScenario.END_TO_END) \
        .set_road_type(RoadType.COMPLEX) \
        .set_road_length(150.0) \
        .set_vehicle_params(4.0, 1.8, 12.0) \
        .set_mpc_params(20, 0.1, 100) \
        .set_constraint_types([
            ConstraintType.CONTOURING,
            ConstraintType.SCENARIO,
            ConstraintType.LINEARIZED,
            ConstraintType.ELLIPSOID
        ]) \
        .set_obstacles(True, 4) \
        .set_visualization(True, 6, (16, 10)) \
        .set_output("all_combined_outputs") \
        .build()


def get_custom_config(**kwargs) -> TestConfig:
    """Get custom configuration with specified parameters."""
    builder = TestConfigBuilder()
    
    # Apply custom parameters
    for key, value in kwargs.items():
        if hasattr(builder.config, key):
            setattr(builder.config, key, value)
    
    return builder.build()


# Configuration registry for easy access
CONFIG_REGISTRY = {
    "simple_demo": get_simple_demo_config,
    "contouring": get_contouring_test_config,
    "scenario": get_scenario_test_config,
    "linearized": get_linearized_test_config,
    "ellipsoid": get_ellipsoid_test_config,
    "gaussian": get_gaussian_test_config,
    "decomposition": get_decomposition_test_config,
    "all_combined": get_all_combined_test_config
}


def get_config(config_name: str) -> TestConfig:
    """Get configuration by name."""
    if config_name in CONFIG_REGISTRY:
        return CONFIG_REGISTRY[config_name]()
    else:
        raise ValueError(f"Unknown configuration: {config_name}. Available: {list(CONFIG_REGISTRY.keys())}")


def list_available_configs() -> List[str]:
    """List all available configuration names."""
    return list(CONFIG_REGISTRY.keys())
