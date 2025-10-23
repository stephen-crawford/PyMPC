"""
Testing framework for PyMPC.

This module provides a comprehensive testing framework for MPC systems including:
- Unified test runner for various MPC scenarios
- Constraint testing framework
- Test configuration and result management
- Visualization and analysis tools
"""

from .test_config import (
    TestConfig, TestResult, ConstraintTestConfig, ConstraintTestResult,
    TestConfigManager, TestType, VehicleType, RoadType, ConstraintType,
    VehicleConfig, RoadConfig, MPCConfig, ObstacleConfig
)

from .unified_test_runner import UnifiedTestRunner
from .unified_constraint_framework import UnifiedConstraintFramework

__all__ = [
    # Test configuration classes
    'TestConfig',
    'TestResult',
    'ConstraintTestConfig',
    'ConstraintTestResult',
    'TestConfigManager',
    
    # Enums
    'TestType',
    'VehicleType',
    'RoadType',
    'ConstraintType',
    
    # Config dataclasses
    'VehicleConfig',
    'RoadConfig',
    'MPCConfig',
    'ObstacleConfig',
    
    # Testing frameworks
    'UnifiedTestRunner',
    'UnifiedConstraintFramework',
]
