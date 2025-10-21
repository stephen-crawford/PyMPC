"""
Standardized testing framework for PyMPC.

This module provides a comprehensive test framework with:
- Easy test implementation and modification
- Clear failure explanations and diagnostics
- Automatic test discovery and execution
- Performance monitoring and reporting
- Integration with logging and visualization systems
"""

from .base_test import BaseMPCTest, TestConfig, TestResult
from .test_suite import TestSuite
from .constraint_tests import ConstraintTestSuite
from .objective_tests import ObjectiveTestSuite
from .integration_tests import IntegrationTestSuite

__all__ = [
    'BaseMPCTest',
    'TestConfig',
    'TestResult',
    'TestSuite',
    'ConstraintTestSuite',
    'ObjectiveTestSuite',
    'IntegrationTestSuite'
]
