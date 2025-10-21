"""
Refactored Constraint Tests Package

This package contains refactored constraint tests using the unified framework
with configurable perception areas, obstacle memory, and trajectory funnels.
"""

from .unified_framework import UnifiedConstraintFramework
from .test_runner import RefactoredTestRunner

__all__ = [
    'UnifiedConstraintFramework',
    'RefactoredTestRunner'
]
