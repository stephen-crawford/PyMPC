"""
Standardized Logging System for PyMPC

This module provides a comprehensive logging system with:
- Clear log levels and formatting
- Context-aware logging
- Performance monitoring
- Error tracking and diagnostics
- Test-specific logging
"""

import logging
import sys
import time
import traceback
import functools
from datetime import datetime
from typing import Any, Dict, Optional, List
import json
import numpy as np


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Add color to level name
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        
        # Add timestamp
        record.timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        return super().format(record)


class TestLogger:
    """Specialized logger for test execution with detailed diagnostics."""
    
    def __init__(self, test_name: str, log_level: str = "INFO"):
        self.test_name = test_name
        self.logger = logging.getLogger(f"test.{test_name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create console handler with colored formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        formatter = ColoredFormatter(
            fmt='%(timestamp)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Test-specific data
        self.test_start_time = None
        self.test_phases = []
        self.errors = []
        self.warnings = []
        self.performance_metrics = {}
        
    def start_test(self):
        """Mark the beginning of a test."""
        self.test_start_time = time.time()
        self.logger.info("🚀 Starting test: %s", self.test_name)
        
    def end_test(self, success: bool = True):
        """Mark the end of a test with summary."""
        if self.test_start_time:
            duration = time.time() - self.test_start_time
            status = "✅ PASSED" if success else "❌ FAILED"
            self.logger.info("%s Test '%s' completed in %.2fs", status, self.test_name, duration)
            
            if self.errors:
                self.logger.error("Errors encountered: %d", len(self.errors))
            if self.warnings:
                self.logger.warning("Warnings: %d", len(self.warnings))
                
    def log_phase(self, phase_name: str, details: str = ""):
        """Log a test phase with timing."""
        phase_start = time.time()
        self.test_phases.append({
            'name': phase_name,
            'start_time': phase_start,
            'details': details
        })
        self.logger.info("📋 Phase: %s - %s", phase_name, details)
        
    def log_success(self, message: str, details: Dict = None):
        """Log a successful operation."""
        self.logger.info("✅ %s", message)
        if details:
            self.logger.debug("   Details: %s", json.dumps(details, indent=2))
            
    def log_warning(self, message: str, details: Dict = None):
        """Log a warning with context."""
        warning_info = {
            'message': message,
            'details': details,
            'timestamp': time.time()
        }
        self.warnings.append(warning_info)
        self.logger.warning("⚠️  %s", message)
        if details:
            self.logger.debug("   Context: %s", json.dumps(details, indent=2))
            
    def log_error(self, message: str, exception: Exception = None, details: Dict = None):
        """Log an error with full context."""
        error_info = {
            'message': message,
            'exception': str(exception) if exception else None,
            'traceback': traceback.format_exc() if exception else None,
            'details': details,
            'timestamp': time.time()
        }
        self.errors.append(error_info)
        self.logger.error("❌ %s", message)
        if exception:
            self.logger.error("   Exception: %s", exception)
        if details:
            self.logger.debug("   Context: %s", json.dumps(details, indent=2))
            
    def log_performance(self, operation: str, duration: float, metrics: Dict = None):
        """Log performance metrics."""
        self.performance_metrics[operation] = {
            'duration': duration,
            'metrics': metrics or {}
        }
        self.logger.info("⏱️  %s: %.3fs", operation, duration)
        
    def log_debug(self, message: str, data: Any = None):
        """Log debug information."""
        self.logger.debug("🔍 %s", message)
        if data is not None:
            if isinstance(data, np.ndarray):
                self.logger.debug("   Array shape: %s, dtype: %s", data.shape, data.dtype)
            else:
                self.logger.debug("   Data: %s", data)
                
    def get_test_summary(self) -> Dict:
        """Get a comprehensive test summary."""
        return {
            'test_name': self.test_name,
            'duration': time.time() - self.test_start_time if self.test_start_time else 0,
            'phases': self.test_phases,
            'errors': self.errors,
            'warnings': self.warnings,
            'performance_metrics': self.performance_metrics,
            'success': len(self.errors) == 0
        }
    
    def log_info(self, message: str):
        """Log info message."""
        self.logger.info(f"ℹ️  {message}")
    
    def get_error_log(self) -> List[str]:
        """Get list of error messages."""
        return [error['message'] for error in self.errors]


class PerformanceMonitor:
    """Context manager for performance monitoring."""
    
    def __init__(self, logger: TestLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.logger.log_performance(self.operation, duration)


class DiagnosticLogger:
    """Specialized logger for diagnostic information."""
    
    def __init__(self, logger: TestLogger):
        self.logger = logger
        
    def log_solver_state(self, solver_name: str, state: Dict):
        """Log solver state information."""
        self.logger.log_debug(f"Solver '{solver_name}' state", {
            'variables': len(state.get('variables', [])),
            'constraints': len(state.get('constraints', [])),
            'objectives': len(state.get('objectives', []))
        })
        
    def log_constraint_violation(self, constraint_name: str, violation: float, 
                                expected_bounds: tuple, actual_value: float):
        """Log constraint violations with detailed analysis."""
        self.logger.log_error(f"Constraint violation in '{constraint_name}'", details={
            'violation_amount': violation,
            'expected_bounds': expected_bounds,
            'actual_value': actual_value,
            'violation_percentage': (violation / (expected_bounds[1] - expected_bounds[0])) * 100
        })
        
    def log_vehicle_state(self, state: Dict, iteration: int):
        """Log vehicle state with position and velocity info."""
        self.logger.log_debug(f"Vehicle state at iteration {iteration}", {
            'position': (state.get('x', 0), state.get('y', 0)),
            'velocity': state.get('v', 0),
            'heading': state.get('psi', 0),
            'spline_position': state.get('spline', 0)
        })
        
    def log_mpc_failure(self, failure_reason: str, solver_output: Dict = None):
        """Log MPC solver failures with detailed diagnostics."""
        self.logger.log_error(f"MPC solver failed: {failure_reason}", details={
            'solver_output': solver_output,
            'possible_causes': [
                "Overconstrained problem (too many constraints)",
                "Infeasible constraints",
                "Numerical issues in solver",
                "Invalid constraint bounds",
                "Missing or invalid parameters"
            ]
        })


# Global logger instances
_test_loggers: Dict[str, TestLogger] = {}
_current_test_logger: Optional[TestLogger] = None


def get_test_logger(test_name: str, log_level: str = "INFO") -> TestLogger:
    """Get or create a test logger."""
    global _test_loggers, _current_test_logger
    
    if test_name not in _test_loggers:
        _test_loggers[test_name] = TestLogger(test_name, log_level)
    
    _current_test_logger = _test_loggers[test_name]
    return _current_test_logger


def get_current_logger() -> Optional[TestLogger]:
    """Get the current active test logger."""
    return _current_test_logger


def log_function_call(func):
    """Decorator to log function calls with performance monitoring."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_current_logger()
        if logger:
            logger.log_debug(f"Calling {func.__name__}")
            with PerformanceMonitor(logger, func.__name__):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper


def log_exceptions(func):
    """Decorator to log exceptions with full context."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_current_logger()
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if logger:
                logger.log_error(f"Exception in {func.__name__}", e, {
                    'args': str(args)[:100] + '...' if len(str(args)) > 100 else str(args),
                    'kwargs': str(kwargs)[:100] + '...' if len(str(kwargs)) > 100 else str(kwargs)
                })
            raise
    return wrapper


# Convenience functions for backward compatibility
def LOG_DEBUG(message: str, data: Any = None):
    """Log debug message."""
    logger = get_current_logger()
    if logger:
        logger.log_debug(message, data)
    else:
        print(f"DEBUG: {message}")

def LOG_INFO(message: str):
    """Log info message."""
    logger = get_current_logger()
    if logger:
        logger.logger.info(message)
    else:
        print(f"INFO: {message}")

def LOG_WARN(message: str):
    """Log warning message."""
    logger = get_current_logger()
    if logger:
        logger.logger.warning(message)
    else:
        print(f"WARNING: {message}")

def LOG_ERROR(message: str, exception: Exception = None):
    """Log error message."""
    logger = get_current_logger()
    if logger:
        logger.log_error(message, exception)
    else:
        print(f"ERROR: {message}")


# Export main classes and functions
__all__ = [
    'TestLogger', 'PerformanceMonitor', 'DiagnosticLogger',
    'get_test_logger', 'get_current_logger',
    'log_function_call', 'log_exceptions',
    'LOG_DEBUG', 'LOG_INFO', 'LOG_WARN', 'LOG_ERROR'
]
