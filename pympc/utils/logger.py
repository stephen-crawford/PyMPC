"""
Advanced logging utilities for MPC framework.

This module provides enhanced logging capabilities with structured logging,
performance monitoring, and debugging support.
"""

import logging
import sys
import os
import time
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
import traceback


class MPCFormatter(logging.Formatter):
    """Custom formatter for MPC logging."""
    
    def __init__(self, include_timestamp: bool = True, include_level: bool = True):
        """
        Initialize MPC formatter.
        
        Args:
            include_timestamp: Whether to include timestamp
            include_level: Whether to include log level
        """
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        
        # Create format string
        format_parts = []
        if include_timestamp:
            format_parts.append('%(asctime)s')
        if include_level:
            format_parts.append('%(levelname)s')
        format_parts.extend(['%(name)s', '%(message)s'])
        
        format_string = ' - '.join(format_parts)
        super().__init__(format_string)
    
    def format(self, record):
        """Format log record."""
        # Add custom fields
        record.function_name = record.funcName
        record.line_number = record.lineno
        
        return super().format(record)


class PerformanceLogger:
    """Logger for performance monitoring."""
    
    def __init__(self, logger_name: str = "pympc.performance"):
        """
        Initialize performance logger.
        
        Args:
            logger_name: Name of the performance logger
        """
        self.logger = logging.getLogger(logger_name)
        self.performance_data: List[Dict[str, Any]] = []
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation_name: str) -> None:
        """
        Start timing an operation.
        
        Args:
            operation_name: Name of the operation
        """
        self.start_times[operation_name] = time.time()
        self.logger.debug(f"Started timing: {operation_name}")
    
    def end_timer(self, operation_name: str, 
                  additional_data: Dict[str, Any] = None) -> float:
        """
        End timing an operation.
        
        Args:
            operation_name: Name of the operation
            additional_data: Additional data to log
            
        Returns:
            Duration in seconds
        """
        if operation_name not in self.start_times:
            self.logger.warning(f"No start time found for operation: {operation_name}")
            return 0.0
        
        start_time = self.start_times.pop(operation_name)
        duration = time.time() - start_time
        
        # Log performance data
        performance_entry = {
            'operation': operation_name,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'additional_data': additional_data or {}
        }
        self.performance_data.append(performance_entry)
        
        self.logger.info(f"Operation '{operation_name}' took {duration:.4f} seconds")
        
        return duration
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.
        
        Returns:
            Performance summary dictionary
        """
        if not self.performance_data:
            return {'total_operations': 0, 'average_duration': 0.0}
        
        durations = [entry['duration'] for entry in self.performance_data]
        
        return {
            'total_operations': len(self.performance_data),
            'total_duration': sum(durations),
            'average_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'operations': self.performance_data
        }
    
    def save_performance_data(self, filepath: str) -> None:
        """
        Save performance data to file.
        
        Args:
            filepath: Path to save performance data
        """
        with open(filepath, 'w') as f:
            json.dump(self.get_performance_summary(), f, indent=2)
    
    def clear_performance_data(self) -> None:
        """Clear all performance data."""
        self.performance_data = []
        self.start_times = {}


class DebugLogger:
    """Logger for debugging MPC operations."""
    
    def __init__(self, logger_name: str = "pympc.debug"):
        """
        Initialize debug logger.
        
        Args:
            logger_name: Name of the debug logger
        """
        self.logger = logging.getLogger(logger_name)
        self.debug_data: List[Dict[str, Any]] = []
    
    def log_state(self, state_name: str, state_data: Any, 
                  context: str = "") -> None:
        """
        Log state information.
        
        Args:
            state_name: Name of the state
            state_data: State data
            context: Additional context
        """
        debug_entry = {
            'type': 'state',
            'name': state_name,
            'data': str(state_data),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        self.debug_data.append(debug_entry)
        
        self.logger.debug(f"State '{state_name}': {state_data} {context}")
    
    def log_constraint(self, constraint_name: str, constraint_value: float,
                      satisfied: bool, context: str = "") -> None:
        """
        Log constraint information.
        
        Args:
            constraint_name: Name of the constraint
            constraint_value: Constraint value
            satisfied: Whether constraint is satisfied
            context: Additional context
        """
        debug_entry = {
            'type': 'constraint',
            'name': constraint_name,
            'value': constraint_value,
            'satisfied': satisfied,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        self.debug_data.append(debug_entry)
        
        status = "SATISFIED" if satisfied else "VIOLATED"
        self.logger.debug(f"Constraint '{constraint_name}': {constraint_value} ({status}) {context}")
    
    def log_objective(self, objective_name: str, objective_value: float,
                     context: str = "") -> None:
        """
        Log objective information.
        
        Args:
            objective_name: Name of the objective
            objective_value: Objective value
            context: Additional context
        """
        debug_entry = {
            'type': 'objective',
            'name': objective_name,
            'value': objective_value,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        self.debug_data.append(debug_entry)
        
        self.logger.debug(f"Objective '{objective_name}': {objective_value} {context}")
    
    def log_solver_info(self, solver_name: str, iterations: int,
                        solve_time: float, status: str, context: str = "") -> None:
        """
        Log solver information.
        
        Args:
            solver_name: Name of the solver
            iterations: Number of iterations
            solve_time: Solve time in seconds
            status: Solver status
            context: Additional context
        """
        debug_entry = {
            'type': 'solver',
            'name': solver_name,
            'iterations': iterations,
            'solve_time': solve_time,
            'status': status,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        self.debug_data.append(debug_entry)
        
        self.logger.debug(f"Solver '{solver_name}': {iterations} iterations, "
                         f"{solve_time:.4f}s, status={status} {context}")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """
        Log error information.
        
        Args:
            error: Exception to log
            context: Additional context
        """
        debug_entry = {
            'type': 'error',
            'name': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        self.debug_data.append(debug_entry)
        
        self.logger.error(f"Error in {context}: {error}")
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """
        Get debug summary.
        
        Returns:
            Debug summary dictionary
        """
        if not self.debug_data:
            return {'total_entries': 0}
        
        # Count by type
        type_counts = {}
        for entry in self.debug_data:
            entry_type = entry['type']
            type_counts[entry_type] = type_counts.get(entry_type, 0) + 1
        
        return {
            'total_entries': len(self.debug_data),
            'type_counts': type_counts,
            'entries': self.debug_data
        }
    
    def save_debug_data(self, filepath: str) -> None:
        """
        Save debug data to file.
        
        Args:
            filepath: Path to save debug data
        """
        with open(filepath, 'w') as f:
            json.dump(self.get_debug_summary(), f, indent=2)
    
    def clear_debug_data(self) -> None:
        """Clear all debug data."""
        self.debug_data = []


class MPCLogger:
    """Main MPC logger with multiple logging capabilities."""
    
    def __init__(self, name: str = "pympc", log_level: int = logging.INFO,
                 log_file: Optional[str] = None, 
                 performance_logging: bool = True,
                 debug_logging: bool = False):
        """
        Initialize MPC logger.
        
        Args:
            name: Logger name
            log_level: Logging level
            log_file: Optional log file path
            performance_logging: Enable performance logging
            debug_logging: Enable debug logging
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = MPCFormatter(include_timestamp=True, include_level=True)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = MPCFormatter(include_timestamp=True, include_level=True)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Initialize sub-loggers
        self.performance_logger = PerformanceLogger(f"{name}.performance") if performance_logging else None
        self.debug_logger = DebugLogger(f"{name}.debug") if debug_logging else None
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    def log_performance(self, operation_name: str, duration: float,
                       additional_data: Dict[str, Any] = None) -> None:
        """
        Log performance data.
        
        Args:
            operation_name: Name of the operation
            duration: Duration in seconds
            additional_data: Additional data
        """
        if self.performance_logger:
            self.performance_logger.performance_data.append({
                'operation': operation_name,
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'additional_data': additional_data or {}
            })
    
    def log_debug_state(self, state_name: str, state_data: Any, 
                       context: str = "") -> None:
        """
        Log debug state.
        
        Args:
            state_name: Name of the state
            state_data: State data
            context: Additional context
        """
        if self.debug_logger:
            self.debug_logger.log_state(state_name, state_data, context)
    
    def log_debug_constraint(self, constraint_name: str, constraint_value: float,
                            satisfied: bool, context: str = "") -> None:
        """
        Log debug constraint.
        
        Args:
            constraint_name: Name of the constraint
            constraint_value: Constraint value
            satisfied: Whether constraint is satisfied
            context: Additional context
        """
        if self.debug_logger:
            self.debug_logger.log_constraint(constraint_name, constraint_value, satisfied, context)
    
    def log_debug_objective(self, objective_name: str, objective_value: float,
                           context: str = "") -> None:
        """
        Log debug objective.
        
        Args:
            objective_name: Name of the objective
            objective_value: Objective value
            context: Additional context
        """
        if self.debug_logger:
            self.debug_logger.log_objective(objective_name, objective_value, context)
    
    def log_debug_solver(self, solver_name: str, iterations: int,
                        solve_time: float, status: str, context: str = "") -> None:
        """
        Log debug solver info.
        
        Args:
            solver_name: Name of the solver
            iterations: Number of iterations
            solve_time: Solve time in seconds
            status: Solver status
            context: Additional context
        """
        if self.debug_logger:
            self.debug_logger.log_solver_info(solver_name, iterations, solve_time, status, context)
    
    def log_debug_error(self, error: Exception, context: str = "") -> None:
        """
        Log debug error.
        
        Args:
            error: Exception to log
            context: Additional context
        """
        if self.debug_logger:
            self.debug_logger.log_error(error, context)
    
    def get_performance_summary(self) -> Optional[Dict[str, Any]]:
        """Get performance summary."""
        if self.performance_logger:
            return self.performance_logger.get_performance_summary()
        return None
    
    def get_debug_summary(self) -> Optional[Dict[str, Any]]:
        """Get debug summary."""
        if self.debug_logger:
            return self.debug_logger.get_debug_summary()
        return None
    
    def save_logs(self, output_dir: str) -> None:
        """
        Save all logs to directory.
        
        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.performance_logger:
            perf_file = os.path.join(output_dir, "performance_log.json")
            self.performance_logger.save_performance_data(perf_file)
        
        if self.debug_logger:
            debug_file = os.path.join(output_dir, "debug_log.json")
            self.debug_logger.save_debug_data(debug_file)
    
    def clear_logs(self) -> None:
        """Clear all logs."""
        if self.performance_logger:
            self.performance_logger.clear_performance_data()
        
        if self.debug_logger:
            self.debug_logger.clear_debug_data()


# Global logger instance
_global_logger: Optional[MPCLogger] = None


def get_logger(name: str = "pympc", **kwargs) -> MPCLogger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name
        **kwargs: Additional logger arguments
        
    Returns:
        Logger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = MPCLogger(name, **kwargs)
    
    return _global_logger


def setup_logging_config(log_level: int = logging.INFO,
                        log_file: Optional[str] = None,
                        performance_logging: bool = True,
                        debug_logging: bool = False) -> MPCLogger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        performance_logging: Enable performance logging
        debug_logging: Enable debug logging
        
    Returns:
        Configured logger instance
    """
    return get_logger(
        log_level=log_level,
        log_file=log_file,
        performance_logging=performance_logging,
        debug_logging=debug_logging
    )


# Convenience functions for backward compatibility
def LOG_DEBUG(message: str) -> None:
    """Log debug message."""
    get_logger().debug(message)


def LOG_INFO(message: str) -> None:
    """Log info message."""
    get_logger().info(message)


def LOG_WARN(message: str) -> None:
    """Log warning message."""
    get_logger().warning(message)


def LOG_ERROR(message: str) -> None:
    """Log error message."""
    get_logger().error(message)