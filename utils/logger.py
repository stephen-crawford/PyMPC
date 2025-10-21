"""
Comprehensive logging framework for MPC tests and analysis.

This module provides structured logging for MPC optimization, constraint analysis,
and performance monitoring.
"""

import logging
import time
import json
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import os
from pathlib import Path


class MPCLogger:
    """
    Comprehensive logger for MPC optimization and analysis.
    """
    
    def __init__(self, 
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 enable_file_logging: bool = True,
                 enable_console_logging: bool = True):
        """
        Initialize MPC logger.
        
        Args:
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            enable_file_logging: Enable file logging
            enable_console_logging: Enable console logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create session timestamp
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_dir = self.log_dir / f"session_{self.session_id}"
        self.session_log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(f"mpc_logger_{self.session_id}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        if enable_console_logging:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if enable_file_logging:
            file_handler = logging.FileHandler(
                self.session_log_dir / "mpc_optimization.log"
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Initialize data storage
        self.optimization_data = []
        self.constraint_data = []
        self.performance_data = []
        
        self.logger.info(f"MPC Logger initialized - Session: {self.session_id}")
    
    def log_optimization_start(self, 
                              test_name: str,
                              horizon_length: int,
                              dt: float,
                              state_dim: int,
                              control_dim: int,
                              num_constraints: int = 0) -> None:
        """Log the start of an optimization."""
        self.logger.info(f"Starting optimization: {test_name}")
        self.logger.info(f"  Horizon: {horizon_length}, dt: {dt}")
        self.logger.info(f"  State dim: {state_dim}, Control dim: {control_dim}")
        self.logger.info(f"  Constraints: {num_constraints}")
        
        self.current_test = {
            'test_name': test_name,
            'start_time': time.time(),
            'horizon_length': horizon_length,
            'dt': dt,
            'state_dim': state_dim,
            'control_dim': control_dim,
            'num_constraints': num_constraints
        }
    
    def log_optimization_end(self, 
                           success: bool,
                           solve_time: float,
                           objective_value: Optional[float] = None,
                           iterations: Optional[int] = None,
                           constraint_violations: Optional[int] = None) -> None:
        """Log the end of an optimization."""
        if not hasattr(self, 'current_test'):
            self.logger.warning("No current test to end")
            return
        
        end_time = time.time()
        total_time = end_time - self.current_test['start_time']
        
        self.current_test.update({
            'end_time': end_time,
            'total_time': total_time,
            'solve_time': solve_time,
            'success': success,
            'objective_value': objective_value,
            'iterations': iterations,
            'constraint_violations': constraint_violations
        })
        
        # Log results
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"Optimization {status}: {self.current_test['test_name']}")
        self.logger.info(f"  Total time: {total_time:.3f}s, Solve time: {solve_time:.3f}s")
        
        if objective_value is not None:
            self.logger.info(f"  Objective value: {objective_value:.6f}")
        if iterations is not None:
            self.logger.info(f"  Iterations: {iterations}")
        if constraint_violations is not None:
            self.logger.info(f"  Constraint violations: {constraint_violations}")
        
        # Store data
        self.optimization_data.append(self.current_test.copy())
        
        # Save to file
        self._save_optimization_data()
    
    def log_constraint_info(self, 
                           constraint_type: str,
                           constraint_data: Dict[str, Any]) -> None:
        """Log constraint information."""
        self.logger.info(f"Adding constraint: {constraint_type}")
        
        constraint_info = {
            'timestamp': time.time(),
            'constraint_type': constraint_type,
            'constraint_data': constraint_data
        }
        
        self.constraint_data.append(constraint_info)
        self.logger.debug(f"Constraint data: {constraint_data}")
    
    def log_performance_metrics(self, 
                              metrics: Dict[str, Any]) -> None:
        """Log performance metrics."""
        self.logger.info("Performance metrics:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")
        
        self.performance_data.append({
            'timestamp': time.time(),
            'metrics': metrics
        })
    
    def log_trajectory_analysis(self, 
                               trajectory: np.ndarray,
                               reference_path: Optional[np.ndarray] = None,
                               obstacles: Optional[List[Dict]] = None) -> None:
        """Log trajectory analysis."""
        if trajectory is None or trajectory.size == 0:
            self.logger.warning("Empty trajectory provided")
            return
        
        # Basic trajectory statistics
        positions = trajectory[:2, :] if trajectory.shape[0] >= 2 else trajectory
        path_length = self._compute_path_length(positions)
        max_velocity = np.max(np.linalg.norm(np.diff(positions, axis=1), axis=0)) / 0.1  # Assuming dt=0.1
        
        self.logger.info(f"Trajectory analysis:")
        self.logger.info(f"  Path length: {path_length:.3f}m")
        self.logger.info(f"  Max velocity: {max_velocity:.3f}m/s")
        self.logger.info(f"  Trajectory shape: {trajectory.shape}")
        
        # Reference path comparison
        if reference_path is not None:
            tracking_error = self._compute_tracking_error(positions, reference_path)
            self.logger.info(f"  Tracking error: {tracking_error:.3f}m")
        
        # Obstacle analysis
        if obstacles is not None:
            min_distances = self._compute_obstacle_distances(positions, obstacles)
            self.logger.info(f"  Min obstacle distance: {min_distances:.3f}m")
    
    def log_constraint_violations(self, 
                                violations: Dict[str, int]) -> None:
        """Log constraint violations."""
        self.logger.warning("Constraint violations detected:")
        for constraint_type, count in violations.items():
            self.logger.warning(f"  {constraint_type}: {count} violations")
    
    def log_solver_info(self, 
                       solver_name: str,
                       solver_options: Dict[str, Any]) -> None:
        """Log solver information."""
        self.logger.info(f"Solver: {solver_name}")
        self.logger.info(f"Solver options: {solver_options}")
    
    def log_info(self, message: str, session_id: Optional[str] = None) -> None:
        """Log info message."""
        if session_id:
            self.logger.info(f"[{session_id}] {message}")
        else:
            self.logger.info(message)
    
    def log_warning(self, message: str, session_id: Optional[str] = None) -> None:
        """Log warning message."""
        if session_id:
            self.logger.warning(f"[{session_id}] {message}")
        else:
            self.logger.warning(message)
    
    def log_error(self, message: str, session_id: Optional[str] = None) -> None:
        """Log error message."""
        if session_id:
            self.logger.error(f"[{session_id}] {message}")
        else:
            self.logger.error(message)
    
    def start_session(self, session_name: str) -> str:
        """Start a new logging session."""
        session_id = f"{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Starting session: {session_name} (ID: {session_id})")
        return session_id
    
    def end_session(self, session_id: str) -> None:
        """End a logging session."""
        self.logger.info(f"Ending session: {session_id}")
    
    def _compute_path_length(self, positions: np.ndarray) -> float:
        """Compute path length."""
        if positions.shape[1] < 2:
            return 0.0
        
        distances = np.linalg.norm(np.diff(positions, axis=1), axis=0)
        return np.sum(distances)
    
    def _compute_tracking_error(self, 
                               trajectory: np.ndarray, 
                               reference_path: np.ndarray) -> float:
        """Compute tracking error."""
        if reference_path.shape[1] != trajectory.shape[1]:
            return float('inf')
        
        errors = np.linalg.norm(trajectory - reference_path, axis=0)
        return np.mean(errors)
    
    def _compute_obstacle_distances(self, 
                                   positions: np.ndarray, 
                                   obstacles: List[Dict]) -> float:
        """Compute minimum obstacle distances."""
        if not obstacles:
            return float('inf')
        
        min_distances = []
        for i in range(positions.shape[1]):
            pos = positions[:, i]
            distances = []
            for obs in obstacles:
                if 'center' in obs:
                    center = obs['center']
                    distance = np.linalg.norm(pos - center)
                    distances.append(distance)
            if distances:
                min_distances.append(min(distances))
        
        return min(min_distances) if min_distances else float('inf')
    
    def _save_optimization_data(self) -> None:
        """Save optimization data to file."""
        data_file = self.session_log_dir / "optimization_data.json"
        
        # Convert numpy arrays to lists for JSON serialization
        data_to_save = []
        for item in self.optimization_data:
            item_copy = item.copy()
            # Convert numpy arrays to lists
            for key, value in item_copy.items():
                if isinstance(value, np.ndarray):
                    item_copy[key] = value.tolist()
            data_to_save.append(item_copy)
        
        with open(data_file, 'w') as f:
            json.dump(data_to_save, f, indent=2)
    
    def save_session_summary(self) -> Dict[str, Any]:
        """Save session summary."""
        summary = {
            'session_id': self.session_id,
            'start_time': self.session_log_dir.name,
            'total_tests': len(self.optimization_data),
            'successful_tests': sum(1 for test in self.optimization_data if test.get('success', False)),
            'failed_tests': sum(1 for test in self.optimization_data if not test.get('success', True)),
            'total_solve_time': sum(test.get('solve_time', 0) for test in self.optimization_data),
            'average_solve_time': np.mean([test.get('solve_time', 0) for test in self.optimization_data]) if self.optimization_data else 0,
            'constraint_types': list(set(item.get('constraint_type', 'unknown') for item in self.constraint_data)),
            'optimization_data': self.optimization_data,
            'constraint_data': self.constraint_data,
            'performance_data': self.performance_data
        }
        
        summary_file = self.session_log_dir / "session_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Session summary saved to {summary_file}")
        return summary
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        if not self.optimization_data:
            return {}
        
        solve_times = [test.get('solve_time', 0) for test in self.optimization_data]
        successful_tests = [test for test in self.optimization_data if test.get('success', False)]
        
        return {
            'total_tests': len(self.optimization_data),
            'successful_tests': len(successful_tests),
            'success_rate': len(successful_tests) / len(self.optimization_data) * 100,
            'average_solve_time': np.mean(solve_times),
            'min_solve_time': np.min(solve_times),
            'max_solve_time': np.max(solve_times),
            'total_solve_time': np.sum(solve_times)
        }
    
    def print_session_summary(self) -> None:
        """Print session summary to console."""
        stats = self.get_session_stats()
        if not stats:
            self.logger.info("No optimization data available")
            return
        
        self.logger.info("=" * 50)
        self.logger.info("SESSION SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Total tests: {stats['total_tests']}")
        self.logger.info(f"Successful tests: {stats['successful_tests']}")
        self.logger.info(f"Success rate: {stats['success_rate']:.1f}%")
        self.logger.info(f"Average solve time: {stats['average_solve_time']:.3f}s")
        self.logger.info(f"Min solve time: {stats['min_solve_time']:.3f}s")
        self.logger.info(f"Max solve time: {stats['max_solve_time']:.3f}s")
        self.logger.info(f"Total solve time: {stats['total_solve_time']:.3f}s")
        self.logger.info("=" * 50)


class MPCProfiler:
    """
    Performance profiler for MPC optimization.
    """
    
    def __init__(self, logger: MPCLogger):
        self.logger = logger
        self.profiles = {}
    
    def start_profile(self, name: str) -> None:
        """Start profiling a section."""
        self.profiles[name] = {
            'start_time': time.time(),
            'end_time': None,
            'duration': None
        }
    
    def end_profile(self, name: str) -> float:
        """End profiling a section."""
        if name not in self.profiles:
            self.logger.logger.warning(f"Profile '{name}' not found")
            return 0.0
        
        end_time = time.time()
        duration = end_time - self.profiles[name]['start_time']
        
        self.profiles[name]['end_time'] = end_time
        self.profiles[name]['duration'] = duration
        
        self.logger.logger.info(f"Profile '{name}': {duration:.3f}s")
        return duration
    
    def get_profile_summary(self) -> Dict[str, float]:
        """Get profile summary."""
        return {name: profile['duration'] for name, profile in self.profiles.items() 
                if profile['duration'] is not None}
