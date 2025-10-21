"""
Performance monitoring and analysis tools for MPC optimization.

This module provides comprehensive performance monitoring, analysis,
and benchmarking capabilities for MPC optimization.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict

# Optional imports
try:
    import seaborn as sns
    import pandas as pd
    SEABORN_AVAILABLE = True
    PANDAS_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    PANDAS_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for MPC optimization.
    """
    # Timing metrics
    solve_time: float = 0.0
    setup_time: float = 0.0
    total_time: float = 0.0
    
    # Optimization metrics
    iterations: int = 0
    objective_value: float = 0.0
    constraint_violations: int = 0
    
    # Problem size metrics
    num_variables: int = 0
    num_constraints: int = 0
    num_equality_constraints: int = 0
    num_inequality_constraints: int = 0
    
    # Convergence metrics
    convergence_status: str = "unknown"
    dual_infeasibility: float = 0.0
    constraint_violation: float = 0.0
    complementarity: float = 0.0
    
    # Trajectory metrics
    path_length: float = 0.0
    max_velocity: float = 0.0
    max_acceleration: float = 0.0
    tracking_error: float = 0.0
    
    # Memory metrics
    memory_usage: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Performance monitor for MPC optimization.
    """
    
    def __init__(self, 
                 log_dir: str = "performance_logs",
                 enable_plotting: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            log_dir: Directory for performance logs
            enable_plotting: Enable performance plotting
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create session directory
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        self.enable_plotting = enable_plotting
        self.metrics_history = []
        self.benchmark_data = defaultdict(list)
        
        # Performance tracking
        self.current_test = None
        self.start_times = {}
        
        # Set plotting style
        if enable_plotting:
            if SEABORN_AVAILABLE:
                plt.style.use('seaborn-v0_8')
                sns.set_palette("husl")
            else:
                plt.style.use('default')
    
    def start_test(self, test_name: str, test_config: Dict[str, Any]) -> None:
        """
        Start monitoring a test.
        
        Args:
            test_name: Name of the test
            test_config: Test configuration
        """
        self.current_test = {
            'test_name': test_name,
            'test_config': test_config,
            'start_time': time.time(),
            'metrics': PerformanceMetrics()
        }
        
        self.start_times[test_name] = time.time()
    
    def record_setup_time(self, setup_time: float) -> None:
        """Record setup time."""
        if self.current_test:
            self.current_test['metrics'].setup_time = setup_time
    
    def record_solve_time(self, solve_time: float) -> None:
        """Record solve time."""
        if self.current_test:
            self.current_test['metrics'].solve_time = solve_time
    
    def record_optimization_metrics(self, 
                                  iterations: int,
                                  objective_value: float,
                                  convergence_status: str = "unknown",
                                  dual_infeasibility: float = 0.0,
                                  constraint_violation: float = 0.0,
                                  complementarity: float = 0.0) -> None:
        """Record optimization metrics."""
        if self.current_test:
            metrics = self.current_test['metrics']
            metrics.iterations = iterations
            metrics.objective_value = objective_value
            metrics.convergence_status = convergence_status
            metrics.dual_infeasibility = dual_infeasibility
            metrics.constraint_violation = constraint_violation
            metrics.complementarity = complementarity
    
    def record_problem_size(self, 
                          num_variables: int,
                          num_constraints: int,
                          num_equality_constraints: int = 0,
                          num_inequality_constraints: int = 0) -> None:
        """Record problem size metrics."""
        if self.current_test:
            metrics = self.current_test['metrics']
            metrics.num_variables = num_variables
            metrics.num_constraints = num_constraints
            metrics.num_equality_constraints = num_equality_constraints
            metrics.num_inequality_constraints = num_inequality_constraints
    
    def record_trajectory_metrics(self, 
                                trajectory: np.ndarray,
                                reference_path: Optional[np.ndarray] = None,
                                dt: float = 0.1) -> None:
        """Record trajectory metrics."""
        if self.current_test and trajectory is not None:
            metrics = self.current_test['metrics']
            
            # Path length
            if trajectory.shape[0] >= 2:
                positions = trajectory[:2, :]
                distances = np.linalg.norm(np.diff(positions, axis=1), axis=0)
                metrics.path_length = np.sum(distances)
                
                # Max velocity
                velocities = distances / dt
                metrics.max_velocity = np.max(velocities)
                
                # Max acceleration
                if len(velocities) > 1:
                    accelerations = np.diff(velocities) / dt
                    metrics.max_acceleration = np.max(np.abs(accelerations))
            
            # Tracking error
            if reference_path is not None and trajectory.shape[0] >= 2:
                positions = trajectory[:2, :]
                if reference_path.shape[0] == 2:
                    ref_positions = reference_path
                else:
                    ref_positions = reference_path.T
                
                if positions.shape[1] == ref_positions.shape[1]:
                    errors = np.linalg.norm(positions - ref_positions, axis=0)
                    metrics.tracking_error = np.mean(errors)
    
    def record_constraint_violations(self, violations: int) -> None:
        """Record constraint violations."""
        if self.current_test:
            self.current_test['metrics'].constraint_violations = violations
    
    def record_memory_usage(self, memory_usage: float) -> None:
        """Record memory usage."""
        if self.current_test:
            self.current_test['metrics'].memory_usage = memory_usage
    
    def record_custom_metric(self, name: str, value: Any) -> None:
        """Record custom metric."""
        if self.current_test:
            self.current_test['metrics'].custom_metrics[name] = value
    
    def end_test(self, success: bool = True) -> PerformanceMetrics:
        """
        End monitoring a test.
        
        Args:
            success: Whether the test was successful
            
        Returns:
            Performance metrics
        """
        if not self.current_test:
            return PerformanceMetrics()
        
        end_time = time.time()
        total_time = end_time - self.current_test['start_time']
        
        self.current_test['metrics'].total_time = total_time
        self.current_test['success'] = success
        
        # Store metrics
        self.metrics_history.append(self.current_test.copy())
        
        # Update benchmark data
        test_name = self.current_test['test_name']
        self.benchmark_data[test_name].append(self.current_test['metrics'])
        
        # Save to file
        self._save_metrics()
        
        metrics = self.current_test['metrics']
        self.current_test = None
        
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {}
        
        successful_tests = [test for test in self.metrics_history if test.get('success', False)]
        solve_times = [test['metrics'].solve_time for test in successful_tests]
        
        return {
            'total_tests': len(self.metrics_history),
            'successful_tests': len(successful_tests),
            'success_rate': len(successful_tests) / len(self.metrics_history) * 100,
            'average_solve_time': np.mean(solve_times) if solve_times else 0,
            'min_solve_time': np.min(solve_times) if solve_times else 0,
            'max_solve_time': np.max(solve_times) if solve_times else 0,
            'total_solve_time': np.sum(solve_times) if solve_times else 0
        }
    
    def plot_performance_comparison(self, 
                                  metric: str = 'solve_time',
                                  save: bool = True,
                                  show: bool = True) -> plt.Figure:
        """
        Plot performance comparison across tests.
        
        Args:
            metric: Metric to plot
            save: Save plot to file
            show: Display plot
            
        Returns:
            matplotlib Figure object
        """
        if not self.metrics_history:
            return None
        
        # Extract data
        test_names = []
        metric_values = []
        
        for test in self.metrics_history:
            if test.get('success', False):
                test_names.append(test['test_name'])
                metric_value = getattr(test['metrics'], metric, 0)
                metric_values.append(metric_value)
        
        if not metric_values:
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        
        bars = ax.bar(test_names, metric_values, color='skyblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Performance Comparison - {metric.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            filename = f"performance_comparison_{metric}.png"
            filepath = self.session_dir / filename
            fig.savefig(filepath, dpi=100, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_solve_time_trend(self, 
                            save: bool = True,
                            show: bool = True) -> plt.Figure:
        """
        Plot solve time trend over tests.
        
        Args:
            save: Save plot to file
            show: Display plot
            
        Returns:
            matplotlib Figure object
        """
        if not self.metrics_history:
            return None
        
        # Extract data
        test_indices = []
        solve_times = []
        
        for i, test in enumerate(self.metrics_history):
            if test.get('success', False):
                test_indices.append(i)
                solve_times.append(test['metrics'].solve_time)
        
        if not solve_times:
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        
        ax.plot(test_indices, solve_times, 'o-', color='blue', linewidth=2, markersize=6)
        ax.set_xlabel('Test Index')
        ax.set_ylabel('Solve Time (s)')
        ax.set_title('Solve Time Trend')
        ax.grid(True, alpha=0.3)
        
        if save:
            filename = "solve_time_trend.png"
            filepath = self.session_dir / filename
            fig.savefig(filepath, dpi=100, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_problem_size_analysis(self, 
                                 save: bool = True,
                                 show: bool = True) -> plt.Figure:
        """
        Plot problem size analysis.
        
        Args:
            save: Save plot to file
            show: Display plot
            
        Returns:
            matplotlib Figure object
        """
        if not self.metrics_history:
            return None
        
        # Extract data
        test_names = []
        num_variables = []
        num_constraints = []
        solve_times = []
        
        for test in self.metrics_history:
            if test.get('success', False):
                test_names.append(test['test_name'])
                num_variables.append(test['metrics'].num_variables)
                num_constraints.append(test['metrics'].num_constraints)
                solve_times.append(test['metrics'].solve_time)
        
        if not test_names:
            return None
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=100)
        
        # Variables vs solve time
        ax1.scatter(num_variables, solve_times, color='red', alpha=0.7, s=100)
        ax1.set_xlabel('Number of Variables')
        ax1.set_ylabel('Solve Time (s)')
        ax1.set_title('Variables vs Solve Time')
        ax1.grid(True, alpha=0.3)
        
        # Constraints vs solve time
        ax2.scatter(num_constraints, solve_times, color='green', alpha=0.7, s=100)
        ax2.set_xlabel('Number of Constraints')
        ax2.set_ylabel('Solve Time (s)')
        ax2.set_title('Constraints vs Solve Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = "problem_size_analysis.png"
            filepath = self.session_dir / filename
            fig.savefig(filepath, dpi=100, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_trajectory_quality(self, 
                               save: bool = True,
                               show: bool = True) -> plt.Figure:
        """
        Plot trajectory quality metrics.
        
        Args:
            save: Save plot to file
            show: Display plot
            
        Returns:
            matplotlib Figure object
        """
        if not self.metrics_history:
            return None
        
        # Extract data
        test_names = []
        path_lengths = []
        max_velocities = []
        tracking_errors = []
        
        for test in self.metrics_history:
            if test.get('success', False):
                test_names.append(test['test_name'])
                path_lengths.append(test['metrics'].path_length)
                max_velocities.append(test['metrics'].max_velocity)
                tracking_errors.append(test['metrics'].tracking_error)
        
        if not test_names:
            return None
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
        
        # Path length
        ax1.bar(test_names, path_lengths, color='blue', alpha=0.7)
        ax1.set_ylabel('Path Length (m)')
        ax1.set_title('Path Length Comparison')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.get_xticklabels(), rotation=45)
        
        # Max velocity
        ax2.bar(test_names, max_velocities, color='green', alpha=0.7)
        ax2.set_ylabel('Max Velocity (m/s)')
        ax2.set_title('Max Velocity Comparison')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # Tracking error
        ax3.bar(test_names, tracking_errors, color='red', alpha=0.7)
        ax3.set_ylabel('Tracking Error (m)')
        ax3.set_title('Tracking Error Comparison')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save:
            filename = "trajectory_quality.png"
            filepath = self.session_dir / filename
            fig.savefig(filepath, dpi=100, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {}
        
        # Basic statistics
        summary = self.get_performance_summary()
        
        # Detailed analysis
        successful_tests = [test for test in self.metrics_history if test.get('success', False)]
        
        if not successful_tests:
            return summary
        
        # Solve time analysis
        solve_times = [test['metrics'].solve_time for test in successful_tests]
        summary['solve_time_stats'] = {
            'mean': np.mean(solve_times),
            'std': np.std(solve_times),
            'min': np.min(solve_times),
            'max': np.max(solve_times),
            'median': np.median(solve_times)
        }
        
        # Problem size analysis
        num_variables = [test['metrics'].num_variables for test in successful_tests]
        num_constraints = [test['metrics'].num_constraints for test in successful_tests]
        
        summary['problem_size_stats'] = {
            'avg_variables': np.mean(num_variables),
            'avg_constraints': np.mean(num_constraints),
            'max_variables': np.max(num_variables),
            'max_constraints': np.max(num_constraints)
        }
        
        # Trajectory quality analysis
        path_lengths = [test['metrics'].path_length for test in successful_tests]
        tracking_errors = [test['metrics'].tracking_error for test in successful_tests]
        
        summary['trajectory_quality_stats'] = {
            'avg_path_length': np.mean(path_lengths),
            'avg_tracking_error': np.mean(tracking_errors),
            'min_tracking_error': np.min(tracking_errors),
            'max_tracking_error': np.max(tracking_errors)
        }
        
        # Save report
        report_file = self.session_dir / "performance_report.json"
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary
    
    def _save_metrics(self) -> None:
        """Save metrics to file."""
        metrics_file = self.session_dir / "performance_metrics.json"
        
        # Convert to serializable format
        serializable_data = []
        for test in self.metrics_history:
            test_data = {
                'test_name': test['test_name'],
                'success': test.get('success', False),
                'metrics': {
                    'solve_time': test['metrics'].solve_time,
                    'setup_time': test['metrics'].setup_time,
                    'total_time': test['metrics'].total_time,
                    'iterations': test['metrics'].iterations,
                    'objective_value': test['metrics'].objective_value,
                    'constraint_violations': test['metrics'].constraint_violations,
                    'num_variables': test['metrics'].num_variables,
                    'num_constraints': test['metrics'].num_constraints,
                    'convergence_status': test['metrics'].convergence_status,
                    'path_length': test['metrics'].path_length,
                    'max_velocity': test['metrics'].max_velocity,
                    'tracking_error': test['metrics'].tracking_error
                }
            }
            serializable_data.append(test_data)
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    def export_to_csv(self, filename: Optional[str] = None) -> Path:
        """
        Export performance data to CSV.
        
        Args:
            filename: Optional filename
            
        Returns:
            Path to CSV file
        """
        if not self.metrics_history:
            return None
        
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for CSV export")
        
        # Create DataFrame
        data = []
        for test in self.metrics_history:
            row = {
                'test_name': test['test_name'],
                'success': test.get('success', False),
                'solve_time': test['metrics'].solve_time,
                'setup_time': test['metrics'].setup_time,
                'total_time': test['metrics'].total_time,
                'iterations': test['metrics'].iterations,
                'objective_value': test['metrics'].objective_value,
                'constraint_violations': test['metrics'].constraint_violations,
                'num_variables': test['metrics'].num_variables,
                'num_constraints': test['metrics'].num_constraints,
                'convergence_status': test['metrics'].convergence_status,
                'path_length': test['metrics'].path_length,
                'max_velocity': test['metrics'].max_velocity,
                'tracking_error': test['metrics'].tracking_error
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        if filename is None:
            filename = f"performance_data_{self.session_id}.csv"
        
        filepath = self.session_dir / filename
        df.to_csv(filepath, index=False)
        
        return filepath
