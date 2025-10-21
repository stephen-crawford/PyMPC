"""
Demo framework for showcasing different constraint combinations.

This module provides a comprehensive demo framework for easily setting up
and running MPC tests with different constraint combinations, objectives,
and scenarios.
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
import json
from datetime import datetime

from .logger import MPCLogger, MPCProfiler
from .advanced_visualizer import MPCVisualizer
from .test_config import TestConfig, TestConfigBuilder, PredefinedTestConfigs
from .performance_monitor import PerformanceMonitor, PerformanceMetrics


class MPCDemoFramework:
    """
    Comprehensive demo framework for MPC testing and analysis.
    """
    
    def __init__(self, 
                 output_dir: str = "demo_outputs",
                 enable_logging: bool = True,
                 enable_visualization: bool = True,
                 enable_performance_monitoring: bool = True):
        """
        Initialize MPC demo framework.
        
        Args:
            output_dir: Directory for demo outputs
            enable_logging: Enable logging
            enable_visualization: Enable visualization
            enable_performance_monitoring: Enable performance monitoring
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create session directory
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"demo_session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.enable_logging = enable_logging
        self.enable_visualization = enable_visualization
        self.enable_performance_monitoring = enable_performance_monitoring
        
        if enable_logging:
            self.logger = MPCLogger(
                log_dir=str(self.session_dir / "logs"),
                log_level="INFO"
            )
            self.profiler = MPCProfiler(self.logger)
        else:
            self.logger = None
            self.profiler = None
        
        if enable_visualization:
            self.visualizer = MPCVisualizer(
                save_dir=str(self.session_dir / "plots")
            )
        else:
            self.visualizer = None
        
        if enable_performance_monitoring:
            self.performance_monitor = PerformanceMonitor(
                log_dir=str(self.session_dir / "performance")
            )
        else:
            self.performance_monitor = None
        
        # Demo results
        self.demo_results = []
        self.current_demo = None
    
    def run_demo(self, 
                config: TestConfig,
                solver_func: Callable,
                demo_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a demo with the given configuration.
        
        Args:
            config: Test configuration
            solver_func: Solver function that takes config and returns results
            demo_name: Optional demo name
            
        Returns:
            Demo results dictionary
        """
        if demo_name is None:
            demo_name = config.test_name
        
        if self.logger:
            self.logger.log_optimization_start(
                test_name=demo_name,
                horizon_length=config.horizon_length,
                dt=config.dt,
                state_dim=config.state_dim,
                control_dim=config.control_dim,
                num_constraints=len(config.constraints)
            )
        
        if self.performance_monitor:
            self.performance_monitor.start_test(demo_name, asdict(config))
        
        # Start profiling
        if self.profiler:
            self.profiler.start_profile("total_demo")
            self.profiler.start_profile("solver_setup")
        
        try:
            # Run solver
            start_time = time.time()
            results = solver_func(config)
            solve_time = time.time() - start_time
            
            if self.profiler:
                self.profiler.end_profile("solver_setup")
                self.profiler.start_profile("post_processing")
            
            # Process results
            success = results is not None and results.get('success', False)
            
            if success:
                # Extract trajectory and controls
                trajectory = results.get('states', np.array([]))
                controls = results.get('controls', np.array([]))
                
                # Record performance metrics
                if self.performance_monitor:
                    self.performance_monitor.record_solve_time(solve_time)
                    self.performance_monitor.record_optimization_metrics(
                        iterations=results.get('iterations', 0),
                        objective_value=results.get('objective_value', 0.0),
                        convergence_status=results.get('convergence_status', 'unknown')
                    )
                    self.performance_monitor.record_problem_size(
                        num_variables=results.get('num_variables', 0),
                        num_constraints=results.get('num_constraints', 0)
                    )
                    
                    if trajectory.size > 0:
                        self.performance_monitor.record_trajectory_metrics(
                            trajectory, 
                            reference_path=config.reference_path
                        )
                
                # Create visualizations
                if self.visualizer and trajectory.size > 0:
                    self._create_visualizations(
                        trajectory, controls, config, demo_name
                    )
                
                # Log results
                if self.logger:
                    self.logger.log_optimization_end(
                        success=True,
                        solve_time=solve_time,
                        objective_value=results.get('objective_value'),
                        iterations=results.get('iterations'),
                        constraint_violations=results.get('constraint_violations', 0)
                    )
                    
                    if trajectory.size > 0:
                        self.logger.log_trajectory_analysis(
                            trajectory,
                            reference_path=config.reference_path,
                            obstacles=self._extract_obstacles(config)
                        )
            
            else:
                # Log failure
                if self.logger:
                    self.logger.log_optimization_end(
                        success=False,
                        solve_time=solve_time
                    )
            
            # End performance monitoring
            if self.performance_monitor:
                metrics = self.performance_monitor.end_test(success)
            
            if self.profiler:
                self.profiler.end_profile("post_processing")
                self.profiler.end_profile("total_demo")
            
            # Store demo results
            demo_result = {
                'demo_name': demo_name,
                'config': asdict(config),
                'success': success,
                'solve_time': solve_time,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.performance_monitor:
                demo_result['performance_metrics'] = asdict(metrics)
            
            if self.profiler:
                demo_result['profile_summary'] = self.profiler.get_profile_summary()
            
            self.demo_results.append(demo_result)
            
            return demo_result
            
        except Exception as e:
            # Log error
            if self.logger:
                self.logger.logger.error(f"Demo failed: {e}")
            
            if self.performance_monitor:
                self.performance_monitor.end_test(False)
            
            if self.profiler:
                self.profiler.end_profile("total_demo")
            
            return {
                'demo_name': demo_name,
                'config': asdict(config),
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_predefined_demos(self, 
                            demo_names: List[str],
                            solver_func: Callable) -> List[Dict[str, Any]]:
        """
        Run predefined demos.
        
        Args:
            demo_names: List of predefined demo names
            solver_func: Solver function
            
        Returns:
            List of demo results
        """
        results = []
        
        for demo_name in demo_names:
            if demo_name == "curving_road_ellipsoid":
                config = PredefinedTestConfigs.curving_road_ellipsoid()
            elif demo_name == "curving_road_gaussian":
                config = PredefinedTestConfigs.curving_road_gaussian()
            elif demo_name == "curving_road_scenario":
                config = PredefinedTestConfigs.curving_road_scenario()
            elif demo_name == "goal_reaching":
                config = PredefinedTestConfigs.goal_reaching()
            elif demo_name == "combined_constraints":
                config = PredefinedTestConfigs.combined_constraints()
            else:
                if self.logger:
                    self.logger.logger.warning(f"Unknown demo: {demo_name}")
                continue
            
            result = self.run_demo(config, solver_func, demo_name)
            results.append(result)
        
        return results
    
    def run_custom_demo(self, 
                      demo_name: str,
                      config_builder: TestConfigBuilder,
                      solver_func: Callable) -> Dict[str, Any]:
        """
        Run a custom demo.
        
        Args:
            demo_name: Name of the demo
            config_builder: Test configuration builder
            solver_func: Solver function
            
        Returns:
            Demo results
        """
        config = config_builder.build()
        return self.run_demo(config, solver_func, demo_name)
    
    def run_benchmark_suite(self, 
                           solver_func: Callable,
                           benchmark_configs: List[TestConfig]) -> Dict[str, Any]:
        """
        Run a benchmark suite.
        
        Args:
            solver_func: Solver function
            benchmark_configs: List of benchmark configurations
            
        Returns:
            Benchmark results
        """
        if self.logger:
            self.logger.logger.info(f"Starting benchmark suite with {len(benchmark_configs)} tests")
        
        benchmark_results = []
        
        for i, config in enumerate(benchmark_configs):
            if self.logger:
                self.logger.logger.info(f"Running benchmark {i+1}/{len(benchmark_configs)}: {config.test_name}")
            
            result = self.run_demo(config, solver_func, f"benchmark_{i+1}")
            benchmark_results.append(result)
        
        # Generate benchmark summary
        successful_tests = [r for r in benchmark_results if r.get('success', False)]
        solve_times = [r.get('solve_time', 0) for r in successful_tests]
        
        benchmark_summary = {
            'total_tests': len(benchmark_results),
            'successful_tests': len(successful_tests),
            'success_rate': len(successful_tests) / len(benchmark_results) * 100,
            'average_solve_time': np.mean(solve_times) if solve_times else 0,
            'min_solve_time': np.min(solve_times) if solve_times else 0,
            'max_solve_time': np.max(solve_times) if solve_times else 0,
            'results': benchmark_results
        }
        
        if self.logger:
            self.logger.logger.info(f"Benchmark suite completed: {benchmark_summary['success_rate']:.1f}% success rate")
        
        return benchmark_summary
    
    def create_comparison_report(self, 
                               demo_names: List[str],
                               metrics: List[str] = None) -> Dict[str, Any]:
        """
        Create comparison report for demos.
        
        Args:
            demo_names: List of demo names to compare
            metrics: List of metrics to compare
            
        Returns:
            Comparison report
        """
        if metrics is None:
            metrics = ['solve_time', 'success', 'iterations', 'objective_value']
        
        # Filter results by demo names
        filtered_results = [r for r in self.demo_results if r['demo_name'] in demo_names]
        
        if not filtered_results:
            return {}
        
        # Create comparison data
        comparison_data = {}
        for metric in metrics:
            values = []
            for result in filtered_results:
                if metric == 'success':
                    values.append(result.get('success', False))
                elif metric in result.get('results', {}):
                    values.append(result['results'][metric])
                elif metric in result.get('performance_metrics', {}):
                    values.append(result['performance_metrics'][metric])
                else:
                    values.append(None)
            comparison_data[metric] = values
        
        # Create comparison report
        report = {
            'demo_names': demo_names,
            'metrics': metrics,
            'comparison_data': comparison_data,
            'summary': {
                'total_demos': len(filtered_results),
                'successful_demos': sum(1 for r in filtered_results if r.get('success', False)),
                'average_solve_time': np.mean([r.get('solve_time', 0) for r in filtered_results if r.get('success', False)])
            }
        }
        
        return report
    
    def generate_session_report(self) -> Dict[str, Any]:
        """Generate comprehensive session report."""
        if not self.demo_results:
            return {}
        
        # Basic statistics
        total_demos = len(self.demo_results)
        successful_demos = sum(1 for r in self.demo_results if r.get('success', False))
        solve_times = [r.get('solve_time', 0) for r in self.demo_results if r.get('success', False)]
        
        # Performance analysis
        performance_analysis = {}
        if self.performance_monitor:
            performance_analysis = self.performance_monitor.get_performance_summary()
        
        # Create session report
        session_report = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'total_demos': total_demos,
            'successful_demos': successful_demos,
            'success_rate': successful_demos / total_demos * 100,
            'average_solve_time': np.mean(solve_times) if solve_times else 0,
            'min_solve_time': np.min(solve_times) if solve_times else 0,
            'max_solve_time': np.max(solve_times) if solve_times else 0,
            'performance_analysis': performance_analysis,
            'demo_results': self.demo_results
        }
        
        # Save report
        report_file = self.session_dir / "session_report.json"
        with open(report_file, 'w') as f:
            json.dump(session_report, f, indent=2, default=str)
        
        if self.logger:
            self.logger.print_session_summary()
        
        return session_report
    
    def _create_visualizations(self, 
                              trajectory: np.ndarray,
                              controls: np.ndarray,
                              config: TestConfig,
                              demo_name: str) -> None:
        """Create visualizations for demo."""
        if not self.visualizer:
            return
        
        # Extract reference path
        reference_path = None
        if config.reference_path:
            # Convert reference path to numpy array
            if isinstance(config.reference_path, dict):
                # Handle different reference path types
                pass  # Implementation depends on reference path format
            else:
                reference_path = np.array(config.reference_path)
        
        # Extract obstacles
        obstacles = self._extract_obstacles(config)
        
        # Create 2D trajectory plot
        self.visualizer.plot_trajectory_2d(
            trajectory=trajectory,
            reference_path=reference_path,
            obstacles=obstacles,
            title=f"{demo_name} - Trajectory",
            save=True,
            show=False
        )
        
        # Create state evolution plot
        if trajectory.shape[0] > 1:
            state_names = ['x', 'y', 'yaw', 'v', 'delta'][:trajectory.shape[0]]
            self.visualizer.plot_state_evolution(
                trajectory=trajectory,
                dt=config.dt,
                state_names=state_names,
                title=f"{demo_name} - State Evolution",
                save=True,
                show=False
            )
        
        # Create control evolution plot
        if controls.size > 0:
            control_names = ['acceleration', 'steering_rate']
            self.visualizer.plot_control_evolution(
                controls=controls,
                dt=config.dt,
                control_names=control_names,
                title=f"{demo_name} - Control Evolution",
                save=True,
                show=False
            )
    
    def _extract_obstacles(self, config: TestConfig) -> List[Dict[str, Any]]:
        """Extract obstacles from configuration."""
        obstacles = []
        
        for constraint in config.constraints:
            if constraint['type'] == 'ellipsoid':
                obstacles.extend(constraint.get('obstacles', []))
            elif constraint['type'] == 'gaussian':
                # Convert Gaussian obstacles to ellipsoid format for visualization
                for obs in constraint.get('uncertain_obstacles', []):
                    if 'mean' in obs and 'shape' in obs:
                        obstacles.append({
                            'center': obs['mean'],
                            'shape': obs['shape'],
                            'rotation': 0.0
                        })
            elif constraint['type'] == 'scenario':
                # Use first scenario for visualization
                scenarios = constraint.get('scenarios', [])
                if scenarios:
                    obstacles.extend(scenarios[0].get('obstacles', []))
        
        return obstacles
    
    def get_session_dir(self) -> Path:
        """Get session directory path."""
        return self.session_dir
    
    def cleanup(self) -> None:
        """Clean up demo framework."""
        if self.logger:
            self.logger.save_session_summary()
        
        if self.performance_monitor:
            self.performance_monitor.generate_performance_report()
        
        if self.visualizer:
            self.visualizer.save_session_summary({
                'session_id': self.session_id,
                'total_demos': len(self.demo_results),
                'successful_demos': sum(1 for r in self.demo_results if r.get('success', False))
            })


def create_demo_framework(output_dir: str = "demo_outputs",
                         enable_logging: bool = True,
                         enable_visualization: bool = True,
                         enable_performance_monitoring: bool = True) -> MPCDemoFramework:
    """
    Create a new MPC demo framework.
    
    Args:
        output_dir: Directory for demo outputs
        enable_logging: Enable logging
        enable_visualization: Enable visualization
        enable_performance_monitoring: Enable performance monitoring
        
    Returns:
        MPC demo framework
    """
    return MPCDemoFramework(
        output_dir=output_dir,
        enable_logging=enable_logging,
        enable_visualization=enable_visualization,
        enable_performance_monitoring=enable_performance_monitoring
    )
