"""
Standardized Test Framework for PyMPC

This module provides a comprehensive test framework with:
- Easy test implementation and modification
- Clear failure explanations and diagnostics
- Automatic test discovery and execution
- Performance monitoring and reporting
- Integration with logging and visualization systems
"""

import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.standardized_logging import get_test_logger, PerformanceMonitor
from utils.standardized_visualization import TestVisualizationManager, VisualizationConfig, VisualizationMode


@dataclass
class TestConfig:
    """Configuration for test execution."""
    test_name: str
    description: str = ""
    timeout: float = 60.0
    max_iterations: int = 100
    goal_tolerance: float = 1.0
    enable_visualization: bool = True
    save_outputs: bool = True
    log_level: str = "INFO"
    visualization_mode: VisualizationMode = VisualizationMode.REALTIME
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of test execution."""
    test_name: str
    success: bool
    duration: float
    iterations: int
    final_error: float
    convergence_rate: float
    performance_metrics: Dict[str, float]
    error_log: List[str]
    visualization_data: Dict[str, Any]


class BaseMPCTest(ABC):
    """
    Base class for standardized MPC tests.
    
    Provides:
    - Standardized test structure
    - Automatic visualization setup
    - Performance monitoring
    - Error handling and reporting
    - Test result generation
    """
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.logger = get_test_logger(config.test_name, config.log_level)
        self.performance_monitor = PerformanceMonitor(self.logger, "test_execution")
        self.visualizer = None
        self.test_result = None
        
        # Test state
        self.current_iteration = 0
        self.start_time = None
        self.trajectory_history = []
        self.error_history = []
        self.constraint_violations = []
        
        # Initialize visualization if enabled
        if config.enable_visualization:
            self._setup_visualization()
    
    def _setup_visualization(self):
        """Setup visualization system."""
        viz_config = VisualizationConfig(
            mode=self.config.visualization_mode,
            realtime=True,
            show_constraint_projection=True,
            save_animation=self.config.save_outputs,
            save_plots=self.config.save_outputs,
            output_dir=f"test_results/{self.config.test_name}/visualizations"
        )
        
        self.visualizer = TestVisualizationManager(self.config.test_name)
        self.visualizer.initialize(viz_config)
    
    @abstractmethod
    def setup_test_environment(self):
        """Setup test environment - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def create_mpc_system(self):
        """Create MPC system - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def run_mpc_iteration(self, iteration: int) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Run a single MPC iteration.
        
        Returns:
            success: bool - Whether iteration was successful
            error: float - Current error value
            state: Dict - Current state information
        """
        pass
    
    def run_test(self) -> TestResult:
        """Run the complete test."""
        self.logger.log_phase("Test Start", f"Starting test: {self.config.test_name}")
        self.start_time = time.time()
        
        try:
            # Setup
            self.setup_test_environment()
            mpc_system = self.create_mpc_system()
            
            # Run iterations
            success = self._run_iterations(mpc_system)
            
            # Generate results
            self.test_result = self._generate_test_result(success)
            
            # Save outputs
            if self.config.save_outputs:
                self._save_test_outputs()
            
            return self.test_result
            
        except Exception as e:
            self.logger.log_error(f"Test failed with exception: {str(e)}")
            self.logger.log_error(f"Traceback: {traceback.format_exc()}")
            
            # Generate error result
            self.test_result = self._generate_error_result(str(e))
            return self.test_result
        
        finally:
            # Cleanup
            self._cleanup()
    
    def _run_iterations(self, mpc_system) -> bool:
        """Run MPC iterations until convergence or timeout."""
        self.logger.log_phase("MPC Iterations", "Starting MPC iterations")
        
        for iteration in range(self.config.max_iterations):
            self.current_iteration = iteration
            
            try:
                # Run iteration
                success, error, state = self.run_mpc_iteration(iteration)
                
                # Update history
                self.trajectory_history.append(state)
                self.error_history.append(error)
                
                # Update visualization
                if self.visualizer:
                    self._update_visualization(state)
                
                # Check convergence
                if self._check_convergence(error):
                    self.logger.log_success(f"Converged after {iteration + 1} iterations")
                    return True
                
                # Check timeout
                if self._check_timeout():
                    self.logger.log_warning(f"Test timed out after {iteration + 1} iterations")
                    return False
                
                # Log progress
                if iteration % 10 == 0:
                    self.logger.log_info(f"Iteration {iteration}: error = {error:.4f}")
                
            except Exception as e:
                self.logger.log_error(f"Iteration {iteration} failed: {str(e)}")
                self.constraint_violations.append({
                    'iteration': iteration,
                    'error': str(e)
                })
                continue
        
        self.logger.log_warning(f"Test did not converge after {self.config.max_iterations} iterations")
        return False
    
    def _check_convergence(self, error: float) -> bool:
        """Check if the test has converged."""
        return error < self.config.goal_tolerance
    
    def _check_timeout(self) -> bool:
        """Check if the test has timed out."""
        elapsed = time.time() - self.start_time
        return elapsed > self.config.timeout
    
    def _update_visualization(self, state: Dict[str, Any]):
        """Update visualization with current state."""
        if not self.visualizer:
            return
        
        # Update vehicle state
        vehicle_state = {
            'x': state.get('x', 0),
            'y': state.get('y', 0),
            'psi': state.get('psi', 0)
        }
        self.visualizer.update_vehicle_state(vehicle_state)
        
        # Update trajectory
        self.visualizer.update_trajectory([vehicle_state])
        
        # Update constraints if available
        if 'constraints' in state:
            self.visualizer.update_constraints(state['constraints'])
        
        # Update constraint projections if available
        if 'constraint_projections' in state:
            self.visualizer.update_constraint_projection(state['constraint_projections'])
        
        # Refresh display
        self.visualizer.refresh_display()
    
    def _generate_test_result(self, success: bool) -> TestResult:
        """Generate test result."""
        duration = time.time() - self.start_time
        
        # Calculate convergence rate
        convergence_rate = 0.0
        if len(self.error_history) > 1:
            initial_error = self.error_history[0]
            final_error = self.error_history[-1]
            if initial_error > 0:
                convergence_rate = (initial_error - final_error) / initial_error
        
        # Get performance metrics
        performance_metrics = {
            'duration': duration,
            'iterations': self.current_iteration + 1,
            'convergence_rate': convergence_rate
        }
        if self.visualizer:
            viz_stats = self.visualizer.get_performance_stats()
            performance_metrics.update(viz_stats)
        
        return TestResult(
            test_name=self.config.test_name,
            success=success,
            duration=duration,
            iterations=self.current_iteration + 1,
            final_error=self.error_history[-1] if self.error_history else float('inf'),
            convergence_rate=convergence_rate,
            performance_metrics=performance_metrics,
            error_log=self.logger.get_error_log(),
            visualization_data=self._get_visualization_data()
        )
    
    def _generate_error_result(self, error_message: str) -> TestResult:
        """Generate error result."""
        duration = time.time() - self.start_time if self.start_time else 0.0
        
        return TestResult(
            test_name=self.config.test_name,
            success=False,
            duration=duration,
            iterations=self.current_iteration,
            final_error=float('inf'),
            convergence_rate=0.0,
            performance_metrics={},
            error_log=[error_message],
            visualization_data={}
        )
    
    def _get_visualization_data(self) -> Dict[str, Any]:
        """Get visualization data for result."""
        if not self.visualizer:
            return {}
        
        return {
            'trajectory_history': self.trajectory_history,
            'constraint_overlays': self.visualizer.constraint_overlays,
            'performance_stats': self.visualizer.get_performance_stats()
        }
    
    def _save_test_outputs(self):
        """Save test outputs."""
        if not self.config.save_outputs:
            return
        
        # Create output directory
        output_dir = Path(f"test_results/{self.config.test_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save test result
        result_file = output_dir / "test_result.json"
        with open(result_file, 'w') as f:
            json.dump(self.test_result.__dict__, f, indent=2, default=str)
        
        # Save trajectory data
        if self.trajectory_history:
            traj_file = output_dir / "trajectory.json"
            with open(traj_file, 'w') as f:
                json.dump(self.trajectory_history, f, indent=2)
        
        # Save error history
        if self.error_history:
            error_file = output_dir / "error_history.json"
            with open(error_file, 'w') as f:
                json.dump(self.error_history, f, indent=2)
        
        # Create animation if visualizer is available
        if self.visualizer and self.trajectory_history:
            animation_file = self.visualizer.create_animation()
            if animation_file:
                self.logger.log_info(f"Animation saved: {animation_file}")
    
    def _cleanup(self):
        """Cleanup test resources."""
        if self.visualizer:
            self.visualizer.cleanup()
        
        # Performance monitor is a context manager, no need to stop
    
    def add_constraint_overlay(self, module_name: str, overlay: Dict[str, Any]):
        """Add constraint overlay from a module."""
        if self.visualizer:
            self.visualizer.add_constraint_overlay(module_name, overlay)
    
    def get_test_summary(self) -> str:
        """Get a summary of the test results."""
        if not self.test_result:
            return "Test not completed"
        
        result = self.test_result
        status = "PASSED" if result.success else "FAILED"
        
        summary = f"""
Test: {result.test_name}
Status: {status}
Duration: {result.duration:.2f}s
Iterations: {result.iterations}
Final Error: {result.final_error:.4f}
Convergence Rate: {result.convergence_rate:.2%}
"""
        
        if result.error_log:
            summary += f"Errors: {len(result.error_log)}\n"
        
        return summary


class TestSuite:
    """Test suite for running multiple tests."""
    
    def __init__(self, suite_name: str):
        self.suite_name = suite_name
        self.tests = []
        self.results = []
    
    def add_test(self, test: BaseMPCTest):
        """Add a test to the suite."""
        self.tests.append(test)
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all tests in the suite."""
        print(f"Running test suite: {self.suite_name}")
        print("=" * 50)
        
        for test in self.tests:
            print(f"\nRunning test: {test.config.test_name}")
            result = test.run_test()
            self.results.append(result)
            
            # Print summary
            print(test.get_test_summary())
        
        return self.results
    
    def get_suite_summary(self) -> str:
        """Get summary of all test results."""
        if not self.results:
            return "No tests run"
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(r.duration for r in self.results)
        avg_duration = total_duration / total_tests if total_tests > 0 else 0
        
        summary = f"""
Test Suite: {self.suite_name}
Total Tests: {total_tests}
Passed: {passed_tests}
Failed: {failed_tests}
Success Rate: {passed_tests/total_tests:.1%}
Total Duration: {total_duration:.2f}s
Average Duration: {avg_duration:.2f}s
"""
        
        return summary


def run_single_test(test: BaseMPCTest) -> TestResult:
    """Run a single test and return result."""
    return test.run_test()


def run_test_suite(suite: TestSuite) -> List[TestResult]:
    """Run a test suite and return results."""
    return suite.run_all_tests()