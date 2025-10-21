#!/usr/bin/env python3
"""
Test script for the visualization and logging framework.

This script tests the basic functionality without requiring all dependencies.
"""

import sys
import os
import numpy as np
import time

# Add the pympc module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pympc'))

def test_basic_imports():
    """Test basic imports."""
    print("Testing basic imports...")
    
    try:
        from utils.logger import MPCLogger, MPCProfiler
        print("✓ Logger imports successful")
    except ImportError as e:
        print(f"✗ Logger import failed: {e}")
        return False
    
    try:
        from utils.test_config import TestConfigBuilder, PredefinedTestConfigs
        print("✓ Test config imports successful")
    except ImportError as e:
        print(f"✗ Test config import failed: {e}")
        return False
    
    try:
        from utils.advanced_visualizer import MPCVisualizer
        print("✓ Visualizer imports successful")
    except ImportError as e:
        print(f"✗ Visualizer import failed: {e}")
        return False
    
    try:
        from utils.performance_monitor import PerformanceMonitor
        print("✓ Performance monitor imports successful")
    except ImportError as e:
        print(f"✗ Performance monitor import failed: {e}")
        return False
    
    return True


def test_logger():
    """Test logger functionality."""
    print("\nTesting logger...")
    
    try:
        from utils.logger import MPCLogger, MPCProfiler
        
        # Create logger
        logger = MPCLogger(log_dir="test_logs", log_level="INFO")
        
        # Test logging
        logger.log_optimization_start(
            test_name="test_optimization",
            horizon_length=20,
            dt=0.1,
            state_dim=5,
            control_dim=2,
            num_constraints=3
        )
        
        logger.log_optimization_end(
            success=True,
            solve_time=1.5,
            objective_value=10.2,
            iterations=25
        )
        
        # Test profiler
        profiler = MPCProfiler(logger)
        profiler.start_profile("test_section")
        time.sleep(0.1)
        profiler.end_profile("test_section")
        
        print("✓ Logger functionality works")
        return True
        
    except Exception as e:
        print(f"✗ Logger test failed: {e}")
        return False


def test_test_config():
    """Test test configuration functionality."""
    print("\nTesting test configuration...")
    
    try:
        from utils.test_config import TestConfigBuilder, PredefinedTestConfigs
        
        # Create a simple configuration
        config = (TestConfigBuilder("test_config")
                  .set_mpc_params(horizon_length=20, dt=0.1)
                  .set_initial_state([0.0, 0.0, 0.0, 1.0, 0.0])
                  .build())
        
        print(f"✓ Test config created: {config.test_name}")
        print(f"  Horizon: {config.horizon_length}")
        print(f"  Initial state: {config.initial_state}")
        
        # Test predefined configurations
        predefined_configs = [
            "curving_road_ellipsoid",
            "curving_road_gaussian",
            "curving_road_scenario",
            "goal_reaching",
            "combined_constraints"
        ]
        
        for config_name in predefined_configs:
            if hasattr(PredefinedTestConfigs, config_name):
                config_func = getattr(PredefinedTestConfigs, config_name)
                config = config_func()
                print(f"✓ Predefined config '{config_name}' created")
            else:
                print(f"✗ Predefined config '{config_name}' not found")
        
        return True
        
    except Exception as e:
        print(f"✗ Test config test failed: {e}")
        return False


def test_visualizer():
    """Test visualizer functionality."""
    print("\nTesting visualizer...")
    
    try:
        from utils.advanced_visualizer import MPCVisualizer
        
        # Create visualizer
        visualizer = MPCVisualizer(save_dir="test_plots")
        
        # Create test data
        t = np.linspace(0, 2*np.pi, 50)
        trajectory = np.array([
            0.3 * t,
            2 * np.sin(0.3 * t),
            np.zeros_like(t),
            np.ones_like(t),
            np.zeros_like(t)
        ])
        
        reference_path = np.column_stack([0.3 * t, 2 * np.sin(0.3 * t)])
        
        obstacles = [
            {'center': np.array([2.0, 1.0]), 'shape': np.array([0.5, 0.3]), 'rotation': 0.0},
            {'center': np.array([4.0, -1.0]), 'shape': np.array([0.4, 0.4]), 'rotation': 0.0}
        ]
        
        # Test 2D trajectory plot
        fig = visualizer.plot_trajectory_2d(
            trajectory=trajectory,
            reference_path=reference_path,
            obstacles=obstacles,
            title="Test Trajectory",
            save=True,
            show=False
        )
        
        print("✓ 2D trajectory plot created")
        
        # Test state evolution plot
        fig = visualizer.plot_state_evolution(
            trajectory=trajectory,
            dt=0.1,
            state_names=['x', 'y', 'yaw', 'v', 'delta'],
            title="Test State Evolution",
            save=True,
            show=False
        )
        
        print("✓ State evolution plot created")
        
        # Test performance metrics plot
        metrics = {
            'solve_time': 1.5,
            'iterations': 25,
            'objective_value': 10.2,
            'constraint_violations': 0
        }
        
        fig = visualizer.plot_performance_metrics(
            metrics=metrics,
            title="Test Performance Metrics",
            save=True,
            show=False
        )
        
        print("✓ Performance metrics plot created")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualizer test failed: {e}")
        return False


def test_performance_monitor():
    """Test performance monitor functionality."""
    print("\nTesting performance monitor...")
    
    try:
        from utils.performance_monitor import PerformanceMonitor
        
        # Create performance monitor
        monitor = PerformanceMonitor(log_dir="test_performance")
        
        # Test monitoring
        monitor.start_test("test_performance", {"test": "config"})
        
        monitor.record_solve_time(1.5)
        monitor.record_optimization_metrics(
            iterations=25,
            objective_value=10.2,
            convergence_status="optimal"
        )
        monitor.record_problem_size(
            num_variables=100,
            num_constraints=50
        )
        
        # Create test trajectory
        t = np.linspace(0, 2*np.pi, 50)
        trajectory = np.array([
            0.3 * t,
            2 * np.sin(0.3 * t),
            np.zeros_like(t),
            np.ones_like(t),
            np.zeros_like(t)
        ])
        
        reference_path = np.column_stack([0.3 * t, 2 * np.sin(0.3 * t)])
        
        monitor.record_trajectory_metrics(trajectory, reference_path)
        
        # End test
        metrics = monitor.end_test(success=True)
        
        print("✓ Performance monitoring works")
        print(f"  Solve time: {metrics.solve_time}")
        print(f"  Iterations: {metrics.iterations}")
        print(f"  Path length: {metrics.path_length:.3f}")
        
        # Test performance summary
        summary = monitor.get_performance_summary()
        print(f"✓ Performance summary: {summary}")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance monitor test failed: {e}")
        return False


def test_demo_framework():
    """Test demo framework functionality."""
    print("\nTesting demo framework...")
    
    try:
        from utils.demo_framework import create_demo_framework
        from utils.test_config import TestConfigBuilder
        
        # Create demo framework
        demo = create_demo_framework(
            output_dir="test_demo_outputs",
            enable_logging=True,
            enable_visualization=True,
            enable_performance_monitoring=True
        )
        
        print("✓ Demo framework created")
        
        # Create test configuration
        config = (TestConfigBuilder("test_demo")
                  .set_mpc_params(horizon_length=10, dt=0.1)
                  .set_initial_state([0.0, 0.0, 0.0, 1.0, 0.0])
                  .build())
        
        print("✓ Test configuration created")
        
        # Mock solver function
        def mock_solver(config):
            return {
                'success': True,
                'states': np.random.rand(5, 11),
                'controls': np.random.rand(2, 10),
                'solve_time': 1.0,
                'iterations': 20,
                'objective_value': 5.0
            }
        
        # Test demo (without actually running MPC)
        print("✓ Mock solver created")
        
        # Cleanup
        demo.cleanup()
        print("✓ Demo framework cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Demo framework test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("MPC Visualization and Logging Framework Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Logger", test_logger),
        ("Test Configuration", test_test_config),
        ("Visualizer", test_visualizer),
        ("Performance Monitor", test_performance_monitor),
        ("Demo Framework", test_demo_framework)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'-'*20} {test_name} {'-'*20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                print(f"✓ {test_name} passed")
            else:
                print(f"✗ {test_name} failed")
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed_tests = sum(1 for r in results.values() if r)
    total_tests = len(results)
    
    print(f"Total tests: {total_tests}")
    print(f"Passed tests: {passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠ {total_tests - passed_tests} tests failed")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
