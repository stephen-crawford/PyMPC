#!/usr/bin/env python3
"""
Comprehensive Test Runner for PyMPC Framework.

This script runs all tests in the organized folder structure.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def run_test_file(test_path, test_name):
    """Run a single test file and return results."""
    print(f"\n{'='*60}")
    print(f"Running {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run([sys.executable, test_path], 
                              capture_output=True, text=True, timeout=300)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {test_name} PASSED ({duration:.1f}s)")
            return True, duration, result.stdout
        else:
            print(f"❌ {test_name} FAILED ({duration:.1f}s)")
            print(f"Error: {result.stderr}")
            return False, duration, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} TIMEOUT (>5min)")
        return False, 300, "Timeout"
    except Exception as e:
        print(f"💥 {test_name} ERROR: {e}")
        return False, 0, str(e)

def main():
    """Main test runner function."""
    print("🚀 PyMPC Framework Test Runner")
    print("=" * 60)
    
    # Define test categories
    test_categories = {
        "Unit Tests": [
            "tests/unit/test_basic_functionality.py",
            "tests/unit/test_installation.py",
            "tests/unit/test_core_functionality.py"
        ],
        "Integration Tests": [
            "tests/integration/test_mpcc_proper.py",
            "tests/integration/test_path_following.py",
            "tests/integration/test_working_trajectory.py"
        ],
        "MPCC Examples": [
            "examples/mpcc/working_mpcc_demo.py",
            "examples/mpcc/test_manual_path_following.py",
            "examples/mpcc/test_simple_path_following.py"
        ],
        "Basic Examples": [
            "examples/basic/test_visualization_framework.py",
            "examples/basic/simple_mpcc_example.py"
        ],
        "Advanced Examples": [
            "examples/advanced/test_realtime_visualization.py",
            "examples/advanced/test_curving_road.py"
        ]
    }
    
    # Results tracking
    results = {
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "total_time": 0,
        "category_results": {}
    }
    
    # Run tests by category
    for category, test_files in test_categories.items():
        print(f"\n{'='*60}")
        print(f"CATEGORY: {category}")
        print(f"{'='*60}")
        
        category_results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "time": 0
        }
        
        for test_file in test_files:
            if os.path.exists(test_file):
                test_name = os.path.basename(test_file)
                results["total_tests"] += 1
                category_results["total"] += 1
                
                passed, duration, output = run_test_file(test_file, test_name)
                category_results["time"] += duration
                
                if passed:
                    results["passed_tests"] += 1
                    category_results["passed"] += 1
                else:
                    results["failed_tests"] += 1
                    category_results["failed"] += 1
            else:
                print(f"⚠️  {test_file} not found, skipping...")
        
        results["category_results"][category] = category_results
        results["total_time"] += category_results["time"]
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for category, cat_results in results["category_results"].items():
        total = cat_results["total"]
        passed = cat_results["passed"]
        failed = cat_results["failed"]
        time_taken = cat_results["time"]
        
        if total > 0:
            success_rate = (passed / total) * 100
            status = "✅" if failed == 0 else "⚠️" if passed > 0 else "❌"
            print(f"{status} {category:20} {passed:2}/{total:2} passed ({success_rate:5.1f}%) - {time_taken:5.1f}s")
        else:
            print(f"⚪ {category:20} No tests found")
    
    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")
    
    total = results["total_tests"]
    passed = results["passed_tests"]
    failed = results["failed_tests"]
    total_time = results["total_time"]
    
    if total > 0:
        success_rate = (passed / total) * 100
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ({success_rate:.1f}%)")
        print(f"Failed: {failed}")
        print(f"Total Time: {total_time:.1f}s")
        
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED!")
            return True
        else:
            print(f"\n⚠️  {failed} TESTS FAILED")
            return False
    else:
        print("❌ NO TESTS FOUND")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
