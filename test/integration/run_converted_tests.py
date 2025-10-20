"""
Test Runner for Converted Standardized Tests

This script runs all the converted integration tests using the standardized
logging, visualization, and testing framework.
"""

import sys
import os
from pathlib import Path
import time
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from test.framework.standardized_test import TestSuite
from utils.standardized_logging import get_test_logger


class ConvertedTestRunner:
    """Runs all converted integration tests."""
    
    def __init__(self):
        self.logger = get_test_logger("converted_test_runner", "INFO")
        self.test_suite = TestSuite("Converted Integration Tests")
        self.results = []
        
    def find_converted_tests(self):
        """Find all converted test files."""
        test_dir = Path(__file__).parent
        converted_tests = []
        
        for test_file in test_dir.rglob("converted_*.py"):
            converted_tests.append(test_file)
        
        return converted_tests
    
    def run_converted_tests(self):
        """Run all converted tests."""
        self.logger.start_test()
        
        # Find converted tests
        converted_tests = self.find_converted_tests()
        self.logger.log_info(f"Found {len(converted_tests)} converted tests")
        
        # Run each test
        for i, test_file in enumerate(converted_tests, 1):
            self.logger.log_phase(f"Running Test {i}/{len(converted_tests)}", f"Executing {test_file.name}")
            
            try:
                # Run the test
                result = subprocess.run(
                    [sys.executable, str(test_file)],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                # Parse result
                success = result.returncode == 0
                duration = 0  # We don't have timing from subprocess
                
                self.results.append({
                    'test_name': test_file.name,
                    'success': success,
                    'duration': duration,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'return_code': result.returncode
                })
                
                if success:
                    self.logger.log_success(f"Test {test_file.name} completed successfully")
                else:
                    self.logger.log_error(f"Test {test_file.name} failed", details={
                        'return_code': result.returncode,
                        'stderr': result.stderr[:200] + '...' if len(result.stderr) > 200 else result.stderr
                    })
                
            except subprocess.TimeoutExpired:
                self.logger.log_error(f"Test {test_file.name} timed out")
                self.results.append({
                    'test_name': test_file.name,
                    'success': False,
                    'duration': 300,
                    'stdout': '',
                    'stderr': 'Test timed out',
                    'return_code': -1
                })
            except Exception as e:
                self.logger.log_error(f"Failed to run test {test_file.name}", e)
                self.results.append({
                    'test_name': test_file.name,
                    'success': False,
                    'duration': 0,
                    'stdout': '',
                    'stderr': str(e),
                    'return_code': -1
                })
        
        # Generate summary
        self.generate_summary()
        
        self.logger.end_test()
        return self.results
    
    def generate_summary(self):
        """Generate test execution summary."""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r['success'])
        failed_tests = total_tests - successful_tests
        
        self.logger.log_info(f"Test execution summary:")
        self.logger.log_info(f"  Total tests: {total_tests}")
        self.logger.log_info(f"  Successful: {successful_tests}")
        self.logger.log_info(f"  Failed: {failed_tests}")
        self.logger.log_info(f"  Success rate: {(successful_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            self.logger.log_warning("Failed tests:")
            for result in self.results:
                if not result['success']:
                    self.logger.log_warning(f"  - {result['test_name']}: {result['stderr'][:100]}...")


def run_all_converted_tests():
    """Run all converted integration tests."""
    print("🚀 Running All Converted Integration Tests")
    print("=" * 60)
    
    runner = ConvertedTestRunner()
    results = runner.run_converted_tests()
    
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - successful_tests
    
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests} ✅")
    print(f"Failed: {failed_tests} ❌")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    if failed_tests > 0:
        print(f"\n❌ FAILED TESTS:")
        for result in results:
            if not result['success']:
                print(f"   - {result['test_name']}: {result['stderr'][:100]}...")
    
    return results


if __name__ == "__main__":
    results = run_all_converted_tests()
