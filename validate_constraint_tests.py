"""
Validate Constraint Tests

This script validates that the converted test system properly works for each type of constraint.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test.framework.standardized_test import BaseMPCTest, TestConfig
from utils.standardized_logging import get_test_logger


class ConstraintTestValidator:
    """Validates constraint tests with standardized framework."""
    
    def __init__(self):
        self.logger = get_test_logger("constraint_validator", "INFO")
        self.results = {}
        
    def validate_constraint_type(self, constraint_type: str, test_files: list):
        """Validate tests for a specific constraint type."""
        self.logger.log_phase(f"Validating {constraint_type}", f"Testing {len(test_files)} files")
        
        results = {
            'constraint_type': constraint_type,
            'total_tests': len(test_files),
            'successful_tests': 0,
            'failed_tests': 0,
            'test_results': []
        }
        
        for test_file in test_files:
            self.logger.logger.info(f"Testing {test_file.name}")
            
            try:
                # Run the test with timeout
                result = subprocess.run(
                    [sys.executable, str(test_file)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(project_root)
                )
                
                success = result.returncode == 0
                
                test_result = {
                    'test_name': test_file.name,
                    'success': success,
                    'return_code': result.returncode,
                    'stdout': result.stdout[:200] + '...' if len(result.stdout) > 200 else result.stdout,
                    'stderr': result.stderr[:200] + '...' if len(result.stderr) > 200 else result.stderr
                }
                
                results['test_results'].append(test_result)
                
                if success:
                    results['successful_tests'] += 1
                    self.logger.log_success(f"✅ {test_file.name} - PASSED")
                else:
                    results['failed_tests'] += 1
                    self.logger.log_error(f"❌ {test_file.name} - FAILED", details={
                        'return_code': result.returncode,
                        'stderr': result.stderr[:100]
                    })
                
            except subprocess.TimeoutExpired:
                results['failed_tests'] += 1
                self.logger.log_error(f"⏰ {test_file.name} - TIMEOUT")
                results['test_results'].append({
                    'test_name': test_file.name,
                    'success': False,
                    'return_code': -1,
                    'stdout': '',
                    'stderr': 'Test timed out'
                })
            except Exception as e:
                results['failed_tests'] += 1
                self.logger.log_error(f"💥 {test_file.name} - ERROR: {e}")
                results['test_results'].append({
                    'test_name': test_file.name,
                    'success': False,
                    'return_code': -1,
                    'stdout': '',
                    'stderr': str(e)
                })
        
        return results
    
    def validate_all_constraint_types(self):
        """Validate all constraint types."""
        self.logger.start_test()
        
        # Define constraint types and their test files
        constraint_tests = {
            'scenario': [
                project_root / "test/integration/converted_test_scenario_contouring_integration.py",
                project_root / "test/integration/converted_test_scenario_contouring_confirmation.py",
                project_root / "test/integration/converted_test_final_scenario_contouring.py",
                project_root / "test/integration/converted_test_working_scenario_contouring.py",
                project_root / "test/integration/converted_test_proper_scenario_constraints.py"
            ],
            'gaussian': [
                project_root / "test/integration/constraints/gaussian/converted_gaussian_and_contouring_constraints_with_contouring_objective.py"
            ],
            'linear': [
                project_root / "test/integration/constraints/linear/converted_linear_constraints_contouring_objective.py",
                project_root / "test/integration/constraints/linear/converted_linear_and_contouring_constraints_with_contouring_objective.py",
                project_root / "test/integration/constraints/linear/converted_linear_and_contouring_constraints_contouring_objective.py"
            ],
            'ellipsoid': [
                project_root / "test/integration/constraints/ellipsoid/converted_ellipsoid_and_contouring_constraints_with_contouring_objective.py"
            ],
            'decomposition': [
                project_root / "test/integration/constraints/decomp/converted_decomp_and_contouring_constraints_with_contouring_objective.py"
            ],
            'goal': [
                project_root / "test/integration/objective/goal/converted_goal_objective_integration_test.py",
                project_root / "test/integration/objective/contouring/converted_goal_contouring_integration_test.py"
            ]
        }
        
        # Filter existing files
        for constraint_type, test_files in constraint_tests.items():
            constraint_tests[constraint_type] = [f for f in test_files if f.exists()]
        
        # Validate each constraint type
        for constraint_type, test_files in constraint_tests.items():
            if test_files:
                self.results[constraint_type] = self.validate_constraint_type(constraint_type, test_files)
            else:
                self.logger.log_warning(f"No test files found for {constraint_type}")
        
        # Generate summary
        self.generate_validation_summary()
        
        self.logger.end_test()
        return self.results
    
    def generate_validation_summary(self):
        """Generate validation summary."""
        self.logger.log_phase("Validation Summary", "Generating comprehensive report")
        
        total_tests = sum(r['total_tests'] for r in self.results.values())
        total_successful = sum(r['successful_tests'] for r in self.results.values())
        total_failed = sum(r['failed_tests'] for r in self.results.values())
        
        self.logger.logger.info(f"📊 VALIDATION SUMMARY")
        self.logger.logger.info(f"Total constraint types: {len(self.results)}")
        self.logger.logger.info(f"Total tests: {total_tests}")
        self.logger.logger.info(f"Successful: {total_successful} ✅")
        self.logger.logger.info(f"Failed: {total_failed} ❌")
        self.logger.logger.info(f"Success rate: {(total_successful/total_tests)*100:.1f}%" if total_tests > 0 else "N/A")
        
        # Per-constraint-type summary
        for constraint_type, result in self.results.items():
            success_rate = (result['successful_tests'] / result['total_tests']) * 100 if result['total_tests'] > 0 else 0
            self.logger.logger.info(f"{constraint_type.title()}: {result['successful_tests']}/{result['total_tests']} ({success_rate:.1f}%)")
            
            if result['failed_tests'] > 0:
                failed_tests = [t['test_name'] for t in result['test_results'] if not t['success']]
                self.logger.log_warning(f"  Failed: {', '.join(failed_tests)}")


def main():
    """Main validation function."""
    print("🧪 Validating Converted Constraint Tests")
    print("=" * 60)
    
    validator = ConstraintTestValidator()
    results = validator.validate_all_constraint_types()
    
    print("\n" + "=" * 60)
    print("📊 FINAL VALIDATION REPORT")
    print("=" * 60)
    
    total_tests = sum(r['total_tests'] for r in results.values())
    total_successful = sum(r['successful_tests'] for r in results.values())
    total_failed = sum(r['failed_tests'] for r in results.values())
    
    print(f"Total constraint types: {len(results)}")
    print(f"Total tests: {total_tests}")
    print(f"Successful: {total_successful} ✅")
    print(f"Failed: {total_failed} ❌")
    print(f"Success rate: {(total_successful/total_tests)*100:.1f}%" if total_tests > 0 else "N/A")
    
    # Per-constraint-type details
    for constraint_type, result in results.items():
        success_rate = (result['successful_tests'] / result['total_tests']) * 100 if result['total_tests'] > 0 else 0
        status = "✅ WORKING" if success_rate >= 80 else "⚠️ PARTIAL" if success_rate > 0 else "❌ FAILED"
        print(f"\n{constraint_type.title()} Constraints: {status}")
        print(f"  Tests: {result['successful_tests']}/{result['total_tests']} ({success_rate:.1f}%)")
        
        if result['failed_tests'] > 0:
            failed_tests = [t['test_name'] for t in result['test_results'] if not t['success']]
            print(f"  Failed: {', '.join(failed_tests)}")
    
    return results


if __name__ == "__main__":
    results = main()
