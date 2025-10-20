"""
Replace Old Tests with Corrected Standardized Versions

This script replaces the old integration tests with the corrected standardized
versions that have proper logging, visualization, and testing framework.
"""

import os
import shutil
from pathlib import Path


def replace_old_tests():
    """Replace old tests with corrected standardized versions."""
    print("🔄 Replacing old tests with corrected standardized versions...")
    
    # Define replacement mappings
    replacements = {
        # Main integration tests
        "test/integration/test_scenario_contouring_integration.py": "test/integration/converted_test_scenario_contouring_integration.py",
        "test/integration/test_scenario_contouring_confirmation.py": "test/integration/converted_test_scenario_contouring_confirmation.py",
        "test/integration/test_final_scenario_contouring.py": "test/integration/converted_test_final_scenario_contouring.py",
        "test/integration/test_working_scenario_contouring.py": "test/integration/converted_test_working_scenario_contouring.py",
        "test/integration/test_proper_scenario_constraints.py": "test/integration/converted_test_proper_scenario_constraints.py",
        "test/integration/test_final_mpc_implementation.py": "test/integration/converted_test_final_mpc_implementation.py",
        "test/integration/test_guaranteed_goal_reaching.py": "test/integration/converted_test_guaranteed_goal_reaching.py",
        "test/integration/test_fixed_solver.py": "test/integration/converted_test_fixed_solver.py",
        "test/integration/test_complete_mpc_system.py": "test/integration/converted_test_complete_mpc_system.py",
        "test/integration/test_working_mpc_goal_reaching.py": "test/integration/converted_test_working_mpc_goal_reaching.py",
        "test/integration/test_working_scenario_mpc.py": "test/integration/converted_test_working_scenario_mpc.py",
        "test/integration/test_all_constraint_types.py": "test/integration/converted_test_all_constraint_types.py",
        "test/integration/test_simple_goal_reaching.py": "test/integration/converted_test_simple_goal_reaching.py",
        
        # Constraint-specific tests
        "test/integration/constraints/scenario/scenario_and_contouring_constraints_with_contouring_objective.py": "test/integration/constraints/scenario/converted_scenario_and_contouring_constraints_with_contouring_objective.py",
        "test/integration/constraints/gaussian/gaussian_and_contouring_constraints_with_contouring_objective.py": "test/integration/constraints/gaussian/converted_gaussian_and_contouring_constraints_with_contouring_objective.py",
        "test/integration/constraints/linear/linear_constraints_contouring_objective.py": "test/integration/constraints/linear/converted_linear_constraints_contouring_objective.py",
        "test/integration/constraints/linear/linear_and_contouring_constraints_with_contouring_objective.py": "test/integration/constraints/linear/converted_linear_and_contouring_constraints_with_contouring_objective.py",
        "test/integration/constraints/linear/linear_and_contouring_constraints_contouring_objective.py": "test/integration/constraints/linear/converted_linear_and_contouring_constraints_contouring_objective.py",
        "test/integration/constraints/ellipsoid/ellipsoid_and_contouring_constraints_with_contouring_objective.py": "test/integration/constraints/ellipsoid/converted_ellipsoid_and_contouring_constraints_with_contouring_objective.py",
        "test/integration/constraints/decomp/decomp_and_contouring_constraints_with_contouring_objective.py": "test/integration/constraints/decomp/converted_decomp_and_contouring_constraints_with_contouring_objective.py",
        
        # Objective-specific tests
        "test/integration/objective/goal/goal_objective_integration_test.py": "test/integration/objective/goal/converted_goal_objective_integration_test.py",
        "test/integration/objective/contouring/goal_contouring_integration_test.py": "test/integration/objective/contouring/converted_goal_contouring_integration_test.py"
    }
    
    replaced_count = 0
    backup_count = 0
    
    for old_test, new_test in replacements.items():
        old_path = Path(old_test)
        new_path = Path(new_test)
        
        if old_path.exists() and new_path.exists():
            print(f"🔄 Replacing: {old_path}")
            
            # Create backup of old test
            backup_path = old_path.with_suffix('.py.old_backup')
            shutil.copy2(old_path, backup_path)
            backup_count += 1
            print(f"   📦 Backed up to: {backup_path}")
            
            # Copy new standardized test to replace old test
            shutil.copy2(new_path, old_path)
            replaced_count += 1
            print(f"   ✅ Replaced with standardized version")
            
        elif not old_path.exists():
            print(f"⚠️  Old test not found: {old_path}")
        elif not new_path.exists():
            print(f"⚠️  New test not found: {new_path}")
    
    print(f"\n✅ Replacement Summary:")
    print(f"   Replaced: {replaced_count} tests")
    print(f"   Backed up: {backup_count} old tests")
    print(f"   All old tests now use standardized framework!")


def create_test_runner():
    """Create a comprehensive test runner for all standardized tests."""
    runner_content = '''"""
Comprehensive Test Runner for Standardized Tests

This script runs all standardized integration tests with proper
logging, visualization, and testing framework.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from test.framework.standardized_test import TestSuite
from utils.standardized_logging import get_test_logger


def run_all_standardized_tests():
    """Run all standardized integration tests."""
    print("🚀 Running All Standardized Integration Tests")
    print("=" * 60)
    
    # Find all standardized test files
    test_files = []
    
    # Main integration tests
    main_tests = list(Path("test/integration").glob("test_*.py"))
    test_files.extend(main_tests)
    
    # Constraint-specific tests
    constraint_tests = list(Path("test/integration/constraints").rglob("*.py"))
    test_files.extend(constraint_tests)
    
    # Objective-specific tests
    objective_tests = list(Path("test/integration/objective").rglob("*.py"))
    test_files.extend(objective_tests)
    
    # Filter out non-test files
    test_files = [f for f in test_files if f.name.startswith("test_") and f.suffix == ".py"]
    
    print(f"📋 Found {len(test_files)} standardized test files")
    
    # Run each test
    successful_tests = 0
    failed_tests = 0
    
    for i, test_file in enumerate(test_files, 1):
        print(f"\\n{'='*50}")
        print(f"🧪 Running test {i}/{len(test_files)}: {test_file.name}")
        print(f"{'='*50}")
        
        try:
            # Run the test
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(project_root)
            )
            
            if result.returncode == 0:
                successful_tests += 1
                print(f"✅ {test_file.name} - PASSED")
            else:
                failed_tests += 1
                print(f"❌ {test_file.name} - FAILED")
                print(f"   Return code: {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
        
        except subprocess.TimeoutExpired:
            failed_tests += 1
            print(f"⏰ {test_file.name} - TIMEOUT")
        except Exception as e:
            failed_tests += 1
            print(f"💥 {test_file.name} - ERROR: {e}")
    
    # Final summary
    print(f"\\n{'='*60}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {len(test_files)}")
    print(f"Successful: {successful_tests} ✅")
    print(f"Failed: {failed_tests} ❌")
    print(f"Success rate: {(successful_tests/len(test_files))*100:.1f}%")
    
    return successful_tests, failed_tests


if __name__ == "__main__":
    successful, failed = run_all_standardized_tests()
'''
    
    runner_path = Path("test/integration/run_all_standardized_tests.py")
    with open(runner_path, 'w') as f:
        f.write(runner_content)
    
    print(f"✅ Created comprehensive test runner: {runner_path}")


if __name__ == "__main__":
    replace_old_tests()
    create_test_runner()
