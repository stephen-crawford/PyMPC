"""
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
        print(f"\n{'='*50}")
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
    print(f"\n{'='*60}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {len(test_files)}")
    print(f"Successful: {successful_tests} ✅")
    print(f"Failed: {failed_tests} ❌")
    print(f"Success rate: {(successful_tests/len(test_files))*100:.1f}%")
    
    return successful_tests, failed_tests


if __name__ == "__main__":
    successful, failed = run_all_standardized_tests()
