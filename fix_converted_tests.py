"""
Fix Converted Tests

This script fixes the import paths in all converted test files.
"""

import os
import re
from pathlib import Path


def fix_converted_test(test_file_path):
    """Fix import paths in a converted test file."""
    print(f"🔧 Fixing: {test_file_path}")
    
    with open(test_file_path, 'r') as f:
        content = f.read()
    
    # Fix the project root path calculation
    old_path = "project_root = Path(__file__).parent.parent.parent.parent"
    new_path = "project_root = Path(__file__).parent.parent.parent"
    
    if old_path in content:
        content = content.replace(old_path, new_path)
        print(f"   ✅ Fixed project root path")
    
    # Write the fixed content back
    with open(test_file_path, 'w') as f:
        f.write(content)
    
    print(f"   ✅ Fixed: {test_file_path}")


def fix_all_converted_tests():
    """Fix all converted test files."""
    print("🔧 Fixing all converted test files...")
    
    # Find all converted test files
    test_dir = Path("test/integration")
    converted_tests = list(test_dir.glob("converted_*.py"))
    
    print(f"📋 Found {len(converted_tests)} converted test files")
    
    # Fix each test file
    for test_file in converted_tests:
        try:
            fix_converted_test(test_file)
        except Exception as e:
            print(f"❌ Failed to fix {test_file}: {e}")
    
    print(f"\n✅ Fixed {len(converted_tests)} converted test files")


if __name__ == "__main__":
    fix_all_converted_tests()
