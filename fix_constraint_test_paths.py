"""
Fix Constraint Test Paths

This script fixes the import path issues in constraint-specific tests.
"""

import os
import re
from pathlib import Path


def fix_constraint_test_paths():
    """Fix import paths in constraint-specific tests."""
    print("🔧 Fixing constraint-specific test paths...")
    
    # Find all constraint-specific converted tests
    constraint_dirs = [
        "test/integration/constraints/scenario",
        "test/integration/constraints/gaussian", 
        "test/integration/constraints/linear",
        "test/integration/constraints/ellipsoid",
        "test/integration/constraints/decomp",
        "test/integration/objective/goal",
        "test/integration/objective/contouring"
    ]
    
    fixed_count = 0
    
    for constraint_dir in constraint_dirs:
        if Path(constraint_dir).exists():
            converted_tests = list(Path(constraint_dir).glob("converted_*.py"))
            
            for test_file in converted_tests:
                print(f"🔧 Fixing: {test_file}")
                
                with open(test_file, 'r') as f:
                    content = f.read()
                
                # Fix the project root path calculation for constraint tests
                # They need to go up 4 levels to reach project root
                old_path = "project_root = Path(__file__).parent.parent.parent"
                new_path = "project_root = Path(__file__).parent.parent.parent.parent"
                
                if old_path in content:
                    content = content.replace(old_path, new_path)
                    print(f"   ✅ Fixed project root path")
                
                # Write the fixed content back
                with open(test_file, 'w') as f:
                    f.write(content)
                
                fixed_count += 1
                print(f"   ✅ Fixed: {test_file}")
    
    print(f"\n✅ Fixed {fixed_count} constraint-specific test files")


if __name__ == "__main__":
    fix_constraint_test_paths()
