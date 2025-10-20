"""
Replace Old Tests with Updated Versions

This script replaces old test files with their updated versions that use
the new logging and visualization framework, then deletes the old files.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple


class TestReplacer:
    """Replaces old tests with updated versions and cleans up old files."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.replaced_tests = []
        self.deleted_files = []
        self.failed_operations = []
        
    def replace_test_files(self):
        """Replace old tests with updated versions."""
        print("🔄 Replacing old tests with updated versions")
        print("=" * 70)
        
        # Define replacement mappings
        replacements = self._get_replacement_mappings()
        
        for old_file, new_file in replacements.items():
            if self._replace_single_test(old_file, new_file):
                self.replaced_tests.append((old_file, new_file))
        
        # Clean up old files
        self._cleanup_old_files()
        
        # Generate summary
        self._generate_summary()
    
    def _get_replacement_mappings(self) -> Dict[Path, Path]:
        """Get mappings of old files to their updated versions."""
        replacements = {}
        
        # Main integration tests
        main_tests = [
            ("test_final_mpc_implementation.py", "converted_test_final_mpc_implementation.py"),
            ("test_guaranteed_goal_reaching.py", "converted_test_guaranteed_goal_reaching.py"),
            ("test_fixed_solver.py", "converted_test_fixed_solver.py"),
            ("test_complete_mpc_system.py", "converted_test_complete_mpc_system.py"),
            ("test_standardized_systems.py", "converted_test_standardized_systems.py"),
            ("test_final_scenario_contouring.py", "converted_test_final_scenario_contouring.py"),
            ("test_scenario_contouring_confirmation.py", "converted_test_scenario_contouring_confirmation.py"),
            ("test_working_mpc_goal_reaching.py", "converted_test_working_mpc_goal_reaching.py"),
            ("test_working_scenario_mpc.py", "converted_test_working_scenario_mpc.py"),
            ("test_all_constraint_types.py", "converted_test_all_constraint_types.py"),
            ("test_working_scenario_contouring.py", "converted_test_working_scenario_contouring.py"),
            ("test_simple_goal_reaching.py", "converted_test_simple_goal_reaching.py"),
            ("test_proper_scenario_constraints.py", "converted_test_proper_scenario_constraints.py"),
            ("test_scenario_contouring_integration.py", "converted_test_scenario_contouring_integration.py")
        ]
        
        for old_name, new_name in main_tests:
            old_path = self.project_root / "test" / "integration" / old_name
            new_path = self.project_root / "test" / "integration" / new_name
            if old_path.exists() and new_path.exists():
                replacements[old_path] = new_path
        
        # Constraint-specific tests
        constraint_tests = [
            ("scenario", "scenario_and_contouring_constraints_with_contouring_objective.py", 
             "converted_scenario_and_contouring_constraints_with_contouring_objective.py"),
            ("gaussian", "gaussian_and_contouring_constraints_with_contouring_objective.py",
             "converted_gaussian_and_contouring_constraints_with_contouring_objective.py"),
            ("decomp", "decomp_and_contouring_constraints_with_contouring_objective.py",
             "converted_decomp_and_contouring_constraints_with_contouring_objective.py"),
            ("linear", "linear_constraints_contouring_objective.py",
             "converted_linear_constraints_contouring_objective.py"),
            ("linear", "linear_and_contouring_constraints_with_contouring_objective.py",
             "converted_linear_and_contouring_constraints_with_contouring_objective.py"),
            ("linear", "linear_and_contouring_constraints_contouring_objective.py",
             "converted_linear_and_contouring_constraints_contouring_objective.py"),
            ("ellipsoid", "ellipsoid_and_contouring_constraints_with_contouring_objective.py",
             "converted_ellipsoid_and_contouring_constraints_with_contouring_objective.py")
        ]
        
        for constraint_type, old_name, new_name in constraint_tests:
            old_path = self.project_root / "test" / "integration" / "constraints" / constraint_type / old_name
            new_path = self.project_root / "test" / "integration" / "constraints" / constraint_type / new_name
            if old_path.exists() and new_path.exists():
                replacements[old_path] = new_path
        
        # Objective tests
        objective_tests = [
            ("goal", "goal_objective_integration_test.py", "converted_goal_objective_integration_test.py"),
            ("contouring", "goal_contouring_integration_test.py", "converted_goal_contouring_integration_test.py")
        ]
        
        for objective_type, old_name, new_name in objective_tests:
            old_path = self.project_root / "test" / "integration" / "objective" / objective_type / old_name
            new_path = self.project_root / "test" / "integration" / "objective" / objective_type / new_name
            if old_path.exists() and new_path.exists():
                replacements[old_path] = new_path
        
        return replacements
    
    def _replace_single_test(self, old_file: Path, new_file: Path) -> bool:
        """Replace a single test file."""
        print(f"🔄 Replacing: {old_file.name}")
        
        try:
            # Create backup of old file
            backup_file = old_file.with_suffix('.py.backup')
            shutil.copy2(old_file, backup_file)
            
            # Replace old file with new file
            shutil.copy2(new_file, old_file)
            
            print(f"   ✅ Replaced: {old_file.name}")
            return True
            
        except Exception as e:
            print(f"   ❌ Failed: {old_file.name} - {e}")
            self.failed_operations.append((old_file, str(e)))
            return False
    
    def _cleanup_old_files(self):
        """Clean up old files and backups."""
        print(f"\n🧹 Cleaning up old files")
        print("-" * 50)
        
        # Delete converted files (now replaced the originals)
        converted_files = list(self.project_root.glob("test/integration/converted_*.py"))
        converted_files.extend(list(self.project_root.glob("test/integration/constraints/**/converted_*.py")))
        converted_files.extend(list(self.project_root.glob("test/integration/objective/**/converted_*.py")))
        
        for file_path in converted_files:
            try:
                file_path.unlink()
                self.deleted_files.append(file_path)
                print(f"   🗑️  Deleted: {file_path.name}")
            except Exception as e:
                print(f"   ❌ Failed to delete: {file_path.name} - {e}")
                self.failed_operations.append((file_path, str(e)))
        
        # Delete backup files
        backup_files = list(self.project_root.glob("test/**/*.py.backup"))
        for file_path in backup_files:
            try:
                file_path.unlink()
                self.deleted_files.append(file_path)
                print(f"   🗑️  Deleted backup: {file_path.name}")
            except Exception as e:
                print(f"   ❌ Failed to delete backup: {file_path.name} - {e}")
                self.failed_operations.append((file_path, str(e)))
    
    def _generate_summary(self):
        """Generate replacement summary."""
        print(f"\n{'='*70}")
        print(f"📊 TEST REPLACEMENT SUMMARY")
        print(f"{'='*70}")
        print(f"Successfully replaced: {len(self.replaced_tests)} ✅")
        print(f"Files deleted: {len(self.deleted_files)} 🗑️")
        print(f"Failed operations: {len(self.failed_operations)} ❌")
        
        if self.replaced_tests:
            print(f"\n✅ REPLACED TESTS:")
            for old_file, new_file in self.replaced_tests:
                print(f"   - {old_file.name} ← {new_file.name}")
        
        if self.deleted_files:
            print(f"\n🗑️ DELETED FILES:")
            for file_path in self.deleted_files:
                print(f"   - {file_path.name}")
        
        if self.failed_operations:
            print(f"\n❌ FAILED OPERATIONS:")
            for file_path, error in self.failed_operations:
                print(f"   - {file_path.name}: {error}")
        
        # Save detailed report
        report_file = self.project_root / "TEST_REPLACEMENT_REPORT.md"
        with open(report_file, 'w') as f:
            f.write("# Test Replacement Report\n\n")
            f.write(f"**Successfully replaced:** {len(self.replaced_tests)}\n")
            f.write(f"**Files deleted:** {len(self.deleted_files)}\n")
            f.write(f"**Failed operations:** {len(self.failed_operations)}\n\n")
            
            if self.replaced_tests:
                f.write("## Replaced Tests\n\n")
                for old_file, new_file in self.replaced_tests:
                    f.write(f"- `{old_file.name}` ← `{new_file.name}`\n")
                f.write("\n")
            
            if self.deleted_files:
                f.write("## Deleted Files\n\n")
                for file_path in self.deleted_files:
                    f.write(f"- `{file_path.name}`\n")
                f.write("\n")
            
            if self.failed_operations:
                f.write("## Failed Operations\n\n")
                for file_path, error in self.failed_operations:
                    f.write(f"- `{file_path.name}`: {error}\n")
        
        print(f"\n📄 Detailed report saved to: {report_file}")


def main():
    """Main replacement function."""
    replacer = TestReplacer()
    replacer.replace_test_files()


if __name__ == "__main__":
    main()
