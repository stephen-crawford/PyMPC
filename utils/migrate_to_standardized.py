"""
Migration Script for Standardized Systems

This script helps migrate existing PyMPC tests to use the new standardized
logging, visualization, and testing systems.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple


class TestMigrator:
    """Migrates existing tests to standardized systems."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.migrations = []
        
    def migrate_test_file(self, test_file_path: str) -> bool:
        """Migrate a single test file to use standardized systems."""
        file_path = Path(test_file_path)
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return False
        
        print(f"🔄 Migrating: {file_path}")
        
        # Read original file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Apply migrations
        migrated_content = self.apply_migrations(content, file_path.name)
        
        # Write migrated file
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
        with open(backup_path, 'w') as f:
            f.write(content)
        
        with open(file_path, 'w') as f:
            f.write(migrated_content)
        
        print(f"✅ Migrated: {file_path}")
        print(f"📄 Backup saved: {backup_path}")
        
        return True
    
    def apply_migrations(self, content: str, filename: str) -> str:
        """Apply all migrations to file content."""
        migrations = [
            self.migrate_imports,
            self.migrate_logging_calls,
            self.migrate_visualization_code,
            self.migrate_test_structure,
            self.migrate_error_handling
        ]
        
        for migration in migrations:
            content = migration(content, filename)
        
        return content
    
    def migrate_imports(self, content: str, filename: str) -> str:
        """Migrate import statements."""
        # Add standardized system imports
        if "from utils.standardized_logging" not in content:
            # Find the last import statement
            import_pattern = r'(import\s+\w+.*\n|from\s+\w+.*\n)'
            imports = re.findall(import_pattern, content)
            
            if imports:
                last_import = imports[-1]
                standardized_imports = '''# Standardized system imports
from utils.standardized_logging import get_test_logger, LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR
from utils.standardized_visualization import TestVisualizationManager, VisualizationConfig, VisualizationMode
from test.framework.standardized_test import BaseMPCTest, TestConfig, TestResult
from utils.debugging_tools import ConstraintAnalyzer, SolverDiagnostics, TrajectoryAnalyzer

'''
                content = content.replace(last_import, last_import + standardized_imports)
        
        return content
    
    def migrate_logging_calls(self, content: str, filename: str) -> str:
        """Migrate logging calls to standardized system."""
        # Replace print statements with logger calls
        content = re.sub(r'print\s*\(\s*["\']([^"\']*)["\']\s*\)', r'logger.log_info("\1")', content)
        content = re.sub(r'print\s*\(\s*f["\']([^"\']*)["\']\s*\)', r'logger.log_info(f"\1")', content)
        
        # Replace debug prints
        content = re.sub(r'print\s*\(\s*["\']DEBUG:([^"\']*)["\']\s*\)', r'logger.log_debug("\1")', content)
        
        # Replace error prints
        content = re.sub(r'print\s*\(\s*["\']ERROR:([^"\']*)["\']\s*\)', r'logger.log_error("\1")', content)
        
        return content
    
    def migrate_visualization_code(self, content: str, filename: str) -> str:
        """Migrate visualization code to standardized system."""
        # Replace matplotlib setup
        if "plt.figure" in content:
            content = re.sub(
                r'plt\.figure\([^)]*\)',
                'visualizer.create_figure("single")',
                content
            )
        
        # Replace plot calls
        if "plt.plot" in content:
            content = re.sub(
                r'plt\.plot\s*\(([^)]+)\)',
                r'visualizer.plot_trajectory(\1)',
                content
            )
        
        # Replace show calls
        content = re.sub(r'plt\.show\(\)', 'visualizer.update_test_progress(state, trajectory_x, trajectory_y, iteration)', content)
        
        return content
    
    def migrate_test_structure(self, content: str, filename: str) -> str:
        """Migrate test structure to standardized framework."""
        # This is a complex migration that would require parsing the file structure
        # For now, we'll add a comment suggesting manual migration
        if "class" in content and "Test" in content:
            content = f"""# MIGRATION NOTE: This test should be converted to use BaseMPCTest
# See STANDARDIZED_SYSTEMS_GUIDE.md for detailed migration instructions
# 
# Example migration:
# 1. Inherit from BaseMPCTest instead of current base class
# 2. Implement setup_test_environment() method
# 3. Implement setup_mpc_system() method  
# 4. Implement execute_mpc_iteration() method
# 5. Implement check_goal_reached() method
# 6. Use standardized logging and visualization

{content}"""
        
        return content
    
    def migrate_error_handling(self, content: str, filename: str) -> str:
        """Migrate error handling to standardized system."""
        # Replace generic exception handling
        content = re.sub(
            r'except\s+Exception\s+as\s+e:\s*\n\s*print\s*\([^)]*\)',
            r'except Exception as e:\n        logger.log_error("Exception occurred", e)',
            content
        )
        
        return content
    
    def create_migration_template(self, test_name: str) -> str:
        """Create a template for migrating a test to standardized systems."""
        template = f'''"""
{test_name} - Migrated to Standardized Systems

This test has been migrated to use the standardized logging, visualization,
and testing systems for PyMPC.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from test.framework.standardized_test import BaseMPCTest, TestConfig
from utils.standardized_logging import get_test_logger
from utils.standardized_visualization import VisualizationConfig, VisualizationMode
from utils.debugging_tools import ConstraintAnalyzer, SolverDiagnostics, TrajectoryAnalyzer


class {test_name.replace(' ', '')}(BaseMPCTest):
    """
    {test_name} using standardized systems.
    
    This test demonstrates:
    - Standardized logging with clear diagnostics
    - Real-time visualization
    - Graceful MPC failure handling
    - Comprehensive debugging tools
    """
    
    def __init__(self):
        config = TestConfig(
            test_name="{test_name.lower().replace(' ', '_')}",
            description="{test_name} using standardized systems",
            timeout=60.0,
            max_iterations=100,
            goal_tolerance=1.0,
            enable_visualization=True,
            visualization_mode=VisualizationMode.REALTIME,
            log_level="INFO"
        )
        super().__init__(config)
        
        # Initialize debugging tools
        self.constraint_analyzer = ConstraintAnalyzer()
        self.solver_diagnostics = SolverDiagnostics()
        self.trajectory_analyzer = TrajectoryAnalyzer()
    
    def setup_test_environment(self):
        """Setup the test environment (roads, obstacles, etc.)."""
        # TODO: Implement environment setup
        # Return dictionary with:
        # - start: (x, y) tuple
        # - goal: (x, y) tuple  
        # - reference_path: dict with x, y, s arrays
        # - left_bound, right_bound: dict with x, y, s arrays
        # - dynamic_obstacles: list of obstacle dicts
        pass
    
    def setup_mpc_system(self, data):
        """Setup the MPC system (solver, planner, etc.)."""
        # TODO: Implement MPC system setup
        # Return (planner, solver) tuple
        pass
    
    def execute_mpc_iteration(self, planner, data, iteration):
        """Execute one MPC iteration and return new state."""
        # TODO: Implement MPC iteration
        # Return new state dict
        pass
    
    def check_goal_reached(self, state, goal):
        """Check if the goal has been reached."""
        # TODO: Implement goal checking
        # Return boolean
        pass


# Run the test
if __name__ == "__main__":
    test = {test_name.replace(' ', '')}()
    result = test.run_test()
    
    print(f"Test {{'PASSED' if result.success else 'FAILED'}}")
    print(f"Duration: {{result.duration:.2f}}s")
    print(f"Iterations: {{result.iterations_completed}}")
    print(f"Final distance: {{result.final_distance_to_goal:.3f}}")
'''
        return template
    
    def generate_migration_report(self) -> str:
        """Generate a report of all migrations performed."""
        report = f"""# Migration Report

## Summary
- Total migrations: {len(self.migrations)}
- Files processed: {len(set(m['file'] for m in self.migrations))}

## Migrations Applied
"""
        
        for migration in self.migrations:
            report += f"- {migration['type']}: {migration['description']}\n"
        
        return report


def migrate_existing_tests(project_root: str):
    """Migrate all existing tests to standardized systems."""
    migrator = TestMigrator(project_root)
    
    # Find test files
    test_files = []
    test_dir = Path(project_root) / "test" / "integration"
    
    if test_dir.exists():
        for test_file in test_dir.rglob("*.py"):
            if "test" in test_file.name.lower() and not test_file.name.startswith("example"):
                test_files.append(str(test_file))
    
    print(f"🔍 Found {len(test_files)} test files to migrate")
    
    # Migrate each file
    successful_migrations = 0
    for test_file in test_files:
        try:
            if migrator.migrate_test_file(test_file):
                successful_migrations += 1
        except Exception as e:
            print(f"❌ Failed to migrate {test_file}: {e}")
    
    print(f"\n✅ Successfully migrated {successful_migrations}/{len(test_files)} files")
    
    # Generate migration report
    report = migrator.generate_migration_report()
    with open(Path(project_root) / "MIGRATION_REPORT.md", "w") as f:
        f.write(report)
    
    print(f"📄 Migration report saved to: MIGRATION_REPORT.md")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."
    
    migrate_existing_tests(project_root)
