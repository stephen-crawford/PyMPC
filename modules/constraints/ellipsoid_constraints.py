import casadi as cd
import numpy as np

from modules.constraints.base_constraint import BaseConstraint
from utils.utils import LOG_DEBUG


class EllipsoidConstraints(BaseConstraint):
	def __init__(self):
		super().__init__()
		self.name = "ellipsoid_constraints"
		LOG_DEBUG("EllipsoidConstraints initialized")

	def update(self, state, data):
		return

	def calculate_constraints(self, state, data, stage_idx):
		return []

	def lower_bounds(self):
		return []

	def upper_bounds(self):
		return []


import numpy as np
import casadi as cs
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ConstraintViolation:
	"""Data structure to track constraint violations"""
	constraint_id: str
	stage: int
	value: float
	violation: float
	tolerance: float
	is_violated: bool


class EllipsoidConstraintVerifier:
	"""Verifies ellipsoidal constraint satisfaction with configurable tolerances"""

	def __init__(self, solver_tolerance: float = 1e-4, verification_tolerance: float = 1e-3):
		self.solver_tolerance = solver_tolerance
		self.verification_tolerance = verification_tolerance
		self.violation_history = []
		self.stats = {
			'max_violation': 0.0,
			'mean_violation': 0.0,
			'violation_count': 0,
			'total_checks': 0
		}

	def verify_ellipsoid_constraints_solution(self, solver, parameter_manager) -> Dict:
		"""
		Verify ellipsoid constraints are satisfied in the current solution

		Args:
			solver: CasADiSolver instance with solution
			parameter_manager: Parameter manager with constraint data

		Returns:
			Dict with verification results
		"""
		if solver.solution is None:
			return {'status': 'no_solution', 'violations': []}

		violations = []
		max_violation = 0.0

		# Get ellipsoid constraint module
		ellipsoid_module = None
		for module in solver.module_manager.get_modules():
			if hasattr(module, 'name') and module.name == 'ellipsoid_constraints':
				ellipsoid_module = module
				break

		if ellipsoid_module is None:
			return {'status': 'no_ellipsoid_module', 'violations': []}

		# Verify constraints at each stage
		horizon_val = solver.horizon if (hasattr(solver, 'horizon') and solver.horizon is not None) else 10
		for stage_idx in range(horizon_val + 1):
			stage_violations = self._verify_stage_constraints(
				solver, ellipsoid_module, parameter_manager, stage_idx
			)
			violations.extend(stage_violations)

			# Track maximum violation
			for violation in stage_violations:
				max_violation = max(max_violation, violation.violation)

		# Update statistics
		self._update_statistics(violations)

		# Determine overall status
		critical_violations = [v for v in violations if v.is_violated]
		status = 'satisfied' if len(critical_violations) == 0 else 'violated'

		return {
			'status': status,
			'violations': violations,
			'critical_violations': critical_violations,
			'max_violation': max_violation,
			'num_violations': len(critical_violations),
			'total_constraints': len(violations)
		}

	def _verify_stage_constraints(self, solver, ellipsoid_module, parameter_manager, stage_idx) -> List[
		ConstraintViolation]:
		"""Verify ellipsoid constraints for a specific stage"""
		violations = []

		# Get robot state at this stage
		pos_x = solver.get_output(stage_idx, 'x')
		pos_y = solver.get_output(stage_idx, 'y')
		try:
			psi = solver.get_output(stage_idx, 'psi')
		except:
			psi = 0.0

		if pos_x is None or pos_y is None:
			return violations

		pos = np.array([pos_x, pos_y])

		# Build rotation matrix
		cos_psi = np.cos(psi)
		sin_psi = np.sin(psi)
		rotation_car = np.array([[cos_psi, -sin_psi], [sin_psi, cos_psi]])

		r_disc = parameter_manager.get("ego_disc_radius")

		# Check each obstacle
		for obs_id in range(ellipsoid_module.max_obstacles):
			base_name = ellipsoid_module.constraint_name(obs_id, stage_idx)

			# Get obstacle parameters
			try:
				obst_x = parameter_manager.get(f"{base_name}_x")
				obst_y = parameter_manager.get(f"{base_name}_y")
				obst_psi = parameter_manager.get(f"{base_name}_psi")
				obst_major = parameter_manager.get(f"{base_name}_major")
				obst_minor = parameter_manager.get(f"{base_name}_minor")
				obst_r = parameter_manager.get(f"{base_name}_r")
			except:
				continue  # Skip if parameters not available

			obstacle_cog = np.array([obst_x, obst_y])

			# Check each disc
			for disc_id in range(ellipsoid_module.num_discs):
				disc_x = parameter_manager.get(f"ego_disc_{disc_id}_offset")
				disc_relative_pos = np.array([disc_x, 0])

				# Compute disc position in global frame
				disc_pos = pos + rotation_car @ disc_relative_pos

				# Compute ellipsoid constraint value
				constraint_value = self._compute_ellipsoid_constraint(
					disc_pos, obstacle_cog, obst_psi,
					obst_major, obst_minor, obst_r, r_disc
				)

				# Check violation (constraint should be >= 1.0)
				violation_amount = max(0.0, 1.0 - constraint_value)
				is_violated = violation_amount > self.verification_tolerance

				constraint_id = f"obs_{obs_id}_disc_{disc_id}_stage_{stage_idx}"

				violations.append(ConstraintViolation(
					constraint_id=constraint_id,
					stage=stage_idx,
					value=constraint_value,
					violation=violation_amount,
					tolerance=self.verification_tolerance,
					is_violated=is_violated
				))

		return violations

	def _compute_ellipsoid_constraint(self, disc_pos: np.ndarray, obstacle_cog: np.ndarray,
									  obst_psi: float, obst_major: float, obst_minor: float,
									  obst_r: float, r_disc: float) -> float:
		"""Compute the ellipsoid constraint value: (p-c)^T * Q * (p-c)"""

		# Create ellipse matrix
		major_denom = obst_major + r_disc + obst_r
		minor_denom = obst_minor + r_disc + obst_r

		# Handle potential division by zero
		major_inv_sq = 1.0 / (major_denom * major_denom) if major_denom > 1e-6 else 1e6
		minor_inv_sq = 1.0 / (minor_denom * minor_denom) if minor_denom > 1e-6 else 1e6

		# Diagonal matrix
		ab = np.diag([major_inv_sq, minor_inv_sq])

		# Obstacle rotation matrix
		cos_obst_psi = np.cos(obst_psi)
		sin_obst_psi = np.sin(obst_psi)
		obstacle_rotation = np.array([[cos_obst_psi, -sin_obst_psi],
									  [sin_obst_psi, cos_obst_psi]])

		# Compute ellipse matrix: R^T * A * R
		obstacle_ellipse_matrix = obstacle_rotation.T @ ab @ obstacle_rotation

		# Compute constraint: (p - c)^T * Q * (p - c)
		disc_to_obstacle = disc_pos - obstacle_cog
		constraint_value = disc_to_obstacle.T @ obstacle_ellipse_matrix @ disc_to_obstacle

		return float(constraint_value)

	def _update_statistics(self, violations: List[ConstraintViolation]):
		"""Update violation statistics"""
		self.stats['total_checks'] += len(violations)

		violation_values = [v.violation for v in violations if v.violation > 0]
		if violation_values:
			current_max = max(violation_values)
			self.stats['max_violation'] = max(self.stats['max_violation'], current_max)
			self.stats['violation_count'] += len([v for v in violations if v.is_violated])

			# Update mean (running average)
			total_violations = sum(violation_values)
			if self.stats['total_checks'] > 0:
				self.stats['mean_violation'] = (
													   self.stats['mean_violation'] * (
														   self.stats['total_checks'] - len(violations)) +
													   total_violations
											   ) / self.stats['total_checks']

	def add_to_solver(self, solver):
		"""Add verification methods to the solver instance"""

		def verify_constraints(self_solver):
			"""Method to add to solver for easy constraint verification"""
			return self.verify_ellipsoid_constraints_solution(self_solver, self_solver.parameter_manager)

		def print_constraint_violations(self_solver, max_violations_to_show=10):
			"""Method to print constraint violations in a readable format"""
			result = verify_constraints(self_solver)

			print(f"\n=== Ellipsoid Constraint Verification ===")
			print(f"Status: {result['status']}")
			print(f"Total constraints checked: {result['total_constraints']}")
			print(f"Violations found: {result['num_violations']}")
			print(f"Maximum violation: {result['max_violation']:.6f}")

			if result['critical_violations']:
				print(f"\nTop {min(max_violations_to_show, len(result['critical_violations']))} violations:")
				sorted_violations = sorted(result['critical_violations'],
										   key=lambda x: x.violation, reverse=True)

				for i, violation in enumerate(sorted_violations[:max_violations_to_show]):
					print(f"  {i + 1}. {violation.constraint_id}: "
						  f"value={violation.value:.6f}, "
						  f"violation={violation.violation:.6f}")

			return result

		# Add methods to solver
		solver.verify_ellipsoid_constraints = lambda: verify_constraints(solver)
		solver.print_constraint_violations = lambda max_show=10: print_constraint_violations(solver, max_show)

	def get_statistics(self) -> Dict:
		"""Get violation statistics"""
		return self.stats.copy()

	def reset_statistics(self):
		"""Reset violation statistics"""
		self.stats = {
			'max_violation': 0.0,
			'mean_violation': 0.0,
			'violation_count': 0,
			'total_checks': 0
		}
		self.violation_history.clear()


# Usage example integration with your solver
def integrate_verifier_with_solver(solver_instance, solver_tol=1e-4, verification_tol=1e-3):
	"""
	Integrate the verifier with your existing solver

	Args:
		solver_instance: Your CasADiSolver instance
		solver_tol: Solver tolerance (from ipopt.tol)
		verification_tol: Verification tolerance (should be >= solver_tol)
	"""
	verifier = EllipsoidConstraintVerifier(solver_tol, verification_tol)
	verifier.add_to_solver(solver_instance)
	return verifier


# Example usage in your solve loop:
def enhanced_solve_with_verification(solver):
	"""Enhanced solve method with constraint verification"""

	# Integrate verifier
	verifier = integrate_verifier_with_solver(solver)

	# Solve as usual
	exit_flag = solver.solve()

	if exit_flag == 1:  # Success
		# Verify constraints
		result = solver.verify_ellipsoid_constraints()

		if result['status'] == 'violated':
			print(f"WARNING: Ellipsoid constraints violated! Max violation: {result['max_violation']:.6f}")
			solver.print_constraint_violations(max_show=5)

		# Optionally, you could:
		# 1. Tighten solver tolerance and re-solve
		# 2. Add penalty terms for violated constraints
		# 3. Log violations for analysis

		else:
			print(f"✓ All ellipsoid constraints satisfied (max residual: {result['max_violation']:.6f})")

	return exit_flag, verifier.get_statistics()