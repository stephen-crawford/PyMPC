import casadi as cd
import numpy as np

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planning.src.types import PredictionType, PredictionStep
from solver.src.parameter_manager import ParameterManager
from utils.const import DETERMINISTIC, GAUSSIAN
from utils.math_utils import exponential_quantile, rotation_matrix, chi_square_quantile
from utils.utils import LOG_DEBUG


class EllipsoidConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)
		self.name = 'ellipsoid_constraints'
		self.num_segments = self.get_config_value("contouring.num_segments")

		self.num_discs = self.get_config_value("num_discs")
		self.robot_radius = self.get_config_value("robot.radius")
		self.risk = self.get_config_value("probabilistic.risk")

		self.max_obstacles = self.get_config_value("max_obstacles")
		self.num_constraints = self.max_obstacles * self.num_discs

		# Dummy values for invalid states
		self._dummy_x = 50.0
		self._dummy_y = 50.0

		LOG_DEBUG("Ellipsoid Constraints successfully initialized")

	def update(self, state, data):
		LOG_DEBUG("EllipsoidConstraints.update")

		# Update dummy values based on current state
		self._dummy_x = state.get("x") + 50.0
		self._dummy_y = state.get("y") + 50.0

	def constraint_name(self, index, step, disc_id=None):
		"""Generate constraint parameter names"""
		if disc_id is None:
			return f"ellipsoid_obs_{index}_step_{step}"
		return f"disc_{disc_id}_ellipsoid_constraint_{index}_step_{step}"

	def define_parameters(self, params):
		params.add(f"ego_disc_radius")

		for disc_id in range(self.num_discs):
			params.add(f"ego_disc_{disc_id}_offset")

		for obs_id in range(self.max_obstacles):
			for step in range(self.solver.horizon + 1):
				base_name = self.constraint_name(obs_id, step)
				params.add(f"{base_name}_x")
				params.add(f"{base_name}_y")
				params.add(f"{base_name}_psi")
				params.add(f"{base_name}_major")
				params.add(f"{base_name}_minor")
				params.add(f"{base_name}_chi")
				params.add(f"{base_name}_r")

	def set_parameters(self, parameter_manager, data, step):
		LOG_DEBUG(
			f"set_parameters called with step={step}, num_obstacles={len(data.dynamic_obstacles) if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles else 0}")

		parameter_manager.set_parameter("ego_disc_radius", self.robot_radius)

		for disc_id in range(self.num_discs):
			if disc_id >= len(data.robot_area):
				LOG_DEBUG(f"ERROR: disc_id {disc_id} >= len(robot_area) {len(data.robot_area)}")
				break
			parameter_manager.set_parameter(f"ego_disc_{disc_id}_offset", data.robot_area[disc_id].offset)

		# Set parameters for ALL obstacles and ALL steps to avoid missing parameter errors
		for obs_id in range(self.max_obstacles):
			for step_idx in range(self.solver.horizon + 1):
				base_name = self.constraint_name(obs_id, step_idx)

				# Default dummy values
				default_values = {
					"x": self._dummy_x, "y": self._dummy_y,
					"psi": 0.0, "r": 0.1,
					"major": 0.1, "minor": 0.1, "chi": 1.0
				}

				# Set dummy values first
				for name, value in default_values.items():
					parameter_manager.set_parameter(f"{base_name}_{name}", value)

				# Override with actual data if available
				if (hasattr(data, 'dynamic_obstacles') and
						data.dynamic_obstacles and
						obs_id < len(data.dynamic_obstacles)):

					current_obstacle = data.dynamic_obstacles[obs_id]

					if (current_obstacle.prediction is not None and
							len(current_obstacle.prediction.steps) > 0):

						# Use the appropriate prediction step (or last available if not enough steps)
						prediction_step_idx = min(step_idx, len(current_obstacle.prediction.steps) - 1)

						LOG_DEBUG(
							f"Setting real data for obstacle {obs_id}, step {step_idx}, using prediction step {prediction_step_idx}")

						# Set actual obstacle data
						parameter_manager.set_parameter(f"{base_name}_x",
														current_obstacle.prediction.steps[prediction_step_idx].position[
															0])
						parameter_manager.set_parameter(f"{base_name}_y",
														current_obstacle.prediction.steps[prediction_step_idx].position[
															1])
						parameter_manager.set_parameter(f"{base_name}_psi",
														current_obstacle.prediction.steps[prediction_step_idx].angle)
						parameter_manager.set_parameter(f"{base_name}_r", current_obstacle.radius)

						# Compute uncertainty scaling
						chi = chi_square_quantile(dof=2, alpha=1.0 - self.risk)

						if current_obstacle.prediction.type == PredictionType.DETERMINISTIC:
							major, minor = 0.1, 0.1

						elif current_obstacle.prediction.type == PredictionType.GAUSSIAN:
							major = max(0.1,
										current_obstacle.prediction.steps[prediction_step_idx].major_radius * np.sqrt(
											chi))
							minor = max(0.1,
										current_obstacle.prediction.steps[prediction_step_idx].minor_radius * np.sqrt(
											chi))

						elif current_obstacle.prediction.type == PredictionType.NONGAUSSIAN:
							major = max(0.1,
										current_obstacle.prediction.steps[prediction_step_idx].major_radius * np.sqrt(
											chi) * 1.5)
							minor = max(0.1,
										current_obstacle.prediction.steps[prediction_step_idx].minor_radius * np.sqrt(
											chi) * 1.5)
						else:
							major, minor = 0.1, 0.1

						# Update with actual values
						parameter_manager.set_parameter(f"{base_name}_major", major)
						parameter_manager.set_parameter(f"{base_name}_minor", minor)

	def get_lower_bound(self):
		lower_bound = []
		for obs in range(self.max_obstacles):
			for disc in range(self.num_discs):
				lower_bound.append(1.0)
		return lower_bound

	def get_upper_bound(self):
		upper_bound = []
		for obs in range(self.max_obstacles):
			for disc in range(self.num_discs):
				upper_bound.append(np.inf)
		return upper_bound

	def get_constraints(self, symbolic_state, parameter_manager: ParameterManager, stage_idx):
		constraints = []
		pos_x = symbolic_state.get("x")
		pos_y = symbolic_state.get("y")
		pos = cd.vertcat(pos_x, pos_y)

		try:
			psi = symbolic_state.get("psi")
		except:
			psi = 0.0

		# Build rotation matrix directly with CasADi operations
		cos_psi = cd.cos(psi)
		sin_psi = cd.sin(psi)
		rotation_car = cd.vertcat(
			cd.horzcat(cos_psi, -sin_psi),
			cd.horzcat(sin_psi, cos_psi)
		)

		r_disc = parameter_manager.get("ego_disc_radius")

		# Constraint for dynamic obstacles
		for obs_id in range(self.max_obstacles):
			base_name = self.constraint_name(obs_id, stage_idx)
			obst_x = parameter_manager.get(f"{base_name}_x")
			obst_y = parameter_manager.get(f"{base_name}_y")
			obstacle_cog = cd.vertcat(obst_x, obst_y)

			obst_psi = parameter_manager.get(f"{base_name}_psi")
			obst_major = parameter_manager.get(f"{base_name}_major")
			obst_minor = parameter_manager.get(f"{base_name}_minor")
			obst_r = parameter_manager.get(f"{base_name}_r")

			# Chi scaling is already applied in set_parameters, so don't apply again
			chi = parameter_manager.get(f"{base_name}_chi")

			# Create ellipse matrix using proper CasADi syntax
			major_denom = obst_major + r_disc + obst_r
			minor_denom = obst_minor + r_disc + obst_r

			# Handle potential division by zero
			major_inv_sq = cd.if_else(major_denom > 1e-6, 1.0 / (major_denom * major_denom), 1e6)
			minor_inv_sq = cd.if_else(minor_denom > 1e-6, 1.0 / (minor_denom * minor_denom), 1e6)

			# Create diagonal matrix properly
			ab = cd.diag(cd.vertcat(major_inv_sq, minor_inv_sq))

			# Build obstacle rotation matrix
			cos_obst_psi = cd.cos(obst_psi)
			sin_obst_psi = cd.sin(obst_psi)
			obstacle_rotation = cd.vertcat(
				cd.horzcat(cos_obst_psi, -sin_obst_psi),
				cd.horzcat(sin_obst_psi, cos_obst_psi)
			)

			# Compute ellipse matrix: R^T * A * R
			obstacle_ellipse_matrix = obstacle_rotation.T @ ab @ obstacle_rotation

			for disc_id in range(self.num_discs):
				# Get disc offset
				disc_x = parameter_manager.get(f"ego_disc_{disc_id}_offset")
				disc_relative_pos = cd.vertcat(disc_x, 0)

				# Compute disc position in global frame
				disc_pos = pos + rotation_car @ disc_relative_pos

				# Compute constraint: (p - c)^T * Q * (p - c) >= 1
				disc_to_obstacle = disc_pos - obstacle_cog
				c_disc_obstacle = disc_to_obstacle.T @ obstacle_ellipse_matrix @ disc_to_obstacle
				constraints.append(c_disc_obstacle)

		return constraints

	def is_data_ready(self, data):
		missing_data = ""
		if not data.has("dynamic_obstacles"):
			missing_data += "Obstacles "

		for i in range(len(data.dynamic_obstacles)):
			if data.dynamic_obstacles[i].prediction.empty():
				print("Found missing obstacle pred: " + str(data.dynamic_obstacles[i].prediction.empty()))

			if (not (data.dynamic_obstacles[i].prediction.type == PredictionType.GAUSSIAN) and
					not (data.dynamic_obstacles[i].prediction.type == PredictionType.DETERMINISTIC)):
				missing_data += "Obstacle Prediction (Type is incorrect) "

		LOG_DEBUG("Missing data for ellipsoid constraints has length: " + str(len(missing_data)))
		LOG_DEBUG("Missing data for ellipsoid constraints is: " + str(missing_data))
		return len(missing_data) < 1


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
		for stage_idx in range(solver.horizon + 1):
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