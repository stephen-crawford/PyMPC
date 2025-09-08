import math

import numpy as np
from traitlets import List
from typing import Tuple, Optional, Callable

from planning.src.types import ScenarioConstraint, Scenario, SupportSubsample, ObstacleType, SupportData, \
	ConstraintSide, ScenarioStatus
from utils.const import BISECTION_TOLERANCE
from utils.utils import LOG_DEBUG, get_config_dotted, read_config_file


class PolygonSearch:
	"""Class for computing minimal polygon constraints"""

	def __init__(self, constraints, y_left, y_right, constraint_size, poly_range):
		self.constraints = constraints
		self.y_left = y_left
		self.y_right = y_right
		self.constraint_size = constraint_size
		self.poly_range = poly_range
		self.polygon_out = []  # Added to store output polygons

	def compute_minimal_polygon(self, points: np.ndarray):
		"""
		Compute minimal polygon from set of constraint points
		Returns list of (a1, a2, b) constraint coefficients where a1*x + a2*y <= b
		"""
		if len(points) < 3:
			return []

		# Simplified convex hull computation for demonstration
		# In practice, you'd use a proper convex hull algorithm
		hull_points = self._compute_convex_hull(points)
		constraints = []

		for i in range(len(hull_points)):
			p1 = hull_points[i]
			p2 = hull_points[(i + 1) % len(hull_points)]

			# Compute line coefficients ax + by = c
			dx = p2[0] - p1[0]
			dy = p2[1] - p1[1]

			# Normal vector (perpendicular)
			a1 = -dy
			a2 = dx

			# Normalize
			norm = np.sqrt(a1 * a1 + a2 * a2)
			if norm > 1e-8:
				a1 /= norm
				a2 /= norm

			b = a1 * p1[0] + a2 * p1[1]
			constraints.append((a1, a2, b))

		return constraints

	def _compute_convex_hull(self, points: np.ndarray):
		"""Simplified convex hull - replace with proper implementation"""
		# For now, just return the points (placeholder)
		return [(p[0], p[1]) for p in points]

	def size(self):
		"""Return size of polygon output"""
		return len(self.polygon_out)


class Constraint:
	def __init__(self, scenario: Scenario, step: int):
		self.scenario = scenario
		self.step = step


class SafetyCertifier:
	"""Singleton class for safety certification using scenario approach"""

	_instance = None

	def __new__(cls):
		if cls._instance is None:
			cls._instance = super(SafetyCertifier, cls).__new__(cls)
			cls._instance._initialized = False
		return cls._instance

	@classmethod
	def get(cls):
		"""Get singleton instance"""
		return cls()

	@classmethod
	def Get(cls):
		"""Alternative method name for compatibility"""
		return cls()

	def __init__(self):
		if self._initialized:
			return

		# Main parameters
		self.max_support_ = 0
		self.sample_size_ = 0
		self.risk_ = 0.0
		self.confidence_ = 0.0
		self.initialized_ = False

		# Look up table
		self.risk_lut_: List[float] = []

		# Support data
		self.support_data_: Optional[SupportData] = None

		self._initialized = True

	def init(self):
		"""Initialize the safety certifier"""
		self.initialized_ = True

	@staticmethod
	def certify_support_problem(epsilon: float, n_collected: int) -> float:
		"""Certify support problem"""
		return math.pow(1.0 - epsilon, float(n_collected))

	def find_sample_size(self, risk: float, confidence: float, max_support: int) -> int:
		"""
		Find the sample size that is safe for the given configuration

		Args:
			risk: Probability of constraint violation (0 - 1)
			confidence: Confidence with which risk < specified risk (0 - 1)
			max_support: The maximum support to be encountered

		Returns:
			The sample size
		"""
		self.risk_ = risk
		self.confidence_ = confidence
		self.max_support_ = max_support

		# Use bisection to find the minimum sample size
		def risk_function(sample_size: int) -> float:
			return self.epsilon_for_max_support(sample_size) - risk

		# Start with reasonable bounds
		low = 1.0
		high = 10000.0

		self.sample_size_ = self.bisection(low, high, risk_function)
		self.compute_risk_for_support_range()

		return self.sample_size_

	def get_sample_size(self) -> int:
		"""Get the current sample size"""
		return self.sample_size_

	def GetSampleSize(self) -> int:
		"""Alternative method name for compatibility"""
		return self.sample_size_

	def get_max_support(self) -> int:
		"""Get the maximum support"""
		return self.max_support_

	def get_safe_support_bound(self) -> int:
		"""Returns the maximum safe support"""
		for i, risk in enumerate(self.risk_lut_):
			if risk > self.risk_:
				return i - 1 if i > 0 else 0
		return len(self.risk_lut_) - 1

	def get_risk_for_support(self, support) -> float:
		"""
		Returns the risk for a particular support

		Args:
			support: Either an integer support value or SupportSubsample object

		Returns:
			The risk at the given support
		"""
		if isinstance(support, SupportSubsample):
			s = support.get_support_size()
		else:
			s = int(support)

		if 0 <= s < len(self.risk_lut_):
			return self.risk_lut_[s]
		else:
			return self.epsilon(self.sample_size_, s)

	def log_support(self, support_size: int):
		"""Log support size"""
		if self.support_data_ is not None:
			self.support_data_.add(support_size)

	def rooted_n_choose_k(self, n: float, k: float, root: float) -> float:
		"""
		Compute N choose k inside of a root numerically

		Args:
			n: N value
			k: k value
			root: root value

		Returns:
			Result of computation
		"""
		if k > n or k < 0:
			return 0.0

		if k == 0 or k == n:
			return 1.0

		# Use logarithms to avoid overflow
		log_result = 0.0
		for i in range(int(k)):
			log_result += math.log(n - i) - math.log(i + 1)

		return math.exp(log_result / root)

	def bisection(self, low: float, high: float, func: Callable[[int], float]) -> int:
		"""
		Apply bisection on the given function

		Args:
			low: minimum output value
			high: maximum output value
			func: function to bisect on

		Returns:
			Result of bisection
		"""
		while high - low > BISECTION_TOLERANCE:
			mid = (low + high) / 2.0
			mid_int = int(mid)

			if func(mid_int) < 0:
				high = mid
			else:
				low = mid

		return int(math.ceil(low))

	def compute_risk_for_support_range(self):
		"""Compute the risk for all support levels < max_support_"""
		self.risk_lut_ = []
		for s in range(self.max_support_ + 1):
			risk = self.epsilon(self.sample_size_, s)
			self.risk_lut_.append(risk)

	def epsilon(self, sample_size: int, support: int) -> float:
		"""
		The risk function

		Args:
			sample_size: S
			support: n

		Returns:
			Resulting risk
		"""
		if support > sample_size:
			return 1.0

		# Compute sum from i=0 to support of (sample_size choose i) * epsilon^i * (1-epsilon)^(sample_size-i)
		# This is a simplified version - you may need to implement the full scenario approach formula
		total = 0.0
		for i in range(support + 1):
			if sample_size >= i:
				# Use beta function approximation or exact computation
				binom_coeff = math.comb(sample_size, i) if hasattr(math, 'comb') else self._compute_binomial(
					sample_size, i)
				prob = binom_coeff * math.pow(0.5, sample_size)  # Simplified - replace with actual epsilon calculation
				total += prob

		return min(1.0, total)

	def epsilon_for_max_support(self, sample_size: int) -> float:
		"""
		The risk function, but with support = max_support

		Args:
			sample_size: S

		Returns:
			Resulting risk
		"""
		return self.epsilon(sample_size, self.max_support_)

	def _compute_binomial(self, n: int, k: int) -> int:
		"""Compute binomial coefficient n choose k"""
		if k > n or k < 0:
			return 0
		if k == 0 or k == n:
			return 1

		k = min(k, n - k)  # Take advantage of symmetry
		result = 1
		for i in range(k):
			result = result * (n - i) // (i + 1)
		return result


class SafeHorizon:
	"""
	Python implementation of Safe Horizon MPC for disc-wise computations
	Converted from C++ trajectory_disc.h
	"""

	def __init__(self, disc_id: int, solver, sampler):
		# Set disc ID and solver first
		self.disc_id_ = disc_id
		self.solver = solver
		self.sampler = sampler

		LOG_DEBUG("SH-MPC Disc")

		# Load configuration - matching C++ structure
		self.config = read_config_file()


		self.enable_visualization_ = self.get_config_value("scenario_constraints.draw_disc", -1) == disc_id

		# Main parameters from SafetyCertifier and config
		self.S = SafetyCertifier.get().get_sample_size() if SafetyCertifier.get().sample_size_ > 0 else 50  # sampling count
		self.N = self.get_config_value("horizon", 10)  # Prediction horizon
		self.horizon = self.N

		# State variables
		self.is_feasible_ = True
		self.scenarios_ = None  # Pointer to scenarios
		self.status = ScenarioStatus.RESET if 'ScenarioStatus' in globals() else 0

		# Get robot radius from config
		self.robot_radius_ = self.get_config_value("robot.radius", 1.0)

		# Calculate constraint size: Dynamic + Range + Static
		max_obstacles = self.get_config_value("max_obstacles", 5)
		num_halfspaces = self.get_config_value("linearized_constraints.add_halfspaces", 0)
		constraint_size = self.S * max_obstacles + 4 + num_halfspaces

		# Resize over the horizon - matching C++ vector initialization
		self.verify_poses_ = [np.zeros(2) for _ in range(self.N)]
		self.old_intersects_ = [[] for _ in range(self.N)]

		# Keeping track of infeasible scenarios per k
		self.infeasible_scenario_poses_ = [[] for _ in range(self.N)]
		self.infeasible_scenario_idxs_ = [[] for _ in range(self.N)]

		# Reserve space for infeasible scenarios
		for k in range(self.N):
			self.infeasible_scenario_poses_[k] = []
			self.infeasible_scenario_idxs_[k] = []

		# Intermediate distance computation arrays (matching C++ structure)
		self.diffs_x_ = [np.zeros(self.S) for _ in range(self.N)]
		self.diffs_y_ = [np.zeros(self.S) for _ in range(self.N)]
		self.distances_ = [np.zeros(self.S) for _ in range(self.N)]

		# Constraint vectors Ax <= b
		self.a1_ = [np.zeros(constraint_size) for _ in range(self.N)]
		self.a2_ = [np.zeros(constraint_size) for _ in range(self.N)]
		self.b_ = [np.zeros(constraint_size) for _ in range(self.N)]

		# Property access compatibility for scenario module
		self.a1 = self.a1_
		self.a2 = self.a2_
		self.b = self.b_

		# The x, y value of all constraints at left and right points from vehicle
		self.y_left_ = [np.zeros(constraint_size) for _ in range(self.N)]
		self.y_right_ = [np.zeros(constraint_size) for _ in range(self.N)]
		self.x_left_ = [0.0 for _ in range(self.N)]
		self.x_right_ = [0.0 for _ in range(self.N)]

		# A vector with the sample and obstacle indices per scenario
		self.scenario_indices_ = [[] for _ in range(self.N)]

		# Meta-data of constructed constraints
		self.constraints_ = [[] for _ in range(self.N)]

		# Initialize scenario indices and constraints for each time step
		for k in range(self.N):
			self.scenario_indices_[k] = []
			self.constraints_[k] = []

			# Dynamic constraints
			for v in range(max_obstacles):
				for s in range(self.S):
					scenario = Scenario(s, v)
					self.scenario_indices_[k].append(scenario)

					constraint = ScenarioConstraint(
						scenario=scenario,
						type_=ObstacleType.DYNAMIC,
						side_=ConstraintSide.UNDEFINED
					)
					self.constraints_[k].append(constraint)

			# External static constraints
			for i in range(num_halfspaces):
				idx = max_obstacles * self.S + i
				scenario = Scenario(idx, -1)
				self.scenario_indices_[k].append(scenario)

				constraint = ScenarioConstraint(
					scenario=scenario,
					type_=ObstacleType.STATIC,
					side_=ConstraintSide.UNDEFINED
				)
				self.constraints_[k].append(constraint)

			# Range constraints (4 constraints)
			for i in range(4):
				idx = max_obstacles * self.S + num_halfspaces + i
				scenario = Scenario(idx, -1)
				self.scenario_indices_[k].append(scenario)

				constraint = ScenarioConstraint(
					scenario = scenario,
					type_=ObstacleType.RANGE,
					side_=ConstraintSide.UNDEFINED
				)
				self.constraints_[k].append(constraint)

		# Initialize polygon constructors
		polygon_range = self.get_config_value("scenario_constraints.polygon_range_", 10.0)
		self.polytopes_ = []
		self.polytopes = []  # Alternative access for compatibility

		for k in range(self.N):
			polytope = PolygonSearch(
				self.constraints_[k],
				self.y_left_[k],
				self.y_right_[k],
				constraint_size,
				polygon_range
			)
			self.polytopes_.append(polytope)
			self.polytopes.append(polytope)  # Duplicate reference for compatibility

		# Joint radii w.r.t. each obstacle (including vehicle radius)
		self.radii_ = [0.0 for _ in range(max_obstacles)]
		self.radii = self.radii_  # Alternative access

		# Tracking of infeasible scenarios
		self.distances_feasible_ = [True for _ in range(self.N)]
		self.distances_feasible = self.distances_feasible_  # Alternative access

		# Support subsample initialization
		self.support_subsample_ = SupportSubsample() if 'SupportSubsample' in globals() else None

		# Property compatibility
		self.verify_poses = self.verify_poses_
		self.old_intersects = self.old_intersects_
		self.infeasible_scenario_poses = self.infeasible_scenario_poses_
		self.infeasible_scenario_idxs = self.infeasible_scenario_idxs_
		self.diffs_x = self.diffs_x_
		self.diffs_y = self.diffs_y_
		self.distances = self.distances_
		self.y_left = self.y_left_
		self.y_right = self.y_right_
		self.scenario_indices = self.scenario_indices_
		self.constraints = self.constraints_
		self.sample_size = self.S

		LOG_DEBUG("SafeHorizon initialized")

	def update(self, data):
		"""
		Update the Safe Horizon with new real-time data

		Args:
			data: Real-time data containing obstacles and vehicle state
		"""
		LOG_DEBUG(f"SafeHorizon disc {self.disc_id_} update started")

		# Clear previous computations
		self.clear_all()

		# Load external data and prepare scenarios
		self.load_data(data)

		# Process each time step and obstacle
		if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles is not None:
			for step in range(self.horizon):
				for obstacle_id in range(len(data.dynamic_obstacles)):
					# Compute distances to all scenarios
					self.compute_distances(data, step, obstacle_id)

					# Check feasibility based on distances
					self.check_feasibility_by_distance(step, obstacle_id)

					# Compute halfspace constraints
					self.compute_halfspaces(step, obstacle_id)

				# Construct polytopes for this time step
				self.construct_polytopes(step, data)

		# Set status to success if we reach here
		self.status = 1  # SUCCESS equivalent

		LOG_DEBUG(f"SafeHorizon disc {self.disc_id_} update completed")

	def set_parameters(self, data, step: int):
		"""
		Set constraint parameters for optimization at time step k
		Note: The update computes constraints for k = 0, ..., N-1
		However, k=0 of optimization is initial state, so we use k-1 when prompted with k

		Args:
			data: Real-time data
			step: Time step index
		"""

		if step == 0:
			return

		step_idx = min(step - 1, len(self.polytopes) - 1)
		if step_idx < 0 or step_idx >= len(self.polytopes):
			return

		polytope = self.polytopes[step_idx]

		# Set parameters based on available constraints
		max_constraints = min(24, len(polytope.polygon_out)) if hasattr(polytope, 'polygon_out') else 0

		for edge in range(max_constraints):
			if edge < len(self.constraints_[step_idx]):
				constraint = self.constraints_[step_idx][edge]
				current_index = edge  # Simplified index calculation

				if hasattr(self.solver, 'set_parameter'):
					self.solver.set_parameter(step, f"disc_{step - 1}_scenario_constraint_{edge}_a1",
											  self.a1_[step_idx][current_index] if current_index < len(
												  self.a1_[step_idx]) else 1.0)
					self.solver.set_parameter(step, f"disc_{step - 1}_scenario_constraint_{edge}_a2",
											  self.a2_[step_idx][current_index] if current_index < len(
												  self.a2_[step_idx]) else 0.0)
					self.solver.set_parameter(step, f"disc_{step - 1}_scenario_constraint_{edge}_b",
											  self.b_[step_idx][current_index] if current_index < len(
												  self.b_[step_idx]) else 100.0)

		# Fill remaining constraints with default values
		ego_x = 0.0
		if hasattr(self.solver, 'get_ego_prediction'):
			try:
				ego_x = self.solver.get_ego_prediction("x")
			except:
				ego_x = 0.0

		for i in range(max_constraints, 24):
			if hasattr(self.solver, 'set_parameter'):
				self.solver.set_parameter(step, f"disc_{step}_scenario_constraint_{i}_a1", 1)
				self.solver.set_parameter(step, f"disc_{step}_scenario_constraint_{i}_a2", 0)
				self.solver.set_parameter(step, f"disc_{step}_scenario_constraint_{i}_b", ego_x + 100)

	def compute_active_constraints(self, active_constraints_aggregate: SupportSubsample,
								   infeasible_scenarios: SupportSubsample) -> bool:
		"""
		Compute active constraints and identify infeasible scenarios

		Args:
			active_constraints_aggregate: Output container for active constraints
			infeasible_scenarios: Output container for infeasible scenarios

		Returns:
			bool: True if computation successful
		"""
		feasible = True
		infeasible_count = 0

		for step in range(min(self.horizon - 1, len(self.polytopes))):
			# Get slack value from solver
			slack = 0.0
			if hasattr(self.solver, 'get_output'):
				try:
					slack = self.solver.get_output(step + 1, "slack")
				except:
					slack = 0.0

			# Get vehicle position
			vehicle_x = vehicle_y = 0.0
			if hasattr(self.solver, 'get_output'):
				try:
					vehicle_x = self.solver.get_output(step + 1, "x")
					vehicle_y = self.solver.get_output(step + 1, "y")
				except:
					pass

			# Check constraints in polytope
			polytope = self.polytopes[step]
			if hasattr(polytope, 'polygon_out'):
				constraints_to_check = polytope.polygon_out
			else:
				constraints_to_check = self.constraints_[step]

			for i, constraint in enumerate(constraints_to_check[:min(len(constraints_to_check), len(self.a1_[step]))]):
				# Calculate constraint violation
				constraint_value = (self.a1_[step][i] * vehicle_x +
									self.a2_[step][i] * vehicle_y -
									(self.b_[step][i] + slack))

				if constraint_value > 1e-3:
					infeasible_count += 1
					if hasattr(constraint, 'type') and constraint.type == ObstacleType.DYNAMIC:
						if hasattr(infeasible_scenarios, 'add') and hasattr(constraint, 'scenario'):
							infeasible_scenarios.add(constraint.scenario)
						feasible = False
						if hasattr(active_constraints_aggregate, 'add') and hasattr(constraint, 'scenario'):
							active_constraints_aggregate.add(constraint.scenario)
				elif (hasattr(constraint, 'type') and constraint.type == ObstacleType.DYNAMIC and
					  constraint_value > -1e-7):
					if hasattr(active_constraints_aggregate, 'add') and hasattr(constraint, 'scenario'):
						active_constraints_aggregate.add(constraint.scenario)

		return feasible

	def visualize(self, data):
		"""Visualize constraints and scenarios if enabled"""
		if not self.enable_visualization_:
			return

		try:
			self.visualize_all_constraints()
			self.visualize_scenarios(data)
			self.visualize_selected_scenarios(data)
			self.visualize_polygons()

		except Exception as e:
			LOG_DEBUG(f"Visualization error: {str(e)}")

	def clear_all(self):
		"""Clear data from previous computations"""
		for k in range(min(self.horizon, len(self.diffs_x_))):
			if k < len(self.diffs_x_):
				self.diffs_x_[k] = np.zeros_like(self.diffs_x_[k])
			if k < len(self.diffs_y_):
				self.diffs_y_[k] = np.zeros_like(self.diffs_y_[k])
			if k < len(self.distances_):
				self.distances_[k] = np.zeros_like(self.distances_[k])
			if k < len(self.a1_):
				self.a1_[k] = np.zeros_like(self.a1_[k])
			if k < len(self.a2_):
				self.a2_[k] = np.zeros_like(self.a2_[k])
			if k < len(self.b_):
				self.b_[k] = np.zeros_like(self.b_[k])
			if k < len(self.y_left_):
				self.y_left_[k] = np.zeros_like(self.y_left_[k])
			if k < len(self.y_right_):
				self.y_right_[k] = np.zeros_like(self.y_right_[k])
			if k < len(self.old_intersects_):
				self.old_intersects_[k].clear()
			if k < len(self.infeasible_scenario_poses_):
				self.infeasible_scenario_poses_[k].clear()
			if k < len(self.infeasible_scenario_idxs_):
				self.infeasible_scenario_idxs_[k].clear()

		self.distances_feasible_ = [True] * len(self.distances_feasible_)

	def load_data(self, data):
		"""
		Load external data, retrieve scenarios and prepare for update

		Args:
			data: Real-time data containing obstacles and vehicle state
		"""
		# Get scenarios from sampler
		if self.sampler is not None:
			self.scenarios_ = getattr(self.sampler, 'samples', None)

		# Initialize radii for obstacles
		self.radii_.clear()
		if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles is not None:
			for obstacle in data.dynamic_obstacles:
				# Get obstacle radius (assume it's provided or use default)
				obs_radius = getattr(obstacle, 'radius', 0.5)
				joint_radius = self.robot_radius_ + obs_radius
				self.radii_.append(joint_radius)

		# Initialize feasibility tracking
		self.distances_feasible_ = [True] * len(self.radii_)

		# Call helper methods
		self.push_algorithm(data)
		self.dr_projection(data)

		# Precompute distances and feasibility
		max_obstacles = self.get_config_value("scenario_constraints.max_obstacles_", 5)
		for step in range(min(self.horizon, self.N)):
			for obs_id in range(min(max_obstacles, len(self.radii_))):
				self.compute_distances(data, step, obs_id)
				self.check_feasibility_by_distance(step, obs_id)

	def compute_distances(self, data, k: int, obstacle_id: int):
		"""
		Compute distances to all scenarios for given time step and obstacle

		Args:
			data: Real-time data
			k: Time step index
			obstacle_id: Obstacle index
		"""
		if self.scenarios_ is None:
			return

		# Access scenarios with bounds checking
		if obstacle_id >= len(self.scenarios_) or k >= len(self.scenarios_[obstacle_id]):
			return

		scenarios_at_k = self.scenarios_[obstacle_id][k]
		if len(scenarios_at_k) == 0:
			return

		# Get vehicle position prediction at time k
		if hasattr(data, 'vehicle_prediction') and len(data.vehicle_prediction) > k:
			vehicle_x = data.vehicle_prediction[k].x
			vehicle_y = data.vehicle_prediction[k].y
		else:
			# Use current position as fallback
			vehicle_x = getattr(data, 'vehicle_x', 0.0)
			vehicle_y = getattr(data, 'vehicle_y', 0.0)

		# Compute distances and differences
		num_scenarios = min(len(scenarios_at_k), self.S)
		distances = np.zeros(num_scenarios)
		diffs_x = np.zeros(num_scenarios)
		diffs_y = np.zeros(num_scenarios)

		for i in range(num_scenarios):
			scenario = scenarios_at_k[i]
			# Handle different scenario data formats
			if hasattr(scenario, 'position'):
				pos_x, pos_y = scenario.position[0], scenario.position[1]
			elif isinstance(scenario, (list, tuple)) and len(scenario) >= 2:
				pos_x, pos_y = scenario[0], scenario[1]
			else:
				pos_x, pos_y = 0.0, 0.0

			# Difference vector from scenario to vehicle
			diffs_x[i] = vehicle_x - pos_x
			diffs_y[i] = vehicle_y - pos_y

			# Euclidean distance
			distances[i] = np.sqrt(diffs_x[i] ** 2 + diffs_y[i] ** 2)

		# Store results with bounds checking
		if k < len(self.distances_):
			# Ensure arrays are right size
			if len(self.distances_[k]) != num_scenarios:
				self.distances_[k] = np.zeros(max(num_scenarios, self.S))
				self.diffs_x_[k] = np.zeros(max(num_scenarios, self.S))
				self.diffs_y_[k] = np.zeros(max(num_scenarios, self.S))

			self.distances_[k][:num_scenarios] = distances
			self.diffs_x_[k][:num_scenarios] = diffs_x
			self.diffs_y_[k][:num_scenarios] = diffs_y

	def check_feasibility_by_distance(self, k: int, obstacle_id: int):
		"""
		Check feasibility based on computed distances

		Args:
			k: Time step index
			obstacle_id: Obstacle index
		"""
		if k >= len(self.distances_) or len(self.distances_[k]) == 0:
			return

		if obstacle_id >= len(self.radii_):
			return

		min_distance = np.min(self.distances_[k])
		required_distance = self.radii_[obstacle_id]

		# Check if any scenario violates the minimum distance requirement
		if min_distance < required_distance:
			if obstacle_id < len(self.distances_feasible_):
				self.distances_feasible_[obstacle_id] = False

			# Record infeasible scenarios
			infeasible_mask = self.distances_[k] < required_distance
			infeasible_indices = np.where(infeasible_mask)[0]

			if k < len(self.infeasible_scenario_idxs_):
				self.infeasible_scenario_idxs_[k].extend(infeasible_indices.tolist())

			# Record poses of infeasible scenarios
			if (self.scenarios_ is not None and obstacle_id < len(self.scenarios_) and
					k < len(self.scenarios_[obstacle_id])):
				scenarios_at_k = self.scenarios_[obstacle_id][k]
				if k < len(self.infeasible_scenario_poses_):
					for idx in infeasible_indices:
						if idx < len(scenarios_at_k):
							scenario = scenarios_at_k[idx]
							if hasattr(scenario, 'position'):
								pose = np.array([scenario.position[0], scenario.position[1]])
							elif isinstance(scenario, (list, tuple)) and len(scenario) >= 2:
								pose = np.array([scenario[0], scenario[1]])
							else:
								pose = np.array([0.0, 0.0])
							self.infeasible_scenario_poses_[k].append(pose)

	def compute_halfspaces(self, k: int, obstacle_id: int):
		"""
		Use distances, diffs_x and diffs_y to compute constraints for all scenarios

		Args:
			k: Time step index
			obstacle_id: Obstacle index
		"""
		if (k >= len(self.distances_) or len(self.distances_[k]) == 0 or
				k >= len(self.diffs_x_) or len(self.diffs_x_[k]) == 0):
			return

		distances = self.distances_[k]
		diffs_x = self.diffs_x_[k]
		diffs_y = self.diffs_y_[k]

		if obstacle_id >= len(self.radii_):
			return

		required_distance = self.radii_[obstacle_id]
		num_scenarios = len(distances)

		# Compute constraint coefficients
		a1_vals = np.zeros(num_scenarios)
		a2_vals = np.zeros(num_scenarios)
		b_vals = np.zeros(num_scenarios)

		for i in range(num_scenarios):
			if distances[i] > 1e-8:  # Avoid division by zero
				# Normalize difference vector
				a1_vals[i] = diffs_x[i] / distances[i]
				a2_vals[i] = diffs_y[i] / distances[i]

				# Constraint: a1*x + a2*y <= b
				# This represents: distance from vehicle to obstacle >= required_distance
				b_vals[i] = required_distance
			else:
				# Handle degenerate case
				a1_vals[i] = 1.0
				a2_vals[i] = 0.0
				b_vals[i] = required_distance

		# Store constraint coefficients with bounds checking
		if k < len(self.a1_):
			# Ensure we have space for all constraints
			max_constraints = len(self.a1_[k])
			start_idx = obstacle_id * self.S
			end_idx = min(start_idx + num_scenarios, max_constraints)
			actual_scenarios = end_idx - start_idx

			if start_idx < max_constraints and actual_scenarios > 0:
				self.a1_[k][start_idx:end_idx] = a1_vals[:actual_scenarios]
				self.a2_[k][start_idx:end_idx] = a2_vals[:actual_scenarios]
				self.b_[k][start_idx:end_idx] = b_vals[:actual_scenarios]

		# Create constraint metadata
		if (self.scenarios_ is not None and obstacle_id < len(self.scenarios_) and
				k < len(self.scenarios_[obstacle_id]) and k < len(self.constraints_)):
			scenarios_at_k = self.scenarios_[obstacle_id][k]
			for i in range(min(num_scenarios, len(scenarios_at_k))):
				if hasattr(scenarios_at_k[i], 'position'):
					scenario_obj = Scenario(sample_idx=i, obstacle_idx=obstacle_id)
				else:
					scenario_obj = Scenario(idx_=i, obstacle_idx_=obstacle_id)

				constraint = ScenarioConstraint(
					a1=a1_vals[i] if i < len(a1_vals) else 1.0,
					a2=a2_vals[i] if i < len(a2_vals) else 0.0,
					b=b_vals[i] if i < len(b_vals) else 1.0,
					scenario_=scenario_obj,
					type_=ObstacleType.DYNAMIC,
					side_=ConstraintSide.UNDEFINED
				)

				# Only add if we have space
				if len(self.constraints_[k]) < len(self.a1_[k]):
					self.constraints_[k].append(constraint)

	def construct_polytopes(self, k: int, data):
		"""
		Construct the polytope given all computed constraints

		Args:
			k: Time step index
			data: Real-time data
		"""
		if k >= len(self.polytopes_):
			return

		# Collect all constraint points for this time step
		constraint_points = []

		if (k < len(self.a1_) and len(self.a1_[k]) > 0 and
				k < len(self.a2_) and len(self.a2_[k]) > 0):

			for i in range(len(self.a1_[k])):
				if abs(self.a1_[k][i]) > 1e-8 or abs(self.a2_[k][i]) > 1e-8:
					# Convert constraint to point representation for polygon computation
					# This is a simplified approach - in practice you'd use proper geometric algorithms
					point = np.array([self.a1_[k][i], self.a2_[k][i]])
					constraint_points.append(point)

		if len(constraint_points) > 2:
			# Compute minimal polygon
			constraint_points_array = np.array(constraint_points)
			polygon_constraints = self.polytopes_[k].compute_minimal_polygon(constraint_points_array)

			# Update constraint data with minimal polygon
			self.polytopes_[k].polygon_out.clear()
			for i, (a1, a2, b) in enumerate(polygon_constraints):
				# Create scenario for the constraint
				scenario_obj = Scenario(idx_=i, obstacle_idx_=-1)  # Aggregate constraint

				constraint = ScenarioConstraint(
					a1=a1,
					a2=a2,
					b=b,
					scenario_=scenario_obj,
					type_=ObstacleType.DYNAMIC,
					side_=ConstraintSide.UNDEFINED
				)
				self.polytopes_[k].polygon_out.append(constraint)

	def push_algorithm(self, data):
		"""Push the initial plan away from scenarios if infeasible (orthogonal to vehicle plan)"""
		# Implementation for projection algorithm
		pass

	def dr_projection(self, data):
		"""Project the plan to feasibility using (Cyclic) Douglas-Rachford Splitting"""
		# Implementation for Douglas-Rachford projection
		pass

	def get_scenario_location(self, k: int, obstacle_index: int, scenario_index: int) -> np.ndarray:
		"""
		Get 3D scenario location for visualization

		Returns:
			np.ndarray: [x, y, z] coordinates
		"""
		if (self.scenarios_ is None or obstacle_index >= len(self.scenarios_) or
				k >= len(self.scenarios_[obstacle_index]) or
				scenario_index >= len(self.scenarios_[obstacle_index][k])):
			return np.array([0.0, 0.0, 0.0])

		scenario = self.scenarios_[obstacle_index][k][scenario_index]
		if hasattr(scenario, 'position'):
			return np.array([scenario.position[0], scenario.position[1], 0.0])
		elif isinstance(scenario, (list, tuple)) and len(scenario) >= 2:
			return np.array([scenario[0], scenario[1], 0.0])
		else:
			return np.array([0.0, 0.0, 0.0])

	def get_scenario_location_2d(self, k: int, obstacle_index: int, scenario_index: int) -> np.ndarray:
		"""
		Get 2D scenario location

		Returns:
			np.ndarray: [x, y] coordinates
		"""
		location_3d = self.get_scenario_location(k, obstacle_index, scenario_index)
		return location_3d[:2]

	def is_data_ready(self, data):
		"""Check if all required data is available for this disc"""
		try:
			# Check if sampler has scenarios ready
			if self.sampler is not None:
				if not hasattr(self.sampler, 'samples_ready') or not self.sampler.samples_ready():
					return False

				# Check if scenarios are available
				if not hasattr(self.sampler, 'samples') or self.sampler.samples is None:
					return False

			# Check if obstacle data is available
			if not hasattr(data, 'dynamic_obstacles') or data.dynamic_obstacles is None:
				return False

			return True

		except Exception as e:
			LOG_DEBUG(f"Error checking data readiness for disc {self.disc_id_}: {e}")
			return False

	def reset(self):
		"""Reset the disc state"""
		self.status = 0  # RESET equivalent
		self.is_feasible_ = True
		self.clear_all()

	# Visualization methods (placeholders - implement based on your visualization framework)
	def visualize_all_constraints(self):
		"""Visualize all scenario constraints"""
		LOG_DEBUG(f"Visualizing constraints for disc {self.disc_id_}")

	def visualize_scenarios(self, data):
		"""Visualize all scenarios (can be slow)"""
		LOG_DEBUG(f"Visualizing all scenarios for disc {self.disc_id_}")

	def visualize_selected_scenarios(self, data):
		"""Visualize support scenarios"""
		LOG_DEBUG(f"Visualizing selected scenarios for disc {self.disc_id_}")

	def visualize_polygon_scenarios(self):
		"""Visualize scenarios that contribute to final constraints"""
		LOG_DEBUG(f"Visualizing polygon scenarios for disc {self.disc_id_}")

	def visualize_polygons(self):
		"""Visualize the final constraint polygons"""
		LOG_DEBUG(f"Visualizing constraint polygons for disc {self.disc_id_}")

	def get_draw_indices(self):
		"""Get indices of scenarios to draw for visualization"""
		return list(range(min(10, self.S)))  # Limit to first 10 for performance

	def get_config_value(self, key: str, default=None):
		"""Get configuration value with fallback"""
		return get_config_dotted(self.config, key) if self.config else default