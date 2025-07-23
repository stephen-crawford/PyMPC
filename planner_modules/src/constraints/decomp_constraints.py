import logging
import numpy as np

from planning.src.types import State
from utils.math_utils import EllipsoidDecomp, rotation_matrix
from utils.utils import LOG_DEBUG, PROFILE_SCOPE

from planner_modules.src.constraints.base_constraint import BaseConstraint


class DecompConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)

		self.name = 'decomp_constraints'
		self.num_segments = self.get_config_value("contouring.num_segments")

		# Create EllipsoidDecomp instance
		self.decomp_util = EllipsoidDecomp()

		# Only look around for obstacles using a box with sides of width 2*range
		self.range = self.get_config_value("decomp.range")

		self.decomp_util.set_local_bbox(np.array([self.range, self.range]))

		self.occ_pos = []  # List to store occupied positions

		self.n_discs = self.get_config_value("num_discs")  # Is overwritten to 1 for topology constraints

		self.max_obstacles = self.get_config_value("max_obstacles")

		self.num_constraints = self.max_obstacles * self.n_discs

		self.use_slack = self.get_config_value("decomp.use_slack")
		self.num_discs = self.get_config_value("num_discs")
		# Initialize constraint storage
		self.a1 = [[[0.0 for _ in range(self.max_obstacles)] for _ in range(self.solver.horizon)] for _ in
				   range(self.n_discs)]
		self.a2 = [[[0.0 for _ in range(self.max_obstacles)] for _ in range(self.solver.horizon)] for _ in
				   range(self.n_discs)]
		self.b = [[[0.0 for _ in range(self.max_obstacles)] for _ in range(self.solver.horizon)] for _ in
				  range(self.n_discs)]

		LOG_DEBUG("Decomp Constraints successfully initialized")

	def update(self, state: State, data):
		PROFILE_SCOPE("DecompConstraints.update")
		LOG_DEBUG("DecompConstraints.update")

		_dummy_b = state.get("x") + 100.0
		_dummy_a1 = 0.0
		_dummy_a2 = 0.0

		self.get_occupied_grid_cells(data)  # Retrieve occupied points from the costmap
		self.decomp_util.set_obs(self.occ_pos)  # Set obstacles

		path = []
		s = state.get("spline")

		for k in range(1, self.solver.horizon):
			# Get reference trajectory position
			ref_states = self.solver.get_reference_trajectory().get_states()
			LOG_DEBUG("fetched reference states: {}".format(ref_states))
			if k >= len(ref_states):
				continue

			pos = np.array([
				ref_states[k].get("x"),
				ref_states[k].get("y")
			])

			path.append(np.array(pos))

			v = ref_states[k].get("v")  # Use the predicted velocity

			s += v * self.solver.timestep

		self.decomp_util.dilate(path, 0.0, False)

		constraints = []
		self.decomp_util.set_constraints(constraints, 0.0)  # Map is already inflated
		polyhedrons = self.decomp_util.get_polyhedrons()

		max_decomp_constraints = 0

		for k in range(self.solver.horizon - 1):
			if k >= len(constraints) or constraints[k] is None:
				continue

			constraint = constraints[k]
			max_decomp_constraints = max(max_decomp_constraints, constraint.A_.shape[0])

			for i in range(min(constraint.A_.shape[0], self.num_constraints)):
				if np.linalg.norm(constraint.A_[i]) < 1e-3 or np.isnan(constraint.A_[i, 0]):
					break

				self.a1[0][k + 1][i] = constraint.A_[i, 0]
				self.a2[0][k + 1][i] = constraint.A_[i, 1]
				self.b[0][k + 1][i] = constraint.b_[i]

			# Fill the remaining with dummy values
			for i in range(min(constraint.A_.shape[0], self.max_obstacles), self.num_constraints):
				self.a1[0][k + 1][i] = _dummy_a1
				self.a2[0][k + 1][i] = _dummy_a2
				self.b[0][k + 1][i] = _dummy_b

		if max_decomp_constraints > self.num_constraints:
			logging.warning(
				f"Maximum number of decomp util constraints exceeds specification: {max_decomp_constraints} > {self.num_constraints}")

		LOG_DEBUG("DecompConstraints::update done")

	def get_occupied_grid_cells(self, data):
		LOG_DEBUG("get_occupied_grid_cells")

		costmap = data.costmap

		# Store all occupied cells in the grid map
		self.occ_pos.clear()

		for i in range(costmap.get_size_in_cells_x()):
			for j in range(costmap.get_size_in_cells_y()):
				if costmap.getCost(i, j) == 0:  # Assuming FREE_SPACE is 0
					continue

				x, y = costmap.map_to_world(i, j)
				self.occ_pos.append(np.array([x, y]))

		return True

	def define_parameters(self, params):
		for d in range(self.n_discs):
				# Set solver parameter for ego disc offset
				params.add(f"ego_disc_{d}_offset")

				for i in range(self.max_obstacles):
					# Use dummy values for k=0
					_dummy_a1 = 0.0
					_dummy_a2 = 0.0
					_dummy_b = 100.0  # Large dummy value
					params.add(f"disc_{d}_decomp_{i}_a1")
					params.add(f"disc_{d}_decomp_{i}_a2")
					params.add(f"disc_{d}_decomp_{i}_b")

	def set_parameters(self, parameter_manager, data, k):
		if k == 0:  # Dummies
			for d in range(self.n_discs):
				# Set solver parameter for ego disc offset
				if data.has('robot_area') and len(data.robot_area) > d:
					LOG_DEBUG("Robot area offset is " + str(data.robot_area[d].offset))
					parameter_manager.set_parameter(f"ego_disc_{d}_offset", data.robot_area[d].offset)

				constraint_counter = 0
				for i in range(self.max_obstacles):
					# Use dummy values for k=0
					_dummy_a1 = 0.0
					_dummy_a2 = 0.0
					_dummy_b = 100.0  # Large dummy value
					parameter_manager.set_parameter(f"disc_{d}_decomp_{i}_a1", _dummy_a1)
					parameter_manager.set_parameter(f"disc_{d}_decomp_{i}_a2", _dummy_a2)
					parameter_manager.set_parameter(f"disc_{d}_decomp_{i}_b", _dummy_b)
					constraint_counter += 1
			return

		if k == 1:
			LOG_DEBUG("DecompConstraints::set_parameters")

		constraint_counter = 0  # Necessary for now to map the disc and obstacle index to a single index
		for d in range(self.n_discs):
			# Set solver parameter for ego disc offset
			if data.has('robot_area') and len(data.robot_area) > d:
				parameter_manager.set_parameter(f"ego_disc_{d}_offset", data.robot_area[d].offset)

			for i in range(self.num_constraints):
				parameter_manager.set_parameter(f"disc_{d}_decomp_{i}_a1", self.a1[d][k][i])
				parameter_manager.set_parameter(f"disc_{d}_decomp_{i}_a2", self.a2[d][k][i])
				parameter_manager.set_parameter(f"disc_{d}_decomp_{i}_b", self.b[d][k][i])
				constraint_counter += 1

	def get_constraints(self, symbolic_state, params, stage_idx):
		"""
		Fixed method signature to match BaseConstraint interface
		"""
		constraints = []

		# States
		pos_x = symbolic_state.get("x")
		pos_y = symbolic_state.get("y")
		pos = np.array([pos_x, pos_y])
		psi = symbolic_state.get("psi")

		try:
			if self.use_slack:
				slack = symbolic_state.get("slack")
			else:
				slack = 0.0
		except:
			slack = 0.0

		rotation_car = rotation_matrix(psi)
		for disc_id in range(self.num_discs):
			disc_x = params.get(f"ego_disc_{disc_id}_offset")
			disc_relative_pos = np.array([disc_x, 0])
			disc_pos = pos + rotation_car.dot(disc_relative_pos)

			for index in range(self.num_constraints):
				a1 = params.get(self.constraint_name(index, disc_id) + "_a1")
				a2 = params.get(self.constraint_name(index, disc_id) + "_a2")
				b = params.get(self.constraint_name(index, disc_id) + "_b")

				expr = a1 * disc_pos[0] + a2 * disc_pos[1] - (b + slack)
				# ✅ Skip constant constraints
				if hasattr(expr, "is_constant") and expr.is_constant():
					continue
				constraints.append(expr)

		return constraints

	def is_data_ready(self, data):
		missing_data = ""
		if  not data.has("costmap"):
			missing_data = "costmap"
		LOG_DEBUG("Missing data: %s" % missing_data)
		return len(missing_data) < 1

	def constraint_name(self, index, disc_id):
		return f"disc_{disc_id}_decomp_{index}"

	def get_lower_bound(self):
		lower_bound = []
		for index in range(0, self.num_constraints):
			lower_bound.append(-np.inf)
		return lower_bound

	def get_upper_bound(self):
		upper_bound = []
		for index in range(0, self.num_constraints):
			upper_bound.append(0.0)
		return upper_bound

