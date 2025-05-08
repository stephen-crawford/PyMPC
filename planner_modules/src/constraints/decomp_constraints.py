import logging
import numpy as np

from planner.src.types import State
from utils.utils import LOG_DEBUG, PROFILE_SCOPE
from utils.visualizer import ROSLine, ROSPointMarker
from utils.utils import EllipsoidDecomp2D

from planner_modules.src.constraints.base_constraint import BaseConstraint


class DecompConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)

		self.name = 'decomp_constraints'
		self.get_num_segments = self.get_config_value("contouring.num_segments")

		# Create EllipsoidDecomp2D instance
		self.decomp_util = EllipsoidDecomp2D()

		# Only look around for obstacles using a box with sides of width 2*range
		self.range = self.get_config_value("decomp.range")
		self.decomp_util.set_local_bbox(np.array([self.range, self.range]))

		self.occ_pos = []  # List to store occupied positions

		self.n_discs = self.get_config_value("n_discs")  # Is overwritten to 1 for topology constraints

		self._max_constraints = self.get_config_value("decomp.max_constraints")

		self.n_constraints = self._max_constraints * self.n_discs

		self.nh = self.n_constraints
		self.use_slack = self.get_config_value("decomp.use_slack")

		# Initialize constraint storage
		self.a1 = [[[0.0 for _ in range(self._max_constraints)] for _ in range(self.get_config_value("N"))] for _ in
				   range(self.n_discs)]
		self.a2 = [[[0.0 for _ in range(self._max_constraints)] for _ in range(self.get_config_value("N"))] for _ in
				   range(self.n_discs)]
		self.b = [[[0.0 for _ in range(self._max_constraints)] for _ in range(self.get_config_value("N"))] for _ in
				  range(self.n_discs)]

		LOG_DEBUG("Decomp Constraints successfully initialized")

	def update(self, state: State, data, module_data):
		PROFILE_SCOPE("DecompConstraints.update")
		LOG_DEBUG("DecompConstraints.update")

		_dummy_b = state.get("x") + 100.0
		_dummy_a1 = 0.0
		_dummy_a2 = 0.0

		self.get_occupied_grid_cells(data)  # Retrieve occupied points from the costmap
		self.decomp_util.set_obs(self.occ_pos)  # Set obstacles

		path = []
		s = state.get("spline")

		for k in range(self.solver.N):
			# Global (reference) path
			path_pos = module_data.path.get_point(s)
			path.append(np.array([path_pos[0], path_pos[1]]))

			v = self.solver.get_ego_prediction(k, "v")  # Use the predicted velocity

			s += v * self.solver.dt

		self.decomp_util.dilate(path, 0.0, False)

		constraints = []
		self.decomp_util.set_constraints(constraints, 0.0)  # Map is already inflated
		polyhedrons = self.decomp_util.get_polyhedrons()

		max_decomp_constraints = 0

		for k in range(self.solver.N - 1):
			if k >= len(constraints) or constraints[k] is None:
				continue

			constraint = constraints[k]
			max_decomp_constraints = max(max_decomp_constraints, constraint.A_.shape[0])

			for i in range(min(constraint.A_.shape[0], self._max_constraints)):
				if np.linalg.norm(constraint.A_[i]) < 1e-3 or np.isnan(constraint.A_[i, 0]):
					break

				self.a1[0][k + 1][i] = constraint.A_[i, 0]
				self.a2[0][k + 1][i] = constraint.A_[i, 1]
				self.b[0][k + 1][i] = constraint.b_[i]

			# Fill the remaining with dummy values
			for i in range(min(constraint.A_.shape[0], self._max_constraints), self._max_constraints):
				self.a1[0][k + 1][i] = _dummy_a1
				self.a2[0][k + 1][i] = _dummy_a2
				self.b[0][k + 1][i] = _dummy_b

		if max_decomp_constraints > self._max_constraints:
			logging.warning(
				f"Maximum number of decomp util constraints exceeds specification: {max_decomp_constraints} > {self._max_constraints}")

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
		for segment_index in range(self.num_segments):
			params.add(f"width_right{segment_index}_a")
			params.add(f"width_right{segment_index}_b")
			params.add(f"width_right{segment_index}_c")
			params.add(f"width_right{segment_index}_d")

			params.add(f"width_left{segment_index}_a")
			params.add(f"width_left{segment_index}_b")
			params.add(f"width_left{segment_index}_c")
			params.add(f"width_left{segment_index}_d")

	def set_parameters(self, parameter_manager, data, module_data, k):
		if k == 0:  # Dummies
			for d in range(self.n_discs):
				# Set solver parameter for ego disc offset
				if hasattr(data, 'robot_area') and len(data.robot_area) > d:
					self.set_solver_parameter("ego_disc_offset", data.robot_area[d].offset, k, d)

				constraint_counter = 0
				for i in range(self._max_constraints):
					# Use dummy values for k=0
					_dummy_a1 = 0.0
					_dummy_a2 = 0.0
					_dummy_b = 100.0  # Large dummy value

					parameter_manager.add(f"decomp_a1{k}", _dummy_a1, k, constraint_counter)
					self.set_solver_parameter("decomp_a2", _dummy_a2, k, constraint_counter)
					self.set_solver_parameter("decomp_b", _dummy_b, k, constraint_counter)
					constraint_counter += 1
			return

		if k == 1:
			LOG_DEBUG("DecompConstraints::set_parameters")

		constraint_counter = 0  # Necessary for now to map the disc and obstacle index to a single index
		for d in range(self.n_discs):
			# Set solver parameter for ego disc offset
			if hasattr(data, 'robot_area') and len(data.robot_area) > d:
				self.set_solver_parameter("ego_disc_offset", data.robot_area[d].offset, k, d)

			for i in range(self._max_constraints):
				self.set_solver_parameter("decomp_a1", self.a1[d][k][i], k, constraint_counter)
				self.set_solver_parameter("decomp_a2", self.a2[d][k][i], k, constraint_counter)
				self.set_solver_parameter("decomp_b", self.b[d][k][i], k, constraint_counter)
				constraint_counter += 1

	def is_data_ready(self, data):
		missing_data = ""
		if "costmap" not in data:
			missing_data = "costmap"

		return len(missing_data) < 1
