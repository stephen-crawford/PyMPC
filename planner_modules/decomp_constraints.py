import logging
import numpy as np
from utils.const import CONSTRAINT
from utils.utils import read_config_file, LOG_DEBUG, PROFILE_SCOPE
from utils.visualizer import ROSLine, ROSPointMarker

from utils.utils import EllipsoidDecomp2D

CONFIG = read_config_file()


class DecompConstraints:
	def __init__(self, solver):
		self.solver = solver
		self.module_type = CONSTRAINT
		self.name = "decomp_constraints"
		LOG_DEBUG("Initializing Decomp Constraints")

		self.get_num_segments = CONFIG["contouring"]["num_segments"]
		# Create EllipsoidDecomp2D instance
		self.decomp_util = EllipsoidDecomp2D()

		# Only look around for obstacles using a box with sides of width 2*range
		self.range = CONFIG["decomp"]["range"]
		self.decomp_util.set_local_bbox(np.array([self.range, self.range]))

		self.occ_pos = []  # List to store occupied positions

		self.n_discs = CONFIG["n_discs"]  # Is overwritten to 1 for topology constraints

		self._max_constraints = CONFIG["decomp"]["max_constraints"]
		# Initialize constraint storage
		self.a1 = [[[0.0 for _ in range(self._max_constraints)] for _ in range(CONFIG["N"])] for _ in
		           range(self.n_discs)]
		self.a2 = [[[0.0 for _ in range(self._max_constraints)] for _ in range(CONFIG["N"])] for _ in
		           range(self.n_discs)]
		self.b = [[[0.0 for _ in range(self._max_constraints)] for _ in range(CONFIG["N"])] for _ in
		          range(self.n_discs)]

		LOG_DEBUG("Decomp Constraints successfully initialized")

	def update(self, state, data, module_data):
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

	def set_parameters(self, data, module_data, k):
		if k == 0:  # Dummies
			for d in range(self.n_discs):
				# Set solver parameter for ego disc offset
				self.set_solver_parameter_ego_disc_offset(k, self.solver.params, data.robot_area[d].offset, d)

				constraint_counter = 0
				for i in range(self._max_constraints):
					# Use dummy values for k=0
					_dummy_a1 = 0.0
					_dummy_a2 = 0.0
					_dummy_b = 100.0  # Large dummy value

					self.set_solver_parameter_decomp_a1(k, self.solver.params, _dummy_a1, constraint_counter)
					self.set_solver_parameter_decomp_a2(k, self.solver.params, _dummy_a2, constraint_counter)
					self.set_solver_parameter_decomp_b(k, self.solver.params, _dummy_b, constraint_counter)
					constraint_counter += 1
			return

		if k == 1:
			LOG_DEBUG("DecompConstraints::set_parameters")

		constraint_counter = 0  # Necessary for now to map the disc and obstacle index to a single index
		for d in range(self.n_discs):
			# Set solver parameter for ego disc offset
			if hasattr(data, 'robot_area') and len(data.robot_area) > d:
				self.set_solver_parameter_ego_disc_offset(k, self.solver.params, data.robot_area[d].offset, d)

			for i in range(self._max_constraints):
				self.set_solver_parameter_decomp_a1(k, self.solver.params, self.a1[d][k][i], constraint_counter)
				self.set_solver_parameter_decomp_a2(k, self.solver.params, self.a2[d][k][i], constraint_counter)
				self.set_solver_parameter_decomp_b(k, self.solver.params, self.b[d][k][i], constraint_counter)
				constraint_counter += 1

	# Add setter methods like in ContouringConstraints
	def set_solver_parameter_ego_disc_offset(self, k, params, value, d):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "ego_disc_offset", value, k, index=d, settings=CONFIG)

	def set_solver_parameter_decomp_a1(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "decomp_a1", value, k, index=i, settings=CONFIG)

	def set_solver_parameter_decomp_a2(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "decomp_a2", value, k, index=i, settings=CONFIG)

	def set_solver_parameter_decomp_b(self, k, params, value, i):
		from solver.solver_interface import set_solver_parameter
		set_solver_parameter(params, "decomp_b", value, k, index=i, settings=CONFIG)

	def is_data_ready(self, data, missing_data):
		if data.costmap is None:
			missing_data += "Costmap "
			return False

		return True

	def project_to_safety(self, pos):
		"""Project to a collision free position if necessary."""
		if not self.occ_pos:  # Empty list check in Python
			return

		# This would need a separate implementation of douglas_rachford_projection
		# For now we'll skip the implementation
		pass

	def visualize(self, data, module_data):
		if not CONFIG["debug_visuals"]:
			return

		LOG_DEBUG("DecompConstraints.Visualize")

		# Create a publisher for free space visualization
		free_space_publisher = ROSLine(self.name + "/free_space")
		point_marker = ROSPointMarker(self.name + "/points")

		points = point_marker.get_new_point_marker("CUBE")
		points.set_scale(0.1, 0.1, 0.1)
		points.set_color(1, 0, 0, 1)

		line = free_space_publisher.add_new_line()
		line.set_scale(0.1)

		visualize_points = False

		polyhedrons = self.decomp_util.get_polyhedrons()

		k = 0
		while k < self.solver.N and k < len(polyhedrons):
			poly = polyhedrons[k]
			line.set_color_int(k, self.solver.N)

			vertices = poly.vertices
			if len(vertices) < 2:
				k += CONFIG["visualization"]["draw_every"]
				continue

			for i in range(len(vertices)):
				if visualize_points:
					points.add_point_marker((vertices[i][0], vertices[i][1], 0))

				if i > 0:
					line.add_line((vertices[i - 1][0], vertices[i - 1][1], 0),
					              (vertices[i][0], vertices[i][1], 0))

			# Close the loop
			line.add_line((vertices[-1][0], vertices[-1][1], 0),
			              (vertices[0][0], vertices[0][1], 0))

			k += CONFIG["visualization"]["draw_every"]

		free_space_publisher.publish()

		if not CONFIG["debug_visuals"]:
			return

		# Create a publisher for map visualization
		map_publisher = ROSPointMarker(self.name + "/map")
		point = map_publisher.get_new_point_marker("CUBE")
		point.set_scale(0.1, 0.1, 0.1)
		point.set_color(0, 0, 0, 1)

		for vec in self.occ_pos:
			point.add_point_marker((vec[0], vec[1], 0))

		map_publisher.publish()