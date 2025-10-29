import logging

import numpy as np
from scipy.interpolate import CubicSpline

from planner_modules.src.constraints.base_constraint import BaseConstraint
from planning.src.types import State
from utils.math_utils import EllipsoidDecomp, rotation_matrix
from utils.utils import LOG_DEBUG, PROFILE_SCOPE

class OccupiedPositionStore():
	def __init__(self):
		self.occupied_positions = []

	def reset(self):
		self.occupied_positions = []

	def add(self, position):
		self.occupied_positions.append(position)

	def set(self, entry, index):
		self.occupied_positions[index] = entry

	def get_occupied_positions(self):
		return self.occupied_positions

	def __len__(self):
		return len(self.occupied_positions)


class DecompConstraints(BaseConstraint):
	def __init__(self, solver):
		super().__init__(solver)

		self.name = 'decomp_constraints'

		# Create EllipsoidDecomp instance
		self.decomp_util = EllipsoidDecomp()

		# Only look around for obstacles using a box with sides of width 2*range
		self.range = self.get_config_value("decomp.range")

		self.decomp_util.set_local_bbox(np.array([self.range, self.range]))

		self.occ_pos = OccupiedPositionStore()

		self.num_discs = self.get_config_value("num_discs")  # Is overwritten to 1 for topology constraints

		self.max_constraints = self.get_config_value("decomp.max_constraints")

		self.max_num_lin_constraints = self.max_constraints * self.num_discs

		self.use_slack = self.get_config_value("decomp.use_slack")

		# Initialize constraint storage
		self._a1 = [None] * self.num_discs
		self._a2 = [None] * self.num_discs
		self._b = [None] * self.num_discs

		for disc_id in range(self.num_discs):
			self._a1[disc_id] = [None] * (self.solver.horizon + 1)
			self._a2[disc_id] = [None] * (self.solver.horizon + 1)
			self._b[disc_id] = [None] * (self.solver.horizon + 1)
			for step in range(self.solver.horizon + 1):
				self._a1[disc_id][step] = [None] * self.max_constraints
				self._a2[disc_id][step] = [None] * self.max_constraints
				self._b[disc_id][step] = [None] * self.max_constraints

		self.num_active_obstacles = 0

		# Store dummy values for invalid states
		self._dummy_a1 = 0.0
		self._dummy_a2 = 0.0
		self._dummy_b = 100.0

		LOG_DEBUG("Decomp Constraints successfully initialized")

	def update(self, state: State, data):
		PROFILE_SCOPE("DecompConstraints.update")
		LOG_DEBUG("DecompConstraints.update")

		self._dummy_b = state.get("x") + 100.0

		self.get_occupied_grid_cells(data)  # Retrieve occupied points from the costmap
		self.decomp_util.set_obs(self.occ_pos.get_occupied_positions())  # Set obstacles

		path = []
		spline_progress = state.get("spline")
		for step in range(self.solver.horizon + 1):
			ref_states = self.solver.get_reference_trajectory().get_states()
			pred_vel = ref_states[step].get("v")
			# Get reference trajectory position
			LOG_DEBUG("fetched reference states: {}".format(ref_states))

			if data.reference_path is not None: # if we have a reference path we use global path values
				data.reference_path.x_spline = CubicSpline(data.reference_path.s, data.reference_path.x)
				data.reference_path.y_spline = CubicSpline(data.reference_path.s, data.reference_path.y)
				path_point_x = data.reference_path.x_spline(spline_progress)
				path_point_y = data.reference_path.y_spline(spline_progress)
				path_position = np.array([path_point_x, path_point_y])
				path.append(path_position)

			else: # without a reference path we use local path values

				pos = np.array([
					ref_states[step].get("x"),
					ref_states[step].get("y")
				])

				path.append(np.array(pos))

			spline_progress += pred_vel * self.solver.timestep

		self.decomp_util.dilate(path, 0.0)
		LOG_DEBUG("After dilation decomp polyhedrons are: " + str(self.decomp_util.get_polyhedrons()))
		constraints = []
		LOG_DEBUG("Constraints before setting: {}".format(constraints))
		self.decomp_util.set_constraints(constraints, 1.0)  # Injects the constraints into the provided list
		constraints_after_set = []
		for constraint in constraints:
			constraints_after_set.append(constraint)
		LOG_DEBUG("Constraints after setting: {}".format(constraints_after_set))
		polyhedrons = self.decomp_util.get_polyhedrons()

		max_decomp_constraints = 0

		for step in range(self.solver.horizon):
			for disc_id in range(self.num_discs):
				LOG_DEBUG("step: {}".format(step))
				constraints_for_step = constraints[step]
				max_decomp_constraints = max(max_decomp_constraints, constraints_for_step.A_.shape[0])
				self.num_active_obstacles = min(constraints_for_step.A_.shape[0], self.max_constraints)
				constraint_counter = 0
				for constraint_id in range(self.num_active_obstacles):
					a1_val = constraints_for_step.A_[constraint_id, 0]
					a2_val = constraints_for_step.A_[constraint_id, 1]
					b_val = constraints_for_step.b_[constraint_id]
					LOG_DEBUG("For step {}, disc {}, constraint_id {}, setting a1 {}, a2 {}, and b {}".format(step, disc_id, constraint_id, a1_val, a2_val, b_val))
					if np.isnan(a1_val) or np.isnan(a2_val) or np.isnan(b_val):
						continue  # or break, depending on intent

					self._a1[disc_id][step + 1][constraint_id] = a1_val
					self._a2[disc_id][step + 1][constraint_id] = a2_val
					self._b[disc_id][step + 1][constraint_id] = b_val
					constraint_counter += 1

				for extra_constraint_id in range(constraint_counter, self.max_constraints):
					self._a1[disc_id][step + 1][extra_constraint_id] = self._dummy_a1
					self._a2[disc_id][step + 1][extra_constraint_id] = self._dummy_a2
					self._b[disc_id][step + 1][extra_constraint_id] = self._dummy_b

		for step in range(self.solver.horizon):
			const_for_step = constraints[step]
			LOG_DEBUG(f"Step {step} has {const_for_step.A_.shape[0]} constraints")
			LOG_DEBUG(f"A: {const_for_step.A_}, b: {const_for_step.b_}")

		if max_decomp_constraints > self.max_constraints:
			logging.warning(
				f"Maximum number of decomp util constraints exceeds specification: {max_decomp_constraints} > {self.max_constraints}")

		LOG_DEBUG("DecompConstraints::update done")


	def constraint_name(self, index, step, disc_id=None):
		"""Generate constraint parameter names"""
		if disc_id is None:
			return f"decomp_obs_{index}_step_{step}"
		return f"disc_{disc_id}_decomp_constraint_{index}_step_{step}"

	def get_occupied_grid_cells(self, data):
		LOG_DEBUG("get_occupied_grid_cells")

		costmap = data.costmap

		# Store all occupied cells in the grid map
		self.occ_pos.reset()

		for x_pos in range(costmap.get_size_in_cells_x()):
			for y_pos in range(costmap.get_size_in_cells_y()):
				if costmap.getCost(x_pos, y_pos) == 0:  # Assuming FREE_SPACE is 0
					continue

				x, y = costmap.map_to_world(x_pos, y_pos)
				self.occ_pos.add(np.array([x, y]))

		return True

	def define_parameters(self, params):
		for disc_id in range(self.num_discs):
				# Set solver parameter for ego disc offset
				params.add(f"ego_disc_{disc_id}_offset")

				for cons_id in range(self.max_num_lin_constraints):
					for step in range(self.solver.horizon + 1):
						base_name = self.constraint_name(cons_id, step, disc_id)
						params.add(f"{base_name}_a1")
						params.add(f"{base_name}_a2")
						params.add(f"{base_name}_b")

	def set_parameters(self, parameter_manager, data, step):
		if step == 0:  # Initialize with dummy values
			for disc_id in range(self.num_discs):
				if data.has('robot_area') and len(data.robot_area) > disc_id:
					parameter_manager.set_parameter(f"ego_disc_{disc_id}_offset", data.robot_area[disc_id].offset)

				for constraint_id in range(self.max_constraints):
					base_name = self.constraint_name(constraint_id, step, disc_id)
					parameter_manager.set_parameter(f"{base_name}_a1", self._dummy_a1)
					parameter_manager.set_parameter(f"{base_name}_a2", self._dummy_a2)
					parameter_manager.set_parameter(f"{base_name}_b", self._dummy_b)
			return

		# Set actual constraint values for the given step

		for disc_id in range(self.num_discs):
			if data.has('robot_area') and len(data.robot_area) > disc_id:
				parameter_manager.set_parameter(f"ego_disc_{disc_id}_offset", data.robot_area[disc_id].offset)

			for constraint_id in range(self.max_constraints):
				for time_step in range(1, self.solver.horizon + 1):
					base_name = self.constraint_name(constraint_id, time_step, disc_id)

					parameter_manager.set_parameter(f"{base_name}_a1", self._a1[disc_id][time_step][constraint_id])
					parameter_manager.set_parameter(f"{base_name}_a2", self._a2[disc_id][time_step][constraint_id])
					parameter_manager.set_parameter(f"{base_name}_b", self._b[disc_id][time_step][constraint_id])

	def get_constraints(self, symbolic_state, params, stage_idx):
		constraints = []

		if stage_idx >= self.solver.horizon:
			return constraints

		# States
		pos_x = symbolic_state.get("x")
		pos_y = symbolic_state.get("y")
		pos = np.array([pos_x, pos_y])
		psi = symbolic_state.get("psi")

		try:
			slack = symbolic_state.get("slack") if self.use_slack else 0.0
		except:
			slack = 0.0

		rotation_car = rotation_matrix(psi)

		for disc_id in range(self.num_discs):
			disc_x = params.get(f"ego_disc_{disc_id}_offset")
			disc_relative_pos = np.array([disc_x, 0])
			disc_pos = pos + rotation_car.dot(disc_relative_pos)
			for index in range(self.num_active_obstacles):
				base_name = self.constraint_name(index, stage_idx, disc_id)
				a1 = params.get(f"{base_name}_a1")
				a2 = params.get(f"{base_name}_a2")
				b = params.get(f"{base_name}_b")
				LOG_DEBUG("for stage index {}, obstacle index {}, disc id: {}, a1: {}, a2: {}, b: {}".format(stage_idx, index, disc_id, a1, a2, b))
				# Skip dummy/invalid constraints
				if abs(a1) < 1e-6 and abs(a2) < 1e-6:
					LOG_DEBUG(f"Skipping dummy constraint for obstacle {index}, disc {disc_id}")
					continue

				constraint_expr = a1 * disc_pos[0] + a2 * disc_pos[1] - (b + slack)
				constraints.append(constraint_expr)

		return constraints

	def is_data_ready(self, data):
		missing_data = ""
		if  not data.has("costmap"):
			missing_data = "costmap"
		LOG_DEBUG("Missing data: %s" % missing_data)
		return len(missing_data) < 1

	def get_lower_bound(self):
		lower_bound = []
		for index in range(0, self.max_num_lin_constraints):
			lower_bound.append(-np.inf)
		return lower_bound

	def get_upper_bound(self):
		upper_bound = []
		for index in range(0, self.max_num_lin_constraints):
			upper_bound.append(0.0)
		return upper_bound

