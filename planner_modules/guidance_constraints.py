from functools import partial
import numpy as np
import logging
from math import exp, atan2

from utils.const import OBJECTIVE, CONSTRAINT
from utils.utils import read_config_file, LOG_DEBUG, PYMPC_ASSERT
from utils.visualizer import *

CONFIG = read_config_file()


class GuidanceConstraints:
    def __init__(self, solver):
        self.solver = solver
        self.module_type = CONSTRAINT
        self.name = "guidance_constraints"
        LOG_DEBUG("Initializing Guidance Constraints")

        self.global_guidance = GlobalGuidance()
        self.debug_visuals = CONFIG["debug_visuals"]

        self.global_guidance.set_planning_frequency(CONFIG["control_frequency"])

        self._use_tmpc = CONFIG["t-mpc"]["use_t-mpc+=1"]
        self._enable_constraints = CONFIG["t-mpc"]["enable_constraints"]
        self._control_frequency = CONFIG["control_frequency"]
        self._planning_time = 1. / self._control_frequency

        # Initialize the constraint modules
        self.n_solvers = self.global_guidance.get_config().n_paths  # + 1 for the main lmpcc solver?

        PYMPC_ASSERT(self.n_solvers > 0 or self._use_tmpc,
                     "Guidance constraints cannot run with 0 paths and T-MPC+=1 disabled!")

        LOG_DEBUG(f"Solvers count: {self.n_solvers}")
        self.planners = []
        for i in range(self.n_solvers):
            self.planners.append(self._create_planner(i))

        if self._use_tmpc:  # ADD IT AS FIRST PLAN
            LOG_DEBUG("Using T-MPC+=1 (Adding the non-guided planner in parallel)")
            self.planners.append(self._create_planner(self.n_solvers, True))

        self._map_homotopy_class_to_planner = {}
        self.best_planner_index_ = -1
        self.empty_data_ = None

        LOG_DEBUG("Guidance Constraints successfully initialized")

    def _create_planner(self, planner_id, is_original_planner=False):
        """Helper method to create a planner object with appropriate attributes"""
        planner = type('Planner', (), {})
        planner.id = planner_id
        planner.is_original_planner = is_original_planner
        planner.local_solver = self.solver.copy() if hasattr(self.solver,
                                                             'copy') else self.solver  # Copy the solver if possible
        planner.taken = False
        planner.existing_guidance = False
        planner.disabled = False

        # Create result object
        planner.result = type('Result', (), {})
        planner.result.Reset = lambda: None  # Placeholder for Reset method
        planner.result.exit_code = 0
        planner.result.success = False
        planner.result.objective = float('inf')
        planner.result.guidance_ID = 0
        planner.result.color = 0

        # Create constraint modules
        planner.guidance_constraints = type('GuidanceConstraints', (), {
            'update': lambda state, data, module_data: None,
            'set_parameters': lambda data, module_data, k: None,
            'visualize': lambda data, module_data: None,
            'is_data_ready': lambda data, missing_data: True,
        })

        planner.safety_constraints = type('SafetyConstraints', (), {
            'update': lambda state, data, module_data: None,
            'set_parameters': lambda data, module_data, k: None,
            'visualize': lambda data, module_data: None,
            'is_data_ready': lambda data, missing_data: True,
            'on_data_received': lambda data, data_name: None,
        })

        return planner

    def update(self, state, data, module_data):
        LOG_DEBUG("Guidance Constraints.update")

        if module_data.path is None:
            LOG_DEBUG("Path data not yet available")
            return

        # Convert static obstacles
        if hasattr(module_data, 'static_obstacles') and module_data.static_obstacles:
            halfspaces = []
            for i in range(len(module_data.static_obstacles[0])):
                halfspaces.append((module_data.static_obstacles[0][i].A, module_data.static_obstacles[0][i].b))
            self.global_guidance.load_static_obstacles(halfspaces)  # Load static obstacles represented by halfspaces

        if self._use_tmpc and self.global_guidance.get_config().n_paths == 0:  # No global guidance
            return

        # Set the goals of the global guidance planner
        self.global_guidance.set_start(state.get_pos(), state.get("psi"), state.get("v"))

        if module_data.path_velocity is not None:
            self.global_guidance.set_reference_velocity(module_data.path_velocity(state.get("spline")))
        else:
            self.global_guidance.set_reference_velocity(CONFIG["weights"]["reference_velocity"])

        if not CONFIG["enable_output"]:
            LOG_DEBUG("Not propagating nodes (output is disabled)")
            self.global_guidance.do_not_propagate_nodes()

        # Set the goals for the guidance planner
        self.set_goals(state, module_data)

        LOG_DEBUG("Running Guidance Search")
        self.global_guidance.update()  # The main update

        self.map_guidance_trajectories_to_planners()

        # Create empty data if needed
        if self.empty_data_ is None:
            self.empty_data_ = data.__class__()  # Create new instance of the same class
            if hasattr(self.empty_data_, 'dynamic_obstacles'):
                self.empty_data_.dynamic_obstacles = []

    def set_goals(self, state, module_data):
        LOG_DEBUG("Setting guidance planner goals")

        current_s = state.get("spline")
        n_discs = CONFIG["n_discs"]

        if (module_data.path_velocity is None or
                module_data.path_width_left is None or
                module_data.path_width_right is None):
            self.global_guidance.load_reference_path(
                max(0., state.get("spline")),
                module_data.path,
                CONFIG["road"]["width"] / 2. - n_discs - 0.1,
                CONFIG["road"]["width"] / 2. - n_discs - 0.1)
            return

        # Define goals along the reference path, taking into account the velocity along the path
        final_s = current_s
        for k in range(self.global_guidance.get_config().N):  # Euler integrate the velocity along the path
            final_s += module_data.path_velocity(final_s) * self.solver.dt

        n_long = self.global_guidance.get_config().longitudinal_goals
        n_lat = self.global_guidance.get_config().vertical_goals

        assert (n_lat % 2) == 1, "Number of lateral grid points should be odd!"
        assert n_long >= 2, "There should be at least two longitudinal goals (start, end)"

        middle_lat = (n_lat - 1) // 2
        s_long = np.linspace(current_s, final_s, n_long)

        assert s_long[1] - s_long[
            0] > 0.05, "Goals should have some spacing between them (Config::reference_velocity_ should not be zero)"

        long_best = s_long[-1]  # Using Python's array indexing for last element

        goals = []
        for i in range(n_long):
            s = s_long[i]  # Distance along the path for these goals

            # Compute its cost (distance to the desired goal)
            long_cost = abs(s - long_best)

            # Compute the normal vector to the reference path
            line_point = module_data.path.get_point(s)
            normal = module_data.path.get_orthogonal(s)
            angle = module_data.path.get_path_angle(s)

            # Place goals orthogonally to the path
            dist_lat = np.linspace(
                -module_data.path_width_left(s) + n_discs,
                module_data.path_width_right(s) - n_discs,
                n_lat)
            # Put the middle goal on the reference path
            dist_lat[middle_lat] = 0.0

            for j in range(n_lat):
                if i == 0 and j != middle_lat:
                    continue  # Only the first goal should be in the center

                d = dist_lat[j]

                lat_cost = abs(d)  # Higher cost, the further away from the center line
                result = []
                res = line_point + normal * d

                # Create space-time point
                result.append(res[0])  # x
                result.append(res[1])  # y

                if self.global_guidance.space_time_point_num_states() == 3:
                    result.append(angle)  # Add angle if needed

                goals.append((result, long_cost + lat_cost))  # Add the goal

        self.global_guidance.set_goals(goals)

    def map_guidance_trajectories_to_planners(self):
        # Map each of the found guidance trajectories to an optimization ID
        # Maintaining the same homotopy class so that its initialization is valid

        remaining_trajectories = []
        for p in range(len(self.planners)):
            self.planners[p].taken = False
            self.planners[p].existing_guidance = False
        self._map_homotopy_class_to_planner.clear()

        for i in range(self.global_guidance.number_of_guidance_trajectories()):
            homotopy_class = self.global_guidance.get_guidance_trajectory(i).topology_class

            # Does it match any of the planners?
            planner_found = False
            for p in range(len(self.planners)):
                # More than one guidance trajectory may map to the same planner
                if (self.planners[p].result.guidance_ID == homotopy_class and
                        not self.planners[p].taken):
                    self._map_homotopy_class_to_planner[i] = p
                    self.planners[p].taken = True
                    self.planners[p].existing_guidance = True
                    planner_found = True
                    break

            if not planner_found:
                remaining_trajectories.append(i)

        # Assign the remaining trajectories to the remaining planners
        for i in remaining_trajectories:
            for p in range(len(self.planners)):
                if not self.planners[p].taken:
                    self._map_homotopy_class_to_planner[i] = p
                    self.planners[p].taken = True
                    self.planners[p].existing_guidance = False
                    break

    def set_parameters(self, data, module_data, k):
        if k == 0:
            self.solver.params.solver_timeout = 0.02  # Should not do anything
            LOG_DEBUG("Guidance Constraints does not need to set parameters")

    def optimize(self, state, data, module_data):
        # Set up for parallel processing
        LOG_DEBUG("Guidance Constraints.optimize")

        if not self._use_tmpc and not self.global_guidance.succeeded():
            return 0

        shift_forward = CONFIG["shift_previous_solution_forward"] and CONFIG["enable_output"]

        for planner in self.planners:
            planner.result.Reset()
            planner.disabled = False

            if planner.id >= self.global_guidance.number_of_guidance_trajectories():
                # Only enable the solvers that are needed
                if not planner.is_original_planner:  # We still want to add the original planner!
                    planner.disabled = True
                    continue

            # Copy the data from the main solver
            solver = planner.local_solver
            LOG_DEBUG(f"Planner [{planner.id}]: Copying data from main solver")
            # Copy solver attributes if necessary

            # CONSTRUCT CONSTRAINTS
            if planner.is_original_planner and not self._enable_constraints:
                planner.guidance_constraints.update(state, self.empty_data_, module_data)
                planner.safety_constraints.update(state, data, module_data)  # updates collision avoidance constraints
            else:
                LOG_DEBUG(f"Planner [{planner.id}]: Loading guidance into the solver and constructing constraints")

                if CONFIG["t-mpc"]["warmstart_with_mpc_solution"] and planner.existing_guidance:
                    planner.local_solver.initialize_warmstart(state, shift_forward)
                else:
                    self.initialize_solver_with_guidance(planner)

                planner.guidance_constraints.update(state, data, module_data)  # updates linearization of constraints
                planner.safety_constraints.update(state, data, module_data)  # updates collision avoidance constraints

            # LOAD PARAMETERS
            LOG_DEBUG(f"Planner [{planner.id}]: Loading updated parameters into the solver")
            for k in range(self.solver.N):
                if planner.is_original_planner:
                    planner.guidance_constraints.set_parameters(self.empty_data_, module_data, k)
                else:
                    planner.guidance_constraints.set_parameters(data, module_data, k)

                planner.safety_constraints.set_parameters(data, module_data, k)

            # Set timeout based on remaining planning time
            import time
            used_time = time.time() - data.planning_start_time if hasattr(data, 'planning_start_time') else 0
            planner.local_solver.params.solver_timeout = self._planning_time - used_time - 0.006

            # SOLVE OPTIMIZATION
            planner.local_solver.load_warm_start()
            LOG_DEBUG(f"Planner [{planner.id}]: Solving ...")
            planner.result.exit_code = solver.solve()
            LOG_DEBUG(f"Planner [{planner.id}]: Done! (exitcode = {planner.result.exit_code})")

            # ANALYSIS AND PROCESSING
            planner.result.success = planner.result.exit_code == 1
            planner.result.objective = solver._info.pobj if hasattr(solver, '_info') else float('inf')

            if planner.is_original_planner:  # We did not use any guidance!
                planner.result.guidance_ID = 2 * self.global_guidance.get_config().n_paths  # one higher than the maximum number of topology classes
                planner.result.color = -1
            else:
                guidance_trajectory = self.global_guidance.get_guidance_trajectory(planner.id)
                planner.result.guidance_ID = guidance_trajectory.topology_class  # We were using this guidance
                planner.result.color = guidance_trajectory.color  # A color index to visualize with

                if guidance_trajectory.previously_selected:  # Prefer the selected trajectory
                    planner.result.objective *= self.global_guidance.get_config().selection_weight_consistency

        # DECISION MAKING
        self.best_planner_index_ = self.find_best_planner()
        if self.best_planner_index_ == -1:
            LOG_DEBUG(f"Failed to find a feasible trajectory in any of the {len(self.planners)} optimizations.")
            return self.planners[0].result.exit_code

        best_planner = self.planners[self.best_planner_index_]
        best_solver = best_planner.local_solver

        # Communicate to the guidance which topology class we follow (none if it was the original planner)
        self.global_guidance.override_selected_trajectory(
            best_planner.result.guidance_ID,
            best_planner.is_original_planner)

        # Load the solution into the main lmpcc solver
        self.solver.output = best_solver.output
        if hasattr(best_solver, '_info'):
            self.solver._info = best_solver._info
        if hasattr(best_solver, 'params'):
            self.solver.params = best_solver.params

        return best_planner.result.exit_code

    def initialize_solver_with_guidance(self, planner):
        solver = planner.local_solver

        # Initialize the solver with the guidance trajectory
        trajectoryspline = self.global_guidance.get_guidance_trajectory(planner.id).spline.get_trajectory()

        # Initialize the solver in the selected local optimum
        for k in range(solver.N):
            index = k
            cur_position = trajectoryspline.get_point(index * solver.dt)
            solver.set_ego_prediction(k, "x", cur_position[0])
            solver.set_ego_prediction(k, "y", cur_position[1])

            cur_velocity = trajectoryspline.get_velocity(index * solver.dt)
            solver.set_ego_prediction(k, "psi", atan2(cur_velocity[1], cur_velocity[0]))
            solver.set_ego_prediction(k, "v", np.linalg.norm(cur_velocity))

    def find_best_planner(self):
        # Find the best feasible solution
        best_solution = float('inf')
        best_index = -1

        for i in range(len(self.planners)):
            planner = self.planners[i]
            if planner.disabled:  # Do not consider disabled planners
                continue

            if planner.result.success and planner.result.objective < best_solution:
                best_solution = planner.result.objective
                best_index = i

        return best_index

    def visualize(self, data, module_data):
        LOG_DEBUG("Guidance Constraints: Visualize()")

        # Visualize global guidance if available
        if not (self._use_tmpc and self.global_guidance.get_config().n_paths == 0):  # If global guidance
            self.global_guidance.visualize(CONFIG["t-mpc"]["highlight_selected"], -1)

        for i in range(len(self.planners)):
            planner = self.planners[i]
            if planner.disabled:
                continue

            if i == 0:
                planner.guidance_constraints.visualize(data, module_data)
                planner.safety_constraints.visualize(data, module_data)

            # Visualize the warmstart
            if CONFIG["debug_visuals"]:
                initial_trajectory = []
                for k in range(planner.local_solver.N):
                    initial_trajectory.append((
                        planner.local_solver.get_ego_prediction(k, "x"),
                        planner.local_solver.get_ego_prediction(k, "y")
                    ))
                visualize_trajectory(initial_trajectory, self.name + "/warmstart_trajectories", False, 0.2, 20, 20)

            # Visualize the optimized trajectory
            if planner.result.success:
                trajectory = []
                for k in range(self.solver.N):
                    trajectory.append((
                        planner.local_solver.get_output(k, "x"),
                        planner.local_solver.get_output(k, "y")
                    ))

                if i == self.best_planner_index_:
                    visualize_trajectory(trajectory, self.name + "/optimized_trajectories", False, 1.0, -1, 12, True,
                                         False)
                elif planner.is_original_planner:
                    visualize_trajectory(trajectory, self.name + "/optimized_trajectories", False, 1.0, 11, 12, True,
                                         False)
                else:
                    visualize_trajectory(trajectory, self.name + "/optimized_trajectories", False, 1.0,
                                         planner.result.color, self.global_guidance.get_config().n_paths, True, False)

        # Publish visualization data
        if hasattr(self, 'VISUALS') and hasattr(self.VISUALS, 'missing_data'):
            self.VISUALS.missing_data(self.name + "/optimized_trajectories").publish()
            if CONFIG["debug_visuals"]:
                self.VISUALS.missing_data(self.name + "/warmstart_trajectories").publish()

    def is_data_ready(self, data, missing_data):
        ready = True

        if len(self.planners) > 0:
            ready = ready and self.planners[0].guidance_constraints.is_data_ready(data, missing_data)
            ready = ready and self.planners[0].safety_constraints.is_data_ready(data, missing_data)

        if not ready:
            return False

        return ready

    def on_data_received(self, data, data_name):
        if data_name == "goal":  # New
            LOG_DEBUG("Goal input is not yet implemented for T-MPC")
            # Implementation details to be added

        # We wait for both the obstacles and state to arrive before we compute here
        if data_name == "dynamic obstacles":
            LOG_DEBUG("Guidance Constraints: Received dynamic obstacles")

            for planner in self.planners:
                planner.safety_constraints.on_data_received(data, data_name)

            obstacles = []
            for obstacle in data.dynamic_obstacles:
                positions = []
                positions.append(obstacle.position)  # Current position

                for k in range(len(obstacle.prediction.modes[0])):
                    positions.append(obstacle.prediction.modes[0][k].position)

                obstacles.append((
                    obstacle.index,
                    positions,
                    obstacle.radius + data.robot_area[0].radius
                ))

            self.global_guidance.load_obstacles(obstacles, {})

    def reset(self):
        self.global_guidance.reset()

        for planner in self.planners:
            planner.local_solver.reset()

    def save_data(self, data_saver):
        data_saver.add_data("runtime_guidance", self.global_guidance.get_last_runtime())

        for i, planner in enumerate(self.planners):
            objective = planner.result.objective if planner.result.success else -1
            data_saver.add_data(f"objective_{i}", objective)

            if planner.is_original_planner:
                data_saver.add_data("lmpcc_objective", objective)
                data_saver.add_data("original_planner_id", planner.id)  # To identify which one is the original planner

        data_saver.add_data("best_planner_idx", self.best_planner_index_)
        if self.best_planner_index_ != -1:
            best_objective = self.planners[self.best_planner_index_].local_solver._info.pobj
        else:
            best_objective = -1

        data_saver.add_data("gmpcc_objective", best_objective)
        self.global_guidance.save_data(data_saver)  # Save data from the guidance planner


# This is a placeholder for the GlobalGuidance class which would need to be implemented
# based on your specific requirements
class GlobalGuidance:
    def __init__(self):
        # Configuration and other parameters would be initialized here
        pass

    def set_planning_frequency(self, freq):
        pass

    def get_config(self):
        # Return a configuration object with attributes like n_paths, N, etc.
        config = type('Config', (), {})
        config.n_paths = CONFIG.get("global_guidance", {}).get("n_paths", 1)
        config.N = CONFIG.get("global_guidance", {}).get("N", 10)
        config.longitudinal_goals = CONFIG.get("global_guidance", {}).get("longitudinal_goals", 5)
        config.vertical_goals = CONFIG.get("global_guidance", {}).get("vertical_goals", 5)
        config.selection_weight_consistency = CONFIG.get("global_guidance", {}).get("selection_weight_consistency", 0.8)
        return config

    def load_static_obstacles(self, halfspaces):
        pass

    def set_start(self, pos, psi, v):
        pass

    def set_reference_velocity(self, velocity):
        pass

    def do_not_propagate_nodes(self):
        pass

    def update(self):
        pass

    def succeeded(self):
        return True

    def number_of_guidance_trajectories(self):
        return 1  # Placeholder

    def get_guidance_trajectory(self, idx):
        # Return a trajectory object with attributes
        trajectory = type('Trajectory', (), {})
        trajectory.topology_class = idx
        trajectory.color = idx
        trajectory.previously_selected = False

        # Create spline attribute
        trajectory.spline = type('Spline', (), {})
        trajectory.spline.get_trajectory = lambda: type('TrajectorySpline', (), {
            'get_point': lambda t: np.array([0, 0]),
            'get_velocity': lambda t: np.array([1, 0])
        })

        return trajectory

    def space_time_point_num_states(self):
        return 2  # Default to x, y

    def set_goals(self, goals):
        pass

    def load_reference_path(self, start_s, path, width_left, width_right):
        pass

    def load_obstacles(self, obstacles, extra_data):
        pass

    def override_selected_trajectory(self, guidance_id, is_original_planner):
        pass

    def visualize(self, highlight_selected, trajectory_nr):
        pass

    def reset(self):
        pass

    def get_last_runtime(self):
        return 0.0

    def save_data(self, data_saver):
        pass