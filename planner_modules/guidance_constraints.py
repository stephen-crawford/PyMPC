from functools import partial
import numpy as np
from math import exp, atan2

from utils.const import CONSTRAINT
from utils.utils import LOG_DEBUG, PYMPC_ASSERT, CONFIG
from utils.visualizer import ROSLine
from planner_modules.base_constraint import BaseConstraint
from functools import partial
import numpy as np
from math import exp, atan2, sqrt
import time

from utils.const import CONSTRAINT
from utils.utils import LOG_DEBUG, PYMPC_ASSERT, CONFIG
from utils.visualizer import ROSLine, ROSMarker


class GlobalGuidance:
    def __init__(self):
        """Initialize the global guidance planner with default parameters"""
        self._planning_frequency = 10.0  # Hz
        self._dt = 1.0 / self._planning_frequency

        # State and configuration parameters
        self._current_pos = np.zeros(2)
        self._current_psi = 0.0
        self._current_v = 0.0
        self._reference_velocity = 0.0

        # Path and obstacle data
        self._reference_path = None
        self._start_s = 0.0
        self._width_left = 0.0
        self._width_right = 0.0
        self._static_obstacles = []
        self._dynamic_obstacles = []

        # Goal related parameters
        self._goals = []
        self._trajectories = []
        self._selected_trajectory_id = -1

        # Control flags
        self._propagate_nodes = True
        self._success = False
        self._runtime = 0.0
        self._original_planner_selected = False

        # Create default config (will be overridden by get_config)
        self._config = type('Config', (), {})
        self._config.n_paths = CONFIG.get("global_guidance", {}).get("n_paths", 1)
        self._config.N = CONFIG.get("global_guidance", {}).get("N", 10)
        self._config.longitudinal_goals = CONFIG.get("global_guidance", {}).get("longitudinal_goals", 5)
        self._config.vertical_goals = CONFIG.get("global_guidance", {}).get("vertical_goals", 5)
        self._config.selection_weight_consistency = CONFIG.get("global_guidance", {}).get(
            "selection_weight_consistency", 0.8)

        LOG_DEBUG("GlobalGuidance initialized")

    def set_planning_frequency(self, freq):
        """Set the planning frequency in Hz"""
        self._planning_frequency = freq
        self._dt = 1.0 / freq
        LOG_DEBUG(f"Planning frequency set to {freq} Hz (dt={self._dt:.4f})")

    def get_config(self):
        """Return configuration object with attributes"""
        return self._config

    def load_static_obstacles(self, halfspaces):
        """Load static obstacles represented as halfspaces (Ax <= b)"""
        self._static_obstacles = halfspaces
        LOG_DEBUG(f"Loaded {len(halfspaces)} static obstacles")

    def set_start(self, pos, psi, v):
        """Set the starting position, orientation and velocity"""
        self._current_pos = np.array(pos)
        self._current_psi = psi
        self._current_v = v
        LOG_DEBUG(f"Start set to pos={pos}, psi={psi:.2f}, v={v:.2f}")

    def set_reference_velocity(self, velocity):
        """Set the reference velocity for planning"""
        self._reference_velocity = velocity
        LOG_DEBUG(f"Reference velocity set to {velocity:.2f}")

    def do_not_propagate_nodes(self):
        """Disable node propagation for planning"""
        self._propagate_nodes = False
        LOG_DEBUG("Node propagation disabled")

    def update(self):
        """Main update function to generate trajectory options"""
        start_time = time.time()

        # Reset trajectories
        self._trajectories = []

        if len(self._goals) == 0:
            LOG_DEBUG("No goals set, cannot update guidance")
            self._success = False
            self._runtime = time.time() - start_time
            return

        # Plan trajectories to reach goals
        self._generate_trajectories()

        # Mark planning as successful if we have trajectories
        self._success = len(self._trajectories) > 0

        # Record runtime
        self._runtime = time.time() - start_time
        LOG_DEBUG(
            f"GlobalGuidance update completed in {self._runtime:.4f}s, generated {len(self._trajectories)} trajectories")

    def _generate_trajectories(self):
        """Generate trajectories to reach goals using simple planning methods"""
        # For each goal, create a different trajectory option
        # Limited to n_paths trajectories maximum
        goals_to_consider = sorted(self._goals, key=lambda g: g[1])[:self._config.n_paths]

        for i, (goal_point, cost) in enumerate(goals_to_consider):
            # Create a simple trajectory from current position to goal
            trajectory = self._create_trajectory_to_goal(goal_point, i)
            self._trajectories.append(trajectory)

            LOG_DEBUG(f"Created trajectory {i} with cost {cost:.2f} to goal {goal_point}")

    def _create_trajectory_to_goal(self, goal_point, traj_id):
        """Create a simple trajectory to reach a goal point"""
        # Create a trajectory class with needed attributes
        trajectory = type('Trajectory', (), {})
        trajectory.topology_class = traj_id  # Unique ID for this trajectory class
        trajectory.color = traj_id  # Color index for visualization
        trajectory.previously_selected = (traj_id == self._selected_trajectory_id)

        # Create spline for the trajectory
        trajectory.spline = type('Spline', (), {})

        # Define point and velocity getters as lambdas
        # This is a simple linear trajectory from start to goal
        start_pos = self._current_pos
        goal_pos = np.array(goal_point[:2])  # Only take x,y from goal
        direction = goal_pos - start_pos
        distance = np.linalg.norm(direction)

        if distance > 0:
            direction = direction / distance
        else:
            direction = np.array([np.cos(self._current_psi), np.sin(self._current_psi)])

        # Create the trajectory function
        def get_trajectory():
            traj = type('TrajectorySpline', (), {})

            def get_point(t):
                # Simple linear interpolation
                t_total = self._config.N * self._dt
                t = min(t, t_total)
                fraction = min(t / t_total, 1.0)
                return start_pos + direction * distance * fraction

            def get_velocity(t):
                # Constant velocity along the path
                t_total = self._config.N * self._dt
                if t >= t_total:
                    return np.zeros(2)
                return direction * self._reference_velocity

            traj.get_point = get_point
            traj.get_velocity = get_velocity
            return traj

        trajectory.spline.get_trajectory = get_trajectory

        return trajectory

    def succeeded(self):
        """Return whether the guidance planning succeeded"""
        return self._success

    def number_of_guidance_trajectories(self):
        """Return the number of available guidance trajectories"""
        return len(self._trajectories)

    def get_guidance_trajectory(self, idx):
        """Get a specific guidance trajectory by index"""
        if idx < 0 or idx >= len(self._trajectories):
            LOG_DEBUG(f"Trajectory index {idx} out of range (0-{len(self._trajectories) - 1})")
            # Return a default trajectory
            trajectory = type('Trajectory', (), {})
            trajectory.topology_class = -1
            trajectory.color = -1
            trajectory.previously_selected = False

            trajectory.spline = type('Spline', (), {})
            trajectory.spline.get_trajectory = lambda: type('TrajectorySpline', (), {
                'get_point': lambda t: self._current_pos,
                'get_velocity': lambda t: np.array([self._current_v * np.cos(self._current_psi),
                                                    self._current_v * np.sin(self._current_psi)])
            })

            return trajectory

        return self._trajectories[idx]

    def space_time_point_num_states(self):
        """Return the number of states for space-time points (x,y,[psi])"""
        # If we're using orientation in goals, return 3, otherwise 2
        return 3 if len(self._goals) > 0 and len(self._goals[0][0]) > 2 else 2

    def set_goals(self, goals):
        """Set the guidance goals (points and costs)"""
        self._goals = goals
        LOG_DEBUG(f"Set {len(goals)} guidance goals")

    def load_reference_path(self, start_s, path, width_left, width_right):
        """Load a reference path with lateral boundaries"""
        self._reference_path = path
        self._start_s = start_s
        self._width_left = width_left
        self._width_right = width_right
        LOG_DEBUG(f"Loaded reference path starting at s={start_s:.2f}, "
                  f"width_left={width_left:.2f}, width_right={width_right:.2f}")

    def load_obstacles(self, obstacles, extra_data):
        """Load dynamic obstacles for planning"""
        self._dynamic_obstacles = obstacles
        LOG_DEBUG(f"Loaded {len(obstacles)} dynamic obstacles")

    def override_selected_trajectory(self, guidance_id, is_original_planner):
        """Override which trajectory is selected as the best one"""
        self._selected_trajectory_id = guidance_id
        self._original_planner_selected = is_original_planner
        LOG_DEBUG(f"Selected trajectory ID {guidance_id}, original planner: {is_original_planner}")

    def visualize(self, highlight_selected, trajectory_nr):
        """Visualize all guidance trajectories"""
        # Visualize goals as points
        for i, (goal_point, _) in enumerate(self._goals):
            # Visualization code would go here - using placeholder
            pass

        # Visualize all trajectories
        for i, traj in enumerate(self._trajectories):
            trajectory_spline = traj.spline.get_trajectory()

            # Create points for visualization
            trajectory_points = []
            for k in range(self._config.N):
                t = k * self._dt
                pos = trajectory_spline.get_point(t)
                trajectory_points.append((pos[0], pos[1]))

            # Highlight selected trajectory if requested
            is_selected = (traj.topology_class == self._selected_trajectory_id) and highlight_selected
            color = traj.color

            # Call visualization function (placeholder)
            visualize_trajectory(trajectory_points, "global_guidance/trajectories",
                                 is_selected, 0.5, color, self._config.n_paths)

    def reset(self):
        """Reset the guidance planner"""
        self._trajectories = []
        self._selected_trajectory_id = -1
        self._success = False
        self._goals = []
        LOG_DEBUG("GlobalGuidance reset")

    def get_last_runtime(self):
        """Get the last planning runtime in seconds"""
        return self._runtime

    def save_data(self, data_saver):
        """Save planning data for analysis"""
        data_saver.add_data("global_guidance/runtime", self._runtime)
        data_saver.add_data("global_guidance/num_trajectories", len(self._trajectories))
        data_saver.add_data("global_guidance/selected_trajectory", self._selected_trajectory_id)
        data_saver.add_data("global_guidance/original_planner_selected", self._original_planner_selected)


def visualize_trajectory(trajectory_points, publisher_name, highlight=False,
                         scale=0.1, color_int=0, max_color=20, add_markers=False,
                         connect_points=True):
    """
    Visualize a trajectory as a series of points and lines

    Args:
        trajectory_points: List of (x,y) points in the trajectory
        publisher_name: Name for the visualization publisher
        highlight: Whether to highlight this trajectory
        scale: Scale factor for visualization
        color_int: Integer color index
        max_color: Maximum number of colors in the palette
        add_markers: Whether to add point markers
        connect_points: Whether to connect points with lines
    """
    if not trajectory_points:
        return

    # Calculate a color from the color integer
    r, g, b = 0.0, 0.0, 0.0

    if color_int < 0:  # Special colors
        if color_int == -1:  # Selected trajectory
            r, g, b = 0.0, 1.0, 0.0  # Green
        else:
            r, g, b = 1.0, 1.0, 1.0  # White
    else:
        # Generate a color from the color_int using hue rotation
        import colorsys
        h = (color_int % max_color) / float(max_color)
        r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.8)

    # Adjust alpha and width based on highlight
    alpha = 1.0 if highlight else 0.7
    width = 0.03 if highlight else 0.01

    # Create line strips
    if connect_points and len(trajectory_points) > 1:
        line = ROSLine()
        line.header.frame_id = "world"
        line.id = 0
        line.ns = publisher_name
        line.type = line.LINE_STRIP
        line.action = line.ADD
        line.scale.x = width * scale
        line.color.r = r
        line.color.g = g
        line.color.b = b
        line.color.a = alpha

        # Add points to the line
        for point in trajectory_points:
            p = type('Point', (), {})
            p.x = point[0]
            p.y = point[1]
            p.z = 0.1  # Slightly above ground
            line.points.append(p)

        # Publish line (this is a placeholder - actual publishing would depend on your system)
        # visualization_publisher.publish(line)

    # Add markers if requested
    if add_markers:
        for i, point in enumerate(trajectory_points):
            marker = ROSMarker()
            marker.header.frame_id = "world"
            marker.id = i
            marker.ns = publisher_name + "/points"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.1 * scale
            marker.scale.y = 0.1 * scale
            marker.scale.z = 0.1 * scale
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = alpha
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = 0.1  # Slightly above ground

            # Publish marker (placeholder)
            # visualization_publisher.publish(marker)

class GuidanceConstraints(BaseConstraint):
    def __init__(self, solver):
        super().__init__(solver)
        LOG_DEBUG("Initializing Guidance Constraints")
        self.name = "guidance_constraints"

        self.global_guidance = GlobalGuidance()
        self.global_guidance.set_planning_frequency(self.get_config_value("control_frequency"))

        self._use_tmpc = self.get_config_value("t-mpc.use_t-mpc+=1", False)
        self._enable_constraints = self.get_config_value("t-mpc.enable_constraints", True)
        self._control_frequency = self.get_config_value("control_frequency")
        self._planning_time = 1. / self._control_frequency

        # Initialize the constraint modules
        self.nsolvers = self.global_guidance.get_config().n_paths
        print("NSolvers is " + str(self.nsolvers))
        PYMPC_ASSERT(self.nsolvers > 0 or self._use_tmpc,
                     "Guidance constraints cannot run with 0 paths and T-MPC+=1 disabled!")

        LOG_DEBUG(f"Solvers count: {self.nsolvers}")
        self.planners = []
        for i in range(self.nsolvers):
            self.planners.append(self._create_planner(i))

        if self._use_tmpc:  # ADD IT AS FIRST PLAN
            LOG_DEBUG("Using T-MPC+=1 (Adding the non-guided planner in parallel)")
            self.planners.append(self._create_planner(self.nsolvers, True))

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
            self.global_guidance.set_reference_velocity(self.get_config_value("weights.reference_velocity"))

        if not self.get_config_value("enable_output"):
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
        n_discs = self.get_config_value("n_discs")

        if (module_data.path_velocity is None or
                module_data.path_width_left is None or
                module_data.path_width_right is None):
            self.global_guidance.load_reference_path(
                max(0., state.get("spline")),
                module_data.path,
                self.get_config_value("road.width") / 2. - n_discs - 0.1,
                self.get_config_value("road.width") / 2. - n_discs - 0.1)
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

        shift_forward = self.get_config_value("shift_previous_solution_forward", True) and self.get_config_value("enable_output")

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

                if self.get_config_value("t-mpc.warmstart_with_mpc_solution", False) and planner.existing_guidance:
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
        trajectory_spline = self.global_guidance.get_guidance_trajectory(planner.id).spline.get_trajectory()

        # Initialize the solver in the selected local optimum
        for k in range(solver.N):
            index = k
            cur_position = trajectory_spline.get_point(index * solver.dt)
            solver.set_ego_prediction(k, "x", cur_position[0])
            solver.set_ego_prediction(k, "y", cur_position[1])

            cur_velocity = trajectory_spline.get_velocity(index * solver.dt)
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
        super().visualize(data, module_data)
        LOG_DEBUG("Guidance Constraints: Visualize()")

        # Visualize global guidance if available
        if not (self._use_tmpc and self.global_guidance.get_config().n_paths == 0):  # If global guidance
            self.global_guidance.visualize(self.get_config_value("t-mpc.highlight_selected", True), -1)

        for i in range(len(self.planners)):
            planner = self.planners[i]
            if planner.disabled:
                continue

            if i == 0:
                planner.guidance_constraints.visualize(data, module_data)
                planner.safety_constraints.visualize(data, module_data)

            # Visualize the warmstart
            if self.get_config_value("debug_visuals", False):
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
            if self.get_config_value("debug_visuals", False):
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
        super().reset()
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

    def visualize_trajectory(self, trajectory, name_suffix="trajectory", scale=0.1, color_int=0):
        """Override the base class method to use our custom visualize_trajectory function"""
        publisher_name = f"{self.name}/{name_suffix}"
        visualize_trajectory(trajectory, publisher_name, False, scale, color_int, 0)