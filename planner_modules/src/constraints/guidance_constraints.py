import numpy as np

from planner.src.planner import Planner
from planner_modules.src.constraints.base_constraint import BaseConstraint
from utils.const import CONSTRAINT
from utils.utils import LOG_DEBUG


class GuidanceConstraints(BaseConstraint):
    def __init__(self, solver):
        super().__init__(solver)
        LOG_DEBUG("Initializing Guidance Constraints")
        self.name = "guidance_constraints"
        self.module_type = CONSTRAINT

        # Core configuration parameters
        self.planning_frequency = self.get_config_value("control_frequency")
        self.control_frequency = self.get_config_value("control_frequency")
        self.planning_time = 1. / self.control_frequency
        self.max_obstacles = self.get_config_value("max_obstacles")
        self.num_other_halfspaces = self.get_config_value("guidance.num_other_halfspaces")
        self.nh = self.max_obstacles + self.num_other_halfspaces

        # Guidance-specific parameters
        self.num_paths = self.get_config_value("num_paths", 1)
        self.longitudinal_goals = self.get_config_value("guidance.longitudinal_goals", 5)
        self.vertical_goals = self.get_config_value("guidance.vertical_goals", 5)
        self.use_tmpc = self.get_config_value("use_tmpc", False)

        # Initialize constraint dimensions
        self.num_constraints = 2  # Position and velocity constraints

        # Create planners for trajectory generation
        self.planners = []
        for path in range(self.num_paths):
            self.planners.append(self._create_guidance_planner())
        if self.use_tmpc:
            self.planners.append(self._create_guidance_planner())
        self.best_planner_index = -1

        # Path and trajectory data
        self.reference_path = None
        self.width_left = None
        self.width_right = None
        self.trajectories = []
        self.selected_trajectory = None

        LOG_DEBUG("Guidance Constraints successfully initialized")

    def _create_guidance_planner(self):
        """Create a guidance planner instance"""
        planner = GuidancePlanner(self.solver)
        planner.longitudinal_goals = self.longitudinal_goals
        planner.vertical_goals = self.vertical_goals
        return planner

    def update(self, state, data, module_data):
        LOG_DEBUG("Guidance Constraints.update")

        if not hasattr(module_data, 'path') or module_data.path is None:
            LOG_DEBUG("Path data not yet available")
            return

        # Process static obstacles if available
        if hasattr(module_data, 'static_obstacles') and module_data.static_obstacles:
            halfspaces = []
            for i in range(len(module_data.static_obstacles[0])):
                halfspaces.append((module_data.static_obstacles[0][i].A, module_data.static_obstacles[0][i].b))

            for planner in self.planners:
                planner.load_static_obstacles(halfspaces)

        # Set up planners with current state

        for planner in self.planners:
            planner.set_start(state.get_pos(), state.get("psi"), state.get("v"))

            # Set reference velocity
            if hasattr(module_data, 'path_velocity') and module_data.path_velocity is not None:
                planner.set_reference_velocity(module_data.path_velocity(state.get("spline")))
            else:
                planner.set_reference_velocity(self.get_config_value("weights.reference_velocity"))

            # Set goals for the guidance planner
            self.set_goals(state, module_data, planner)

        # Set the width parameters if available
        if hasattr(module_data, 'path_width_left') and module_data.path_width_left is not None and \
                hasattr(module_data, 'path_width_right') and module_data.path_width_right is not None:
            self.width_left = module_data.path_width_left
            self.width_right = module_data.path_width_right

        # Run the planners
        LOG_DEBUG("Running Guidance Search")
        for planner in self.planners:
            planner.update()

        # Store trajectories
        self.trajectories = []
        for planner in self.planners:
            for trajectory in planner.trajectories:
                self.trajectories.append(trajectory)

        # Find the best trajectory
        self.optimize(state, data, module_data)

    def set_goals(self, state, module_data, planner):
        LOG_DEBUG("Setting guidance planner goals")

        current_s = state.get("spline")
        num_discs = self.get_config_value("num_discs", 0.1)

        # Handle case when path data is incomplete
        if not hasattr(module_data, 'path_velocity') or module_data.path_velocity is None or \
                not hasattr(module_data, 'path_width_left') or module_data.path_width_left is None or \
                not hasattr(module_data, 'path_width_right') or module_data.path_width_right is None:
            planner.load_reference_path(
                max(0., state.get("spline")),
                module_data.path,
                self.get_config_value("road.width") / 2. - num_discs - 0.1,
                self.get_config_value("road.width") / 2. - num_discs - 0.1)
            return

        # Calculate final position based on velocity
        final_s = current_s
        for k in range(planner.horizon):
            final_s += module_data.path_velocity(final_s) * self.solver.dt

        n_long = self.longitudinal_goals
        n_lat = self.vertical_goals

        assert (n_lat % 2) == 1, "Number of lateral grid points should be odd!"
        assert n_long >= 2, "There should be at least two longitudinal goals (start, end)"

        middle_lat = (n_lat - 1) // 2
        s_long = np.linspace(current_s, final_s, n_long)

        long_best = s_long[-1]

        goals = []
        for i in range(n_long):
            s = s_long[i]

            # Compute cost (distance to desired goal)
            long_cost = abs(s - long_best)

            # Get path information at this point
            line_point = module_data.path.get_point(s)
            normal = module_data.path.get_orthogonal(s)
            angle = module_data.path.get_path_angle(s)

            # Place goals orthogonally to the path
            dist_lat = np.linspace(
                -module_data.path_width_left(s) + num_discs,
                module_data.path_width_right(s) - num_discs,
                n_lat)

            # Put the middle goal on the reference path
            dist_lat[middle_lat] = 0.0

            for j in range(n_lat):
                if i == 0 and j != middle_lat:
                    continue  # Only the first goal should be in the center

                d = dist_lat[j]
                lat_cost = abs(d)

                # Calculate position
                res = line_point + normal * d

                # Create space-time point
                result = [res[0], res[1]]  # x, y

                if planner.space_time_point_num_states() == 3:
                    result.append(angle)  # Add angle if needed

                goals.append((result, long_cost + lat_cost))

        planner.set_goals(goals)

    def define_parameters(self, params):
        """Define the parameters needed for guidance constraints"""
        LOG_DEBUG("Defining guidance parameters")

        # Define trajectory guidance parameters for each timestep
        for k in range(self.solver.horizon):
            params.add(f"guidance_x_{k}")
            params.add(f"guidance_y_{k}")
            params.add(f"guidance_vx_{k}")
            params.add(f"guidance_vy_{k}")

    def set_parameters(self, parameter_manager, data, module_data, k):
        """Set parameter values for the solver"""
        print("Setting guidance parameters")
        if k == 0:
            print("k is 0 ")
            LOG_DEBUG("Setting guidance parameters")

            # If we have a selected trajectory, use it to set parameters
            if self.selected_trajectory:
                trajectory_spline = self.selected_trajectory.spline.get_trajectory()

                # Set parameters based on the trajectory
                print("horizon is " + str(self.solver.horizon))
                print("trajectories len is " + str(len(self.trajectories)))
                for segment_idx in range(min(self.solver.horizon, len(self.trajectories))):
                    pos = trajectory_spline.get_point(segment_idx * self.solver.dt)
                    vel = trajectory_spline.get_velocity(segment_idx * self.solver.dt)

                    # Set position and velocity guidance parameters for this timestep
                    parameter_manager.set_parameter(f"guidance_x_{segment_idx}", pos[0])
                    parameter_manager.set_parameter(f"guidance_y_{segment_idx}", pos[1])
                    parameter_manager.set_parameter(f"guidance_vx_{segment_idx}", vel[0])
                    parameter_manager.set_parameter(f"guidance_vy_{segment_idx}", vel[1])

    def optimize(self, state, data, module_data):
        LOG_DEBUG("Guidance Constraints.optimize")

        # Check if planners have successful solutions
        all_successful = True
        for planner in self.planners:
            if not planner.succeeded():
                all_successful = False

        if not all_successful or not self.trajectories:
            LOG_DEBUG("No successful trajectories found")
            return 0

        # Find the best trajectory based on cost
        best_traj_index = -1
        best_cost = float('inf')

        for i, traj in enumerate(self.trajectories):
            # Calculate cost - distance to reference path + other factors
            cost = self.calculate_trajectory_cost(traj, state)
            if cost < best_cost:
                best_cost = cost
                best_traj_index = i

        if best_traj_index >= 0:
            self.selected_trajectory = self.trajectories[best_traj_index]
            self.best_planner_index = best_traj_index
            LOG_DEBUG(f"Selected trajectory {best_traj_index} with cost {best_cost}")
            return 1

        return 0

    def calculate_trajectory_cost(self, trajectory, state):
        """Calculate cost of a trajectory based on various factors"""
        # Get trajectory spline
        traj_spline = trajectory.spline.get_trajectory()

        # Basic cost is the stored cost
        cost = trajectory.cost if hasattr(trajectory, 'cost') else 0

        print("basic cost is " + str(cost))
        # Add distance to current position

        current_pos = np.array(state.get_pos())
        start_pos = np.array(traj_spline.get_point(0))
        cost += np.linalg.norm(current_pos - start_pos) * 0.5
        print("cost plus norm is ", cost)

        # Prefer previously selected trajectory for consistency
        if hasattr(trajectory, 'previously_selected') and trajectory.previously_selected:
            cost *= 0.8  # Give 20% discount to previously selected trajectory

        return cost

    def lower_bounds(self):
        """Return lower bounds for the constraints"""
        # Return soft guidance constraint bounds
        return [-float('inf'), -float('inf')]  # Position and velocity constraints

    def upper_bounds(self):
        """Return upper bounds for the constraints"""
        # Return soft guidance constraint bounds
        return [0.0, 0.0]  # Position and velocity constraints

    def calculate_constraints(self, params, settings, stage_idx):
        """Define the guidance constraints for the optimization model"""
        constraints = []

        # Get state variables from the model
        model = settings.model
        pos_x = model.get("x")
        pos_y = model.get("y")
        vel_x = model.get("vx") if "vx" in model.vars else model.get("v") * np.cos(model.get("psi"))
        vel_y = model.get("vy") if "vy" in model.vars else model.get("v") * np.sin(model.get("psi"))

        # Get guidance parameters
        guidance_x = params.get(f"guidance_x_{stage_idx}")
        guidance_y = params.get(f"guidance_y_{stage_idx}")
        guidance_vx = params.get(f"guidance_vx_{stage_idx}")
        guidance_vy = params.get(f"guidance_vy_{stage_idx}")

        # Get slack variable if available
        try:
            slack = model.get("slack")
        except:
            slack = 0.0

        # Calculate position error relative to guidance
        pos_error_x = pos_x - guidance_x
        pos_error_y = pos_y - guidance_y

        # Calculate velocity error relative to guidance
        vel_error_x = vel_x - guidance_vx
        vel_error_y = vel_y - guidance_vy

        # Add position constraint (weighted squared norm)
        pos_weight = self.get_config_value("weights.guidance_position", 1.0)
        pos_constraint = pos_weight * (pos_error_x ** 2 + pos_error_y ** 2) - slack

        # Add velocity constraint (weighted squared norm)
        vel_weight = self.get_config_value("weights.guidance_velocity", 0.5)
        vel_constraint = vel_weight * (vel_error_x ** 2 + vel_error_y ** 2) - slack

        constraints.append(pos_constraint)
        constraints.append(vel_constraint)

        return constraints

    def is_data_ready(self, data):
        """Check if all required data is available"""

        missing_data = ""

        # Check for required data fields
        if not hasattr(data, 'path') or data.path is None:
            missing_data += "path"

        return len(missing_data) < 1

    def on_data_received(self, data, data_name):
        """Handle incoming data"""
        if data_name == "dynamic obstacles":
            LOG_DEBUG("Guidance Constraints: Received dynamic obstacles")

            obstacles = []
            for obstacle in data.dynamic_obstacles:
                positions = [obstacle.position]  # Current position

                for k in range(len(obstacle.prediction.modes[0])):
                    positions.append(obstacle.prediction.modes[0][k].position)

                obstacles.append((
                    obstacle.index,
                    positions,
                    obstacle.radius + data.robot_area[0].radius
                ))

            for planner in self.planners:
                planner.load_obstacles(obstacles)

    def reset(self):
        """Reset the guidance constraints"""
        super().reset()
        for planner in self.planners:
            planner.reset()
        self.trajectories = []
        self.selected_trajectory = None


class GuidancePlanner(Planner):
    def __init__(self, solver):
        """Initialize the guidance planner with default parameters"""
        super().__init__(solver)
        self._success = False
        self._planning_frequency = 10.0  # Hz
        self._dt = 1.0 / self._planning_frequency

        # State and configuration
        self.current_pos = np.zeros(2)
        self.current_psi = 0.0
        self.current_v = 0.0
        self._reference_velocity = 0.0

        # Path and obstacle data
        self.reference_path = None
        self.start_s = 0.0
        self.width_left = 0.0
        self.width_right = 0.0
        self.static_obstacles = []
        self.dynamic_obstacles = []

        # Goals and trajectories
        self.goals = []
        self.trajectories = []
        self.selected_trajectory_id = -1

        # Configuration
        self.horizon = solver.horizon if hasattr(solver, 'horizon') else 10
        self.longitudinal_goals = 5
        self.vertical_goals = 5

        LOG_DEBUG("GuidancePlanner initialized")

    def set_start(self, pos, psi, v):
        """Set the starting position, orientation and velocity"""
        self.current_pos = np.array(pos)
        self.current_psi = psi
        self.current_v = v

    def set_reference_velocity(self, velocity):
        """Set the reference velocity for planning"""
        self._reference_velocity = velocity

    def load_static_obstacles(self, halfspaces):
        """Load static obstacles represented as halfspaces"""
        self.static_obstacles = halfspaces

    def load_obstacles(self, obstacles):
        """Load dynamic obstacles for planning"""
        self.dynamic_obstacles = obstacles

    def set_goals(self, goals):
        """Set the guidance goals (points and costs)"""
        self.goals = goals

    def load_reference_path(self, start_s, path, width_left, width_right):
        """Load a reference path with lateral boundaries"""
        self.reference_path = path
        self.start_s = start_s
        self.width_left = width_left
        self.width_right = width_right

    def update(self):
        """Main update function to generate trajectories"""
        # Reset trajectories
        self.trajectories = []

        if len(self.goals) == 0:
            LOG_DEBUG("No goals set, cannot update guidance")
            self._success = False
            return

        # Plan trajectories to reach goals
        self.generate_trajectories()

        # Mark planning as successful if we have trajectories
        self._success = len(self.trajectories) > 0

    def generate_trajectories(self):
        """Generate trajectories to reach goals"""
        # Sort goals by cost
        sorted_goals = sorted(self.goals, key=lambda g: g[1])

        # Create trajectories for each goal
        for i, (goal_point, cost) in enumerate(sorted_goals[:self.longitudinal_goals]):
            trajectory = self.create_trajectory_to_goal(goal_point, i, cost)
            self.trajectories.append(trajectory)

    def create_trajectory_to_goal(self, goal_point, traj_id, cost):
        """Create a trajectory to reach a goal point"""
        # Create trajectory object
        trajectory = type('Trajectory', (), {})
        trajectory.topology_class = traj_id
        trajectory.color = traj_id
        trajectory.previously_selected = (traj_id == self.selected_trajectory_id)
        trajectory.cost = cost

        # Create spline object
        trajectory.spline = type('Spline', (), {})

        # Define start and goal
        start_pos = self.current_pos
        goal_pos = np.array(goal_point[:2])  # Only take x,y from goal
        direction = goal_pos - start_pos
        distance = np.linalg.norm(direction)

        if distance > 0:
            direction = direction / distance
        else:
            direction = np.array([np.cos(self.current_psi), np.sin(self.current_psi)])

        # Create trajectory function
        def get_trajectory():
            traj = type('TrajectorySpline', (), {})

            def get_point(t):
                # Simple linear interpolation
                t_total = self.horizon * self._dt
                t = min(t, t_total)
                fraction = min(t / t_total, 1.0)
                return start_pos + direction * distance * fraction

            def get_velocity(t):
                # Constant velocity along the path
                t_total = self.horizon * self._dt
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
        return len(self.trajectories)

    def get_guidance_trajectory(self, idx):
        """Get a specific guidance trajectory by index"""
        if idx < 0 or idx >= len(self.trajectories):
            # Return a default trajectory
            trajectory = type('Trajectory', (), {})
            trajectory.topology_class = -1
            trajectory.color = -1
            trajectory.previously_selected = False

            trajectory.spline = type('Spline', (), {})
            trajectory.spline.get_trajectory = lambda: type('TrajectorySpline', (), {
                'get_point': lambda t: self.current_pos,
                'get_velocity': lambda t: np.array([self.current_v * np.cos(self.current_psi),
                                                    self.current_v * np.sin(self.current_psi)])
            })

            return trajectory

        return self.trajectories[idx]

    def space_time_point_num_states(self):
        """Return the number of states for space-time points"""
        return 3 if len(self.goals) > 0 and len(self.goals[0][0]) > 2 else 2

    def override_selected_trajectory(self, guidance_id):
        """Override which trajectory is selected as the best one"""
        self.selected_trajectory_id = guidance_id

    def reset(self):
        """Reset the guidance planner"""
        self.trajectories = []
        self.selected_trajectory_id = -1
        self._success = False
        self.goals = []