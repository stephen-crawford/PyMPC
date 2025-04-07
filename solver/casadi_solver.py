import casadi as ca
import numpy as np
from solver.solver_interface import BaseSolver


class CasADiSolver(BaseSolver):
    def __init__(self, dt=0.1, N=20):
        super().__init__(dt, N)
        # Properties to match the existing framework
        self.nx = 0  # Will be set later
        self.nu = 0  # Will be set later
        self.nvar = 0  # Will be set later
        self.npar = 0  # Will be set later

        # Parameter storage - matching the expected interface
        self.params = {"solver_timeout": 0.1}
        self.all_parameters = []  # Will be initialized properly later

        # Stage-specific objectives and constraints
        self.stage_objectives = [None] * N
        self.stage_constraints = [None] * N
        self.constraint_numbers = [0] * N
        self.constraint_lbs = [None] * N
        self.constraint_ubs = [None] * N

        # Dynamics function
        self.dynamics_func = None
        self.xinit_indices = None

        # Initialize CasADi optimizer
        self.opti = ca.Opti()
        self.solution = None

        # State vector map for name-based access
        self.state_map = {}  # Maps "x", "y", "v", etc. to indices

        # Previous solution for warmstart
        self.prev_solution = None

        # These will be initialized in finalize_problem()
        self.X = None
        self.U = None
        self.P = None  # Parameters
        self.X0 = None  # Initial state parameter

        # Parameter bundles to match the set_solver_parameter interface
        self.parameter_bundles = {}
        self.parameter_length = 0

    def initialize_parameters(self, parameter_bundles, length):
        """Initialize parameter storage according to bundles configuration"""
        self.parameter_bundles = parameter_bundles
        self.parameter_length = length
        total_params = self.N * length
        self.all_parameters = np.zeros(total_params)
        self.npar = total_params

    def get_ego_prediction(self, k, var_name):
        """Get predicted state/input variable at stage k - used by planner"""
        if self.solution is None:
            # If no solution yet, return current initial values
            if var_name in self.state_map:
                idx = self.state_map[var_name]
                return self.opti.debug.value(self.X[k][idx])
            return 0.0
        else:
            # Return solution value
            if var_name in self.state_map:
                idx = self.state_map[var_name]
                return self.solution.value(self.X[k][idx])
            return 0.0

    def set_state_map(self, state_map):
        """Set the mapping from state names to indices"""
        self.state_map = state_map

    def set_stage_objective(self, stage_idx, objective_func):
        """Set the objective function for a specific stage"""
        self.stage_objectives[stage_idx] = objective_func

    def set_stage_constraints(self, stage_idx, constraint_func):
        """Set the constraint function for a specific stage"""
        self.stage_constraints[stage_idx] = constraint_func

    def set_constraint_bounds(self, stage_idx, lb, ub, number):
        """Set constraint bounds for a specific stage"""
        self.constraint_lbs[stage_idx] = lb
        self.constraint_ubs[stage_idx] = ub
        self.constraint_numbers[stage_idx] = number

    def set_constraint_number(self, stage_idx, number):
        """Set number of constraints for a specific stage"""
        self.constraint_numbers[stage_idx] = number

    def set_dynamics(self, dynamics_func):
        """Set the dynamics function"""
        self.dynamics_func = dynamics_func

    def set_initial_state_indices(self, xinit_indices):
        """Set indices for initial state variables"""
        self.xinit_indices = xinit_indices

    def finalize_problem(self):
        """Set up the optimization problem with all configured components"""
        # Create state and input variables for each stage
        self.X = [self.opti.variable(self.nx) for _ in range(self.N + 1)]
        self.U = [self.opti.variable(self.nu) for _ in range(self.N)]

        # Parameter vector
        self.P = self.opti.parameter(self.npar)

        # Initial state parameter
        self.X0 = self.opti.parameter(self.nx)

        # Initialize cost
        total_cost = 0

        # Add objective and constraints for each stage
        for k in range(self.N):
            # Create stage variable z = [u_k, x_k]
            z_k = ca.vertcat(self.U[k], self.X[k])

            # Add stage cost if objective is defined
            if self.stage_objectives[k] is not None:
                stage_cost = self.stage_objectives[k](z_k, self.P)
                total_cost += stage_cost

            # Add stage constraints if defined
            if k > 0 and self.stage_constraints[k] is not None and self.constraint_numbers[k] > 0:
                g = self.stage_constraints[k](z_k, self.P)
                self.opti.subject_to(
                    self.constraint_lbs[k] <= g
                )
                self.opti.subject_to(
                    g <= self.constraint_ubs[k]
                )

            # Add dynamics constraint for next state
            if self.dynamics_func is not None:
                # Extract current state and input
                x_k = self.X[k]
                u_k = self.U[k]

                # Compute next state using dynamics
                z_k_full = ca.vertcat(u_k, x_k)
                x_next = self.dynamics_func(z_k_full, self.P)

                # Enforce next state equals predicted next state
                self.opti.subject_to(self.X[k + 1] == x_next)

        # Add initial state constraint
        self.opti.subject_to(self.X[0] == self.X0)

        # Set the objective
        self.opti.minimize(total_cost)

        # Configure solver
        solver_opts = {
            "expand": True,
            "max_iter": 100,
            "print_level": 0,
            "max_cpu_time": self.params["solver_timeout"]
        }
        self.opti.solver("ipopt", {}, solver_opts)

    def set_xinit(self, state):
        """Set initial state value - implements abstract method"""
        # Convert state object to vector based on state_map
        state_vector = np.zeros(self.nx)

        # If state is an object with getPos(), getVel() methods
        try:
            pos = state.getPos()
            state_vector[self.state_map.get("x", 0)] = pos[0]
            state_vector[self.state_map.get("y", 1)] = pos[1]

            vel = state.getVel()
            state_vector[self.state_map.get("v", 2)] = np.linalg.norm(vel)

            # Add other state variables like heading, etc.
            if hasattr(state, "get") and callable(getattr(state, "get")):
                for key, idx in self.state_map.items():
                    if key not in ["x", "y", "v"] and key in self.state_map:
                        state_vector[idx] = state.get(key, 0.0)

            # Special case for spline parameter
            if "spline" in self.state_map and hasattr(state, "get"):
                state_vector[self.state_map["spline"]] = state.get("spline", 0.0)

        except AttributeError:
            # If state is a simple vector or dictionary
            if isinstance(state, dict):
                for key, idx in self.state_map.items():
                    state_vector[idx] = state.get(key, 0.0)
            elif hasattr(state, "__getitem__"):
                # Assume it's directly a state vector
                state_vector = state

        self.opti.set_value(self.X0, state_vector)

    def load_warmstart(self):
        """Load warmstart values into the optimizer"""
        if self.prev_solution is not None:
            for k in range(self.N + 1):
                if k < self.N:
                    self.opti.set_initial(self.U[k], self.prev_solution['U'][k])
                self.opti.set_initial(self.X[k], self.prev_solution['X'][k])

    def initialize_warmstart(self, state, shift_forward=True):
        """Initialize warmstart from previous solution"""
        if self.solution is None:
            self.initialize_with_braking(state)
            return

        # Get the previous solution
        prev_X = [self.solution.value(self.X[k]) for k in range(self.N + 1)]
        prev_U = [self.solution.value(self.U[k]) for k in range(self.N)]

        if shift_forward:
            # Shift solution forward
            for k in range(self.N - 1):
                self.opti.set_initial(self.X[k], prev_X[k + 1])
                self.opti.set_initial(self.U[k], prev_U[min(k + 1, self.N - 1)])

            # Last point - extrapolate or repeat last
            self.opti.set_initial(self.X[self.N], prev_X[self.N])

        else:
            # Just use previous solution as is
            for k in range(self.N + 1):
                if k < self.N:
                    self.opti.set_initial(self.U[k], prev_U[k])
                self.opti.set_initial(self.X[k], prev_X[k])

        # Store for later use
        self.prev_solution = {'X': prev_X, 'U': prev_U}

    def initialize_with_braking(self, state):
        """Initialize with a simple braking trajectory"""
        # Get initial velocity
        v0 = 0.0
        try:
            vel = state.getVel()
            v0 = np.linalg.norm(vel)
        except AttributeError:
            if "v" in self.state_map and isinstance(state, dict):
                v0 = state.get("v", 0.0)
            elif hasattr(state, "__getitem__") and "v" in self.state_map:
                v0 = state[self.state_map["v"]]

        # Initialize X using straight-line braking trajectory
        decel = 2.0  # m/s^2, reasonable deceleration

        # Initial position
        x0, y0 = 0.0, 0.0
        try:
            pos = state.getPos()
            x0, y0 = pos[0], pos[1]
        except AttributeError:
            if "x" in self.state_map and "y" in self.state_map and isinstance(state, dict):
                x0, y0 = state.get("x", 0.0), state.get("y", 0.0)
            elif hasattr(state, "__getitem__"):
                x_idx = self.state_map.get("x", 0)
                y_idx = self.state_map.get("y", 1)
                if len(state) > max(x_idx, y_idx):
                    x0, y0 = state[x_idx], state[y_idx]

        # Get initial heading
        heading = 0.0
        try:
            heading = state.get("heading", 0.0)
        except AttributeError:
            if "heading" in self.state_map and isinstance(state, dict):
                heading = state.get("heading", 0.0)
            elif hasattr(state, "__getitem__") and "heading" in self.state_map:
                heading = state[self.state_map["heading"]]

        # Create trajectory
        x_traj = []
        u_traj = []

        for k in range(self.N + 1):
            t = k * self.dt
            v = max(0, v0 - decel * t)
            s = v0 * t - 0.5 * decel * t * t

            # State vector for this time step
            x_k = np.zeros(self.nx)

            # Fill in known values
            if "x" in self.state_map:
                x_k[self.state_map["x"]] = x0 + s * np.cos(heading)
            if "y" in self.state_map:
                x_k[self.state_map["y"]] = y0 + s * np.sin(heading)
            if "v" in self.state_map:
                x_k[self.state_map["v"]] = v
            if "heading" in self.state_map:
                x_k[self.state_map["heading"]] = heading
            if "spline" in self.state_map and hasattr(state, "get"):
                x_k[self.state_map["spline"]] = state.get("spline", 0.0) + s

            x_traj.append(x_k)

            # Control inputs - deceleration
            if k < self.N:
                u_k = np.zeros(self.nu)
                if "acceleration" in self.state_map and self.nu > 0:
                    u_k[0] = -decel  # Assuming first control is acceleration
                u_traj.append(u_k)

        # Set initial values for optimizer
        for k in range(self.N + 1):
            if k < self.N:
                self.opti.set_initial(self.U[k], u_traj[k])
            self.opti.set_initial(self.X[k], x_traj[k])

        # Save for warm starting
        self.prev_solution = {'X': x_traj, 'U': u_traj}

    def print_if_bound_limited(self):
        """Print if any variable is at its bounds - for debugging"""
        # This would require knowing variable bounds
        pass

    def solve(self):
        """Solve the optimization problem - implements abstract method"""
        try:
            # Set parameter values from all_parameters
            self.opti.set_value(self.P, self.all_parameters)

            # Update solver timeout
            solver_opts = self.opti.solver_options()
            solver_opts["max_cpu_time"] = self.params["solver_timeout"]

            # Solve the problem
            self.solution = self.opti.solve()

            # Store solution for warm starting
            prev_X = [self.solution.value(self.X[k]) for k in range(self.N + 1)]
            prev_U = [self.solution.value(self.U[k]) for k in range(self.N)]
            self.prev_solution = {'X': prev_X, 'U': prev_U}

            return 1  # Success
        except Exception as e:
            print(f"Solver failed: {e}")
            return -1  # Failure

    def get_output(self, k, var_name):
        """Get variable value from solution - implements abstract method"""
        if self.solution is None:
            return 0.0

        if var_name in self.state_map:
            # State variable
            idx = self.state_map[var_name]
            return self.solution.value(self.X[k][idx])
        else:
            # Try to look for control input
            try:
                # Assuming control inputs might be indexed numerically
                if var_name.startswith("u"):
                    # Input variable
                    idx = int(var_name[1:]) if len(var_name) > 1 else 0
                    return self.solution.value(self.U[k][idx])
            except (ValueError, IndexError):
                pass

        return 0.0  # Default return if not found

    def reset(self):
        """Reset the solver - implements abstract method"""
        self.opti = ca.Opti()
        self.solution = None
        self.prev_solution = None

        # Re-initialize all_parameters
        if hasattr(self, 'parameter_length'):
            total_params = self.N * self.parameter_length
            self.all_parameters = np.zeros(total_params)

        # Re-setup the problem
        self.finalize_problem()

    def explain_exit_flag(self, code):
        """Return a human-readable explanation of the exit flag - implements abstract method"""
        if code == 1:
            return "Solved successfully"
        elif code == -1:
            return "Solver failed - infeasible or numerical issues"
        else:
            return f"Unknown exit code: {code}"