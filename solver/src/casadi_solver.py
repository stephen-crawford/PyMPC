import copy

import casadi as cs
import numpy as np

from planning.src.dynamic_models import DynamicsModel
from planning.src.types import State, Trajectory
from solver.src.base_solver import BaseSolver
from utils.const import OBJECTIVE
from utils.utils import LOG_DEBUG, LOG_WARN, LOG_INFO, CONFIG

'''
Casadi solver used for trajectory optimization. 

Usage follows: 

1. Create solver object my_solver
2. Take a DynamicsModel and call my_solver.set_dynamics_model(model)
    > This will set the dynamics model for the solver and initialize the optimization variable dictionary to match
    > Will apply the upper and lower bounds for the state and control variables of the model as defined by the class 
3. Form an initial State instance for the trajectory you want to solve 
5. Call initialize_rollout(state, shift_forward)
6. Call solve()

'''

DEFAULT_BRAKING = -2.0

class CasADiSolver(BaseSolver):
    def __init__(self, timestep=0.1, horizon=30):
        super().__init__(timestep, horizon)
        LOG_INFO("CasADiSolver: Initializing solver")

        # CasADi specific attributes
        self.opti = cs.Opti()  # CasADi Opti stack
        self.var_dict = {}  # Dictionary to store optimization variables

        self.dynamics_model = None

        self.warmstart_values = {}
        self.forecast = []

        # Solution storage
        self.solution = None
        self.exit_flag = None
        self.info = {} # Used for delayed logging


    def set_dynamics_model(self, dynamics_model: DynamicsModel):
        """
        Incorporates the given dynamics model into the optimization problem.
        """

        self.dynamics_model = dynamics_model

        for var_name in self.dynamics_model.get_dependent_vars():
            self.var_dict[var_name] = self.opti.variable(self.horizon + 1) # This makes a horizon + 1 length symbolic vector for dependent variables
            self.warmstart_values[var_name] = np.zeros(self.horizon + 1)
        for var_name in self.dynamics_model.get_inputs():
            self.var_dict[var_name] = self.opti.variable(self.horizon)  # This makes a horizon length symbolic vector for inputs
            self.warmstart_values[var_name] = np.zeros(self.horizon)

        LOG_DEBUG(f"After setting dynamics model, vars are: {self.var_dict}")

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 1000,
            'ipopt.tol': 1e-4,
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.mu_init': 1e-1,  # Barrier parameter
            'ipopt.bound_push': 1e-8,  # Keep variables away from bounds
            'ipopt.bound_frac': 1e-8,  # Fraction of bound violation allowed
        }
        self.opti.solver('ipopt', opts)

        self._apply_model_bounds()

    def initialize_rollout(self, state: State, shift_forward=True):
        """Initialize solver with warmstart from previous solution

        Args:
            state: Current state object
            shift_forward: Whether to shift previous solution forward
        """
        self._set_initial_state(state)
        self._initialize_warmstart(state, shift_forward)

    def _initialize_warmstart(self, state: State, shift_forward=True):
        """Initialize with a braking trajectory using proper numerical integration"""
        if self.solution is None or not shift_forward:
            # Initialize with simple trajectory (e.g., constant velocity)
            self._initialize_base_warmstart(state)
        else:
            # Shift previous solution forward
            LOG_DEBUG("Previous solution found, shifting forward")
            self._shift_warmstart_forward()

        # Set initial values for optimization variables
        self._set_opti_initial_values()

        initial_forecast = self._create_trajectory_from_warmstart()
        self.forecast.append(copy.deepcopy(initial_forecast))

    def _initialize_base_warmstart(self, state: State):
        """Create initial warmstart trajectory using simple integration"""
        # Initialize states
        for var_name in self.dynamics_model.get_dependent_vars():
            current_value = state.get(var_name) if state.get(var_name) is not None else 0.0

            if var_name == 'v':
                for k in range(self.horizon + 1):

                    v_k = max(0.0, current_value + DEFAULT_BRAKING * k * self.timestep)
                    self.warmstart_values[var_name][k] = v_k
            elif var_name == 'psi':
                for k in range(self.horizon + 1):
                    # Integrate heading: psi_k = psi_0 + w * k * dt
                    #psi_k = current_value + state.get('w') * k * self.timestep
                    #psi_k = max(-np.pi * 4, psi_k)  # Apply bounds after integration
                    psi_k = current_value
                    self.warmstart_values[var_name][k] = psi_k
        for var_name in self.dynamics_model.get_dependent_vars():
            current_value = state.get(var_name) if state.get(var_name) is not None else 0.0

            if var_name in ['x', 'y', 'z']:  # Position - integrate velocity
                for k in range(self.horizon + 1):
                    if k == 0:
                        self.warmstart_values[var_name][k] = current_value
                    else:
                        # Use the updated velocity and heading from current time step
                        v_k = self.warmstart_values['v'][k]  # Current velocity
                        psi_k = self.warmstart_values['psi'][k]  # Current heading

                        if var_name == 'x':
                            self.warmstart_values[var_name][k] = (
                                    self.warmstart_values[var_name][k - 1] +
                                    v_k * self.timestep * np.cos(psi_k)
                            )
                        elif var_name == 'y':
                            self.warmstart_values[var_name][k] = (
                                    self.warmstart_values[var_name][k - 1] +
                                    v_k * self.timestep * np.sin(psi_k)
                            )
                    # ignore z since this should not change

        LOG_DEBUG("After base warmstart initialization, warm is " + str(self.warmstart_values))
        # Initialize inputs to predict based on current controls
        for var_name in self.dynamics_model.get_inputs():
            self.warmstart_values[var_name][:] = state.get(var_name)

    def _shift_warmstart_forward(self):
        """Shift previous solution forward by one timestep"""
        for var_name in self.dynamics_model.get_all_vars():
            if var_name in self.warmstart_values:
                # Shift forward
                self.warmstart_values[var_name][:-1] = self.warmstart_values[var_name][1:]

                # Provide a better prediction for the last value instead of duplicating
                if len(self.warmstart_values[var_name]) >= 2:
                    # Use linear extrapolation based on the last two values
                    last_val = self.warmstart_values[var_name][-2]
                    second_last_val = self.warmstart_values[var_name][-3] if len(
                        self.warmstart_values[var_name]) >= 3 else last_val

                    # Linear extrapolation: next_val = last_val + (last_val - second_last_val)
                    predicted_val = last_val + (last_val - second_last_val)
                    self.warmstart_values[var_name][-1] = predicted_val

    def _update_warmstart_from_solution(self):
        """Update warmstart values from solution"""
        for var_name in self.var_dict:
            try:
                self.warmstart_values[var_name] = np.array(self.solution.value(self.var_dict[var_name]))
            except:
                LOG_WARN(f"Could not update warmstart for {var_name}")

    def _create_trajectory_from_warmstart(self):
        """Create a Trajectory object from current warmstart values"""
        traj = Trajectory(timestep=self.timestep, length=self.horizon + 1)

        for k in range(self.horizon + 1):
            state_k = State(model_type=self.dynamics_model)

            # Extract state variables from warmstart
            for var_name in self.dynamics_model.get_dependent_vars():
                if var_name in self.warmstart_values:
                    value = self.warmstart_values[var_name][k]
                    state_k.set(var_name, value)

            # Extract input variables (if within horizon)
            if k < self.horizon:
                for var_name in self.dynamics_model.get_inputs():
                    if var_name in self.warmstart_values:
                        value = self.warmstart_values[var_name][k]
                        state_k.set(var_name, value)

            traj.add_state(state_k)

        return traj

    def _set_opti_initial_values(self):
        """Set initial values for optimization variables"""
        for var_name in self.dynamics_model.get_dependent_vars():
            for k in range(self.horizon + 1):
                self.opti.set_initial(self.var_dict[var_name][k], self.warmstart_values[var_name][k])

        for var_name in self.dynamics_model.get_inputs():
            for k in range(self.horizon):
                self.opti.set_initial(self.var_dict[var_name][k], self.warmstart_values[var_name][k])

    def _set_initial_state(self, state: State):
        """Set initial state constraints"""
        LOG_DEBUG("Setting initial state based on given state " + str(state))
        self.initial_state = state

        # Constrain initial state variables to actual values
        for var_name in self.dynamics_model.get_dependent_vars():
            value = state.get(var_name)
            if value is not None:
                symbolic_var = self.var_dict[var_name][0]
                print(f"Setting {symbolic_var} == {value}")
                self.opti.subject_to(symbolic_var == value)
                self.warmstart_values[var_name][0] = value

    def _apply_model_bounds(self):
        """
        Makes a horizon size set of upper and lower bounds for each dependent variable
        Makes a horizon-1 size set of upper and lower bounds for each input
        """

        # Apply state variable bounds
        LOG_DEBUG("CasADi solver applying model lower and upper bounds")
        for var_name in self.dynamics_model.get_dependent_vars():

            var = self.var_dict[var_name]
            lb, ub, _ = self.dynamics_model.get_bounds(var_name)
            LOG_DEBUG("{} lb, ub {}, {}".format(var_name,lb, ub))
            for k in range(self.horizon + 1):
                self.opti.subject_to(self.opti.bounded(lb, var[k], ub))

        for var_name in self.dynamics_model.get_inputs():
            var = self.var_dict[var_name]
            lb, ub, _ = self.dynamics_model.get_bounds(var_name)
            LOG_DEBUG("{} lb, ub {}, {}".format(var_name, lb, ub))
            for k in range(self.horizon):
                self.opti.subject_to(self.opti.bounded(lb, var[k], ub))

        self._add_dynamics_constraints()

    def _add_dynamics_constraints(self):
        LOG_DEBUG("CasADi solver adding dynamics constraints")

        initial_ng = self.opti.ng
        final_ng = 0
        for k in range(self.horizon): # Creates an array of symbolic variables representing the inputs at the kth timestep

            x_k = [self.var_dict[var_name][k] for var_name in self.dynamics_model.get_dependent_vars()]
            u_k = [self.var_dict[var_name][k] for var_name in self.dynamics_model.get_inputs()]

            # Combine into z_k [a_0, ... a_horizon-1, w_0, ..., w_horizon -1, x_0, ..., x_horizon-, etc.]
            z_k = cs.vertcat(*u_k, *x_k)

            # Load into dynamics model
            self.dynamics_model.load(z_k)

            # Symbolic predicted next state
            next_state = self.dynamics_model.discrete_dynamics(z_k, self.parameter_manager, self.timestep)

            # For each dependent variables make the associated symbolic var. forced to equal the integrated next state value for that variable
            for i, var in enumerate(self.dynamics_model.get_dependent_vars()):
                if (hasattr(self.dynamics_model, 'state_dimension_integrate') and
                        self.dynamics_model.state_dimension_integrate is not None and
                        i >= self.dynamics_model.state_dimension_integrate):
                    continue
                self.opti.subject_to(self.var_dict[var][k + 1] == next_state[i])
            final_ng = self.opti.ng

        LOG_DEBUG(f"Number of dynamic constraints added for each timestep: {final_ng - initial_ng}")


    def solve(self):
        """Solve the optimization problem

        Returns:
            int: Exit flag (1 for success, -1 for failure)
        """
        LOG_DEBUG("Attempting to solve in Casadi solver")
        cost_comps_by_stage = []

        try:
            # Get solver timeout parameter
            timeout = self.parameter_manager.get("solver_timeout")
            # Update solver options with timeout
            options = {}
            if timeout > 0:
                options['ipopt.max_cpu_time'] = timeout

            total_objective = 0
            for stage_idx in range(self.horizon + 1):

                LOG_DEBUG("Trying to get objective cost for stage " + str(stage_idx) + " curr var dict: " + str(self.var_dict))
                symbolic_state = State(self.dynamics_model)

                # Add state variables
                for var_name in self.dynamics_model.get_all_vars():
                    if stage_idx < self.var_dict[var_name].shape[0]:
                        symbolic_state.set(var_name, self.var_dict[var_name][stage_idx])

                LOG_DEBUG("Going to fetch costs for symbolic state: " + str(symbolic_state))
                objective_costs = self.get_objective_cost(symbolic_state, stage_idx)

                LOG_DEBUG("Objective cost: " + str(objective_costs))
                cost_comps_by_stage.append(objective_costs)

                objective_value = 0
                for dic in objective_costs:
                    for item in dic.items():
                        objective_value += item[1]

                # Ensure the objective is a scalar
                if isinstance(objective_value, cs.MX) and objective_value.shape[0] > 1:
                    # Sum all elements if it's a vector
                    objective_value = cs.sum1(objective_value)

                total_objective += objective_value

                constraints = self.get_constraints(stage_idx)
                if constraints is not None:
                    for (c, lb, ub) in constraints:
                        self.opti.subject_to(self.opti.bounded(lb, c, ub))
            # Set the problem objective

                penalty_terms = self.get_penalty_terms(stage_idx)
                for penalty in penalty_terms:
                    total_objective += penalty

            self.opti.minimize(total_objective) #minimizes the accumulated cost over all stages

            try:
                sol = self.opti.solve()

            except RuntimeError as e:
                stats = self.opti.stats()
                LOG_WARN(f"Solver status: {stats.get('return_status', 'unknown')}")
                LOG_WARN(f"Solver message: {stats.get('iterations', {})}")
                LOG_WARN(f"Number of iterations: {stats.get('iter_count', 'N/A')}")
                LOG_WARN(f"Infeasibilities: {stats.get('infeasibilities', 'N/A')}")
                LOG_WARN(f"CPU time: {stats.get('t_proc_total', 'N/A')}")

                LOG_WARN(f"[ERROR] CasADi solver failed at iteration: {self.opti.stats()['iter_count']} ")
                LOG_WARN("[ERROR] Exception:" + str(e))
                g_vals = self.opti.debug.value(self.opti.g)
                lbg_vals = self.opti.debug.value(self.opti.lbg)
                ubg_vals = self.opti.debug.value(self.opti.ubg)

                for i, (g, l, u) in enumerate(zip(g_vals, lbg_vals, ubg_vals)):
                    if g < l - 1e-6 or g > u + 1e-6:
                        print(f"[VIOLATION] Constraint[{i}] = {g:.6f}, bounds: [{l}, {u}]")

                # LOG_DEBUG all decision variable values at failure
                for i in range(self.opti.nx):
                    LOG_WARN(f"x[{i}] = {self.opti.debug.value(self.opti.x[i])}")

                # print constraints at failure
                for i in range(self.opti.ng):
                    LOG_WARN(f"g[{i}] = {self.opti.debug.value(self.opti.g[i])}")

                LOG_WARN("Solver failed. Dumping debug values...")
                for name in self.var_dict:
                    try:
                        val = self.opti.debug.value(self.var_dict[name])
                        LOG_WARN(f"{name}: {val}")
                    except RuntimeError:
                        LOG_DEBUG(f"Could not extract value for {name}")

                raise

            # Store solution
            self.solution = sol
            if self.solution:
                for var_name, var in self.var_dict.items():
                    values = self.solution.value(var)
                    lb, ub, _ = self.dynamics_model.get_bounds(var_name)

                    for i, val in enumerate(values):
                        if abs(val - lb) < 1e-4:
                            LOG_WARN(f"{var_name}[{i}] at lower bound: {val}")
                        elif abs(val - ub) < 1e-4:
                            LOG_WARN(f"{var_name}[{i}] at upper bound: {val}")
            self.exit_flag = 1
            self.info["status"] = "success"
            LOG_DEBUG("CasADi solver solved successfully")

            self._update_warmstart_from_solution()
            for k in range(self.horizon + 1):
                for i in range(len(self.module_manager.get_modules())):
                    module = self.module_manager.get_modules()[i]
                    if module.module_type == OBJECTIVE:
                        mod_name = module.get_name()
                        LOG_INFO(f"=== Stage {k} Costs for Module {mod_name} ===")
                        dic = cost_comps_by_stage[k][i]
                        for name, expr in dic.items():
                            # Substitute optimized variable values into exprf
                            try:
                                # Create CasADi function for evaluation
                                f = cs.Function(f"eval_{name}_{k}", [self.opti.x], [expr])
                                val = f(sol.value(self.opti.x))

                                if isinstance(val, cs.DM):
                                    val_np = np.array(val)
                                    if val_np.size == 1:
                                        LOG_INFO(f"{name} = {float(val)}")
                                    else:
                                        LOG_WARN(f"{name} evaluated to non-scalar: {val_np}")
                                else:
                                    LOG_WARN(f"{name} is not a DM: {type(val)}")
                                LOG_INFO(f"{name} = {float(val)}")  # convert CasADi DM to float
                            except Exception as e:
                                LOG_DEBUG(f"Could not evaluate {name} at stage {k}: {e}")

            # Update warmstart values for next iteration
            LOG_DEBUG(f"Var dict is {self.var_dict}")
            for var_name in self.var_dict:
                if var_name in self.dynamics_model.get_dependent_vars():
                    self.warmstart_values[var_name] = np.array(sol.value(self.var_dict[var_name]))
                elif var_name in self.dynamics_model.get_inputs():
                    self.warmstart_values[var_name] = np.array(sol.value(self.var_dict[var_name]))

            LOG_DEBUG("Optimization solved successfully")
            LOG_INFO("=== SOLUTION FOUND ===")
            for var_name, var_vector in self.var_dict.items():
                try:
                    values = self.solution.value(var_vector)
                    values = np.round(values, decimals=4)  # Round for cleaner logs
                    LOG_INFO(f"{var_name}: {values}")
                except Exception as e:
                    LOG_WARN(f"Failed to log variable {var_name}: {e}")
            return 1
        except Exception as e:
            LOG_WARN(f"Solver failed: {str(e)}")
            self.exit_flag = -1
            self.info["status"] = "failed"
            self.info["error"] = str(e)
            return -1

    def get_initial_state(self):
        return self.initial_state

    ################################# UTILITY FUNCTIONS #################################

    def get_output(self, k, var_name):
        """Get the output value for a specific variable at time step k

        Args:
            k (int): Time step index (0 to horizon)
            var_name (str): Variable name

        Returns:
            float: Variable value at time step k, or None if unavailable
        """
        if self.solution is None:
            return None

        # Check if the variable exists
        if var_name not in self.var_dict:
            return None

        # Check if k is valid for the variable
        var_length = self.horizon + 1 if var_name in self.dynamics_model.get_dependent_vars() else self.horizon
        if k < 0 or k >= var_length:
            return None
        # Return the value from the solution
        try:
            return float(self.solution.value(self.var_dict[var_name][k]))
        except:
            return None

    def get_reference_trajectory(self):
        """Extract optimized trajectory from solution"""
        traj = Trajectory(timestep=self.timestep, length=self.horizon + 1)

        LOG_DEBUG("Extracting trajectory from solution")

        if self.solution is None:
            LOG_WARN("No solution found. Cannot extract reference trajectory.")
            return traj

        for k in range(self.horizon + 1):
            state_k = State(model_type=self.dynamics_model)

            # Extract state variables
            for var_name in self.dynamics_model.get_dependent_vars():
                try:
                    value = float(self.solution.value(self.var_dict[var_name][k]))
                    state_k.set(var_name, value)
                except Exception as e:
                    LOG_WARN(f"Failed to extract {var_name}[{k}]: {e}")
                    state_k.set(var_name, 0.0)

            # Extract input variables (if within horizon)
            if k < self.horizon:
                for var_name in self.dynamics_model.get_inputs():
                    try:
                        value = float(self.solution.value(self.var_dict[var_name][k]))
                        state_k.set(var_name, value)
                    except Exception as e:
                        LOG_WARN(f"Failed to extract {var_name}[{k}]: {e}")
                        state_k.set(var_name, 0.0)

            traj.add_state(state_k)

        return traj

    def print_if_bound_limited(self):
        """Print which variables are at their bounds (for debugging)"""
        if self.solution is None:
            return

        LOG_DEBUG("Checking variable bounds:")
        for var_name, var in self.var_dict.items():
            # Get solution values
            try:
                values = self.solution.value(var)
                if isinstance(values, (float, int)):
                    values = [values]  # Convert to list for consistency

                # Check for values close to typical bounds
                for i, val in enumerate(values):
                    if abs(val) > 1e6:
                        LOG_WARN(f"Variable {var_name}[{i}] has very large value: {val}")
            # You can add more specific bound checks here
            except:
                pass

    def explain_exit_flag(self, code=None):
        """Explain the solver exit code

        Args:
            code (int, optional): Exit code to explain

        Returns:
            str: Human-readable explanation
        """
        code = code if code is not None else self.exit_flag

        explanations = {
            1: "Optimization successful",
            -1: "Optimization failed" + (
                f": {self.info.get('error', '')}" if hasattr(self, 'info') and 'error' in self.info else ""),
            -2: "Problem is infeasible",
            -3: "Maximum iterations reached",
            -10: "Solver timeout reached"
        }

        return explanations.get(code, f"Unknown exit code: {code}")

    def get_forecasts(self):
        return self.forecast


    def reset(self):
        """Reset the solver to its initial state."""

        # Clear and reinitialize variables and parameters
        self.opti = cs.Opti()

        self.solution = None
        self.exit_flag = None
        self.info = {}
        self.var_dict.clear()
        LOG_DEBUG("After reset, var dict is: " + str(self.var_dict.items()))
        self.set_dynamics_model(self.dynamics_model)
        for module in self.module_manager.modules:
            module.reset()
        LOG_DEBUG("CasADi solver reset.")

    # For compatibility with the model interface expected by modules
    def load(self, z):
        """Compatibility method for BaseSolver's static methods
        In CasADi, this is a no-op as we're directly using the solver instance.
        """
        pass

    def copy(self):
        """Create a copy of the solver (for solver management)

        Returns:
            E
        """
        # Create a new solver with same parameters
        new_solver = CasADiSolver(self.timestep, self.horizon)

        # Copy module manager
        new_solver.module_manager = self.module_manager

        # Copy parameter manager
        new_solver.parameter_manager = self.parameter_manager.copy()

        return new_solver
