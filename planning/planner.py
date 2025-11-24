import time
import numpy as np

from planning.types import *
from planning.types import propagate_obstacles
from planning.obstacle_manager import ObstacleManager
from planning.modules_manager import ModuleManager
from planning.parameter_manager import ParameterManager
from solver.casadi_solver import CasADiSolver
from utils.const import OBJECTIVE
from utils.utils import CONFIG, LOG_INFO, LOG_WARN, LOG_DEBUG


class Planner:
  def __init__(self, problem, config=None):
    LOG_INFO("=== Planner Initialization ===")
    
    if config is None:
      self.config = CONFIG
      LOG_DEBUG("Using default CONFIG")

    else:
      self.config = config
      LOG_DEBUG("Using provided config")

    self.problem = problem
    self.module_manager = ModuleManager()
    self.parameter_manager = ParameterManager()
    self.obstacle_manager = ObstacleManager(self.config)
    self.output = PlannerOutput()
    self.state = None
    LOG_INFO("Starting problem setup...")
    self.problem_setup()
    LOG_INFO("Planner initialization complete")


  def problem_setup(self):
    LOG_INFO("Setting up problem configuration...")
    
    self.model_type = self.problem.get_model_type()
    LOG_INFO(f"Model type: {self.model_type.__class__.__name__ if self.model_type else 'None'}")
    
    self.modules = self.problem.get_modules()
    LOG_INFO(f"Number of modules: {len(self.modules) if self.modules else 0}")
    for i, module in enumerate(self.modules):
      module_name = getattr(module, 'name', 'Unknown')
      module_type = getattr(module, 'module_type', 'Unknown')
      LOG_DEBUG(f"  Module {i+1}: {module_name} (type: {module_type})")
    
    self.obstacles = self.problem.get_obstacles()
    LOG_INFO(f"Number of obstacles: {len(self.obstacles) if self.obstacles else 0}")
    
    self.data = self.problem.get_data()
    if self.data:
      LOG_INFO("Data object provided")
      # Log data fields
      data_fields = [attr for attr in dir(self.data) if not attr.startswith('_')]
      LOG_DEBUG(f"Data has {len(data_fields)} attributes")
      key_fields = ['horizon', 'timestep', 'dynamics_model', 'reference_path', 'robot_area', 'dynamic_obstacles']
      for field in key_fields:
        has_field = hasattr(self.data, field)
        field_val = getattr(self.data, field, None)
        if field == 'robot_area' and field_val is not None:
          # Log detailed robot_area information
          disc_info = []
          for i, disc in enumerate(field_val):
            if hasattr(disc, 'offset') and hasattr(disc, 'radius'):
              disc_info.append(f"disc[{i}]: offset={disc.offset:.3f}, radius={disc.radius:.3f}")
          LOG_INFO(f"  data.robot_area: {len(field_val)} disc(s) - {', '.join(disc_info)}")
        else:
          LOG_DEBUG(f"  data.{field}: {'present' if has_field else 'missing'}, value={'set' if field_val is not None else 'None'}")
    else:
      LOG_WARN("No data object provided")
    
    self.x0 = self.problem.get_x0()
    if self.x0 is not None:
      LOG_INFO("Initial state (x0) provided")
      state_vars = {var: self.x0.get(var) for var in self.x0.get_all_vars() if self.x0.has(var)}
      LOG_DEBUG(f"Initial state variables: {list(state_vars.keys())}")
    else:
      LOG_DEBUG("No initial state provided, will create default")
    
    # Initialize state from x0 if provided
    if self.x0 is not None:
      self.state = self.x0
      LOG_DEBUG("Using provided x0 as initial state")
    else:
      # Create default state if not provided
      self.state = State(self.model_type)
      LOG_DEBUG("Created default state from model_type")
    
    solver_type = self.config.get("solver", {}).get("solver", "casadi") if isinstance(self.config.get("solver"), dict) else self.config.get("solver", "casadi")
    LOG_INFO(f"Creating solver: {solver_type}")
    if solver_type == "casadi":
      self.solver = CasADiSolver(self.config)
      LOG_DEBUG("CasADiSolver created successfully")
    else:
      raise ValueError("Solver not supported")
    
    # Add modules to solver's module manager
    LOG_INFO("Adding modules to solver's module manager...")
    for module in self.modules:
      module_name = getattr(module, 'name', 'Unknown')
      LOG_DEBUG(f"Adding module '{module_name}' to solver module manager")
      self.solver.module_manager.add_module(module)
    
    # Also add to planner's module manager for compatibility
    LOG_INFO("Adding modules to planner's module manager...")
    self.module_manager.add_modules(self.modules)
    
    objective_modules = self.module_manager.get_objective_modules()
    if objective_modules is None:
      LOG_WARN("No objective modules found!")
      raise ValueError("No objective modules found")
    else:
      LOG_INFO(f"Found {len(objective_modules)} objective module(s)")
      for obj_mod in objective_modules:
        LOG_DEBUG(f"  Objective: {getattr(obj_mod, 'name', 'Unknown')}")
    
    if self.module_manager.check_dependencies() is False:
      LOG_WARN("Module dependencies are not satisfied!")
      raise ValueError("Dependencies are not satisfied")
    else:
      LOG_DEBUG("All module dependencies satisfied")

    LOG_INFO("Adding obstacles to obstacle manager...")
    if self.data is not None:
        if self.obstacles:
            # Use obstacle manager's create_obstacles_from_config if available
            if hasattr(self.obstacle_manager, 'create_obstacles_from_config'):
                # Convert obstacles to config format if needed
                obstacle_configs = []
                for i, obs in enumerate(self.obstacles):
                    from planning.obstacle_manager import ObstacleConfig, PredictionType
                    obs_radius = getattr(obs, 'radius', 0.5)  # Default radius
                    LOG_DEBUG(f"  Obstacle {i}: position={getattr(obs, 'position', 'unknown')}, radius={obs_radius:.3f}")
                    obstacle_configs.append(ObstacleConfig(
                        obstacle_id=i,
                        initial_position=obs.position if hasattr(obs, 'position') else np.array([0.0, 0.0]),
                        initial_velocity=obs.velocity if hasattr(obs, 'velocity') else np.array([1.0, 0.0]),
                        dynamics_type="unicycle"  # Default, may need to detect from obstacle type
                    ))
                self.data.dynamic_obstacles = self.obstacle_manager.create_obstacles_from_config(obstacle_configs)
                # Log obstacle radii after creation
                for i, obs in enumerate(self.data.dynamic_obstacles):
                    obs_radius = getattr(obs, 'radius', 'unknown')
                    LOG_DEBUG(f"  Created obstacle {i}: radius={obs_radius}")
            else:
                # Fallback: just use obstacles directly
                self.data.dynamic_obstacles = self.obstacles if self.obstacles else []
                # Log obstacle radii
                for i, obs in enumerate(self.data.dynamic_obstacles):
                    obs_radius = getattr(obs, 'radius', 'unknown')
                    LOG_DEBUG(f"  Obstacle {i}: radius={obs_radius}")
        else:
            self.data.dynamic_obstacles = []
        LOG_INFO(f"Obstacle manager now has {len(self.data.dynamic_obstacles) if self.data.dynamic_obstacles else 0} obstacles")
    else:
        LOG_WARN("Cannot add obstacles: data is None")
    
    # Store data reference in solver and set required attributes
    self.solver.data = self.data
    LOG_DEBUG("Data reference stored in solver")
    
    # Set dynamics model on solver
    if hasattr(self.solver, 'set_dynamics_model'):
      self.solver.set_dynamics_model(self.model_type)
      LOG_DEBUG("Dynamics model set via set_dynamics_model method (should be in data)")
    else:
      # DO NOT set solver.dynamics_model directly - it should only come from data
      # Ensure data has dynamics_model set instead
      if self.data and hasattr(self.data, 'dynamics_model'):
        self.data.dynamics_model = self.model_type
        LOG_DEBUG("Dynamics model set in data.dynamics_model")
      else:
        LOG_WARN("Cannot set dynamics_model: data is None or has no dynamics_model attribute")
    
    # Ensure data has horizon and dynamics_model for solver compatibility
    if self.data:
      self.data.horizon = self.problem.get_horizon()
      self.data.timestep = self.problem.get_timestep()
      self.data.dynamics_model = self.model_type
      LOG_INFO(f"Data configured: horizon={self.data.horizon}, timestep={self.data.timestep}")
    else:
      LOG_WARN("Cannot set data attributes: data is None")
    
    LOG_INFO("Problem setup complete")

  
  def solve(self):
    LOG_INFO("=== Starting Planner.solve() ===")
    solver_iterations = 0
    max_iterations = self.config.get("solver_iterations", 1)
    LOG_INFO(f"Maximum solver iterations: {max_iterations}")

    while solver_iterations < max_iterations:
      LOG_INFO(f"--- Solver iteration {solver_iterations + 1}/{max_iterations} ---")
      
      # Log current state before solve
      current_x = self.state.get('x') if self.state.has('x') else None
      current_y = self.state.get('y') if self.state.has('y') else None
      current_psi = self.state.get('psi') if self.state.has('psi') else None
      current_v = self.state.get('v') if self.state.has('v') else None
      current_spline = self.state.get('spline') if self.state.has('spline') else None
      LOG_INFO(f"Current state before solve: x={current_x:.2f}, y={current_y:.2f}, psi={current_psi:.2f}, v={current_v:.2f}" + 
               (f", spline={current_spline:.3f}" if current_spline is not None else ""))
      
      mpc_output = self.solve_mpc()
      
      if mpc_output.success:
        LOG_DEBUG(f"Iteration {solver_iterations + 1}: MPC solve successful")
        if hasattr(mpc_output, 'control') and mpc_output.control:
          LOG_INFO(f"Control output: {mpc_output.control}")
          # Log angular velocity specifically
          if 'w' in mpc_output.control:
            LOG_INFO(f"  Angular velocity w = {mpc_output.control['w']:.4f} rad/s")
          if 'a' in mpc_output.control:
            LOG_INFO(f"  Acceleration a = {mpc_output.control['a']:.4f} m/s²")
        else:
          LOG_WARN(f"Iteration {solver_iterations + 1}: MPC solved but no control output!")
        
        if mpc_output.control:
          self.output.control_history.append(mpc_output.control)
          # Log state before propagation
          before_x = self.state.get('x') if self.state.has('x') else None
          before_y = self.state.get('y') if self.state.has('y') else None
          
          # Propagate state using dynamics model (ensuring spline/state updates match solver)
          dynamics_model = None
          if hasattr(self.solver, 'dynamics_model') and self.solver.dynamics_model is not None:
            dynamics_model = self.solver.dynamics_model
          else:
            dynamics_model = self.model_type
          parameter_manager = getattr(self.solver, 'parameter_manager', None)
          
          self.state = self.problem.get_state().propagate(
            mpc_output.control,
            self.solver.timestep,
            dynamics_model=dynamics_model,
            parameter_manager=parameter_manager
          )
          
          # Log state after propagation
          after_x = self.state.get('x') if self.state.has('x') else None
          after_y = self.state.get('y') if self.state.has('y') else None
          after_psi = self.state.get('psi') if self.state.has('psi') else None
          after_v = self.state.get('v') if self.state.has('v') else None
          after_spline = self.state.get('spline') if self.state.has('spline') else None
          
          LOG_INFO(f"State propagated: ({before_x:.2f}, {before_y:.2f}) -> ({after_x:.2f}, {after_y:.2f})")
          LOG_DEBUG(f"  After: x={after_x:.2f}, y={after_y:.2f}, psi={after_psi:.2f}, v={after_v:.2f}" + 
                   (f", spline={after_spline:.3f}" if after_spline is not None else ""))
          
          self.data.update(self.state)
          self.output.realized_trajectory.add_state(self.state)
        else:
          LOG_WARN(f"Iteration {solver_iterations + 1}: No control to apply, skipping propagation")
        
        solver_iterations += 1
        
        if self.state.is_objective_reached(self.data):
          LOG_INFO(f"Objective reached at iteration {solver_iterations}!")
          break
      else:
        LOG_WARN(f"Iteration {solver_iterations + 1}: MPC solve failed")
        break
    
    LOG_INFO(f"Planner.solve() completed after {solver_iterations} iterations")
    return self.output
  
  def solve_mpc(self, data=None):
    """Solve MPC optimization problem.
    
    Args:
      data: Optional Data object. If provided, updates self.data.
    """
    LOG_INFO("=== Starting Planner.solve_mpc() ===")
    
    # Update data if provided
    if data is not None:
      LOG_DEBUG("Updating data from provided argument")
      self.data = data
      # Log goal if available
      if hasattr(data, 'goal') and data.goal is not None:
        LOG_INFO(f"Planner.solve_mpc: data.goal = ({data.goal[0]:.3f}, {data.goal[1]:.3f})")
      if hasattr(data, 'parameters') and data.parameters is not None:
        goal_x = data.parameters.get("goal_x")
        goal_y = data.parameters.get("goal_y")
        if goal_x is not None and goal_y is not None:
          LOG_INFO(f"Planner.solve_mpc: parameters.goal = ({goal_x:.3f}, {goal_y:.3f})")
    else:
      LOG_DEBUG("Using existing self.data")
    
    # Ensure state is set in data - data is the single source of truth
    if self.data is not None:
      if not hasattr(self.data, 'state') or self.data.state is None:
        if self.state is not None:
          self.data.state = self.state
          LOG_DEBUG("Set data.state from planner.state")
      else:
        # Update planner.state to match data.state (data is authoritative)
        self.state = self.data.state
        LOG_DEBUG("Updated planner.state from data.state")
      
      # Log current state
      if self.state is not None:
        state_x = self.state.get('x') if self.state.has('x') else None
        state_y = self.state.get('y') if self.state.has('y') else None
        LOG_INFO(f"Planner.solve_mpc: current state = ({state_x:.3f}, {state_y:.3f})")
    
    # Ensure solver horizon and timestep are set
    if not hasattr(self.solver, 'horizon') or self.solver.horizon is None:
      planner_config = self.config.get("planner", {})
      self.solver.horizon = planner_config.get("horizon", 10)
      self.solver.timestep = planner_config.get("timestep", 0.1)
      LOG_DEBUG(f"Solver horizon/timestep set from config: {self.solver.horizon}/{self.solver.timestep}")
    else:
      LOG_DEBUG(f"Solver horizon/timestep already set: {self.solver.horizon}/{self.solver.timestep}")

    LOG_INFO("Checking data readiness for all modules...")
    modules = self.solver.module_manager.get_modules()
    data_ready_status = {}
    for module in modules:
      module_name = getattr(module, 'name', 'Unknown')
      is_ready = module.is_data_ready(self.data)
      data_ready_status[module_name] = is_ready
      LOG_DEBUG(f"  Module '{module_name}': data_ready={is_ready}")
    
    is_data_ready = all(data_ready_status.values())
    if not is_data_ready:
      not_ready = [name for name, ready in data_ready_status.items() if not ready]
      LOG_WARN(f"Data not ready for modules: {not_ready}")
      return self.output
    else:
      LOG_INFO("All modules report data is ready")

    # Ensure state is in data before passing to solver
    if self.data is not None and (not hasattr(self.data, 'state') or self.data.state is None):
      self.data.state = self.state
      LOG_DEBUG("Set data.state from planner.state before solver initialization")

    # Ensure solver is initialized before calling initialize_rollout
    if not hasattr(self.solver, 'opti') or self.solver.opti is None:
      LOG_DEBUG("Initializing solver (opti is None)")
      if self.data and hasattr(self.data, 'dynamics_model') and self.data.dynamics_model:
        # DO NOT set solver.dynamics_model - it should only come from data
        self.solver.intialize_solver(self.data)
        LOG_DEBUG("Solver initialized with dynamics model from data")
      else:
        LOG_WARN("Cannot initialize solver: data or dynamics_model missing")
    else:
      LOG_DEBUG("Solver already initialized")
    
    LOG_INFO("Initializing rollout...")
    # Pass state via data - solver will get it from data.state
    # Still pass state parameter for backward compatibility, but data.state is authoritative
    self.solver.initialize_rollout(self.state, self.data)

    # Ensure horizon is set for propagate_obstacles
    horizon_val = self.solver.horizon if self.solver.horizon is not None else 10
    timestep_val = self.solver.timestep if self.solver.timestep is not None else 0.1
    LOG_DEBUG(f"Propagating obstacles: horizon={horizon_val}, timestep={timestep_val}")
    # Call with Data first per function signature: propagate_obstacles(data, dt, horizon, ...)
    propagate_obstacles(self.data, dt=timestep_val, horizon=horizon_val)
    LOG_DEBUG("Obstacles propagated")

    LOG_INFO("Updating all modules...")
    for module in self.solver.module_manager.get_modules():
      module_name = getattr(module, 'name', 'Unknown')
      LOG_DEBUG(f"Updating module '{module_name}'")
      module.update(self.state, self.data)

    # Set data for all stages including final stage (horizon + 1 stages: 0 to horizon)
    LOG_INFO(f"Setting parameters for {horizon_val + 1} stages...")
    # CRITICAL FIX: Use solver's parameter_manager, not planner's separate one
    # The solver's parameter_manager is what modules access during get_value()
    for k in range(horizon_val + 1):
      for module in self.solver.module_manager.get_modules():
        self.solver.parameter_manager.set_parameters(module, self.data, k)
        # Log goal parameters for goal objective at stage 0
        if k == 0 and hasattr(module, 'name') and module.name == 'goal_objective':
          params_0 = self.solver.parameter_manager.get_all(0)
          goal_x = params_0.get('goal_x')
          goal_y = params_0.get('goal_y')
          if goal_x is not None and goal_y is not None:
            LOG_INFO(f"Planner: After set_parameters, goal_x={goal_x:.3f}, goal_y={goal_y:.3f} at stage 0")
    
    # CRITICAL: Constraints and objectives are now computed symbolically in the solver
    # The solver creates symbolic states for each stage and passes them to modules.
    # We no longer pre-collect constraints/objectives here since they must use symbolic states.
    # Reference: https://github.com/tud-amr/mpc_planner - all computation is symbolic in the solver.
    LOG_INFO(f"Parameters set for {horizon_val + 1} stages. Constraints and objectives will be computed symbolically in solver.")
    
    # Set data.parameters for backward compatibility (used by some modules)
    if not hasattr(self.data, 'parameters') or self.data.parameters is None:
      self.data.parameters = {}
    params_0 = self.solver.parameter_manager.get_all(0)
    self.data.parameters.update(params_0)
    # Log goal parameters
    if 'goal_x' in params_0 and 'goal_y' in params_0:
      LOG_INFO(f"Planner: Set data.parameters with goal_x={params_0['goal_x']:.3f}, goal_y={params_0['goal_y']:.3f}")

    LOG_INFO("Calling solver.solve()...")
    LOG_DEBUG(f"  Solving MPC over horizon={horizon_val}, timestep={timestep_val}")
    LOG_DEBUG(f"  Current state: x={self.state.get('x') if self.state.has('x') else 'N/A'}, y={self.state.get('y') if self.state.has('y') else 'N/A'}, v={self.state.get('v') if self.state.has('v') else 'N/A'}")
    exit_flag = self.solver.solve(self.state, self.data)

    if exit_flag != 1:
      self.output.success = False
      error_msg = self.solver.explain_exit_flag(exit_flag)
      LOG_WARN(f"MPC solve failed: {error_msg}")
      LOG_WARN("  No control extracted - solver did not find a solution")
      return self.output

    self.output.success = True
    LOG_INFO("=== MPC SOLVE SUCCESSFUL ===")
    LOG_INFO("  Solver found optimal trajectory over the horizon")
    
    # Get reference trajectory (optimal trajectory over horizon)
    reference_trajectory = self.solver.get_reference_trajectory()
    LOG_INFO(f"  Reference trajectory length: {len(reference_trajectory.get_states()) if hasattr(reference_trajectory, 'get_states') else 'unknown'}")
    
    # Log trajectory details for diagnosis
    if hasattr(reference_trajectory, 'get_states'):
      traj_states = reference_trajectory.get_states()
      if traj_states and len(traj_states) > 0:
        first_state = traj_states[0]
        last_state = traj_states[-1] if len(traj_states) > 1 else first_state
        first_x = first_state.get('x') if first_state.has('x') else None
        first_y = first_state.get('y') if first_state.has('y') else None
        last_x = last_state.get('x') if last_state.has('x') else None
        last_y = last_state.get('y') if last_state.has('y') else None
        LOG_INFO(f"  Trajectory: start=({first_x:.2f}, {first_y:.2f})" + 
                 (f", end=({last_x:.2f}, {last_y:.2f})" if last_x is not None and last_y is not None else ""))
        LOG_DEBUG(f"  Trajectory has {len(traj_states)} states (horizon + 1)")
        # Log first few trajectory points
        for i in range(min(3, len(traj_states))):
          state_i = traj_states[i]
          x_i = state_i.get('x') if state_i.has('x') else None
          y_i = state_i.get('y') if state_i.has('y') else None
          v_i = state_i.get('v') if state_i.has('v') else None
          spline_i = state_i.get('spline') if state_i.has('spline') else None
          # Log control inputs for first state (k=0)
          if i == 0:
            u_a = state_i.get('a') if state_i.has('a') else None
            u_w = state_i.get('w') if state_i.has('w') else None
            LOG_DEBUG(f"  Traj[{i}]: x={x_i:.2f}, y={y_i:.2f}, v={v_i:.2f}" + 
                     (f", spline={spline_i:.3f}" if spline_i is not None else "") +
                     (f", u=(a={u_a:.3f}, w={u_w:.3f})" if u_a is not None and u_w is not None else ""))
          else:
            LOG_DEBUG(f"  Traj[{i}]: x={x_i:.2f}, y={y_i:.2f}, v={v_i:.2f}" + 
                     (f", spline={spline_i:.3f}" if spline_i is not None else ""))
    
    self.output.trajectory_history.append(reference_trajectory)

    if self.output.success and self.config.get("planner", {}).get("debug_limits", False):
      LOG_DEBUG("Checking variable bounds...")
      self.solver.print_if_bound_limited()

    # Extract first-step control from solver so caller can apply it
    # CRITICAL: This is the control input at k=0 (first time step) that should be applied
    LOG_INFO("Extracting first control input (k=0) from optimal trajectory...")
    try:
      control_out = {}
      if hasattr(self.solver, 'dynamics_model') and self.solver.dynamics_model is not None:
        input_vars = self.solver.dynamics_model.get_inputs()
        LOG_DEBUG(f"  Looking for control inputs: {input_vars}")
        LOG_DEBUG(f"  Extracting control at k=0 (first time step of horizon)")
        for u_name in input_vars:
          # Get control input at k=0 (first time step)
          # Input variables have shape [horizon], so k=0 is the first control
          val0 = self.solver.get_output(0, u_name)
          if val0 is not None:
            control_out[u_name] = float(val0)
            LOG_DEBUG(f"  Extracted u[{u_name}][0] = {float(val0):.4f}")
            # Log w specifically for debugging turning
            if u_name == 'w':
              LOG_INFO(f"  ⚠️  Angular velocity w[0] = {float(val0):.4f} rad/s (vehicle should turn if non-zero)")
          else:
            LOG_WARN(f"  Could not extract {u_name} from solver (returned None)")
            # Optional fallback: extract from trajectory if enabled in config
            if self.config.get("planner", {}).get("fallback_control_enabled", False):
              if hasattr(reference_trajectory, 'get_states') and len(reference_trajectory.get_states()) > 0:
                first_state = reference_trajectory.get_states()[0]
                if first_state.has(u_name):
                  fallback_val = first_state.get(u_name)
                  control_out[u_name] = float(fallback_val)
                  LOG_DEBUG(f"  Fallback: extracted {u_name} from trajectory state: {float(fallback_val):.4f}")
      # Store on output if anything was found
      if control_out:
        self.output.control = control_out
        LOG_INFO(f"  ✓ First control input extracted: {control_out}")
        LOG_INFO(f"  This control will be applied to transition from current state to next state")
        
        # Log detailed control values for debugging vehicle movement
        if 'a' in control_out:
          LOG_INFO(f"    Acceleration a[0] = {control_out['a']:.4f} m/s²")
        if 'w' in control_out:
          LOG_INFO(f"    Angular velocity w[0] = {control_out['w']:.4f} rad/s")
          if abs(control_out['w']) < 1e-6:
            LOG_WARN(f"    ⚠️  Angular velocity is near zero - vehicle may not turn")
        if 'a' in control_out and abs(control_out['a']) < 1e-6:
          LOG_WARN(f"    ⚠️  Acceleration is near zero - vehicle may not move forward")
      else:
        LOG_WARN("  ⚠️  No control extracted from solver - all inputs returned None!")
        LOG_WARN("  ⚠️  Vehicle will NOT move - solver may have failed or solution is invalid")
    except Exception as e:
      LOG_WARN(f"Error extracting control: {e}")
      import traceback
      LOG_DEBUG(f"Traceback: {traceback.format_exc()}")

    LOG_INFO("Planner.solve_mpc() completed successfully")
    return self.output

  def get_symbolic_dynamics(self, dynamics_model, x, u, timestep, data=None):
    """
    Provide symbolic next-state for the solver.
    - If the model exposes symbolic_dynamics(x,u,p,timestep), use it.
    - Otherwise, default to Euler: x_next = x + dt * f(x,u,p), where f comes from continuous_model.
    The parameter accessor p is provided as a callable if data has parameters.
    """
    # Parameter getter callable (modules expect callable get)
    def _param_getter(key):
      try:
        if data is not None and hasattr(data, 'parameters'):
          return data.parameters.get(key)
      except Exception:
        pass
      return 0.0

    # Prefer model-provided symbolic dynamics if available
    if hasattr(dynamics_model, 'symbolic_dynamics') and callable(getattr(dynamics_model, 'symbolic_dynamics')):
      try:
        return dynamics_model.symbolic_dynamics(x, u, _param_getter, timestep)
      except Exception:
        # Fall back to Euler if model symbolic fails
        pass

    # Default Euler step using model's continuous dynamics
    f = dynamics_model.continuous_model(x, u, _param_getter)
    try:
      # x and f are casadi symbols; ensure same shape
      return x + timestep * f
    except Exception:
      # As last resort, return x (no motion) to avoid crash
      return x

  def get_numeric_dynamics(self, dynamics_model, x, u, timestep, data=None):
    """Numeric next-state using Euler: x_next = x + dt * f(x,u,p)."""
    def _param_getter(key):
      try:
        if data is not None and hasattr(data, 'parameters'):
          return data.parameters.get(key)
      except Exception:
        pass
      return 0.0
    try:
      f = dynamics_model.continuous_model(x, u, _param_getter)
      return x + timestep * f
    except Exception as e:
      LOG_DEBUG(f"get_numeric_dynamics failed: {e}")
      return x

  def reset(self):

    self.solver.reset()
    self.state.reset()
    self.data.reset()

  def is_objective_reached(self, data):
    # Only check modules that have is_objective_reached method (e.g., GoalObjective, ContouringObjective)
    # Other objectives like ControlEffortObjective don't have this method
    objective_modules = [module for module in self.solver.module_manager.modules if module.module_type == OBJECTIVE]
    reached_results = []
    for module in objective_modules:
      if hasattr(module, 'is_objective_reached'):
        try:
          reached = module.is_objective_reached(self.state, data)
          reached_results.append(reached)
        except Exception as e:
          LOG_DEBUG(f"Error checking objective reached for {module.name}: {e}")
          # If a module can't check, don't block goal reaching
          continue
    # Goal is reached if at least one objective module says it's reached
    # (typically only GoalObjective or ContouringObjective will have this method)
    return any(reached_results) if reached_results else False

  def visualize(self, stage_idx=1):
    LOG_DEBUG("Planner::visualize")
    for module in self.solver.module_manager.modules:
      if hasattr(module, 'get_visualizer'):
        visualizer = module.get_visualizer()
        if visualizer and hasattr(visualizer, 'visualize'):
          visualizer.visualize(self.state, self.data, stage_idx)

  def get_state(self):
    return self.state

  def set_state(self, state):
    self.state = state
    LOG_DEBUG("Planner::set_state")


class PlannerOutput:
  def __init__(self):
    self.success = False
    self.control_history = [] # The control history is a list of the actual control inputs used during the execution
    self.trajectory_history = [] # This is a list of horizon length trajectories which is added to each time the solver finds a solution
    self.realized_trajectory = Trajectory() # this is the trajectory executed by the physical robot when integrating based on the control history
