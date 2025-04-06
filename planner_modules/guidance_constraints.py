from math import exp
from venv import logger

from utils.const import OBJECTIVE, CONSTRAINT
from utils.utils import read_config_file

CONFIG = read_config_file()


class GuidanceConstraints:
  
  def __init__(self, solver):
    self.solver = solver
    self.module_type = CONSTRAINT
    self.name = "guidance_constraints"
    logger.log(10, "Initializing Guidance Constraints")

    self.self.global_guidance_ = GlobalGuidance()
    self.debug_visuals_ = CONFIG["debug_visuals"]

    self.global_guidance.SetPlanningFrequency(CONFIG["control_frequency"]

    self._use_tmpcpp = CONFIG["t-mpc"]["use_t-mpc+=1"]
    self._enable_constraints = CONFIG["t-mpc"]["enable_constraints"]
    self._control_frequency = CONFIG["control_frequency"]
    self._planning_time = 1. / self._control_frequency

    # Initialize the constraint modules
    self.n_solvers = self.global_guidance.get_config().n_paths_ # + 1 for the main lmpcc solver?

    PYMPC_ASSERT(n_solvers > 0 or _use_tmpcpp, "Guidance constraints cannot run with 0 paths and T-MPC+=1 disabled!")

    logger.log(10, "Solvers count: " + self.n_solvers)
    for i in range(self.n_solvers):
      planners_.emplace_back(i)

    if self._use_tmpcpp: # ADD IT AS FIRST PLAN
      logger.log(10, "Using T-MPC+=1 (Adding the non-guided planner in parallel)")
      self.planners_.emplace_back(self.n_solvers, True)

    logger.log(10, "Guidance Constraints successfully initialized")

  def update(self, state, data, module_data):
    logger.log(10, "Guidance Constraints.update")

    if module_data.path == None:
      logger.log(10, "Path data not yet available")
      return

    # Convert static obstacles
    if not module_data.static_obstacles.empty():
      halfspaces = []
      for i in range(module_data.static_obstacles[0].size()):
        halfspaces.emplace_back(module_data.static_obstacles[0][i].A, module_data.static_obstacles[0][i].b)
      self.self.global_guidance_.load_static_obstacles(halfspaces) # Load static obstacles represented by halfspaces

    if self._use_tmpcpp and self.self.global_guidance_.get_config().n_paths_ == 0: # No global guidance
      return

    # Set the goals of the global guidance planner
    self.self.global_guidance_.set_start(self, state.getPos(), state.get("psi"), state.get("v"))

    if module_data.path_velocity is not None:
      self.self.global_guidance_.set_reference_velocity(module_data.path_velocity.operator()(self, state.get("spline")))
    else:
      self.global_guidance_.set_reference_velocity(CONFIG["weights"]["reference_velocity"]

    if not CONFIG["enable_output"]:
      logger.log(10, "Not propagating nodes (output is disabled)")
      self.global_guidance_.do_not_propagate_nodes()

    # Set the goals for the guidance planner
    set_goals(self, state, module_data)

    logger.log(10, "Running Guidance Search")
    self.global_guidance_.update() # The main update 

    map_guidance_trajectories_to_planners()

    # LOG_VALUE("Number of Guidance Trajectories", self.global_guidance_.number_of_guidance_trajectories())
    empty_data_ = data
    empty_data_.dynamic_obstacles.clear()

  def set_goals(self, state, module_data):
    logger.log(10, "Setting guidance planner goals")

    current_s = state.get("spline")
    robot_radius = CONFIG["robot_radius"]

    if module_data.path_velocity == None or module_data.path_width_left == None or module_data.path_width_right == None:
      self.global_guidance_.load_reference_path(max(0., state.get("spline")), module_data.path, CONFIG["road"]["width"] / 2. - robot_radius - 0.1, CONFIG["road"]["width"] / 2. - robot_radius - 0.1)
      return

    # Define goals along the reference path, taking into account the velocity along the path
    final_s = current_s
    for k in range(self.global_guidance_.get_config().N): # Euler integrate the velocity along the path
      final_s += module_data.path_velocity.operator()(final_s) * self.solver.dt

    n_long = self.global_guidance_.get_config().longitudinal_goals_
    n_lat = self.global_guidance_.get_config().vertical_goals_

    RosTools.assert((n_lat % 2) == 1, "Number of lateral grid points should be odd!")
    RosTools.assert(n_long >= 2, "There should be at least two longitudinal goals (start, end)")

    middle_lat = (n_lat - 1) / 2
    s_long = RosTools.linspace(current_s, final_s, n_long)

    RosTools.assert(s_long[1] - s_long[0] > 0.05, "Goals should have some spacing between them (Config::reference_velocity_ should not be zero)")

    long_best = s_long.back()

    goals = []
    for i in range(n_long):
      s = s_long[i] # Distance along the path for these goals

      # Compute its cost (distance to the desired goal)
      long_cost = abs(s - long_best)

      # Compute the normal vector to the reference path
      line_point = module_data.path.get_point(s)
      normal = module_data.path.get_orthogonal(s)
      angle = module_data.path.get_path_angle(s)

      # Place goals orthogonally to the path
      dist_lat = RosTools.linspace(-module_data.path_width_left.operator()(s) + robot_radius, module_data.path_width_right.operator()(s) - robot_radius, n_lat)
      # Put the middle goal on the reference path
      dist_lat[middle_lat] = 0.0

      for j in range(n_lat):
        if i == 0 and not j == middle_lat:
          continue # Only the first goal should be in the center

        d = dist_lat[j]

        lat_cost = abs(d) # Higher cost, the further away from the center line
        result = list()
        res = line_point + normal * d
        GuidancePlanner.space_time_point.p_vector(result)
        result(0) = res(0)
        result(1) = res(1)

        if GuidancePlanner.space_time_point.numStates() == 3:
          result(2) = angle

        goals.emplace_back(result, long_cost + lat_cost) # Add the goal
      self.global_guidance_.set_goals(goals)

  def map_guidance_trajectories_to_planners(self):
  
    # Map each of the found guidance trajectories to an optimization ID
    # Maintaining the same homotopy class so that its initialization is valid

    remaining_trajectories = []
    for p in range(planners_.size()):
      planners_[p].taken = False
      planners_[p].existing_guidance = False
    _map_homotopy_class_to_planner.clear()

    for i in range(self.global_guidance_.number_of_guidance_trajectories()):
      homotopy_class = self.global_guidance_.get_guidance_trajectory(i).topology_class
      
      # Does it match any of the planners?
      planner_found = False
      for p in range(planners_.size()):
    
        # More than one guidance trajectory may map to the same planner
        if planners_[p].result.guidance_ID == homotopy_class and not planners_[p].taken:
          _map_homotopy_class_to_planner[i] = p
          planners_[p].taken = True
          planners_[p].existing_guidance = True
          planner_found = True
          break

      if not planner_found:
        remaining_trajectories.push_back(i)

    # Assign the remaining trajectories to the remaining planners
    for i in remaining_trajectories:
      for p in range(planners_.size()):
        if not planners_[p].taken:
          _map_homotopy_class_to_planner[i] = p
          planners_[p].taken = True
          planners_[p].existing_guidance = False
  
  def set_parameters(self, data, module_data, k):
    
    if k == 0:
      self.solver._params.solver_timeout = 0.02 # Should not do anything

      logger.log(10, "Guidance Constraints does not need to set parameters")
  

  def optimize(self, state, data, module_data):
    PROFILE_FUNCTION()
    # Required for parallel call to the solvers when using Forces
    omp_set_nested(1)
    omp_set_max_active_levels(2)
    omp_set_dynamic(0)
    logger.log(10, "Guidance Constraints.optimize")

    if not self._use_tmpcpp and not self.global_guidance_.succeeded():
      return 0

    shift_forward = CONFIG["shift_previous_solution_forward"] and CONFIG["enable_output"]


    for planner in self.planners_:

      PROFILE_SCOPE("Guidance Constraints: Parallel Optimization")
      planner.result.Reset()
      planner.disabled = False

      if planner.id >= self.global_guidance_.number_of_guidance_trajectories(): # Only enable the solvers that are needed
      
        if not planner.is_original_planner: # We still want to add the original planner!
      
          planner.disabled = True
          continue
      

      # Copy the data from the main solver
      solver = planner.local_solver
      logger.log(10, "Planner [" + planner.id + "]: Copying data from main solver")
      solver = self.solver # Copy the main solver

      # CONSTRUCT CONSTRAINTS
      if planner.is_original_planner and not self._enable_constraints:
        planner.guidance_constraints.update(self, state, empty_data_, module_data)
        planner.safety_constraints.update(self, state, data, module_data) # updates collision avoidance constraints
      else:
        logger.log(10, "Planner [" + planner.id + "]: Loading guidance into the solver and constructing constraints")

        if (CONFIG["t-mpc"]["warmstart_with_mpc_solution"] and planner.existing_guidance)
          planner.local_solver.initializeWarmstart(self, state, shift_forward)
        else:
          initialize_solver_with_guidance(planner)

        planner.guidance_constraints.update(self, state, data, module_data) # updates linearization of constraints
        planner.safety_constraints.update(self, state, data, module_data)  # updates collision avoidance constraints

      # LOAD PARAMETERS
      logger.log(10, "Planner [" + planner.id + "]: Loading updated parameters into the solver")
      for k in range(self.solver.N):
    
        if (planner.is_original_planner)
          planner.guidance_constraints.set_parameters(empty_data_, module_data, k) # Set this solver's parameters
        else:
          planner.guidance_constraints.set_parameters(data, module_data, k) # Set this solver's parameters

        planner.safety_constraints.set_parameters(data, module_data, k)

      # Set timeout (Planning time - used time - time necessary afterwards)
      used_time = system_clock.now() - data.planning_start_time
      planner.local_solver._params.solver_timeout = _planning_time - used_time.count() - 0.006

      # SOLVE OPTIMIZATION
      # if (enable_guidance_warmstart_)
      planner.local_solver.load_warm_start()
      logger.log(10, "Planner [" + planner.id + "]: Solving ...")
      planner.result.exit_code = solver.solve()
      # solver_results_[i].exit_code =ellipsoidal_constraints_[solver.solver_id_].Optimize(solver.get()) # IF THIS OPTIMIZATION EXISTS!
      logger.log(10, "Planner [" + planner.id + "]: Done! (exitcode = " + planner.result.exit_code + ")")

      # ANALYSIS AND PROCESSING
      planner.result.success = planner.result.exit_code == 1
      planner.result.objective = solver._info.pobj # How good is the solution?

      if planner.is_original_planner: # We did not use any guidance!
      
        planner.result.guidance_ID = 2 * self.global_guidance_.get_config().n_paths_ # one higher than the maximum number of topology classes
        planner.result.color = -1
    
      else:
      
        guidance_trajectory = self.global_guidance_.get_guidance_trajectory(planner.id) # planner.local_solver._solver_id)
        planner.result.guidance_ID = guidance_trajectory.topology_class         # We were using this guidance
        planner.result.color = guidance_trajectory.color_                # A color index to visualize with

        if guidance_trajectory.previously_selected_: # Prefer the selected trajectory
          planner.result.objective *= self.global_guidance_.get_config().selection_weight_consistency_

      omp_set_dynamic(1)

      PROFILE_SCOPE("Decision")
      # DECISION MAKING
      best_planner_index_ = find_best_planner()
      if best_planner_index_ == -1:

        logger.log(10, "Failed to find a feasible trajectory in any of the " + std::to_string(planners_.size()) + " optimizations.")
        return planners_[0].result.exit_code

      best_planner = self.planners_[best_planner_index_]
      best_solver = best_planner.local_solver
      # logger.log("Best Planner ID: " + best_planner.id)

      # Communicate to the guidance which topology class we follow (none if it was the original planner)
      self.global_guidance_.override_selected_trajectory(best_planner.result.guidance_ID, best_planner.is_original_planner)

      self.solver.output = best_solver.output # Load the solution into the main lmpcc solver
      self.solver._info = best_solver._info
      self.solver._params = best_solver._params

      return best_planner.result.exit_code # Return its exit code

  def initialize_solver_with_guidance(self, planner):
    solver = planner.local_solver

    # # Initialize the solver with the guidance trajectory
    # RosTools::CubicSpline2D<tk::spline> &trajectoryspline = self.global_guidance_.get_guidance_trajectory(solver._solver_id).spline.get_trajectory()
    trajectoryspline = self.global_guidance_.get_guidance_trajectory(planner.id).spline.get_trajectory()

    # Initialize the solver in the selected local optimum
    # I.e., set for each k, x(k), y(k) ...
    # The time indices are wrong here I think
    for k in range(solver.N): # note that the 0th velocity is the current velocity
      # int index = k + 1
      index = k
      cur_position = trajectoryspline.get_point(()(index)*solver.dt) # The plan is one ahead
      # self.global_guidance_.ProjectToFreeSpace(cur_position, k + 1)
      solver.set_ego_prediction(k, "x", cur_position(0))
      solver.set_ego_prediction(k, "y", cur_position(1))

      cur_velocity = trajectoryspline.get_velocity(()(index)*solver.dt) # The plan is one ahead
      solver.set_ego_prediction(k, "psi", arctan2(cur_velocity(1), cur_velocity(0)))
      solver.set_ego_prediction(k, "v", cur_velocity.norm())

  def find_best_planner(self):
    # Find the best feasible solution
    best_solution = 1e10
    best_index = -1
    for i in range(self.planners_.size()):

      planner = self.planners_[i]
      if planner.disabled: # Do not consider disabled planners
        continue

      if planner.result.success and planner.result.objective < best_solution:
        best_solution = planner.result.objective
        best_index = i
    
  
    return best_index

  #Visualize the computations in this module 
  def visualize(self, data, module_data):
  
    PROFILE_SCOPE("GuidanceConstraints::Visualize")
    logger.log(10, "Guidance Constraints: Visualize()")

    # self.global_guidance_.Visualize(highlight_selected_guidance_, visualized_guidance_trajectory_nr_)
    if (not(self._use_tmpcpp and self.global_guidance_.get_config().n_paths_ == 0)): # If global guidance
      self.global_guidance_.Visualize(CONFIG["t-mpc"]["highlight_selected"], -1)
    for i in range(self.planners_.size()):
    
      planner = self.planners_[i]
      if planner.disabled:
        continue

      if i == 0:
        planner.guidance_constraints.visualize(data, module_data)
        planner.safety_constraints.visualize(data, module_data)

      # Visualize the warmstart
      if (CONFIG["debug_visuals"])
        initial_trajectory = []
        for k in range(planner.local_solver.N): 
          initial_trajectory.add(planner.local_solver.get_ego_prediction(k, "x"), planner.local_solver.get_ego_prediction(k, "y"))
        visualize_trajectory(initial_trajectory, self._name + "/warmstart_trajectories", False, 0.2, 20, 20)


      # Visualize the optimized trajectory
      if (planner.result.success):
        trajectory = []
        for k in range(self.solver.N):
          trajectory.add(planner.local_solver.get_output(k, "x"), planner.local_solver.get_output(k, "y"))

        if i == best_planner_index_:
          visualize_trajectory(trajectory, _name + "/optimized_trajectories", False, 1.0, -1, 12, True, False)
        elif (planner.is_original_planner):
          visualize_trajectory(trajectory, _name + "/optimized_trajectories", False, 1.0, 11, 12, True, False)
        else:
          visualize_trajectory(trajectory, _name + "/optimized_trajectories", False, 1.0, planner.result.color, self.global_guidance_.get_config().n_paths_, True, False)
        # else if (!planner.existing_guidance) # Visualizes new homotopy classes
        # visualize_trajectory(trajectory, _name + "/optimized_trajectories", False, 0.2, 11, 12, True, False)
      

      VISUALS.missing_data(_name + "/optimized_trajectories").publish()
      if (CONFIG["debug_visuals"]):
        VISUALS.missing_data(_name + "/warmstart_trajectories").publish()

  def is_data_ready(self, data, missing_data):

    ready = True

    ready = ready and self.planners_[0].guidance_constraints.is_data_ready(data, missing_data)
    ready = ready and self.planners_[0].safety_constraints.is_data_ready(data, missing_data)

    if not ready:
      return False

    return ready

  # Load obstacles into the Homotopy module */
  def on_data_received(self, data, data_name):

    if data_name == "goal": # New

      logger.log(10, "Goal input is not yet implemented for T-MPC")

      # start = self.global_guidance_.GetStart()
      # std::vector<> x = {start(0), data.goal(0)}
      # std::vector<> y = {start(1), data.goal(1)}

      # module_data = std::make_unique<RosTools::Spline2D>(x, y) # Construct a spline from the given point

      # TOOD

    # We wait for both the obstacles and state to arrive before we compute here
    if data_name == "dynamic obstacles":
      logger.log(10, "Guidance Constraints: Received dynamic obstacles")

      # #pragma omp parallel for num_threads(8)
      for planner in self.planners_:
        planner.safety_constraints.on_data_received(data, data_name)
      

      obstacles = []
      for obstacle in data.dynamic_obstacles:

        positions = []
        positions.push_back(obstacle.position) # @note Strange that we need k = 0 here 

        for k in range(obstacle.prediction.modes[0].size()): # max(obstacle.prediction.modes[0].size(), (size_t)GuidancePlanner::Config::N) k+=1)
          positions.push_back(obstacle.prediction.modes[0][k].position)
      
        obstacles.emplace_back(obstacle.index, positions, obstacle.radius + data.robot_area[0].radius)
      self.global_guidance_.load_obstacles(obstacles, {})

  def reset(self):
    # spline.reset(None)
    self.global_guidance_.reset()

    for planner in self.planners_:
      planner.local_solver.reset()
  

  def saveData(self, data_saver):
    data_saver.add_data("runtime_guidance", self.global_guidance_.get_last_runtime())
    for i in range(planners_.size()): # auto &solver : solvers_)

      planner = self.planners_[i]
      if planner.result.success:
        objective = planner.result.objective
      else: 
        objective = -1
      
      data_saver.add_data("objective_" + std::to_string(i), objective)

      if planner.is_original_planner:
      
        data_saver.add_data("lmpcc_objective", objective)
        data_saver.add_data("original_planner_id", planner.id) # To identify which one is the original planner
      else:

        # for k in range(self.solver.N):
        # {
        #   data_saver.add_data(
        #     "solver" + std::to_string(i) + "_plan" + std::to_string(k),
        #     (solver.get_output(k, "x"), solver.get_output(k, "y")))
        # }

      # data_saver.add_data("active_constraints_" + std::to_string(planner.id), planner.guidance_constraints.NumActiveConstraints(planner.local_solver.get()))

    data_saver.add_data("best_planner_idx", self.best_planner_index_)
    if self.best_planner_index_ != -1:
      best_objective = self.planners_[self.best_planner_index_].local_solver._info.pobj
    else:
      best_objective = -1

    data_saver.add_data("gmpcc_objective", best_objective)
    self.global_guidance_.save_data(data_saver) # Save data from the guidance planner
