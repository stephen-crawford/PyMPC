from venv import logger

from planner.src.data_prep import ensure_obstacle_size
from utils.utils import read_config_file

CONFIG = read_config_file()


class ROSNavigationPlanner:
  def __init__(self, name, buffer, costmap):
    self.name = name
    self.buffer = buffer
    self.costmap = costmap
    self.initialized = False


  def initialize(self, name, buffer, costmap_ros):
    if not self.initialized:
      self.node_handle = node_handle("~/" + name)

      self.buffer = buffer

      costmap_ros_ = costmap_ros
      costmap_ = costmap_ros_.getCostmap()
      self.costmap = costmap_

      self.initialized_ = True

      LOG_DEBUG( "Started ROSNavigation Planner")

      VISUALS.init(general_node_handler)

      # Initialize the configuration
      get_configuration_instance().initialize(SYSTEM_CONFIG_PATH(__FILE__, "settings"))

      _data.robot_area = Disc(0., CONFIG["n_discs"])

      # Initialize the planner
      _planner = make_unique(Planner)

      # Initialize the ROS interface
      initializeSubscribersAndPublishers(node_handler)

      start_environment()

      _reconfigure = make_unique(ROSNavigationReconfigure)

      self_timeout_timer.set_duration(60.)
      self._timeout_timer.start()
      for i in range(CAMERA_BUFFER):

        _x_buffer[i] = 0.
        _y_buffer[i] = 0.


      RosTools.Instrumentor.get().begin_session("mpc_planner_rosnavigation")

      LOG_DIVIDER()

  def exit(self):

    logger.log("Stopped ROSNavigation Planner")
    BENCHMARKERS.print()

    RosTools.Instrumentor.get().end_sssion()

  def set_plan(self, orig_global_plan):
    # check if plugin is initialized
    if not self.initialized:

      LOG_DEBUG( "ROS planner has not been initialized, please call initialize() before using this planner")
      return False


    # store the global plan
    global_plan_.clear()
    global_plan_ = orig_global_plan

    # we do not clear the local planner here, since setPlan is called frequently whenever the global planner updates the plan.
    # the local planner checks whether it is required to reinitialize the trajectory or not within each velocity computation step.

    # reset goal_reached_ flag
    # goal_reached_ = False

    return True

  def compute_velocity_commands(self, cmd_vel):
    if not self.initialized:
      LOG_DEBUG( "This planner has not been initialized")
      return False

    path = make_shared(Path)
    path.poses = global_plan_
    path_callback(path)

    if (_rotate_to_goal):
      rotateToGoal(cmd_vel)
    else:
      loop(cmd_vel)

    return True


  def initialize_subscribers_and_publishers(self, node_handler):
    LOG_DEBUG( "initializeSubscribersAndPublishers")

    _state_sub = node_handler.subscribe("/input/state", 5, bind(state_callback, self, _1))

    _state_pose_sub = node_handler.subscribe("/input/state_pose", 5, bind(state_pose_callback, self, _1))

    _goal_sub = node_handler.subscribe("/input/goal", 1, bind(goal_callback, self, _1))

    _path_sub = node_handler.subscribe("/input/reference_path", 1, bind(path_callback, self, _1))

    _obstacle_sim_sub = node_handler.subscribe("/input/obstacles", 1, bind(obstacleCallback, self, _1))

    _cmd_pub = node_handler.advertise("/output/command", 1)

    _pose_pub = node_handler.advertise("/output/pose", 1)

    _collisions_sub = node_handler.subscribe("/feedback/collisions", 1, bind(collision_callback, self, _1))

    # Environment Reset
    _reset_simulation_pub = node_handler.advertise("/lmpcc/reset_environment", 1)
    _reset_simulation_client = node_handler.service_client("/gazebo/reset_world")
    _reset_ekf_client = node_handler.service_client("/set_pose")

    # Pedestrian simulator
    _ped_horizon_pub = node_handler.advertise("/pedestrian_simulator/horizon", 1)
    _ped_integrator_step_pub = node_handler.advertise("/pedestrian_simulator/integrator_step", 1)
    _ped_clock_frequency_pub = node_handler.advertise("/pedestrian_simulator/clock_frequency", 1)
    _ped_start_client = node_handler.service_client("/pedestrian_simulator/start")

  def start_environment(self):

    # Manually add obstacles in the costmap!
    # int mx, my
    # costmap_.worldToMapEnforceBounds(2., 2., mx, my)
    # LOG_VALUE("mx", mx)
    # LOG_VALUE("my", my)

    # for (int i = 0 i < 10 i++)
    # {
    #   costmap_.setCost(mx + i, my, costmap_2d::LETHAL_OBSTACLE)
    # }

    LOG_DEBUG( "Starting pedestrian simulator")
    i = 0
    while i < 20:
      horizon_msg = None
      horizon_msg.data = CONFIG["N"]
      _ped_horizon_pub.publish(horizon_msg)

      integrator_step_msg = None
      integrator_step_msg.data = CONFIG["integrator_step"]
      _ped_integrator_step_pub.publish(integrator_step_msg)

      clock_frequency_msg = None
      clock_frequency_msg.data = CONFIG["control_frequency"]
      _ped_clock_frequency_pub.publish(clock_frequency_msg)

      empty_msg = None
      if (_ped_start_client.call(empty_msg)):
        break
      else:
        logger.log(3, "Waiting for pedestrian simulator to start")
        ros.duration(1.0).sleep()
        _reset_simulation_pub.publish(std_msgs.empty())
    _enable_output = CONFIG["enable_output"]
    logger.log10, ("Environment ready.")

  def is_goal_reached(self):

    if not self.initialized:
      LOG_DEBUG( "This planner has not been initialized")
      return False

    goal_reached = _planner.is_objective_reached(_state, _data) and not done_ # Activate once
    if (goal_reached):
      LOG_DEBUG( "Goal Reached!")
      done_ = True
      self.reset()

    return goal_reached

  def rotate_to_goal(self, cmd_vel):
    LOG_DEBUG( "Rotating to the goal")
    if not _data.goal_received:
      LOG_DEBUG( "Waiting for the goal")
      return
    goal_angle = 0.

    if _data.reference_path.x.size() > 2:
      goal_angle = arctan2(_data.reference_path.y[2] - _state.get("y"), _data.reference_path.x[2] - _state.get("x"))
    else:
      goal_angle = arctan2(_data.goal(1) - _state.get("y"), _data.goal(0) - _state.get("x"))

    angle_diff = goal_angle - _state.get("psi")

    if angle_diff > M_PI:
      angle_diff -= 2 * M_PI


    if abs(angle_diff) > M_PI / 4.:
      cmd_vel.linear.x = 0.0
      if _enable_output:
        cmd_vel.angular.z = 1.5 * RosTool.sgn(angle_diff)
      else:
        cmd_vel.angular.z = 0.

    else:
      LOG_DEBUG( "Robot rotated and is ready to follow the path")
      _rotate_to_goal = False

  def loop(self, cmd_vel):
  
    # Copy data for thread safety
    data = self._data
    state = self._state

    data.planning_start_time = system_clock.now()

    LOG_DEBUG( "============= Loop =============")

    if _timeout_timer.hasFinished():
    
      reset(False)
      cmd_vel.linear.x = 0.
      cmd_vel.angular.z = 0.
      return
    

    if CONFIG["debug_output"]:
      state.print()

    loop_benchmarker = BENCHMARKERS.getBenchmarker("loop")
    loop_benchmarker.start()

    output = self._planner.solve_mpc(state, data)

    LOG_MARK("Success: " + output.success)

    if self._enable_output and output.success:
      # Publish the command
      cmd_vel.linear.x = _planner.get_solution(1, "v") # = x1
      cmd_vel.angular.z = _planner.get_solution(0, "w") # = u0
      LOG_DEBUG( "Commanded v: " + cmd.linear.x)
      LOG_VALUE_DEBUG(10, "Commanded w: " + cmd.angular.z)
    else:
      deceleration = CONFIG["deceleration_at_infeasible"]
      dt = 1. / CONFIG["control_frequency"]

      velocity = self._state.get("v")
      velocity_after_braking = velocity - deceleration * dt  # Brake with the given deceleration
      cmd_vel.linear.x = max(velocity_after_braking, 0.) # Don't drive backwards when braking
      cmd_vel.angular.z = 0.0
    _cmd_pub.publish(cmd)

    publish_pose()
    publish_camera()

    loop_benchmarker.stop()

    if CONFIG["recording"]["enable"]:
      # Save control inputs
      if (output.success):
        data_saver = self._planner.get_data_saver()
        data_saver.AddData("input_a", state.get("a"))
        data_saver.AddData("input_v", self._planner.get_solution(1, "v"))
        data_saver.AddData("input_w", self._planner.get_solution(0, "w"))
    
      self._planner.save_data(state, data)
    if output.success:
  
      self._planner.visualize(state, data)
      visualize()
    
    LOG_DEBUG( "============= End Loop =============")

  def state_callback(self, msg):
  
    LOG_DEBUG( "State callback")
    self._state.set("x", msg.pose.pose.position.x)
    self._state.set("y", msg.pose.pose.position.y)
    self._state.set("psi", RosTools.quaternion_to_angle(msg.pose.pose.orientation))
    self._state.set("v", sqrt(pow(msg.twist.twist.linear.x, 2.) + pow(msg.twist.twist.linear.y, 2.)))

    if (abs(msg.pose.pose.orientation.x) > (M_PI / 8.) or abs(msg.pose.pose.orientation.y) > (M_PI / 8.)):

      LOG_DEBUG( "Detected flipped robot. Resetting.")
      reset(False) # Reset without success

  def state_pose_callback(self, msg):
    LOG_DEBUG( "State callback")

    self._state.set("x", msg.pose.position.x)
    self._state.set("y", msg.pose.position.y)
    self._state.set("psi", msg.pose.orientation.z)
    self._state.set("v", msg.pose.position.z)

    if (abs(msg.pose.orientation.x) > (M_PI / 8.) or abs(msg.pose.orientation.y) > (M_PI / 8.)):
      LOG_DEBUG( "Detected flipped robot. Resetting.")
      reset(False) # Reset without success
  

  def goal_callback(self, msg):
    LOG_DEBUG( "Goal callback")

    self._data.goal(0) = msg.pose.position.x
    self._data.goal(1) = msg.pose.position.y
    self._data.goal_received = True

    _rotate_to_goal = True

  def is_path_the_same(self, msg):
    # Check if the path is the same
    if self._data.reference_path.x.size() != msg.poses.size():
      return False

    # Check up to the first two points
    num_points = min(2, self._data.reference_path.x.size())
    for i in range(num_points):
    
      if not self._data.reference_path.point_in_path(i, msg.poses[i].pose.position.x, msg.poses[i].pose.position.y)):
        return False
    
    return True

  def path_callback(self, msg):
    LOG_DEBUG( "Path callback")

    downsample = CONFIG["downsample_path"]

    if is_the_same_path(msg) or msg.poses.size() < downsample + 1:
      return

    self._data.reference_path.clear()

    int count = 0
    for pose in msg.poses:
      if count % downsample == 0 or count == msg.poses.size() - 1: # Todo
        self._data.reference_path.x.push_back(pose.pose.position.x)
        self._data.reference_path.y.push_back(pose.pose.position.y)
        self._data.reference_path.psi.push_back(RosTools.quaternion_to_angle(pose.pose.orientation))
      
      count+=1

    # Fit a clothoid on the global path to sample points on the spline from
    # RosTools::Clothoid2D clothoid(_data.reference_path.x, _data.reference_path.y, _data.reference_path.psi, 2.0)
    # _data.reference_path.clear()
    # clothoid.getPointsOnClothoid(_data.reference_path.x, _data.reference_path.y, _data.reference_path.s)

    # Velocity
    # LOG_VALUE("velocity reference", CONFIG["weights"]["reference_velocity"])
    # for i in range(self._data.reference_path.x.size()):
    # 
    #   if (i != self._data.reference_path.x.size() - 1):
    #     self._data.reference_path.v.push_back(CONFIG["weights"]["reference_velocity"])
    #   else:
    #     self._data.reference_path.v.push_back(0.)
    # }

    self._planner.on_data_received(_data, "reference_path")

  def obstacle_callback(self, msg):
    LOG_DEBUG( "Obstacle callback")

    self._data.dynamic_obstacles.clear()

    for obstacle in msg.obstacles:
  
      # Save the obstacle
      self._data.dynamic_obstacles.emplace_back(obstacle.id, (obstacle.pose.position.x, obstacle.pose.position.y), RosTools.quaternion_to_angle(obstacle.pose), CONFIG["obstacle_radius"])
      dynamic_obstacle = self._data.dynamic_obstacles.back()

      if (obstacle.probabilities.size() == 0): # No Predictions!
        continue

      # Save the prediction
      if (obstacle.probabilities.size() == 1): # One mode
        dynamic_obstacle.prediction = Prediction(GAUSSIAN)

        mode = obstacle.gaussians[0]
        for k in range(mode.mean.poses.size()):
          dynamic_obstacle.prediction.modes[0].emplace_back((mode.mean.poses[k].pose.position.x, mode.mean.poses[k].pose.position.y), RosTools.quaternion_to_angle(mode.mean.poses[k].pose.orientation), mode.major_semiaxis[k], mode.minor_semiaxis[k])

        if (mode.major_semiaxis.back() == 0. or not CONFIG["probabilistic"]["enable"]):
          dynamic_obstacle.prediction.type = DETERMINISTIC
        else:
          dynamic_obstacle.prediction.type = GAUSSIAN
      
      else:
        PYMPC_ASSERT(False, "Multiple modes not yet supported")
    
    ensure_obstacle_size(self._data.dynamic_obstacles, self._state)

    if (CONFIG["probabilistic"]["propagate_uncertainty"]):
      propogate_prediction_uncertainty(self._data.dynamic_obstacles)

    self._planner.on_data_received(self._data, "dynamic obstacles")

  def visualize(self):
    publisher = VISUALS.get_publisher("angle")
    line = publisher.get_new_line()

    line.add_line((_state.get("x"), _state.get("y")), (_state.get("x") + 1.0 * cos(_state.get("psi")), _state.get("y") + 1.0 * sin(_state.get("psi"))))
    publisher.publish()

  def reset(self, success):
    LOG_DEBUG( "Resetting")
    l(_reset_mutex)

    _reset_simulation_client.call(_reset_msg)
    _reset_ekf_client.call(_reset_pose_msg)
    _reset_simulation_pub.publish(empty_message())

    for i in range(CAMERA_BUFFER):
      _x_buffer[i] = 0.
      _y_buffer[i] = 0.


    _planner.reset(_state, _data, success)
    _data.costmap = costmap_

    Duration(1.0 / CONFIG["control_frequency"]).sleep()

    done_ = False
    _rotate_to_goal = False

    _timeout_timer.start()

  def collision_callback(self, msg):
    LOG_DEBUG( "Collision callback")

    _data.intrusion = (float)(msg.data)

    if (_data.intrusion > 0.)
      LOG_DEBUG( "Collision detected (Intrusion: " + _data.intrusion + ")")

  def publish_pose(self):
    pose.pose.position.x = _state.get("x")
    pose.pose.position.y = _state.get("y")
    pose.pose.orientation = RosTools.angle_to_quaternion(_state.get("psi"))

    pose.header.stamp = ros::Time::now()
    pose.header.frame_id = "map"

    _pose_pub.publish(pose)

  def publish_camera(self):
    msg.header.stamp = ros.Time.now()

    if ((msg.header.stamp - _prev_stamp) < ros::Duration(0.5 / CONFIG["control_frequency"])):
      return

    _prev_stamp = msg.header.stamp

    msg.header.frame_id = "map"
    msg.child_frame_id = "camera"

    # Smoothen the camera
    for i in range(CAMERA_BUFFER - 1):
      _x_buffer[i] = _x_buffer[i + 1]
      _y_buffer[i] = _y_buffer[i + 1]
    _x_buffer[CAMERA_BUFFER - 1] = _state.get("x")
    _y_buffer[CAMERA_BUFFER - 1] = _state.get("y")
    camera_x = 0.
    camera_y = 0.
    for i in range(CAMERA_BUFFER):
      camera_x += _x_buffer[i]
      camera_y += _y_buffer[i]
    msg.transform.translation.x = camera_x / CAMERA_BUFFER #_state.get("x")
    msg.transform.translation.y = camera_y / CAMERA_BUFFER #_state.get("y")
    msg.transform.translation.z = 0.0
    msg.transform.rotation.x = 0
    msg.transform.rotation.y = 0
    msg.transform.rotation.z = 0
    msg.transform.rotation.w = 1

    _camera_pub.send_transform(msg)
