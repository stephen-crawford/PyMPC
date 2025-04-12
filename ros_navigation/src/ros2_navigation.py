#!/usr/bin/env python3

import numpy as np
import math
import threading
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.callback_groups import ReentrantCallbackGroup

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from std_msgs.msg import Empty, Float32, Float64, Int32
from std_srvs.srv import Empty as EmptySrv
from robot_localization.srv import SetPose
from tf2_ros import TransformBroadcaster
from nav2_core.controller import Controller  # Replacing nav_core::BaseLocalPlanner
from nav2_costmap_2d.costmap_2d import Costmap2DROS, Costmap2D  # Updated import path

from mpc_planner.planner import Planner
from mpc_planner.data_preparation import ensure_obstacle_size, propagate_prediction_uncertainty
from mpc_planner_util.parameters import Configuration
from mpc_planner_util.load_yaml import load_yaml_file
from mpc_planner_msgs.msg import ObstacleArray

from ros_tools.visuals import Visuals
from ros_tools.logging import LOG_INFO, LOG_DEBUG, LOG_SUCCESS, LOG_WARN, LOG_ERROR, LOG_MARK, LOG_VALUE_DEBUG, \
    LOG_DIVIDER
from ros_tools.convertions import quaternion_to_angle, angle_to_quaternion
from ros_tools.math import sgn
from ros_tools.data_saver import DataSaver
from ros_tools.spline import Clothoid2D
from ros_tools.profiling import Instrumentor, Benchmarkers

from mpc_planner.prediction import Prediction, PredictionType
from mpc_planner.state import State
from mpc_planner.real_time_data import RealTimeData
from mpc_planner.models import Disc

# Constants
CAMERA_BUFFER = 10
SYSTEM_CONFIG_PATH = lambda file_path, folder: f"{file_path}/{folder}"

# Global objects
CONFIG = None  # Will be loaded during initialization
VISUALS = Visuals()
BENCHMARKERS = Benchmarkers()


class Timer:
    def __init__(self):
        self.duration = 0.0
        self.start_time = 0.0

    def setDuration(self, duration):
        self.duration = duration

    def start(self):
        self.start_time = time.time()

    def hasFinished(self):
        return (time.time() - self.start_time) > self.duration


class ROS2NavigationPlanner(Controller):
    def __init__(self):
        self.costmap_ros_ = None
        self.initialized_ = False

        self._data = RealTimeData()
        self._state = State()
        self._planner = None
        self._reconfigure = None

        # Initialize buffers and variables
        self._x_buffer = [0.0] * CAMERA_BUFFER
        self._y_buffer = [0.0] * CAMERA_BUFFER
        self._prev_stamp = Time()
        self._timeout_timer = Timer()
        self._rotate_to_goal = False
        self._enable_output = False
        self.done_ = False
        self.global_plan_ = []

        self._reset_mutex = threading.Lock()
        self._callback_group = ReentrantCallbackGroup()

        # Node will be set during configure

    # ROS2 Controller interface method (replaces initialize)
    def configure(self, node, name, tf, costmap_ros):
        """
        Configure the controller with ROS2 interfaces
        """
        self.node = node
        self.logger = node.get_logger()
        self.tf_ = tf

        # Load config
        global CONFIG
        CONFIG = load_yaml_file(SYSTEM_CONFIG_PATH(__file__, "settings"))

        self.costmap_ros_ = costmap_ros
        self.costmap_ = costmap_ros.getCostmap()
        self._data.costmap = self.costmap_

        self.initialized_ = True

        self.logger.info("Started ROS2Navigation Planner")

        VISUALS.init(self.node)

        # Initialize the configuration
        Configuration.getInstance().initialize(SYSTEM_CONFIG_PATH(__file__, "settings"))

        self._data.robot_area = [Disc(0.0, CONFIG["robot_radius"])]

        # Initialize the planner
        self._planner = Planner()

        # Initialize the ROS interface
        self.initializeSubscribersAndPublishers()

        self.startEnvironment()

        self._timeout_timer.setDuration(60.0)
        self._timeout_timer.start()
        for i in range(CAMERA_BUFFER):
            self._x_buffer[i] = 0.0
            self._y_buffer[i] = 0.0

        Instrumentor.Get().BeginSession("mpc_planner_ros2navigation")

        LOG_DIVIDER()
        return True

    def cleanup(self):
        """
        Clean up resources used by the controller
        """
        self.logger.info("Stopped ROS2Navigation Planner")
        BENCHMARKERS.print()

        Instrumentor.Get().EndSession()
        return True

    def activate(self):
        """
        Activate the controller
        """
        self.logger.info("Activating controller")
        return True

    def deactivate(self):
        """
        Deactivate the controller
        """
        self.logger.info("Deactivating controller")
        return True

    def setPlan(self, path):
        """
        Set the plan that the controller is following
        """
        # check if plugin is initialized
        if not self.initialized_:
            self.logger.error("Planner has not been initialized, please call configure() before using this planner")
            return False

        # store the global plan
        self.global_plan_ = []
        self.global_plan_ = path.poses

        # we do not clear the local planner here, since setPlan is called frequently whenever the global planner updates the plan.
        # the local planner checks whether it is required to reinitialize the trajectory or not within each velocity computation step.

        return True

    def computeVelocityCommands(self, pose, velocity, goal_checker_reset):
        """
        Given the current state of the robot, compute velocity command to send to base
        """
        if not self.initialized_:
            self.logger.error("This planner has not been initialized")
            return Twist()

        # Convert Path format
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.node.get_clock().now().to_msg()
        path.poses = self.global_plan_
        self.pathCallback(path)

        cmd_vel = Twist()
        if self._rotate_to_goal:
            self.rotateToGoal(cmd_vel)
        else:
            self.loop(cmd_vel)

        return cmd_vel

    def initializeSubscribersAndPublishers(self):
        """
        Set up ROS2 subscribers and publishers
        """
        self.logger.info("initializeSubscribersAndPublishers")

        # QoS profiles
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Subscribers
        self._state_sub = self.node.create_subscription(
            Odometry,
            "/input/state",
            self.stateCallback,
            qos_profile,
            callback_group=self._callback_group
        )

        self._state_pose_sub = self.node.create_subscription(
            PoseStamped,
            "/input/state_pose",
            self.statePoseCallback,
            qos_profile,
            callback_group=self._callback_group
        )

        self._goal_sub = self.node.create_subscription(
            PoseStamped,
            "/input/goal",
            self.goalCallback,
            1,
            callback_group=self._callback_group
        )

        self._path_sub = self.node.create_subscription(
            Path,
            "/input/reference_path",
            self.pathCallback,
            1,
            callback_group=self._callback_group
        )

        self._obstacle_sim_sub = self.node.create_subscription(
            ObstacleArray,
            "/input/obstacles",
            self.obstacleCallback,
            1,
            callback_group=self._callback_group
        )

        self._collisions_sub = self.node.create_subscription(
            Float32,
            "/feedback/collisions",
            self.collisionCallback,
            1,
            callback_group=self._callback_group
        )

        # Publishers
        self._cmd_pub = self.node.create_publisher(
            Twist,
            "/output/command",
            1
        )

        self._pose_pub = self.node.create_publisher(
            PoseStamped,
            "/output/pose",
            1
        )

        # TF broadcaster
        self._camera_pub = TransformBroadcaster(self.node)

        # Environment Reset
        self._reset_simulation_pub = self.node.create_publisher(
            Empty,
            "/lmpcc/reset_environment",
            1
        )

        # Service clients
        self._reset_simulation_client = self.node.create_client(
            EmptySrv,
            "/gazebo/reset_world",
            callback_group=self._callback_group
        )

        self._reset_ekf_client = self.node.create_client(
            SetPose,
            "/set_pose",
            callback_group=self._callback_group
        )

        # Pedestrian simulator
        self._ped_horizon_pub = self.node.create_publisher(
            Int32,
            "/pedestrian_simulator/horizon",
            1
        )

        self._ped_integrator_step_pub = self.node.create_publisher(
            Float32,
            "/pedestrian_simulator/integrator_step",
            1
        )

        self._ped_clock_frequency_pub = self.node.create_publisher(
            Float32,
            "/pedestrian_simulator/clock_frequency",
            1
        )

        self._ped_start_client = self.node.create_client(
            EmptySrv,
            "/pedestrian_simulator/start",
            callback_group=self._callback_group
        )

    def startEnvironment(self):
        """
        Initialize the environment and pedestrian simulator
        """
        self.logger.info("Starting pedestrian simulator")
        for i in range(20):
            horizon_msg = Int32()
            horizon_msg.data = CONFIG["N"]
            self._ped_horizon_pub.publish(horizon_msg)

            integrator_step_msg = Float32()
            integrator_step_msg.data = CONFIG["integrator_step"]
            self._ped_integrator_step_pub.publish(integrator_step_msg)

            clock_frequency_msg = Float32()
            clock_frequency_msg.data = CONFIG["control_frequency"]
            self._ped_clock_frequency_pub.publish(clock_frequency_msg)

            # Check if service is available
            if not self._ped_start_client.wait_for_service(timeout_sec=1.0):
                self.logger.info("Waiting for pedestrian simulator service...")
                continue

            # Call service
            req = EmptySrv.Request()
            future = self._ped_start_client.call_async(req)

            # Wait for response (in ROS2 we need to use futures)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=1.0)

            if future.result() is not None:
                break
            else:
                self.logger.info("Waiting for pedestrian simulator to start")
                # Sleep in ROS2
                self.node.create_rate(1.0).sleep()

                self._reset_simulation_pub.publish(Empty())

        self._enable_output = CONFIG["enable_output"]
        self.logger.info("Environment ready.")

    def isGoalReached(self):
        """
        Check if the goal has been reached
        """
        if not self.initialized_:
            self.logger.error("This planner has not been initialized")
            return False

        goal_reached = self._planner.isObjectiveReached(self._state, self._data) and not self.done_
        if goal_reached:
            LOG_SUCCESS("Goal Reached!")
            self.done_ = True
            self.reset()

        return goal_reached

    def rotateToGoal(self, cmd_vel):
        """
        Rotate the robot to face the goal before moving
        """
        if not hasattr(self, "_log_throttle"):
            self._log_throttle = 0

        current_time = time.time()
        if current_time - self._log_throttle > 1.5:
            self.logger.info("Rotating to the goal")
            self._log_throttle = current_time

        if not self._data.goal_received:
            self.logger.info("Waiting for the goal")
            return

        goal_angle = 0.0

        if len(self._data.reference_path.x) > 2:
            goal_angle = math.atan2(self._data.reference_path.y[2] - self._state.get("y"),
                                    self._data.reference_path.x[2] - self._state.get("x"))
        else:
            goal_angle = math.atan2(self._data.goal[1] - self._state.get("y"),
                                    self._data.goal[0] - self._state.get("x"))

        angle_diff = goal_angle - self._state.get("psi")

        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi

        if abs(angle_diff) > math.pi / 4.0:
            cmd_vel.linear.x = 0.0
            if self._enable_output:
                cmd_vel.angular.z = 1.5 * sgn(angle_diff)
            else:
                cmd_vel.angular.z = 0.0
        else:
            LOG_SUCCESS("Robot rotated and is ready to follow the path")
            self._rotate_to_goal = False

    def loop(self, cmd_vel):
        """
        Main control loop
        """
        # Copy data for thread safety
        data = self._data  # In Python, this is a reference, not a deep copy
        state = self._state  # In Python, this is a reference, not a deep copy

        data.planning_start_time = datetime.now()

        LOG_MARK("============= Loop =============")

        if self._timeout_timer.hasFinished():
            self.reset(False)
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            return

        if CONFIG["debug_output"]:
            state.print()

        loop_benchmarker = BENCHMARKERS.getBenchmarker("loop")
        loop_benchmarker.start()

        output = self._planner.solveMPC(state, data)

        LOG_MARK("Success: " + str(output.success))

        if self._enable_output and output.success:
            # Publish the command
            cmd_vel.linear.x = self._planner.getSolution(1, "v")  # = x1
            cmd_vel.angular.z = self._planner.getSolution(0, "w")  # = u0
            LOG_VALUE_DEBUG("Commanded v", cmd_vel.linear.x)
            LOG_VALUE_DEBUG("Commanded w", cmd_vel.angular.z)
        else:
            deceleration = CONFIG["deceleration_at_infeasible"]
            velocity_after_braking = 0.0
            velocity = 0.0
            dt = 1.0 / CONFIG["control_frequency"]

            velocity = self._state.get("v")
            velocity_after_braking = velocity - deceleration * dt  # Brake with the given deceleration
            cmd_vel.linear.x = max(velocity_after_braking, 0.0)  # Don't drive backwards when braking
            cmd_vel.angular.z = 0.0

        # Create a copy to publish (to preserve cmd_vel for the caller)
        cmd = Twist()
        cmd.linear.x = cmd_vel.linear.x
        cmd.angular.z = cmd_vel.angular.z
        self._cmd_pub.publish(cmd)

        self.publishPose()
        self.publishCamera()

        loop_benchmarker.stop()

        if CONFIG["recording"]["enable"]:
            # Save control inputs
            if output.success:
                data_saver = self._planner.getDataSaver()
                data_saver.AddData("input_a", state.get("a"))
                data_saver.AddData("input_v", self._planner.getSolution(1, "v"))
                data_saver.AddData("input_w", self._planner.getSolution(0, "w"))

            self._planner.saveData(state, data)

        if output.success:
            self._planner.visualize(state, data)
            self.visualize()

        LOG_MARK("============= End Loop =============")

    def stateCallback(self, msg):
        """
        Process state updates from odometry
        """
        LOG_MARK("State callback")
        self._state.set("x", msg.pose.pose.position.x)
        self._state.set("y", msg.pose.pose.position.y)
        self._state.set("psi", quaternion_to_angle(msg.pose.pose.orientation))
        self._state.set("v",
                        math.sqrt(math.pow(msg.twist.twist.linear.x, 2.0) + math.pow(msg.twist.twist.linear.y, 2.0)))

        if abs(msg.pose.pose.orientation.x) > (math.pi / 8.0) or abs(msg.pose.pose.orientation.y) > (math.pi / 8.0):
            LOG_WARN("Detected flipped robot. Resetting.")
            self.reset(False)  # Reset without success

    def statePoseCallback(self, msg):
        """
        Process state updates from pose
        """
        LOG_MARK("State callback")

        self._state.set("x", msg.pose.position.x)
        self._state.set("y", msg.pose.position.y)
        self._state.set("psi", msg.pose.orientation.z)
        self._state.set("v", msg.pose.position.z)

        if abs(msg.pose.orientation.x) > (math.pi / 8.0) or abs(msg.pose.orientation.y) > (math.pi / 8.0):
            LOG_ERROR("Detected flipped robot. Resetting.")
            self.reset(False)  # Reset without success

    def goalCallback(self, msg):
        """
        Handle new goal
        """
        LOG_MARK("Goal callback")

        self._data.goal[0] = msg.pose.position.x
        self._data.goal[1] = msg.pose.position.y
        self._data.goal_received = True

        self._rotate_to_goal = True

    def isPathTheSame(self, msg):
        """
        Check if the received path is the same as the current one
        """
        # Check if the path is the same
        if len(self._data.reference_path.x) != len(msg.poses):
            return False

        # Check up to the first two points
        num_points = min(2, len(self._data.reference_path.x))
        for i in range(num_points):
            if not self._data.reference_path.pointInPath(i, msg.poses[i].pose.position.x, msg.poses[i].pose.position.y):
                return False
        return True

    def pathCallback(self, msg):
        """
        Handle path updates
        """
        LOG_MARK("Path callback")

        downsample = CONFIG["downsample_path"]

        if self.isPathTheSame(msg) or len(msg.poses) < downsample + 1:
            return

        self._data.reference_path.clear()

        count = 0
        for pose in msg.poses:
            if count % downsample == 0 or count == len(msg.poses) - 1:
                self._data.reference_path.x.append(pose.pose.position.x)
                self._data.reference_path.y.append(pose.pose.position.y)
                self._data.reference_path.psi.append(quaternion_to_angle(pose.pose.orientation))
            count += 1

        # Notify planner of updated path
        self._planner.onDataReceived(self._data, "reference_path")

    def obstacleCallback(self, msg):
        """
        Handle obstacle updates
        """
        LOG_MARK("Obstacle callback")

        self._data.dynamic_obstacles.clear()

        for obstacle in msg.obstacles:
            # Save the obstacle
            self._data.dynamic_obstacles.append(
                DynamicObstacle(
                    obstacle.id,
                    np.array([obstacle.pose.position.x, obstacle.pose.position.y]),
                    quaternion_to_angle(obstacle.pose),
                    CONFIG["obstacle_radius"]
                )
            )
            dynamic_obstacle = self._data.dynamic_obstacles[-1]

            if len(obstacle.probabilities) == 0:  # No Predictions!
                continue

            # Save the prediction
            if len(obstacle.probabilities) == 1:  # One mode
                dynamic_obstacle.prediction = Prediction(PredictionType.GAUSSIAN)

                mode = obstacle.gaussians[0]
                for k in range(len(mode.mean.poses)):
                    dynamic_obstacle.prediction.modes[0].append(
                        PredictionPoint(
                            np.array([mode.mean.poses[k].pose.position.x, mode.mean.poses[k].pose.position.y]),
                            quaternion_to_angle(mode.mean.poses[k].pose.orientation),
                            mode.major_semiaxis[k],
                            mode.minor_semiaxis[k]
                        )
                    )

                if mode.major_semiaxis[-1] == 0.0 or not CONFIG["probabilistic"]["enable"]:
                    dynamic_obstacle.prediction.type = PredictionType.DETERMINISTIC
                else:
                    dynamic_obstacle.prediction.type = PredictionType.GAUSSIAN
            else:
                assert False, "Multiple modes not yet supported"

        ensure_obstacle_size(self._data.dynamic_obstacles, self._state)

        if CONFIG["probabilistic"]["propagate_uncertainty"]:
            propagate_prediction_uncertainty(self._data.dynamic_obstacles)

        self._planner.onDataReceived(self._data, "dynamic obstacles")

    def visualize(self):
        """
        Visualize the robot's angle
        """
        publisher = VISUALS.getPublisher("angle")
        line = publisher.getNewLine()

        line.addLine(
            np.array([self._state.get("x"), self._state.get("y")]),
            np.array([self._state.get("x") + 1.0 * math.cos(self._state.get("psi")),
                      self._state.get("y") + 1.0 * math.sin(self._state.get("psi"))])
        )
        publisher.publish()

    def reset(self, success=True):
        """
        Reset the environment
        """
        self.logger.info("Resetting")
        with self._reset_mutex:
            # Call reset world service
            if not self._reset_simulation_client.wait_for_service(timeout_sec=1.0):
                self.logger.warn("Reset world service not available")
            else:
                req = EmptySrv.Request()
                future = self._reset_simulation_client.call_async(req)
                rclpy.spin_until_future_complete(self.node, future, timeout_sec=1.0)

            # Call reset EKF service
            if not self._reset_ekf_client.wait_for_service(timeout_sec=1.0):
                self.logger.warn("Reset EKF service not available")
            else:
                req = SetPose.Request()
                future = self._reset_ekf_client.call_async(req)
                rclpy.spin_until_future_complete(self.node, future, timeout_sec=1.0)

            # Publish reset message
            self._reset_simulation_pub.publish(Empty())

            # Reset camera buffer
            for i in range(CAMERA_BUFFER):
                self._x_buffer[i] = 0.0
                self._y_buffer[i] = 0.0

            # Reset planner
            self._planner.reset(self._state, self._data, success)
            self._data.costmap = self.costmap_

            # Sleep (ROS2 way)
            self.node.create_rate(CONFIG["control_frequency"]).sleep()

            # Reset state variables
            self.done_ = False
            self._rotate_to_goal = False

            # Restart timer
            self._timeout_timer.start()

    def collisionCallback(self, msg):
        """
        Handle collision feedback
        """
        LOG_MARK("Collision callback")

        self._data.intrusion = float(msg.data)

        # Throttled logging for collisions
        if self._data.intrusion > 0.0:
            if not hasattr(self, "_collision_log_time"):
                self._collision_log_time = 0

            current_time = time.time()
            if current_time - self._collision_log_time > 0.5:
                self.logger.info(f"Collision detected (Intrusion: {self._data.intrusion})")
                self._collision_log_time = current_time

    def publishPose(self):
        """
        Publish the current pose
        """
        pose = PoseStamped()
        pose.pose.position.x = self._state.get("x")
        pose.pose.position.y = self._state.get("y")
        pose.pose.orientation = angle_to_quaternion(self._state.get("psi"))

        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.header.frame_id = "map"

        self._pose_pub.publish(pose)

    def publishCamera(self):
        """
        Publish camera transform for visualization
        """
        now = self.node.get_clock().now()

        # Convert ROS2 time to Duration
        duration_since_last = Duration(seconds=0)
        if hasattr(self, "_prev_stamp") and self._prev_stamp != Time():
            duration_since_last = now - self._prev_stamp

        # Check if we need to publish based on frequency
        if duration_since_last.nanoseconds < int(0.5 / CONFIG["control_frequency"] * 1e9):
            return

        self._prev_stamp = now

        msg = TransformStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "map"
        msg.child_frame_id = "camera"

        # Smoothen the camera
        for i in range(CAMERA_BUFFER - 1):
            self._x_buffer[i] = self._x_buffer[i + 1]
            self._y_buffer[i] = self._y_buffer[i + 1]
        self._x_buffer[CAMERA_BUFFER - 1] = self._state.get("x")
        self._y_buffer[CAMERA_BUFFER - 1] = self._state.get("y")
        camera_x = 0.0
        camera_y = 0.0
        for i in range(CAMERA_BUFFER):
            camera_x += self._x_buffer[i]
            camera_y += self._y_buffer[i]
        msg.transform.translation.x = camera_x / float(CAMERA_BUFFER)
        msg.transform.translation.y = camera_y / float(CAMERA_BUFFER)
        msg.transform.translation.z = 0.0
        msg.transform.rotation.x = 0
        msg.transform.rotation.y = 0
        msg.transform.rotation.z = 0
        msg.transform.rotation.w = 1

        self._camera_pub.sendTransform(msg)


# Additional required classes (simplified implementations)
class DynamicObstacle:
    def __init__(self, id, position, orientation, radius):
        self.id = id
        self.position = position  # Eigen::Vector2d or numpy array
        self.orientation = orientation
        self.radius = radius
        self.prediction = None


class PredictionPoint:
    def __init__(self, position, orientation, major_semiaxis, minor_semiaxis):
        self.position = position
        self.orientation = orientation
        self.major_semiaxis = major_semiaxis
        self.minor_semiaxis = minor_semiaxis


# ROS2 plugin export
def register_ros2navigation_planner():
    return ROS2NavigationPlanner()