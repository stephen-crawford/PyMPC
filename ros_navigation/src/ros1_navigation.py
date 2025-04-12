#!/usr/bin/env python3

import numpy as np
import math
import threading
import time
from datetime import datetime
import chrono

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from std_msgs.msg import Empty, Float32, Float64, Int32
from std_srvs.srv import Empty as EmptySrv
from robot_localization.srv import SetPose
from tf2_ros import TransformBroadcaster
from nav_core import BaseLocalPlanner
from costmap_2d import Costmap2DROS, Costmap2D

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
CONFIG = load_yaml_file(SYSTEM_CONFIG_PATH(__file__, "settings"))
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


class ROSNavigationPlanner(BaseLocalPlanner):
    def __init__(self, name=None, tf=None, costmap_ros=None):
        self.costmap_ros_ = None
        self.tf_ = None
        self.initialized_ = False
        self.general_nh_ = rospy.NodeHandle()

        self._data = RealTimeData()
        self._state = State()
        self._planner = None
        self._reconfigure = None

        # Initialize buffers and variables
        self._x_buffer = [0.0] * CAMERA_BUFFER
        self._y_buffer = [0.0] * CAMERA_BUFFER
        self._prev_stamp = rospy.Time(0)
        self._timeout_timer = Timer()
        self._rotate_to_goal = False
        self._enable_output = False
        self.done_ = False
        self.global_plan_ = []

        self._reset_mutex = threading.Lock()
        self._reset_msg = EmptySrv()
        self._reset_pose_msg = SetPose()

        self._camera_pub = TransformBroadcaster()

        if name is not None and tf is not None and costmap_ros is not None:
            self.initialize(name, tf, costmap_ros)

    def initialize(self, name, tf, costmap_ros):
        if not self.initialized_:
            nh = rospy.NodeHandle(f"~/{name}")

            self.tf_ = tf

            self.costmap_ros_ = costmap_ros
            self.costmap_ = self.costmap_ros_.getCostmap()
            self._data.costmap = self.costmap_

            self.initialized_ = True

            LOG_INFO("Started ROSNavigation Planner")

            VISUALS.init(self.general_nh_)

            # Initialize the configuration
            Configuration.getInstance().initialize(SYSTEM_CONFIG_PATH(__file__, "settings"))

            self._data.robot_area = [Disc(0.0, CONFIG["robot_radius"])]

            # Initialize the planner
            self._planner = Planner()

            # Initialize the ROS interface
            self.initializeSubscribersAndPublishers(nh)

            self.startEnvironment()

            # TODO: implement reconfigure class
            # self._reconfigure = RosnavigationReconfigure()

            self._timeout_timer.setDuration(60.0)
            self._timeout_timer.start()
            for i in range(CAMERA_BUFFER):
                self._x_buffer[i] = 0.0
                self._y_buffer[i] = 0.0

            Instrumentor.Get().BeginSession("mpc_planner_rosnavigation")

            LOG_DIVIDER()

    def __del__(self):
        LOG_INFO("Stopped ROSNavigation Planner")
        BENCHMARKERS.print()

        Instrumentor.Get().EndSession()

    def setPlan(self, orig_global_plan):
        # check if plugin is initialized
        if not self.initialized_:
            rospy.logerr("planner has not been initialized, please call initialize() before using this planner")
            return False

        # store the global plan
        self.global_plan_ = []
        self.global_plan_ = orig_global_plan

        # we do not clear the local planner here, since setPlan is called frequently whenever the global planner updates the plan.
        # the local planner checks whether it is required to reinitialize the trajectory or not within each velocity computation step.

        # reset goal_reached_ flag
        # self.goal_reached_ = False

        return True

    def computeVelocityCommands(self, cmd_vel):
        if not self.initialized_:
            rospy.logerr("This planner has not been initialized")
            return False

        path = Path()
        path.poses = self.global_plan_
        self.pathCallback(path)

        if self._rotate_to_goal:
            self.rotateToGoal(cmd_vel)
        else:
            self.loop(cmd_vel)

        return True

    def initializeSubscribersAndPublishers(self, nh):
        LOG_INFO("initializeSubscribersAndPublishers")

        self._state_sub = nh.subscribe("/input/state", 5, self.stateCallback)

        self._state_pose_sub = nh.subscribe("/input/state_pose", 5, self.statePoseCallback)

        self._goal_sub = nh.subscribe("/input/goal", 1, self.goalCallback)

        self._path_sub = nh.subscribe("/input/reference_path", 1, self.pathCallback)

        self._obstacle_sim_sub = nh.subscribe("/input/obstacles", 1, self.obstacleCallback)

        self._cmd_pub = nh.advertise("/output/command", 1)

        self._pose_pub = nh.advertise("/output/pose", 1)

        self._collisions_sub = nh.subscribe("/feedback/collisions", 1, self.collisionCallback)

        # Environment Reset
        self._reset_simulation_pub = nh.advertise("/lmpcc/reset_environment", 1)
        self._reset_simulation_client = nh.serviceClient("/gazebo/reset_world", EmptySrv)
        self._reset_ekf_client = nh.serviceClient("/set_pose", SetPose)

        # Pedestrian simulator
        self._ped_horizon_pub = nh.advertise("/pedestrian_simulator/horizon", 1)
        self._ped_integrator_step_pub = nh.advertise("/pedestrian_simulator/integrator_step", 1)
        self._ped_clock_frequency_pub = nh.advertise("/pedestrian_simulator/clock_frequency", 1)
        self._ped_start_client = nh.serviceClient("/pedestrian_simulator/start", EmptySrv)

    def startEnvironment(self):
        # Manually add obstacles in the costmap!
        # mx, my = 0, 0
        # self.costmap_.worldToMapEnforceBounds(2.0, 2.0, mx, my)
        # LOG_VALUE("mx", mx)
        # LOG_VALUE("my", my)

        # for i in range(10):
        #     self.costmap_.setCost(mx + i, my, costmap_2d.LETHAL_OBSTACLE)

        LOG_INFO("Starting pedestrian simulator")
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

            empty_msg = EmptySrv()
            if self._ped_start_client.call(empty_msg):
                break
            else:
                LOG_INFO_THROTTLE(3, "Waiting for pedestrian simulator to start")
                rospy.Duration(1.0).sleep()

                self._reset_simulation_pub.publish(Empty())

        self._enable_output = CONFIG["enable_output"]
        LOG_INFO("Environment ready.")

    def isGoalReached(self):
        if not self.initialized_:
            rospy.logerr("This planner has not been initialized")
            return False

        goal_reached = self._planner.isObjectiveReached(self._state, self._data) and not self.done_  # Activate once
        if goal_reached:
            LOG_SUCCESS("Goal Reached!")
            self.done_ = True
            self.reset()

        return goal_reached

    def rotateToGoal(self, cmd_vel):
        LOG_INFO_THROTTLE(1500, "Rotating to the goal")
        if not self._data.goal_received:
            LOG_INFO("Waiting for the goal")
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

        cmd = Twist()
        if self._enable_output and output.success:
            # Publish the command
            cmd_vel.linear.x = self._planner.getSolution(1, "v")  # = x1
            cmd_vel.angular.z = self._planner.getSolution(0, "w")  # = u0
            LOG_VALUE_DEBUG("Commanded v", cmd.linear.x)
            LOG_VALUE_DEBUG("Commanded w", cmd.angular.z)
        else:
            deceleration = CONFIG["deceleration_at_infeasible"]
            velocity_after_braking = 0.0
            velocity = 0.0
            dt = 1.0 / CONFIG["control_frequency"]

            velocity = self._state.get("v")
            velocity_after_braking = velocity - deceleration * dt  # Brake with the given deceleration
            cmd_vel.linear.x = max(velocity_after_braking, 0.0)  # Don't drive backwards when braking
            cmd_vel.angular.z = 0.0

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
        LOG_MARK("State callback")

        self._state.set("x", msg.pose.position.x)
        self._state.set("y", msg.pose.position.y)
        self._state.set("psi", msg.pose.orientation.z)
        self._state.set("v", msg.pose.position.z)

        if abs(msg.pose.orientation.x) > (math.pi / 8.0) or abs(msg.pose.orientation.y) > (math.pi / 8.0):
            LOG_ERROR("Detected flipped robot. Resetting.")
            self.reset(False)  # Reset without success

    def goalCallback(self, msg):
        LOG_MARK("Goal callback")

        self._data.goal[0] = msg.pose.position.x
        self._data.goal[1] = msg.pose.position.y
        self._data.goal_received = True

        self._rotate_to_goal = True

    def isPathTheSame(self, msg):
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
        LOG_MARK("Path callback")

        downsample = CONFIG["downsample_path"]

        if self.isPathTheSame(msg) or len(msg.poses) < downsample + 1:
            return

        self._data.reference_path.clear()

        count = 0
        for pose in msg.poses:
            if count % downsample == 0 or count == len(msg.poses) - 1:  # Todo
                self._data.reference_path.x.append(pose.pose.position.x)
                self._data.reference_path.y.append(pose.pose.position.y)
                self._data.reference_path.psi.append(quaternion_to_angle(pose.pose.orientation))
            count += 1

        # Fit a clothoid on the global path to sample points on the spline from
        # clothoid = Clothoid2D(self._data.reference_path.x, self._data.reference_path.y, self._data.reference_path.psi, 2.0)
        # self._data.reference_path.clear()
        # clothoid.getPointsOnClothoid(self._data.reference_path.x, self._data.reference_path.y, self._data.reference_path.s)

        # Velocity
        """
    LOG_VALUE("velocity reference", CONFIG["weights"]["reference_velocity"])
    for i in range(len(self._data.reference_path.x)):
        if i != len(self._data.reference_path.x) - 1:
            self._data.reference_path.v.append(CONFIG["weights"]["reference_velocity"])
        else:
            self._data.reference_path.v.append(0.0)
    """

        self._planner.onDataReceived(self._data, "reference_path")

    def obstacleCallback(self, msg):
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
        publisher = VISUALS.getPublisher("angle")
        line = publisher.getNewLine()

        line.addLine(
            np.array([self._state.get("x"), self._state.get("y")]),
            np.array([self._state.get("x") + 1.0 * math.cos(self._state.get("psi")),
                      self._state.get("y") + 1.0 * math.sin(self._state.get("psi"))])
        )
        publisher.publish()

    def reset(self, success=True):
        LOG_INFO("Resetting")
        with self._reset_mutex:
            self._reset_simulation_client.call(self._reset_msg)
            self._reset_ekf_client.call(self._reset_pose_msg)
            self._reset_simulation_pub.publish(Empty())

            for i in range(CAMERA_BUFFER):
                self._x_buffer[i] = 0.0
                self._y_buffer[i] = 0.0

            self._planner.reset(self._state, self._data, success)
            self._data.costmap = self.costmap_

            rospy.Duration(1.0 / CONFIG["control_frequency"]).sleep()

            self.done_ = False
            self._rotate_to_goal = False

            self._timeout_timer.start()

    def collisionCallback(self, msg):
        LOG_MARK("Collision callback")

        self._data.intrusion = float(msg.data)

        if self._data.intrusion > 0.0:
            LOG_INFO_THROTTLE(500.0, "Collision detected (Intrusion: " + str(self._data.intrusion) + ")")

    def publishPose(self):
        pose = PoseStamped()
        pose.pose.position.x = self._state.get("x")
        pose.pose.position.y = self._state.get("y")
        pose.pose.orientation = angle_to_quaternion(self._state.get("psi"))

        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"

        self._pose_pub.publish(pose)

    def publishCamera(self):
        msg = TransformStamped()
        msg.header.stamp = rospy.Time.now()

        if (msg.header.stamp - self._prev_stamp) < rospy.Duration(0.5 / CONFIG["control_frequency"]):
            return

        self._prev_stamp = msg.header.stamp

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
        msg.transform.translation.x = camera_x / float(CAMERA_BUFFER)  # self._state.get("x")
        msg.transform.translation.y = camera_y / float(CAMERA_BUFFER)  # self._state.get("y")
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
        self.position = position  # Eigen::Vector2d
        self.orientation = orientation
        self.radius = radius
        self.prediction = None


class PredictionPoint:
    def __init__(self, position, orientation, major_semiaxis, minor_semiaxis):
        self.position = position
        self.orientation = orientation
        self.major_semiaxis = major_semiaxis
        self.minor_semiaxis = minor_semiaxis


# Register the plugin with ROS
def register_plugin():
    from pluginlib.class_list_macros import register
    register("local_planner/ROSNavigationPlanner", ROSNavigationPlanner)


if __name__ == "__main__":
    rospy.init_node("ros_navigation_planner")
    # Plugin initialization will be handled by ROS navigation stack
    rospy.spin()