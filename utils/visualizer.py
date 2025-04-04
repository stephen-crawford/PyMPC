import numpy as np
import tf2_py
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Quaternion
import tf2_ros
import math
from enum import Enum


class Colormap(Enum):
    VIRIDIS = 1
    INFERNO = 2
    BRUNO = 3


class ROSMarker:
    # Color palettes
    VIRIDIS = [253, 231, 37, 234, 229, 26, 210, 226, 27, 186, 222, 40, 162, 218, 55, 139, 214, 70, 119, 209, 83, 99,
               203, 95, 80, 196, 106, 63, 188, 115, 49, 181, 123, 38, 173, 129, 33, 165, 133, 30, 157, 137, 31, 148,
               140, 34, 140, 141, 37, 131, 142, 41, 123, 142, 44, 115, 142, 47, 107, 142, 51, 98, 141, 56, 89, 140]
    INFERNO = [252, 255, 164, 241, 237, 113, 246, 213, 67, 251, 186, 31, 252, 161, 8, 248, 135, 14, 241, 113, 31, 229,
               92, 48, 215, 75, 63, 196, 60, 78, 177, 50, 90, 155, 41, 100, 135, 33, 107, 113, 25, 110, 92, 18, 110, 69,
               10, 105, 47, 10, 91, 24, 12, 60]
    BRUNO = [217, 83, 25, 0, 114, 189, 119, 172, 48, 126, 47, 142, 237, 177, 32, 77, 190, 238, 162, 19, 47, 256, 153,
             256, 0, 103, 256]

    def __init__(self, ros_publisher, frame_id):
        self.ros_publisher_ = ros_publisher
        self.marker_ = Marker()
        self.marker_.header.frame_id = frame_id

    def stamp(self):
        self.marker_.header.stamp = Clock().now().to_msg()

    def set_color(self, r, g, b, alpha=1.0):
        self.marker_.color.r = r
        self.marker_.color.g = g
        self.marker_.color.b = b
        self.marker_.color.a = alpha

    def set_color_ratio(self, ratio, alpha=1.0):
        r, g, b = self.get_color_from_range(ratio)
        self.set_color(r, g, b, alpha)

    def set_color_int(self, select, alpha=1.0, colormap=Colormap.VIRIDIS):
        r, g, b = self.get_color_from_range_int(select, colormap)
        self.set_color(r, g, b, alpha)

    def set_color_int_range(self, select, range_val, alpha=1.0, colormap=Colormap.VIRIDIS):
        colors = self.get_colors(colormap)
        select = math.floor(select * (len(colors) // 3) / range_val)  # Scale selection with range
        r, g, b = self.get_color_from_range_int(select, colormap)
        self.set_color(r, g, b, alpha)

    def set_scale(self, x, y=None, z=None):
        self.marker_.scale.x = x
        if y is not None:
            self.marker_.scale.y = y
            if z is not None:
                self.marker_.scale.z = z

    def set_orientation(self, val):
        if isinstance(val, float):  # Assume it's a psi angle (yaw)
            q = tf2_ros.transformations.quaternion_from_euler(0, 0, val)
            self.marker_.pose.orientation.x = q[0]
            self.marker_.pose.orientation.y = q[1]
            self.marker_.pose.orientation.z = q[2]
            self.marker_.pose.orientation.w = q[3]
        elif isinstance(val, Quaternion):
            self.marker_.pose.orientation = val
        else:  # Assume it's a tf2 quaternion
            self.marker_.pose.orientation.x = val[0]
            self.marker_.pose.orientation.y = val[1]
            self.marker_.pose.orientation.z = val[2]
            self.marker_.pose.orientation.w = val[3]

    def set_lifetime(self, lifetime):
        self.marker_.lifetime = rclpy.duration.Duration(seconds=lifetime).to_msg()

    def set_action_delete(self):
        self.marker_.action = Marker.DELETE

    @staticmethod
    def vec_to_point(v):
        p = Point()
        p.x = v[0]
        p.y = v[1]
        p.z = v[2]
        return p

    @staticmethod
    def get_color_from_range_int(select, colormap=Colormap.VIRIDIS):
        colors = ROSMarker.get_colors(colormap)

        select %= len(colors) // 3  # We only have limited values
        # Invert the color range
        select = (len(colors) // 3) - 1 - select

        red = colors[select * 3 + 0]
        green = colors[select * 3 + 1]
        blue = colors[select * 3 + 2]

        return red / 256.0, green / 256.0, blue / 256.0

    @staticmethod
    def get_colors(colormap):
        if colormap == Colormap.VIRIDIS:
            return ROSMarker.VIRIDIS
        elif colormap == Colormap.INFERNO:
            return ROSMarker.INFERNO
        elif colormap == Colormap.BRUNO:
            return ROSMarker.BRUNO
        else:
            raise RuntimeError("Invalid colormap given")

    @staticmethod
    def get_color_from_range(ratio):
        # Normalize ratio to fit into 6 regions of 256 units each
        normalized = int(ratio * 256 * 6)

        # Find distance to start of closest region
        x = normalized % 256

        region = normalized // 256
        if region == 0:
            red, green, blue = 255, x, 0  # red
        elif region == 1:
            red, green, blue = 255 - x, 255, 0  # yellow
        elif region == 2:
            red, green, blue = 0, 255, x  # green
        elif region == 3:
            red, green, blue = 0, 255 - x, 255  # cyan
        elif region == 4:
            red, green, blue = x, 0, 255  # blue
        elif region == 5:
            red, green, blue = 255, 0, 255 - x  # magenta

        return red / 256.0, green / 256.0, blue / 256.0


class ROSMarkerPublisher:
    def __init__(self, node, topic_name, frame_id, max_size=1000):
        self.frame_id_ = frame_id
        self.max_size_ = max_size

        # Use the node directly if passed a Node object, otherwise assume it's a SharedPtr
        if isinstance(node, Node):
            self.pub_ = node.create_publisher(MarkerArray, topic_name, 3)
        else:
            self.pub_ = node.create_publisher(MarkerArray, topic_name, 3)

        self.id_ = 0
        self.prev_id_ = 0
        self.topic_name_ = topic_name

        # Initialize marker list
        self.marker_list_ = MarkerArray()
        self.ros_markers_ = []

        # Clear any left-over markers
        remove_all_marker_list = MarkerArray()
        remove_all_marker = Marker()
        remove_all_marker.action = Marker.DELETEALL
        remove_all_marker.header.frame_id = frame_id
        remove_all_marker.header.stamp = Clock().now().to_msg()
        remove_all_marker_list.markers.append(remove_all_marker)
        self.pub_.publish(remove_all_marker_list)

    def add(self, marker):
        if marker.id > self.max_size_ - 1:
            # If we exceed the max size, allocate 1.5 times the space
            prev_size = self.max_size_
            self.max_size_ = math.ceil(self.max_size_ * 1.5)
            # We would log a warning here in ROS

        # Add the marker
        self.marker_list_.markers.append(marker)

    def get_new_line(self):
        # Create a line
        ros_line = ROSLine(self, self.frame_id_)
        self.ros_markers_.append(ros_line)
        return ros_line

    def get_new_point_marker(self, marker_type="CUBE"):
        # Create a point marker
        ros_point = ROSPointMarker(self, self.frame_id_, marker_type)
        self.ros_markers_.append(ros_point)
        return ros_point

    def get_new_multiple_point_marker(self, marker_type="CUBE"):
        # Create multiple point marker
        ros_points = ROSMultiplePointMarker(self, self.frame_id_, marker_type)
        self.ros_markers_.append(ros_points)
        return ros_points

    def get_new_text_marker(self):
        # Create text marker
        ros_text = ROSTextMarker(self, self.frame_id_)
        self.ros_markers_.append(ros_text)
        return ros_text

    def get_new_model_marker(self, model_path):
        # Create model marker
        ros_model = ROSModelMarker(self, self.frame_id_, model_path)
        self.ros_markers_.append(ros_model)
        return ros_model

    def publish(self, keep_markers=False):
        # If less markers are published, remove the extra markers explicitly
        remove_marker_ = Marker()
        remove_marker_.action = Marker.DELETE
        remove_marker_.header.frame_id = self.frame_id_
        remove_marker_.header.stamp = Clock().now().to_msg()

        if self.prev_id_ > self.id_:
            for i in range(self.id_, self.prev_id_):
                remove_marker_.id = i
                self.marker_list_.markers.append(remove_marker_)

        # Add new markers
        for marker in self.ros_markers_:
            marker.stamp()

        self.pub_.publish(self.marker_list_)

        if not keep_markers:
            # Clear marker data for the next iteration
            self.marker_list_ = MarkerArray()
            self.ros_markers_ = []

            self.prev_id_ = self.id_
            self.id_ = 0

    def __del__(self):
        # Remove all markers
        remove_marker_ = Marker()
        remove_marker_.action = Marker.DELETE
        remove_marker_.header.frame_id = self.frame_id_

        # Clear current markers
        self.marker_list_ = MarkerArray()

        # Delete all previous markers
        for i in range(self.prev_id_):
            remove_marker_.id = i
            self.marker_list_.markers.append(remove_marker_)

        self.pub_.publish(self.marker_list_)

    def get_id(self):
        cur_id = self.id_
        self.id_ += 1
        return cur_id


class ROSLine(ROSMarker):
    def __init__(self, ros_publisher, frame_id):
        super().__init__(ros_publisher, frame_id)

        self.marker_.type = Marker.LINE_LIST

        # LINE_STRIP/LINE_LIST markers use only the x component of scale, for the line width
        self.marker_.scale.x = 0.5

        # Line strip is red
        self.set_color(1, 0, 0)

        self.marker_.pose.orientation.x = 0.0
        self.marker_.pose.orientation.y = 0.0
        self.marker_.pose.orientation.z = 0.0
        self.marker_.pose.orientation.w = 1.0

    def add_line(self, p1, p2, z=None):
        if isinstance(p1, np.ndarray) and len(p1) == 2 and z is not None:
            # Handle 2D points with z value
            point1 = self.vec_to_point(np.array([p1[0], p1[1], z]))
            point2 = self.vec_to_point(np.array([p2[0], p2[1], z]))
            self.add_line_points(point1, point2)
        elif isinstance(p1, np.ndarray) and len(p1) == 3:
            # Handle 3D points
            point1 = self.vec_to_point(p1)
            point2 = self.vec_to_point(p2)
            self.add_line_points(point1, point2)
        elif isinstance(p1, Point) and isinstance(p2, Point):
            # Handle Point messages
            self.add_line_points(p1, p2)

    def add_line_points(self, p1, p2):
        # Request an ID
        self.marker_.id = self.ros_publisher_.get_id()

        # Add the points
        self.marker_.points.append(p1)
        self.marker_.points.append(p2)

        # Add line to the publisher
        self.ros_publisher_.add(self.marker_)

        # Clear points
        self.marker_.points = []

    def add_broken_line(self, p1, p2, dist):
        if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
            # Convert to Point messages
            p1_msg = self.vec_to_point(p1)
            p2_msg = self.vec_to_point(p2)
            self.add_broken_line_points(p1_msg, p2_msg, dist)
        else:
            # Assume they're already Point messages
            self.add_broken_line_points(p1, p2, dist)

    def add_broken_line_points(self, p1, p2, dist):
        # Interpolate the points, ensure 0.5 of the broken line at both sides!
        dpoints = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
        num_lines = math.floor(dpoints / (2.0 * dist))  # Also includes spaces!
        extra = dpoints - num_lines * (2.0 * dist)  # The difference

        num_elements = num_lines * 2

        dir_vec = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
        dir_vec = dir_vec / np.linalg.norm(dir_vec)  # Normalize

        cur_p = p1
        for i in range(num_elements + 1):  # The first line is split in an end and start
            next_p = Point()

            if i == 0 or i == num_elements:
                next_p.x = cur_p.x + dir_vec[0] * 0.5 * (dist + extra / 2.0)
                next_p.y = cur_p.y + dir_vec[1] * 0.5 * (dist + extra / 2.0)
                next_p.z = cur_p.z + dir_vec[2] * 0.5 * (dist + extra / 2.0)
            else:
                next_p.x = cur_p.x + dir_vec[0] * dist
                next_p.y = cur_p.y + dir_vec[1] * dist
                next_p.z = cur_p.z + dir_vec[2] * dist

            if i % 2 == 0:  # If this is a line, not a space
                self.add_line_points(cur_p, next_p)

            cur_p = next_p


class ROSPointMarker(ROSMarker):
    def __init__(self, ros_publisher, frame_id, marker_type):
        super().__init__(ros_publisher, frame_id)
        self.marker_type_ = marker_type

        self.marker_.type = self.get_marker_type(marker_type)

        # Set default scale
        self.marker_.scale.x = 0.5
        self.marker_.scale.y = 0.5

        # Default color (red)
        self.set_color(1.0, 0.0, 0.0)

        # Default orientation (identity quaternion)
        self.marker_.pose.orientation.x = 0.0
        self.marker_.pose.orientation.y = 0.0
        self.marker_.pose.orientation.z = 0.0
        self.marker_.pose.orientation.w = 1.0

    def add_point_marker(self, p, z=None):
        if isinstance(p, np.ndarray) and len(p) == 2 and z is not None:
            # Handle 2D point with z
            result = Point()
            result.x = p[0]
            result.y = p[1]
            result.z = z
            self.add_point_from_msg(result)
        elif isinstance(p, np.ndarray) and len(p) == 3:
            # Handle 3D point
            result = self.vec_to_point(p)
            self.add_point_from_msg(result)
        elif isinstance(p, Point):
            # Handle Point message
            self.add_point_from_msg(p)
        elif isinstance(p, Pose):
            # Handle Pose message
            self.marker_.id = self.ros_publisher_.get_id()
            self.marker_.pose = p
            self.ros_publisher_.add(self.marker_)

    def add_point_from_msg(self, point):
        # Request an ID
        self.marker_.id = self.ros_publisher_.get_id()

        # Set position
        self.marker_.pose.position = point

        # Add to publisher
        self.ros_publisher_.add(self.marker_)

    def set_z(self, z):
        self.marker_.pose.position.z = z

    @staticmethod
    def get_marker_type(marker_type):
        marker_types = {
            "CUBE": Marker.CUBE,
            "ARROW": Marker.ARROW,
            "SPHERE": Marker.SPHERE,
            "POINTS": Marker.POINTS,
            "CYLINDER": Marker.CYLINDER
        }
        return marker_types.get(marker_type, Marker.CUBE)


class ROSMultiplePointMarker(ROSMarker):
    def __init__(self, ros_publisher, frame_id, marker_type="POINTS"):
        super().__init__(ros_publisher, frame_id)

        self.marker_.type = self.get_multiple_marker_type(marker_type)

        # Set default scale
        self.marker_.scale.x = 0.5
        self.marker_.scale.y = 0.5

        # Default color (green)
        self.set_color(0, 1, 0)

        # Default orientation (identity quaternion)
        self.marker_.pose.orientation.x = 0.0
        self.marker_.pose.orientation.y = 0.0
        self.marker_.pose.orientation.z = 0.0
        self.marker_.pose.orientation.w = 1.0

    def add_point_marker(self, p):
        if isinstance(p, np.ndarray):
            # Handle numpy array
            result = self.vec_to_point(p)
            self.marker_.points.append(result)
        elif isinstance(p, Point):
            # Handle Point message
            self.marker_.points.append(p)
        elif isinstance(p, Pose):
            # Handle Pose message
            result = Point()
            result.x = p.position.x
            result.y = p.position.y
            result.z = p.position.z
            self.marker_.points.append(result)

    def finish_points(self):
        self.marker_.id = self.ros_publisher_.get_id()
        self.ros_publisher_.add(self.marker_)

    @staticmethod
    def get_multiple_marker_type(marker_type):
        marker_types = {
            "CUBE": Marker.CUBE_LIST,
            "SPHERE": Marker.SPHERE_LIST,
            "POINTS": Marker.POINTS
        }
        return marker_types.get(marker_type, Marker.CUBE_LIST)


class ROSTextMarker(ROSMarker):
    def __init__(self, ros_publisher, frame_id):
        super().__init__(ros_publisher, frame_id)

        self.marker_.type = Marker.TEXT_VIEW_FACING

        # Set default scale (z sets the size of an uppercase "A")
        self.marker_.scale.x = 1.0
        self.marker_.scale.y = 1.0
        self.marker_.scale.z = 0.2

        # Default text
        self.marker_.text = "TEST"

        # Default color (green)
        self.set_color(0, 1, 0)

        # Default orientation (identity quaternion)
        self.marker_.pose.orientation.x = 0.0
        self.marker_.pose.orientation.y = 0.0
        self.marker_.pose.orientation.z = 0.0
        self.marker_.pose.orientation.w = 1.0

    def add_point_marker(self, p):
        if isinstance(p, np.ndarray):
            # Handle numpy array
            result = self.vec_to_point(p)
            self.add_point_from_msg(result)
        elif isinstance(p, Point):
            # Handle Point message
            self.add_point_from_msg(p)
        elif isinstance(p, Pose):
            # Handle Pose message
            result = Point()
            result.x = p.position.x
            result.y = p.position.y
            result.z = p.position.z
            self.add_point_from_msg(result)

    def add_point_from_msg(self, point):
        # Request an ID
        self.marker_.id = self.ros_publisher_.get_id()

        # Set position
        self.marker_.pose.position = point

        # Add to publisher
        self.ros_publisher_.add(self.marker_)

    def set_z(self, z):
        self.marker_.pose.position.z = z

    def set_text(self, text):
        self.marker_.text = text

    def set_scale(self, z):
        self.marker_.scale.z = z  # Only scale "z" is used for text


class ROSModelMarker(ROSMarker):
    def __init__(self, ros_publisher, frame_id, model_path):
        super().__init__(ros_publisher, frame_id)

        self.marker_.type = Marker.MESH_RESOURCE
        self.marker_.mesh_resource = model_path

        # Set default scale
        self.marker_.scale.x = 1.0
        self.marker_.scale.y = 1.0
        self.marker_.scale.z = 1.0

        # Default color (green)
        self.set_color(0, 1, 0)

        # Default orientation (identity quaternion)
        self.marker_.pose.orientation.x = 0.0
        self.marker_.pose.orientation.y = 0.0
        self.marker_.pose.orientation.z = 0.0
        self.marker_.pose.orientation.w = 1.0

    def add_point_marker(self, p, z=None):
        if isinstance(p, np.ndarray) and len(p) == 2 and z is not None:
            # Handle 2D point with z
            result = Point()
            result.x = p[0]
            result.y = p[1]
            result.z = z
            self.add_point_from_msg(result)
        elif isinstance(p, np.ndarray) and len(p) == 3:
            # Handle 3D point
            result = self.vec_to_point(p)
            self.add_point_from_msg(result)
        elif isinstance(p, Point):
            # Handle Point message
            self.add_point_from_msg(p)
        elif isinstance(p, Pose):
            # Handle Pose message
            result = Point()
            result.x = p.position.x
            result.y = p.position.y
            result.z = p.position.z
            self.add_point_from_msg(result)

    def add_point_from_msg(self, point):
        # Request an ID
        self.marker_.id = self.ros_publisher_.get_id()

        # Set position
        self.marker_.pose.position = point

        # Add to publisher
        self.ros_publisher_.add(self.marker_)