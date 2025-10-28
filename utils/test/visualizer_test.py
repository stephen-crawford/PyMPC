import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import math

# Import your modules - adjust import paths as needed
from utils.visualizer_compat import ROSMarker, ROSMarkerPublisher, ROSLine, ROSPointMarker, ROSMultiplePointMarker, ROSTextMarker, \
    ROSModelMarker, Colormap


class TestROSMarker(unittest.TestCase):
    def setUp(self):
        # Mock the publisher
        self.publisher = MagicMock()
        self.frame_id = "map"
        self.marker = ROSMarker(self.publisher, self.frame_id)

    def test_init(self):
        self.assertEqual(self.marker.marker_.header.frame_id, self.frame_id)

    def test_set_color(self):
        self.marker.set_color(0.5, 0.6, 0.7, 0.8)
        self.assertEqual(self.marker.marker_.color.r, 0.5)
        self.assertEqual(self.marker.marker_.color.g, 0.6)
        self.assertEqual(self.marker.marker_.color.b, 0.7)
        self.assertEqual(self.marker.marker_.color.a, 0.8)

    def test_set_color_ratio(self):
        # Test a few key points in the color range
        self.marker.set_color_ratio(0.0)  # Should be red
        self.assertAlmostEqual(self.marker.marker_.color.r, 1.0)
        self.assertAlmostEqual(self.marker.marker_.color.g, 0.0)
        self.assertAlmostEqual(self.marker.marker_.color.b, 0.0)

        self.marker.set_color_ratio(1 / 6)  # Should be yellow
        self.assertAlmostEqual(self.marker.marker_.color.r, 1.0)
        self.assertAlmostEqual(self.marker.marker_.color.g, 1.0)
        self.assertAlmostEqual(self.marker.marker_.color.b, 0.0)

        self.marker.set_color_ratio(0.5)  # Should be cyan
        self.assertAlmostEqual(self.marker.marker_.color.r, 0.0)
        self.assertAlmostEqual(self.marker.marker_.color.g, 1.0)
        self.assertAlmostEqual(self.marker.marker_.color.b, 1.0)

    def test_set_color_int(self):
        self.marker.set_color_int(0, 1.0, Colormap.VIRIDIS)
        # The last color in VIRIDIS palette (because it's inverted)
        viridis_len = len(ROSMarker.VIRIDIS) // 3
        red = ROSMarker.VIRIDIS[(viridis_len - 1) * 3 + 0] / 256.0
        green = ROSMarker.VIRIDIS[(viridis_len - 1) * 3 + 1] / 256.0
        blue = ROSMarker.VIRIDIS[(viridis_len - 1) * 3 + 2] / 256.0

        self.assertAlmostEqual(self.marker.marker_.color.r, red)
        self.assertAlmostEqual(self.marker.marker_.color.g, green)
        self.assertAlmostEqual(self.marker.marker_.color.b, blue)

    def test_set_scale(self):
        self.marker.set_scale(0.5)
        self.assertEqual(self.marker.marker_.scale.x, 0.5)

        self.marker.set_scale(1.0, 2.0)
        self.assertEqual(self.marker.marker_.scale.x, 1.0)
        self.assertEqual(self.marker.marker_.scale.y, 2.0)

        self.marker.set_scale(3.0, 4.0, 5.0)
        self.assertEqual(self.marker.marker_.scale.x, 3.0)
        self.assertEqual(self.marker.marker_.scale.y, 4.0)
        self.assertEqual(self.marker.marker_.scale.z, 5.0)

    @patch('utils.visualizer_compat.time.time')
    def test_set_orientation(self, mock_quaternion):
        # Test with float (yaw angle)
        mock_quaternion.return_value = [0.1, 0.2, 0.3, 0.4]
        self.marker.set_orientation(1.57)  # ~90 degrees
        mock_quaternion.assert_called_with(0, 0, 1.57)
        self.assertEqual(self.marker.marker_.pose.orientation.x, 0.1)
        self.assertEqual(self.marker.marker_.pose.orientation.y, 0.2)
        self.assertEqual(self.marker.marker_.pose.orientation.z, 0.3)
        self.assertEqual(self.marker.marker_.pose.orientation.w, 0.4)

        # Test with Quaternion
        q = Quaternion()
        q.x, q.y, q.z, q.w = 0.5, 0.6, 0.7, 0.8
        self.marker.set_orientation(q)
        self.assertEqual(self.marker.marker_.pose.orientation.x, 0.5)
        self.assertEqual(self.marker.marker_.pose.orientation.y, 0.6)
        self.assertEqual(self.marker.marker_.pose.orientation.z, 0.7)
        self.assertEqual(self.marker.marker_.pose.orientation.w, 0.8)

        # Test with list/tuple
        self.marker.set_orientation([0.9, 0.8, 0.7, 0.6])
        self.assertEqual(self.marker.marker_.pose.orientation.x, 0.9)
        self.assertEqual(self.marker.marker_.pose.orientation.y, 0.8)
        self.assertEqual(self.marker.marker_.pose.orientation.z, 0.7)
        self.assertEqual(self.marker.marker_.pose.orientation.w, 0.6)

    def test_set_lifetime(self):
        with patch('utils.visualizer_compat.time.time') as mock_time:
            mock_duration_instance = MagicMock()
            mock_duration.return_value = mock_duration_instance

            self.marker.set_lifetime(10.0)
            mock_duration.assert_called_with(seconds=10.0)
            mock_duration_instance.to_msg.assert_called_once()

    def test_set_action_delete(self):
        self.marker.set_action_delete()
        self.assertEqual(self.marker.marker_.action, Marker.DELETE)

    def test_vec_to_point(self):
        v = np.array([1.0, 2.0, 3.0])
        p = ROSMarker.vec_to_point(v)
        self.assertEqual(p.x, 1.0)
        self.assertEqual(p.y, 2.0)
        self.assertEqual(p.z, 3.0)

    def test_get_color_from_range_int(self):
        # Test getting a color from the VIRIDIS colormap
        r, g, b = ROSMarker.get_color_from_range_int(0, Colormap.VIRIDIS)
        viridis_len = len(ROSMarker.VIRIDIS) // 3
        red = ROSMarker.VIRIDIS[(viridis_len - 1) * 3 + 0] / 256.0
        green = ROSMarker.VIRIDIS[(viridis_len - 1) * 3 + 1] / 256.0
        blue = ROSMarker.VIRIDIS[(viridis_len - 1) * 3 + 2] / 256.0

        self.assertAlmostEqual(r, red)
        self.assertAlmostEqual(g, green)
        self.assertAlmostEqual(b, blue)

        # Test INFERNO colormap
        r, g, b = ROSMarker.get_color_from_range_int(0, Colormap.INFERNO)
        inferno_len = len(ROSMarker.INFERNO) // 3
        red = ROSMarker.INFERNO[(inferno_len - 1) * 3 + 0] / 256.0
        green = ROSMarker.INFERNO[(inferno_len - 1) * 3 + 1] / 256.0
        blue = ROSMarker.INFERNO[(inferno_len - 1) * 3 + 2] / 256.0

        self.assertAlmostEqual(r, red)
        self.assertAlmostEqual(g, green)
        self.assertAlmostEqual(b, blue)

    def test_get_colors(self):
        # Test getting different colormaps
        self.assertEqual(ROSMarker.get_colors(Colormap.VIRIDIS), ROSMarker.VIRIDIS)
        self.assertEqual(ROSMarker.get_colors(Colormap.INFERNO), ROSMarker.INFERNO)
        self.assertEqual(ROSMarker.get_colors(Colormap.BRUNO), ROSMarker.BRUNO)

        # Test invalid colormap
        with self.assertRaises(RuntimeError):
            ROSMarker.get_colors(999)  # Invalid colormap value

    def test_get_color_from_range(self):
        # Test a few key points in the rainbow
        r, g, b = ROSMarker.get_color_from_range(0.0)  # red
        self.assertAlmostEqual(r, 255 / 256.0)
        self.assertAlmostEqual(g, 0.0)
        self.assertAlmostEqual(b, 0.0)

        r, g, b = ROSMarker.get_color_from_range(1 / 6)  # yellow
        self.assertAlmostEqual(r, 255 / 256.0)
        self.assertAlmostEqual(g, 255 / 256.0)
        self.assertAlmostEqual(b, 0.0)

        r, g, b = ROSMarker.get_color_from_range(2 / 6)  # green
        self.assertAlmostEqual(r, 0.0)
        self.assertAlmostEqual(g, 255 / 256.0)
        self.assertAlmostEqual(b, 0.0)


class TestROSMarkerPublisher(unittest.TestCase):
    def setUp(self):
        self.node = MagicMock(spec=Node)
        self.node.create_publisher.return_value = MagicMock()
        self.topic_name = "test_markers"
        self.frame_id = "test_frame"
        self.publisher = ROSMarkerPublisher(self.node, self.topic_name, self.frame_id)

    def test_init(self):
        self.assertEqual(self.publisher.frame_id_, self.frame_id)
        self.assertEqual(self.publisher.max_size_, 1000)
        self.assertEqual(self.publisher.id_, 0)
        self.assertEqual(self.publisher.prev_id_, 0)
        self.assertEqual(self.publisher.topic_name_, self.topic_name)
        self.node.create_publisher.assert_called_with(MarkerArray, self.topic_name, 3)

    def test_add(self):
        marker = MagicMock()
        marker.id = 5
        self.publisher.add(marker)
        self.assertIn(marker, self.publisher.marker_list_.markers)

    def test_add_exceeding_max_size(self):
        marker = MagicMock()
        marker.id = 1001  # Exceeds default max_size of 1000
        self.publisher.add(marker)
        # Check if max_size was increased (should be 1.5x the original)
        self.assertGreaterEqual(self.publisher.max_size_, 1500)

    def test_get_new_line(self):
        line = self.publisher.get_new_line()
        self.assertIsInstance(line, ROSLine)
        self.assertIn(line, self.publisher.ros_markers_)

    def test_get_new_point_marker(self):
        point = self.publisher.get_new_point_marker()
        self.assertIsInstance(point, ROSPointMarker)
        self.assertIn(point, self.publisher.ros_markers_)

        # Test with custom marker type
        sphere = self.publisher.get_new_point_marker("SPHERE")
        self.assertIsInstance(sphere, ROSPointMarker)
        self.assertEqual(sphere.marker_.type, Marker.SPHERE)

    def test_get_new_multiple_point_marker(self):
        points = self.publisher.get_new_multiple_point_marker()
        self.assertIsInstance(points, ROSMultiplePointMarker)
        self.assertIn(points, self.publisher.ros_markers_)

    def test_get_new_text_marker(self):
        text = self.publisher.get_new_text_marker()
        self.assertIsInstance(text, ROSTextMarker)
        self.assertIn(text, self.publisher.ros_markers_)

    def test_get_new_model_marker(self):
        model_path = "package://my_package/meshes/model.dae"
        model = self.publisher.get_new_model_marker(model_path)
        self.assertIsInstance(model, ROSModelMarker)
        self.assertIn(model, self.publisher.ros_markers_)
        self.assertEqual(model.marker_.mesh_resource, model_path)

    def test_publish(self):
        line = self.publisher.get_new_line()
        line.stamp = MagicMock()

        self.publisher.publish()

        line.stamp.assert_called_once()
        self.publisher.pub_.publish.assert_called_once()
        # Check that markers were cleared
        self.assertEqual(len(self.publisher.marker_list_.markers), 0)
        self.assertEqual(len(self.publisher.ros_markers_), 0)
        # Check that ids were properly updated
        self.assertEqual(self.publisher.prev_id_, self.publisher.id_)
        self.assertEqual(self.publisher.id_, 0)

    def test_publish_keep_markers(self):
        line = self.publisher.get_new_line()
        line.stamp = MagicMock()

        self.publisher.publish(keep_markers=True)

        line.stamp.assert_called_once()
        self.publisher.pub_.publish.assert_called_once()
        # Check that markers were kept
        self.assertNotEqual(len(self.publisher.ros_markers_), 0)

    def test_publish_remove_extra_markers(self):
        # Simulate having published more markers previously
        self.publisher.prev_id_ = 5
        self.publisher.id_ = 2

        self.publisher.publish()

        # Check that DELETE markers were added for the extra markers
        published_markers = self.publisher.pub_.publish.call_args[0][0]
        delete_markers_count = sum(1 for m in published_markers.markers if m.action == Marker.DELETE)
        self.assertEqual(delete_markers_count, 3)  # Should delete markers 2, 3, 4

    def test_get_id(self):
        id1 = self.publisher.get_id()
        id2 = self.publisher.get_id()

        self.assertEqual(id1, 0)
        self.assertEqual(id2, 1)
        self.assertEqual(self.publisher.id_, 2)


class TestROSLine(unittest.TestCase):
    def setUp(self):
        self.publisher = MagicMock()
        # Mock get_id to return predictable values
        self.publisher.get_id.return_value = 42
        self.publisher.add = MagicMock()
        self.frame_id = "map"
        self.line = ROSLine(self.publisher, self.frame_id)

    def test_init(self):
        self.assertEqual(self.line.marker_.type, Marker.LINE_LIST)
        self.assertEqual(self.line.marker_.scale.x, 0.5)
        self.assertEqual(self.line.marker_.color.r, 1.0)
        self.assertEqual(self.line.marker_.color.g, 0.0)
        self.assertEqual(self.line.marker_.color.b, 0.0)

    def test_add_line_numpy_2d(self):
        p1 = np.array([1.0, 2.0])
        p2 = np.array([3.0, 4.0])
        z = 0.5

        self.line.add_line(p1, p2, z)

        self.publisher.get_id.assert_called_once()
        self.publisher.add.assert_called_once()

        # Check that the points were added correctly
        marker = self.publisher.add.call_args[0][0]
        self.assertEqual(marker.id, 42)
        self.assertEqual(len(marker.points), 2)
        self.assertEqual(marker.points[0].x, 1.0)
        self.assertEqual(marker.points[0].y, 2.0)
        self.assertEqual(marker.points[0].z, 0.5)
        self.assertEqual(marker.points[1].x, 3.0)
        self.assertEqual(marker.points[1].y, 4.0)
        self.assertEqual(marker.points[1].z, 0.5)

    def test_add_line_numpy_3d(self):
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([4.0, 5.0, 6.0])

        self.line.add_line(p1, p2)

        self.publisher.get_id.assert_called_once()
        self.publisher.add.assert_called_once()

        # Check that the points were added correctly
        marker = self.publisher.add.call_args[0][0]
        self.assertEqual(marker.id, 42)
        self.assertEqual(len(marker.points), 2)
        self.assertEqual(marker.points[0].x, 1.0)
        self.assertEqual(marker.points[0].y, 2.0)
        self.assertEqual(marker.points[0].z, 3.0)
        self.assertEqual(marker.points[1].x, 4.0)
        self.assertEqual(marker.points[1].y, 5.0)
        self.assertEqual(marker.points[1].z, 6.0)

    def test_add_line_points(self):
        p1 = Point()
        p1.x, p1.y, p1.z = 1.0, 2.0, 3.0
        p2 = Point()
        p2.x, p2.y, p2.z = 4.0, 5.0, 6.0

        self.line.add_line(p1, p2)

        self.publisher.get_id.assert_called_once()
        self.publisher.add.assert_called_once()

        # Check that the points were added correctly
        marker = self.publisher.add.call_args[0][0]
        self.assertEqual(marker.id, 42)
        self.assertEqual(len(marker.points), 2)
        self.assertEqual(marker.points[0].x, 1.0)
        self.assertEqual(marker.points[0].y, 2.0)
        self.assertEqual(marker.points[0].z, 3.0)
        self.assertEqual(marker.points[1].x, 4.0)
        self.assertEqual(marker.points[1].y, 5.0)
        self.assertEqual(marker.points[1].z, 6.0)

    def test_add_broken_line(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([10.0, 0.0, 0.0])  # 10 units in x direction
        dist = 1.0  # 1 unit segments

        self.line.add_broken_line(p1, p2, dist)

        # Should create 5 line segments (10 units / 2 units per segment)
        self.assertEqual(self.publisher.add.call_count, 5)


class TestROSPointMarker(unittest.TestCase):
    def setUp(self):
        self.publisher = MagicMock()
        self.publisher.get_id.return_value = 42
        self.frame_id = "map"
        self.point_marker = ROSPointMarker(self.publisher, self.frame_id, "CUBE")

    def test_init(self):
        self.assertEqual(self.point_marker.marker_.type, Marker.CUBE)
        self.assertEqual(self.point_marker.marker_.scale.x, 0.5)
        self.assertEqual(self.point_marker.marker_.scale.y, 0.5)

        # Test different marker types
        sphere = ROSPointMarker(self.publisher, self.frame_id, "SPHERE")
        self.assertEqual(sphere.marker_.type, Marker.SPHERE)

        arrow = ROSPointMarker(self.publisher, self.frame_id, "ARROW")
        self.assertEqual(arrow.marker_.type, Marker.ARROW)

    def test_add_point_marker_numpy_2d(self):
        p = np.array([1.0, 2.0])
        z = 3.0

        self.point_marker.add_point_marker(p, z)

        self.publisher.get_id.assert_called_once()
        self.publisher.add.assert_called_once()

        # Check that the point was added correctly
        marker = self.publisher.add.call_args[0][0]
        self.assertEqual(marker.id, 42)
        self.assertEqual(marker.pose.position.x, 1.0)
        self.assertEqual(marker.pose.position.y, 2.0)
        self.assertEqual(marker.pose.position.z, 3.0)

    def test_add_point_marker_numpy_3d(self):
        p = np.array([1.0, 2.0, 3.0])

        self.point_marker.add_point_marker(p)

        self.publisher.get_id.assert_called_once()
        self.publisher.add.assert_called_once()

        # Check that the point was added correctly
        marker = self.publisher.add.call_args[0][0]
        self.assertEqual(marker.id, 42)
        self.assertEqual(marker.pose.position.x, 1.0)
        self.assertEqual(marker.pose.position.y, 2.0)
        self.assertEqual(marker.pose.position.z, 3.0)

    def test_add_point_marker_pose(self):
        pose = Pose()
        pose.position.x = 1.0
        pose.position.y = 2.0
        pose.position.z = 3.0
        pose.orientation.x = 0.1
        pose.orientation.y = 0.2
        pose.orientation.z = 0.3
        pose.orientation.w = 0.4

        self.point_marker.add_point_marker(pose)

        self.publisher.get_id.assert_called_once()
        self.publisher.add.assert_called_once()

        # Check that the pose was added correctly
        marker = self.publisher.add.call_args[0][0]
        self.assertEqual(marker.id, 42)
        self.assertEqual(marker.pose.position.x, 1.0)
        self.assertEqual(marker.pose.position.y, 2.0)
        self.assertEqual(marker.pose.position.z, 3.0)
        self.assertEqual(marker.pose.orientation.x, 0.1)
        self.assertEqual(marker.pose.orientation.y, 0.2)
        self.assertEqual(marker.pose.orientation.z, 0.3)
        self.assertEqual(marker.pose.orientation.w, 0.4)

    def test_set_z(self):
        self.point_marker.set_z(5.0)
        self.assertEqual(self.point_marker.marker_.pose.position.z, 5.0)

    def test_get_marker_type(self):
        self.assertEqual(ROSPointMarker.get_marker_type("CUBE"), Marker.CUBE)
        self.assertEqual(ROSPointMarker.get_marker_type("ARROW"), Marker.ARROW)
        self.assertEqual(ROSPointMarker.get_marker_type("SPHERE"), Marker.SPHERE)
        self.assertEqual(ROSPointMarker.get_marker_type("POINTS"), Marker.POINTS)
        self.assertEqual(ROSPointMarker.get_marker_type("CYLINDER"), Marker.CYLINDER)
        # Default to CUBE for unknown types
        self.assertEqual(ROSPointMarker.get_marker_type("UNKNOWN"), Marker.CUBE)


class TestROSMultiplePointMarker(unittest.TestCase):
    def setUp(self):
        self.publisher = MagicMock()
        self.publisher.get_id.return_value = 42
        self.frame_id = "map"
        self.points_marker = ROSMultiplePointMarker(self.publisher, self.frame_id)

    def test_init(self):
        self.assertEqual(self.points_marker.marker_.type, Marker.POINTS)
        self.assertEqual(self.points_marker.marker_.scale.x, 0.5)
        self.assertEqual(self.points_marker.marker_.scale.y, 0.5)
        self.assertEqual(self.points_marker.marker_.color.g, 1.0)  # Default is green

        # Test different marker types
        cubes = ROSMultiplePointMarker(self.publisher, self.frame_id, "CUBE")
        self.assertEqual(cubes.marker_.type, Marker.CUBE_LIST)

        spheres = ROSMultiplePointMarker(self.publisher, self.frame_id, "SPHERE")
        self.assertEqual(spheres.marker_.type, Marker.SPHERE_LIST)

    def test_add_point_marker_numpy(self):
        p = np.array([1.0, 2.0, 3.0])

        self.points_marker.add_point_marker(p)

        self.assertEqual(len(self.points_marker.marker_.points), 1)
        self.assertEqual(self.points_marker.marker_.points[0].x, 1.0)
        self.assertEqual(self.points_marker.marker_.points[0].y, 2.0)
        self.assertEqual(self.points_marker.marker_.points[0].z, 3.0)

    def test_add_point_marker_point(self):
        p = Point()
        p.x, p.y, p.z = 1.0, 2.0, 3.0

        self.points_marker.add_point_marker(p)

        self.assertEqual(len(self.points_marker.marker_.points), 1)
        self.assertEqual(self.points_marker.marker_.points[0].x, 1.0)
        self.assertEqual(self.points_marker.marker_.points[0].y, 2.0)
        self.assertEqual(self.points_marker.marker_.points[0].z, 3.0)

    def test_add_point_marker_pose(self):
        p = Pose()
        p.position.x, p.position.y, p.position.z = 1.0, 2.0, 3.0

        self.points_marker.add_point_marker(p)

        self.assertEqual(len(self.points_marker.marker_.points), 1)
        self.assertEqual(self.points_marker.marker_.points[0].x, 1.0)
        self.assertEqual(self.points_marker.marker_.points[0].y, 2.0)
        self.assertEqual(self.points_marker.marker_.points[0].z, 3.0)

    def test_finish_points(self):
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([4.0, 5.0, 6.0])

        self.points_marker.add_point_marker(p1)
        self.points_marker.add_point_marker(p2)
        self.points_marker.finish_points()

        self.publisher.get_id.assert_called_once()
        self.publisher.add.assert_called_once()

        # Check that both points were in the published marker
        marker = self.publisher.add.call_args[0][0]
        self.assertEqual(marker.id, 42)
        self.assertEqual(len(marker.points), 2)
        self.assertEqual(marker.points[0].x, 1.0)
        self.assertEqual(marker.points[0].y, 2.0)
        self.assertEqual(marker.points[0].z, 3.0)
        self.assertEqual(marker.points[1].x, 4.0)
        self.assertEqual(marker.points[1].y, 5.0)
        self.assertEqual(marker.points[1].z, 6.0)

    def test_get_multiple_marker_type(self):
        self.assertEqual(ROSMultiplePointMarker.get_multiple_marker_type("CUBE"), Marker.CUBE_LIST)
        self.assertEqual(ROSMultiplePointMarker.get_multiple_marker_type("SPHERE"), Marker.SPHERE_LIST)
        self.assertEqual(ROSMultiplePointMarker.get_multiple_marker_type("POINTS"), Marker.POINTS)
        # Default to CUBE_LIST for unknown types
        self.assertEqual(ROSMultiplePointMarker.get_multiple_marker_type("UNKNOWN"), Marker.CUBE_LIST)

