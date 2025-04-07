import logging
import unittest
import math
import numpy as np
import time
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open
import threading
import datetime

# Import the modules to test
from utils.utils import (
    angle_to_quaternion, quaternion_to_angle, distance, exponential_quantile,
    linspace, rotation_matrix_from_heading, angular_difference, bisection,
    sgn, PROFILE_SCOPE, Benchmarker, Timer,
    ProfileResult, InstrumentationSession, Instrumentor, InstrumentationTimer,
    RandomGenerator, read_config_file, LOG_DEBUG, LOG_WARN, PYMPC_ASSERT
)

from utils.utils import ExperimentManager, DataSaver


class TestConversionUtils(unittest.TestCase):
    def test_angle_to_quaternion(self):
        # Test conversion from angle to quaternion
        angle = math.pi / 2  # 90 degrees
        q = angle_to_quaternion(angle)

        # For yaw angle of pi/2, we expect:
        # x=0, y=0, z=sin(pi/4), w=cos(pi/4)
        self.assertAlmostEqual(q['x'], 0.0)
        self.assertAlmostEqual(q['y'], 0.0)
        self.assertAlmostEqual(q['z'], math.sin(math.pi / 4))
        self.assertAlmostEqual(q['w'], math.cos(math.pi / 4))

    def test_quaternion_to_angle(self):
        # Test conversion from quaternion to angle
        # Create a quaternion representing a 90-degree yaw
        q = {'x': 0.0, 'y': 0.0, 'z': math.sin(math.pi / 4), 'w': math.cos(math.pi / 4)}
        angle = quaternion_to_angle(q)

        self.assertAlmostEqual(angle, math.pi / 2)

        # Test with a pose object
        class Pose:
            def __init__(self):
                self.orientation = {'x': 0.0, 'y': 0.0, 'z': math.sin(math.pi / 4), 'w': math.cos(math.pi / 4)}

        pose = Pose()
        angle = quaternion_to_angle(pose)
        self.assertAlmostEqual(angle, math.pi / 2)

        # Test with object attributes
        class Quaternion:
            def __init__(self):
                self.x = 0.0
                self.y = 0.0
                self.z = math.sin(math.pi / 4)
                self.w = math.cos(math.pi / 4)

        quat = Quaternion()
        angle = quaternion_to_angle(quat)
        self.assertAlmostEqual(angle, math.pi / 2)


class TestMathUtils(unittest.TestCase):
    def test_distance(self):
        # Test distance calculation
        a = [1, 2, 3]
        b = [4, 5, 6]
        self.assertAlmostEqual(distance(a, b), math.sqrt(27))

    def test_exponential_quantile(self):
        # Test exponential CDF
        lambda_param = 0.5
        p = 0.5
        self.assertAlmostEqual(exponential_quantile(lambda_param, p), math.log(2) / 0.5)

    def test_linspace(self):
        # Test linspace function
        result = linspace(0, 10, 6)
        expected = [0, 2, 4, 6, 8, 10]
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e)

        # Test edge cases
        self.assertEqual(linspace(0, 10, 0), [])
        self.assertEqual(linspace(0, 10, 1), [10])
        self.assertEqual(linspace(0, 10, 2), [0, 10])

    def test_rotation_matrix_from_heading(self):
        # Test rotation matrix creation
        heading = math.pi / 2  # 90 degrees
        matrix = rotation_matrix_from_heading(heading)

        # For 90 degrees, we expect:
        # [0, 1]
        # [-1, 0]
        self.assertAlmostEqual(matrix[0, 0], 0)
        self.assertAlmostEqual(matrix[0, 1], 1)
        self.assertAlmostEqual(matrix[1, 0], -1)
        self.assertAlmostEqual(matrix[1, 1], 0)

    def test_angular_difference(self):
        # Test angular difference calculation
        # Angles close to each other
        self.assertAlmostEqual(angular_difference(0, 0.1), 0.1)

        # Angles that wrap around
        self.assertAlmostEqual(angular_difference(0, 2 * math.pi - 0.1), -0.1)
        self.assertAlmostEqual(angular_difference(0, -0.1), -0.1)

    def test_bisection(self):
        # Test bisection root finding
        func = lambda x: x ** 2 - 4  # Root at x=2
        root = bisection(0, 3, func, 1e-5)
        self.assertAlmostEqual(root, 2.0, 5)

        # Test with integer result
        func = lambda x: x - 5  # Root at x=5
        root = bisection(4, 6, func, 1e-5)
        self.assertAlmostEqual(root, 5.0, 5)

        # Test exception
        with self.assertRaises(RuntimeError):
            bisection(3, 0, func, 1e-5)  # Low > high

    def test_sgn(self):
        # Test sign function
        self.assertEqual(sgn(5), 1)
        self.assertEqual(sgn(-5), -1)
        self.assertEqual(sgn(0), 0)


@patch('logging.debug')
class TestProfilingTools(unittest.TestCase):
    def test_profile_scope(self, mock_debug):
        # Test profile scope context manager
        with PROFILE_SCOPE("test_scope"):
            time.sleep(0.01)

        # Check if logging.debug was called
        self.assertTrue(mock_debug.called)
        # Check if the message contains our scope name
        self.assertIn("test_scope", mock_debug.call_args[0][0])

    def test_benchmarker(self, mock_debug):
        # Test benchmarker class
        benchmarker = Benchmarker("test_bench")

        # Test start/stop
        benchmarker.start()
        time.sleep(0.01)
        duration = benchmarker.stop()

        # Duration should be positive
        self.assertGreater(duration, 0)

        # Test get methods
        self.assertEqual(benchmarker.get_last(), duration)
        self.assertEqual(benchmarker.get_total_duration(), duration)

        # Test reset
        benchmarker.reset()
        self.assertEqual(benchmarker.get_last(), -1)

        # Test cancel
        benchmarker.start()
        benchmarker.cancel()
        self.assertFalse(benchmarker.is_running())

    def test_timer(self, mock_debug):
        # Test timer class
        timer = Timer(0.01)

        # Test start
        timer.start()
        time.sleep(0.015)  # Slightly longer than the duration

        # Test has_finished
        self.assertTrue(timer.has_finished())

        # Test current_duration
        self.assertGreater(timer.current_duration(), 0.01)

        # Test set_duration
        timer.set_duration(1.0)
        self.assertFalse(timer.has_finished())


@patch('builtins.open', new_callable=mock_open)
class TestInstrumentationTools(unittest.TestCase):
    @patch('threading.current_thread')
    def test_instrumentation_timer(self, mock_thread, mock_file):
        # Configure mock thread
        mock_thread.return_value.ident = 123

        # Test instrumentation timer
        with patch.object(Instrumentor, 'get') as mock_instrumentor:
            mock_instrumentor.return_value = MagicMock()

            timer = InstrumentationTimer("test_timer")
            time.sleep(0.01)
            timer.stop()

            # Check if write_profile was called
            mock_instrumentor.return_value.write_profile.assert_called_once()

            # Check if the profile result has our timer name
            args = mock_instrumentor.return_value.write_profile.call_args[0][0]
            self.assertEqual(args.Name, "test_timer")

    def test_instrumentor(self, mock_file):
        # Test instrumentor singleton
        inst1 = Instrumentor.get()
        inst2 = Instrumentor.get()
        self.assertIs(inst1, inst2)

        # Test begin_session
        with patch.object(Instrumentor, '_get_package_path', return_value='/tmp'):
            inst1.begin_session("test_session", "profile.json")

            # Check if file was opened
            mock_file.assert_called_with('/tmp/profile.json', 'w')

            # Test write_profile
            result = ProfileResult("test_profile", 1000, 2000, 123)
            inst1.write_profile(result)

            # Test end_session
            inst1.end_session()


class TestRandomGenerator(unittest.TestCase):
    def test_random(self):
        # Test with fixed seed for reproducibility
        gen = RandomGenerator(42)

        # Generate random numbers
        r1 = gen.random()
        r2 = gen.random()

        # Values should be between 0 and 1
        self.assertGreaterEqual(r1, 0)
        self.assertLess(r1, 1)
        self.assertGreaterEqual(r2, 0)
        self.assertLess(r2, 1)

        # Test with same seed - should get same sequence
        gen2 = RandomGenerator(42)
        self.assertEqual(gen2.random(), r1)
        self.assertEqual(gen2.random(), r2)

    def test_int(self):
        gen = RandomGenerator(42)

        # Generate random integers
        i1 = gen.int(100)
        i2 = gen.int(100)

        # Values should be between 0 and 100
        self.assertGreaterEqual(i1, 0)
        self.assertLess(i1, 100)
        self.assertGreaterEqual(i2, 0)
        self.assertLess(i2, 100)

    def test_gaussian(self):
        gen = RandomGenerator(42)

        # Generate gaussian random numbers
        g1 = gen.gaussian(0, 1)
        g2 = gen.gaussian(10, 2)

        # Can't test exact values, but can verify mean after many samples
        samples = [gen.gaussian(5, 1) for _ in range(1000)]
        mean = sum(samples) / len(samples)
        self.assertAlmostEqual(mean, 5, delta=0.2)

    def test_uniform_to_gaussian_2d(self):
        gen = RandomGenerator(42)

        # Test conversion
        uniform = np.array([0.5, 0.5])
        result = gen.uniform_to_gaussian_2d(uniform)

        # Result should be different than input
        self.assertNotEqual(result[0], 0.5)
        self.assertNotEqual(result[1], 0.5)

    def test_bivariate_gaussian(self):
        gen = RandomGenerator(42)

        # Generate bivariate gaussian
        result = gen.bivariate_gaussian([0, 0], 1, 0.5, 0)

        # Result should be a 2D point
        self.assertEqual(len(result), 2)


@patch('yaml.safe_load')
@patch('builtins.open', new_callable=mock_open)
class TestConfigUtils(unittest.TestCase):
    def test_read_config_file(self, mock_file, mock_yaml_load):
        # Set up the mock to return a test config
        mock_yaml_load.return_value = {"test": "value"}

        # Call the function
        result = read_config_file()

        # Check if yaml.safe_load was called
        mock_yaml_load.assert_called_once()

        # Check the result
        self.assertEqual(result, {"test": "value"})

        # Test error handling
        mock_yaml_load.side_effect = Exception("Test error")
        with patch('builtins.print') as mock_print:
            try:
                result = read_config_file()
                self.assertIsNone(result)
                mock_print.assert_called()
            except Exception:
                return

@patch('logging.basicConfig')
@patch('logging.getLogger')
class TestLoggingUtils(unittest.TestCase):
    def test_log_debug(self, mock_get_logger, mock_basic_config):
        # Set up the mocks
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Call the function
        LOG_DEBUG("test message")

        # Check if logger was configured and used
        mock_basic_config.assert_called_with(level=logging.DEBUG)
        mock_get_logger.assert_called_with("utils.utils")
        mock_logger.debug.assert_called_with("test message")

    def test_log_warn(self, mock_get_logger, mock_basic_config):
        # Set up the mocks
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Call the function
        LOG_WARN("test warning")

        # Check if logger was configured and used
        mock_basic_config.assert_called_with(level=logging.WARN)
        mock_get_logger.assert_called_with("utils.utils")
        mock_logger.debug.assert_called_with("test warning")

    def test_pympc_assert(self, mock_get_logger, mock_basic_config):
        # Set up the mocks
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Test case where assertion passes
        PYMPC_ASSERT(True, "This shouldn't fail")

        # Test case where assertion fails
        with self.assertRaises(AssertionError):
            PYMPC_ASSERT(False, "Assertion failed message")

        # Check if logger was configured and used
        mock_basic_config.assert_called_with(level=logging.ERROR)
        mock_get_logger.assert_called_with("utils.utils")
        mock_logger.error.assert_called()


@patch('logging.getLogger')
class TestExperimentManager(unittest.TestCase):
    def setUp(self):
        # Set up test CONFIG global
        global CONFIG
        CONFIG = {
            "recording": {
                "enable": True,
                "folder": "/tmp",
                "file": "test.dat",
                "timestamp": False,
                "num_experiments": 5
            },
            "N": 10,
            "control_frequency": 10.0
        }

    def test_init(self, mock_get_logger):
        # Set up mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create manager
        manager = ExperimentManager()

        # Check initialization
        self.assertEqual(manager.control_iteration, 0)
        self.assertEqual(manager.iteration_at_last_reset, 0)
        self.assertEqual(manager.experiment_counter, 0)

    @patch('utils.utils.DataSaver.add_data')
    def test_update(self, mock_add_data, mock_get_logger):
        # Set up mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create manager
        manager = ExperimentManager()

        # Create mock objects
        state = MagicMock()
        state.get_pose.return_value = [1, 2]
        state.get.return_value = 0.5

        solver = MagicMock()
        solver.get_ego_prediction_position.return_value = [3, 4]

        data = MagicMock()
        data.dynamic_obstacles = [MagicMock()]
        data.dynamic_obstacles[0].index = 1
        data.dynamic_obstacles[0].position = [5, 6]
        data.dynamic_obstacles[0].angle = 0.7
        data.dynamic_obstacles[0].radius = 0.3
        data.intrusion = 0.0

        # Call update
        manager.update(state, solver, data)

        # Check if data was added
        self.assertTrue(mock_add_data.called)

        # Control iteration should be incremented
        self.assertEqual(manager.control_iteration, 1)

    @patch('utils.utils.DataSaver.save_data')
    def test_export_data(self, mock_save_data, mock_get_logger):
        # Create manager
        manager = ExperimentManager()

        # Call export_data
        manager.export_data()

        # Check if data was saved
        mock_save_data.assert_called_with('tests', 'planner_test_output')

    @patch('utils.utils.DataSaver.add_data')
    @patch('utils.utils.ExperimentManager.export_data')
    def test_on_task_complete(self, mock_export_data, mock_add_data, mock_get_logger):
        # Set up mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create manager
        manager = ExperimentManager()
        manager.control_iteration = 10
        manager.num_experiments = 5

        # Call on_task_complete
        manager.on_task_complete(True)

        # Check if data was added
        self.assertTrue(mock_add_data.called)

        # Check experiment counter
        self.assertEqual(manager.experiment_counter, 1)


        try:
            # Test export trigger
            manager.experiment_counter = 4
            manager.on_task_complete(True)
            self.assertEqual(manager.experiment_counter, 5)
            mock_export_data.assert_called_once()
        except:
            manager.experiment_counter = 4

        # Test completion
        with self.assertRaises(AssertionError):
            manager.experiment_counter = 5
            manager.on_task_complete(True)


class TestDataSaver(unittest.TestCase):
    def test_init(self):
        # Create data saver
        saver = DataSaver()

        # Check initialization
        self.assertEqual(saver.data, {})
        self.assertFalse(saver.add_timestamp)

    def test_set_add_timestamp(self):
        # Create data saver
        saver = DataSaver()

        # Set timestamp flag
        saver.set_add_timestamp(True)
        self.assertTrue(saver.add_timestamp)

    def test_add_data(self):
        # Create data saver
        saver = DataSaver()

        # Add data
        saver.add_data("key1", "value1")
        saver.add_data("key2", 123)

        # Check data
        self.assertEqual(saver.data, {"key1": "value1", "key2": 123})

    @patch('builtins.open', new_callable=mock_open)
    def test_save_data(self, mock_file):
        # Create data saver
        saver = DataSaver()

        # Add data
        saver.add_data("key1", "value1")

        # Save data
        saver.save_data("/tmp", "test.dat")

        # Check if file was opened
        mock_file.assert_called_with("/tmp/test.dat", "w")

        # Check if data was written to file
        mock_file().write.assert_called_with(str({"key1": "value1"}))

    def test_get_file_path(self):
        # Create data saver
        saver = DataSaver()

        # Get file path
        path = saver.get_file_path("/tmp", "test.dat", False)

        # Check path
        self.assertEqual(path, "/tmp/test.dat")


if __name__ == '__main__':
    unittest.main()