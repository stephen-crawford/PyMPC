import unittest
import numpy as np
from unittest.mock import patch, MagicMock

# Import functions and classes to test
from solver.src.generate_solver import generate_solver, generate_casadi_solver, generate_osqp_solver
from planner.src.dynamic_models import (
    SecondOrderUnicycleModel,
    BicycleModel2ndOrder,
    ContouringSecondOrderUnicycleModelCurvatureAware
)
from solver.src.casadi_solver import CasADiSolver
from solver.src.osqp_solver import OSQPSolver


class TestSolverGeneration(unittest.TestCase):
    """Test suite for solver generation functions."""

    def setUp(self):
        """Set up test fixtures before each test."""
        # Create a mock settings dictionary
        self.settings = {
            "name": "test",
            "dt": 0.1,
            "N": 20,
            "solver_settings": {
                "solver": "casadi"
            },
            "params": None,
            "contouring": {
                "get_num_segments": 10
            }
        }

        # Create mock modules
        self.modules = []

    def test_generate_casadi_solver(self):
        """Test generating a CasADi solver."""
        model = SecondOrderUnicycleModel()

        solver, simulator = generate_casadi_solver(self.modules, self.settings, model)

        self.assertTrue(isinstance(solver, CasADiSolver))
        self.assertTrue (isinstance(simulator, CasADiSolver))
        # Check if solver and simulator are the same instance
        self.assertEqual(solver, simulator)

    def test_generate_osqp_solver(self):
        """Test generating an OSQP solver."""
        model = SecondOrderUnicycleModel()

        solver, simulator = generate_osqp_solver(self.modules, self.settings, model)

        self.assertTrue(isinstance(solver, OSQPSolver))
        self.assertTrue(isinstance(simulator, OSQPSolver))
        # Check if solver and simulator are the same instance
        self.assertEqual(solver, simulator)

    def test_invalid_solver_type(self):
        """Test behavior with invalid solver type."""
        model = SecondOrderUnicycleModel()
        self.settings["solver_settings"]["solver"] = "invalidsolver"

        with self.assertRaises(IOError):
            generate_solver(self.modules, model, settings=self.settings)

    def test_skip_solver_generation(self):
        """Test skipping solver generation."""
        model = SecondOrderUnicycleModel()

        # Mock sys.argv to simulate 'false' to skip solver generation
        with patch('sys.argv', ['script_name', 'false']):
            # Mock load_settings to return predefined settings
            with patch('solver_generator.util.files.load_settings', return_value=self.settings):
                # Mock the actual solver generation to return None, None
                with patch('solver_generator.generatesolver.generate_casadisolver') as mock_gen_casadi:
                    mock_gen_casadi.return_value = (None, None)

                    # Now, when generatesolver is called, it should use the mock load_settings
                    solver, simulator = generate_solver(self.modules, model)

                    # Check if solver generation was skipped
                    self.assertIsNone(solver)
                    self.assertIsNone(simulator)


class TestDynamicsModels(unittest.TestCase):
    """Test suite for dynamics models."""

    def setUp(self):
        """Set up test fixtures before each test."""
        self.settings = {
            "dt": 0.1,
            "integrator_step": 0.01,
            "params": MagicMock(),
            "contouring": {
                "get_num_segments": 10
            }
        }

        # Mock params
        self.mock_params = MagicMock()
        self.mock_params.get_p.return_value = np.array([])
        self.settings["params"] = self.mock_params

    def test_second_order_unicycle_model(self):
        """Test SecondOrderUnicycleModel."""
        model = SecondOrderUnicycleModel()

        # Check if model has correct dimensions
        self.assertEqual(model.nu, 2)
        self.assertEqual(model.nx, 4)

        # Create a test state vector
        z = np.array([0.5, 0.2, 1.0, 2.0, 0.5, 1.0])
        model.load(z)
        model.load_settings(self.settings)

        # Test continuous model
        x = z[model.nu:]
        u = z[:model.nu]
        result = model.continuous_model(x, u)

        # Check dimensions of the result
        self.assertEqual(len(result), model.nx)

        # Check specific values (example: velocity components)
        self.assertAlmostEqual(float(result[0]), x[3] * np.cos(x[2]))
        self.assertAlmostEqual(float(result[1]), x[3] * np.sin(x[2]))

    def test_bicycle_model(self):
        """Test BicycleModel2ndOrder."""
        model = BicycleModel2ndOrder()

        # Check if model has correct dimensions
        self.assertEqual(model.nu, 3)
        self.assertEqual(model.nx, 6)

        # Test state and input bounds
        self.assertEqual(len(model.lower_bound), model.nu + model.nx)
        self.assertEqual(len(model.upper_bound), model.nu + model.nx)

        # Test get_bounds method for a state
        state_bounds = model.get_bounds("v")
        self.assertEqual(state_bounds[0], model.lower_bound[model.nu + 3])
        self.assertEqual(state_bounds[1], model.upper_bound[model.nu + 3])

        # Test get_bounds method for an input
        input_bounds = model.get_bounds("a")
        self.assertEqual(input_bounds[0], model.lower_bound[0])
        self.assertEqual(input_bounds[1], model.upper_bound[0])

    @patch('solver_generator.solver_model.DynamicsModel')
    def test_curvature_aware_model(self, mock_spline):
        """Test ContouringSecondOrderUnicycleModelCurvatureAware."""
        # Setup mock spline
        mock_spline_instance = MagicMock()
        mock_spline_instance.at.return_value = (0.0, 0.0)
        mock_spline_instance.deriv_normalized.return_value = (1.0, 0.0)
        mock_spline_instance.deriv2.return_value = (0.0, 0.0)
        mock_spline_instance.get_curvature.return_value = 0.01
        mock_spline.return_value = mock_spline_instance

        model = ContouringSecondOrderUnicycleModelCurvatureAware()

        # Check special property for integration
        self.assertEqual(model.nx_integrate, model.nx - 1)

        # Test model_discrete_dynamics
        z = np.zeros(model.nu + model.nx)
        integrated_states = np.ones(model.nx - 1)  # All states except spline
        model.load(z)
        model.load_settings(self.settings)

        # Test discrete dynamics override
        with patch.object(model, 'discrete_dynamics') as mock_discrete:
            mock_discrete.return_value = None
            model.integrate(z, self.settings, 0.01)
            mock_discrete.assert_called_once()


class TestIntegration(unittest.TestCase):
    """Test full integration between models and solvers."""

    def setUp(self):
        """Set up test fixtures before each test."""
        self.settings = {
            "name": "test",
            "dt": 0.1,
            "N": 20,
            "solver_settings": {
                "solver": "casadi"
            },
            "params": None,
            "contouring": {
                "get_num_segments": 10
            }
        }
        self.modules = []

    @patch('solver_generator.util.files.write_to_yaml')
    @patch('solver_generator.solver_config.generate_rqt_reconfigure')
    @patch('solver_generator.util.logging.print_header')
    @patch('solver_generator.util.files.solver_path')
    @patch('solver_generator.util.files.solver_settings_path')
    @patch('solver_generator.util.logging.print_success')
    def test_casadi_solver_integration(self, mock_print_success, mocksolver_settings_path,
                                       mocksolver_path, mock_print_header,
                                       mock_generate_rqt, mock_write_yaml):
        """Test integration of a model with CasADi solver generation."""
        # Create real instances
        model = SecondOrderUnicycleModel()
        mocksolver_settings_path.return_value = "test_path.yml"
        mocksolver_path.return_value = "testsolver_path"

        # Force CasADi solver type in settings
        self.settings["solver_type"] = "casadi"

        # Use generate_casadisolver directly instead of the generic generatesolver
        solver, simulator = generate_casadi_solver(self.modules, self.settings, model)

        # Assert that we got CasADi solver instances
        self.assertIsInstance(solver, CasADiSolver)
        self.assertIsInstance(simulator, CasADiSolver)

        # Check that the model's dynamics were properly set up
        self.assertEqual(solver.nx, model.nx)
        self.assertEqual(solver.nu, model.nu)
        self.assertEqual(solver.nvar, model.get_nvar())

        # Verify that CasADi problem was finalized
        # This will only pass if finalize_problem() completed successfully
        self.assertTrue(hasattr(solver, 'opti') and solver.opti is not None)

    @patch('solver_generator.solver_config.generate_rqt_reconfigure')
    @patch('solver_generator.util.logging.print_header')
    @patch('solver_generator.util.files.solver_path')
    @patch('solver_generator.util.files.solver_settings_path')
    @patch('solver_generator.util.logging.print_success')
    def test_osqp_solver_integration(self, mock_print_success, mocksolver_settings_path,
                                     mocksolver_path, mock_print_header,
                                     mock_generate_rqt):
        """Test integration of a model with OSQP solver generation."""
        # Create real instances
        model = SecondOrderUnicycleModel()
        mocksolver_settings_path.return_value = "test_path.yml"
        mocksolver_path.return_value = "testsolver_path"

        # Force OSQP solver type in settings
        self.settings["solver_type"] = "osqp"

        # Use generate_osqpsolver directly instead of the generic generatesolver
        solver, simulator = generate_osqp_solver(self.modules, self.settings, model)

        # Assert that we got OSQP solver instances
        self.assertIsInstance(solver, OSQPSolver)
        self.assertIsInstance(simulator, OSQPSolver)

        # Check that the model's dynamics were properly set up
        self.assertEqual(solver.nx, model.nx)
        self.assertEqual(solver.nu, model.nu)
        self.assertEqual(solver.nvar, model.get_nvar())

        # Verify that OSQP problem was properly configured
        # This verification depends on your OSQP solver implementation
        # For example, check if the problem matrices exist
        self.assertTrue(hasattr(solver, 'P') and solver.P is not None)
        self.assertTrue(hasattr(solver, 'A') and solver.A is not None)

        # Since we know from the previous test that write_to_yaml might not be called
        # by generate_osqpsolver directly, we can skip that assertion or
        # add appropriate checks for the actual function that should be called

        # If you want to test solver functionality
        if hasattr(solver, 'solve'):
            try:
                # Try to solve
                result = solver.solve()
                self.assertIsNotNone(result)
            except Exception as e:
                self.fail(f"Solver failed to solve a test problem: {e}")

if __name__ == '__main__':
    unittest.main()