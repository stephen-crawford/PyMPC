"""
Integration example demonstrating unified scenario-based MPC with the existing framework.

This example shows how to:
1. Use the existing Planner with scenario-based constraints
2. Combine scenario_mpc modules with existing modules
3. Bridge types between systems
4. Run the complete MPC loop
"""

import sys
import numpy as np
sys.path.insert(0, '/home/stephen/PyMPC')

from planning.planner import Planner
from planning.types import (
    State, Data, DynamicObstacle, Prediction, PredictionStep,
    PredictionType, ReferencePath, define_robot_area, generate_reference_path,
)
from planning.scenario_adapter import (
    ScenarioDataAdapter,
    state_to_ego_state,
    ego_state_to_state,
    dynamic_obstacle_to_obstacle_state,
)
from planning.dynamic_models import SecondOrderUnicycleModel
from modules.constraints.scenario_constraint import ScenarioConstraint
from modules.constraints.contouring_constraints import ContouringConstraints
from modules.objectives.goal_objective import GoalObjective
from modules.objectives.control_effort_objective import ControlEffortObjective
from utils.utils import LOG_INFO, LOG_DEBUG


def create_test_config():
    """Create test configuration."""
    return {
        "planner": {
            "horizon": 15,
            "timestep": 0.1,
        },
        "solver": {
            "solver": "casadi",
            "shift_previous_solution_forward": True,
        },
        "solver_iterations": 50,
        "max_obstacles": 5,
        "max_obstacle_distance": 50.0,
        "integrator_step": 0.1,
        # Scenario constraint settings
        "scenario_constraint": {
            "num_scenarios": 8,
            "confidence_level": 0.95,
            "ego_radius": 0.5,
            "obstacle_radius": 0.5,
            "weight_type": "frequency",
            "enable_pruning": True,
        },
        # Contouring constraints settings
        "contouring_constraints": {
            "road_width": 4.0,
        },
        # Goal objective settings
        "goal_objective": {
            "weight": 10.0,
        },
        # Control effort settings
        "control_effort_objective": {
            "acceleration_weight": 0.1,
            "steering_weight": 0.1,
        },
    }


def create_reference_path():
    """Create a simple reference path."""
    start = [0.0, 0.0, 0.0]
    goal = [20.0, 0.0, 0.0]
    return generate_reference_path(start, goal, path_type="straight", num_points=50)


def create_obstacles():
    """Create test dynamic obstacles."""
    obstacles = []

    # Obstacle 1: Coming towards ego from ahead
    obs1 = DynamicObstacle(
        index=0,
        position=np.array([10.0, 0.5]),
        angle=np.pi,  # Facing left (towards ego)
        radius=0.5
    )
    obs1.prediction = Prediction(PredictionType.GAUSSIAN)
    obs1.velocity = np.array([-0.5, 0.0])
    obstacles.append(obs1)

    # Obstacle 2: Moving perpendicular to path
    obs2 = DynamicObstacle(
        index=1,
        position=np.array([8.0, -2.0]),
        angle=np.pi / 2,  # Facing up
        radius=0.5
    )
    obs2.prediction = Prediction(PredictionType.GAUSSIAN)
    obs2.velocity = np.array([0.0, 0.3])
    obstacles.append(obs2)

    return obstacles


class ScenarioProblem:
    """Problem definition combining scenario constraints with existing modules."""

    def __init__(self, config):
        self.config = config
        self.model_type = SecondOrderUnicycleModel()
        self.modules = []
        self.obstacles = []
        self.data = None
        self.x0 = None
        self._state = None

    def setup(self, initial_state, reference_path, obstacles, goal):
        """Setup the problem with initial conditions."""
        # Create data container
        self.data = Data()

        # Set reference path
        self.data.reference_path = reference_path

        # Set goal
        self.data.goal = goal
        self.data.parameters = {
            'goal_x': float(goal[0]),
            'goal_y': float(goal[1]),
        }

        # Set dynamics model
        self.data.dynamics_model = self.model_type

        # Set horizon and timestep
        planner_config = self.config.get("planner", {})
        self.data.horizon = planner_config.get("horizon", 15)
        self.data.timestep = planner_config.get("timestep", 0.1)

        # Set robot area (collision discs)
        self.data.robot_area = define_robot_area(
            length=2.0,
            width=1.0,
            n_discs=1
        )

        # Set road width for contouring
        self.data.road_width = self.config.get("contouring_constraints", {}).get("road_width", 4.0)

        # Store obstacles
        self.obstacles = obstacles
        self.data.dynamic_obstacles = obstacles

        # Create initial state
        self.x0 = State(self.model_type)
        for key, val in initial_state.items():
            self.x0.set(key, val)
        self.data.state = self.x0
        self._state = self.x0

        # Create modules
        self._create_modules()

    def _create_modules(self):
        """Create constraint and objective modules."""
        self.modules = []

        # Goal objective - drives towards goal
        goal_obj = GoalObjective(self.config)
        self.modules.append(goal_obj)

        # Control effort objective - penalizes control effort
        control_obj = ControlEffortObjective(self.config)
        self.modules.append(control_obj)

        # Scenario constraint - scenario-based collision avoidance
        scenario_constraint = ScenarioConstraint(self.config)
        self.modules.append(scenario_constraint)

        # Contouring constraints - keeps vehicle on path
        contouring_constraint = ContouringConstraints(self.config)
        self.modules.append(contouring_constraint)

    def get_model_type(self):
        return self.model_type

    def get_modules(self):
        return self.modules

    def get_obstacles(self):
        return self.obstacles

    def get_data(self):
        return self.data

    def get_x0(self):
        return self.x0

    def get_state(self):
        if self._state is not None:
            return self._state
        return self.x0

    def get_horizon(self):
        return self.data.horizon if self.data else 15

    def get_timestep(self):
        return self.data.timestep if self.data else 0.1


def propagate_obstacles_simple(data, dt, horizon, speed=0.5):
    """Simple obstacle propagation for testing."""
    if not hasattr(data, 'dynamic_obstacles') or not data.dynamic_obstacles:
        return

    for obstacle in data.dynamic_obstacles:
        pred = obstacle.prediction
        pred.steps = []

        velocity = getattr(obstacle, 'velocity', np.array([0.0, 0.0]))
        current_pos = obstacle.position.copy()

        for k in range(int(horizon) + 1):
            future_pos = current_pos + velocity * k * dt
            angle = np.arctan2(velocity[1], velocity[0]) if np.linalg.norm(velocity) > 0 else 0.0

            # Gaussian uncertainty grows with time
            uncertainty = 0.1 + k * 0.05
            pred.steps.append(PredictionStep(
                position=future_pos,
                angle=angle,
                major_radius=obstacle.radius + uncertainty,
                minor_radius=obstacle.radius + uncertainty * 0.5
            ))


def run_integrated_simulation():
    """Run the integrated simulation."""
    print("=" * 60)
    print("Integrated Scenario-Based MPC Example")
    print("=" * 60)

    # Create configuration
    config = create_test_config()

    # Create reference path
    ref_path = create_reference_path()
    print(f"Reference path: {len(ref_path.x)} points, length={ref_path.s[-1]:.2f}m")

    # Create obstacles
    obstacles = create_obstacles()
    print(f"Created {len(obstacles)} obstacles")

    # Define goal
    goal = np.array([18.0, 0.0])
    print(f"Goal: ({goal[0]:.1f}, {goal[1]:.1f})")

    # Initial state
    initial_state = {
        'x': 0.0,
        'y': 0.0,
        'psi': 0.0,
        'v': 0.5,
        'spline': 0.0,  # Arc length along path
    }
    print(f"Initial state: x={initial_state['x']}, y={initial_state['y']}, "
          f"psi={initial_state['psi']:.2f}, v={initial_state['v']}")

    # Create problem
    problem = ScenarioProblem(config)
    problem.setup(initial_state, ref_path, obstacles, goal)

    # Create planner
    print("\nInitializing planner...")
    planner = Planner(problem, config)

    # Simulation loop
    print("\nStarting simulation...")
    max_steps = 50
    ego_trajectory = []
    step = 0

    while step < max_steps:
        # Get current state
        current_state = planner.get_state()
        ego_x = current_state.get('x')
        ego_y = current_state.get('y')
        ego_trajectory.append((ego_x, ego_y))

        # Check if goal reached
        dist_to_goal = np.sqrt((ego_x - goal[0])**2 + (ego_y - goal[1])**2)
        if dist_to_goal < 1.0:
            print(f"\nGoal reached at step {step}!")
            break

        # Propagate obstacles for current predictions
        propagate_obstacles_simple(
            planner.data,
            config["planner"]["timestep"],
            config["planner"]["horizon"]
        )

        # Solve MPC
        output = planner.solve_mpc(planner.data)

        if output.success:
            # Apply control and propagate
            if output.control:
                control_dict = output.control
                # Propagate state using dynamics
                planner.state = planner.state.propagate(
                    control_dict,
                    config["planner"]["timestep"],
                    dynamics_model=planner.model_type
                )
                planner.data.state = planner.state

                # Update obstacles (simple constant velocity)
                for obs in planner.data.dynamic_obstacles:
                    if hasattr(obs, 'velocity'):
                        obs.position = obs.position + obs.velocity * config["planner"]["timestep"]

            if step % 10 == 0:
                print(f"Step {step}: position=({ego_x:.2f}, {ego_y:.2f}), "
                      f"dist_to_goal={dist_to_goal:.2f}")
        else:
            print(f"Step {step}: MPC solve failed!")
            # Use fallback (braking)
            planner.state.set('v', max(0, planner.state.get('v') - 0.1))

        step += 1

    print(f"\nSimulation complete after {step} steps")
    print(f"Final position: ({ego_trajectory[-1][0]:.2f}, {ego_trajectory[-1][1]:.2f})")

    return ego_trajectory, obstacles, goal, ref_path


def visualize_results(ego_trajectory, obstacles, goal, ref_path):
    """Visualize simulation results."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot reference path
        ax.plot(ref_path.x, ref_path.y, 'k--', linewidth=1, alpha=0.5, label='Reference path')

        # Plot ego trajectory
        ego_x = [p[0] for p in ego_trajectory]
        ego_y = [p[1] for p in ego_trajectory]
        ax.plot(ego_x, ego_y, 'b-', linewidth=2, label='Ego trajectory')
        ax.plot(ego_x[0], ego_y[0], 'bo', markersize=10, label='Start')
        ax.plot(ego_x[-1], ego_y[-1], 'bs', markersize=10, label='End')

        # Plot initial obstacle positions
        for i, obs in enumerate(obstacles):
            ax.plot(obs.position[0], obs.position[1], 'ro', markersize=8)
            circle = Circle(obs.position[:2], obs.radius, fill=False, color='red', linestyle='--')
            ax.add_patch(circle)
            ax.annotate(f'Obs {i}', obs.position[:2], fontsize=8)

        # Plot goal
        ax.plot(goal[0], goal[1], 'g*', markersize=15, label='Goal')
        goal_circle = Circle(goal[:2], 1.0, fill=False, color='green', linestyle='--')
        ax.add_patch(goal_circle)

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Integrated Scenario-Based MPC Simulation')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim(-2, 22)
        ax.set_ylim(-4, 4)

        plt.tight_layout()
        plt.savefig('/home/stephen/PyMPC/examples/scenario_integration_result.png', dpi=150)
        plt.show()
        print("Plot saved to scenario_integration_result.png")

    except Exception as e:
        print(f"Visualization skipped: {e}")


if __name__ == "__main__":
    # Run simulation
    ego_traj, obstacles, goal, ref_path = run_integrated_simulation()

    # Visualize
    visualize_results(ego_traj, obstacles, goal, ref_path)
