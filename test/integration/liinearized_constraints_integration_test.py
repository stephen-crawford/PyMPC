from solver.src.modules_manager import initialize_module_manager
from planner_modules.src.constraints.linearized_constraints import LinearizedConstraints
from planner_modules.src.objectives.goal_objective import GoalModule


def initialize_modules(solver=None):
	"""Initialize all modules and return the module manager"""
	module_manager = initialize_module_manager(solver)

	# Add your modules here
	module_manager.add_module(LinearizedConstraints(solver))
	module_manager.add_module(GoalModule(solver))  # Make sure to pass the solver
	# Add more modules as needed

	return module_manager


def integrate_with_solver_generator():
	"""Hook into the solver generator pipeline"""
	# Override solver_definition functions with our module-based versions
	import solver_generator.solver_definition as solver_def

	# Create module manager without solver (will be assigned later)
	modules = initialize_modules()

	# Replace solver_definition functions with our module-based versions
	# Store the original define_parameters to call it within our wrapped function
	original_define_parameters = solver_def.define_parameters

	# Create wrapper functions with correct signatures
	def wrapped_define_parameters(params, settings):
		return define_parameters(modules, params, settings)

	def wrapped_objective(z, p, model, settings, stage_idx):
		return objective(modules, z, p, model, settings, stage_idx)

	def wrapped_constraints(z, p, model, settings, stage_idx):
		return constraints(modules, z, p, model, settings, stage_idx)

	# Assign the wrapper functions
	solver_def.define_parameters = wrapped_define_parameters
	solver_def.objective = wrapped_objective
	solver_def.constraints = wrapped_constraints
	solver_def.constraint_lower_bounds = lambda: constraint_lower_bounds(modules)
	solver_def.constraint_upper_bounds = lambda: constraint_upper_bounds(modules)
	solver_def.constraint_number = lambda: constraint_number(modules)

	return modules


# Import necessary functions outside the main block so they're available
from solver.src.modules_manager import define_parameters, objective, constraints
from solver.src.modules_manager import constraint_lower_bounds, constraint_upper_bounds, constraint_number

# Example usage in your main script
if __name__ == "__main__":
	from solver_generator.util.files import load_settings
	from solver_generator.dynamics_models import BicycleModel2ndOrder
	from solver.src.generate_solver import generate_casadi_solver

	# Initialize modules and integrate with solver generator
	modules = integrate_with_solver_generator()

	# Load settings and create model
	settings = load_settings()
	model = BicycleModel2ndOrder()

	# Generate solver - using the correct function with correct parameters
	# Note: generate_solver is now expecting modules first, unlike in your error
	solver, simulator = generate_casadi_solver(modules, settings, model, False)

	# Now attach the solver to the modules
	for module in modules.modules:  # Use modules.modules
		module.solver = solver

	# Setup a simple test case
	state = {"x": 0.0, "y": 0.0, "v": 5.0, "psi": 0.0}


	# Create dummy data for the modules that matches what LinearizedConstraints expects
	class DummyObstacle:
		def __init__(self):
			self.radius = 0.5
			self.prediction = DummyPrediction()


	class DummyPrediction:
		def __init__(self):
			self.modes = [[DummyMode() for _ in range(solver.N)]]
			self.type = "DETERMINISTIC"  # Match the DETERMINISTIC constant

		def empty(self):
			return False


	class DummyMode:
		def __init__(self):
			self.position = [10.0, 0.0]  # Position away from ego vehicle


	class DummyDisc:
		def __init__(self):
			self.offset = 0.325  # Matching your config

		def get_position(self, pos, psi):
			return pos  # Simplified for testing


	class DummyData:
		def __init__(self):
			self.dynamic_obstacles = DummyObstacleList()
			self.robot_area = [DummyDisc() for _ in range(1)]  # Use the correct number of discs


	class DummyObstacleList:
		def __init__(self):
			self._obstacles = [DummyObstacle() for _ in range(3)]  # Match max_obstacles in config

		def __getitem__(self, idx):
			return self._obstacles[idx]

		def size(self):
			return len(self._obstacles)

		def empty(self):
			return len(self._obstacles) == 0


	class DummyModuleData:
		def __init__(self):
			self.static_obstacles = DummyStaticList()


	class DummyStaticList:
		def __init__(self):
			self._list = [[] for _ in range(solver.N + 1)]

		def __getitem__(self, idx):
			return self._list[idx]

		def empty(self):
			return True


	data = DummyData()
	module_data = DummyModuleData()

	# Initialize the solver with the state
	solver.set_xinit(state)

	# Update the modules with the current state
	modules.update_all(state, data, module_data)

	# Set the parameters for all modules
	modules.set_parameters_all(data, module_data)

	# Solve the optimization problem
	result = solver.solve()

	print(f"Solver result: {result}")
	print(f"Explanation: {solver.explain_exit_flag(result)}")

	# Get some outputs if successful
	if result == 1:
		for k in range(solver.N):
			x = solver.get_output(k, "x")
			y = solver.get_output(k, "y")
			v = solver.get_output(k, "v")
			print(f"Step {k}: x={x:.2f}, y={y:.2f}, v={v:.2f}")