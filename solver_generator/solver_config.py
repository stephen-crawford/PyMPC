import os

from solver_generator.util.files import get_package_path, get_current_package
from solver_generator.util.logging import print_path, print_success

from utils.utils import CONFIG

def generate_rqt_reconfigure(settings):
    """Generate configuration files for RQT Reconfigure."""
    current_package = get_current_package()
    system_name = "".join(current_package.split("_")[2:])
    path = f"{get_package_path(current_package)}/cfg/"
    os.makedirs(path, exist_ok=True)
    path += f"{system_name}.cfg"
    print_path("RQT Reconfigure", path, end="", tab=True)

    with open(path, "w") as rqt_file:
        rqt_file.write("#!/usr/bin/env python\n")
        rqt_file.write(f'PACKAGE = "{current_package}"\n')
        rqt_file.write("from dynamic_reconfigure.parameter_generator_catkin import *\n")
        rqt_file.write("gen = ParameterGenerator()\n\n")

        rqt_file.write('weight_params = gen.add_group("Weights", "Weights")\n')
        rqt_params = settings["params"].rqt_params
        for idx, param in enumerate(rqt_params):
            rqt_file.write(
                f'weight_params.add("{param}", double_t, 1, "{param}", 1.0, '
                f'{settings["params"].rqt_param_min_values[idx]}, '
                f'{settings["params"].rqt_param_max_values[idx]})\n'
            )
        rqt_file.write(f'exit(gen.generate(PACKAGE, "{current_package}", "{system_name}"))\n')

    print_success(" -> generated")


def get_parameter_bundle_values(settings):
    """Get parameter bundle values from settings."""
    parameter_bundles = {}

    for key, indices in settings["params"].parameter_bundles.items():
        function_name = key.replace("_", " ").title().replace(" ", "")
        parameter_bundles[function_name] = indices

    return parameter_bundles


def set_solver_parameters(k, params, parameter_name, value, index=0):
    """Set solver parameters based on parameter name."""
    parameter_bundles = get_parameter_bundle_values(CONFIG)

    if parameter_name in parameter_bundles:
        indices = parameter_bundles[parameter_name]
        if len(indices) == 1:
            params.all_parameters[k * CONFIG['params'].length() + indices[0]] = value
        else:
            if 0 <= index < len(indices):
                params.all_parameters[k * CONFIG['params'].length() + indices[index]] = value
    else:
        raise ValueError(f"Unknown parameter: {parameter_name}")