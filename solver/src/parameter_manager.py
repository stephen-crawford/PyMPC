import copy

import numpy as np
import casadi as cd
from utils.utils import print_header, print_value, parameter_map_path, write_to_yaml, LOG_INFO, LOG_DEBUG


class ParameterManager:
    def __init__(self):
        self.parameter_lookup = {}  # param -> (start_idx, length)
        self.parameter_count = 0
        self.parameter_values = None

        # RQT support
        self.rqt_parameters = []
        self.rqt_parameter_config_names = []
        self.rqt_parameter_min_values = []
        self.rqt_parameter_max_values = []

    def add(self, parameter, length=1,
            add_to_rqt_reconfigure=False,
            rqt_config_name=lambda p: f'["weights"]["{p}"]',
            rqt_min_value=0.0,
            rqt_max_value=100000000000.0):
        """Add a parameter with given length and optional RQT settings."""
        if parameter in self.parameter_lookup:
            return

        start_idx = self.parameter_count
        self.parameter_lookup[parameter] = (start_idx, length)
        self.parameter_count += length

        if add_to_rqt_reconfigure:
            self.rqt_parameters.append(parameter)
            self.rqt_parameter_config_names.append(rqt_config_name)
            self.rqt_parameter_min_values.append(rqt_min_value)
            self.rqt_parameter_max_values.append(rqt_max_value)

    def load(self, p):
        """Load a flat vector of parameter values."""
        p = np.array(p, dtype=float).flatten()
        if p.shape[0] != self.parameter_count:
            raise ValueError(f"Expected {self.parameter_count} parameters, got {p.shape[0]}")
        self.parameter_values = p

    def set_parameter(self, key, value):
        """Set a parameter value (scalar or vector) in the flat array."""
        if self.parameter_values is None:
            self.parameter_values = []
        if key not in self.parameter_lookup:
            LOG_DEBUG("Parameter manager set parameter failing because key missing")
            raise KeyError(f"{id(self)} Parameter '{key}' not found.")

        start, length = self.parameter_lookup[key]
        value = np.atleast_1d(value).astype(float)

        if value.shape[0] != length:
            raise ValueError(f"Expected length {length} for '{key}', got {value.shape[0]}")

        if key in self.rqt_parameters:
            idx = self.rqt_parameters.index(key)
            min_val = self.rqt_parameter_min_values[idx]
            max_val = self.rqt_parameter_max_values[idx]
            value = np.clip(value, min_val, max_val)

        self.parameter_values[start:start + length] = value

    def get(self, key):
        """Retrieve a parameter value (scalar or vector)."""
        if self.parameter_values is None:
            raise RuntimeError("Parameters not loaded.")
        if key not in self.parameter_lookup:
            LOG_DEBUG("Parameter manager get key failing because key missing")
            raise KeyError(f"Parameter '{key}' not found.")
        start, length = self.parameter_lookup[key]
        val = self.parameter_values[start:start + length]

        return val[0] if length == 1 else val

    def get_all(self):
        """Return all parameters as a dictionary: {key: value}"""
        if self.parameter_values is None:
            raise RuntimeError("Parameters not loaded.")

        all_params = {}
        for key, (start, length) in self.parameter_lookup.items():
            value = self.parameter_values[start:start + length]
            all_params[key] = value[0] if length == 1 else value.copy()
        return all_params

    def update_from_dict(self, param_dict):
        for key, value in param_dict.items():
            if self.has_parameter(key):
                self.set_parameter(key, value)
            else:
                LOG_DEBUG(f"Parameter '{key}' not found, skipping update.")

    def has_parameter(self, key):
        return key in self.parameter_lookup

    def get_casadi_parameters(self):
        """Get symbolic vector for use in CasADi problem definition."""
        return cd.SX.sym('p', self.parameter_count)

    def get_parameter_values(self):
        return self.parameter_values

    def save_map(self):
        """Save the parameter map to a YAML file."""
        file_path = parameter_map_path()
        save_map = {
            "parameter_indices": {k: {"start": v[0], "length": v[1]} for k, v in self.parameter_lookup.items()},
            "num_parameters": self.parameter_count
        }
        write_to_yaml(file_path, save_map)

    def __str__(self):
        lines = [self.format_header("Parameters")]
        for key, (start, length) in self.parameter_lookup.items():
            tag = " (in rqt_reconfigure)" if key in self.rqt_parameters else ""

            # Default to "Not loaded" in case of errors
            val_str = "Not loaded"
            if self.parameter_values is not None:
                try:
                    value = self.parameter_values[start:start + length]
                    if len(value) == length:
                        val_str = value[0] if length == 1 else value.tolist()
                except Exception as e:
                    val_str = f"Error: {str(e)}"

            lines.append(self.format_value(f"[{start}:{start + length}]", f"{key}{tag} = {val_str}", tab=True))
        return "\n".join(lines)

    def format_header(self, title):
        return f"\n=== {title} ==="

    def format_value(self, name, value, tab=False):
        prefix = "  " if tab else ""
        return f"{prefix}{name}: {value}"

    def copy(self):
        return copy.deepcopy(self)


