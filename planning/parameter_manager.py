from collections import defaultdict
from typing import Any, Dict, Optional


class ParameterManager:
    """
    Aggregates per-stage parameters provided by modules for the solver.

    Planner contract:
      - set_parameters(module, data, k)
      - get_all(k) -> Dict[str, Any]

    Module-side supported patterns:
      - module.set_parameters(parameter_manager, data, k): pushes params into manager
      - module.get_parameters(data, k) -> Dict
      - module.parameters or module.params: static or precomputed dict
    """

    def __init__(self):
        self._stage_to_params: Dict[int, Dict[str, Any]] = defaultdict(dict)

    # --------- API used by planner ---------
    def set_parameters(self, module: Any, data: Any, stage_index: int) -> None:
        """Collect parameters for a given module at a stage.

        Tries push-style first (module calls back into this manager), then pull-style.
        """
        # Push style
        if hasattr(module, "set_parameters"):
            module.set_parameters(self, data, stage_index)
            return

        # Pull style common variants
        params: Optional[Dict[str, Any]] = None
        if hasattr(module, "get_parameters"):
            try:
                params = module.get_parameters(data, stage_index)
            except TypeError:
                # Fallback: some modules may only accept data
                params = module.get_parameters(data)
        elif hasattr(module, "parameters"):
            params = getattr(module, "parameters")
        elif hasattr(module, "params"):
            params = getattr(module, "params")

        if params:
            # Namespace by module name when possible to avoid collisions
            module_name = getattr(module, "name", None)
            if module_name:
                for key, value in params.items():
                    self._stage_to_params[stage_index][f"{module_name}.{key}"] = value
            else:
                self._stage_to_params[stage_index].update(params)

    def get_all(self, stage_index: int) -> Dict[str, Any]:
        return self._stage_to_params.get(stage_index, {})

    # --------- API for modules using push-style ---------
