from utils.utils import read_config_file

CONFIG = read_config_file()

class State:

  def __init__(self):
    initialize()
    loadConfigYaml(SYSTEM_CONFIG_PATH(__FILE__, "solver_settings"), _config)
    loadConfigYaml(SYSTEM_CONFIG_PATH(__FILE__, "model_map"), _model_map)


  def initialize(self):
    _state = (_config["nx"], 0.0)
    _nu = _config["nu"]


  def get(self, var_name):
    return _state[_model_map[var_name][1] - _nu] # States come after the inputs

  def getPos(self):
    return (get("x"), get("y"))

  def set(self, var_name, value):
    _state[_model_map[var_name][1] - _nu] = value

  def print(self):
    it = _model_map.begin()
    for it !=_model_map.end():
      if (str(it.second[0]) == "x"):
      LOG_DEBUG( it.first + get(it.first))
