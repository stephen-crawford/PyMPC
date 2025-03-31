import yaml

def read_config_file():
    with open("CONFIG.yml", 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: e")
            return None
