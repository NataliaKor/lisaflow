import yaml
from ruamel.yaml import YAML


def get_config_yaml(path):
    """
    Read configuration parameter from file
    Args:
        path -- Path to the yaml configuration file
    Return:
        Dict with parameter
    """
    yaml = YAML()
    with open(path, 'r') as stream:
        return yaml.load(stream)

def get_config(path):
    """
    Read configuration parameter from file
    Args:
        path -- Path to the yaml configuration file
    Return:
        Dict with parameter
    """

    with open(path, 'r') as stream:
        return yaml.load(stream, yaml.FullLoader)
