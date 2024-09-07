import yaml
from pathlib import Path
import collections.abc


def load_yaml_config(config_path):
    """Load a YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def merge_configs(base_config, override_config):
    """Merge two configuration dictionaries."""
    merged_config = base_config.copy()
    merged_config = update(merged_config, override_config)
    return merged_config


def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the config.yaml file.
    
    Returns:
        dict: Configuration dictionary.
    """
    config = load_yaml_config(config_path)
    
    if 'parent_config' in config:
        parent_path = Path(config['parent_config'])
        parent_config = load_config(parent_path)  # Recursively load parent
        config = merge_configs(parent_config, config)
        config.pop('parent_config', None)

    return config