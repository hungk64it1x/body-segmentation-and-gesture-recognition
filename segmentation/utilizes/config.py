import logging
import sys
import os
import yaml
import imp
import pprint


def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    cfg : dict
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, "r") as stream:
        cfg = yaml.load(stream)
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    return cfg


def make_paths_absolute(dir_, cfg):
    """
    Make all values for keys ending with `_path` or `_dir` absolute to dir_.

    Parameters
    ----------
    dir_ : str
    cfg : dict

    Returns
    -------
    cfg : dict
    """
    for key in cfg.keys():
        if str(key).endswith("_path") or str(key).endswith("_dir"):
            if type(cfg[key]) is list:
                cfg[key] = [os.path.abspath(os.path.join(dir_, i)) for i in cfg[key]]
            else:
                cfg[key] = os.path.abspath(os.path.join(dir_, cfg[key]))
            if cfg is dict and not os.path.isfile(cfg[key]):
                logging.error("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg
