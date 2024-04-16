from tomllib import load
import cytomine
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_config(config_path:str):
    with open(config_path, "rb") as f:
        config = load(f)
    return config

if __name__ == '__main__':
    api_keys = load_config('keys.toml')
