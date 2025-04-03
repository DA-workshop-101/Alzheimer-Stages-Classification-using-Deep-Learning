from distutils.command.config import config # type: ignore
import os
import requests
import shutil
import random
import argparse
import pandas as pd
import numpy as np
import yaml

def read_params(config_file):
    with open(config_file) as yaml_file:
        config=yaml.safe_load(yaml_file)
    return config

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml')
    passed_args=parser.parse_args()
    config = read_params(config_file=passed_args.config)
    # print(f"Classes: {config['raw_data']['classes']}")