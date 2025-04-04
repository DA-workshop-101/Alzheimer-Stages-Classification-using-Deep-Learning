import os
import shutil
import argparse
import yaml
import pandas as pd
import numpy as np
from get_data import read_params

def create_folder(config_file: str, image=None) -> None:
    config=read_params(config_file)
    dest_dir = config['load_data']['preprocessed_data']
    classes = config['raw_data']['classes']

    train_dir = os.path.join(dest_dir, 'train')
    test_dir = os.path.join(dest_dir, 'test')

    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print('Train and Test folders already exist....!')
        print("Skipping folder creation....!")
        return
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    print('Folders created successfully....!')

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml')
    passed_args=parser.parse_args()
    create_folder(config_file=passed_args.config)
