from logging import root
import os
import shutil
import argparse
import yaml
import pandas as pd
import numpy as np
import random
from get_data import read_params
from create_folder import create_folder

def is_class_folder_empty(dest, train_folder, test_folder):
    """Check if all class subdirectories inside train/ and test/ are empty."""
    for folder in [train_folder, test_folder]:
        for class_folder in os.listdir(folder):  # Iterate through class subdirectories
            class_path = os.path.join(folder, class_folder)
            if os.path.isdir(class_path) and os.listdir(class_path):  
                return False  

    return True  

def train_and_test(config_file):
    config = read_params(config_file)

    create_folder(config_file)

    dest = config['load_data']['preprocessed_data']

    train_folder = os.path.join(dest, "train")
    test_folder = os.path.join(dest, "test")

    if not is_class_folder_empty(dest,  train_folder, test_folder):
        print("Train and Test class folders already contain data. Skipping copying.")
        return 

    root_dir = config['raw_data']['data_src']
    # full_path = config['load_data']['full_path']
    classes = config['raw_data']['classes']
    split_ratio = config['train']['split_ratio']

 

    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"Warning: Class directory '{class_name}' does not exist. Skipping...")
            continue

        files = os.listdir(class_dir)
        total_files = len(files)

        print(f"Class {class_name} -> {total_files} files")

        random.shuffle(files)

        split_index = round(split_ratio * total_files)
        train_files, test_files = files[:split_index], files[split_index:]

        train_dest = os.path.join(train_folder, class_name)
        test_dest = os.path.join(test_folder, class_name)

        os.makedirs(train_dest, exist_ok=True)
        os.makedirs(test_dest, exist_ok=True)

        for file_name in train_files:
            src_path = os.path.join(class_dir, file_name)
            shutil.copy2(src_path, train_dest)

        for file_name in test_files:
            src_path = os.path.join(class_dir, file_name)
            shutil.copy2(src_path, test_dest)

        print(f"'{class_name}' processed: {len(train_files)} train, {len(test_files)} test.")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml')
    passed_args=parser.parse_args()
    train_and_test(config_file=passed_args.config)