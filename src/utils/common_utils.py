import yaml
import time
import os

''' 
Read Config.yaml file
'''


def read_config(filepath):
    # Read config.yaml file
    with open(filepath, "r") as file:
        content = yaml.safe_load(file)
    return content


# Creating Tensorboard Log folder with unique name
def get_log_path(log_dir="artifacts/log/fit"):
    dirname = time.strftime("LOG_%Y_%M_%d__%H_%M_%S")
    dirpath = os.path.join(log_dir, dirname)
    print(f"Saving logs at : {dirpath}")
    return dirpath
