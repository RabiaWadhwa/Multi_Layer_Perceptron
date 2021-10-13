from src.utils.common_utils import read_config
from src.utils.data_mgmt import get_data
import argparse


def training(config_path):
    config = read_config(config_path)
    print(config)

    validation_datasize = config["params"]["validation_datasize"]
    print(validation_datasize)
    (X_train,y_train),(X_valid,y_valid),(X_test,y_test) = get_data(validation_datasize)

if __name__ == '__main__':
     
     # Create the parser
     parser = argparse.ArgumentParser()
     
     # Add an argument , # arg name - config or c , default file - config.yaml
     parser.add_argument("--config","-c",default="config.yaml") 
    
     # Parse the argument
     args = parser.parse_args()

    # Print User-input argument 
     print("Config input by user ",args.config)
     training(args.config)
