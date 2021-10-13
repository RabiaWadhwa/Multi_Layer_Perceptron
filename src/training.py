from utils.common_utils import read_config
import argparse
def trainings(config_path):
    config = read_config(config_path)
    print(config)


if __name__ == '__main__':
     
     # Create the parser
     parser = argparse.ArgumentParser()
     
     # Add an argument , # arg name - config or c , default file - config.yaml
     parser.add_argument("--config","-c",default="config.yaml") 
    
     # Parse the argument
     args = parser.parse_args()

    # Print User-input argument 
     print("Config input by user ",args.config)
     trainings(args.config)
