from src.utils.common_utils import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model
import argparse


def training(config_path):
    config = read_config(config_path)
    print(config)

    validation_datasize = config["params"]["validation_datasize"]
    print(f"Validation Datasize {validation_datasize}")
    (X_train,y_train),(X_valid,y_valid),(X_test,y_test) = get_data(validation_datasize)

    no_classes,loss_fn,optimiser,metric = config["params"]['no_classes'],config["params"]['loss_function'] ,config["params"]['optimizer'],config["params"]['metrics']
    print(f"#Classes-{no_classes} Loss function-{loss_fn} Optimizer-{optimiser} Metrics-{metric}")
    
    model_classifier = create_model(metric,optimiser,loss_fn,no_classes)
    history = model_classifier.fit(X_train,y_train, epochs =config["params"]["epochs"], validation_data=(X_valid,y_valid) )


if __name__ == '__main__':
     
     # Create the parser
     parser = argparse.ArgumentParser()
     
     # Add an argument , # arg name - config or c , default value - config.yaml
     parser.add_argument("--config","-c",default="config.yaml") 
    
     # Parse the argument
     args = parser.parse_args()

    # Print User-input argument 
     print("Config input by user ",args.config)
     training(args.config)
    
    # To pass the argument at runtime,
    # python training.py -c = config.yaml 
    # python training.py --config = config.yaml  --secret= secret.yaml 
    # python training.py config.yaml secret.yaml  # positional arguments
    # # can pass more arguments too