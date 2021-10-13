import yaml

def read_config(filepath):
    with open(filepath,"r") as file:
        content = yaml.safe_load(file)
    
    return content
