import yaml
''' 
Read Config.yaml file
'''
def read_config(filepath):
    # Read config.yaml file
    with open(filepath, "r") as file:
        content = yaml.safe_load(file)
    return content

