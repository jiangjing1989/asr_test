import yaml
from collections import UserDict

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

class UserConfig(UserDict):
    def __init__(self, common):
        custom = load_yaml(common)
        super(UserConfig, self).__init__(custom)

    def __missing__(self, key):
        return None
        
