import json

def load_config(pth):
    with open(pth, 'r') as f:
        return json.load(f)

