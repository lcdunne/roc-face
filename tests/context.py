import json
import models
import utils
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


with open('example/example_data.json', 'r') as f:
    datasets = json.load(f)
