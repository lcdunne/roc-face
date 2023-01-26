import json
import os
from signal_detection import models, utils

with open('example/example_data.json', 'r') as f:
    datasets = json.load(f)
