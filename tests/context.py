import json
import os
from roc_face import models, utils

with open('example/example_data.json', 'r') as f:
    datasets = json.load(f)
