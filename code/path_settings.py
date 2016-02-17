import os

currentPath = os.path.dirname(os.path.realpath(__file__))

# Setup does not use this reference, adjust in setup.py if changing.
DATA_PATH = os.path.join(currentPath,'../data')

BEST_MODEL_PATH = os.path.join(currentPath,'../models/best')
LAST_MODEL_PATH = os.path.join(currentPath,'../models/last')
