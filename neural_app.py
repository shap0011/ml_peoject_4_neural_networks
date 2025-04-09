# Loading all the necessary packages

import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'debug'

import logging

# Configure logging manually again
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

#import numpy as np for numerical computing
import numpy as np
# import pandas for data manipulation
import pandas as pd
# import matplotlib library for creating plots
import matplotlib.pyplot as plt
# import seaborn library for data visualization
import seaborn as sns
# import warnings library to manage warning messages
import warnings
# import functions from app_module
from app_module import functions as func

warnings.filterwarnings("ignore")
