# Loading all the necessary packages

import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'debug'

import logging

# Configure logging manually again
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# import Streamlit for building web app
import streamlit as st
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
import io
# import train test split
from sklearn.model_selection import train_test_split
# import standard scaler
from sklearn.preprocessing import MinMaxScaler
# import the model
from sklearn.neural_network import MLPClassifier
import contextlib
# import evaluation metrics
from sklearn.metrics import confusion_matrix, accuracy_score

# import functions from app_module
from app_module import functions as func

warnings.filterwarnings("ignore")

# Load dataset
def load_data(filepath):
    logging.info(f"Loading data from {filepath}")
    # load the dataset from a CSV file
    df = pd.read_csv(filepath)
    logging.info(f"Data loaded successfully with shape {df.shape}")
    return df