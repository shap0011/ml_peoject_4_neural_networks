# Loading all the necessary packages

import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'info'

import logging

# Configure logging manually again
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

import streamlit as st
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Load dataset
def load_data(filepath):
    logging.info(f"Loading data from {filepath}")
    # load the dataset from a CSV file
    df = pd.read_csv(filepath)
    logging.info(f"Data loaded successfully with shape {df.shape}")
    return df

