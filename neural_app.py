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
# import functions from app_module
from app_module import functions as func

warnings.filterwarnings("ignore")

try:
    # Set page configuration
    st.set_page_config(page_title="Loan Eligibility App", layout="wide")

    # Define color variables
    header_color = "#c24d2c"  # red color
    div_color = "#feffe0"  # yellow color
    subheader_color = "#000"  # yellow color

    # set the title of the Streamlit app
    st.markdown(f"<h1 style='color: {header_color};'>Project 4. Classification Algorithms</h1>", unsafe_allow_html=True)

    # add subheader
    st.markdown(f"<h2 style='color: {subheader_color};'>Loan Eligibility Prediction model</h2>", unsafe_allow_html=True)

    # load the dataset from a CSV file located in the 'data' folder
    df = func.load_data('data/admission.csv')
    
except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    st.error("Something went wrong! Please check the logs or try again later.")