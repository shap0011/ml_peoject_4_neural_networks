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
    st.markdown(f"<h1 style='color: {header_color};'>Project 4. Neural Networks</h1>", unsafe_allow_html=True)

    # add subheader
    st.markdown(f"<h2 style='color: {subheader_color};'>Predicting Chances of Admission at UCLA</h2>", unsafe_allow_html=True)
    
    # add subheader
    st.markdown("""
                ### Project Scope:
                
                The world is developing rapidly, and continuously looking for the best knowledge and experience among people. 
                This motivates people all around the world to stand out in their jobs and look for higher degrees that can help 
                them in improving their skills and knowledge. As a result, the number of students applying for Master's programs 
                has increased substantially.

                The current admission dataset was created for the prediction of admissions into the University of California, 
                Los Angeles (UCLA). It was built to help students in shortlisting universities based on their profiles. 
                The predicted output gives them a fair idea about their chances of getting accepted.

                **Your Role:**

                Build a classification model using **Neural Networks** to predict a student's chance of admission into UCLA.

                **Specifics:**

                - Target variable: Admit_Chance
                - Machine Learning task: Classification model
                - Input variables: Refer to data dictionary below
                - Success Criteria: Accuracy of 90% and above                
                """)


    # load the dataset from a CSV file located in the 'data' folder
    df = func.load_data('data/admission.csv')
    
except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    st.error("Something went wrong! Please check the logs or try again later.")