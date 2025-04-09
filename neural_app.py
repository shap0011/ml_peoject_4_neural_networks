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
# import functions from app_module
from app_module import functions as func

warnings.filterwarnings("ignore")

try:
    # Set page configuration
    st.set_page_config(page_title="Loan Eligibility App", layout="wide")

    # Define color variables
    header_color = "#1c8787"  # dark red color
    # div_color = "#e3969c"  # dark pink
    subheader_color = "#dc2e2e"  # dust rose

    # set the title of the Streamlit app
    st.markdown(f"<h1 style='color: {header_color};'>Project 4. Neural Networks</h1>", unsafe_allow_html=True)

    # add subheader
    st.markdown(f"<h2 style='color: {subheader_color};'>Predicting Chances of Admission at UCLA</h2>", unsafe_allow_html=True)
    
    # add project scope
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
    
    # add data dictionary
    st.markdown("""
                ### Data Dictionary:
                                
                The dataset contains several parameters which are considered important during 
                the application for Masters Programs. The parameters included are :

                - **GRE_Score:** (out of 340)
                - **TOEFL_Score:** (out of 120)
                - **University_Rating:** It indicates the Bachelor University ranking (out of 5)
                - **SOP:** Statement of Purpose Strength (out of 5)
                - **LOR:** Letter of Recommendation Strength (out of 5)
                - **CGPA:** Student's Undergraduate GPA(out of 10)
                - **Research:** Whether the student has Research Experience (either 0 or 1)
                - **Admit_Chance:** (ranging from 0 to 1)              
                """)
    
    
    # add subheader
    st.markdown(f"<h3 style='color: {subheader_color};'>Loading the libraries and the dataset</h3>", unsafe_allow_html=True)

    # load the dataset from a CSV file located in the 'data' folder
    data = func.load_data('data/admission.csv')

    st.write(f"The first five rows of the dataset")
    
    # first_five_rows = data.head()
    # st.dataframe(first_five_rows)
    
    # first_five_rows = 
    st.dataframe(data.head(), use_container_width=False)
    
    # add dataset and task explanation
    st.markdown("""
                - In the above dataset, the target variable is **Admit_Chance**
                - To make this a classification task, let's convert the target variable into a categorical variable by using a threshold of 80%
                - We are assuming that if **Admit_Chance** is more than 80% then chance of **Admit** would be 1 (i.e. yes) otherwise it would be 0 (i.e. no)           
                """)
    
    #--------------------------------
    
    st.write("Convert the target variable into a categorical variable.")

    # Converting the target variable into a categorical variable
    data['Admit_Chance'] = (data['Admit_Chance'] >= 0.8).astype(int)

    # Confirm the transformation
    st.success("Target variable 'Admit_Chance' converted: values are now 0 or 1.")
    
    #--------------------------------

    # Show the first five rows
    st.markdown(f"<h3 style='color: {subheader_color};'>First Five Rows of the Dataset (After Transformation)</h3>", unsafe_allow_html=True)
    st.dataframe(data.head(), use_container_width=False)
    
    #-------------------------------
    
    st.markdown(f"<h3 style='color: {subheader_color};'>Drop any unnecessary columns and check the info of the data</h3>", unsafe_allow_html=True)
    
    # Dropping columns
    data = data.drop(['Serial_No'], axis=1)
    st.dataframe(data.head(), use_container_width=False)
    
    # Get the number of rows and columns
    rows, columns = data.shape

    # Create a small DataFrame
    shape_df = pd.DataFrame({
        'Rows': [rows],
        'Columns': [columns]
    })
    
    #-------------------------------

    # Display as a table
    st.markdown(f"<h3 style='color: {subheader_color};'>Dataset Shape</h3>", unsafe_allow_html=True)

    st.dataframe(shape_df, use_container_width=False)
    
    #-------------------------------
    
    # Display dataset info
    st.markdown(f"<h3 style='color: {subheader_color};'>Dataset Info</h3>", unsafe_allow_html=True)

    # Create a string buffer
    buffer = io.StringIO()

    # Capture the output of data.info() into the buffer
    data.info(buf=buffer)

    # Get the string from the buffer
    info_str = buffer.getvalue()

    # Display it as preformatted text
    # st.text(info_str)
    st.code(info_str, language='text')
    
    #-----------------------------------
    
    # Display the 'SOP' column's unique values
    st.markdown(f"<h3 style='color: {subheader_color};'>The 'SOP' Column's Unique Values</h3>", unsafe_allow_html=True)

    # Get unique values
    unique_values = data['SOP'].unique()

    # Convert array to a string, separated by commas
    unique_values_str = ", ".join(map(str, unique_values))

    # Display as simple horizontal text
    st.write(unique_values_str)
    
    #-----------------------------------
    
    # Display the 'CGPA' column's unique values
    st.markdown(f"<h3 style='color: {subheader_color};'>The 'CGPA' Column's Unique Values</h3>", unsafe_allow_html=True)

    # Get unique values
    unique_values = data['CGPA'].unique()

    # Convert array to a string, separated by commas
    unique_values_str = ", ".join(map(str, unique_values))

    # Display as simple horizontal text
    st.write(unique_values_str)
    
    #-------------------------------------
    
    # display observations
    st.markdown("""
                **Observations:**
                
                - There are 500 observations and 8 columns in the data
                - All the columns are of numeric data type.
                - There are no missing values in the data       
                """)

    #----------------------------------------
    
    # Display summary statistics of the data
    st.markdown(f"<h3 style='color: {subheader_color};'>Summary statistics of the data</h3>", unsafe_allow_html=True)
    st.dataframe(data.describe(), use_container_width=False)
        
    #-------------------------------------
    
    # display observations
    st.markdown("""
                **Observations:**

                - The average GRE score of students applying for UCLA is ~316 out of 340. Some students scored full marks on GRE.
                - The average TOEFL score of students applying for UCLA is ~107 out of 120. Some students scored full marks on TOEFL.
                - There are students with all kinds of ratings for bachelor's University, SOP, and LOR - ratings ranging from 1 to 5.
                - The average CGPA of students applying for UCLA is 8.57.
                - Majority of students (~56%) have research experience.
                - As per our assumption, on average 31% of students would get admission to UCLA.     
                """)
    
    #----------------------------------------
    
    # visualize the dataset
    st.markdown(f"<h3 style='color: {subheader_color};'>Visualize the dataset to see some patterns</h3>", unsafe_allow_html=True)
    st.dataframe(data.corr(), use_container_width=False)

       
except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    st.error("Something went wrong! Please check the logs or try again later.")