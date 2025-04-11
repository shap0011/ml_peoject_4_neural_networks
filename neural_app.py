# Loading all the necessary packages

import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'info'

import logging

# Configure logging manually again
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

import logging
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import warnings

# import functions from app_module
from app_module import functions as func

warnings.filterwarnings("ignore")

try:
    
    #-------- page setting, header, intro ---------------------
    
    # Set page configuration
    st.set_page_config(page_title="🎓 UCLA Admission Prediction App", layout="wide")

    # Define color variables
    header_color = "#1c8787"  # dark green

    # set the title of the Streamlit app
    st.markdown(f"<h1 style='color: {header_color};'>🎓 UCLA Admission Prediction App</h1>", unsafe_allow_html=True)

    #-------- the app overview -----------------------------
    
    
    #-------- user instructions -------------------------------
    
    
    #-------- the dataset loading -----------------------------
    
    # load the dataset from a CSV file located in the 'data' folder  
    try:
        data = func.load_data('data/admission.csv')
    except Exception as e:
        logging.error(f"Error loading data: {e}", exc_info=True)
        st.error("Failed to load the dataset. Please check the file path or format.")
        st.stop()

    #-------- preprocessing data -----------------------------

    data['Admit_Chance'] = (data['Admit_Chance'] >= 0.8).astype(int)
    data = data.drop('Serial_No', axis=1)
    data['University_Rating'] = data['University_Rating'].astype('object')
    data['Research'] = data['Research'].astype('object')
    clean_data = pd.get_dummies(data, columns=['University_Rating', 'Research'], dtype=int)

    #-------- split and scale data -----------------------------

    # Split features/target
    X = clean_data.drop('Admit_Chance', axis=1)
    y = clean_data['Admit_Chance']

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    #-------- Train a Neural Network -----------------------------

    # Train a Neural Network (MLP Classifier)
    model = MLPClassifier(hidden_layer_sizes=(3, 3), batch_size=50, max_iter=200, random_state=123)
    model.fit(X_scaled, y)

    st.markdown("---")
    st.subheader("📝 Enter Your Details to Predict Admission Chance")

    with st.form(key="admission_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            GRE_Score = st.number_input("GRE Score (out of 340)", min_value=260, max_value=340, value=320)
            TOEFL_Score = st.number_input("TOEFL Score (out of 120)", min_value=0, max_value=120, value=105)

        with col2:
            SOP = st.slider("SOP Strength (1-5)", 1.0, 5.0, 3.0, step=0.5)
            LOR = st.slider("LOR Strength (1-5)", 1.0, 5.0, 3.0, step=0.5)

        with col3:
            CGPA = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, value=8.5)
            Research = st.selectbox("Research Experience", options=[0, 1])
            University_Rating = st.selectbox("University Rating (1-5)", options=[1, 2, 3, 4, 5])

        submit_button = st.form_submit_button(label="Predict Admission")

    #-------- after submit -----------------------------
    
    if submit_button:
        try:
            # Create user input DataFrame
            user_input = pd.DataFrame({
                'GRE_Score': [GRE_Score],
                'TOEFL_Score': [TOEFL_Score],
                'SOP': [SOP],
                'LOR': [LOR],
                'CGPA': [CGPA],
                f'University_Rating_{University_Rating}': [1],
                f'Research_{Research}': [1]
            })

            # Add missing dummy columns
            for col in X.columns:
                if col not in user_input.columns:
                    user_input[col] = 0

            # Reorder columns
            user_input = user_input[X.columns]

            # Scale user input
            user_input_scaled = scaler.transform(user_input)

            # Predict
            prediction = model.predict(user_input_scaled)[0]

    #-------- Display result -----------------------------

            if prediction == 1:
                st.success("✅ Congratulations! Based on your profile, you have a good chance of being admitted to UCLA.")
            else:
                st.error("❌ Unfortunately, your profile shows a low chance of admission. Consider improving some aspects!")

        except Exception as e:
            logging.error(f"Prediction failed: {e}", exc_info=True)
            st.error("Prediction failed. Please check your inputs or try again.")
      
except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    st.error("Something went wrong! Please check the logs or try again later.")