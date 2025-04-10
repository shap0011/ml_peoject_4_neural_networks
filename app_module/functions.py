# Loading all the necessary packages

import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'info'

import logging

# Configure logging manually again
logging.basicConfig(
    level=logging.INFO,
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

warnings.filterwarnings("ignore")

# Load dataset
def load_data(filepath):
    logging.info(f"Loading data from {filepath}")
    # load the dataset from a CSV file
    df = pd.read_csv(filepath)
    logging.info(f"Data loaded successfully with shape {df.shape}")
    return df

# Display dataset shape
def display_shape(df):
    # display subheader
    st.subheader(f"Dataset Shape")
    # Get the number of rows and columns
    rows, columns = df.shape
    # Create a small DataFrame
    shape_df = pd.DataFrame({'Rows': [rows], 'Columns': [columns]})
    st.dataframe(shape_df, use_container_width=False)
    logging.info(f"Dataset shape displayed: {rows} rows, {columns} columns")
    

# Display dataset info
def display_info(df):
    # display subheader
    st.subheader(f"Dataset Info")
    # Create a string buffer
    buffer = io.StringIO()
    # Capture the output of data.info() into the buffer
    df.info(buf=buffer)
    # Get the string from the buffer
    info_str = buffer.getvalue()
    # Display it as preformatted text
    st.code(info_str, language="text")
    logging.info("Dataset info displayed.")


# Display unique values of a column
def display_unique_values(df, column_name):
    # display subheader
    st.subheader(f"Unique values in '{column_name}' column")
    # Get unique values
    unique_vals = df[column_name].unique()
    # Convert array to a string, separated by commas
    unique_vals_str = ", ".join(map(str, unique_vals))
    # Display as simple horizontal text
    st.write(unique_vals_str)
    logging.info(f"Displayed unique values for column: {column_name}")


# Display correlation matrix
def display_correlation(df):
    # display subheader
    st.subheader("Correlation Matrix")
    # compute correlation
    corr_matrix = df.corr()
    #display correlation matrix
    st.dataframe(corr_matrix, use_container_width=False)
    logging.info("Correlation matrix displayed.")


# Plot scatterplot
def plot_scatter(data, x_col, y_col, hue_col):
    # add subheader
    st.subheader(f"Scatterplot of {x_col} vs {y_col}")
    # Create a figure
    fig, ax = plt.subplots(figsize=(4, 2))
     # Create the scatterplot with dots
    sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax, s=8)
    # Set labels
    ax.set_xlabel(x_col, fontsize=6)
    ax.set_ylabel(y_col, fontsize=6)
    # Set title
    ax.set_title(f"{x_col} vs {y_col}", fontsize=8)
    # Tick labels
    ax.tick_params(axis='both', labelsize=5)
    # Legend
    legend = ax.legend(title=hue_col, fontsize=5, title_fontsize=6)
     # Make legend background slightly transparent
    legend.get_frame().set_alpha(0.7)
    # Display the plot in Streamlit
    st.pyplot(fig, use_container_width=False)
    logging.info(f"Scatterplot displayed: {x_col} vs {y_col}")


# Fit and display MinMaxScaler info
def fit_scaler_and_display(xtrain):
    # Add a subheader
    st.subheader("Fitting MinMaxScaler")
    # Initialize and fit the scaler
    scaler = MinMaxScaler()
    scaler.fit(xtrain)
    # Get minimum and maximum values calculated by the scaler
    data_min = scaler.data_min_
    data_max = scaler.data_max_
     # Create a message
    scaler_message = f"""
    Scaler fitted successfully!

    **Features:**
    - {', '.join(xtrain.columns)}

    **Minimum Values (Before Scaling):**
    - {', '.join(map(str, data_min))}

    **Maximum Values (Before Scaling):**
    - {', '.join(map(str, data_max))}
    """
    # Display the message in Streamlit
    st.info(scaler_message)
    logging.info("Scaler fitted and info displayed.")
    return scaler


# Plot confusion matrix heatmap
def plot_confusion_matrix(y_true, y_pred):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Display it as a table
    st.subheader("Confusion Matrix (Table)")
    st.dataframe(pd.DataFrame(cm), use_container_width=False)
    # Create a figure
    fig, ax = plt.subplots(figsize=(2.4, 2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted Labels', fontsize=6)
    ax.set_ylabel('True Labels', fontsize=6)
    ax.set_title('Confusion Matrix', fontsize=8)
    ax.tick_params(axis='both', labelsize=5)
    # add a subheader
    st.subheader("Confusion Matrix (Heatmap)")
    # Display the heatmap
    st.pyplot(fig, use_container_width=False)
    logging.info("Confusion matrix heatmap displayed.")


# Plot loss curve
def plot_loss_curve(model):
    # Add a subheader
    st.subheader("Loss Curve During Neural Network Training")
    # Extract the loss values from the trained model
    loss_values = model.loss_curve_
    # Create a figure
    fig, ax = plt.subplots(figsize=(5, 3))
    # Plot the loss values
    ax.plot(loss_values, label='Loss', color='blue')
     # Customize the plot
    ax.set_title('Loss Curve', fontsize=8)
    ax.set_xlabel('Iterations', fontsize=6)
    ax.set_ylabel('Loss', fontsize=6)
    ax.legend(fontsize=5)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=5)
     # Display the plot
    st.pyplot(fig, use_container_width=False)
    logging.info("Loss curve plotted and displayed.")