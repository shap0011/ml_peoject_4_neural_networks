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

try:
    # Set page configuration
    st.set_page_config(page_title="Loan Eligibility App", layout="wide")

    # Define color variables
    header_color = "#1c8787"  # dark green
    subheader_color = "#dc2e2e"  # dark red

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
    
    # first five rows
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
    
    #---------------------------------------
    
    # Add a subheader
    st.markdown(f"<h3 style='color: {subheader_color};'>Scatterplot of GRE Score vs TOEFL Score</h3>", unsafe_allow_html=True)

    # Add a subheader
    # st.subheader("Scatterplot of GRE Score vs TOEFL Score (Mini Version)")

    # Create a figure
    fig, ax = plt.subplots(figsize=(4, 2)) 

    # Create the scatterplot with dots
    sns.scatterplot(
        data=data,
        x='GRE_Score',
        y='TOEFL_Score',
        hue='Admit_Chance',
        ax=ax,
        s=8 
    )

    # Set labels
    ax.set_xlabel("GRE Score", fontsize=6)
    ax.set_ylabel("TOEFL Score", fontsize=6)

    # Set title
    ax.set_title("GRE vs TOEFL by Admit Chance", fontsize=8)

    # Tick labels
    ax.tick_params(axis='both', labelsize=5)

    # Legend
    legend = ax.legend(title='Admit Chance', fontsize=5, title_fontsize=6)
    # Make legend background slightly transparent
    legend.get_frame().set_alpha(0.7)  

    # Display the plot in Streamlit
    st.pyplot(fig, use_container_width=False)

    #------------------------------------------
         
    # display observations
    st.markdown("""
                **Observations:**

                - There is a linear relationship between GRE and TOEFL scores. 
                This implies that students scoring high in one of them would score high in the other as well.
                - We can see a distinction between students who were admitted (denoted by orange) 
                vs those who were not admitted (denoted by blue). We can see that majority of students 
                who were admitted have GRE score greater than 320, TOEFL score greater than 105.    
                """)
    
    #----------------------------------------
    
    # Add a subheader
    st.markdown(f"<h3 style='color: {subheader_color};'>Data Preparation</h3>", unsafe_allow_html=True)
                 
    # display task explanation
    st.markdown("""
                This dataset contains both numerical and categorical variables. 
                We need to treat them first before we pass them onto the neural network. 
                We will perform below pre-processing steps:

                - One hot encoding of categorical variables
                - Scaling numerical variables
                
                An important point to remember: Before we scale numerical variables, 
                we would first split the dataset into train and test datasets and perform 
                scaling separately. Otherwise, we would be leaking information from the test 
                data to the train data and the resulting model might give a false sense of good performance. 
                This is known as **data leakage** which we would want to avoid. 
                
                In this dataset, although the variable **University Rating** is encoded as a numerical variable. 
                But it is denoting or signifying the quality of the university, so that is why this is 
                a categorical variable and we would be creating one-hot encoding or dummy variables for this variable.
                """)
    
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
    
    #---------------------------------
               
    # Display dataset info
    st.markdown(f"<h3 style='color: {subheader_color};'>Get unique values for LOR</h3>", unsafe_allow_html=True)
    
    # Get unique values
    unique_values = data['LOR'].unique()

    # Convert array to a string, separated by commas
    unique_values_str = ", ".join(map(str, unique_values))

    # Display as simple horizontal text
    st.write(unique_values_str)
        
    #---------------------------------
                 
    # Display dataset info
    st.markdown(f"<h3 style='color: {subheader_color};'>Convert 'University_Rating' to categorical type. Dataset Info</h3>", unsafe_allow_html=True)    
    
    # convert University_Rating to categorical type
    data['University_Rating'] = data['University_Rating'].astype('object')
    data['Research'] = data['Research'].astype('object')

    # Create a string buffer
    buffer = io.StringIO()

    # Capture the output of data.info() into the buffer
    data.info(buf=buffer)

    # Get the string from the buffer
    info_str = buffer.getvalue()

    # Display it as preformatted text
    # st.text(info_str)
    st.code(info_str, language='text')
    
    #-------------------------------
    
    st.write("The first five rows of the dataset:")    
    # first five rows 
    st.dataframe(data.head(), use_container_width=False)
    
    #--------------------------------
    
    # Dummy variables
    st.markdown(f"<h3 style='color: {subheader_color};'>Create dummy variables for all 'object' type variables except 'Loan_Status'</h3>", unsafe_allow_html=True)    
    st.write("The first two rows of the dataset:")
    
    # Create dummy variables for all 'object' type variables except 'Loan_Status'
    clean_data = pd.get_dummies(data, columns=['University_Rating','Research'],dtype='int')
    clean_data.head(2)
    # first two rows of the dataset
    st.dataframe(clean_data.head(2), use_container_width=False)
    
    #-------------------------------
    
    # Split the Data into Train and Test Sets
    # add subheader
    st.markdown(f"<h3 style='color: {subheader_color};'>Split the Data into Train and Test Sets</h3>", unsafe_allow_html=True)

    # Separate features (X) and target variable (y)
    x = clean_data.drop(['Admit_Chance'], axis=1)
    y = clean_data['Admit_Chance']

    # Split the data (Stratify on target to keep class balance)
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=0.2, random_state=123, stratify=y
    )

    st.success("Data split into training and testing sets.")

    # Scaling Numerical Variables
    st.markdown(f"<h3 style='color: {subheader_color};'>Scaling Numerical Variables</h3>", unsafe_allow_html=True)

    st.write(
        "Perform scaling on the numerical variables separately for train and test sets. "
        "We will use `.fit()` to calculate the mean and standard deviation, and `.transform()` to apply the scaling."
    )
    
    #-------------------------------

    # Add a subheader
    st.subheader("Fitting the Scaler (MinMaxScaler)")

    # Initialize and fit the scaler
    scaler = MinMaxScaler()
    scaler.fit(xtrain)

    # Get minimum and maximum values calculated by the scaler
    data_min = scaler.data_min_
    data_max = scaler.data_max_

    # Create a nice message
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
    
    #------------------------------- 

    # Initialize the scaler
    scaler = MinMaxScaler()

    # Fit the scaler on training data only
    scaler.fit(xtrain)

    st.success("Scaler fitted to the training data.")

    # Display Scaler Maximum Values
    st.subheader("Scaler Data Max Values")

    # Extract maximum values after scaling fit
    scaler_data_max = scaler.data_max_

    # Convert array to a comma-separated string
    scaler_data_max_array = ", ".join(map(str, scaler_data_max))

    # Display max values horizontally
    st.write(scaler_data_max_array)

    # Display Scaler Minimum Values
    st.subheader("Scaler Data Min Values")

    # Extract minimum values after scaling fit
    scaler_data_min = scaler.data_min_

    # Convert array to a comma-separated string
    scaler_data_min_array = ", ".join(map(str, scaler_data_min))

    # Display min values horizontally
    st.write(scaler_data_min_array)
    
    
    # Transform training and testing sets
    xtrain_scaled = scaler.transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    # Convert scaled training set back into a DataFrame for better display
    xtrain_scaled_df = pd.DataFrame(xtrain_scaled, columns=xtrain.columns)

    # Show first few rows
    st.subheader("First 5 Rows of Scaled Training Data")
    st.dataframe(xtrain_scaled_df.head(), use_container_width=False)

    # Show descriptive statistics of the scaled data
    st.subheader("Statistics of Scaled Training Data")
    st.dataframe(xtrain_scaled_df.describe(), use_container_width=False)
    
    #----------------------------------
    
    # Add a subheader
    st.subheader("Distribution of GRE and TOEFL Scores (Before and After Scaling)")

    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(4, 3)) 

    # Top-left: Original GRE Score
    sns.histplot(data['GRE_Score'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Original GRE Score", fontsize=6)
    axes[0, 0].set_xlabel("GRE_Score", fontsize=4)   
    axes[0, 0].set_ylabel("Count", fontsize=4)        
    axes[0, 0].tick_params(axis='both', labelsize=3)  

    # Top-right: Scaled GRE Score
    sns.histplot(xtrain.iloc[:, 0], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Scaled GRE Score", fontsize=6)
    axes[0, 1].set_xlabel("GRE_Score", fontsize=4)
    axes[0, 1].set_ylabel("Count", fontsize=4)
    axes[0, 1].tick_params(axis='both', labelsize=4)

    # Bottom-left: Original TOEFL Score
    sns.histplot(data['TOEFL_Score'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Original TOEFL Score", fontsize=6)
    axes[1, 0].set_xlabel("TOEFL_Score", fontsize=4)
    axes[1, 0].set_ylabel("Count", fontsize=4)
    axes[1, 0].tick_params(axis='both', labelsize=4)

    # Bottom-right: Scaled TOEFL Score
    sns.histplot(xtrain.iloc[:, 1], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Scaled TOEFL Score", fontsize=6)
    axes[1, 1].set_xlabel("TOEFL_Score", fontsize=4)
    axes[1, 1].set_ylabel("Count", fontsize=4)
    axes[1, 1].tick_params(axis='both', labelsize=4)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig, use_container_width=False)

    #---------------------------
    
    # Scaling Numerical Variables
    st.markdown(f"<h3 style='color: {subheader_color};'>Neural Network Architecture</h3>", unsafe_allow_html=True)
    
    st.markdown("""
                In neural networks, there are so many hyper-parameters that you can play around with and tune the network to get the best results. Some of them are -

                1. Number of hidden layers
                2. Number of neurons in each hidden layer
                3. Activation functions in hidden layers
                4. Batch size
                5. Learning rate
                6. Dropout
                """)
        
    st.write("Let's build a feed forward neural network with 2 hidden layers. Remember, always start small.")
    # import io
    # import contextlib
    # from sklearn.neural_network import MLPClassifier

    # Add subheader
    st.markdown(f"<h3 style='color: {subheader_color};'>Training the model</h3>", unsafe_allow_html=True)
    st.write("Feedforward Neural Network Training (MLP Classifier)")

    # Create a string buffer to capture output
    buffer = io.StringIO()

    # Train the model and capture the training output
    with contextlib.redirect_stdout(buffer):
        MLP = MLPClassifier(
            hidden_layer_sizes=(3, 3),
            batch_size=50,
            max_iter=200,
            random_state=123,
            verbose=True    # <<< IMPORTANT
        )
        MLP.fit(xtrain_scaled, ytrain)

    # Get the captured training output
    training_log = buffer.getvalue()

    # Display the training log in Streamlit
    st.code(training_log, language="text")
    
    #-------------------------------
    
    st.write("Make predictions on train and check accuracy of the model")
    
    # make predictions on train
    ypred_train = MLP.predict(xtrain_scaled)
    # check accuracy of the model
    model_train_ac = accuracy_score(ytrain, ypred_train)
    st.write(f"Accuracy of the train model: `{model_train_ac}`")
    
    #---------------------------------------
    
    st.write("Make predictions on test and check accuracy of the model")
    # make Predictions
    ypred = MLP.predict(xtest_scaled)
    # check accuracy of the model
    model_test_ac = accuracy_score(ytest, ypred)
    st.write(f"Accuracy of the test model: `{model_test_ac}`")
    
    #----------------------------------

    # Calculate the confusion matrix
    cm = confusion_matrix(ytest, ypred)

    # Display it as a table
    st.subheader("Confusion Matrix (Table)")
    st.dataframe(pd.DataFrame(cm), use_container_width=False)
    
    
    # Calculate the confusion matrix
    cm = confusion_matrix(ytest, ypred)

    # Create a heatmap
    st.subheader("Confusion Matrix (Heatmap)")

    fig, ax = plt.subplots(figsize=(2.4, 2))  # Small figure
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)

    ax.set_xlabel('Predicted Labels', fontsize=6)
    ax.set_ylabel('True Labels', fontsize=6)
    ax.set_title('Confusion Matrix', fontsize=8)
    ax.tick_params(axis='both', labelsize=5)

    # Display the heatmap in Streamlit
    st.pyplot(fig, use_container_width=False)
    
    #----------------------------------
    
    # Plotting loss curve
    # Add a subheader
    st.markdown(f"<h3 style='color: {subheader_color};'>Loss Curve During Neural Network Training</h3>", unsafe_allow_html=True)

    # Extract the loss values from the trained model
    loss_values = MLP.loss_curve_

    # Create a figure
    fig, ax = plt.subplots(figsize=(5, 3))  # smaller figure for Streamlit

    # Plot the loss values
    ax.plot(loss_values, label='Loss', color='blue')

    # Customize the plot
    ax.set_title('Loss Curve', fontsize=8)
    ax.set_xlabel('Iterations', fontsize=6)
    ax.set_ylabel('Loss', fontsize=6)
    ax.legend(fontsize=5)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=5)

    # Display the plot in Streamlit
    st.pyplot(fig, use_container_width=False)
    
    #----------------------------------
    
    # Add a subheader
    st.markdown(f"<h3 style='color: {subheader_color};'>Conclusion</h3>", unsafe_allow_html=True)
    st.markdown("""
                In this case study,

                - We have learned how to build a neural network for a classification task.
                - **Can you think of a reason why, we could get such low accuracy?**
                - You can further analyze the misclassified points and see if there is a pattern or if they were outliers that our model could not identify.               
                """)
      
except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    st.error("Something went wrong! Please check the logs or try again later.")