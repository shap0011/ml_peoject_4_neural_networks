# Predicting Chances of Admission at UCLA - Neural Networks Project

This project builds a **classification model** using **Neural Networks (MLPClassifier)** to predict the **chances of admission** into the **University of California, Los Angeles (UCLA)** based on a student's academic and research profile.

It is developed as a **Streamlit Web App** that walks through the entire Machine Learning pipeline:

- Data Loading and Cleaning

- Data Visualization

- Data Preparation

- Model Building (Neural Network)

- Model Evaluation (Confusion Matrix, Loss Curve)

## Technologies Used

- [Streamlit](https://streamlit.io/) - For building the interactive web app
- [Scikit-learn](https://scikit-learn.org/) - For machine learning models
- [Pandas](https://pandas.pydata.org/) - For data manipulation
- [Logging](https://docs.python.org/3/library/logging.html) - For backend log management

## Project Structure

- **.streamlit/**
  - `config.toml` — Theme setting
- `neural_app.py` — Main Streamlit app
- **app_module/**
  - `__init__.py`
  - `functions.py` — All helper functions
- **data/**
  - `admission.csv` — Raw dataset
- `requirements.txt` — List of Python dependencies
- `README.md` — Project documentation

## Dataset

The dataset contains the following features:

| Feature	         | Description                                     |
|--------------------|-------------------------------------------------|
| GRE_Score	         | GRE exam score (out of 340)                     |
| TOEFL_Score	     | TOEFL exam score (out of 120)                   |
| University_Rating	 | Ranking of the Bachelor University (out of 5)   |
| SOP	             | Strength of Statement of Purpose (out of 5)     |
| LOR	             | Strength of Letter of Recommendation (out of 5) |
| CGPA	             | Undergraduate GPA (out of 10)                   |
| Research	         | Research experience (0 = No, 1 = Yes)           |
| Admit_Chance	     | Probability of admission (0 to 1)               |

## App Features
- **Data Exploration:**

    - View raw data

    - See summary statistics

    - Analyze unique values

    - View correlation matrix and scatterplots

- **Data Preparation:**

    - Target transformation (`Admit_Chance` → categorical: 0 or 1)

    - One-Hot Encoding for categorical features

    - Scaling of numerical features (MinMaxScaler)

- **Model Training:**

    - Build a **Feedforward Neural Network** using **MLPClassifier**

    - Architecture: 2 hidden layers, each with 3 neurons

    - Batch Size: 50

    - Max Iterations: 200

- **Model Evaluation:**

    - Display training log

    - Show training and test accuracy

    - Plot **Confusion Matrix** (table + heatmap)

    - Plot **Loss Curve** during training

- **Error Handling:**

    - Handles errors during data loading, model training, and prediction gracefully.

    - Logs all errors and important messages.

## Key Learning Outcomes
- Building a **classification neural network** with **MLPClassifier**.

- Understanding the full **ML workflow:** preprocessing, modeling, evaluation.

- Importance of **data scaling** and **avoiding data leakage**.

- **Capturing and displaying** real-time training logs inside Streamlit.

- Implementing **robust error handling** and **debugging** practices.

## How to Run the App Locally

1. **Clone the repository**

```bash```
git clone https://github.com/shap0011/ml_peoject_4_neural_networks.git
cd ml_peoject_4_neural_networks

2. **Install the required packages**

```bash```
    pip install -r requirements.txt

3. **Run the App**

```bash```
streamlit run neural_app.py

4. Open the URL shown (usually http://localhost:8501) to view the app in your browser!

## Deployment
The app is also deployed on Streamlit Cloud.
Click [![Here](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ucla-admission-prediction-app-shap0011.streamlit.app/) to view the live app.

## Author
Name: Olga Durham

LinkedIn: [\[Olga Durham LinkedIn Link\]](https://www.linkedin.com/in/olga-durham/)

GitHub: [\[Olga Durham GitHub Link\]](https://github.com/shap0011)


## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://literate-potato-w5v6wwwrvpj357qx.github.dev/)

## License

This project is licensed under the MIT License.  
Feel free to use, modify, and share it.  
See the [LICENSE](./LICENSE) file for details.