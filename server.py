import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

model = joblib.load("Model/depression_prediction_model.pkl")
scaler = joblib.load("Scaler/depression_prediction_scaler.pkl")

st.set_page_config(page_title="Depression Prediction System", layout="wide", page_icon="./logo.png")
st.title("Depression Prediction System")

st.markdown("""# Mental Health Prediction Using Machine Learning

This project aims to predict mental health conditions using comprehensive survey data and a variety of machine learning models. The purpose is to assist in early identification of mental health concerns, supporting timely and effective intervention strategies.

## Data Preprocessing & Deep Cleaning

We performed an extensive and thoughtful data cleaning and transformation pipeline to prepare the dataset for modeling:
- **Drew Inferences**: Performed EDA to Draw Inferences in order to identify relevant columns
- **Missing Value Handling**: Utilized **KNN Imputer** to fill in missing values in categorical columns and filled missing values in numeric columns with either 0 or by the mean of the column.
- **Encoding**: Applied **one-hot encoding** to multi-class categorical features such as `Gender`, `Sleep Duration` and  `Dietary Habits`.
- **Age Binning**: Created age categories (`Teenagers`, `Young Adults`, `Adults`, `Middle-aged and Older`) while preserving the original age column.
- **Data Type Consistency**: Ensured all columns had consistent and model-friendly data types.
- **Feature Scaling**: Standardized numerical features using **StandardScaler** to normalize data.

These cleaning steps significantly enhanced data quality, improved model training, and reduced noise.

## Models Evaluated

We tested a diverse range of machine learning models to find the most accurate and generalizable solution:

- **Logistic Regression**
- **RandomForestClassifier**
- **ExtraTreesClassifier**
- **BaggingClassifier**
- **GradientBoostingClassifier**
- **LinearSVC**
- **SVC**
- **DecisionTreeClassifier**
- **XGBClassifier**
- **GaussianNB**
- **AdaBoostClassifier**
- **LGBMClassifier**
- **CatBoostClassifier**
- **Voting Classifier** (ensemble of: Logistic Regression, CatBoost, LightGBM, XGBoost)

## Hyperparameter Tuning

The four models in Voting Classifier were fine-tuned using **Optuna** with **5-fold cross-validation**, optimizing primarily for **accuracy**. This allowed for efficient and automated discovery of the best hyperparameters for each model.

## Best Model

After evaluating all models, **CatBoostClassifier** consistently outperformed others and achieved an **Accuracy** of **94%**

## Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
---

By combining thoughtful preprocessing with extensive model experimentation and tuning, this project demonstrates how machine learning can be effectively used to predict mental health challenges with high accuracy.
""")

with st.form("depression_form"):
    st.subheader("Survey Input")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        profession = st.selectbox("Are you a Student or a Working Professional", ["Student", "Working Professional"])

    col3, col4 = st.columns(2)
    with col3:
        age = st.number_input("Age", min_value=10, max_value=100, step=1)
    with col4:
        cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, step=0.1)

    col5, col6 = st.columns(2)
    with col5:
        sleep_duration = st.slider("Average Sleep per Day (hours)", min_value=0, max_value=24)
    with col6:
        work_study_hours = st.slider("Daily Study/Work Hours", min_value=0, max_value=24)

    col7, col8 = st.columns(2)
    with col7:
        financial_stress = st.slider("Financial Stress Level (1-5)", min_value=1, max_value=5)
    with col8:
        pressure = st.slider("Work/Study Pressure Level (0-5)", min_value=0, max_value=5)

    satisfaction = st.slider("Work/Study Satisfaction Level (0-5)", min_value=0, max_value=5)

    suicidal_thoughts = st.radio("Have you ever had suicidal thoughts?", ["Yes", "No"])
    family_history = st.radio("Do you have a family history of mental illness?", ["Yes", "No"])
    college_level = st.radio("Are you currently in college or studied till college only?", ["Yes", "No"])
    dietary_habit = st.selectbox("Dietary Habit", ["Healthy", "Moderate", "Unhealthy"])

    submitted = st.form_submit_button("Predict")

if submitted:
    male = 1 if gender == "Male" else 0
    female = 1 - male
    student = 1 if profession == "Student" else 0
    working_professional = 1 - student
    suicidal_thoughts = 1 if suicidal_thoughts == "Yes" else 0
    family_history = 1 if family_history == "Yes" else 0
    college_level = 1 if college_level == "Yes" else 0

    healthy = 1 if dietary_habit == "Healthy" else 0
    moderate = 1 if dietary_habit == "Moderate" else 0
    unhealthy = 1 if dietary_habit == "Unhealthy" else 0

    sleep_duration_less_than_five = int(sleep_duration < 5)
    sleep_duration_between_five_and_eight = int(5 <= sleep_duration <= 8)
    sleep_duration_more_than_eight = int(sleep_duration > 8)

    teenager = int(age <= 20)
    young_adult = int(21 <= age <= 29)
    adult = int(30 <= age <= 39)
    old = int(age >= 40)

    X = np.array([[
        age, cgpa, suicidal_thoughts, work_study_hours, financial_stress,
        family_history, pressure, satisfaction, college_level,
        male, female, student, working_professional,
        sleep_duration_less_than_five, sleep_duration_between_five_and_eight, sleep_duration_more_than_eight,
        healthy, moderate, unhealthy,
        teenager, young_adult, adult, old
    ]])

    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][prediction]
    confidence = round(proba * 100, 2)

    result = "😔 Depressed" if prediction == 1 else "😊 Not Depressed"

    st.success(f"**Prediction:** You are likely **{result}**")

    st.write("**Confidence Level:**")
    st.progress(int(confidence))
    st.write(f"{confidence}%")
