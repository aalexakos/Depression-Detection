import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle

# Page configuration
st.set_page_config(page_title="Predictive Analytics", layout="wide")

# Title
st.title("🔮Predictive Analytics")

# Load the pre-trained model
model_name = st.selectbox("Select a model:", ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Support Vector Machine', 'Gradient Boosting', 'XGBoost'])
model_file = f'{model_name}.pkl'

with open(model_file, 'rb') as f:
    model = pickle.load(f)

# User Input Section
st.subheader("Input Features")
# Assuming the input features are the same as the model's training features
# Replace these with your actual feature names
age = st.number_input("Age", min_value=0)
gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
duration_symptoms = st.number_input("Duration of Symptoms (months)", min_value=0)
# Add more inputs as necessary based on your dataset
# For example, for each of your features, add an input widget

# Create a DataFrame from user input
user_input = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Duration of Symptoms (months)': [duration_symptoms],
    # Include other features here
})

# Predict based on user input
if st.button("Predict"):
    prediction = model.predict(user_input)
    st.write(f"The predicted Depression Diagnosis is: {'Depressed' if prediction[0] == 1 else 'Not Depressed'}")