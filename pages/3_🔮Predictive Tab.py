import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(page_title="Predictive Analytics", layout="wide")

# Title
st.title("🔮Predictive Analytics")

# Load the pre-trained model from the assets folder
file_path = "./assets/trained_model.pickle"
with open(file_path, "rb") as file:
    model = pickle.load(file)

# Main section for user input
st.header("Input Features")

# Function to take user inputs for prediction
def user_input_features():
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    duration_symptoms = st.slider('Duration of Symptoms (months)', 0, 60, 12)
    family_history = st.selectbox('Family History of OCD', ['Yes', 'No'])
    ybocs_obsession = st.slider('Y-BOCS Score (Obsessions)', 0, 40, 10)
    ybocs_compulsion = st.slider('Y-BOCS Score (Compulsions)', 0, 40, 10)
    anxiety_diag = st.selectbox('Anxiety Diagnosis', ['Yes', 'No'])

    # Compulsion Type: Single dropdown for all types
    compulsion_type = st.selectbox(
        'Compulsion Type',
        ['Checking', 'Washing', 'Ordering', 'Praying', 'Counting']
    )

    # Obsession Type: Single dropdown for all types
    obsession_type = st.selectbox(
        'Obsession Type',
        ['Harm-related', 'Contamination', 'Symmetry', 'Hoarding', 'Religious']
    )

    # Medications: Single dropdown for medication types
    medication = st.selectbox(
        'Medications',
        ['SNRI', 'SSRI', 'Benzodiazepine', 'None']
    )

    # Marital Status
    marital_status = st.selectbox('Marital Status', ['Single', 'Divorced', 'Married'])

    # Education Level
    education_level = st.selectbox('Education Level', ['Some College', 'College Degree', 'High School', 'Graduate Degree'])

    # Previous Diagnoses
    previous_diagnosis = st.selectbox('Previous Diagnosis', ['MDD', 'No Previous Diagnoses', 'PTSD', 'GAD', 'Panic Disorder'])

    # Ethnicity (Add ethnicity input with default values)
    ethnicity = st.selectbox('Ethnicity', ['African', 'Asian', 'Caucasian', 'Hispanic'])

    # Convert categorical variables to numerical representations
    data = {
        'Age': age,
        'Gender': 0 if gender == 'Male' else 1,  # Encoding Gender
        'Duration of Symptoms (months)': duration_symptoms,
        'Family History of OCD': 1 if family_history == 'Yes' else 0,
        'Y-BOCS Score (Obsessions)': ybocs_obsession,
        'Y-BOCS Score (Compulsions)': ybocs_compulsion,
        'Anxiety Diagnosis': 1 if anxiety_diag == 'Yes' else 0,
        
        # Encoding the compulsion type
        'Compulsion_Type_Checking': 1 if compulsion_type == 'Checking' else 0,
        'Compulsion_Type_Washing': 1 if compulsion_type == 'Washing' else 0,
        'Compulsion_Type_Ordering': 1 if compulsion_type == 'Ordering' else 0,
        'Compulsion_Type_Praying': 1 if compulsion_type == 'Praying' else 0,
        'Compulsion_Type_Counting': 1 if compulsion_type == 'Counting' else 0,

        # Encoding the obsession type
        'Obsession Type_Harm-related': 1 if obsession_type == 'Harm-related' else 0,
        'Obsession Type_Contamination': 1 if obsession_type == 'Contamination' else 0,
        'Obsession Type_Symmetry': 1 if obsession_type == 'Symmetry' else 0,
        'Obsession Type_Hoarding': 1 if obsession_type == 'Hoarding' else 0,
        'Obsession Type_Religious': 1 if obsession_type == 'Religious' else 0,

        # Encoding the medication type
        'Medications_SNRI': 1 if medication == 'SNRI' else 0,
        'Medications_SSRI': 1 if medication == 'SSRI' else 0,
        'Medications_Benzodiazepine': 1 if medication == 'Benzodiazepine' else 0,
        'Medications_None': 1 if medication == 'None' else 0,

        # Marital Status
        'Marital Status_Single': 1 if marital_status == 'Single' else 0,
        'Marital Status_Divorced': 1 if marital_status == 'Divorced' else 0,
        'Marital Status_Married': 1 if marital_status == 'Married' else 0,

        # Education Level
        'Education Level_Some College': 1 if education_level == 'Some College' else 0,
        'Education Level_College Degree': 1 if education_level == 'College Degree' else 0,
        'Education Level_High School': 1 if education_level == 'High School' else 0,
        'Education Level_Graduate Degree': 1 if education_level == 'Graduate Degree' else 0,

        # Previous Diagnoses
        'Previous Diagnoses_MDD': 1 if previous_diagnosis == 'MDD' else 0,
        'Previous Diagnoses_No Previous Diagnoses': 1 if previous_diagnosis == 'No Previous Diagnoses' else 0,
        'Previous Diagnoses_PTSD': 1 if previous_diagnosis == 'PTSD' else 0,
        'Previous Diagnoses_GAD': 1 if previous_diagnosis == 'GAD' else 0,
        'Previous Diagnoses_Panic Disorder': 1 if previous_diagnosis == 'Panic Disorder' else 0,

        # Ethnicity encoding
        'Ethnicity_African': 1 if ethnicity == 'African' else 0,
        'Ethnicity_Asian': 1 if ethnicity == 'Asian' else 0,
        'Ethnicity_Caucasian': 1 if ethnicity == 'Caucasian' else 0,
        'Ethnicity_Hispanic': 1 if ethnicity == 'Hispanic' else 0,
    }

    # Convert the dictionary to a DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Store user input features
input_df = user_input_features()

# Reorder the input features based on the order used in model training
input_df = input_df[model.feature_names_in_]

# Show input dataframe to the user
st.subheader("User Input Features")
st.write(input_df)

# Make predictions using the pre-trained model
prediction = model.predict(input_df)

# Display the prediction result
st.subheader("Prediction Result")

# Assuming 1 represents depression diagnosis and 0 represents no depression diagnosis
if prediction[0] == 1:
    st.write("The model predicts **Depression**.")
else:
    st.write("The model predicts **No Depression**.")



# SHAP Explanation Section
st.subheader("SHAP Explanation")

# Create a SHAP explainer using the trained model
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for the input data
shap_values = explainer.shap_values(input_df)

# Initialize shap_plot as None
shap_plot = None

# Check the shape of shap_values to determine how many classes are present
if isinstance(shap_values, list):
    # If there are two arrays (for binary classification)
    class_index = prediction[0]  # Get the predicted class index
    st.write(f"Prediction class: {class_index}")

    # Check if we can plot the SHAP values
    if len(shap_values) > class_index:
        # Plot the SHAP values for this specific input (use the predicted class)
        shap.initjs()
        shap_plot = shap.force_plot(explainer.expected_value[class_index], 
                                     shap_values[class_index][0], 
                                     input_df.iloc[0, :])

# If only one class present, give a warning message
if shap_plot is None:
    st.warning("SHAP values could not be computed for both classes; only one class is present in the output.")
else:
    # Convert the SHAP force plot to an HTML component to display in Streamlit
    st.components.v1.html(shap_plot.html(), height=300)