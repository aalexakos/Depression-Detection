import streamlit as st
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Page configuration
st.set_page_config(page_title="Predictive Analytics", layout="wide")

# Sidebar configuration
st.sidebar.header("Depression Detection")
st.sidebar.image("./assets/sidebar.png",)

# Custom CSS to make the sidebar prettier
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f9f6f1;
    }
    .sidebar .sidebar-content h2 {
        color: #4b72b8;
    }
    .css-17eq0hr { 
        background-color: #f9f6f1; 
    }
    </style>
    """, unsafe_allow_html=True)

# Custom CSS to reduce space between elements
st.markdown("""
    <style>
    .css-1aumxhk {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    .stTextInput, .stSelectbox, .stNumberInput, .stSlider {
        margin-bottom: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🔮Predictive Tab - Depression Prediction")

# Load the pre-trained model from the assets folder
file_path = "./assets/best_model.pickle"
with open(file_path, "rb") as file:
    model = pickle.load(file)

# Function to take user inputs for prediction, organized into sections
def user_input_features():
    st.header("Demographic & Lifestyle Data")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Age', min_value=18, max_value=100, value=30)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        marital_status = st.selectbox('Marital Status', ['Single', 'Divorced', 'Married'])
    with col2:
        education_level = st.selectbox('Education Level', ['Some College', 'College Degree', 'High School', 'Graduate Degree'])
        duration_symptoms = st.slider('Duration of Symptoms (months)', 0, 60, 12)
        family_history = st.selectbox('Family History of OCD', ['Yes', 'No'])

    st.header("Medical Conditions")

    col3, col4 = st.columns(2)
    with col3:
        ybocs_obsession = st.slider('Y-BOCS Score (Obsessions)', 0, 20, 10)
        ybocs_compulsion = st.slider('Y-BOCS Score (Compulsions)', 0, 20, 10)
        anxiety_diag = st.selectbox('Anxiety Diagnosis', ['Yes', 'No'])
        previous_diagnosis = st.selectbox('Previous Diagnosis', ['MDD', 'None', 'PTSD', 'GAD', 'Panic Disorder'])
    with col4:
        compulsion_type = st.selectbox(
            'Compulsion Type', ['Checking', 'Washing', 'Ordering', 'Praying', 'Counting'])
        obsession_type = st.selectbox(
            'Obsession Type', ['Harm-related', 'Contamination', 'Symmetry', 'Hoarding', 'Religious'])
        medication = st.selectbox(
            'Medications', ['SNRI', 'SSRI', 'Benzodiazepine', 'None'])

    # Convert categorical variables to numerical representations
    data = {
        'Age': age,
        'Gender': 0 if gender == 'Male' else 1,
        'Duration of Symptoms (months)': duration_symptoms,
        'Family History of OCD': 1 if family_history == 'Yes' else 0,
        'Y-BOCS Score (Obsessions)': ybocs_obsession,
        'Y-BOCS Score (Compulsions)': ybocs_compulsion,
        'Anxiety Diagnosis': 1 if anxiety_diag == 'Yes' else 0,

        'Compulsion_Type_Checking': 1 if compulsion_type == 'Checking' else 0,
        'Compulsion_Type_Washing': 1 if compulsion_type == 'Washing' else 0,
        'Compulsion_Type_Ordering': 1 if compulsion_type == 'Ordering' else 0,
        'Compulsion_Type_Praying': 1 if compulsion_type == 'Praying' else 0,
        'Compulsion_Type_Counting': 1 if compulsion_type == 'Counting' else 0,

        'Obsession Type_Harm-related': 1 if obsession_type == 'Harm-related' else 0,
        'Obsession Type_Contamination': 1 if obsession_type == 'Contamination' else 0,
        'Obsession Type_Symmetry': 1 if obsession_type == 'Symmetry' else 0,
        'Obsession Type_Hoarding': 1 if obsession_type == 'Hoarding' else 0,
        'Obsession Type_Religious': 1 if obsession_type == 'Religious' else 0,

        'Medications_SNRI': 1 if medication == 'SNRI' else 0,
        'Medications_SSRI': 1 if medication == 'SSRI' else 0,
        'Medications_Benzodiazepine': 1 if medication == 'Benzodiazepine' else 0,
        'Medications_None': 1 if medication == 'None' else 0,

        'Marital Status_Single': 1 if marital_status == 'Single' else 0,
        'Marital Status_Divorced': 1 if marital_status == 'Divorced' else 0,
        'Marital Status_Married': 1 if marital_status == 'Married' else 0,

        'Education Level_Some College': 1 if education_level == 'Some College' else 0,
        'Education Level_College Degree': 1 if education_level == 'College Degree' else 0,
        'Education Level_High School': 1 if education_level == 'High School' else 0,
        'Education Level_Graduate Degree': 1 if education_level == 'Graduate Degree' else 0,

        'Previous Diagnoses_MDD': 1 if previous_diagnosis == 'MDD' else 0,
        'Previous Diagnoses_None': 1 if previous_diagnosis == 'None' else 0,
        'Previous Diagnoses_PTSD': 1 if previous_diagnosis == 'PTSD' else 0,
        'Previous Diagnoses_GAD': 1 if previous_diagnosis == 'GAD' else 0,
        'Previous Diagnoses_Panic Disorder': 1 if previous_diagnosis == 'Panic Disorder' else 0,
    }

    # Convert the dictionary to a DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Store user input features
input_df = user_input_features()

# Select appropriate SHAP explainer based on the model type
if isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
    explainer = shap.TreeExplainer(model)
elif isinstance(model, (LogisticRegression, GaussianNB)):
    # Use KernelExplainer for models not supported by TreeExplainer
    if not input_df.empty:
        background_data = input_df.sample(min(100, len(input_df)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background_data)
else:
    st.error("Unsupported model type for SHAP analysis.")
    st.stop()

# Button to trigger prediction
if st.button('Predict') and not input_df.empty:
    # Reorder the input features based on the order used in model training
    input_df = input_df[model.feature_names_in_]

    # Show input dataframe to the user
    st.subheader("User Input Features")
    st.write(input_df)

    # Make predictions using the pre-trained model
    prediction = model.predict(input_df)

    # Store prediction result in session state
    st.session_state.prediction_result = prediction[0]

    # Display the prediction result
    st.subheader("Prediction Result")
    if st.session_state.prediction_result == 1:
        st.markdown("Depression")
    else:
        st.markdown("No Depression")

    # SHAP Explanation Section with Waterfall Plot
    with st.expander("Show SHAP Explanation", expanded=False):
        st.subheader("SHAP Waterfall Plot")
        
        # Compute SHAP values for the input
        shap_values = explainer.shap_values(input_df)
        
        # Make sure to define predicted_class based on the prediction
        predicted_class = st.session_state.prediction_result  # 0 for "No Depression", 1 for "Depression"
        
        # Check if the SHAP values are returned as a list (for multi-class outputs) or a single array
        if isinstance(shap_values, list):
            # Multiple outputs (e.g., for binary classification)
            shap_values_for_predicted_class = shap_values[predicted_class]  # Select SHAP values for the predicted class
        else:
            # Single output, use the shap_values directly
            shap_values_for_predicted_class = shap_values
        
        # Select the SHAP values for the first instance and the predicted class
        shap_values_for_instance = shap_values_for_predicted_class[0]  # SHAP values for the first prediction instance
        
        # If there are multiple outputs, select the SHAP values for the specific output (predicted class)
        if shap_values_for_instance.ndim > 1:
            shap_values_for_instance = shap_values_for_instance[:, predicted_class]
        
        # Create an Explanation object for the SHAP values (for the first instance)
        shap_explanation = shap.Explanation(
            values=shap_values_for_instance,  # SHAP values for the first instance
            base_values=explainer.expected_value[predicted_class],  # Expected value for the predicted class
            data=input_df.values[0],  # Input features for the first instance
            feature_names=input_df.columns  # Feature names
        )

        # Generate SHAP waterfall plot for a single prediction, showing all features
        fig, ax = plt.subplots()
        shap.waterfall_plot(shap_explanation, max_display=10, )  # Display all 37 features
        st.pyplot(fig)

        # Add explanatory text
        st.markdown("""
        **Interpreting the SHAP Waterfall Plot**
        **Features:** The features are listed along the vertical axis.
        **SHAP Values:** The horizontal bars represent the SHAP values, indicating the impact of each feature on the prediction.
        """)
    
    # Detailed insights based on SHAP values
    with st.expander("Detailed Insights from SHAP Values", expanded=False):
        top_features = pd.Series(shap_explanation.values, index=shap_explanation.feature_names).sort_values(ascending=False)       # Displaying top features with explanations
        for feature, value in top_features.items():
            impact = "positive" if value > 0 else "negative"
            st.write(f"- **{feature}:** {value:.3f} (This feature has a {impact} impact on the prediction.)")
