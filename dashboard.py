import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle

# Page configuration
st.set_page_config(
    page_title="Depression Detection Dashboard",
    page_icon="🧠",
)

# Sidebar configuration
st.sidebar.image("./assets/sidebar.png",)

# Custom CSS to make the sidebar prettier
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content h2 {
        color: #4b72b8;
    }
    .css-17eq0hr { 
        background-color: #f0f2f6; 
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📊 Navigation Menu")
tabs = ["🏠 Home", "📈 Descriptive Analytics", "🔍 Diagnostic Analytics", "🔮 Predictive Analytics", "ℹ️ About"]
selection = st.sidebar.radio("Choose a page", tabs)

# Dummy Data
def load_data():
    return pd.DataFrame({
        'Age': np.random.randint(20, 80, 100),
        'Blood Pressure': np.random.randint(100, 180, 100),
        'Cholesterol': np.random.randint(150, 250, 100),
        'Outcome': np.random.choice([0, 1], 100)
    })

data = load_data()

# Home Section
if selection == "🏠 Home":
    st.title("Welcome to the OCD Detection Dashboard")
    st.write("Explore the medical data regarding OCD using the tabs on the left to navigate through different types of analysis.")
    st.image("assets/home.png", caption="The cycle of OCD", use_column_width=True)

# Descriptive Analytics
if selection == "📈 Descriptive Analytics":
    st.title("Descriptive Analytics")
    st.write("This section shows some descriptive statistics and plots of the dataset.")

    st.write("### Dataset Overview")
    st.dataframe(data)

    st.write("### Summary Statistics")
    st.write(data.describe())

    st.write("### Distribution of Age")
    fig, ax = plt.subplots()
    sns.histplot(data['Age'], kde=True, ax=ax)
    st.pyplot(fig)

    st.write("### Blood Pressure vs. Cholesterol")
    fig, ax = plt.subplots()
    sns.scatterplot(x=data['Blood Pressure'], y=data['Cholesterol'], hue=data['Outcome'], ax=ax)
    st.pyplot(fig)

# Diagnostic Analytics
if selection == "🔍 Diagnostic Analytics":
    st.title("Diagnostic Analytics")
    st.write("This section shows diagnostic relationships in the data.")

    st.write("### Correlation Matrix")
    corr_matrix = data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.write("### Cholesterol vs Blood Pressure by Outcome")
    fig, ax = plt.subplots()
    sns.lmplot(x="Cholesterol", y="Blood Pressure", hue="Outcome", data=data, height=6)
    st.pyplot(fig)

# Predictive Analytics
if selection == "🔮 Predictive Analytics":
    st.title("Predictive Analytics")
    st.write("This section allows you to input data and get predictions from the pre-trained model.")
    
    # Load pre-trained model (replace with your own model)
    # model = pickle.load(open('model.pkl', 'rb'))  # Replace with your actual model
    
    st.write("### Input new data:")
    age = st.slider("Age", 20, 80, 50)
    bp = st.slider("Blood Pressure", 100, 180, 120)
    chol = st.slider("Cholesterol", 150, 250, 200)
    
    user_input = pd.DataFrame({
        'Age': [age],
        'Blood Pressure': [bp],
        'Cholesterol': [chol]
    })
    
    st.write("### User Input")
    st.dataframe(user_input)

    # Dummy prediction (replace with actual model prediction)
    # prediction = model.predict(user_input)
    prediction = np.random.choice([0, 1])
    
    st.write("### Prediction Outcome")
    if prediction == 0:
        st.success("Low risk of condition")
    else:
        st.error("High risk of condition")

    # SHAP Explanation (optional)
    if st.checkbox("Show SHAP Explanation"):
        st.write("### SHAP Visualization (dummy)")
        fig, ax = plt.subplots()
        shap_values = np.random.randn(100, 3)  # Dummy SHAP values
        shap.summary_plot(shap_values, data)
        st.pyplot(fig)

# About Section
if selection == "ℹ️ About":
    st.title("About")
    st.write("""The "OCD Patient Dataset: Demographics & Clinical Data" is a comprehensive collection of information pertaining to 1500 individuals diagnosed with Obsessive-Compulsive Disorder (OCD). This dataset encompasses a wide range of parameters, providing a detailed insight into the demographic and clinical profiles of these individuals.""")
    st.write("### Dataset")
    st.write("""Included in this dataset are key demographic details such as age, gender, ethnicity, marital status, and education level, offering a comprehensive overview of the sample population. Additionally, clinical information like the date of OCD diagnosis, duration of symptoms, and any previous psychiatric diagnoses are recorded, providing context to the patients' journeys.

The dataset also delves into the specific nature of OCD symptoms, categorizing them into obsession and compulsion types. Severity of these symptoms is assessed using the Yale-Brown Obsessive-Compulsive Scale (Y-BOCS) scores for both obsessions and compulsions. Furthermore, it documents any co-occurring mental health conditions, including depression and anxiety diagnoses.

Notably, the dataset outlines the medications prescribed to patients, offering valuable insights into the treatment approaches employed. It also records whether there is a family history of OCD, shedding light on potential genetic or environmental factors.

Overall, this dataset serves as a valuable resource for researchers, clinicians, and mental health professionals seeking to gain a deeper understanding of OCD and its manifestations within a diverse patient population.""")
    st.write("### Team Members")
    st.write("Alexandros Alexakos, João Calixto, Yin Shea Lai, Umiah Gohar, Katja Wilde, Kevin Arjona")
    st.write("### Contact")
    st.write("alexandros.alexakos@stud.ki.se")
