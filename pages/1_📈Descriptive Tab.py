import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Descriptive Analytics", layout="wide")

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

# Title
st.title("📈Descriptive Analytics")

# Load dataset
df = pd.read_csv('depression_dataset_processed.csv')

# Define a function to compute the required metrics based on the selection
def compute_metrics(selection):
    if selection == 'None':
        total_patients = df.shape[0]
        return total_patients, None, None  # Returning None for pie chart data
    elif selection == 'Depression Diagnosis':
        total_with = df['Depression Diagnosis'].sum()
        total_without = df.shape[0] - total_with
        return total_with, total_with, total_without
    elif selection == 'Anxiety Diagnosis':
        total_with = df['Anxiety Diagnosis'].sum()
        total_without = df.shape[0] - total_with
        return total_with, total_with, total_without
    elif selection == 'Depression & Anxiety Diagnosis':
        total_with = df[(df['Depression Diagnosis'] == 1) & (df['Anxiety Diagnosis'] == 1)].shape[0]
        total_without = df.shape[0] - total_with
        return total_with, total_with, total_without

# Top layout with two columns (number display and pie chart)
col1, col2 = st.columns(2)

# Left Top Area: Number Display
with col1:
    st.subheader("Total Amount of Patients with")

    # Dropdown for selecting diagnosis type
    diagnosis_type = st.selectbox(
        'Select diagnosis type',
        ['None', 'Depression Diagnosis', 'Anxiety Diagnosis', 'Depression & Anxiety Diagnosis']
    )

    # Compute metrics based on selection
    total_patients, total_with, total_without = compute_metrics(diagnosis_type)

    # Display the total patients with the selected diagnosis
    if diagnosis_type == 'None':
        st.markdown(f"<h1 style='text-align: center;'>{total_patients}</h1>", unsafe_allow_html=True)
        st.write("Displaying the total number of patients in the dataset.")
    else:
        st.markdown(f"<h1 style='text-align: center;'>{total_with}</h1>", unsafe_allow_html=True)
        st.write(f"Total number of patients with {diagnosis_type.lower()}.")

# Right Top Area: Pie Chart Visualization
with col2:
    if total_with is not None and total_without is not None:
        # Prepare the pie chart data
        labels = ['With Diagnosis', 'Without Diagnosis']
        sizes = [total_with, total_without]
        colors = ['#001f3f', '#99ccff']  # Navy blue shades

        # Create the pie chart
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'color': "black"})
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Display the pie chart in the app
        st.pyplot(fig)
    else:
        st.write("No diagnosis selected, nothing to display here.")

# Helper function to calculate distribution for one-hot encoded columns
def calculate_distribution(column_prefix, df):
    # Get columns that start with the given prefix
    columns = [col for col in df.columns if col.startswith(column_prefix)]
    
    # Sum the values for each column to get the distribution
    distribution = df[columns].sum().sort_values(ascending=False)
    
    # Rename the index to strip the prefix for cleaner visualization
    distribution.index = [col.replace(column_prefix, '').strip('_') for col in distribution.index]
    
    return distribution

# Left Bottom Area: Distribution Bar Chart based on selected feature
st.markdown("---")  # separator line

st.subheader("Feature Distribution Analysis")

# Create two columns for the bottom section
col3, col4 = st.columns(2)

# Left Bottom Area (col3)
with col3:
    st.subheader("Distribution of")
    
    # Dropdown menu for selecting the feature
    feature_choice = st.selectbox(
        'Select a feature for distribution analysis',
        ['Age', 'Gender', 'Marital Status', 'Education Level']
    )
    
    # Radio buttons for patient filter (all patients or only those with depression)
    patient_filter = st.radio(
        "Show distribution for",
        ['In All Patients', 'Patients with Depression Diagnosis']
    )
    
    # Filter the data based on the radio button selection
    if patient_filter == 'Patients with Depression Diagnosis':
        data_to_plot = df[df['Depression Diagnosis'] == 1]
    else:
        data_to_plot = df
    
    # Plot distribution based on selected feature
    if feature_choice == 'Age':
        age_bins = [0, 18, 30, 45, 60, 100]
        age_labels = ['0-18', '19-30', '31-45', '46-60', '60+']
        data_to_plot['age_group'] = pd.cut(data_to_plot['Age'], bins=age_bins, labels=age_labels)
        chart_data = data_to_plot['age_group'].value_counts().sort_index()

        # Find the age group with the maximum count
        max_group = chart_data.idxmax()
        
        # Create a bar chart for age distribution
        fig, ax = plt.subplots()
        colors = ['#99ccff' if label == max_group else '#001f3f' for label in chart_data.index]  # Highlight max count
        ax.bar(chart_data.index, chart_data.values, color=colors)
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Age', color='#001f3f')
        st.pyplot(fig)

    elif feature_choice == 'Gender':
        # Assuming Gender in the dataset is encoded (e.g., 0 for Female, 1 for Male), map them to labels
        gender_mapping = {0: 'Male', 1: 'Female'}
        chart_data = data_to_plot['Gender'].map(gender_mapping).value_counts()

        # Find the gender with the maximum count
        max_gender = chart_data.idxmax()

        # Create a bar chart for gender distribution
        fig, ax = plt.subplots()
        colors = ['#99ccff' if label == max_gender else '#001f3f' for label in chart_data.index]  # Highlight max count
        ax.bar(chart_data.index, chart_data.values, color=colors)
        ax.set_xlabel('Gender')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Gender', color='#001f3f')

        st.pyplot(fig)

    elif feature_choice == 'Marital Status':
        chart_data = calculate_distribution('Marital Status_', data_to_plot)

        # Find the marital status with the maximum count
        max_status = chart_data.idxmax()

        # Create a bar chart for marital status distribution
        fig, ax = plt.subplots()
        colors = ['#99ccff' if label == max_status else '#001f3f' for label in chart_data.index]  # Highlight max count
        ax.bar(chart_data.index, chart_data.values, color=colors)
        ax.set_xlabel('Marital Status')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Marital Status', color='#001f3f')
        st.pyplot(fig)

    elif feature_choice == 'Education Level':
        chart_data = calculate_distribution('Education Level_', data_to_plot)

        # Find the education level with the maximum count
        max_education = chart_data.idxmax()

        # Create a bar chart for education level distribution
        fig, ax = plt.subplots()
        colors = ['#99ccff' if label == max_education else '#001f3f' for label in chart_data.index]  # Highlight max count
        ax.bar(chart_data.index, chart_data.values, color=colors)
        ax.set_xlabel('Education Level')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Education Level', color='#001f3f')
        st.pyplot(fig)

# Right Bottom Area: Distribution of Medications and Previous Diagnoses
with col4:
    st.subheader("Distribution of Medications and Previous Diagnoses")
    
    # Dropdown menu for selecting either medications or previous diagnoses
    med_diag_choice = st.selectbox(
        'Select a category',
        ['Medications', 'Previous Diagnosis']
    )
    
    # Radio buttons for patient filter (all patients or only those with depression)
    med_patient_filter = st.radio(
        "Show distribution for",
        ['In All Patients', 'Patients with Depression Diagnosis'],
        key='med_radio'  # Separate key to avoid interference with previous radio button
    )
    
    # Filter the data based on the radio button selection
    if med_patient_filter == 'Patients with Depression Diagnosis':
        med_data_to_plot = df[df['Depression Diagnosis'] == 1]
    else:
        med_data_to_plot = df
    
    # Plot distribution based on selected category (medications or previous diagnosis)
    if med_diag_choice == 'Medications':
        med_chart_data = {
            "SNRI": med_data_to_plot['Medications_SNRI'].sum(),
            "SSRI": med_data_to_plot['Medications_SSRI'].sum(),
            "Benzodiazepine": med_data_to_plot['Medications_Benzodiazepine'].sum(),
            "None": med_data_to_plot['Medications_None'].sum()
        }
        med_chart_df = pd.DataFrame(list(med_chart_data.items()), columns=['Medication', 'Count'])

        # Find the medication with the maximum count
        max_med = med_chart_df.loc[med_chart_df['Count'].idxmax(), 'Medication']

        # Create a bar chart using matplotlib for medications
        fig, ax = plt.subplots()
        colors = ['#99ccff' if label == max_med else '#001f3f' for label in med_chart_df['Medication']]  # Highlight max count
        ax.bar(med_chart_df['Medication'], med_chart_df['Count'], color=colors)
        ax.set_xlabel('Medication')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Medications', color='#001f3f')

        # Display the bar chart
        st.pyplot(fig)
    
    elif med_diag_choice == 'Previous Diagnosis':
        diag_chart_data = calculate_distribution('Previous Diagnoses_', med_data_to_plot)

        # Find the diagnosis with the maximum count
        max_diag = diag_chart_data.idxmax()

        # Create a bar chart using matplotlib for previous diagnoses
        fig, ax = plt.subplots()
        colors = ['#99ccff' if label == max_diag else '#001f3f' for label in diag_chart_data.index]  # Highlight max count
        ax.bar(diag_chart_data.index, diag_chart_data.values, color=colors)
        ax.set_xlabel('Previous Diagnoses')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Previous Diagnoses', color='#001f3f')

        # Display the bar chart
        st.pyplot(fig)

st.write("ℹ️ Explanation for the Feature Distribution Analysis:")
st.write("""In the bar charts, one of the bars is highlighted in light blue, indicating the bar with the highest count in each chart. This visual cue helps to quickly identify key insights in each chart.""")

# Creating an expander for extra information and abbreviations
with st.expander("ℹ️ More Information & Abbreviations"):
    st.write("""
    **Education Level:**
    - Graduate Degree: Completed an advanced degree (e.g., Master's, PhD)
    - College Degree: Completed a 4-year undergraduate program (e.g., Bachelor's)
    - Some College: Attended college but did not complete a degree
    - High School: Completed high school education
    """)
    st.write("""
    **Medications:** 
    - SNRI (Serotonin-Norepinephrine Reuptake Inhibitors)
    - SSRI (Selective Serotonin Reuptake Inhibitors)
    - Benzodiazepine (anti-anxiety medication)
    """)
    st.write("""
    **Previous Diagnosis:** 
    - MDD (Major Depressive Disorder)
    - GAD (Generalized Anxiety Disorder)
    - PTSD (Post-Traumatic Stress Disorder)
    """)
