import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Diagnostic Analytics", layout="wide")

# Title
st.title("🔍Diagnostic Analytics")

# Load dataset
df = pd.read_csv('depression_dataset_processed.csv')

# Exclude unnecessary features (Patient ID, Duration of Symptoms, Ethnicity)
dfh = df.drop(columns=['Patient ID', 'Duration of Symptoms (months)', 
                      'Ethnicity_African', 'Ethnicity_Hispanic', 
                      'Ethnicity_Asian', 'Ethnicity_Caucasian'])

# Top Left Area: Heatmap of correlations between all features
st.subheader("Feature Correlation Heatmap")

# Split features into strong and weak correlation categories
strong_corr_features = [
    'Age', 'Gender', 'Family History of OCD', 'Education Level_High School', 
    'Education Level_Some College', 'Education Level_College Degree', 'Education Level_Graduate Degree', 
    'Marital Status_Single', 'Marital Status_Divorced', 'Marital Status_Married', 
    'Previous Diagnoses_MDD', 'Previous Diagnoses_None', 'Previous Diagnoses_PTSD', 
    'Previous Diagnoses_GAD', 'Previous Diagnoses_Panic Disorder'
]

weak_corr_features = [
    'Y-BOCS Score (Obsessions)', 'Y-BOCS Score (Compulsions)', 'Anxiety Diagnosis', 
    'Obsession Type_Harm-related', 'Obsession Type_Contamination', 'Obsession Type_Symmetry', 
    'Obsession Type_Hoarding', 'Obsession Type_Religious', 
    'Compulsion_Type_Checking', 'Compulsion_Type_Washing', 'Compulsion_Type_Ordering', 
    'Compulsion_Type_Praying', 'Compulsion_Type_Counting'
]

# Radio button to select the correlation strength
corr_strength = st.radio("Choose correlation strength to display:", ('Strong Correlation', 'Weak Correlation'))

# Filter the correlation matrix based on the selected features
correlation_matrix = df.corr()

if corr_strength == 'Strong Correlation':
    selected_features = strong_corr_features + ['Depression Diagnosis']
else:
    selected_features = weak_corr_features + ['Depression Diagnosis']

# Display the heatmap
fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(correlation_matrix.loc[selected_features, selected_features], annot=False, cmap='coolwarm', ax=ax, center=0)
st.pyplot(fig)

# Top Right Area: Correlation with Depression Diagnosis
st.subheader("Correlation with Depression Diagnosis")

# Dropdown to select a specific feature
feature_choice = st.selectbox(
    "Select a specific feature to see the correlation with depression diagnosis:",
    selected_features[:-1]  # Exclude 'Depression Diagnosis' from the dropdown
)

# Get the correlation value between the selected feature and 'Depression Diagnosis'
correlation_value = correlation_matrix.loc['Depression Diagnosis', feature_choice]

# Display the correlation value with explanation
st.markdown(f"### Correlation with Depression Diagnosis: {correlation_value:.2f}")

# Interpretation based on the strength of correlation
if correlation_value > 0.5:
    st.write("This is a **strong positive** correlation with depression diagnosis.")
elif 0.3 < correlation_value <= 0.5:
    st.write("This is a **moderate positive** correlation with depression diagnosis.")
elif 0 < correlation_value <= 0.3:
    st.write("This is a **weak positive** correlation with depression diagnosis.")
elif -0.3 < correlation_value < 0:
    st.write("This is a **weak negative** correlation with depression diagnosis.")
elif -0.5 <= correlation_value <= -0.3:
    st.write("This is a **moderate negative** correlation with depression diagnosis.")
else:
    st.write("This is a **strong negative** correlation with depression diagnosis.")

# Split the lower part into two side-by-side columns
left_col, right_col = st.columns(2)

# Left Bottom Area: Associations between Compulsion & Obsession Types and Depression
with left_col:
    st.subheader("Associations between Depression with Compulsion & Obsession Types")
    
    # Radio button to choose between Compulsion and Obsession types
    assoc_choice = st.radio(
        "Select an association to display:",
        ['Compulsion Types', 'Obsession Types']
    )

    # Plot correlations based on the selection
    if assoc_choice == 'Compulsion Types':
        # Calculate the correlation between compulsion types and Depression Diagnosis
        compulsion_correlations = df[['Compulsion_Type_Checking', 'Compulsion_Type_Washing', 'Compulsion_Type_Ordering',
                                    'Compulsion_Type_Praying', 'Compulsion_Type_Counting', 'Depression Diagnosis']].corr()['Depression Diagnosis'][:-1]

        # Plot the correlations for compulsion types
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(compulsion_correlations)), compulsion_correlations)
        ax.set_xticks(range(len(compulsion_correlations)))
        ax.set_xticklabels(['Checking', 'Washing', 'Ordering', 'Praying', 'Counting'], rotation=45)
        ax.set_xlabel('Compulsion Type')
        ax.set_ylabel('Correlation with Depression Diagnosis')
        ax.set_title('Correlation between Compulsion Types and Depression Diagnosis')
        st.pyplot(fig)

    elif assoc_choice == 'Obsession Types':
        # Calculate the correlation between obsession types and Depression Diagnosis
        obsession_correlations = df[['Obsession Type_Harm-related', 'Obsession Type_Contamination', 'Obsession Type_Symmetry',
                                    'Obsession Type_Hoarding', 'Obsession Type_Religious', 'Depression Diagnosis']].corr()['Depression Diagnosis'][:-1]

        # Plot the correlations for obsession types
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(obsession_correlations)), obsession_correlations)
        ax.set_xticks(range(len(obsession_correlations)))
        ax.set_xticklabels(['Harm-related', 'Contamination', 'Symmetry', 'Hoarding', 'Religious'], rotation=45)
        ax.set_xlabel('Obsession Type')
        ax.set_ylabel('Correlation with Depression Diagnosis')
        ax.set_title('Correlation between Obsession Types and Depression Diagnosis')
        st.pyplot(fig)


# Right Bottom Area: Relationship between Demographics and Depression Likelihood
with right_col:
    st.subheader("Relationship between Patients' Demographic Factors and the Likelihood of Depression")

    # Dropdown to select demographic factor
    demographic_choice = st.selectbox(
        "Select a demographic factor to analyze:",
        ['Age', 'Gender', 'Ethnicity']
    )

    # Display relationship based on the selected demographic factor
    if demographic_choice == 'Age':
        # Calculate the percentage of patients with and without depression in each age group
        age_bins = [0, 18, 30, 45, 60, 100]
        age_labels = ['0-18', '19-30', '31-45', '46-60', '60+']
        df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
        age_depression = df.groupby('age_group')['Depression Diagnosis'].mean() * 100
        age_no_depression = 100 - age_depression  # Patients without depression

        # Define positions for side-by-side bars
        bar_width = 0.35
        age_groups = np.arange(len(age_depression))  # positions of age groups
        
        # Plot side-by-side bars for depression and no-depression
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(age_groups - bar_width / 2, age_depression, bar_width, label='Depression Diagnosis')
        ax.bar(age_groups + bar_width / 2, age_no_depression, bar_width, label='No Depression Diagnosis')

        ax.set_xticks(age_groups)
        ax.set_xticklabels(age_depression.index, rotation=45)
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Percentage')
        ax.set_title('Depression Diagnosis by Age Group')
        ax.legend()
        st.pyplot(fig)

    elif demographic_choice == 'Gender':
        # Calculate the percentage of patients with and without depression by gender
        gender_depression = df.groupby('Gender')['Depression Diagnosis'].mean() * 100
        gender_no_depression = 100 - gender_depression
        
        # Map 0 to 'Male' and 1 to 'Female' for better readability
        gender_labels = {0: 'Male', 1: 'Female'}
        gender_depression.index = gender_depression.index.map(gender_labels)
        gender_no_depression.index = gender_no_depression.index.map(gender_labels)

        # Define positions for side-by-side bars
        bar_width = 0.35
        gender_groups = np.arange(len(gender_depression))  # positions of gender groups

        # Plot side-by-side bars for depression and no-depression
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(gender_groups - bar_width / 2, gender_depression, bar_width, label='Depression Diagnosis')
        ax.bar(gender_groups + bar_width / 2, gender_no_depression, bar_width, label='No Depression Diagnosis')

        ax.set_xticks(gender_groups)
        ax.set_xticklabels(gender_depression.index, rotation=45)
        ax.set_xlabel('Gender')
        ax.set_ylabel('Percentage')
        ax.set_title('Depression Diagnosis by Gender')
        ax.legend()
        st.pyplot(fig)

    elif demographic_choice == 'Ethnicity':
        # Calculate the number of patients with and without depression by ethnicity
        ethnicity_columns = ['Ethnicity_African', 'Ethnicity_Hispanic', 'Ethnicity_Asian', 'Ethnicity_Caucasian']
        
        # Calculate percentage with depression diagnosis for each ethnicity
        ethnicity_depression = df[ethnicity_columns].mul(df['Depression Diagnosis'], axis=0).mean() * 100
        
        # Calculate percentage without depression diagnosis for each ethnicity
        ethnicity_no_depression = df[ethnicity_columns].mul(1 - df['Depression Diagnosis'], axis=0).mean() * 100
        
        # Rename the columns for better readability
        ethnicity_depression.index = ['African', 'Hispanic', 'Asian', 'Caucasian']
        ethnicity_no_depression.index = ethnicity_depression.index

        # Define positions for side-by-side bars
        bar_width = 0.35
        ethnicity_groups = np.arange(len(ethnicity_depression))  # positions of ethnicity groups

        # Plot side-by-side bars for depression and no-depression
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(ethnicity_groups - bar_width / 2, ethnicity_depression, bar_width, label='Depression Diagnosis')
        ax.bar(ethnicity_groups + bar_width / 2, ethnicity_no_depression, bar_width, label='No Depression Diagnosis')

        ax.set_xticks(ethnicity_groups)
        ax.set_xticklabels(ethnicity_depression.index, rotation=45)
        ax.set_xlabel('Ethnicity')
        ax.set_ylabel('Percentage')
        ax.set_title('Depression Diagnosis by Ethnicity')
        ax.legend()
        st.pyplot(fig)
