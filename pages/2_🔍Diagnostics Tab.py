import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Diagnostic Analytics", layout="wide")

# Sidebar configuration
st.sidebar.header("Depression Detection")
st.sidebar.image("./assets/sidebar.png",)

# Custom CSS to style the sidebar and main content area
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
    .section {
        padding: 20px;
        background-color: #e6f0ff; /* Light navy for section background */
        margin-bottom: 20px; /* Space between sections */
    }
    .subheader {
        color: #003366; /* Navy color for subheaders */
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🔍 Diagnostic Analytics")

st.subheader("What is correlation?")
st.write("""Correlation measures the strength and direction of a relationship between two variables. It helps identify patterns, showing how one variable may increase or decrease in relation to another. A positive correlation means both variables move in the same direction, while a negative correlation means they move in opposite directions. Correlation values range from -1 to 1, with 0 indicating no relationship. It’s useful for understanding connections between data points, like how education level might relate to depression diagnosis.""")

# Load dataset
df = pd.read_csv('depression_dataset_processed.csv')

# Exclude unnecessary features (Patient ID, Duration of Symptoms, Ethnicity)
dfh = df.drop(columns=['Patient ID', 'Duration of Symptoms (months)', 
                      'Ethnicity_African', 'Ethnicity_Hispanic', 
                      'Ethnicity_Asian', 'Ethnicity_Caucasian'])

# Top Left Area: Bar chart of correlations between all features and Depression Diagnosis
st.subheader("Feature Correlation with Depression Diagnosis")

# Calculate correlations with Depression Diagnosis
correlation_matrix = df.corr()
correlation_with_depression = correlation_matrix['Depression Diagnosis'].drop('Depression Diagnosis')

# Define thresholds for strong and weak correlations
strong_threshold = 0.05
weak_threshold = -0.05

# Split features into strong and weak correlation categories
strong_corr_features = correlation_with_depression[(correlation_with_depression >= strong_threshold) | (correlation_with_depression <= -strong_threshold)].index.tolist()
weak_corr_features = correlation_with_depression[(correlation_with_depression < strong_threshold) & (correlation_with_depression > weak_threshold)].index.tolist()

# Radio button to select the correlation strength
corr_strength = st.radio("Choose correlation strength to display:", ('Strong Correlation', 'Weak Correlation'))

# Creating an expander for extra information
with st.expander("ℹ️ More Information"):
    st.write("""Based on a review of the literature, the dataset features have been categorized into two groups. Features with a higher correlation to depression are displayed in the bar chart corresponding to the "Strong Correlation" option, while those with weaker correlations are shown in the bar chart for the "Weak Correlation" option.""")

# Select features based on correlation strength
if corr_strength == 'Strong Correlation':
    selected_features = strong_corr_features
else:
    selected_features = weak_corr_features

# Get correlation values with Depression Diagnosis for the selected features
correlation_values = correlation_with_depression.loc[selected_features]

# Display the bar chart of correlations
fig, ax = plt.subplots(figsize=(15, 10))
correlation_values.sort_values().plot(kind='barh', ax=ax, color='skyblue')
ax.set_xlabel('Correlation Coefficient')
ax.set_title('Correlation between Selected Features and Depression Diagnosis')
st.pyplot(fig)

st.markdown("---")  # separator line

# Top Right Area: Correlation with Depression Diagnosis
st.subheader("Correlation with Depression Diagnosis")

# Dropdown to select a specific feature
feature_choice = st.selectbox(
    "Select a specific feature to see the correlation with depression diagnosis:",
    selected_features
)

# Get the correlation value between the selected feature and 'Depression Diagnosis'
correlation_value = correlation_matrix.loc['Depression Diagnosis', feature_choice]

# Display the correlation value with explanation
st.markdown(f"### Correlation between {feature_choice} and Depression Diagnosis: {correlation_value:.2f}")

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
    st.markdown("---")  # separator line
    st.subheader("Associations between Depression with Compulsion & Obsession Types")
    
    # Radio button to choose between Compulsion and Obsession types
    assoc_choice = st.radio(
        "Select an association to display:",
        ['Compulsion Types', 'Obsession Types']
    )

    if assoc_choice == 'Compulsion Types':
        st.write("The different types of compulsions can have varying correlations with depression. Below is a chart showing the strength of these correlations based on data from patients' reported compulsions.")

        # Calculate the correlation between compulsion types and Depression Diagnosis
        compulsion_correlations = df[['Compulsion_Type_Checking', 'Compulsion_Type_Washing', 'Compulsion_Type_Ordering',
                                    'Compulsion_Type_Praying', 'Compulsion_Type_Counting', 'Depression Diagnosis']].corr()['Depression Diagnosis'][:-1]

        # Plot the correlations for compulsion types
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(compulsion_correlations)), compulsion_correlations, color='#003366')  # Navy color
        ax.set_xticks(range(len(compulsion_correlations)))
        ax.set_xticklabels(['Checking', 'Washing', 'Ordering', 'Praying', 'Counting'], rotation=45)
        ax.set_xlabel('Compulsion Type')
        ax.set_ylabel('Correlation with Depression Diagnosis')
        ax.set_title('Correlation between Compulsion Types and Depression Diagnosis')
        st.pyplot(fig)

        # Explanation of the findings
        st.write("""
            Compulsions, or repetitive behaviors, are a hallmark of OCD and can influence the likelihood of a comorbid 
            depression diagnosis. For example, compulsions like checking and washing, which can dominate daily life, 
            are often associated with higher levels of anxiety and stress, potentially contributing to depressive symptoms. 
            Understanding these correlations helps in developing more targeted interventions for individuals with co-occurring OCD and depression.
        """)

    elif assoc_choice == 'Obsession Types':
        st.write("Obsessions can vary in their impact on depression. This chart visualizes how different types of obsessive thoughts correlate with depression diagnosis.")

        # Calculate the correlation between obsession types and Depression Diagnosis
        obsession_correlations = df[['Obsession Type_Harm-related', 'Obsession Type_Contamination', 'Obsession Type_Symmetry',
                                    'Obsession Type_Hoarding', 'Obsession Type_Religious', 'Depression Diagnosis']].corr()['Depression Diagnosis'][:-1]

        # Plot the correlations for obsession types
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(obsession_correlations)), obsession_correlations, color='#003366')  # Navy color
        ax.set_xticks(range(len(obsession_correlations)))
        ax.set_xticklabels(['Harm-related', 'Contamination', 'Symmetry', 'Hoarding', 'Religious'], rotation=45)
        ax.set_xlabel('Obsession Type')
        ax.set_ylabel('Correlation with Depression Diagnosis')
        ax.set_title('Correlation between Obsession Types and Depression Diagnosis')
        st.pyplot(fig)

        # Explanation of the findings
        st.write("""
            Obsessive thoughts, such as those related to harm, contamination, or symmetry, can also influence the severity of depression in patients with OCD. 
            For instance, harm-related obsessions, which are often associated with intense guilt and fear, may be linked to a higher risk of depression. 
            These insights help clinicians understand the interplay between OCD symptoms and depression, aiding in more holistic treatment approaches.
        """)


# Right Bottom Area: Y-BOCS Score and Depression Diagnosis Analysis
with right_col:
    st.markdown("---")  # Optional separator line
    st.subheader("Y-BOCS Scores and Their Impact on Depression Diagnosis")

    # Dropdown to select the type of Y-BOCS score to analyze
    ybocs_choice = st.selectbox(
        "Select Y-BOCS score type to analyze:",
        ['Y-BOCS Obsession Scores', 'Y-BOCS Compulsion Scores', 'Total Y-BOCS Score']
    )

    if ybocs_choice == 'Y-BOCS Obsession Scores':
        st.write("The Y-BOCS obsession score reflects the severity of obsessive thoughts. Below, we use a box plot to compare obsession scores for patients with and without depression.")

        # Box plot for Y-BOCS Obsession Scores vs Depression Diagnosis
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=df['Depression Diagnosis'], y=df['Y-BOCS Score (Obsessions)'], ax=ax)
        ax.set_xlabel('Depression Diagnosis (0 = No, 1 = Yes)')
        ax.set_ylabel('Y-BOCS Obsession Score')
        ax.set_title('Y-BOCS Obsession Scores vs Depression Diagnosis')
        st.pyplot(fig)

    elif ybocs_choice == 'Y-BOCS Compulsion Scores':
        st.write("The Y-BOCS compulsion score reflects the severity of compulsive behaviors. Below, we use a box plot to compare compulsion scores for patients with and without depression.")

        # Box plot for Y-BOCS Compulsion Scores vs Depression Diagnosis
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=df['Depression Diagnosis'], y=df['Y-BOCS Score (Compulsions)'], ax=ax)
        ax.set_xlabel('Depression Diagnosis (0 = No, 1 = Yes)')
        ax.set_ylabel('Y-BOCS Compulsion Score')
        ax.set_title('Y-BOCS Compulsion Scores vs Depression Diagnosis')
        st.pyplot(fig)

    elif ybocs_choice == 'Total Y-BOCS Score':
        # Calculate Total Y-BOCS score
        df['Total Y-BOCS Score'] = df['Y-BOCS Score (Obsessions)'] + df['Y-BOCS Score (Compulsions)']
        st.write("The total Y-BOCS score is a sum of both obsession and compulsion scores. Below, we use a box plot to compare total Y-BOCS scores for patients with and without depression.")

        # Box plot for Total Y-BOCS Scores vs Depression Diagnosis
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=df['Depression Diagnosis'], y=df['Total Y-BOCS Score'], ax=ax)
        ax.set_xlabel('Depression Diagnosis (0 = No, 1 = Yes)')
        ax.set_ylabel('Total Y-BOCS Score')
        ax.set_title('Total Y-BOCS Score vs Depression Diagnosis')
        st.pyplot(fig)

    # Interpretation and diagnostic relevance
    st.write("""
        The Y-BOCS score is a standard clinical measure to assess the severity of obsessive-compulsive symptoms. The box 
        plots allow us to see how patients with depression compare in their OCD symptom severity, which can help in diagnosing 
        and tailoring treatment for comorbid conditions.
    """)
