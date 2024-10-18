import streamlit as st

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

st.title("About")
st.write("""We are a group of health informatics students and professionals, bringing together a team of passionate data scientists and mental health specialists. Our shared goal is to develop innovative solutions for mental health prediction and analytics, focusing on the early detection of depression in individuals diagnosed with Obsessive-Compulsive Disorder (OCD).""")
st.write("""For our Depression Detection in OCD Patients project, we are utilizing advanced machine learning techniques and clinical insights to provide accurate and timely predictions of depression in OCD populations. By leveraging predictive analytics on this rich dataset, we aim to offer valuable tools that can support clinicians in identifying depression symptoms earlier and more effectively, ultimately improving patient care and treatment outcomes.""")
st.write("### OCD Patient Dataset for Depression Detection")
st.write("""The "OCD Patient Dataset: Demographics & Clinical Data" is a comprehensive collection of data from 1500 individuals diagnosed with Obsessive-Compulsive Disorder (OCD), aimed at supporting research into mental health, particularly focusing on depression detection. This dataset includes a wide range of demographic and clinical information to help researchers explore potential links between OCD and co-occurring depression, as well as develop models for depression detection.""")
st.write("**Reference to the dataset:** https://www.kaggle.com/datasets/ohinhaque/ocd-patient-dataset-demographics-and-clinical-data")

st.write("### Team Members")
st.write("Alexandros Alexakos, João Calixto, Yin Shea Lai, Tugba Cetinkaya, Katja Wilde, Kevin Arjona")
st.write("""This project is a collaboration with Karolinska Hospital, bringing together data science expertise and clinical experience to drive impactful advancements in depression detection and patient care.""")
st.write("For any inquiries, please contact us via our email: alexandros.alexakos@stud.ki.se")
st.image("assets/KarolinskaHospital.jpg")