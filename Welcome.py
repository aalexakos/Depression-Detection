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

st.title("Welcome to the Depression Detection Dashboard")
st.write("Explore the medical data regarding OCD and depression using the tabs on the left to navigate through different types of analysis.")
st.write("**Aim:** The early detection tool is for clinicians at the Karolinska University Hospital to help identify depression at its early stages based on the observed clinical parameters of the patients, thereby improving patient outcomes, reducing healthcare costs, and enhancing overall operational efficiency in diagnosis and treatment.")
st.write("**Intended Users:** The primary users are medical doctors (MDs) and other healthcare professionals at the Karolinska University Hospital. The tool is not intended to replace healthcare practitioners, instead it will serve as a complementary aid in their daily practice, enhancing their capacity to detect and treat depression early.")