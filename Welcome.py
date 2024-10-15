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
st.image("assets/home.png", caption="The cycle of OCD", use_column_width=True)