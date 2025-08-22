import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from joblib import parallel_backend
from io import BytesIO
import warnings 
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import yaml
from src.functions import *

if "sample_data_loaded" not in st.session_state:
    st.session_state.sample_data_loaded = False

# Initialize session state variables
if "training_completed" not in st.session_state:
    st.session_state.training_completed = False

if "sample_data_loaded" not in st.session_state:
    st.session_state.sample_data_loaded = False



# Configure page settings
st.set_page_config(
    page_title="Enterprise Income Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)






# Professional styling with corporate theme
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Corporate header */
    .enterprise-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .header-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 400;
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin: 0.5rem 0 0 0;
    }
    
    /* Professional sections */
    .section-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e1e8ed;
    }
    
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #f8fbff 0%, #ffffff 100%);
        border: 1px solid #e3f2fd;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s ease;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .kpi-value {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1976d2;
        margin: 0;
    }
    
    .kpi-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
        color: #546e7a;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0.5rem 0 0 0;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.3px;
    }
    
    .status-success {
        background: #e8f5e8;
        color: #2e7d32;
        border: 1px solid #81c784;
    }
    
    .status-info {
        background: #e3f2fd;
        color: #1976d2;
        border: 1px solid #64b5f6;
    }
    
    .status-warning {
        background: #fff3e0;
        color: #f57c00;
        border: 1px solid #ffb74d;
    }
    
    /* Professional sidebar */
    .sidebar-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
        border-bottom: 2px solid #667eea;
    }
    
    /* Data insights */
    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .insight-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .insight-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        opacity: 0.95;
        line-height: 1.4;
    }
    
    /* Progress styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #7c8db5;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        border-top: 1px solid #e1e8ed;
        margin-top: 3rem;
    }
    
    /* Hide Streamlit elements for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Utility Functions (unchanged)
# ---------------------------


# ---------------------------
# Enterprise Dashboard
# ---------------------------

try:
    # Corporate Header
    st.markdown("""
    <div class="enterprise-header">
        <h1 class="header-title">📊 Enterprise Income Analytics Platform</h1>
        <p class="header-subtitle">Advanced Machine Learning Solutions for Strategic Income Prediction & Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Executive Summary Banner
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <p class="kpi-value">AI/ML</p>
            <p class="kpi-label">Powered Analytics</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <p class="kpi-value">99%+</p>
            <p class="kpi-label">Accuracy Target</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <p class="kpi-value">Real-time</p>
            <p class="kpi-label">Processing</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="kpi-card">
            <p class="kpi-value">Enterprise</p>
            <p class="kpi-label">Grade Security</p>
        </div>
        """, unsafe_allow_html=True)

    # Professional Sidebar
    st.sidebar.markdown('<div class="sidebar-header">🎯 Platform Navigation</div>', unsafe_allow_html=True)
    
    # Sample Data Download Section - Load only once and cache in session state
    st.sidebar.markdown('<div class="sidebar-header">📁 Sample Datasets</div>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div class="insight-card">
        <div class="insight-title">📊 Demo Data Files</div>
        <div class="insight-text">
            Download these sample datasets to test the platform functionality.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    
    
    # Display dataset and model links in the sidebar without checking session state
    col1, col2, col3 = st.sidebar.columns(3)

    with col1:
        st.sidebar.write('<div style="font-size: 0.85rem; color: #000000;">Link for Train Data</div>', unsafe_allow_html=True)
        st.sidebar.code("https://huggingface.co/datasets/Garvitarunav/Train/resolve/main/Train.csv", language='text')

    with col2:
        st.sidebar.write('<div style="font-size: 0.85rem; color: #000000;">Link for Test Data</div>', unsafe_allow_html=True)
        st.sidebar.code("https://huggingface.co/datasets/Garvitarunav/Train/resolve/main/Test.csv", language='text')

    with col3:
        st.sidebar.write('<div style="font-size: 0.85rem; color: #000000;">Link for ML Model Pre-trained</div>', unsafe_allow_html=True)
        st.sidebar.code("https://huggingface.co/Garvitarunav/Income/resolve/main/enterprise_income_model_2025-08-21_14-26.joblib", language='text')

    st.sidebar.markdown("""
    <div style="font-size: 0.8rem; color: #546e7a; margin: 0.5rem 0; font-family: Inter;">
        💡 Copy these URLs to use Train.csv for training, Test.csv for predictions, and the pre-trained model.
    </div>
    """, unsafe_allow_html=True)



    # Add timestamp and session info
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.sidebar.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">📅 Session Information</div>
        <div class="insight-text">
            <strong>Date:</strong> {current_time}<br>
            <strong>Platform:</strong> Enterprise ML Suite<br>
            <strong>Version:</strong> 2.1.0 Enterprise
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    option = st.sidebar.selectbox(
        "Select Analytics Module:",
        ["Model Training & Optimization", "Predictive Analytics Engine"],
        help="Choose between training new models or generating predictions"
    )
    #### First Option
    if option == "Model Training & Optimization":
        first()



    ### Second Option
    elif option == "Predictive Analytics Engine":
        second()

    # Professional Footer
    st.markdown("""
    <div class="footer">
        <strong>Enterprise Income Analytics Platform</strong> • Powered by Advanced Machine Learning<br>
        <em>Confidential & Proprietary • Enterprise Security Compliant • Version 2.1.0</em>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.warning(f"Platform error: {e}")
    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">🛠️ System Support</div>
        <div class="insight-text">
            Contact enterprise support for technical assistance:<br>
            • Platform Status: Active<br>
            • Support Level: Enterprise<br>
            • Response Time: < 15 minutes
        </div>
    </div>
    """, unsafe_allow_html=True)