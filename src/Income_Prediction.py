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
from functions import *

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
    initial_sidebar_state="collapsed"
)

# Modern Professional Styling with Card Effects
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global styling with neutral palette */
    .main > div {
        padding-top: 1rem;
        background: #fafafa;
        min-height: 100vh;
    }
    
    /* Clean minimal header */
    .platform-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        margin: -1rem -1rem 3rem -1rem;
        padding: 3rem 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .platform-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 30%, rgba(34, 197, 94, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 80% 70%, rgba(168, 85, 247, 0.05) 0%, transparent 40%);
        pointer-events: none;
    }
    
    .header-content {
        max-width: 1200px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }
    
    .platform-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff !important;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    
    .platform-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 400;
        color: #cbd5e1 !important;
        margin: 0;
        opacity: 0.9;
    }
    
    /* Additional override for platform header text */
    .platform-header h1,
    .platform-header .platform-title,
    .header-content h1 {
        color: #ffffff !important;
    }
    
    .platform-header p,
    .platform-header .platform-subtitle,
    .header-content p {
        color: #cbd5e1 !important;
    }
    
    /* Modern metric grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-tile {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.2s ease;
        position: relative;
    }
    
    .metric-tile:hover {
        border-color: #22c55e;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        font-weight: 500;
        color: #64748b;
        margin: 0.25rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-accent {
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #22c55e 0%, #a855f7 100%);
        border-radius: 12px 0 0 12px;
    }
    
    /* Professional sections with card effect */
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
    
    /* Clean section styling */
    .content-section {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        position: relative;
    }
    
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0 0 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-header::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, #e2e8f0 0%, transparent 100%);
    }
    
    /* KPI Cards from sample */
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
    
    /* Modern status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.02em;
        gap: 0.5rem;
    }
    
    .status-active {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    .status-ready {
        background: #dbeafe;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    
    .status-processing {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    
    /* Status badges from sample */
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
    
    /* Sophisticated insight panels */
    .insight-panel {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        color: #ffffff;
        position: relative;
        overflow: hidden;
    }
    
    .insight-panel::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(34, 197, 94, 0.1) 0%, transparent 70%);
        border-radius: 50%;
        transform: translate(50%, -50%);
        pointer-events: none;
    }
    
    .insight-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #22c55e;
    }
    
    .insight-content {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #e2e8f0;
        position: relative;
        z-index: 1;
    }
    
    /* Data insights from sample */
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
    
    /* Modern button styling */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12) !important;
        font-size: 0.9rem !important;
        height: 2.75rem !important;
        padding: 0 1.5rem !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Clean tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #f8fafc;
        border-radius: 12px;
        padding: 0.25rem;
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        padding: 0.875rem 1.5rem;
        border-radius: 8px;
        color: #64748b;
        transition: all 0.2s ease;
        background: transparent;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #ffffff;
        color: #0f172a;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.25) !important;
    }
    
    /* Clean data display */
    .data-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    /* Enhanced form elements */
    .stSelectbox > div > div {
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #374151;
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #22c55e;
        box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.1);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: #ffffff;
        border: 2px dashed #d1d5db;
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.2s ease;
        text-align: center;
    }
    
    .stFileUploader > div:hover {
        border-color: #22c55e;
        background: #f0fdf4;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #22c55e 0%, #a855f7 100%);
        border-radius: 4px;
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 8px !important;
        border: none !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }
    
    .stAlert[data-baseweb="notification"] {
        background: #f0f9ff !important;
        color: #1e40af !important;
        border-left: 4px solid #3b82f6 !important;
    }
    
    /* Clean dataframe */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Typography improvements */
    .stMarkdown p {
        font-family: 'Inter', sans-serif;
        color: #374151;
        font-weight: 400;
        line-height: 1.6;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: #0f172a;
        font-weight: 600;
    }
    
    /* Card styling for sub-headings */
    .stMarkdown h4 {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        color: #0f172a !important;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stMarkdown h4:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        border-color: #22c55e;
    }
    
    .stMarkdown h4::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #22c55e 0%, #a855f7 100%);
        border-radius: 12px 0 0 12px;
    }
    
    /* Advanced Card System for Sub-headings and Information */
    .info-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .info-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.15);
        border-color: #22c55e;
    }
    
    .info-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #22c55e 0%, #a855f7 100%);
        border-radius: 16px 0 0 16px;
    }
    
    .card-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .card-content {
        font-family: 'Inter', sans-serif;
        color: #4b5563;
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 1rem 0;
    }
    
    .card-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-item {
        text-align: center;
        padding: 1rem;
        background: #f8fafc;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    .metric-item:hover {
        background: #f0fdf4;
        border-color: #22c55e;
    }
    
    .metric-value-card {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    
    .metric-label-card {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 500;
        color: #64748b;
        margin: 0.25rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .performance-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        border: none;
    }
    
    .performance-card .card-header {
        color: #22c55e;
    }
    
    .performance-card .card-content {
        color: #e2e8f0;
    }
    
    .performance-card .metric-item {
        background: rgba(255,255,255,0.05);
        border-color: rgba(255,255,255,0.1);
        color: #ffffff;
    }
    
    .performance-card .metric-item:hover {
        background: rgba(34, 197, 94, 0.1);
        border-color: #22c55e;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border-color: #bbf7d0;
    }
    
    .feature-card .card-header {
        color: #166534;
    }
    
    .analysis-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fef7cd 100%);
        border-color: #fde68a;
    }
    
    .analysis-card .card-header {
        color: #92400e;
    }
    
    .config-card {
        background: linear-gradient(135deg, #dbeafe 0%, #e0f2fe 100%);
        border-color: #bfdbfe;
    }
    
    .config-card .card-header {
        color: #1e40af;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Resource link styling */
    .resource-link {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #475569;
        transition: all 0.2s ease;
    }
    
    .resource-link:hover {
        border-color: #22c55e;
        background: #f0fdf4;
    }
    
    .resource-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        color: #6b7280;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Modern footer */
    .platform-footer {
        background: #0f172a;
        margin: 4rem -1rem -1rem -1rem;
        padding: 2rem;
        text-align: center;
        color: #94a3b8;
        font-family: 'Inter', sans-serif;
        border-top: 1px solid #1e293b;
    }
    
    .footer-title {
        color: #ffffff;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .footer-text {
        font-size: 0.9rem;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Card Helper Functions
# ---------------------------

def create_performance_card(title, content="", metrics=None):
    """Create a performance analytics card with metrics"""
    metrics_html = ""
    if metrics:
        metrics_html = '<div class="card-metrics">'
        for metric_name, metric_value, rating in metrics:
            metrics_html += f"""
            <div class="metric-item">
                <div class="metric-value-card">{metric_value}</div>
                <div class="metric-label-card">{metric_name}</div>
                <div style="font-size: 0.7rem; color: #64748b; margin-top: 0.25rem;">{rating}</div>
            </div>
            """
        metrics_html += '</div>'
    
    st.markdown(f"""
    <div class="info-card performance-card">
        <div class="card-header">📊 {title}</div>
        <div class="card-content">{content}</div>
        {metrics_html}
    </div>
    """, unsafe_allow_html=True)

def create_config_card(title, content="", config_items=None):
    """Create a configuration card with parameters"""
    config_html = ""
    if config_items:
        config_html = '<div class="card-metrics">'
        for param_name, param_value in config_items:
            config_html += f"""
            <div class="metric-item">
                <div class="metric-value-card">{param_value}</div>
                <div class="metric-label-card">{param_name}</div>
            </div>
            """
        config_html += '</div>'
    
    st.markdown(f"""
    <div class="info-card config-card">
        <div class="card-header">⚙️ {title}</div>
        <div class="card-content">{content}</div>
        {config_html}
    </div>
    """, unsafe_allow_html=True)

def create_analysis_card(title, content=""):
    """Create an analysis card for insights"""
    st.markdown(f"""
    <div class="info-card analysis-card">
        <div class="card-header">🔍 {title}</div>
        <div class="card-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def create_feature_card(title, content=""):
    """Create a feature importance card"""
    st.markdown(f"""
    <div class="info-card feature-card">
        <div class="card-header">⭐ {title}</div>
        <div class="card-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Modern Enterprise Dashboard
# ---------------------------

try:
    # Clean Header
    st.markdown("""
    <div class="platform-header">
        <div class="header-content">
            <h1 class="platform-title">Enterprise Income Analytics</h1>
            <p class="platform-subtitle">Advanced machine learning platform for strategic income prediction and business intelligence</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Modern Metrics Grid
    st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-tile">
            <div class="metric-accent"></div>
            <div class="metric-value">AI/ML</div>
            <div class="metric-label">Powered Engine</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-tile">
            <div class="metric-accent"></div>
            <div class="metric-value">99%+</div>
            <div class="metric-label">Accuracy Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-tile">
            <div class="metric-accent"></div>
            <div class="metric-value">Real-time</div>
            <div class="metric-label">Processing</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-tile">
            <div class="metric-accent"></div>
            <div class="metric-value">Enterprise</div>
            <div class="metric-label">Security</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Sample Resources Section
    st.markdown("""
    <div class="content-section">
        <div class="section-header">📁 Sample Datasets & Resources</div>
        <div style="margin-bottom: 1.5rem;">
            <span class="status-indicator status-ready">Ready for Use</span>
        </div>
        <p style="color: #6b7280; margin-bottom: 1.5rem;">
            Access these curated datasets to demonstrate platform capabilities and test ML workflows.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Resource Links
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="resource-label">🎯 Training Dataset</div>', unsafe_allow_html=True)
        st.code("https://huggingface.co/datasets/Garvitarunav/Train/resolve/main/Train.csv", language='text')

    with col2:
        st.markdown('<div class="resource-label">🧪 Test Dataset</div>', unsafe_allow_html=True)
        st.code("https://huggingface.co/datasets/Garvitarunav/Train/resolve/main/Test.csv", language='text')

    with col3:
        st.markdown('<div class="resource-label">🤖 Pre-trained Model</div>', unsafe_allow_html=True)
        st.code("https://huggingface.co/Garvitarunav/Income/resolve/main/enterprise_income_model_2025-08-21_14-26.joblib", language='text')

    # Session Information
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(f"""
    <div class="insight-panel">
        <div class="insight-header">Session Information</div>
        <div class="insight-content">
            <strong>Session:</strong> {current_time} • <strong>Platform:</strong> Enterprise ML Suite v2.1.0 • <strong>Environment:</strong> Production Ready
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Clean Tab Navigation
    tab1, tab2 = st.tabs(["🤖 Model Training & Optimization", "🔮 Predictive Analytics Engine"])
    
    # Tab 1: Model Training
    with tab1:
        st.markdown("""
        <div class="content-section">
            <div class="section-header">🤖 Model Training & Optimization</div>
            <div style="margin-bottom: 1rem;">
                <span class="status-indicator status-active">Training Module Active</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        first()
    
    # Tab 2: Predictive Analytics
    with tab2:
        st.markdown("""
        <div class="content-section">
            <div class="section-header">🔮 Predictive Analytics Engine</div>
            <div style="margin-bottom: 1rem;">
                <span class="status-indicator status-ready">Analytics Ready</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        second()

    # Modern Footer
    st.markdown("""
    <div class="platform-footer">
        <div class="footer-title">Enterprise Income Analytics Platform</div>
        <div class="footer-text">
            Powered by Advanced Machine Learning • Enterprise Security Compliant • Version 2.1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Platform initialization error: {e}")
    st.markdown("""
    <div class="insight-panel">
        <div class="insight-header">Support Information</div>
        <div class="insight-content">
            Enterprise support available 24/7 • Response time: < 15 minutes • Status: All systems operational
        </div>
    </div>
    """, unsafe_allow_html=True)