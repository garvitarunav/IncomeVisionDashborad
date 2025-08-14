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

def convert_df_to_csv(df):
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode('utf-8')

def preprocess_data(df, is_train=True):
    """Clean and preprocess dataset."""
    drop_cols = [
        "ID", "class", "education_institute", "unemployment_reason", "is_labor_union",
        "occupation_code_main", "under_18_family", "veterans_admin_questionnaire",
        "residence_1_year_ago", "old_residence_reg", "old_residence_state",
        "migration_prev_sunbelt"
    ]
    df = df.drop(drop_cols, axis=1, errors="ignore")

    # Encode categorical fields
    df['gender'] = df['gender'].apply(lambda x: 1 if x != ' Female' else 0)

    if is_train and "income_above_limit" in df.columns:
        df["income_above_limit"] = df["income_above_limit"].apply(lambda x: 1 if x == "Above limit" else 0)

    # Feature engineering
    df['wwpy+te'] = df["working_week_per_year"] + df["total_employed"]
    df['wwpy-oc'] = df["working_week_per_year"] - df["occupation_code"]

    return df

def balance_classes(df, target_col):
    """Upsample minority class to match majority."""
    df_majority = df[df[target_col] == 0]
    df_minority = df[df[target_col] == 1]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    return pd.concat([df_majority, df_minority_upsampled])

def train_model(data):
    """Train a RandomForest model with GridSearchCV using threading backend."""
    data = preprocess_data(data, is_train=True)

    # Select final features
    features = ['working_week_per_year', 'gains', 'total_employed',
                'industry_code', 'stocks_status', 'wwpy+te', 'wwpy-oc', 'income_above_limit']
    final_df = data[features]
    final_df = balance_classes(final_df, "income_above_limit")

    X = final_df.drop("income_above_limit", axis=1)
    y = final_df["income_above_limit"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_features': ['sqrt'],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1],
        'bootstrap': [True]
    }

    with parallel_backend('threading'):
        grid_search = GridSearchCV(
            estimator=rfc,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=0,
            scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluation
    y_pred = best_model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

    # Enhanced visualizations
    cm = confusion_matrix(y_test, y_pred)
    
    # Professional confusion matrix
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: Below Limit', 'Predicted: Above Limit'],
        y=['Actual: Below Limit', 'Actual: Above Limit'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    fig_cm.update_layout(
        title={'text': 'Model Performance: Confusion Matrix', 'x': 0.5, 'font': {'size': 18, 'family': 'Inter'}},
        xaxis_title="Predicted Classes",
        yaxis_title="Actual Classes",
        font=dict(family="Inter", size=12),
        height=400,
        plot_bgcolor='white'
    )

    # Professional feature importance
    feature_importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=True)
    fig_fi = go.Figure(go.Bar(
        x=feature_importances.values,
        y=feature_importances.index,
        orientation='h',
        marker=dict(
            color=feature_importances.values,
            colorscale='Viridis',
            showscale=True
        ),
        text=[f'{val:.3f}' for val in feature_importances.values],
        textposition='outside'
    ))
    fig_fi.update_layout(
        title={'text': 'Feature Importance Analysis', 'x': 0.5, 'font': {'size': 18, 'family': 'Inter'}},
        xaxis_title="Importance Score",
        yaxis_title="Features",
        font=dict(family="Inter", size=12),
        height=500,
        plot_bgcolor='white',
        showlegend=False
    )

    return best_model, grid_search.best_params_, metrics, fig_cm, fig_fi
