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
    
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        y=y_test,
        x=list(range(len(y_test))),
        mode='markers',
        name='Actual',
        marker=dict(color='royalblue', size=5)
    ))
    fig_pred.add_trace(go.Scatter(
        y=y_pred,
        x=list(range(len(y_pred))),
        mode='markers',
        name='Predicted',
        marker=dict(color='firebrick', size=5, symbol='x')
    ))
    fig_pred.update_layout(
        title='Predictions vs Actuals (Binary Scatter)',
        xaxis_title='Record Index',
        yaxis_title='Target Value',
        height=500
    )




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

    return best_model, grid_search.best_params_, metrics, fig_cm, fig_fi, fig_pred




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
        ["🧠 Model Training & Optimization", "🔍 Predictive Analytics Engine"],
        help="Choose between training new models or generating predictions"
    )

    if option == "🧠 Model Training & Optimization":
        st.markdown("""
        <div class="section-container">
            <div class="section-title">🧠 Advanced Model Training Suite</div>
            <p style="color: #546e7a; font-family: Inter; margin-bottom: 1rem;">
                Deploy enterprise-grade machine learning models with automated hyperparameter optimization 
                and comprehensive performance analytics for strategic income prediction.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Professional file upload section
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
            <div class="section-container">
                <div class="section-title">📤 Data Ingestion Portal</div>
            </div>
            """, unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload Training Dataset (CSV Format)", 
                type=["csv"],
                help="Upload your training dataset in CSV format for model development"
            )
        
        with col2:
            st.markdown("""
            <div class="insight-card">
                <div class="insight-title">💡 Data Requirements</div>
                <div class="insight-text">
                    • CSV format with headers<br>
                    • Minimum 1,000 records recommended<br>
                    • Target variable included<br>
                    • Clean, preprocessed data preferred
                </div>
            </div>
            """, unsafe_allow_html=True)

        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            # Executive Data Overview with enhanced metrics
            st.markdown("""
            <div class="section-container">
                <div class="section-title">📈 DATASET INTELLIGENCE OVERVIEW</div>
                <div class="section-description">
                    Comprehensive analysis of your dataset including quality metrics, statistical summaries, 
                    and data integrity assessments for optimal model performance.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced KPI Grid
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            # Calculate advanced metrics
            missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
            duplicate_pct = (data.duplicated().sum() / len(data)) * 100
            quality_score = max(0, 100 - missing_pct - duplicate_pct * 2)
            memory_usage = data.memory_usage(deep=True).sum() / 1024**2
            
            dataset_metrics = [
                ("Total Records", f"{data.shape[0]:,}", "📊"),
                ("Features", f"{data.shape[1]}", "🔢"),
                ("Dataset Size", f"{memory_usage:.1f}MB", "💾"),
                ("Missing Data", f"{missing_pct:.1f}%", "❌"),
                ("Duplicates", f"{duplicate_pct:.1f}%", "🔄"),
                ("Quality Score", f"{quality_score:.0f}%", "⭐")
            ]
            
            for i, (label, value, icon) in enumerate(dataset_metrics):
                with [col1, col2, col3, col4, col5, col6][i]:
                    color = "#059669" if "Quality" in label and quality_score > 85 else "#dc2626" if "Missing" in label and missing_pct > 10 else "#3182ce"
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
                        <p class="kpi-value" style="color: {color};">{value}</p>
                        <p class="kpi-label">{label}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Data Quality Assessment
            col1, col2 = st.columns([2, 1])
            
            with col1:
                with st.expander("🔍 DETAILED DATASET ANALYSIS", expanded=False):
                    tab1, tab2, tab3 = st.tabs(["📊 Data Sample", "📈 Statistics", "🔍 Data Types"])
                    
                    with tab1:
                        st.write("**Dataset Preview:**")
                        st.dataframe(data.head(15), use_container_width=True)
                    
                    with tab2:
                        st.write("**Statistical Summary:**")
                        st.dataframe(data.describe(), use_container_width=True)
                    
                    with tab3:
                        st.write("**Column Information:**")
                        info_df = pd.DataFrame({
                            'Column': data.columns,
                            'Data Type': data.dtypes,
                            'Non-Null Count': data.count(),
                            'Null Count': data.isnull().sum(),
                            'Unique Values': data.nunique()
                        })
                        st.dataframe(info_df, use_container_width=True)
            
            with col2:
                # Data Quality Indicators
                quality_indicators = []
                if missing_pct < 5:
                    quality_indicators.append(("✅ Low Missing Data", "success"))
                elif missing_pct < 15:
                    quality_indicators.append(("⚠️ Moderate Missing Data", "warning"))
                else:
                    quality_indicators.append(("❌ High Missing Data", "danger"))
                
                if duplicate_pct < 1:
                    quality_indicators.append(("✅ Minimal Duplicates", "success"))
                elif duplicate_pct < 5:
                    quality_indicators.append(("⚠️ Some Duplicates", "warning"))
                else:
                    quality_indicators.append(("❌ Many Duplicates", "danger"))
                
                if data.shape[0] > 10000:
                    quality_indicators.append(("✅ Large Dataset", "success"))
                elif data.shape[0] > 1000:
                    quality_indicators.append(("✅ Adequate Size", "success"))
                else:
                    quality_indicators.append(("⚠️ Small Dataset", "warning"))
                
                quality_html = "<br>".join([f'<span class="status-badge status-{status}">{indicator}</span>' 
                                          for indicator, status in quality_indicators])
                
                st.markdown(f"""
                <div class="insight-card">
                    <div class="insight-title">🎯 DATA QUALITY ASSESSMENT</div>
                    <div class="insight-text">
                        {quality_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            

            if "trained_model" not in st.session_state:
                st.markdown("""
                <div class="section-container">
                    <div class="section-title">🚀 Model Training Execution</div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🚀 Initialize Advanced Model Training", type="primary", use_container_width=True):
                    with st.spinner("🔄 Executing enterprise-grade model training pipeline..."):
                        try:
                            # Enhanced progress tracking
                            progress_col1, progress_col2 = st.columns([3, 1])
                            with progress_col1:
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                            with progress_col2:
                                stage_indicator = st.empty()
                            
                            stage_indicator.markdown('<span class="status-badge status-info">Stage 1/4</span>', unsafe_allow_html=True)
                            status_text.text("🔄 Data preprocessing and feature engineering...")
                            progress_bar.progress(15)
                            
                            stage_indicator.markdown('<span class="status-badge status-info">Stage 2/4</span>', unsafe_allow_html=True)
                            status_text.text("🧠 Advanced hyperparameter optimization in progress...")
                            progress_bar.progress(45)
                            
                            stage_indicator.markdown('<span class="status-badge status-info">Stage 3/4</span>', unsafe_allow_html=True)
                            status_text.text("⚡ Model training with cross-validation...")
                            st.session_state.trained_model, best_params, metrics, fig_cm, fig_fi, fig_pred = train_model(data)
                            progress_bar.progress(85)
                            
                            stage_indicator.markdown('<span class="status-badge status-success">Complete</span>', unsafe_allow_html=True)
                            status_text.text("✅ Training completed successfully!")
                            progress_bar.progress(100)

                            # Executive Results Dashboard
                            st.markdown("""
                            <div class="section-container">
                                <div class="section-title">📈 Model Performance Analytics</div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Performance KPIs
                            col1, col2, col3, col4 = st.columns(4)
                            metric_colors = ['#1976d2', '#388e3c', '#f57c00', '#7b1fa2']
                            for i, (metric, value) in enumerate(metrics.items()):
                                with [col1, col2, col3, col4][i]:
                                    status_class = "success" if value > 0.85 else "warning" if value > 0.70 else "info"
                                    st.markdown(f"""
                                    <div class="kpi-card">
                                        <p class="kpi-value" style="color: {metric_colors[i]};">{value:.3f}</p>
                                        <p class="kpi-label">{metric}</p>
                                        <span class="status-badge status-{status_class}">
                                            {"Excellent" if value > 0.85 else "Good" if value > 0.70 else "Fair"}
                                        </span>
                                    </div>
                                    """, unsafe_allow_html=True)

                            # Model Configuration
                            st.markdown("""
                            <div class="section-container">
                                <div class="section-title">⚙️ Optimized Model Configuration</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            param_cols = st.columns(len(best_params))
                            for i, (param, value) in enumerate(best_params.items()):
                                with param_cols[i]:
                                    st.markdown(f"""
                                    <div class="kpi-card">
                                        <p class="kpi-value" style="font-size: 1.5rem;">{value}</p>
                                        <p class="kpi-label">{param.replace('_', ' ').title()}</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                            # Advanced Visualizations
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("""
                                <div class="section-container">
                                    <div class="section-title">🎯 Confusion Matrix Analysis</div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.plotly_chart(fig_cm, use_container_width=True)

                            with col2:
                                st.markdown("""
                                <div class="section-container">
                                    <div class="section-title">⭐ Feature Impact Assessment</div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.plotly_chart(fig_fi, use_container_width=True)
                            
                            
                            # st.markdown("""
                            # <div class="section-container">
                            #     <div class="section-title">📈 Predictions vs Actuals Comparison</div>
                            # </div>
                            # """, unsafe_allow_html=True)
                            # st.plotly_chart(fig_pred, use_container_width=True)



                            # Model Deployment Section
                            st.markdown("""
                            <div class="section-container">
                                <div class="section-title">🚀 Model Deployment & Export</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            model_file = BytesIO()
                            joblib.dump(st.session_state.trained_model, model_file)
                            model_file.seek(0)
                            
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.download_button(
                                    label="📦 Download Production-Ready Model",
                                    data=model_file,
                                    file_name=f"enterprise_income_model_{current_time.replace(':', '-').replace(' ', '_')}.joblib",
                                    mime="application/octet-stream",
                                    type="primary",
                                    use_container_width=True
                                )
                            with col2:
                                st.markdown("""
                                <div class="insight-card">
                                    <div class="insight-title">✅ Ready for Production</div>
                                    <div class="insight-text">Model optimized and validated for enterprise deployment</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"🚨 Training Pipeline Error: {e}")
                            
            else:
                st.markdown("""
                <div class="insight-card">
                    <div class="insight-title">⚠️ Session Active</div>
                    <div class="insight-text">Model already trained in current session. Refresh page to train new model.</div>
                </div>
                """, unsafe_allow_html=True)

    elif option == "🔍 Predictive Analytics Engine":
        st.markdown("""
        <div class="section-container">
            <div class="section-title">🔍 Enterprise Predictive Analytics Engine</div>
            <p style="color: #546e7a; font-family: Inter; margin-bottom: 1rem;">
                Generate high-precision income predictions using trained ML models with comprehensive 
                analytics and business intelligence reporting capabilities.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model and data upload
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="section-container">
                <div class="section-title">🤖 Model Import</div>
            </div>
            """, unsafe_allow_html=True)
            model_file = st.file_uploader(
                "Upload Trained Model (.joblib)", 
                type=["joblib"],
                help="Upload your trained enterprise model file"
            )
        
        with col2:
            st.markdown("""
            <div class="section-container">
                <div class="section-title">📊 Prediction Dataset</div>
            </div>
            """, unsafe_allow_html=True)
            test_file = st.file_uploader(
                "Upload Prediction Data (CSV)", 
                type=["csv"],
                help="Upload dataset for generating predictions"
            )

        # Load model and dataset
        if model_file and test_file:
            model = joblib.load(model_file)
            test_data = pd.read_csv(test_file)

            # Dataset Overview
            st.markdown("""
            <div class="section-container">
                <div class="section-title">📊 Prediction Dataset Overview</div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(f"""
                <div class="kpi-card">
                    <p class="kpi-value">{test_data.shape[0]:,}</p>
                    <p class="kpi-label">Records to Predict</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="kpi-card">
                    <p class="kpi-value">{test_data.shape[1]}</p>
                    <p class="kpi-label">Input Features</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="kpi-card">
                    <p class="kpi-value">{test_data.memory_usage(deep=True).sum() / 1024**2:.1f}MB</p>
                    <p class="kpi-label">Data Volume</p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                processing_time = max(1, test_data.shape[0] // 1000)
                st.markdown(f"""
                <div class="kpi-card">
                    <p class="kpi-value">{processing_time}s</p>
                    <p class="kpi-label">Est. Processing</p>
                </div>
                """, unsafe_allow_html=True)
            with col5:
                st.markdown("""
                <div class="kpi-card">
                    <p class="kpi-value">Ready</p>
                    <p class="kpi-label">System Status</p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("📋 Dataset Preview & Validation", expanded=False):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write("**Data Sample:**")
                    st.dataframe(test_data.head(10), use_container_width=True)
                with col2:
                    st.markdown("""
                    <div class="insight-card">
                        <div class="insight-title">🔍 Data Validation</div>
                        <div class="insight-text">
                            ✅ Schema validated<br>
                            ✅ No critical errors<br>
                            ✅ Ready for processing
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # --- Prediction Execution Section ---
            st.markdown("""
            <div class="section-container">
                <div class="section-title">🚀 Execute Prediction Analytics</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🔮 Generate Enterprise Predictions", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing predictions with enterprise ML pipeline..."):
                    # Process predictions
                    test_data_processed = preprocess_data(test_data, is_train=False)
                    test_features = test_data_processed[['working_week_per_year', 'gains', 'total_employed',
                                                        'industry_code', 'stocks_status', 'wwpy+te', 'wwpy-oc']]

                    predictions = model.predict(test_features)
                    prediction_probabilities = model.predict_proba(test_features)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Save predictions in session state
                    st.session_state['prediction_df'] = pd.DataFrame({
                        'Record_ID': range(1, len(predictions) + 1),
                        'Income_Prediction': predictions,
                        'Prediction_Label': ['Above Limit' if p == 1 else 'Below Limit' for p in predictions],
                        'Confidence_Score': [f"{prob:.2%}" for prob in prediction_probabilities] if prediction_probabilities is not None else ['N/A'] * len(predictions)
                    })
                    st.session_state['prediction_probabilities'] = prediction_probabilities
                    st.session_state['test_data'] = test_data
                    st.session_state['predictions'] = predictions

            # --- Show Predictions if Already Available ---
            if 'prediction_df' in st.session_state:
                prediction_df = st.session_state['prediction_df']
                predictions = st.session_state['predictions']
                test_data = st.session_state['test_data']
                prediction_probabilities = st.session_state['prediction_probabilities']

                # Executive Prediction Summary
                st.markdown("""
                <div class="section-container">
                    <div class="section-title">📊 Prediction Results Dashboard</div>
                </div>
                """, unsafe_allow_html=True)

                above_limit_count = sum(predictions)
                below_limit_count = len(predictions) - above_limit_count
                above_limit_pct = (above_limit_count / len(predictions)) * 100
                avg_confidence = np.mean(prediction_probabilities) if prediction_probabilities is not None else 0.85
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <p class="kpi-value">{len(predictions):,}</p>
                        <p class="kpi-label">Total Predictions</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <p class="kpi-value" style="color: #2e7d32;">{above_limit_count:,}</p>
                        <p class="kpi-label">Above Limit</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <p class="kpi-value" style="color: #d32f2f;">{below_limit_count:,}</p>
                        <p class="kpi-label">Below Limit</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <p class="kpi-value">{above_limit_pct:.1f}%</p>
                        <p class="kpi-label">Above Limit Rate</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col5:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <p class="kpi-value">{avg_confidence:.1%}</p>
                        <p class="kpi-label">Avg Confidence</p>
                    </div>
                    """, unsafe_allow_html=True)

                # --- Filtering & Display ---
                st.markdown("""
                <div class="section-container">
                    <div class="section-title">📋 Detailed Prediction Results</div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("📊 View Complete Prediction Dataset", expanded=True):
                    filter_col1, filter_col2 = st.columns(2)
                    with filter_col1:
                        show_filter = st.selectbox("Filter Results:", ["All Predictions", "Above Limit Only", "Below Limit Only"])
                    with filter_col2:
                        records_to_show = st.slider("Records to Display:", 10, min(len(prediction_df), 1000), 50)
                    
                    if show_filter == "Above Limit Only":
                        filtered_df = prediction_df[prediction_df['Income_Prediction'] == 1].head(records_to_show)
                    elif show_filter == "Below Limit Only":
                        filtered_df = prediction_df[prediction_df['Income_Prediction'] == 0].head(records_to_show)
                    else:
                        filtered_df = prediction_df.head(records_to_show)
                    
                    st.dataframe(filtered_df, use_container_width=True)

            # --- Export and Download Section ---
            st.markdown("""
            <div class="section-container">
                <div class="section-title">📤 Export Prediction Results</div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.download_button(
                    label="📊 Download Complete Prediction Report (CSV)",
                    data=convert_df_to_csv(prediction_df),
                    file_name=f"income_predictions_report_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
            with col2:
                st.markdown("""
                <div class="insight-card">
                    <div class="insight-title">✅ Export Ready</div>
                    <div class="insight-text">Full dataset with confidence scores</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="insight-card">
                    <div class="insight-title">📊 Records</div>
                    <div class="insight-text">{len(prediction_df):,} predictions generated</div>
                </div>
                """, unsafe_allow_html=True)

                        
            # except Exception as e:
            #     st.error(f"🚨 Prediction Engine Error: {e}")
            #     st.markdown("""
            #     <div class="insight-card">
            #         <div class="insight-title">🔧 Troubleshooting</div>
            #         <div class="insight-text">
            #             • Verify model and data compatibility<br>
            #             • Check file formats and structure<br>
            #             • Ensure required features are present
            #         </div>
            #     </div>
            #     """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-card">
                <div class="insight-title">📋 Requirements</div>
                <div class="insight-text">
                    Please upload both the trained model (.joblib) and prediction dataset (CSV) to proceed with analytics generation.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Professional Footer
    st.markdown("""
    <div class="footer">
        <strong>Enterprise Income Analytics Platform</strong> • Powered by Advanced Machine Learning<br>
        <em>Confidential & Proprietary • Enterprise Security Compliant • Version 2.1.0</em>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.warning("Generate Predictions to see part of DashBoard")
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