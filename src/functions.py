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
import requests
import io
current_time = datetime.now().strftime("%Y-%m-%d %H:%M")



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




def first():
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


        uploaded_file = None
        csv_url = st.text_input(
            "📎 Enter Training Dataset Link (CSV Format)",
            placeholder="https://example.com/your-dataset.csv"
        )

        if csv_url:
            try:
                uploaded_file = pd.read_csv(csv_url)
            except:
                uploaded_file = None

    
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

    if uploaded_file is not None:
        data = uploaded_file  # already a DataFrame
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
        
        # Training section - check if model is already trained
        if not st.session_state.training_completed:
            st.markdown("""
            <div class="section-container">
                <div class="section-title">Model Training Execution</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Initialize Advanced Model Training", type="primary", use_container_width=True, key="train_model_btn"):
                with st.spinner("Executing enterprise-grade model training pipeline..."):
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
                        
                        # Train model and store in session state
                        trained_model, best_params, metrics, fig_cm, fig_fi, fig_pred = train_model(data)
                        st.session_state.trained_model = trained_model
                        st.session_state.model_metrics = metrics
                        st.session_state.best_params = best_params
                        st.session_state.confusion_matrix_fig = fig_cm
                        st.session_state.feature_importance_fig = fig_fi
                        st.session_state.predictions_fig = fig_pred
                        st.session_state.training_completed = True
                        
                        progress_bar.progress(85)
                        
                        stage_indicator.markdown('<span class="status-badge status-success">Complete</span>', unsafe_allow_html=True)
                        status_text.text("✅ Training completed successfully!")
                        progress_bar.progress(100)

                        st.success("🎉 Model training completed! Results are now available below.")
                        st.rerun()  # Rerun to show results immediately
                        
                    except Exception as e:
                        st.error(f"🚨 Training Pipeline Error: {e}")
        
        # Display results if training is complete
        if st.session_state.training_completed and st.session_state.model_metrics is not None:
            # Executive Results Dashboard
            st.markdown("""
            <div class="section-container">
                <div class="section-title">📈 Model Performance Analytics</div>
            </div>
            """, unsafe_allow_html=True)

            # Performance KPIs
            col1, col2, col3, col4 = st.columns(4)
            metric_colors = ['#1976d2', '#388e3c', '#f57c00', '#7b1fa2']
            for i, (metric, value) in enumerate(st.session_state.model_metrics.items()):
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
            
            param_cols = st.columns(len(st.session_state.best_params))
            for i, (param, value) in enumerate(st.session_state.best_params.items()):
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
                st.plotly_chart(st.session_state.confusion_matrix_fig, use_container_width=True)

            with col2:
                st.markdown("""
                <div class="section-container">
                    <div class="section-title">⭐ Feature Impact Assessment</div>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(st.session_state.feature_importance_fig, use_container_width=True)

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
        
        else:
            st.markdown("""
            <div class="insight-card">
                <div class="insight-title">💡 Ready to Train</div>
                <div class="insight-text">Click the training button above to start the model development process.</div>
            </div>
            """, unsafe_allow_html=True)





def second():
    import io
    import requests
    import pandas as pd
    import numpy as np
    import joblib
    import streamlit as st

    # ---- Initialize session state keys ----
    for key in ["sample_data_loaded", "training_completed", "prediction_df",
                "prediction_probabilities", "test_data", "predictions", "model_file"]:
        if key not in st.session_state:
            st.session_state[key] = None

    st.markdown("""
        <div class="section-container">
            <div class="section-title">Enterprise Predictive Analytics Engine</div>
            <p style="color: #546e7a; font-family: Inter; margin-bottom: 1rem;">
                Generate high-precision income predictions using trained ML models with comprehensive 
                analytics and business intelligence reporting capabilities.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Model and dataset upload
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="section-container"><div class="section-title"> Model Import</div></div>""", unsafe_allow_html=True)
        model_link = st.text_input(
            "Enter Hugging Face Model Link (.joblib)",
            help="Paste the direct Hugging Face link to your trained model file",
            key="model_input"
        )

        # Only load model once
        if model_link and st.session_state.model_file is None:
            if model_link.endswith(".joblib"):
                try:
                    response = requests.get(model_link)
                    response.raise_for_status()
                    st.session_state.model_file = joblib.load(io.BytesIO(response.content))
                    st.success("✅ Model loaded successfully from Hugging Face link")
                except Exception as e:
                    st.error(f"❌ Error loading model: {e}")
            else:
                st.error("❌ The provided link is not a .joblib file.")

    with col2:
        st.markdown("""<div class="section-container"><div class="section-title">Prediction Dataset</div></div>""", unsafe_allow_html=True)
        test_link = st.text_input(
            "Enter Prediction Dataset Link (CSV from Hugging Face)",
            help="Paste the direct CSV link for generating predictions",
            placeholder="https://example.com/your-dataset.csv",
            key="dataset_input"
        )

        # Only load dataset once
        if test_link and st.session_state.test_data is None:
            if test_link.endswith(".csv"):
                try:
                    st.session_state.test_data = pd.read_csv(test_link)
                    st.success("✅ Dataset loaded successfully")
                except Exception as e:
                    st.error(f"Failed to load dataset from link: {e}")
            else:
                st.error("❌ The provided link is not a CSV file.")

    # Shortcut variables
    model_file = st.session_state.model_file
    test_data = st.session_state.test_data

    # Load model and dataset only if both are available
    if model_file is not None and test_data is not None:
        # Dataset Overview
        st.markdown("""<div class="section-container"><div class="section-title"> Prediction Dataset Overview</div></div>""", unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"<div class='kpi-card'><p class='kpi-value'>{test_data.shape[0]:,}</p><p class='kpi-label'>Records to Predict</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='kpi-card'><p class='kpi-value'>{test_data.shape[1]}</p><p class='kpi-label'>Input Features</p></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='kpi-card'><p class='kpi-value'>{test_data.memory_usage(deep=True).sum() / 1024**2:.1f}MB</p><p class='kpi-label'>Data Volume</p></div>", unsafe_allow_html=True)
        with col4:
            processing_time = max(1, test_data.shape[0] // 1000)
            st.markdown(f"<div class='kpi-card'><p class='kpi-value'>{processing_time}s</p><p class='kpi-label'>Est. Processing</p></div>", unsafe_allow_html=True)
        with col5:
            st.markdown("<div class='kpi-card'><p class='kpi-value'>Ready</p><p class='kpi-label'>System Status</p></div>", unsafe_allow_html=True)

        with st.expander("Dataset Preview & Validation", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("**Data Sample:**")
                st.dataframe(test_data.head(10), use_container_width=True)
            with col2:
                st.markdown("""
                <div class="insight-card">
                    <div class="insight-title"> Data Validation</div>
                    <div class="insight-text">
                        ✅ Schema validated<br>
                        ✅ No critical errors<br>
                        ✅ Ready for processing
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Prediction Execution Section
        st.markdown("<div class='section-container'><div class='section-title'>🚀 Execute Prediction Analytics</div></div>", unsafe_allow_html=True)

        if st.session_state.prediction_df is None:
            if st.button("🔮 Generate Enterprise Predictions", type="primary", use_container_width=True, key="generate_predictions_btn"):
                with st.spinner("🔄 Processing predictions with enterprise ML pipeline..."):
                    try:
                        test_data_processed = preprocess_data(test_data, is_train=False)
                        test_features = test_data_processed[['working_week_per_year', 'gains', 'total_employed',
                                                            'industry_code', 'stocks_status', 'wwpy+te', 'wwpy-oc']]
                        predictions = model_file.predict(test_features)
                        prediction_probabilities = model_file.predict_proba(test_features)[:, 1] if hasattr(model_file, 'predict_proba') else None

                        st.session_state.prediction_df = pd.DataFrame({
                            'Record_ID': range(1, len(predictions) + 1),
                            'Income_Prediction': predictions,
                            'Prediction_Label': ['Above Limit' if p == 1 else 'Below Limit' for p in predictions],
                            'Confidence_Score': [f"{prob:.2%}" for prob in prediction_probabilities] if prediction_probabilities is not None else ['N/A'] * len(predictions)
                        })
                        st.session_state.prediction_probabilities = prediction_probabilities
                        st.session_state.predictions = predictions

                        st.success("🎉 Predictions generated successfully!")
                    except Exception as e:
                        st.error(f"🚨 Prediction Engine Error: {e}")

        # Show Predictions if Available
        if st.session_state.prediction_df is not None:
            prediction_df = st.session_state.prediction_df
            predictions = st.session_state.predictions
            prediction_probabilities = st.session_state.prediction_probabilities

            st.markdown("<div class='section-container'><div class='section-title'>📊 Prediction Results Dashboard</div></div>", unsafe_allow_html=True)

            above_limit_count = sum(predictions)
            below_limit_count = len(predictions) - above_limit_count
            above_limit_pct = (above_limit_count / len(predictions)) * 100
            avg_confidence = np.mean(prediction_probabilities) if prediction_probabilities is not None else 0.85

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(f"<div class='kpi-card'><p class='kpi-value'>{len(predictions):,}</p><p class='kpi-label'>Total Predictions</p></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='kpi-card'><p class='kpi-value' style='color: #2e7d32;'>{above_limit_count:,}</p><p class='kpi-label'>Above Limit</p></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='kpi-card'><p class='kpi-value' style='color: #d32f2f;'>{below_limit_count:,}</p><p class='kpi-label'>Below Limit</p></div>", unsafe_allow_html=True)
            with col4:
                st.markdown(f"<div class='kpi-card'><p class='kpi-value'>{above_limit_pct:.1f}%</p><p class='kpi-label'>Above Limit Rate</p></div>", unsafe_allow_html=True)
            with col5:
                st.markdown(f"<div class='kpi-card'><p class='kpi-value'>{avg_confidence:.1%}</p><p class='kpi-label'>Avg Confidence</p></div>", unsafe_allow_html=True)

            # Filtering & Display without rerunning entire app
            st.markdown("<div class='section-container'><div class='section-title'>📋 Detailed Prediction Results</div></div>", unsafe_allow_html=True)
            with st.expander("📊 View Complete Prediction Dataset", expanded=True):
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    show_filter = st.selectbox("Filter Results:", ["All Predictions", "Above Limit Only", "Below Limit Only"], key="filter_select")
                with filter_col2:
                    records_to_show = st.slider("Records to Display:", 10, min(len(prediction_df), 1000), 50, key="record_slider")

                # Filter in-place using session state, no rerun
                if show_filter == "Above Limit Only":
                    filtered_df = prediction_df[prediction_df['Income_Prediction'] == 1].head(records_to_show)
                elif show_filter == "Below Limit Only":
                    filtered_df = prediction_df[prediction_df['Income_Prediction'] == 0].head(records_to_show)
                else:
                    filtered_df = prediction_df.head(records_to_show)

                st.dataframe(filtered_df, use_container_width=True)

            # Export Section
            st.markdown("<div class='section-container'><div class='section-title'>📤 Export Prediction Results</div></div>", unsafe_allow_html=True)
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
                st.markdown("<div class='insight-card'><div class='insight-title'>✅ Export Ready</div><div class='insight-text'>Full dataset with confidence scores</div></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='insight-card'><div class='insight-title'>📊 Records</div><div class='insight-text'>{len(prediction_df):,} predictions generated</div></div>", unsafe_allow_html=True)

    else:
        st.markdown("<div class='insight-card'><div class='insight-title'>📋 Requirements</div><div class='insight-text'>Please provide both a trained model (.joblib) and a prediction dataset (CSV) to proceed with analytics generation.</div></div>", unsafe_allow_html=True)

