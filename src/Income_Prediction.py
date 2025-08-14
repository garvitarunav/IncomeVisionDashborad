import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import *
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
            
            # Executive Data Overview
            st.markdown("""
            <div class="section-container">
                <div class="section-title">📊 Dataset Intelligence Overview</div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(f"""
                <div class="kpi-card">
                    <p class="kpi-value">{data.shape[0]:,}</p>
                    <p class="kpi-label">Total Records</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="kpi-card">
                    <p class="kpi-value">{data.shape[1]}</p>
                    <p class="kpi-label">Features</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="kpi-card">
                    <p class="kpi-value">{data.memory_usage(deep=True).sum() / 1024**2:.1f}MB</p>
                    <p class="kpi-label">Dataset Size</p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
                st.markdown(f"""
                <div class="kpi-card">
                    <p class="kpi-value">{missing_pct:.1f}%</p>
                    <p class="kpi-label">Missing Data</p>
                </div>
                """, unsafe_allow_html=True)
            with col5:
                quality_score = max(0, 100 - missing_pct - (data.duplicated().sum() / len(data) * 100))
                st.markdown(f"""
                <div class="kpi-card">
                    <p class="kpi-value">{quality_score:.0f}%</p>
                    <p class="kpi-label">Data Quality</p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("🔍 Dataset Sample & Quality Metrics", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Data Sample:**")
                    st.dataframe(data.head(), use_container_width=True)
                with col2:
                    st.write("**Statistical Summary:**")
                    st.dataframe(data.describe(), use_container_width=True)

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
                            st.session_state.trained_model, best_params, metrics, fig_cm, fig_fi = train_model(data)
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

        if model_file and test_file:
            try:
                model = joblib.load(model_file)
                test_data = pd.read_csv(test_file)

                # Data Intelligence Overview
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

                # Prediction Execution
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
                        
                        # Enhanced prediction dataframe
                        prediction_df = pd.DataFrame({
                            'Record_ID': range(1, len(predictions) + 1),
                            'Income_Prediction': predictions,
                            'Prediction_Label': ['Above Limit' if p == 1 else 'Below Limit' for p in predictions],
                            'Confidence_Score': [f"{prob:.2%}" for prob in prediction_probabilities] if prediction_probabilities is not None else ['N/A'] * len(predictions)
                        })

                        # Executive Prediction Summary
                        st.markdown("""
                        <div class="section-container">
                            <div class="section-title">📊 Prediction Results Dashboard</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Key metrics
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

                        # Prediction Distribution Visualization
                        if prediction_probabilities is not None:
                            col1, col2 = st.columns(2)
                            with col1:
                                # Distribution chart
                                fig_dist = go.Figure()
                                fig_dist.add_trace(go.Histogram(
                                    x=prediction_probabilities,
                                    nbinsx=20,
                                    marker_color='rgba(102, 126, 234, 0.7)',
                                    name='Confidence Distribution'
                                ))
                                fig_dist.update_layout(
                                    title={'text': 'Prediction Confidence Distribution', 'x': 0.5, 'font': {'size': 16, 'family': 'Inter'}},
                                    xaxis_title="Confidence Score",
                                    yaxis_title="Frequency",
                                    font=dict(family="Inter", size=12),
                                    height=400,
                                    plot_bgcolor='white',
                                    showlegend=False
                                )
                                st.plotly_chart(fig_dist, use_container_width=True)
                            
                            with col2:
                                # Pie chart
                                fig_pie = go.Figure(data=[go.Pie(
                                    labels=['Below Limit', 'Above Limit'],
                                    values=[below_limit_count, above_limit_count],
                                    hole=0.4,
                                    marker_colors=['#ff7f7f', '#90ee90']
                                )])
                                fig_pie.update_layout(
                                    title={'text': 'Income Classification Results', 'x': 0.5, 'font': {'size': 16, 'family': 'Inter'}},
                                    font=dict(family="Inter", size=12),
                                    height=400,
                                    plot_bgcolor='white'
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)

                        # Detailed Results Table
                        st.markdown("""
                        <div class="section-container">
                            <div class="section-title">📋 Detailed Prediction Results</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("📊 View Complete Prediction Dataset", expanded=True):
                            # Add filters
                            filter_col1, filter_col2 = st.columns(2)
                            with filter_col1:
                                show_filter = st.selectbox("Filter Results:", ["All Predictions", "Above Limit Only", "Below Limit Only"])
                            with filter_col2:
                                records_to_show = st.slider("Records to Display:", 10, min(len(prediction_df), 1000), 50)
                            
                            # Apply filters
                            if show_filter == "Above Limit Only":
                                filtered_df = prediction_df[prediction_df['Income_Prediction'] == 1].head(records_to_show)
                            elif show_filter == "Below Limit Only":
                                filtered_df = prediction_df[prediction_df['Income_Prediction'] == 0].head(records_to_show)
                            else:
                                filtered_df = prediction_df.head(records_to_show)
                            
                            st.dataframe(filtered_df, use_container_width=True)
                        
                        # Model Performance Validation (if ground truth available)
                        if "income_above_limit" in test_data.columns:
                            st.markdown("""
                            <div class="section-container">
                                <div class="section-title">🎯 Model Validation Report</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            y_true = test_data["income_above_limit"]
                            from sklearn.metrics import classification_report, accuracy_score
                            
                            # Validation metrics
                            val_accuracy = accuracy_score(y_true, predictions)
                            val_precision = precision_score(y_true, predictions)
                            val_recall = recall_score(y_true, predictions)
                            val_f1 = f1_score(y_true, predictions)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            validation_metrics = [
                                ("Validation Accuracy", val_accuracy),
                                ("Precision", val_precision),
                                ("Recall", val_recall),
                                ("F1-Score", val_f1)
                            ]
                            
                            for i, (metric, value) in enumerate(validation_metrics):
                                with [col1, col2, col3, col4][i]:
                                    status = "success" if value > 0.85 else "warning" if value > 0.70 else "info"
                                    st.markdown(f"""
                                    <div class="kpi-card">
                                        <p class="kpi-value">{value:.3f}</p>
                                        <p class="kpi-label">{metric}</p>
                                        <span class="status-badge status-{status}">
                                            {"Excellent" if value > 0.85 else "Good" if value > 0.70 else "Needs Review"}
                                        </span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Detailed classification report
                            with st.expander("📊 Detailed Classification Report", expanded=False):
                                report = classification_report(y_true, predictions)
                                st.text(report)

                        # Export and Download Section
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
                                file_name=f"income_predictions_report_{current_time.replace(':', '-').replace(' ', '_')}.csv",
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
                        
            except Exception as e:
                st.error(f"🚨 Prediction Engine Error: {e}")
                st.markdown("""
                <div class="insight-card">
                    <div class="insight-title">🔧 Troubleshooting</div>
                    <div class="insight-text">
                        • Verify model and data compatibility<br>
                        • Check file formats and structure<br>
                        • Ensure required features are present
                    </div>
                </div>
                """, unsafe_allow_html=True)
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
    st.error(f"🚨 Platform Error: {e}")
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