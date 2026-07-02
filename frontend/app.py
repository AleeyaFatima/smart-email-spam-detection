import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from src.config import BACKEND_URL
from frontend.api_client import APIClient
from frontend.components import render_kpi_card, render_prediction_badge, render_keyword_analyzer

# ==========================================
# DESIGN DECISION: SaaS Premium Layout
# Set page configuration, load custom CSS, and configure sidebar navigation.
# Load FontAwesome to support professional icons.
# Include REST API server health status check dynamically.
# ==========================================

st.set_page_config(
    page_title="Smart Email Spam Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load FontAwesome Icons & Styles CSS
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """,
    unsafe_allow_html=True
)

# Read and inject custom CSS rules
with open("frontend/styles.css", "r") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Initialize API Client
client = APIClient()

# Check server health
api_online = client.check_health()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 15px 0;">
            <h2 class="gradient-text" style="font-size: 1.6rem; margin: 0; font-weight: 800;">🛡️ SpamGuard AI</h2>
            <p style="color: #6b7280; font-size: 0.8rem; margin: 5px 0 5px 0;">Enterprise Email Security</p>
            <p style="color: #6366f1; font-size: 0.85rem; font-weight: 700; margin: 0 0 15px 0; letter-spacing: 0.05em;"><i class="fa-solid fa-code"></i> DEVELOPED BY ALEEYA</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # API Server Status Indicator
    if api_online:
        st.markdown(
            """
            <div style="
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid rgba(16, 185, 129, 0.2);
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 20px;
                text-align: center;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            ">
                <span style="height: 10px; width: 10px; background-color: #10b981; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #10b981;"></span>
                <span style="color: #10b981; font-size: 0.85rem; font-weight: 600;">REST API: ONLINE</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.2);
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 20px;
                text-align: center;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            ">
                <span style="height: 10px; width: 10px; background-color: #ef4444; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #ef4444;"></span>
                <span style="color: #ef4444; font-size: 0.85rem; font-weight: 600;">REST API: OFFLINE</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("<hr style='border-color: #1f2937; margin: 0 0 20px 0;'>", unsafe_allow_html=True)
    
    # Navigation Choices
    nav_selection = st.radio(
        "NAVIGATION",
        [
            "📊 Dashboard Overview",
            "🔍 Single Email Analyzer",
            "📁 Batch CSV Predictor",
            "📈 Model Comparison",
            "📜 Prediction Logs"
        ]
    )
    
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center; color: #4b5563; font-size: 0.75rem;">
            <p>Smart Spam Detector v1.0.0</p>
            <p>Admin Control Dashboard</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Warning if backend is not reachable
if not api_online:
    st.warning("⚠️ Cannot connect to the FastAPI backend. Some features may not work. Please launch the backend server.")

# ==========================================
# PAGE 1: DASHBOARD OVERVIEW
# ==========================================
if "Dashboard Overview" in nav_selection:
    st.markdown("<h1 style='margin-bottom: 5px;'>Dashboard Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #9ca3af; margin-bottom: 25px;'>Aggregate real-time statistics from active scans and database logs.</p>", unsafe_allow_html=True)
    
    if api_online:
        try:
            # Fetch stats from db
            stats = client.get_stats()
            # Fetch model comparison metadata
            metadata = client.get_model_comparison()
            active_model = metadata.get("active_model", {})
            
            # Display KPIs inside a 4-column layout
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_kpi_card(
                    title="Total Emails Scanned",
                    value=stats["total_scans"],
                    subtitle="Lifetime queries processed",
                    icon_class="fa-solid fa-envelope",
                    card_type="primary"
                )
            with col2:
                render_kpi_card(
                    title="Spam Emails Caught",
                    value=stats["total_spam"],
                    subtitle=f"{stats['spam_percentage']}% of total volume",
                    icon_class="fa-solid fa-shield-halved",
                    card_type="warning"
                )
            with col3:
                render_kpi_card(
                    title="Avg Confidence Score",
                    value=f"{stats['avg_confidence']}%",
                    subtitle="Calculated certainty",
                    icon_class="fa-solid fa-bullseye",
                    card_type="success"
                )
            with col4:
                render_kpi_card(
                    title="Avg Processing Latency",
                    value=f"{stats['avg_latency']} ms",
                    subtitle="Single email prediction speed",
                    icon_class="fa-solid fa-bolt",
                    card_type="info"
                )

            # Charts section (Spam distribution and logs history)
            st.markdown("<h3 style='margin: 30px 0 15px 0;'>System Data & Scan Analytics</h3>", unsafe_allow_html=True)
            col_chart1, col_chart2, col_chart3 = st.columns(3)
            
            with col_chart1:
                st.markdown("<h5 style='text-align: center; color: #f0ede4;'>Training Dataset Distribution</h5>", unsafe_allow_html=True)
                train_stats = metadata.get("dataset_stats", {})
                if train_stats and train_stats.get("total", 0) > 0:
                    fig_train = px.pie(
                        names=["Safe (Ham)", "Spam"],
                        values=[train_stats.get("ham", 0), train_stats.get("spam", 0)],
                        hole=0.5,
                        color_discrete_sequence=["#10b981", "#ef4444"],
                    )
                    fig_train.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#ffffff",
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                        margin=dict(t=10, b=10, l=10, r=10),
                        height=280
                    )
                    st.plotly_chart(fig_train, use_container_width=True)
                else:
                    st.info("No training dataset stats found.")
                    
            with col_chart2:
                st.markdown("<h5 style='text-align: center; color: #f0ede4;'>Live Scan Logs Distribution</h5>", unsafe_allow_html=True)
                # Donut chart for Spam vs Ham in history logs
                if stats["total_scans"] > 0:
                    fig_donut = px.pie(
                        names=["Safe (Ham)", "Spam"],
                        values=[stats["total_ham"], stats["total_spam"]],
                        hole=0.5,
                        color_discrete_sequence=["#10b981", "#ef4444"],
                    )
                    fig_donut.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#ffffff",
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                        margin=dict(t=10, b=10, l=10, r=10),
                        height=280
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)
                else:
                    st.info("No logs in database yet. Scan some emails to show charts.")
                    
            with col_chart3:
                st.markdown("<h5 style='text-align: center; color: #f0ede4;'>Scanner Certainty Trend</h5>", unsafe_allow_html=True)
                # Line chart representing scan confidence level trends
                history_scans = stats.get("history_scans", [])
                if history_scans:
                    df_scans = pd.DataFrame(history_scans)
                    # Convert labels
                    df_scans["label"] = df_scans["is_spam"].apply(lambda x: "Spam" if x == 1 else "Safe (Ham)")
                    
                    fig_line = px.line(
                        df_scans,
                        x=df_scans.index + 1,
                        y="confidence",
                        color="label",
                        labels={"index": "Scan Number", "confidence": "Certainty (%)"},
                        color_discrete_map={"Spam": "#ef4444", "Safe (Ham)": "#10b981"},
                        markers=True
                    )
                    fig_line.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#ffffff",
                        xaxis=dict(showgrid=False, title="Sequential Scan #"),
                        yaxis=dict(showgrid=True, gridcolor="#374151", title="Certainty %"),
                        margin=dict(t=10, b=10, l=10, r=10),
                        height=280
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.info("No logs trends available yet. Scan emails to populate chart.")

            # Display currently active model
            st.markdown("<h3 style='margin: 30px 0 15px 0;'>Production ML Model Node</h3>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
                    border: 1px solid #4f46e5;
                    border-radius: 12px;
                    padding: 20px;
                    color: #ffffff;
                ">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 15px;">
                        <div>
                            <span style="background: #4f46e5; border-radius: 6px; padding: 4px 8px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">ACTIVE NODAL ROUTE</span>
                            <h4 style="margin: 8px 0 2px 0; font-size: 1.4rem;">{active_model.get('model_name', 'Unknown Model')}</h4>
                            <p style="margin: 0; color: #9ca3af; font-size: 0.85rem;">Feature Vectorizer: <b>{active_model.get('vectorizer_name', 'Unknown')}</b></p>
                        </div>
                        <div style="text-align: right;">
                            <span style="color: #9ca3af; font-size: 0.85rem;">Pipeline Training F1 Score</span>
                            <h3 style="color: #6366f1; margin: 2px 0 0 0; font-size: 1.8rem; font-weight: 700;">{round(active_model.get('f1_score', 0.0) * 100, 2)}%</h3>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        except Exception as e:
            st.error(f"Error loading dashboard statistics: {e}")
    else:
        st.info("Please start the backend API server to view dashboard stats.")

# ==========================================
# PAGE 2: SINGLE EMAIL ANALYZER
# ==========================================
elif "Single Email Analyzer" in nav_selection:
    st.markdown("<h1 style='margin-bottom: 5px;'>Single Email Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #9ca3af; margin-bottom: 25px;'>Analyze raw email texts and highlight key spam trigger keywords.</p>", unsafe_allow_html=True)
    
    if api_online:
        with st.form("analyze_form"):
            email_input = st.text_area(
                "Paste Email Content Here",
                placeholder="Enter the full text of the email you want to scan...",
                height=180
            )
            
            submit_btn = st.form_submit_button("Scan Email Content")
            
        if submit_btn:
            if not email_input.strip():
                st.error("Please enter email text to analyze.")
            else:
                with st.spinner("Analyzing email patterns..."):
                    try:
                        # Make API request
                        res = client.analyze_email(email_input)
                        
                        # Display custom prediction badge
                        render_prediction_badge(
                            is_spam=res["is_spam"],
                            probability=res["spam_probability"],
                            confidence=res["confidence_score"]
                        )
                        
                        # Display highlighted keyword analyzer
                        st.markdown("<h3 style='margin: 25px 0 10px 0;'>Trigger Keyword Inspector</h3>", unsafe_allow_html=True)
                        render_keyword_analyzer(email_input)
                        
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
    else:
        st.error("Backend offline. Cannot perform analysis.")

# ==========================================
# PAGE 3: BATCH CSV PREDICTOR
# ==========================================
elif "Batch CSV Predictor" in nav_selection:
    st.markdown("<h1 style='margin-bottom: 5px;'>Batch CSV Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #9ca3af; margin-bottom: 25px;'>Upload a list of emails to check them for spam in bulk.</p>", unsafe_allow_html=True)
    
    if api_online:
        col_up, col_info = st.columns([2, 1])
        
        with col_up:
            # File Uploader
            uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
            
        with col_info:
            st.markdown("##### Template Format")
            st.write("Ensure your CSV has at least a column named **text** containing the email bodies.")
            
            # Simple download sample template CSV file
            sample_df = pd.DataFrame({
                "id": [1, 2, 3],
                "text": [
                    "Hey, are we still meeting for lunch at 1pm?",
                    "URGENT! Winner of £2000 cash prize! Claim free cash reward now!",
                    "Your bank account is locked. Log in to claim your refund."
                ]
            })
            
            csv_sample = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV Template",
                data=csv_sample,
                file_name="spam_detector_sample_template.csv",
                mime="text/csv"
            )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if "text" not in df.columns:
                    st.error("Invalid CSV structure: The CSV file must contain a column named 'text'!")
                else:
                    st.success("CSV loaded successfully! Ready to run batch prediction.")
                    
                    if st.button("🚀 Process Batch Predictions"):
                        # Format emails for API payload
                        emails_payload = []
                        for idx, row in df.iterrows():
                            # Generate an id if missing
                            row_id = str(row.get("id", idx + 1))
                            emails_payload.append({
                                "id": row_id,
                                "text": str(row["text"])
                            })
                            
                        progress_bar = st.progress(0.0)
                        
                        with st.spinner("Scouting email batch..."):
                            progress_bar.progress(0.3)
                            # Make REST call
                            result = client.analyze_batch(emails_payload)
                            progress_bar.progress(1.0)
                            
                        # Show stats
                        st.markdown("<h3 style='margin: 25px 0 15px 0;'>Batch Prediction Statistics</h3>", unsafe_allow_html=True)
                        
                        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                        with kpi_col1:
                            render_kpi_card(
                                title="Processed Emails",
                                value=result["total_processed"],
                                subtitle="Total batch size",
                                icon_class="fa-solid fa-list",
                                card_type="primary"
                            )
                        with kpi_col2:
                            render_kpi_card(
                                title="Spam Detected",
                                value=result["spam_count"],
                                subtitle="Blocked messages",
                                icon_class="fa-solid fa-shield-halved",
                                card_type="warning"
                            )
                        with kpi_col3:
                            render_kpi_card(
                                title="Clean Emails",
                                value=result["ham_count"],
                                subtitle="Legitimate messages",
                                icon_class="fa-solid fa-circle-check",
                                card_type="success"
                            )
                        with kpi_col4:
                            render_kpi_card(
                                title="Spam Percentage",
                                value=f"{result['spam_percentage']}%",
                                subtitle="Batch spam density",
                                icon_class="fa-solid fa-percent",
                                card_type="info"
                            )
                            
                        # Convert output predictions back to DataFrame for display & download
                        preds_out = result["predictions"]
                        df_out = pd.DataFrame(preds_out)
                        
                        # Re-map columns for cleaner UI
                        df_out["prediction"] = df_out["is_spam"].apply(lambda x: "Spam 🚨" if x else "Safe (Ham) ✅")
                        df_out["probability_pct"] = df_out["spam_probability"].apply(lambda x: f"{round(x * 100, 1)}%")
                        df_out["confidence_pct"] = df_out["confidence_score"].apply(lambda x: f"{x}%")
                        
                        df_display = df_out[["id", "text", "prediction", "probability_pct", "confidence_pct"]]
                        
                        st.markdown("<h4 style='margin: 25px 0 10px 0;'>Results Preview</h4>", unsafe_allow_html=True)
                        st.dataframe(df_display, use_container_width=True)
                        
                        # Downloader for final csv file
                        csv_output = df_out.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Batch Prediction Results",
                            data=csv_output,
                            file_name="spam_predictions_output.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Failed to process CSV file: {e}")
    else:
        st.error("Backend offline. Cannot perform batch predictions.")

# ==========================================
# PAGE 4: MODEL COMPARISON
# ==========================================
elif "Model Comparison" in nav_selection:
    st.markdown("<h1 style='margin-bottom: 5px;'>Model Training & Evaluation</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #9ca3af; margin-bottom: 25px;'>Analyze and compare different machine learning algorithms over TF-IDF and Word2Vec features.</p>", unsafe_allow_html=True)
    
    if api_online:
        try:
            metadata = client.get_model_comparison()
            all_models = metadata.get("all_models", [])
            active_model = metadata.get("active_model", {})

            # 1. Model comparison table
            df_models = pd.DataFrame(all_models)
            
            # Sort by F1 Score descending
            df_models = df_models.sort_values(by="f1_score", ascending=False)
            
            # Render model table
            df_table = df_models.copy()
            df_table["accuracy"] = df_table["accuracy"].apply(lambda x: f"{round(x * 100, 2)}%")
            df_table["precision"] = df_table["precision"].apply(lambda x: f"{round(x * 100, 2)}%")
            df_table["recall"] = df_table["recall"].apply(lambda x: f"{round(x * 100, 2)}%")
            df_table["f1_score"] = df_table["f1_score"].apply(lambda x: f"{round(x * 100, 2)}%")
            
            st.markdown("### Model Evaluation Metrics Matrix")
            st.table(df_table[["model_name", "vectorizer_name", "accuracy", "precision", "recall", "f1_score"]])

            # 2. Graph comparison (Plotly)
            st.markdown("### Comparative Performance Chart")
            
            # Melt DataFrame for Plotly grouped bar chart
            df_melted = pd.melt(
                df_models,
                id_vars=["model_name", "vectorizer_name"],
                value_vars=["accuracy", "precision", "recall", "f1_score"],
                var_name="Metric",
                value_name="Score"
            )
            df_melted["Model Node"] = df_melted["model_name"] + " (" + df_melted["vectorizer_name"] + ")"
            df_melted["Score (%)"] = df_melted["Score"] * 100.0

            fig_bar = px.bar(
                df_melted,
                x="Model Node",
                y="Score (%)",
                color="Metric",
                barmode="group",
                color_discrete_sequence=["#6366f1", "#06b6d4", "#f59e0b", "#10b981"],
                height=380
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ffffff",
                xaxis=dict(tickangle=-15, title="Algorithm & Feature Configuration"),
                yaxis=dict(gridcolor="#374151", title="Percentage Score (%)"),
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # 3. Retraining control panel
            st.markdown("<hr style='border-color: #1f2937; margin: 30px 0;'>", unsafe_allow_html=True)
            st.markdown("### Admin Control Panel: Model Training & Dataset Upload")
            
            col_retrain, col_upload = st.columns(2)
            
            with col_retrain:
                st.markdown("#### Quick Retrain")
                st.write(
                    "Retrain all 8 model combinations using the current training dataset pool on the server. "
                    "The system will automatically select the best model based on F1-score once complete."
                )
                if st.button("🔄 Trigger Model Training Pipeline"):
                    with st.spinner("Instructing training node..."):
                        try:
                            res = client.retrain_models()
                            st.success("SUCCESS: Model training has been successfully scheduled in the background. It will rebuild metrics in a few seconds.")
                        except Exception as e:
                            st.error(f"Failed to trigger retraining: {e}")
                            
            with col_upload:
                st.markdown("#### Upload Labeled Dataset")
                st.write(
                    "Upload a labeled CSV/TSV dataset to update the model training pool. "
                    "The file must contain text and labels (spam/ham)."
                )
                
                training_file = st.file_uploader("Choose training dataset file", type=["csv", "tsv"])
                upload_mode = st.selectbox("Upload Mode", ["overwrite", "append"], help="Overwrite replaces the current dataset; Append adds new data to it.")
                
                if st.button("📤 Upload & Retrain Models"):
                    if training_file is not None:
                        file_bytes = training_file.read()
                        with st.spinner("Uploading and parsing dataset..."):
                            try:
                                # 1. Upload to API
                                upload_res = client.upload_dataset(
                                    file_bytes=file_bytes,
                                    filename=training_file.name,
                                    mode=upload_mode
                                )
                                st.success(upload_res.get("message", "Dataset uploaded successfully."))
                                
                                # 2. Retrain
                                with st.spinner("Rebuilding machine learning models..."):
                                    client.retrain_models()
                                    st.success("Models retrained successfully! Reloading stats.")
                                    st.rerun()
                                    
                            except Exception as e:
                                st.error(f"Error: {e}")
                    else:
                        st.warning("Please select a valid CSV or TSV file to upload.")
                    
        except Exception as e:
            st.error(f"Error loading model comparisons: {e}")
    else:
        st.error("Backend offline. Cannot show comparisons.")

# ==========================================
# PAGE 5: PREDICTION LOGS
# ==========================================
elif "Prediction Logs" in nav_selection:
    st.markdown("<h1 style='margin-bottom: 5px;'>Prediction Logs</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #9ca3af; margin-bottom: 25px;'>Search, filter, and audit past emails scanned by this system.</p>", unsafe_allow_html=True)
    
    if api_online:
        col_search, col_label, col_lim = st.columns([2, 1, 1])
        
        with col_search:
            search_query = st.text_input("🔍 Search email text", placeholder="Type keywords to search logs...")
        with col_label:
            label_filter = st.selectbox("Label Filter", ["All", "Spam", "Ham"])
        with col_lim:
            limit_rows = st.number_input("Limit rows", min_value=10, max_value=500, value=100, step=10)
            
        try:
            # Fetch history logs
            history = client.get_history(limit=limit_rows, search=search_query, label=label_filter)
            
            if history:
                df_history = pd.DataFrame(history)
                
                # Format visual fields
                df_history["id"] = df_history["id"].astype(str)
                df_history["spam_prediction"] = df_history["is_spam"].apply(lambda x: "Spam 🚨" if x == 1 else "Safe ✅")
                df_history["probability"] = df_history["probability"].apply(lambda x: f"{round(x * 100, 1)}%")
                df_history["confidence"] = df_history["confidence"].apply(lambda x: f"{x}%")
                df_history["execution_time"] = df_history["execution_time_ms"].apply(lambda x: f"{x} ms")
                
                # Format ISO Datetime string to user-friendly string
                def format_date(iso_str):
                    try:
                        dt = datetime.fromisoformat(iso_str)
                        return dt.strftime("%b %d, %Y - %H:%M:%S")
                    except Exception:
                        return iso_str
                        
                df_history["scanned_at"] = df_history["timestamp"].apply(format_date)
                
                df_display = df_history[["id", "scanned_at", "email_text", "spam_prediction", "probability", "confidence", "model_used", "execution_time"]]
                st.dataframe(df_display, use_container_width=True)
                
                # Database Admin Panel
                st.markdown("<hr style='border-color: #1f2937; margin: 30px 0;'>", unsafe_allow_html=True)
                st.markdown("### Database Administration")
                
                # Danger Zone: clear database logs
                st.markdown(
                    """
                    <div style="background: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                        <span style="color: #ef4444; font-weight: 700; font-size: 0.95rem;">⚠️ DANGER ZONE</span>
                        <p style="color: #9ca3af; font-size: 0.85rem; margin: 5px 0 0 0;">Clearing the database will permanently delete all scan query logs and histories. This action cannot be undone.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Custom Clear Button inside columns
                col_clear, col_space = st.columns([1, 4])
                with col_clear:
                    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
                    if st.button("Permanently Clear Logs"):
                        with st.spinner("Deleting database..."):
                            client.clear_history()
                            st.success("Database logs cleared successfully!")
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            else:
                st.info("No matching records found in prediction history logs.")
                
        except Exception as e:
            st.error(f"Error fetching database logs: {e}")
    else:
        st.error("Backend offline. Cannot fetch database logs.")
