import streamlit as st

# We import the previous modules created
from frontend.ui_components.components import render_metric_card, render_leaderboard, render_charts

# We try to import the previously created proactive/report modules.
# Fallbacks included so the UI doesn't crash if they aren't fully wired yet.
try:
    from frontend.ui_components.recommendation_panel import render_consultant_advice
except ImportError:
    render_consultant_advice = lambda x: st.write("Consultant module not loaded.", x)

try:
    from frontend.ui_components.report_section import render_report_section
except ImportError:
    render_report_section = lambda x: st.write("Report module not loaded.")

def main_dashboard():
    # Page configuration MUST be the first Streamlit command
    st.set_page_config(page_title="AI Data Copilot Dashboard", page_icon="🚀", layout="wide")
    
    st.title("🚀 AI Data Copilot")
    st.markdown("Transform your raw data into professional insights instantly.")

    # Sidebar for actions
    with st.sidebar:
        st.header("Actions")
        uploaded_file = st.file_uploader("Upload Dataset (CSV/Excel)", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            st.session_state["dataset_name"] = uploaded_file.name
            st.success(f"Uploaded: {uploaded_file.name}")
        
        if st.button("▶ Run Full Analysis", use_container_width=True, type="primary"):
            st.session_state["analysis_running"] = True
            st.info("Backend call triggered (Simulated for demo).")
            # In real usage, call requests.post(...) here and populate session_state
            
    # Main Tabs structure
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "🛡 Data Quality", 
        "📈 Visualizations", 
        "🤖 Models", 
        "💡 Insights & Recs", 
        "📄 Report"
    ])

    with tab1:
        st.header("Dataset Overview")
        if st.session_state.get("dataset_summary"):
            summary = st.session_state["dataset_summary"]
            col1, col2, col3 = st.columns(3)
            with col1:
                render_metric_card("Total Rows", f"{summary.get('rows', 0):,}")
            with col2:
                render_metric_card("Total Columns", str(summary.get('columns', 0)))
            with col3:
                render_metric_card("Dataset Size", summary.get('size', 'N/A'))
            
            st.subheader("Key Features")
            st.write(", ".join(summary.get('features', [])))
        else:
            st.info("Upload a dataset and run analysis to view the overview.")

    with tab2:
        st.header("Data Quality")
        if st.session_state.get("data_quality"):
            dq = st.session_state["data_quality"]
            col1, col2 = st.columns(2)
            with col1:
                render_metric_card("Duplicate Records", str(dq.get("duplicates", 0)))
            with col2:
                missing = dq.get("missing_values", {})
                total_missing = sum(missing.values())
                render_metric_card("Total Missing Values", str(total_missing))
            
            if missing:
                st.write("**Missing Values Breakdown:**")
                st.json(missing)
        else:
            st.info("Data quality metrics will appear here.")

    with tab3:
        st.header("Visualizations")
        if st.session_state.get("visualizations"):
            render_charts(st.session_state["visualizations"])
        else:
            st.info("No charts generated yet.")

    with tab4:
        st.header("Machine Learning")
        if st.session_state.get("ml_results"):
            ml = st.session_state["ml_results"]
            st.subheader(f"🏆 Best Model: {ml.get('best_model', 'N/A')}")
            render_leaderboard(ml)
        else:
            st.info("ML Leaderboard will appear here.")

    with tab5:
        st.header("Insights & Recommendations")
        if st.session_state.get("consultant_advice"):
            render_consultant_advice(st.session_state["consultant_advice"])
        else:
            st.info("Consultant advice will be generated after analysis.")

    with tab6:
        session_id = st.session_state.get("session_id", "default_session")
        render_report_section(session_id)

if __name__ == "__main__":
    main_dashboard()
