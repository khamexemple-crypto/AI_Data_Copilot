import streamlit as st
import pandas as pd
import time
from backend.services.full_analysis_runner import run_full_analysis

def render_full_analysis_section(df: pd.DataFrame):
    """
    Renders the UI controls to trigger a one-click end-to-end analysis.
    """
    st.markdown("---")
    st.subheader("⚡ One-Click Full Analysis")
    st.write("Automatically run EDA, cleaning, modeling, explainability, and recommendations in a single sweep.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # User selects the target column for ML
        target_col = st.selectbox(
            "Select Target Column (Optional, needed for Machine Learning)", 
            options=["None"] + list(df.columns)
        )
        
    with col2:
        st.write("") # Spacing alignment
        st.write("")
        gen_report = st.checkbox("Generate Consultant Report automatically", value=True)
        
    if st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True):
        target = target_col if target_col != "None" else None
        
        with st.status("Executing End-to-End Pipeline...", expanded=True) as status:
            st.write("🔍 Running EDA & Data Profiling...")
            time.sleep(0.5) # UX element
            
            st.write("📈 Generating Visualizations...")
            if target:
                st.write(f"🤖 Training AutoML models on target: **{target}**...")
                st.write("🧠 Extracting XAI (Explainable AI) features...")
            else:
                st.write("⏭️ Skipping Machine Learning (No target selected)...")
                
            st.write("💡 Synthesizing Consultant Recommendations...")
            
            try:
                # Execute backend coordination logic
                results = run_full_analysis(
                    df=df,
                    target_column=target,
                    generate_report_flag=gen_report
                )
                
                # Push aggregated results directly into the Streamlit session state
                # so the dashboard tabs update instantly.
                st.session_state["dataset_summary"] = results.get("dataset_summary")
                st.session_state["data_quality"] = results.get("data_quality")
                st.session_state["visualizations"] = results.get("visualizations")
                st.session_state["ml_results"] = results.get("ml_results")
                st.session_state["xai_results"] = results.get("xai_results")
                st.session_state["consultant_advice"] = results.get("consultant_advice")
                
                if "report" in results:
                     st.session_state["generated_report"] = results["report"]
                
                status.update(label="Pipeline Complete!", state="complete", expanded=False)
                st.success("Analysis finished successfully! Check the dashboard tabs above for insights.")
                
            except Exception as e:
                status.update(label="Pipeline Failed", state="error", expanded=True)
                st.error(f"An error occurred during execution: {str(e)}")
