import streamlit as st

def render_ml_warning_panel(issues_dict: dict):
    """
    Renders structured warnings and recommendations for automatically detected ML issues.
    """
    if not issues_dict:
        return
        
    warnings = issues_dict.get("warnings", [])
    detected_issues = issues_dict.get("detected_issues", [])
    recommendations = issues_dict.get("recommendations", [])
    
    st.markdown("---")
    st.subheader("🚨 Automated ML Issue Detection")
    
    if not detected_issues:
        st.success("✅ " + (warnings[0] if warnings else "No issues detected."))
        return

    # If issues exist, display them prominently
    st.warning("⚠️ **Potential Machine Learning Risks Detected**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Detected Warnings:**")
        for w in warnings:
            st.markdown(f"- {w}")
            
    with col2:
        st.markdown("**Recommended Actions:**")
        for r in recommendations:
            st.markdown(f"- {r}")
