import streamlit as st

def render_consultant_advice(consultant_data: dict):
    """
    Renders the proactive consultant recommendations beautifully in Streamlit.
    Call this automatically when the backend returns 'consultant_advice'.
    """
    if not consultant_data:
        return
        
    st.markdown("---")
    st.subheader("💡 AI Consultant Proactive Insights")
    st.caption("Based on your recent analysis, here are grounded recommendations.")

    # 1. Top Insights
    if consultant_data.get("top_insights"):
        st.markdown("**🔍 Key Takeaways:**")
        for insight in consultant_data["top_insights"]:
            st.markdown(f"- {insight}")

    # 2. Risk Flags (Show prominently if they exist)
    if consultant_data.get("risk_flags"):
        st.warning("**⚠️ Risks Detected:**")
        for risk in consultant_data["risk_flags"]:
            st.markdown(f"- {risk}")

    col1, col2 = st.columns(2)
    
    # 3. Recommendations
    with col1:
        st.info("**🎯 Actionable Recommendations:**")
        for rec in consultant_data.get("recommendations", []):
            st.markdown(f"- {rec}")
            
    # 4. Next Steps
    with col2:
        st.success("**🚀 Recommended Next Steps:**")
        for step in consultant_data.get("next_steps", []):
            st.markdown(f"- {step}")
