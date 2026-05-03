import streamlit as st
import pandas as pd
import plotly.express as px

def render_xai_panel(xai_results: dict):
    """
    Renders the Explainable AI (XAI) feature importances and natural language summary.
    """
    if not xai_results:
        st.info("No explainability data available for this model.")
        return
        
    st.markdown("---")
    st.subheader("🧠 Model Explainability (XAI)")
    st.caption("Understand what drives the model's predictions.")

    # Render Natural Language Summary
    summary = xai_results.get("natural_language_summary", [])
    if summary:
        st.info(" ".join(summary))

    # Render Feature Importance Chart
    feature_importances = xai_results.get("feature_importance", [])
    if feature_importances:
        # Convert to DataFrame for easier plotting
        df_imp = pd.DataFrame(feature_importances)
        
        # Take top 15 features to avoid clutter
        df_imp = df_imp.head(15)
        
        # Sort ascending for horizontal bar chart (so biggest is at top)
        df_imp = df_imp.sort_values(by="importance", ascending=True)

        fig = px.bar(
            df_imp, 
            x="importance", 
            y="feature", 
            orientation="h",
            title="Top Feature Importances",
            labels={"importance": "Relative Importance", "feature": ""},
            text_auto='.1%'
        )
        
        fig.update_layout(
            xaxis_tickformat='.0%',
            yaxis_title=None,
            margin=dict(l=0, r=0, t=40, b=0),
            height=300 + (len(df_imp) * 15) # Dynamic height
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Feature importance data is missing or not supported by this model.")
