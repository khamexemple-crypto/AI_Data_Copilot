import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_metric_card(title: str, value: str, delta: str = None):
    """Renders a nice metric card."""
    st.metric(label=title, value=value, delta=delta)

def render_leaderboard(ml_results: dict):
    """Renders the AutoML leaderboard."""
    if not ml_results or "leaderboard" not in ml_results:
        st.info("No model leaderboard available yet. Run the ML pipeline.")
        return
        
    df_lb = pd.DataFrame(ml_results["leaderboard"])
    st.dataframe(df_lb, use_container_width=True, hide_index=True)

def render_charts(charts: list):
    """Renders a list of Plotly figures or descriptions."""
    if not charts:
        st.info("No visualizations available.")
        return
        
    cols = st.columns(2)
    for i, chart in enumerate(charts):
        col = cols[i % 2]
        with col:
            st.subheader(chart.get("title", f"Chart {i+1}"))
            st.write(chart.get("description", ""))
            # If actual plotly JSON is passed, render it
            if "plotly_json" in chart:
                try:
                    fig = go.Figure(chart["plotly_json"])
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not render chart: {e}")
