import streamlit as st
from backend.services.multi_dataset_manager import load_multiple_datasets, compare_datasets, suggest_merge_keys

def render_multi_file_section():
    """
    Renders a Streamlit UI component to handle multi-file uploads, 
    schema comparison, and merge suggestions.
    """
    st.markdown("---")
    st.subheader("📂 Multi-Dataset Management")
    st.caption("Upload, compare, inspect, and find merge links across multiple datasets.")
    
    uploaded_files = st.file_uploader(
        "Upload Multiple Datasets (CSV/Excel)", 
        type=["csv", "xlsx"], 
        accept_multiple_files=True,
        key="multi_file_uploader"
    )
    
    if uploaded_files:
        if st.button("Load & Compare Datasets", type="primary"):
            with st.spinner("Processing datasets..."):
                # Load datasets into memory
                datasets = load_multiple_datasets(uploaded_files)
                st.session_state["multi_datasets"] = datasets
                st.success(f"Successfully loaded {len(datasets)} dataset(s) into memory!")
    
    # If datasets are loaded in the session state, display management UI
    if "multi_datasets" in st.session_state and st.session_state["multi_datasets"]:
        datasets = st.session_state["multi_datasets"]
        
        tab_compare, tab_merge, tab_inspect = st.tabs([
            "📊 Schema Comparison", 
            "🔗 Merge Suggestions", 
            "🔍 Quick Inspect"
        ])
        
        with tab_compare:
            st.write("### Schema Overview")
            comparison = compare_datasets(datasets)
            for name, meta in comparison.items():
                with st.expander(f"📄 {name} ({meta['rows']:,} rows, {meta['columns']} cols)"):
                    st.json(meta['schema'])
                    
        with tab_merge:
            st.write("### Merge Suggestions")
            if len(datasets) >= 2:
                suggestions = suggest_merge_keys(datasets)
                for pair, details in suggestions.items():
                    if details.get("can_merge"):
                        st.success(f"**{pair}**: {details['message']}")
                    else:
                        st.warning(f"**{pair}**: {details['message']}")
            else:
                st.info("Upload at least 2 datasets to see automatic merge suggestions.")
                
        with tab_inspect:
            st.write("### Quick Data Preview")
            selected_ds = st.selectbox("Select dataset to view:", list(datasets.keys()))
            if selected_ds:
                st.dataframe(datasets[selected_ds].head(10), use_container_width=True)
