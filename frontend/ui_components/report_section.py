import streamlit as st
import requests

def render_report_section(session_id: str, api_url: str = "http://localhost:8000"):
    st.header("📄 Consultant Report")
    st.write("Generate a professional business report based on your recent analysis.")
    
    format_option = st.radio("Output Format", ["markdown", "pdf"], horizontal=True)
    
    if st.button("Generate Report", type="primary"):
        with st.spinner("Compiling insights into a presentation-ready report..."):
            try:
                payload = {
                    "session_id": session_id,
                    "output_format": format_option
                    # Include data dicts here if not handled natively by backend session
                }
                
                response = requests.post(f"{api_url}/api/report/generate", json=payload)
                response.raise_for_status()
                data = response.json()
                
                st.success("Report generated successfully!")
                
                if "markdown" in data:
                    st.markdown("### Preview")
                    with st.expander("Review Generated Report", expanded=True):
                        st.markdown(data["markdown"])
                
                if format_option == "pdf":
                    if "pdf_path" in data:
                        st.info(f"PDF successfully exported to your backend server.")
                        # Serve the PDF for download to the user
                        with open(data["pdf_path"], "rb") as f:
                            st.download_button(
                                label="📥 Download Consultant PDF",
                                data=f,
                                file_name="AI_Data_Copilot_Report.pdf",
                                mime="application/pdf"
                            )
                    elif "pdf_error" in data:
                        st.error(f"Failed to generate PDF: {data['pdf_error']}")
                        
            except Exception as e:
                st.error(f"Error communicating with backend: {str(e)}")
