import streamlit as st
import requests
import os

# Backend API URL (adjust if needed)
API_URL = os.getenv("FAB_API_URL", "http://localhost:8000")

st.set_page_config(page_title="FAB Financial Analyst", layout="centered")
st.title("FAB Financial Analyst")

st.header("1. Upload Financial Documents (PDF)")
uploaded_files = st.file_uploader(
    "Upload one or more PDF files (Financial Statement, Earnings Presentation, or Results Call)",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Upload Documents") and uploaded_files:
    with st.spinner("Uploading and processing documents..."):
        files = [("files", (f.name, f, "application/pdf")) for f in uploaded_files]
        try:
            response = requests.post(f"{API_URL}/upload", files=files)
            if response.status_code == 200:
                results = response.json()
                for res in results:
                    if res.get("success"):
                        st.success(f"{res['filename']}: {res['message']}")
                    else:
                        st.error(f"{res['filename']}: {res['message']}")
            else:
                st.error(f"Upload failed: {response.text}")
        except Exception as e:
            st.error(f"Error: {e}")

st.header("2. Analyze a Financial Query")
query = st.text_area(
    "Enter your financial question (e.g., 'What was the year-over-year percentage change in Net Profit between Q3 2023 and Q3 2024?')",
    height=100
)

if st.button("Analyze Query") and query:
    with st.spinner("Analyzing query..."):
        try:
            payload = {"query": query}
            response = requests.post(f"{API_URL}/query", json=payload)
            if response.status_code == 200:
                data = response.json()
                st.subheader("Final Answer:")
                st.markdown(f"**{data.get('final_answer', 'No answer generated')}**")
                st.subheader("Sources Used:")
                sources = data.get("sources_used", [])
                if sources:
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:** {src}")
                else:
                    st.info("No sources returned.")
            else:
                st.error(f"Query failed: {response.text}")
        except Exception as e:
            st.error(f"Error: {e}")
