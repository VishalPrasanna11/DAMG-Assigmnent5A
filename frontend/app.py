# frontend/app.py
import streamlit as st
import requests
import json
from typing import List, Dict, Any

# API endpoint
API_URL = "http://localhost:8000"  # When running in Docker

st.title("NVIDIA Research Assistant")
st.write("Ask questions about NVIDIA's financial performance and get comprehensive research insights.")

# User input
query = st.text_input("Research Question", "What was NVIDIA's revenue growth in the past year?")

# Filtering options
col1, col2 = st.columns(2)
with col1:
    year = st.selectbox("Year", [None, 2020, 2021, 2022, 2023, 2024])
with col2:
    quarter = st.selectbox("Quarter", [None, 1, 2, 3, 4])

# Agent selection
st.write("Select Research Agents:")
col1, col2, col3 = st.columns(3)
with col1:
    use_rag = st.checkbox("Historical Data (RAG)", value=True)
with col2:
    use_snowflake = st.checkbox("Financial Metrics", value=True)
with col3:
    use_websearch = st.checkbox("Latest News & Trends", value=True)

# Generate report button
if st.button("Generate Research Report"):
    agents = []
    if use_rag:
        agents.append("rag")
    if use_snowflake:
        agents.append("snowflake")
    if use_websearch:
        agents.append("websearch")
        
    # Prepare request
    payload = {
        "query": query,
        "year": year,
        "quarter": quarter,
        "agents": agents
    }
    print(f"Payload for API: {json.dumps(payload)}")  # Debugging line to check payload
    
    with st.spinner("Generating research report..."):
        try:
            response = requests.post(f"{API_URL}/research", json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Display report sections
            st.subheader("Research Report")
            
            if "historical_data" in result and use_rag:
                st.markdown("### Historical Performance")
                st.write(result["historical_data"]["content"])
                
            if "financial_metrics" in result and use_snowflake:
                st.markdown("### Financial Metrics")
                st.write(result["financial_metrics"]["content"])
                # Display chart if available
                if "chart" in result["financial_metrics"]:
                    st.image(result["financial_metrics"]["chart"])
                    
            if "latest_insights" in result and use_websearch:
                st.markdown("### Latest Insights & Trends")
                st.write(result["latest_insights"]["content"])
                
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")