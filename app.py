# dummy_app.py (rename to app.py if that's your preference)

import streamlit as st
import requests
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import components
from frontend.components.rag_component import rag_search_ui, rag_analytics_ui
from frontend.pages.Discussion import discussion_ui

API_URL = "http://127.0.0.1:8000"

# Page configuration
st.set_page_config(
    page_title="Climate Research Assistant",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .paper-card {
        background-color: #f8f9fa;
        border-left: 3px solid #1E88E5;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .response-box {
        background-color: #e3f2fd;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.logo("green-lens-logo.png", size="large")
    st.title("Green Lens 🌍")

    # About section
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    This application uses RAG (Retrieval Augmented Generation) to provide evidence-based answers
    to your climate research questions.
    
    Data sourced from scientific research papers on climate change.
    """)

# # Main content
# if page == "Search":
#     rag_search_ui()
# elif page == "Analytics":
#     rag_analytics_ui()
# elif page == "Discussion":
#     discussion_ui()

page = st.navigation([rag_search_ui, rag_analytics_ui, discussion_ui])
page.run()