import streamlit as st
import pandas as pd
import time
from PIL import Image
import random

# Set page configuration
st.set_page_config(
    page_title="Climate Research Paper Finder",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: white !important;
    }
    .sub-header {
        font-size: 1rem !important;
        font-weight: 400 !important;
        color: #d1fae5 !important;
    }
    .paper-title {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #115e59 !important;
    }
    .paper-info {
        font-size: 0.9rem !important;
        color: #6b7280 !important;
    }
    .support-score-high {
        background-color: #d1fae5;
        color: #047857;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        font-weight: 600;
    }
    .support-score-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        font-weight: 600;
    }
    .summary-box {
        background-color: #ecfdf5;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #a7f3d0;
    }
    .stButton>button {
        background-color: #0f766e;
        color: white;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #134e4a;
    }
    .secondary-button>button {
        background-color: #f3f4f6;
        color: #1f2937;
    }
    .secondary-button>button:hover {
        background-color: #e5e7eb;
    }
    footer {
        text-align: center;
        padding: 1rem;
        background-color: #1f2937;
        color: white;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
header_container = st.container()
with header_container:
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown(
            '<p class="main-header">Climate Research Paper Finder</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sub-header">Find evidence-based research to support your climate change arguments</p>', unsafe_allow_html=True)

# Main content
main_container = st.container()
with main_container:
    # Search section
    st.markdown("### What climate argument are you researching?")

    search_query = st.text_area(
        "Enter your climate change question, argument, or research topic",
        placeholder="For example: 'Evidence that sea levels are rising faster in the past decade' or 'Impact of climate change on agricultural yields'",
        height=100,
        label_visibility="collapsed"
    )

    # Search options in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        source_type = st.selectbox(
            "Source Type:",
            ["All Sources", "Peer-Reviewed Only", "Recent Publications"],
        )
    with col2:
        date_range = st.selectbox(
            "Date Range:",
            ["Any Time", "Last 5 Years", "Last 10 Years", "Custom Range"],
        )
    with col3:
        relevance = st.selectbox(
            "Relevance:",
            ["High Support", "High Citations", "Most Recent"],
        )

    # Search button
    search_clicked = st.button("🔍 Find Evidence")

    # Sample data for demonstration
    sample_papers = [
        {
            "id": 1,
            "title": "Global temperature changes and their impact on agricultural yields",
            "authors": "Zhang, J., Smith, K., Patel, R.",
            "journal": "Nature Climate Change",
            "year": 2023,
            "relevance": 98,
            "abstract": "This study examines the correlation between rising global temperatures and decreasing agricultural yields across various climate zones. Our findings indicate a significant negative impact on staple crop production in tropical and subtropical regions.",
            "citations": 145,
            "support_score": 97
        },
        {
            "id": 2,
            "title": "Sea level rise projections for coastal urban planning",
            "authors": "Rodriguez, M., Johnson, T., Lee, S.",
            "journal": "Environmental Research Letters",
            "year": 2024,
            "relevance": 91,
            "abstract": "We present updated projections for sea level rise through 2100 based on the latest climate models. Our research demonstrates that many coastal cities will face increased flooding risks much sooner than previously anticipated.",
            "citations": 87,
            "support_score": 93
        },
        {
            "id": 3,
            "title": "Carbon sequestration potential of reforestation in temperate zones",
            "authors": "Anderson, P., Williams, H., Chen, Y.",
            "journal": "Global Change Biology",
            "year": 2022,
            "relevance": 85,
            "abstract": "This research quantifies the carbon sequestration capacity of reforestation efforts in temperate regions. We found that strategic reforestation could offset up to 15% of current annual carbon emissions in these areas.",
            "citations": 118,
            "support_score": 89
        }
    ]

    # Function to simulate paper search
    def search_papers(query):
        with st.spinner('Searching for relevant research papers...'):
            # Simulate API call with a delay
            time.sleep(2)
            return sample_papers

    # Display results when search is clicked
    if search_clicked and search_query:
        results = search_papers(search_query)

        # Results summary
        st.markdown("### Research Results")
        st.markdown(f"""
        <div class="summary-box">
            <h4>Evidence Summary</h4>
            <p>Found {len(results)} high-quality research papers supporting your query on 
            <strong>"{search_query}"</strong>. These studies collectively demonstrate 
            strong scientific consensus on this topic, with an average of 116 citations 
            per paper and publication dates ranging from 2022-2024.</p>
        </div>
        """, unsafe_allow_html=True)

        # Display each research paper
        for paper in results:
            with st.container():
                st.markdown(f"""
                <div style="border: 1px solid #e5e7eb; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <p class="paper-title">{paper['title']}</p>
                        <span class="support-score-high">👍 {paper['support_score']}% Support</span>
                    </div>
                    <p class="paper-info">
                        {paper['authors']} • {paper['journal']}, {paper['year']} • 🏆 {paper['citations']} citations
                    </p>
                    <p style="margin-top: 0.75rem; font-size: 0.95rem;">
                        {paper['abstract']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Action buttons
                col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
                with col1:
                    st.button("📄 View Full Paper", key=f"view_{paper['id']}")
                with col2:
                    st.markdown('<div class="secondary-button">',
                                unsafe_allow_html=True)
                    st.button("📚 Key Findings", key=f"findings_{paper['id']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col3:
                    st.markdown('<div class="secondary-button">',
                                unsafe_allow_html=True)
                    st.button("Save", key=f"save_{paper['id']}")
                    st.markdown('</div>', unsafe_allow_html=True)

        # Pagination
        st.markdown("""
        <div style="display: flex; justify-content: center; margin-top: 1rem;">
            <button style="padding: 0.25rem 0.5rem; border: 1px solid #d1d5db; border-radius: 0.25rem; margin-right: 0.25rem;">Previous</button>
            <button style="padding: 0.25rem 0.5rem; background-color: #0f766e; color: white; border: none; border-radius: 0.25rem; margin-right: 0.25rem;">1</button>
            <button style="padding: 0.25rem 0.5rem; border: 1px solid #d1d5db; border-radius: 0.25rem; margin-right: 0.25rem;">2</button>
            <button style="padding: 0.25rem 0.5rem; border: 1px solid #d1d5db; border-radius: 0.25rem; margin-right: 0.25rem;">3</button>
            <button style="padding: 0.25rem 0.5rem; border: 1px solid #d1d5db; border-radius: 0.25rem;">Next</button>
        </div>
        """, unsafe_allow_html=True)

    # Display empty state when no search has been performed
    elif search_clicked and not search_query:
        st.warning(
            "Please enter a search query to find relevant research papers.")

# Footer
st.markdown("""
<footer>
    <p>Climate Research Paper Finder © 2025 | Powered by NLP and RAG technology</p>
    <p style="font-size: 0.8rem; color: #9ca3af; margin-top: 0.5rem;">
        Data sources: arXiv, Nature, Environmental Research Letters, and other academic repositories
    </p>
</footer>
""", unsafe_allow_html=True)

# Sidebar with additional options
with st.sidebar:
    st.header("Advanced Options")

    st.subheader("Filter Papers")
    min_citations = st.slider("Minimum Citations", 0, 200, 50)
    publication_years = st.slider(
        "Publication Years", 2000, 2025, (2015, 2025))

    st.subheader("Export Options")
    st.download_button(
        label="Export Results as CSV",
        data=pd.DataFrame(sample_papers).to_csv(index=False),
        file_name="climate_research_results.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("About This Tool")
    st.markdown("""
    This application uses advanced NLP and RAG (Retrieval-Augmented Generation) 
    techniques to find relevant climate change research papers that support your arguments or answer your questions.
    
    **How it works:**
    1. Enter your climate-related question or argument
    2. Our AI analyzes your query to understand your intent
    3. The system searches through thousands of peer-reviewed papers
    4. Results are ranked by relevance and support for your query
    """)
