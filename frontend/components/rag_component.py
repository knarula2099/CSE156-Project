# frontend/components/rag_component.py

import streamlit as st
import pandas as pd
from typing import Dict, List, Any
import plotly.express as px
import time
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.services.rag_service import ClimateRAGService

# Initialize the RAG service
@st.cache_resource
def get_rag_service():
    """Get or create the RAG service"""
    return ClimateRAGService(
        chroma_path="./chroma_db",
        collection_name="climate_research",
        cache_dir="./data/processed/cache"
    )

def render_papers(documents: List[str], metadata: List[Dict]):
    """
    Render retrieved papers in the UI.
    
    Args:
        documents: List of document texts
        metadata: List of document metadata
    """
    if not documents:
        st.warning("No relevant papers found.")
        return
    
    st.subheader("Retrieved Research Papers")
    
    for i, (doc, meta) in enumerate(zip(documents, metadata)):
        with st.expander(f"{i+1}. {meta.get('title', 'Untitled Paper')}"):
            st.markdown(f"**Authors:** {meta.get('authors', 'Unknown')}")
            st.markdown(f"**Year:** {meta.get('year', meta.get('published', 'Unknown'))}")
            st.markdown(f"**Abstract:** {doc}")
            if 'link' in meta and meta['link']:
                st.markdown(f"[View Paper]({meta['link']})")

def render_response(response: str):
    """
    Render the generated response in the UI.
    
    Args:
        response: Generated response text
    """
    st.subheader("Answer")
    st.markdown(response)

def render_evaluation(evaluation: Dict):
    """
    Render retrieval evaluation metrics.
    
    Args:
        evaluation: Dictionary of evaluation metrics
    """
    if not evaluation:
        return
    
    st.subheader("Search Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Retrieved Papers", evaluation.get("num_results", 0))
    
    with col2:
        # Format as percentage
        relevance = evaluation.get("relevance_estimate", 0) * 100
        st.metric("Relevance Score", f"{relevance:.1f}%")
    
    with col3:
        st.metric("Year Span", evaluation.get("temporal_diversity", 0))

def rag_search_ui():
    """Render the RAG search UI component"""
    st.title("Climate Research Assistant")
    
    st.markdown("""
    Ask questions about climate change research, and I'll find relevant papers and generate a 
    response based on the scientific literature.
    """)
    
    # Get the RAG service
    rag_service = get_rag_service()
    
    # Query input
    query = st.text_input("Enter your question:", 
                        placeholder="e.g., How does climate change affect agricultural yields?")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        top_k = st.slider("Number of papers:", min_value=3, max_value=10, value=5)
    
    with col2:
        use_openai = st.checkbox("Use OpenAI for response", value=True)
    
    with col3:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    # Advanced options in expander
    with st.expander("Advanced Options"):
        model = st.selectbox(
            "OpenAI Model:",
            options=["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o", "gpt-4"],
            index=0
        )
        
        use_cache = st.checkbox("Use cache (faster for repeated queries)", value=True)
        use_hybrid = st.checkbox("Use hybrid search (semantic + keyword)", value=True)
    
    # Process search
    if search_button and query:
        with st.spinner("Searching and generating response..."):
            # Perform RAG
            result = rag_service.perform_rag(
                query=query,
                top_k=top_k,
                use_openai=use_openai,
                use_cache=use_cache,
                model=model
            )
            
            # Display processing time
            st.caption(f"Processing time: {result['processing_time']:.2f} seconds" + 
                      (" (from cache)" if result.get("from_cache") else ""))
            
            # Display error if any
            if result.get("error"):
                st.error(f"Error: {result['error']}")
            else:
                # Display response
                if result.get("response"):
                    render_response(result["response"])
                
                # Show evaluation metrics
                if result.get("evaluation"):
                    render_evaluation(result["evaluation"])
                
                # Display retrieved papers
                render_papers(result.get("documents", []), result.get("metadata", []))
                
                # Add user feedback section
                with st.expander("Provide Feedback"):
                    feedback_quality = st.select_slider(
                        "How useful was this response?",
                        options=["Not useful", "Somewhat useful", "Useful", "Very useful", "Extremely useful"],
                        value="Useful"
                    )
                    
                    feedback_text = st.text_area("Additional feedback (optional):", 
                                               placeholder="What could be improved?")
                    
                    if st.button("Submit Feedback"):
                        # Save feedback
                        feedback = {
                            "query": query,
                            "quality": feedback_quality,
                            "comments": feedback_text,
                            "timestamp": time.time()
                        }
                        
                        # In a real app, you would store this feedback
                        st.session_state.setdefault("feedback", []).append(feedback)
                        st.success("Thank you for your feedback!")

def rag_analytics_ui():
    """Render analytics for the RAG system"""
    st.title("RAG System Analytics")
    
    # Get the RAG service
    rag_service = get_rag_service()
    
    # Get cached queries
    cache = rag_service.query_cache
    
    if not cache:
        st.info("No data available yet. Try searching for some queries first.")
        return
    
    # Create DataFrame from cache
    cache_data = []
    for key, value in cache.items():
        cache_data.append({
            "query": value.get("query", "Unknown"),
            "timestamp": value.get("timestamp", 0),
            "num_results": len(value.get("documents", [])),
            "relevance": value.get("evaluation", {}).get("relevance_estimate", 0),
            "processing_time": value.get("processing_time", 0)
        })
    
    cache_df = pd.DataFrame(cache_data)
    
    # Convert timestamp to datetime
    if "timestamp" in cache_df.columns:
        cache_df["date"] = pd.to_datetime(cache_df["timestamp"], unit="s")
    
    # Display summary metrics
    st.subheader("Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Queries", len(cache_df))
    
    with col2:
        avg_relevance = cache_df["relevance"].mean() * 100 if "relevance" in cache_df.columns else 0
        st.metric("Avg Relevance Score", f"{avg_relevance:.1f}%")
    
    with col3:
        avg_time = cache_df["processing_time"].mean() if "processing_time" in cache_df.columns else 0
        st.metric("Avg Processing Time", f"{avg_time:.2f}s")
    
    # Display charts
    st.subheader("Visualizations")
    
    tab1, tab2 = st.tabs(["Performance Metrics", "Recent Queries"])
    
    with tab1:
        if "relevance" in cache_df.columns and "processing_time" in cache_df.columns:
            fig = px.scatter(
                cache_df,
                x="relevance",
                y="processing_time",
                size="num_results",
                hover_data=["query"],
                title="Query Performance",
                labels={
                    "relevance": "Relevance Score",
                    "processing_time": "Processing Time (s)",
                    "num_results": "Number of Results"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if "date" in cache_df.columns:
            # Sort by date descending
            recent_df = cache_df.sort_values("date", ascending=False).head(10)
            
            # Display recent queries
            st.dataframe(
                recent_df[["query", "date", "num_results", "relevance"]].rename(
                    columns={
                        "query": "Query",
                        "date": "Date",
                        "num_results": "Results",
                        "relevance": "Relevance"
                    }
                ),
                use_container_width=True
            )