# frontend/components/rag_component.py
import streamlit as st
from streamlit.components.v1 import html
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
    st.title("🔍 Green Lens")
    
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
        use_gemini = st.checkbox("Use Gemini for response", value=True)
    
    with col3:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    # Advanced options in expander
    with st.expander("Advanced Options"):
        use_cache = st.checkbox("Use cache (faster for repeated queries)", value=True)
        use_hybrid = st.checkbox("Use hybrid search (semantic + keyword)", value=True)
    
    # Process search
    if search_button and query:
        with st.spinner("Searching and generating response..."):
            # Perform RAG
            result = rag_service.perform_rag(
                query=query,
                top_k=top_k,
                use_gemini=use_gemini,
                use_cache=use_cache
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
def query_workflow_ui():
    """Render the query workflow UI component"""
    st.title("Query Workflow")
    
    # Placeholder for future implementation
    # Your Mermaid diagram code
    st.markdown("""
            [![](https://mermaid.ink/img/pako:eNp9Vm1v2zgM_iuGhx02oBkSb0vX4HCHrhm64pJDt2wDds59UG0qEeq3SXK3oO1_HyVatux48webLw8lkiIp34dJmUK4CHlWfk_2TOrg03JbBPio-mYnWbUPPiuQ8TY0n-Cq0CA5S2Ab_k8w83y-ijdaAsszoYPzqvJUH2qQh9jaWhJXqGrtAT6CqjOt4uYbLIWqMnZoEFCkRAx8umDJHtAp-w02B6Uh77lkNyMU7WtpD2D5VVne1tU9rfJe6L8fB4CNLiXE9u089T0bePXx_PJaVJCJwviGXODYnm8t3np2LcsElBLFDm3I107UszPPtYSKtLHDOgHCB-B3PypWpA3QMkqUxRCU30CLMXTaX8gLtB8saCngjmUmVEePHYV5vkCCOdwAk8k-JiYgbgD8Bw7fS5k2yIYbh74_3EiRxvQZnE5XXZIVtzF9MK7gDyywO5BKcJEw3c_GryK9hALtDRhD7ZijIC9K7I4fegMZBhg3XEDscd7xkPNKX2DbaIiJMSsoLesx-CXkohDn11cxUQGSR8GqCheA2BGe58Mwf1XCptTZzpRvQ_WivNijn2z5Nn7mqOeedlXuqF-e2THBbBhGqp7_pmnWOIAyhRsS0dtvA7hQkcAnPD7FS5njJGJZNlljAlbryWo-uYt8B1breGezM4leTCc8Y2o_wZkEx9s_fRqsmSgCM_dIQh0wmfzljwZSeQIDePgK6sEV3Dji3_LB60x_hHVSuxe1KKmIJrFpRN_MCqzKb6Whrtc9pBxJ4fgO_roWQY1Fyt7CR1pXDMGf4w7Su2lUA6F-JDHRlHi_gXzTnsJC_e5xie0kFtJ2DOmxOMbE9G6FjXvUP87Bppva2rBV7h083Q9G3V07I7a9imnLEPtjZyZTV4k9I9dUA6uNPmTmfrB8goWulsCDGu_Z9ooOuMiyxROY81POT3CulLeweDJ7czblfGCX2Cjs6G6s-JxzSFur11HyasYGVt_c9eVsOD-FeWvD2ZtkNh_YyPamaG1mfNrZvI6iKB3Y7Nop5ozO-NQL6TSKIJ0NjBSNry4JnN90SXiZREk0sMjt_HFbJBh_FwubpmezmX8G1sz-G_Wz7mvpv8JLrq8cXP5dNn1Qd7G2ifPV3Xz3kuQDmiHusuGraNw2UYcnIQ6GnIkU_wXvDWwb6j3kOP4XSKbAGZbuNtwWjwhltS43hyIJF3hZwUkoy3q3DxecZQq5ukqxBZeC4XzPWymOtv_KsuMhFejTmv4-7U_o408fOWUl?type=png)](https://mermaid.live/edit#pako:eNp9Vm1v2zgM_iuGhx02oBkSb0vX4HCHrhm64pJDt2wDds59UG0qEeq3SXK3oO1_HyVatux48webLw8lkiIp34dJmUK4CHlWfk_2TOrg03JbBPio-mYnWbUPPiuQ8TY0n-Cq0CA5S2Ab_k8w83y-ijdaAsszoYPzqvJUH2qQh9jaWhJXqGrtAT6CqjOt4uYbLIWqMnZoEFCkRAx8umDJHtAp-w02B6Uh77lkNyMU7WtpD2D5VVne1tU9rfJe6L8fB4CNLiXE9u089T0bePXx_PJaVJCJwviGXODYnm8t3np2LcsElBLFDm3I107UszPPtYSKtLHDOgHCB-B3PypWpA3QMkqUxRCU30CLMXTaX8gLtB8saCngjmUmVEePHYV5vkCCOdwAk8k-JiYgbgD8Bw7fS5k2yIYbh74_3EiRxvQZnE5XXZIVtzF9MK7gDyywO5BKcJEw3c_GryK9hALtDRhD7ZijIC9K7I4fegMZBhg3XEDscd7xkPNKX2DbaIiJMSsoLesx-CXkohDn11cxUQGSR8GqCheA2BGe58Mwf1XCptTZzpRvQ_WivNijn2z5Nn7mqOeedlXuqF-e2THBbBhGqp7_pmnWOIAyhRsS0dtvA7hQkcAnPD7FS5njJGJZNlljAlbryWo-uYt8B1breGezM4leTCc8Y2o_wZkEx9s_fRqsmSgCM_dIQh0wmfzljwZSeQIDePgK6sEV3Dji3_LB60x_hHVSuxe1KKmIJrFpRN_MCqzKb6Whrtc9pBxJ4fgO_roWQY1Fyt7CR1pXDMGf4w7Su2lUA6F-JDHRlHi_gXzTnsJC_e5xie0kFtJ2DOmxOMbE9G6FjXvUP87Bppva2rBV7h083Q9G3V07I7a9imnLEPtjZyZTV4k9I9dUA6uNPmTmfrB8goWulsCDGu_Z9ooOuMiyxROY81POT3CulLeweDJ7czblfGCX2Cjs6G6s-JxzSFur11HyasYGVt_c9eVsOD-FeWvD2ZtkNh_YyPamaG1mfNrZvI6iKB3Y7Nop5ozO-NQL6TSKIJ0NjBSNry4JnN90SXiZREk0sMjt_HFbJBh_FwubpmezmX8G1sz-G_Wz7mvpv8JLrq8cXP5dNn1Qd7G2ifPV3Xz3kuQDmiHusuGraNw2UYcnIQ6GnIkU_wXvDWwb6j3kOP4XSKbAGZbuNtwWjwhltS43hyIJF3hZwUkoy3q3DxecZQq5ukqxBZeC4XzPWymOtv_KsuMhFejTmv4-7U_o408fOWUl)
            """)