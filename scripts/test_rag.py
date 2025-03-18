# scripts/test_rag.py

import sys
import os
import argparse
import json
from datetime import datetime

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from backend.services.rag_service import ClimateRAGService

def format_result(result):
    """Format the RAG result for display in the terminal"""
    output = []
    
    # Add header
    output.append("\n" + "=" * 80)
    output.append(f"CLIMATE RESEARCH ASSISTANT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("=" * 80)
    
    # Add query
    output.append(f"Query: {result['query']}")
    output.append("")
    
    # Add response
    if result.get("response"):
        output.append("Response:")
        output.append("-" * 80)
        output.append(result["response"])
        output.append("-" * 80)
    
    # Add retrieved documents
    if result.get("documents"):
        output.append(f"\nRetrieved {len(result['documents'])} documents:")
        for i, (doc, meta) in enumerate(zip(result["documents"], result["metadata"])):
            output.append(f"\n[{i+1}] {meta.get('title', 'Untitled')} ({meta.get('year', meta.get('published', 'Unknown'))})")
            if meta.get("authors"):
                output.append(f"Authors: {meta['authors']}")
            output.append(f"Abstract: {doc[:300]}...")
    
    # Add processing info
    if result.get("processing_time"):
        output.append(f"\nProcessing time: {result['processing_time']:.2f}s" + 
                     (" (from cache)" if result.get("from_cache") else ""))
    
    return "\n".join(output)

def save_result(result, output_dir="./data/processed/rag_results"):
    """Save the RAG result to a file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    query_slug = result["query"].lower().replace(" ", "_")[:30]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{query_slug}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to file
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    
    return filepath

def run_test_query(query, top_k=5, use_openai=True, model="gpt-4o-mini"):
    """Run a test query through the RAG system"""
    # Initialize RAG service
    rag_service = ClimateRAGService(
        chroma_path="./chroma_db",
        collection_name="climate_research",
        cache_dir="./data/processed/cache"
    )
    
    # Perform RAG
    result = rag_service.perform_rag(
        query=query,
        top_k=top_k,
        use_openai=use_openai,
        model=model
    )
    
    return result

def interactive_mode():
    """Run the RAG system in interactive mode"""
    print("\n" + "=" * 80)
    print("CLIMATE RESEARCH RAG SYSTEM - INTERACTIVE MODE")
    print("=" * 80)
    print("Type your questions about climate research and press Enter.")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("Type 'save' to save the last result to a file.")
    print("=" * 80)
    
    # Initialize RAG service
    rag_service = ClimateRAGService(
        chroma_path="./chroma_db",
        collection_name="climate_research",
        cache_dir="./data/processed/cache"
    )
    
    last_result = None
    
    while True:
        query = input("\nYour question: ")
        
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting interactive mode. Goodbye!")
            break
        
        if query.lower() == "save" and last_result:
            filepath = save_result(last_result)
            print(f"Result saved to {filepath}")
            continue
        
        if not query:
            print("Please enter a question or type 'exit' to quit.")
            continue
        
        # Process query
        print(f"Processing query: '{query}'...")
        result = rag_service.perform_rag(query)
        last_result = result
        
        # Display result
        print(format_result(result))

def main():
    parser = argparse.ArgumentParser(description="Test the Climate Research RAG System")
    parser.add_argument("--query", type=str, help="Run a single query and exit")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve (default: 5)")
    parser.add_argument("--no_openai", action="store_true", help="Don't use OpenAI for response generation")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--save", action="store_true", help="Save the result to a file")
    
    args = parser.parse_args()
    
    if args.query:
        # Run a single query
        result = run_test_query(
            query=args.query,
            top_k=args.top_k,
            use_openai=not args.no_openai,
            model=args.model
        )
        
        # Display result
        print(format_result(result))
        
        # Save result if requested
        if args.save:
            filepath = save_result(result)
            print(f"Result saved to {filepath}")
    else:
        # Run in interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()