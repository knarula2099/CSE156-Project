# scripts/vertex_rag.py

import os
import sys
import argparse
import json
from datetime import datetime
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

# Third-party imports
import chromadb
from sentence_transformers import SentenceTransformer

# Google Cloud imports
from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextGenerationModel

# Load environment variables
load_dotenv()

class VertexClimateRAG:
    """Climate Research RAG System using Google Vertex AI"""
    
    def __init__(self, chroma_path: str = "./chroma_db", collection_name: str = "climate_research", 
                project_id: str = None, location: str = "us-central1"):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.project_id = project_id
        self.location = location
        
        # Initialize components
        self._init_vector_db()
        self._init_embedding_model()
        self._init_vertex_ai()
        
        print(f"Vertex RAG System initialized with collection '{collection_name}'")
    
    def _init_vector_db(self):
        """Initialize the vector database connection"""
        try:
            self.client = chromadb.PersistentClient(path=self.chroma_path)
            try:
                self.collection = self.client.get_collection(self.collection_name)
                print(f"Connected to existing collection with {self.collection.count()} documents")
            except ValueError:
                print(f"Collection '{self.collection_name}' not found. Trying other collection names...")
                # Try to find any collection
                collections = self.client.list_collections()
                if collections:
                    self.collection_name = collections[0].name
                    self.collection = self.client.get_collection(self.collection_name)
                    print(f"Using collection '{self.collection_name}' with {self.collection.count()} documents")
                else:
                    print("No collections found. Creating an empty collection.")
                    self.collection = self.client.create_collection(self.collection_name)
        except Exception as e:
            print(f"Error initializing vector database: {e}")
            raise
    
    def _init_embedding_model(self):
        """Initialize the embedding model"""
        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("Embedding model loaded")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise
    
    def _init_vertex_ai(self):
        """Initialize the Vertex AI client"""
        try:
            # Check for project ID
            if not self.project_id:
                # Try to get from environment
                self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            
            if not self.project_id:
                print("Warning: No project ID provided. Please set GOOGLE_CLOUD_PROJECT environment variable or provide project_id parameter.")
                self.vertex_ai_available = False
                return
            
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.location)
            self.vertex_ai_available = True
            
            # Load the model
            self.generation_model = TextGenerationModel.from_pretrained("text-bison@002")
            print(f"Vertex AI initialized with project {self.project_id}")
            
        except Exception as e:
            print(f"Error initializing Vertex AI: {e}")
            self.vertex_ai_available = False
    
    def preprocess_query(self, query: str) -> str:
        """Clean and enhance the query for better retrieval"""
        # Convert to lowercase
        query = query.lower()
        
        # Remove excessive punctuation and normalize spacing
        query = re.sub(r'[^\w\s]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Expand common climate abbreviations
        abbreviations = {
            "ghg": "greenhouse gas",
            "ipcc": "intergovernmental panel on climate change",
            "co2": "carbon dioxide",
            "co₂": "carbon dioxide",
            "ch4": "methane",
            "slr": "sea level rise"
        }
        
        for abbr, full in abbreviations.items():
            pattern = r'\b' + abbr + r'\b'
            query = re.sub(pattern, full, query, flags=re.IGNORECASE)
        
        return query
    
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List, List]:
        """Retrieve relevant documents for a query"""
        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(processed_query).tolist()
        
        # Retrieve documents
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            if not results["documents"] or not results["documents"][0]:
                print(f"No results found for query: {query}")
                return [], []
            
            return results["documents"][0], results["metadatas"][0]
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return [], []
    
    def generate_response(self, query: str, documents: List[str], metadatas: List[Dict]) -> str:
        """Generate a response using Vertex AI"""
        if not self.vertex_ai_available:
            return "Vertex AI client not available. Please check your Google Cloud credentials."
        
        if not documents:
            return "No relevant documents found to answer your question."
        
        try:
            # Create prompt
            system_instruction = "You are a climate research assistant that provides detailed, evidence-based answers. Base your answers on the provided research papers and include specific citations using the [1], [2] format."
            
            user_prompt = f"Question: {query}\n\nBased on these research papers:\n\n"
            
            # Add documents with citation numbers
            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                title = meta.get("title", "Untitled")
                authors = meta.get("authors", "Unknown")
                year = meta.get("year", meta.get("published", "Unknown"))
                
                user_prompt += f"[{i+1}] {title} ({year}) by {authors}\n"
                user_prompt += f"Abstract: {doc}\n\n"
            
            user_prompt += "Please provide a comprehensive answer with citations to the specific papers [1], [2], etc."
            
            # Combine system instruction and user prompt
            full_prompt = f"{system_instruction}\n\n{user_prompt}"
            
            # Generate response
            response = self.generation_model.predict(
                prompt=full_prompt,
                temperature=0.2,
                max_output_tokens=1024,
                top_k=40,
                top_p=0.8,
            )
            
            return response.text
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {e}"
    
    def create_fallback_response(self, query: str, documents: List[str], metadatas: List[Dict]) -> str:
        """Create a fallback response when Vertex AI is not available"""
        if not documents:
            return "No relevant documents found to answer your question."
        
        response = [f"Here's what I found about '{query}':\n"]
        
        # Extract key points from documents
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            title = meta.get("title", "Untitled")
            year = meta.get("year", meta.get("published", "Unknown"))
            
            # Extract first sentence as key finding
            first_sentence = doc.split('.')[0] + '.'
            
            response.append(f"[{i+1}] {title} ({year}):")
            response.append(f"- {first_sentence}")
            response.append("")
        
        return "\n".join(response)
    
    def perform_rag(self, query: str, top_k: int = 5, use_vertex: bool = True) -> Dict[str, Any]:
        """Perform the complete RAG pipeline"""
        # Start timer
        start_time = time.time()
        
        # Initialize result dictionary
        result = {
            "query": query,
            "documents": [],
            "metadata": [],
            "response": None,
            "error": None,
            "processing_time": 0
        }
        
        try:
            # Retrieve documents
            print(f"Retrieving top {top_k} relevant papers...")
            documents, metadatas = self.retrieve(query, top_k=top_k)
            result["documents"] = documents
            result["metadata"] = metadatas
            print(f"Retrieved {len(documents)} papers")
            
            # Generate response
            if use_vertex and self.vertex_ai_available:
                print("Generating response using Vertex AI...")
                result["response"] = self.generate_response(query, documents, metadatas)
            else:
                print("Creating fallback response...")
                result["response"] = self.create_fallback_response(query, documents, metadatas)
            
        except Exception as e:
            print(f"Error in perform_rag: {e}")
            result["error"] = str(e)
        
        # Calculate processing time
        result["processing_time"] = time.time() - start_time
        
        return result

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
        output.append(f"\nProcessing time: {result['processing_time']:.2f}s")
    
    return "\n".join(output)

def interactive_mode(project_id=None):
    """Run the RAG system in interactive mode"""
    print("\n" + "=" * 80)
    print("CLIMATE RESEARCH RAG SYSTEM - INTERACTIVE MODE")
    print("=" * 80)
    print("Type your questions about climate research and press Enter.")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("=" * 80)
    
    # Initialize RAG service
    try:
        rag_system = VertexClimateRAG(project_id=project_id)
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        return
    
    while True:
        query = input("\nYour question: ")
        
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting interactive mode. Goodbye!")
            break
        
        if not query:
            print("Please enter a question or type 'exit' to quit.")
            continue
        
        # Process query
        print(f"Processing query: '{query}'...")
        result = rag_system.perform_rag(query)
        
        # Display result
        print(format_result(result))

def main():
    parser = argparse.ArgumentParser(description="Climate Research RAG System with Vertex AI")
    parser.add_argument("--query", type=str, help="Run a single query and exit")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve (default: 5)")
    parser.add_argument("--no_vertex", action="store_true", help="Don't use Vertex AI for response generation")
    parser.add_argument("--project_id", type=str, help="Google Cloud project ID")
    
    args = parser.parse_args()
    
    if args.query:
        try:
            # Initialize RAG system
            rag_system = VertexClimateRAG(project_id=args.project_id)
            
            # Run a single query
            result = rag_system.perform_rag(
                query=args.query,
                top_k=args.top_k,
                use_vertex=not args.no_vertex
            )
            
            # Display result
            print(format_result(result))
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        # Run in interactive mode
        interactive_mode(project_id=args.project_id)

if __name__ == "__main__":
    main()