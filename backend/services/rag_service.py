# backend/services/rag_service.py

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime

import chromadb
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ClimateRAGService:
    """Service for Climate Research RAG (Retrieval Augmented Generation)"""
    
    def __init__(self, chroma_path: str = "./chroma_db", collection_name: str = "climate_research", 
                 embedding_model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "./data/processed/cache"):
        """
        Initialize the RAG service.
        
        Args:
            chroma_path: Path to ChromaDB
            collection_name: Name of the ChromaDB collection
            embedding_model_name: Name of the embedding model
            cache_dir: Directory to store cache files
        """
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize services
        self._init_vector_db()
        self._init_embedding_model()
        self._init_openai()
        
        # Initialize cache
        self.query_cache = self._load_cache()
        
        logger.info(f"ClimateRAGService initialized with collection '{collection_name}'")

    def _init_vector_db(self):
        """Initialize the vector database connection"""
        try:
            self.client = chromadb.PersistentClient(path=self.chroma_path)
            try:
                self.collection = self.client.get_collection(self.collection_name)
                logger.info(f"Connected to existing collection with {self.collection.count()} documents")
            except ValueError:
                logger.warning(f"Collection '{self.collection_name}' not found. Creating new collection.")
                self.collection = self.client.create_collection(self.collection_name)
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {str(e)}")
            raise

    def _init_embedding_model(self):
        """Initialize the embedding model"""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Embedding model '{self.embedding_model_name}' loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise

    def _init_openai(self):
        """Initialize the OpenAI client"""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            self.openai_client = None
        else:
            try:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                self.openai_client = None

    def _load_cache(self) -> Dict:
        """Load query cache from disk"""
        cache_file = os.path.join(self.cache_dir, "query_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                logger.info(f"Loaded cache with {len(cache)} entries")
                return cache
            except Exception as e:
                logger.error(f"Failed to load cache: {str(e)}")
                return {}
        return {}

    def _save_cache(self):
        """Save query cache to disk"""
        cache_file = os.path.join(self.cache_dir, "query_cache.json")
        try:
            # Trim cache if it's too large (keep most recent 1000 entries)
            if len(self.query_cache) > 1000:
                # Sort by timestamp and keep most recent 1000
                sorted_keys = sorted(self.query_cache.keys(), 
                                    key=lambda k: self.query_cache[k].get('timestamp', 0),
                                    reverse=True)
                trimmed_cache = {k: self.query_cache[k] for k in sorted_keys[:1000]}
                self.query_cache = trimmed_cache
            
            with open(cache_file, 'w') as f:
                json.dump(self.query_cache, f)
            logger.info(f"Saved cache with {len(self.query_cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {str(e)}")

    def preprocess_query(self, query: str) -> str:
        """
        Preprocess the user query to improve retrieval.
        
        Args:
            query: User's raw query
            
        Returns:
            Preprocessed query
        """
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
            "ch₄": "methane",
            "n2o": "nitrous oxide",
            "slr": "sea level rise",
            "ccs": "carbon capture and storage",
            "ev": "electric vehicle",
            "re": "renewable energy"
        }
        
        for abbr, full in abbreviations.items():
            pattern = r'\b' + abbr + r'\b'
            query = re.sub(pattern, full, query, flags=re.IGNORECASE)
        
        return query

    def expand_query(self, query: str) -> List[str]:
        """
        Expand the query with variations to improve retrieval.
        
        Args:
            query: Preprocessed user query
            
        Returns:
            List of query variations
        """
        variations = [query]
        
        # Add "climate change" if not in query and appropriate
        if "climate" not in query and "warming" not in query and "environment" not in query:
            variations.append(f"{query} climate change")
        
        # Add "effects of" or "impacts of" if query starts with certain words
        if not any(query.startswith(word) for word in ["effect", "impact", "consequence", "how"]):
            if "on" in query:
                parts = query.split("on", 1)
                variations.append(f"effects of {parts[0].strip()} on {parts[1].strip()}")
            else:
                variations.append(f"effects of {query}")
        
        return variations

    def retrieve(self, query: str, top_k: int = 5, use_hybrid: bool = True) -> Tuple[List, List]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            use_hybrid: Whether to use hybrid retrieval (semantic + keyword)
            
        Returns:
            Tuple of (documents, metadata)
        """
        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(processed_query).tolist()
        
        # Retrieve documents using semantic search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2  # Get more for diversity and filtering
        )
        
        if not results["documents"] or not results["documents"][0]:
            logger.warning(f"No results found for query: {query}")
            return [], []
        
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[]])[0]
        
        # If hybrid search is enabled, also try keyword search
        if use_hybrid and len(documents) < top_k:
            try:
                # Expand query with variations
                expanded_queries = self.expand_query(processed_query)
                for expanded_query in expanded_queries:
                    # Skip if it's the same as the original
                    if expanded_query == processed_query:
                        continue
                    
                    # Retrieve using expanded query
                    keyword_results = self.collection.query(
                        query_texts=[expanded_query],
                        n_results=top_k
                    )
                    
                    if keyword_results["documents"] and keyword_results["documents"][0]:
                        # Add new documents not already in results
                        for i, doc in enumerate(keyword_results["documents"][0]):
                            if doc not in documents:
                                documents.append(doc)
                                metadatas.append(keyword_results["metadatas"][0][i])
                                # Use a default distance/score since keyword results don't have comparable distances
                                distances.append(0.5)  # Middle score
            except Exception as e:
                logger.error(f"Error in hybrid search: {str(e)}")
        
        # Rerank and diversify results
        if documents:
            combined_results = list(zip(documents, metadatas, distances))
            
            # Sort by relevance (distance)
            combined_results.sort(key=lambda x: x[2])
            
            # Take top half based on pure relevance
            half_k = min(top_k // 2, len(combined_results))
            diversified_results = combined_results[:half_k]
            
            # Diversify the rest based on publication year and content
            remaining = combined_results[half_k:]
            
            # Group by year
            year_groups = {}
            for doc, meta, dist in remaining:
                year = meta.get("year", meta.get("published", "Unknown"))
                if year not in year_groups:
                    year_groups[year] = []
                year_groups[year].append((doc, meta, dist))
            
            # Get diverse results from different years, starting with most recent
            sorted_years = sorted(year_groups.keys(), reverse=True)
            remaining_slots = top_k - len(diversified_results)
            
            for year in sorted_years:
                if remaining_slots <= 0:
                    break
                if year_groups[year]:
                    # Sort year group by relevance and take the most relevant
                    year_group_sorted = sorted(year_groups[year], key=lambda x: x[2])
                    diversified_results.append(year_group_sorted[0])
                    remaining_slots -= 1
            
            # Unzip the combined results
            final_documents, final_metadatas, _ = zip(*diversified_results) if diversified_results else ([], [], [])
            
            return list(final_documents)[:top_k], list(final_metadatas)[:top_k]
        
        return [], []

    def select_context_chunks(self, documents: List[str], metadatas: List[Dict], 
                            max_tokens: int = 3000) -> Tuple[List[str], List[Dict]]:
        """
        Select the most informative chunks of documents to use as context.
        
        Args:
            documents: List of document texts
            metadatas: List of document metadata
            max_tokens: Maximum number of tokens to include in context
            
        Returns:
            Tuple of (selected document chunks, corresponding metadata)
        """
        if not documents:
            return [], []
        
        # Estimate tokens (rough approximation: 4 chars ≈ 1 token)
        doc_token_counts = [len(doc) // 4 for doc in documents]
        
        # If everything fits, return all
        if sum(doc_token_counts) <= max_tokens:
            return documents, metadatas
        
        # Otherwise, prioritize and select chunks
        selected_docs = []
        selected_meta = []
        token_count = 0
        
        # Always include the full first document (assumed most relevant)
        selected_docs.append(documents[0])
        selected_meta.append(metadatas[0])
        token_count += doc_token_counts[0]
        
        # For remaining documents, take introductory chunks
        for i in range(1, len(documents)):
            # Extract introduction (first paragraph or two)
            paragraphs = documents[i].split('\n\n')
            intro = paragraphs[0]
            if len(paragraphs) > 1:
                intro += '\n\n' + paragraphs[1]
            
            # Estimate tokens for intro
            intro_tokens = len(intro) // 4
            
            # If it fits, add it
            if token_count + intro_tokens <= max_tokens:
                selected_docs.append(intro)
                selected_meta.append(metadatas[i])
                token_count += intro_tokens
            else:
                # If even first paragraph doesn't fit, we're done
                break
        
        return selected_docs, selected_meta

    def create_prompt(self, query: str, documents: List[str], metadatas: List[Dict]) -> str:
        """
        Create a prompt for the LLM based on the query and retrieved documents.
        
        Args:
            query: User query
            documents: Retrieved document texts
            metadatas: Retrieved document metadata
            
        Returns:
            Formatted prompt
        """
        system_prompt = """You are a climate research assistant that provides detailed, evidence-based answers.
When responding to questions:
1. Base your answers on the provided research papers
2. Include specific citations using the [1], [2] format when referencing papers
3. Be accurate and nuanced - acknowledge uncertainties and limitations in the research
4. If the papers don't fully address the question, acknowledge this
5. Provide balanced views representing different scientific perspectives
6. Explain scientific concepts clearly for a general audience"""
        
        user_prompt = f"""Question: {query}

Based on these research papers:

"""
        
        # Add documents with citation numbers
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            title = meta.get("title", "Untitled")
            authors = meta.get("authors", "Unknown")
            year = meta.get("year", meta.get("published", "Unknown"))
            
            user_prompt += f"[{i+1}] {title} ({year}) by {authors}\n"
            user_prompt += f"Abstract: {doc}\n\n"
        
        user_prompt += """Please provide a comprehensive answer with citations to the specific papers [1], [2], etc.
If the information provided is insufficient, please state what is missing."""
        
        return {"system": system_prompt, "user": user_prompt}

    def generate_response(self, query: str, documents: List[str], metadatas: List[Dict], 
                         model: str = "gpt-4o-mini") -> str:
        """
        Generate a response using OpenAI's LLM.
        
        Args:
            query: User query
            documents: Retrieved document texts
            metadatas: Retrieved document metadata
            model: OpenAI model to use
            
        Returns:
            Generated response
        """
        if not self.openai_client:
            return "OpenAI client not available. Please check your API key and try again."
        
        if not documents:
            return "No relevant documents found to answer your question."
        
        try:
            # Select context chunks that fit within token limits
            context_docs, context_meta = self.select_context_chunks(documents, metadatas)
            
            # Create prompt
            prompt = self.create_prompt(query, context_docs, context_meta)
            
            # Generate response
            logger.info(f"Generating response using {model}")
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def create_fallback_response(self, query: str, documents: List[str], metadatas: List[Dict]) -> str:
        """
        Create a fallback response when OpenAI is not available.
        
        Args:
            query: User query
            documents: Retrieved document texts
            metadatas: Retrieved document metadata
            
        Returns:
            Fallback response
        """
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

    def evaluate_retrieval(self, query: str, documents: List[str], metadatas: List[Dict]) -> Dict[str, float]:
        """
        Calculate basic retrieval evaluation metrics.
        
        Args:
            query: User query
            documents: Retrieved documents
            metadatas: Retrieved metadata
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            "num_results": len(documents),
            "temporal_diversity": 0,
            "relevance_estimate": 0
        }
        
        if not documents:
            return metrics
        
        # Calculate temporal diversity (years spanned)
        years = []
        for meta in metadatas:
            year_str = meta.get("year", meta.get("published", ""))
            if isinstance(year_str, str):
                # Extract year from date string
                match = re.search(r'\b(19|20)\d{2}\b', year_str)
                if match:
                    years.append(int(match.group(0)))
            elif isinstance(year_str, (int, float)):
                years.append(int(year_str))
        
        if len(years) > 1:
            metrics["temporal_diversity"] = max(years) - min(years)
        
        # Simple relevance estimate based on query terms in documents
        query_terms = set(self.preprocess_query(query).split())
        relevance_scores = []
        
        for doc in documents:
            doc_lower = doc.lower()
            term_matches = sum(1 for term in query_terms if term in doc_lower)
            relevance_scores.append(term_matches / max(1, len(query_terms)))
        
        if relevance_scores:
            metrics["relevance_estimate"] = sum(relevance_scores) / len(relevance_scores)
        
        return metrics

    def log_interaction(self, query: str, result: Dict):
        """
        Log user interaction for analytics.
        
        Args:
            query: User query
            result: Result dictionary
        """
        log_file = os.path.join(self.cache_dir, "interactions.jsonl")
        
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "num_results": len(result.get("documents", [])),
            "has_response": result.get("response") is not None,
            "evaluation": result.get("evaluation", {})
        }
        
        # Append to log file
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log interaction: {str(e)}")

    def perform_rag(self, query: str, top_k: int = 5, use_openai: bool = True, 
                   use_cache: bool = True, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """
        Perform the complete RAG pipeline.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            use_openai: Whether to use OpenAI for response generation
            use_cache: Whether to use the query cache
            model: OpenAI model to use
            
        Returns:
            Dictionary with query results
        """
        # Start timer
        start_time = time.time()
        
        # Initialize result dictionary
        result = {
            "query": query,
            "documents": [],
            "metadata": [],
            "response": None,
            "error": None,
            "from_cache": False,
            "processing_time": 0
        }
        
        # Check cache
        cache_key = f"{query}_{top_k}_{use_openai}_{model}"
        if use_cache and cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            # Check if cache is recent (less than 1 day old)
            cache_time = cached_result.get("timestamp", 0)
            if time.time() - cache_time < 86400:  # 24 hours
                logger.info(f"Using cached result for query: {query}")
                cached_result["from_cache"] = True
                return cached_result
        
        try:
            # Retrieve documents
            documents, metadatas = self.retrieve(query, top_k=top_k)
            result["documents"] = documents
            result["metadata"] = metadatas
            
            # Generate response
            if use_openai and self.openai_client:
                result["response"] = self.generate_response(query, documents, metadatas, model=model)
            else:
                result["response"] = self.create_fallback_response(query, documents, metadatas)
            
            # Evaluate retrieval
            result["evaluation"] = self.evaluate_retrieval(query, documents, metadatas)
            
        except Exception as e:
            logger.error(f"Error in perform_rag: {str(e)}")
            result["error"] = str(e)
        
        # Calculate processing time
        result["processing_time"] = time.time() - start_time
        
        # Add timestamp
        result["timestamp"] = time.time()
        
        # Cache result
        if use_cache and not result["error"]:
            self.query_cache[cache_key] = result
            self._save_cache()
        
        # Log interaction
        self.log_interaction(query, result)
        
        return result