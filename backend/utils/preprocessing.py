# backend/utils/preprocessing.py

import re
from typing import List

def preprocess_query(query: str) -> str:
    """
    Clean and enhance the query for better retrieval.
    
    Args:
        query: Raw query string
        
    Returns:
        Processed query
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

def expand_query(query: str) -> List[str]:
    """
    Generate variations of the query to improve retrieval.
    
    Args:
        query: Preprocessed query
        
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

def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to a maximum length while preserving complete sentences.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Find the last sentence boundary before max_length
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    
    if last_period > 0:
        return truncated[:last_period + 1]
    
    return truncated + "..."