import os
import pandas as pd
import numpy as np
import pickle
import chromadb
from chromadb.config import Settings
import argparse
from tqdm import tqdm

def setup_args():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description='Load generated embeddings into ChromaDB')
    parser.add_argument('--embeddings_dir', type=str, required=True, 
                        help='Directory containing the embeddings and metadata')
    parser.add_argument('--db_path', type=str, default='./chroma_db',
                        help='Path to store the ChromaDB database')
    parser.add_argument('--collection_name', type=str, default='climate_research',
                        help='Name of the collection in ChromaDB')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for loading documents into ChromaDB')
    return parser.parse_args()

def load_embeddings_and_metadata(embeddings_dir):
    """Load the generated embeddings and metadata."""
    print(f"Loading embeddings and metadata from {embeddings_dir}...")
    
    # Load abstract embeddings
    abstract_embeddings_path = os.path.join(embeddings_dir, 'abstract_embeddings.pkl')
    with open(abstract_embeddings_path, 'rb') as f:
        abstract_embeddings = pickle.load(f)
    
    # Load metadata
    metadata_path = os.path.join(embeddings_dir, 'document_metadata.csv')
    metadata_df = pd.read_csv(metadata_path)
    
    # Load embedding metadata
    embedding_metadata_path = os.path.join(embeddings_dir, 'embedding_metadata.pkl')
    with open(embedding_metadata_path, 'rb') as f:
        embedding_metadata = pickle.load(f)
    
    print(f"Loaded {len(metadata_df)} documents with embeddings.")
    print(f"Embedding model: {embedding_metadata['model_name']}")
    
    return abstract_embeddings, metadata_df, embedding_metadata

def setup_chromadb(db_path, collection_name):
    """Setup ChromaDB client and collection."""
    print(f"Setting up ChromaDB at {db_path}...")
    
    # Create the client
    client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
    
    # Create or get the collection
    try:
        # Try to get the collection if it exists
        collection = client.get_collection(name=collection_name)
        print(f"Using existing collection: {collection_name}")
    except:
        # Create a new collection if it doesn't exist
        collection = client.create_collection(name=collection_name)
        print(f"Created new collection: {collection_name}")
    
    return client, collection

def load_documents_to_chromadb(collection, abstract_embeddings, metadata_df, batch_size):
    """Load documents and embeddings into ChromaDB."""
    print(f"Loading documents into ChromaDB in batches of {batch_size}...")
    
    # Total number of documents
    total_docs = len(metadata_df)
    
    # Process in batches
    for i in tqdm(range(0, total_docs, batch_size)):
        end_idx = min(i + batch_size, total_docs)
        batch_df = metadata_df.iloc[i:end_idx]
        batch_embeddings = abstract_embeddings[i:end_idx]
        
        # Prepare document IDs - ensure they are strings
        ids = batch_df['document_id'].astype(str).tolist()
        
        # Prepare metadata
        metadatas = []
        for _, row in batch_df.iterrows():
            metadata = {
                'title': row['Title'],
                'authors': row['Authors'],
                'published': row['Published'],
                'year': int(row['year_source']) if pd.notna(row['year_source']) else 0,
                'link': row['Link'] if pd.notna(row['Link']) else ''
            }
            metadatas.append(metadata)
        
        # Prepare documents (abstracts)
        documents = batch_df['Abstract'].fillna('').tolist()
        
        # Add to collection
        collection.add(
            ids=ids,
            embeddings=batch_embeddings.tolist(),
            metadatas=metadatas,
            documents=documents
        )
    
    print(f"Successfully loaded {total_docs} documents into ChromaDB collection")

def main():
    """Main function to load embeddings into ChromaDB."""
    args = setup_args()
    
    # Load embeddings and metadata
    abstract_embeddings, metadata_df, embedding_metadata = load_embeddings_and_metadata(args.embeddings_dir)
    
    # Setup ChromaDB
    client, collection = setup_chromadb(args.db_path, args.collection_name)
    
    # Load documents into ChromaDB
    load_documents_to_chromadb(collection, abstract_embeddings, metadata_df, args.batch_size)
    
    # Print collection info
    print(f"ChromaDB collection '{args.collection_name}' info:")
    print(f"Total documents: {collection.count()}")
    print(f"Embedding dimensions: {abstract_embeddings.shape[1]}")
    print(f"Embedding model: {embedding_metadata['model_name']}")
    
    print("Loading embeddings to ChromaDB complete!")

if __name__ == "__main__":
    main()  