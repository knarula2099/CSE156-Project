import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import pickle
import argparse

def setup_args():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description='Generate embeddings for climate research papers')
    parser.add_argument('--input_file', type=str, required=True, 
                        help='Path to the preprocessed CSV file')
    parser.add_argument('--output_dir', type=str, default='./data/processed/embeddings',
                        help='Directory to save the embeddings')
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model to use for embeddings')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for embedding generation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation (cuda/cpu)')
    return parser.parse_args()

def load_data(file_path):
    """Load the preprocessed data from CSV."""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} records.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def generate_embeddings(df, model_name, batch_size, device):
    """Generate embeddings for the preprocessed abstracts and titles."""
    print(f"Generating embeddings using {model_name} on {device}...")
    
    # Load the sentence transformer model
    model = SentenceTransformer(model_name, device=device)
    
    # Check if both processed columns exist
    has_processed_title = 'Processed_Title' in df.columns
    has_processed_abstract = 'Processed_Abstract' in df.columns
    
    if not has_processed_abstract:
        raise ValueError("Dataset must contain 'Processed_Abstract' column")
    
    # Generate embeddings for abstracts
    print("Generating abstract embeddings...")
    abstract_texts = df['Processed_Abstract'].fillna('').tolist()
    abstract_embeddings = []
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(abstract_texts), batch_size)):
        batch = abstract_texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        abstract_embeddings.extend(batch_embeddings)
    
    # Convert to numpy array for easier storage
    abstract_embeddings = np.array(abstract_embeddings)
    
    # Generate embeddings for titles if available
    title_embeddings = None
    if has_processed_title:
        print("Generating title embeddings...")
        title_texts = df['Processed_Title'].fillna('').tolist()
        title_embeddings = []
        
        for i in tqdm(range(0, len(title_texts), batch_size)):
            batch = title_texts[i:i+batch_size]
            batch_embeddings = model.encode(batch, show_progress_bar=False)
            title_embeddings.extend(batch_embeddings)
        
        title_embeddings = np.array(title_embeddings)
    
    # Return the embeddings
    return {
        'abstract_embeddings': abstract_embeddings,
        'title_embeddings': title_embeddings,
        'model_name': model_name
    }

def save_embeddings(embeddings, df, output_dir):
    """Save the generated embeddings and a dataframe with metadata."""
    print(f"Saving embeddings to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the raw embeddings
    with open(os.path.join(output_dir, 'abstract_embeddings.pkl'), 'wb') as f:
        pickle.dump(embeddings['abstract_embeddings'], f)
    
    if embeddings['title_embeddings'] is not None:
        with open(os.path.join(output_dir, 'title_embeddings.pkl'), 'wb') as f:
            pickle.dump(embeddings['title_embeddings'], f)
    
    # Save metadata about the embeddings
    metadata = {
        'model_name': embeddings['model_name'],
        'abstract_shape': embeddings['abstract_embeddings'].shape,
        'title_shape': embeddings['title_embeddings'].shape if embeddings['title_embeddings'] is not None else None,
        'dataset_size': len(df)
    }
    
    with open(os.path.join(output_dir, 'embedding_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save a dataframe with metadata and document IDs for later reference
    df_with_ids = df.copy()
    # Ensure document_id is a string to avoid ChromaDB errors
    df_with_ids['document_id'] = [f"doc_{i}" for i in range(len(df))]
    
    # Select relevant columns for storage
    cols_to_keep = ['document_id', 'Title', 'Authors', 'Published', 'year_source', 'Link', 'Abstract']
    df_to_save = df_with_ids[cols_to_keep].copy()
    
    # Save as CSV for easy inspection
    df_to_save.to_csv(os.path.join(output_dir, 'document_metadata.csv'), index=False)
    
    print(f"Embeddings and metadata successfully saved to {output_dir}")

def main():
    """Main function to generate and save embeddings."""
    args = setup_args()
    
    # Load the data
    df = load_data(args.input_file)
    if df is None:
        return
    
    # Generate embeddings
    embeddings = generate_embeddings(df, args.model_name, args.batch_size, args.device)
    
    # Save embeddings and metadata
    save_embeddings(embeddings, df, args.output_dir)
    
    print("Embedding generation complete!")

if __name__ == "__main__":
    main()