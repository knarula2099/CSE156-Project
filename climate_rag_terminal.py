import chromadb
from sentence_transformers import SentenceTransformer
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the vector database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("climate_research")

# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# OpenAI API Key (ensure it's set in environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key! Set OPENAI_API_KEY as an environment variable.")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

def retrieve_documents(query, top_k=5):
    """Retrieve top-k most relevant documents from ChromaDB."""
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    if not results["documents"]:
        return [], []

    return results["documents"][0], results["metadatas"][0]

def generate_response(query, retrieved_docs):
    """Generate a response based on retrieved documents using GPT-4."""
    if not retrieved_docs:
        return "No relevant documents found for your query."

    context = "\n\n".join(retrieved_docs[:3])  # Use top 3 abstracts as context
    prompt = f"Based on the following climate research abstracts, answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that provides detailed answers based on climate research papers."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def main():
    print("Climate Research RAG System - Terminal Interface")
    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        documents, metadata = retrieve_documents(query)
        
        if not documents:
            print("\nNo relevant documents found. Try refining your query.")
            continue
        
        print("\nTop Retrieved Papers:")
        for i, (doc, meta) in enumerate(zip(documents, metadata)):
            print(f"\n[{i+1}] {meta.get('title', 'Unknown Title')} ({meta.get('published', 'Unknown Date')})")
            print(f"Abstract: {doc[:500]}...")  # Show first 500 chars
            print(f"Link: {meta.get('link', 'No link available')}\n")

        # Generate a response using LLM
        answer = generate_response(query, documents)
        print("\nGenerated Answer:\n", answer)

if __name__ == "__main__":
    main()
