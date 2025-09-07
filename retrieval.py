# retrieval.py

import pickle
import ollama
import numpy as np

def cosine_similarity(v1, v2):
    """Calculates the cosine similarity between two vectors."""
    # Ensure the vectors are numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Calculate dot product and norms
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return dot_product / (norm_v1 * norm_v2)

def retrieve_relevant_chunks(user_query: str, database_path: str, embedding_model: str, top_n: int = 5):
    """
    Retrieves the top N most relevant chunks from the vector database for a given user query.
    """
    # 1. Load the pre-computed vector database
    try:
        with open(database_path, 'rb') as f:
            vector_database = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Database file not found at '{database_path}'. Please run build_database.py first.")
        return []

    # 2. Generate the embedding for the user's query
    print(f"Generating embedding for the query: '{user_query}'")
    query_embedding_response = ollama.embeddings(
        model=embedding_model,
        prompt=user_query
    )
    query_embedding = query_embedding_response["embedding"]

    # 3. Calculate similarity between the query and each chunk in the database
    similarity_scores = []
    for entry in vector_database:
        similarity = cosine_similarity(query_embedding, entry["embedding"])
        similarity_scores.append({
            "score": similarity,
            "chunk": entry["chunk"]
        })
        
    # 4. Sort the chunks by similarity score in descending order
    sorted_chunks = sorted(similarity_scores, key=lambda x: x["score"], reverse=True)
    
    # 5. Return the top N most relevant chunks
    print(f"Successfully retrieved top {top_n} chunks.")
    return sorted_chunks[:top_n]

# This block allows for direct testing of the retrieval script
if __name__ == '__main__':
    DATABASE_FILE = 'vector_database_constitution.pkl'
    EMBEDDING_MODEL = 'bge-m3:567m'
    
    # Example query
    test_query = "What are the fundamental rights of a citizen?"
    
    # Retrieve the top 5 chunks
    top_chunks = retrieve_relevant_chunks(
        user_query=test_query,
        database_path=DATABASE_FILE,
        embedding_model=EMBEDDING_MODEL,
        top_n=5
    )
    
    if top_chunks:
        print(f"\n--- Top 5 most relevant articles for the query: '{test_query}' ---")
        for i, chunk_info in enumerate(top_chunks):
            print(f"\n{i+1}. Similarity Score: {chunk_info['score']:.4f}")
            print(f"   Article: {chunk_info['chunk']}")