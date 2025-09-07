# build_database.py

import pandas as pd 
import ollama 
import pickle
from tqdm import tqdm 

def create_vector_database_from_csv(csv_file_path: str, text_column_index: int, embedding_model: str, output_file: str):
    """
    Reads a CSV file, extracts text from a specific column, generates embeddings,
    and saves them to a file.
    """
    print(f"Starting Database Creation from {csv_file_path} ...")
    
    try:
        df = pd.read_csv(csv_file_path, header=None)
        
        # Extract the text chunks from the specified column and drop any missing values.
        chunks = df.iloc[1:, text_column_index].dropna().tolist()
        
        # Filter out any non-string or empty string chunks, just in case
        chunks = [str(chunk) for chunk in chunks if str(chunk).strip()]
        
        print(f"Found {len(chunks)} text chunks to process from column {text_column_index}.")
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_path}' not found.")
        return
    except IndexError:
        print(f"Error: Column with index {text_column_index} not found in the CSV. Please check the file.")
        return

    # An empty list to store the vector database
    vector_database = []
    
    print(f"Generating embeddings using '{embedding_model}'...")
    # Use tqdm to show a progress bar
    for chunk in tqdm(chunks, desc="Processing Chunks"):
        try:
            # Generate the embedding for the chunk
            response = ollama.embeddings(model=embedding_model, prompt=chunk)
            embedding = response["embedding"]
            
            # Store the chunk and its embedding together
            vector_database.append({
                "chunk": chunk,
                "embedding": embedding
            })
        except Exception as e:
            print(f"\nError processing chunk: '{chunk[:50]}...'. Error: {e}")
            continue # Skip this chunk and move to the next
        
    # Save the vector database to a file using pickle
    with open(output_file, 'wb') as f:
        pickle.dump(vector_database, f)
        
    print(f"\nVector database successfully created and saved to '{output_file}'.")
    print(f"Total vectors in database: {len(vector_database)}")

if __name__ == '__main__':
    
    KNOWLEDGE_BASE_FILE = 'Constitution of India.csv'
    TEXT_COLUMN_INDEX = 0
    EMBEDDING_MODEL_NAME = 'bge-m3:567m' 
    OUTPUT_DATABASE_FILE = 'vector_database_constitution.pkl'
    
    create_vector_database_from_csv(
        KNOWLEDGE_BASE_FILE, 
        TEXT_COLUMN_INDEX, 
        EMBEDDING_MODEL_NAME, 
        OUTPUT_DATABASE_FILE
    )