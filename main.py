import ollama
import sys
import os

from build_database import create_vector_database_from_csv
from retrieval import retrieve_relevant_chunks

# Config Variables
KNOWLEDGE_BASE_FILE = 'Constitution of India.csv'
DATABASE_FILE = 'vector_database_constitution.pkl'
TEXT_COLUMN_INDEX = 0 # The column in the CSV with the text

EMBEDDING_MODEL = 'bge-m3:567m'
LLM_MODEL = 'llama3'

CONTEXT_SIZE = 5 # Number of relevant chunks to retrieve

def main():
    """
    The main function to run the RAG chat application.
    Checks for the database, builds it if necessary, and then starts the chat loop.
    """
    
    # Check for Vector DB
    print("Initializing Constitution of India RAG Chatbot....")
    if not os.path.exists(DATABASE_FILE):
        print(f"Vector database ('{DATABASE_FILE}') not found.")
        
        # Check if the source CSV file exists before trying to build
        if not os.path.exists(KNOWLEDGE_BASE_FILE):
            print(f"Error: Source file ('{KNOWLEDGE_BASE_FILE}') not found.")
            print("Please download the CSV file and place it in the same directory.")
            sys.exit(1) # Exit with an error code
            
        print("Building the database... (This may take a few minutes)")
        create_vector_database_from_csv(
            KNOWLEDGE_BASE_FILE,
            TEXT_COLUMN_INDEX,
            EMBEDDING_MODEL,
            DATABASE_FILE
        )
        print("Database built successfully!")
    else:
        print(f"Vector database found: '{DATABASE_FILE}'")

    print("\nAsk a question about the Indian Constitution. Type 'exit' or press Ctrl+C to quit.")
    print("-" * 50)

    try:
        while True:
            # Get user input
            user_query = input("You: ")
            if user_query.lower().strip() == 'exit':
                break

            # Retrieve relevant context
            print("\nSearching for relevant articles...")
            relevant_context = retrieve_relevant_chunks(
                user_query, DATABASE_FILE, EMBEDDING_MODEL, top_n=CONTEXT_SIZE
            )

            if not relevant_context:
                print("Could not find relevant information. Please try rephrasing your question.")
                print("-" * 50)
                continue

            context_str = "\n\n".join([item['chunk'] for item in relevant_context])
            
            print("Found relevant context. Generating answer...")
            print("-" * 50)

            # Construct the prompt
            prompt_template = f"""
            **Instruction:** You are a helpful assistant specialized in the Constitution of India.
            Answer the user's question based *only* on the following context.
            If the information is not in the context, clearly state that you cannot answer based on the provided information.

            **Context:**
            {context_str}

            **User's Question:** {user_query}

            **Answer:**
            """

            # Generate the answer with streaming
            print("AI Assistant: ", end="")
            stream = ollama.chat(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt_template}],
                stream=True,
            )
            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)
            
            print("\n" + "-" * 50)

    except KeyboardInterrupt:
        print("\n\nExiting chatbot. Goodbye!")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    
    finally:
        print("\nThank you for using the RAG chatbot!")


if __name__ == '__main__':
    main()