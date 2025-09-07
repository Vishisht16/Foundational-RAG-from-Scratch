# Foundational RAG from Scratch

> A RAG model that can be locally run using Llama 3 for generation and bge-m3 for embedding. Stores a deliberately non-indexed vector database in pickle file, and answers any questions related to Consitution of India by fetching top 5 contexts through cosine similarity.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python)](https://www.python.org/downloads/release/python-390/) [![Ollama](https://img.shields.io/badge/Ollama-Local%20LLMs-blueviolet)](https://ollama.com/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Core Philosophy: Understanding the Engine

In an era of high-level AI frameworks like LangChain, it's easy to chain together components without understanding their core mechanics. This project rejects that approach. It is a **deliberate, from-scratch implementation** of the Retrieval-Augmented Generation (RAG) pipeline to build a foundational understanding of how Large Language Models can be made smarter, more factual, and more trustworthy.

By building the vector database, the similarity search, and the context-injection prompt manually, we gain a deep appreciation for the engineering that powers modern AI agents. The vector search is **intentionally a brute-force, non-indexed cosine similarity search** to expose the raw computational logic that sophisticated libraries like FAISS and Annoy abstract away.

This project is not just a tool; it's an educational deep-dive into the heart of RAG.

---

## How It Works: The RAG Pipeline

The entire system operates in two distinct phases: an offline indexing phase and an online querying phase.

### Phase 1: Indexing (Offline, One-Time Setup)
This phase builds the knowledge library. The `main.py` script automates this if the database is not found.

```
Constitution of India.csv → [build_database.py]
                          → Read & Chunk Text
                            → Generate Vector Embeddings (bge-m3)
                              → Store (Chunk + Embedding) Pairs
                                → vector_database_constitution.pkl
```

### Phase 2: Querying (Online, Interactive)
This is the interactive loop where the user asks questions and the system generates answers.

```
User Question → [retrieval.py]
              → Generate Query Embedding (bge-m3)
                → Brute-Force Cosine Similarity Search
                  → Retrieve Top 5 Most Relevant Articles (Context)
                    → [main.py]
                      → Construct Grounded Prompt (Context + Question)
                        → Generate Answer with LLM (Llama 3)
                          → Stream Response to User
```

---

## Key Features

-   **100% Local & Private:** The entire pipeline runs on your local machine using **Ollama**. No API keys, no data leaving your system.
-   **State-of-the-Art Open-Source Models:** Leverages the powerful **Llama 3** for generation and the high-performance **BGE-M3** for embeddings.
-   **"From Scratch" RAG Logic:** Implements the core vector search and context retrieval logic using Python and NumPy, providing deep insight into the mechanics.
-   **Self-Sufficient & Robust:** The main script automatically detects if the vector database exists. If not, it builds it from the source CSV, making the setup foolproof.
-   **Real-World Knowledge Base:** Answers questions based on the full text of the **Constitution of India**, making it a powerful and practical demonstration.
-   **Interactive Streaming:** Responses from the LLM are streamed token-by-token, providing a responsive, ChatGPT-like user experience.

---

## Technology Stack

-   **LLM & Embedding Server:** **Ollama**
-   **Generation LLM:** **Llama 3**
-   **Embedding Model:** **BGE-M3**
-   **Backend Logic:** **Python 3.9+**
-   **Data Handling:** **Pandas** & **NumPy**
-   **Serialization:** **Pickle**

---

## How to Run This Project

### Step 1: Prerequisites - Install and Set Up Ollama

This project will not work without Ollama.

1.  **Install Ollama:** Download and install the application from the [official Ollama website](https://ollama.com/).
2.  **Pull the Necessary Models:** Open your terminal and run the following commands to download the LLM and the embedding model.
    ```bash
    ollama pull llama3
    ollama pull bge-m3:567m
    ```
3.  **Ensure Ollama is Running:** Make sure the Ollama application is running in the background before you proceed.

### Step 2: Project Setup

```bash
# Clone the repository
git clone https://github.com/your-username/Foundational-RAG-from-Scratch.git
cd Foundational-RAG-from-Scratch

# Create and activate a Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the required Python packages
pip install -r requirements.txt 
```

### Step 3: Add the Knowledge Base

-   Download the `Constitution of India.csv` file.
-   Place it in the root directory of the project.

### Step 4: Run the Application

With Ollama running, simply execute the main script from your terminal.

```bash
python main.py
```

-   **First Run:** The script will detect that `vector_database_constitution.pkl` is missing. It will automatically build the database, which may take a few minutes.
-   **Subsequent Runs:** The script will find the existing database and launch the chatbot immediately.

---

## Example Interaction

```
$ python main.py
Initializing Constitution of India RAG Chatbot....
Vector database ('vector_database_constitution.pkl') not found.
Building the database... (This may take a few minutes)
Starting Database Creation from Constitution of India.csv ...
Found 456 text chunks to process from column 0.
Generating embeddings using 'bge-m3:567m'...
Processing Chunks: 100%|█████████████████████████████████████████████████████████████| 456/456 [01:41<00:00,  4.51it/s]

Vector database successfully created and saved to 'vector_database_constitution.pkl'.
Total vectors in database: 456
Database built successfully!

Ask a question about the Indian Constitution. Type 'exit' or press Ctrl+C to quit.
--------------------------------------------------
You: What are my fundamental rights?

Searching for relevant articles...
Generating embedding for the query: 'What are my fundamental rights?'
Successfully retrieved top 5 chunks.
Found relevant context. Generating answer...
--------------------------------------------------
AI Assistant: Based on the provided context, it appears that you are referring to Part III of the Constitution of India, which deals with Fundamental Rights. Article 25 of the Constitution specifies the Freedom of Conscience and Free Profession, Practice and Propagation of Religion.

According to this article, all persons are equally entitled to freedom of conscience and the right freely to profess, practise and propagate religion, subject to public order, morality and health, as well as other provisions of Part III.

As for your specific question about fundamental rights, I would like to clarify that there is no single definitive answer. The Constitution has provided a comprehensive set of rights under various articles, such as:

1. Right to Equality (Article 14-17)
2. Right to Freedom (Article 19-22)
3. Right against Exploitation (Article 23-24)
4. Right to Life and Personal Liberty (Article 21)
5. Protection of Lawful Assembly (Article 19(1)(a))
6. Freedom of Speech and Expression (Article 19(1)(a))

These rights are guaranteed by the Constitution to ensure that every citizen is treated with dignity and respect.

Please let me know if you would like more information or clarification on specific aspects!
--------------------------------------------------
You: exit

Thank you for using the RAG chatbot!
```

---

## Project Structure

```
Foundational-RAG-from-Scratch/
├── main.py                          # Main application entry point, orchestrates the RAG pipeline.
├── build_database.py                # Logic for creating the vector database from the source CSV.
├── retrieval.py                     # Logic for embedding queries and retrieving chunks via cosine similarity.
│
├── Constitution of India.csv        # The source knowledge base file.
├── vector_database_constitution.pkl # The generated vector database (created on first run).
│
├── requirements.txt
└── README.md
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.