# RAG Multi-Agent Document Intelligence

A Retrieval-Augmented Generation (RAG) system built with a multi-agent architecture. This project uses **LangGraph** to coordinate specialized agents, **FastAPI** for the backend API, **Qdrant** as the vector database, and **Streamlit** for a user interface.

##  Overview

This system implements RAG by having an **Analysis Agent** that evaluates retrieval quality before generating an answer. This prevents "hallucination" by ensuring the LLM only answers when relevant context is found.

### The Multi-Agent Workflow:
1.  **Query Agent**: Converts the user query into embeddings and retrieves the top-k relevant document chunks from Qdrant.
2.  **Analysis Agent**: Evaluates the similarity scores of the retrieved chunks. 
    - If scores are high, it proceeds to generation.
    - If scores are medium, it proceeds but adds a caveat about potential incompleteness.
    - If scores are low/non-existent, it triggers a fallback response explaining why it cannot answer.
3.  **Response Agent**: Generates the final answer using an LLM (Large Language Model) based *only* on the provided context.

##  Key Features

-   **Intelligent Retrieval**: Uses state-of-the-art embedding models (`all-MiniLM-L6-v2`) and vector similarity search.
-   **Quality Gates**: Automatic confidence scoring and retrieval analysis to ensure factual accuracy.
-   **Multi-Format Support**: Ingest and query `.txt`, `.pdf`, and `.csv` files.
-   **Full CRUD Operations**: Upload, list, and delete documents from the vector store via the UI or API.
-   **Professional UI**: Streamlit-based dashboard with chat history, document management, and real-time metrics.
-   **Robust API**: FastAPI implementation with Pydantic models for data validation.

##  Tech Stack

-   **Orchestration**: LangGraph / Python
-   **API Framework**: FastAPI
-   **Frontend**: Streamlit
-   **Vector Database**: Qdrant
-   **LLM Integration**: LangChain / Groq
-   **Document Processing**: PyPDF, CSVLoader, RecursiveCharacterTextSplitter

##  Project Structure

```text
├── agents/
│   ├── query_agent.py      # Vector search logic
│   ├── analysis_agent.py   # Quality evaluation & confidence scoring
│   ├── response_agent.py   # LLM response synthesis
│   └── state.py            # TypedDict defining the agent state
├── api/
│   ├── models.py           # Pydantic request/response schemas
│   └── routers/
│       ├── query.py        # Chat/Query endpoints
│       └── documents.py    # Document management endpoints
├── data_ingestion/
│   └── ingestion.py        # Logic for processing and embedding files
├── main.py                 # FastAPI application entry point
├── frontend/
|   └── app.py                  # Streamlit frontend
├── config.py               # Global configurations (thresholds, paths)
├── utils.py                # Shared utilities (clients, model loaders)
└── requirements.txt        # Project dependencies
```

##  Setup & Installation

### 1. Prerequisites
- Python 3.9+
- A running Qdrant instance (Local or Cloud)
- Groq API Key (for LLM inference)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### 4. Running the System

**Start the Backend API:**
```bash
uvicorn main:app --reload
```

**Start the Frontend UI:**
```bash
streamlit run app.py
```

##  Usage

1.  **Upload Documents**: Use the "Upload Documents" tab in the Streamlit UI to add your knowledge base.
2.  **Ask Questions**: Navigate to the "Chat" tab and ask questions. 
3.  **Monitor Quality**: Check confidence scores and retrieval quality indicators provided in the system messages.
4.  **Manage Data**: Use the "Current Documents" section to refresh the list or delete old files to maintain your index.
