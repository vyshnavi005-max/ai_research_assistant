# AI Research Assistant MVP Implementation Plan

The AI Research Assistant is a Retrieval-Augmented Generation (RAG) system. Users can upload PDF documents, and then ask natural language questions. The system processes the PDFs, stores embeddings in ChromaDB, and uses a language model to answer questions with grounded citations.

**Current Scope**: For Week 1, we are only building the core RAG pipeline (Python script). No backend (FastAPI) or frontend (React) will be implemented yet.

## User Review Required

> [!IMPORTANT]
> - **Ollama & Mistral 7B**: This implementation plan assumes Ollama is running locally on the system (usually at `http://localhost:11434`) and has the `mistral` model pulled.
> - **Python Dependencies**: We will update `requirements.txt` to be compatible with Python 3.13 (e.g., using newer versions of LangChain) so that `pip install` works in your virtual environment.

## Folder Structure

We will structure the project under the workspace root `c:/ai_research_assitant` as follows:

```
c:/ai_research_assitant/
├── backend/
│   ├── chroma_db/
│   ├── data/                 # Folder to put test PDFs
│   ├── rag_pipeline.py       # Core RAG pipeline script
│   └── requirements.txt      # Python dependencies
```

## Proposed Changes

### Core RAG Script
#### [MODIFY] [requirements.txt](file:///c:/ai_research_assitant/backend/requirements.txt)
Remove strict version pinning to allow installation on Python 3.13 (e.g., use `langchain>=0.3.0`).

#### [NEW] [rag_pipeline.py](file:///c:/ai_research_assitant/backend/rag_pipeline.py)
A standalone Python script that performs the following:
1. **Document Loading**: Loads a PDF from the `data/` folder using PyMuPDF (or PyPDF).
2. **Chunking**: Splits the document into manageable chunks using `RecursiveCharacterTextSplitter`.
3. **Embedding & Storage**: Embeds the chunks using `sentence-transformers` and stores them in ChromaDB.
4. **Retrieval & Generation**: Uses Ollama (Mistral) to answer a user-provided question based on the retrieved chunks.

## Verification Plan

### Automated Tests
- N/A for this phase.

### Manual Verification
1. Place a sample PDF in `backend/data/`.
2. Run `python rag_pipeline.py` inside the virtual environment.
3. The script should successfully process the PDF, index it in ChromaDB, and output the answer to a test question.
