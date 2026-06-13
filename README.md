# AI Research Assistant

A full-stack Retrieval-Augmented Generation (RAG) system to query PDF documents using LangChain, ChromaDB, FastAPI, React, and Gemini API.

## Folder Structure

```
c:/ai_research_assitant/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routes/endpoints
│   │   ├── services/     # PDF Processing logic
│   │   ├── rag/          # Embeddings and vector DB retrieval
│   │   ├── models/       # Pydantic schemas/models
│   │   └── utils/        # Utility helpers
│   ├── uploads/          # Directory where uploaded PDFs are stored
│   ├── chroma_db/        # Persistent vector store db directory
│   └── main.py           # Entry point for FastAPI
│
└── frontend/
    ├── src/
    │   ├── components/   # UI components (Upload, Chat, Citations)
    │   ├── pages/        # Main views/pages
    │   └── services/     # API services client (Axios)
```

## Setup & Running Instructions

### Backend
1. Create a virtual environment and install dependencies:
   ```bash
   cd backend
   pip install fastapi uvicorn langchain langchain-community chromadb pymupdf sentence-transformers python-multipart
   ```
2. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend
1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```
2. Start development server:
   ```bash
   npm run dev
   ```
