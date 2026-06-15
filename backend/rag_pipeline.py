import os
import argparse
import sys

# Ensure the app module can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.rag.document_loader import DocumentLoader
from app.rag.vector_store import VectorStore
from app.rag.llm_generator import LLMGenerator
from app.rag.retriever import Retriever

# Configuration
DATA_DIR = "data"
CHROMA_PATH = "chroma_db"

def index_data():
    """Indexes the PDFs in the data directory using modular classes."""
    print(f"Scanning for PDFs in '{DATA_DIR}'...")
    loader = DocumentLoader()
    documents = loader.load_directory(DATA_DIR)
    
    if not documents:
        print("No documents found to index.")
        return
    
    print(f"Loaded {len(documents)} document pages. Splitting into chunks...")
    chunks = loader.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks. Initializing embedding model and Vector Store...")
    vector_store = VectorStore(persist_directory=CHROMA_PATH)
    
    print("Embedding chunks and saving to ChromaDB (this might take a while)...")
    vector_store.add_documents(chunks)
    print("Indexing complete.")

def query_rag(question):
    """Queries the existing RAG database using Gemini."""
    if not os.path.exists(CHROMA_PATH):
        print("Chroma database not found. Please run with --index first.")
        return
        
    vector_store = VectorStore(persist_directory=CHROMA_PATH)
    retriever_logic = Retriever(vector_store)
    retriever = retriever_logic.get_retriever_interface()
    
    generator = LLMGenerator()
    
    print(f"\nQuestion: {question}")
    print("Generating answer with Gemini...")
    
    response = generator.generate_answer(retriever, question)
    
    print("\nAnswer:")
    print("-" * 40)
    print(response["answer"])
    print("-" * 40)
    
    if response["citations"]:
        print("\nCitations:")
        for citation in response["citations"]:
            print(f"[{citation['id']}] {citation['source']} (Page {citation['page']})")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Research Assistant RAG Pipeline (Gemini Edition)")
    parser.add_argument("--index", action="store_true", help="Index PDFs in the data directory")
    parser.add_argument("--query", type=str, help="Question to ask the RAG system")
    
    args = parser.parse_args()
    
    if args.index:
        index_data()
    elif args.query:
        query_rag(args.query)
    else:
        parser.print_help()
