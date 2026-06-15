import os
import logging
import hashlib
from typing import List
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyMuPDFLoader
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initializes the document loader and text splitter.
        
        Args:
            chunk_size (int): The maximum size of each text chunk.
            chunk_overlap (int): The overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Loads a single PDF file.
        
        Args:
            file_path (str): The path to the PDF file.
            
        Returns:
            List[Document]: A list of loaded Document objects.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
            
        logger.info(f"Loading PDF: {file_path}")
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        
        filename = os.path.basename(file_path)
        document_id = hashlib.md5(filename.encode()).hexdigest()[:8]
        
        for doc in docs:
            doc.metadata["source"] = filename
            doc.metadata["document_id"] = document_id
            
        return docs

    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Loads all PDF files from a specified directory.
        
        Args:
            directory_path (str): The path to the directory containing PDFs.
            
        Returns:
            List[Document]: A list of loaded Document objects.
        """
        documents = []
        if not os.path.exists(directory_path):
            logger.warning(f"Directory not found: {directory_path}. Creating it now.")
            os.makedirs(directory_path, exist_ok=True)
            return documents

        for filename in os.listdir(directory_path):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(directory_path, filename)
                try:
                    docs = self.load_pdf(file_path)
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {str(e)}")
                    
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents into smaller chunks suitable for embedding.
        
        Args:
            documents (List[Document]): The documents to split.
            
        Returns:
            List[Document]: The chunked documents.
        """
        if not documents:
            logger.warning("No documents provided to split.")
            return []
            
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def process_directory(self, directory_path: str) -> List[Document]:
        """
        Loads all PDFs from a directory and splits them into chunks in one go.
        
        Args:
            directory_path (str): The path to the directory containing PDFs.
            
        Returns:
            List[Document]: The chunked documents ready for embedding.
        """
        documents = self.load_directory(directory_path)
        return self.split_documents(documents)
