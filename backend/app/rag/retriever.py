import logging
from typing import List
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, vector_store: VectorStore, k: int = 3):
        """
        Initializes the Retriever.
        
        Args:
            vector_store (VectorStore): The underlying vector store.
            k (int): Number of top documents to retrieve.
        """
        self.vector_store = vector_store
        self.k = k
        self._retriever = None

    def get_retriever_interface(self):
        """
        Returns the LangChain retriever interface.
        """
        if self._retriever is None:
            vs = self.vector_store.get_vectorstore()
            self._retriever = vs.as_retriever(search_kwargs={"k": self.k})
        return self._retriever

    def retrieve(self, query: str) -> List[Document]:
        """
        Directly retrieves relevant documents for a given query.
        
        Args:
            query (str): The user query.
            
        Returns:
            List[Document]: Retrieved documents.
        """
        retriever = self.get_retriever_interface()
        logger.info(f"Retrieving top {self.k} documents for query: {query}")
        return retriever.invoke(query)
