import logging
from typing import List
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
# pyrefly: ignore [missing-import]
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "chroma_db", embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the VectorStore with ChromaDB and an embedding model.
        
        Args:
            persist_directory (str): Path to persist the Chroma database.
            embedding_model_name (str): Name of the sentence transformer model to use.
        """
        self.persist_directory = persist_directory
        self.embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        self._vectorstore = None

    def get_vectorstore(self) -> Chroma:
        """
        Returns the initialized Chroma vector store. If not loaded, attempts to load existing one.
        """
        if self._vectorstore is None:
            logger.info(f"Loading existing vector store from {self.persist_directory}")
            self._vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
        return self._vectorstore

    def add_documents(self, documents: List[Document]) -> Chroma:
        """
        Embeds and adds a list of documents to the vector store.
        
        Args:
            documents (List[Document]): The chunked documents to add.
            
        Returns:
            Chroma: The updated vector store.
        """
        if not documents:
            logger.warning("No documents provided to add to vector store.")
            return self.get_vectorstore()

        logger.info(f"Adding {len(documents)} documents to Chroma database at {self.persist_directory}")
        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory
        )
        logger.info("Documents successfully added to the vector store.")
        return self._vectorstore

