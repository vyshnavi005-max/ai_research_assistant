import os
import logging
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_core.prompts import PromptTemplate
# pyrefly: ignore [missing-import]
from langchain_core.runnables import RunnablePassthrough
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import StrOutputParser
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# Load environment variables from .env
# The user saved it in backend/app/.env, and this file is in backend/app/rag/
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

class LLMGenerator:
    def __init__(self, model_name: str = "gemini-3.5-flash"):
        """
        Initializes the LLM Generator with Gemini model.
        
        Args:
            model_name (str): Name of the Gemini model to use.
        """
        # Ensure the API key is set in environment variables
        # We check both GEMINI_API_KEY and GOOGLE_API_KEY since users might use either
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("No Gemini/Google API key found in environment variables. Make sure it's set in your .env file.")
            
        logger.info(f"Initializing Gemini model: {model_name}")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
            convert_system_message_to_human=True
        )
        
        template = """Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:"""
        self.prompt = PromptTemplate.from_template(template)

    def _format_docs(self, docs):
        """Formats the documents into a single string for the prompt."""
        return "\n\n".join(doc.page_content for doc in docs)

    def build_chain(self, retriever):
        """
        Builds the final RAG chain using the provided retriever.
        
        Args:
            retriever: The vector store retriever interface.
            
        Returns:
            The executable RAG chain.
        """
        logger.info("Building RAG chain with Gemini model")
        rag_chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def generate_answer(self, retriever, question: str) -> str:
        """
        Convenience method to generate an answer for a given question using a retriever.
        
        Args:
            retriever: The vector store retriever interface.
            question (str): The user's question.
            
        Returns:
            str: The generated answer.
        """
        chain = self.build_chain(retriever)
        return chain.invoke(question)
