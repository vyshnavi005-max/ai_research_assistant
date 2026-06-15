import os
import logging
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_core.prompts import PromptTemplate
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import StrOutputParser
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
# pyrefly: ignore [missing-import]
from langchain_core.chat_history import InMemoryChatMessageHistory
# pyrefly: ignore [missing-import]
from langchain_core.messages import HumanMessage, AIMessage
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

class CitationTracker:
    """Tracks citations for documents used in generation."""
    def __init__(self):
        self.citations = []

    def extract_citations(self, docs):
        self.citations = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            self.citations.append({
                "id": i + 1, 
                "source": source, 
                "page": page, 
                "content": doc.page_content
            })
        return self.citations

class LLMGenerator:
    def __init__(self, model_name: str = "gemini-1.5-flash-latest"):
        """
        Initializes the LLM Generator with Gemini model.
        """
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
        
        # Initialize memory
        self.memory = InMemoryChatMessageHistory()
        self.citation_tracker = CitationTracker()
        
        template = """Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Please cite the sources using the [ID] provided in the context.

Chat History:
{chat_history}

Context: {context}

Question: {question}

Answer:"""
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["chat_history", "context", "question"]
        )

    def _format_docs(self, docs):
        """Formats the documents into a single string with citation IDs for the prompt."""
        formatted_docs = []
        for i, doc in enumerate(docs):
            formatted_docs.append(f"[{i+1}] {doc.page_content}")
        return "\n\n".join(formatted_docs)

    def generate_answer(self, retriever, question: str) -> dict:
        """
        Generates an answer for a given question using a retriever.
        Includes chat memory and citation tracking.
        """
        # 1. Retrieve documents and track citations
        docs = retriever.invoke(question)
        citations = self.citation_tracker.extract_citations(docs)
        
        # 2. Format context
        context = self._format_docs(docs)
        
        # Format the prompt with chat history
        chat_history = "\n".join(
            f"{msg.type}: {msg.content}"
            for msg in self.memory.messages
        )
        
        # 3. Run the LLM chain
        print("Invoking Gemini...")
        chain = self.prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "chat_history": chat_history,
            "context": context,
            "question": question
        })
        print("Gemini returned successfully")
        
        # 4. Save context to memory
        self.memory.add_message(HumanMessage(content=question))
        self.memory.add_message(AIMessage(content=answer))
        
        return {
            "answer": answer,
            "citations": citations
        }
