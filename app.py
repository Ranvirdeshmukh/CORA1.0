from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scripts.chatbot import parse_filters_from_query
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
import time
import os
import asyncio
from threading import Lock

app = FastAPI()

LOCAL_DB_PATH = "/tmp/vector_db"  # Cloud Run's temporary storage

class ChatbotManager:
    _instance = None
    _lock = Lock()
    _last_used = None
    _cleanup_task = None
    CLEANUP_THRESHOLD = 300  # 5 minutes of inactivity

    def __init__(self):
        self.vector_db = None
        self.chat_model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def initialize(self):
        """Initialize only if not already loaded"""
        if self.vector_db is None:
            # Initialize embedding model
            embedding_model = OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Load vector store from disk
            self.vector_db = Chroma(
                persist_directory=LOCAL_DB_PATH,
                embedding_function=embedding_model
            )
            
            # Initialize chat model
            self.chat_model = ChatOpenAI(
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        
        self._last_used = time.time()

    def cleanup(self):
        """Release memory but keep files on disk"""
        if self.vector_db is not None:
            del self.vector_db
            self.vector_db = None
        if self.chat_model is not None:
            del self.chat_model
            self.chat_model = None

    @classmethod
    async def start_cleanup_task(cls):
        """Periodically check and cleanup unused resources"""
        while True:
            await asyncio.sleep(60)  # Check every minute
            instance = cls.get_instance()
            if (instance._last_used is not None and 
                time.time() - instance._last_used > cls.CLEANUP_THRESHOLD):
                instance.cleanup()

class QueryInput(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    # Start the cleanup task
    ChatbotManager._cleanup_task = asyncio.create_task(
        ChatbotManager.start_cleanup_task()
    )

@app.on_event("shutdown")
async def shutdown_event():
    if ChatbotManager._cleanup_task:
        ChatbotManager._cleanup_task.cancel()
    instance = ChatbotManager.get_instance()
    instance.cleanup()

@app.post("/chat")
async def chat_endpoint(input_data: QueryInput):
    try:
        # Get or initialize chatbot
        manager = ChatbotManager.get_instance()
        manager.initialize()

        # Parse filters
        filters = parse_filters_from_query(input_data.query)
        
        # Set up retriever
        if filters:
            retriever = manager.vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "filter": filters}
            )
        else:
            retriever = manager.vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3}
            )

        # Define prompt template
        stuff_template = """You are CORA 1.0, a helpful college course advisor.

- If asked your name, reply: "I am CORA 1.0— your AI college advisor."
- If asked something unrelated to your data, say you don't have info.
- If asked about courses or professors, answer with the data.

Here is some context from your documents:
{context}

User question:
{question}

Answer:
"""
        stuff_prompt = PromptTemplate(
            template=stuff_template,
            input_variables=["context", "question"]
        )

        # Build and run the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=manager.chat_model,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": stuff_prompt
            }
        )

        result = qa_chain.invoke({"query": input_data.query})
        
        # Format response
        sources = [{
            "professor": doc.metadata.get("professor", "N/A"),
            "term": doc.metadata.get("term", "N/A"),
            "course_name": doc.metadata.get("course_name", "N/A")
        } for doc in result["source_documents"]]

        return {
            "answer": result["result"],
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}