# src/api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from config.settings import settings
import logging
from src.core.vector_store import VectorStoreManager
from src.core.llm_manager import LLMManager
from src.chains.qa_chain import UniversityQASystem
from src.processing.translator import ScriptTranslator
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="University Chatbot API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    script: str = "latinic"  # cyrillic or latinic
    language: str = "sr"     # sr, mk, bg

class ChatResponse(BaseModel):
    question: str
    answer: str
   # sources: list
   # confidence: float

def initialize_system():
    """Initialize all components"""
    logger.info("Initializing system components...")
    
    # Initialize components
    translator = ScriptTranslator()
    llm_manager = LLMManager(settings)
    vector_store = VectorStoreManager(settings)
    
    # Create QA system (reads from existing ChromaDB — scraper populates it)
    qa_system = UniversityQASystem(llm_manager, vector_store, translator)
    
    logger.info("System initialized successfully!")
    return qa_system

qa_system = initialize_system()


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream response token by token"""
    try:
        return StreamingResponse(
            stream_response(request.question),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def stream_response(question: str) -> AsyncGenerator[str, None]:
    """Generate streaming response"""
    try:
        # Get streaming response from your QA system
        async for token in qa_system.query_stream(question):
            # Send each token as SSE (Server-Sent Events)
            yield f"data: {json.dumps({'token': token})}\n\n"
        
        # Send end marker
        yield f"data: {json.dumps({'done': True})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )