"""
FastAPI-based REST API for Document Retrieval RAG.

Run with: uvicorn api_app:app --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from rag.agent import build_retrieval_agent
from vertexai.agent_engines import AdkApp
import vertexai
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Document Retrieval RAG API",
    description="AI-powered document Q&A with RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent_app = None


class QueryRequest(BaseModel):
    """Query request model."""
    question: str
    user_id: Optional[str] = "api_user"


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    citations: Optional[List[str]] = None


def extract_response_text(response):
    """Extract text from response dict."""
    if response and isinstance(response, dict):
        if 'parts' in response:
            for part in response['parts']:
                if 'text' in part:
                    return part['text']
        elif 'content' in response and 'parts' in response['content']:
            for part in response['content']['parts']:
                if 'text' in part:
                    return part['text']
    return None


@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup."""
    global agent_app
    
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    vertexai.init(project=project, location=location)
    
    agent = build_retrieval_agent()
    agent_app = AdkApp(agent=agent)
    
    print("✓ Document Retrieval Agent initialized")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Document Retrieval RAG API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/query",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent_initialized": agent_app is not None
    }


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents using RAG.
    
    Args:
        request: QueryRequest with question and optional user_id
        
    Returns:
        QueryResponse with answer and citations
    """
    if not agent_app:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        full_response = ""
        
        for response in agent_app.stream_query(
            message=request.question,
            user_id=request.user_id
        ):
            text = extract_response_text(response)
            if text:
                full_response += text
        
        return QueryResponse(
            answer=full_response,
            citations=None  # You can extract citations from response if needed
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_documents_stream(request: QueryRequest):
    """
    Query documents using RAG with streaming response.
    
    Args:
        request: QueryRequest with question and optional user_id
        
    Returns:
        Streaming response
    """
    if not agent_app:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    async def generate():
        try:
            for response in agent_app.stream_query(
                message=request.question,
                user_id=request.user_id
            ):
                text = extract_response_text(response)
                if text:
                    yield f"data: {text}\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
