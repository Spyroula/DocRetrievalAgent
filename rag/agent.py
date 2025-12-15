"""
Document Retrieval Agent powered by Vertex AI RAG Engine.

This module defines an ADK-based agent that leverages Vertex AI's managed RAG
service to retrieve and synthesize information from document collections.
"""

import os
import logging
from typing import Optional

import google.auth
from google.adk.agents import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag
from dotenv import load_dotenv

from .prompts import get_agent_instructions

logger = logging.getLogger(__name__)


def _initialize_gcp_environment() -> str:
    """
    Initialize GCP environment and return project ID.
    
    Falls back gracefully if ADC is not available (no Google account).
    """
    load_dotenv()
    
    # Check if project is already set
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if project_id:
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))
        return project_id
    
    # Try to get ADC credentials
    try:
        _, project_id = google.auth.default()
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
        return project_id
    except Exception as e:
        logger.warning("Could not retrieve GCP credentials: %s. Set GOOGLE_CLOUD_PROJECT env var for production use.", e)
        # Set safe defaults for local/test mode
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "local-project")
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
        return os.environ["GOOGLE_CLOUD_PROJECT"]


def _build_retrieval_tool(corpus_resource: Optional[str] = None):
    """
    Construct the retrieval tool.
    
    In local mode (no GCP credentials), returns a stub tool to avoid cloud calls.
    In production, returns a Vertex AI RAG retrieval tool.
    """
    # Check if we can use real Vertex AI RAG
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    has_real_credentials = project and project != "local-project"
    
    if not has_real_credentials:
        # Return a stub tool for local testing
        return LocalRetrievalStub()
    
    # Production: use real Vertex AI RAG
    corpus_path = corpus_resource or os.environ.get("RAG_CORPUS")
    
    return VertexAiRagRetrieval(
        name="search_documents",
        description=(
            "Search the document corpus to find relevant information. "
            "This tool performs semantic search across all uploaded documents "
            "and returns the most relevant passages based on your query."
        ),
        rag_resources=[rag.RagResource(rag_corpus=corpus_path)],
        similarity_top_k=10,
        vector_distance_threshold=0.6,
    )


class LocalRetrievalStub:
    """
    Stub retrieval tool for local testing without GCP.
    
    Returns empty results but allows agent instantiation for testing imports and logic.
    """

    def __init__(self):
        self.name = "search_documents"
        self.description = "Local stub retrieval tool (testing/no GCP access)."

    def __call__(self, query: str, **kwargs):
        """Return empty results for local testing."""
        logger.warning("Using local retrieval stub (no real search performed).")
        return {"results": [], "message": "Local mode: no documents to search."}


def build_retrieval_agent(
    model: str = "gemini-2.0-flash-001",
    corpus_resource: Optional[str] = None,
) -> Agent:
    """
    Build and configure the document retrieval agent.
    
    Args:
        model: LLM model to use for the agent.
        corpus_resource: Path to RAG corpus. If None, uses RAG_CORPUS env var.
    
    Returns:
        Configured ADK Agent instance ready for deployment or local testing.
    """
    _initialize_gcp_environment()
    
    retrieval_tool = _build_retrieval_tool(corpus_resource)
    
    agent = Agent(
        model=model,
        name="document_assistant",
        instruction=get_agent_instructions(),
        tools=[retrieval_tool],
    )
    
    return agent


# Lazy initialization: root_agent is created on first access via get_root_agent()
_root_agent_cache = None


def get_root_agent() -> Agent:
    """
    Get the module-level root agent instance, creating it lazily.
    
    This defers GCP initialization until the agent is actually needed.
    """
    global _root_agent_cache
    if _root_agent_cache is None:
        _root_agent_cache = build_retrieval_agent()
    return _root_agent_cache


# For backwards compatibility, provide root_agent as a lazy property reference
root_agent = None  # Will be set on first access


def __getattr__(name):
    """Handle lazy loading of root_agent module attribute."""
    if name == "root_agent":
        return get_root_agent()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
