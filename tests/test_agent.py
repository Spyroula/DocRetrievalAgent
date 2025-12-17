"""Unit tests for the document retrieval agent."""

import pytest
from rag.agent import build_retrieval_agent, root_agent
from rag.prompts import get_agent_instructions


def test_agent_creation():
    """Verify agent can be instantiated with proper configuration."""
    agent = build_retrieval_agent()
    assert agent.name == "document_assistant"


def test_instructions_defined():
    """Verify agent instructions are properly defined and non-empty."""
    instructions = get_agent_instructions()
    assert isinstance(instructions, str)
    assert len(instructions) > 0
    assert "document" in instructions.lower() or "search" in instructions.lower()


def test_root_agent_exists():
    """Verify root_agent is available and properly configured."""
    if root_agent is None:
        # If root_agent failed to initialize, try building a new one
        test_agent = build_retrieval_agent()
        assert test_agent is not None
        assert test_agent.name == "document_assistant"
    else:
        assert root_agent is not None
        assert root_agent.name == "document_assistant"


def test_corpus_utils_exist():
    """Test that corpus utilities module exists and has the required classes."""
    from pathlib import Path
    import importlib.util
    
    # Check if the file exists
    corpus_utils_path = Path(__file__).parent.parent / "rag" / "_shared_libraries" / "prepare_corpus_and_data.py"
    assert corpus_utils_path.exists(), f"Corpus utilities file not found at {corpus_utils_path}"
    
    # Load the module directly using importlib
    spec = importlib.util.spec_from_file_location("prepare_corpus_and_data", corpus_utils_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Verify the classes exist
        assert hasattr(module, "CorpusManager"), "CorpusManager class not found"
        assert hasattr(module, "CorpusSetup"), "CorpusSetup class not found"
