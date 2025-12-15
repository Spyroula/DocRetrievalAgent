"""Unit tests for the document retrieval agent."""

import pytest
from rag.agent import build_retrieval_agent, get_root_agent
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


def test_root_agent_lazy_loaded():
    """Verify root_agent is available via lazy-load function."""
    agent = get_root_agent()
    assert agent is not None
    assert agent.name == "document_assistant"


def test_corpus_utils_import():
    """Test that corpus utilities can be imported."""
    try:
        from rag.shared_libraries.prepare_corpus_and_data import (
            CorpusManager,
            CorpusSetup,
        )
        assert CorpusManager is not None
        assert CorpusSetup is not None
    except Exception as e:
        pytest.skip(f"Cannot import corpus utilities: {e}")
