"""
Interactive testing utility for deployed document retrieval agent.

Provides session-based interaction with a Vertex AI Agent Engine instance,
including streaming response handling and formatted output.
"""

import os
import sys
import json
import logging
from typing import Optional

import vertexai
from vertexai import agent_engines
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

sa_path = os.getenv("SERVICE_ACCOUNT_JSON_PATH")
if sa_path:
    if not os.path.exists(sa_path):
        logger.error("SERVICE_ACCOUNT_JSON_PATH is set but file not found: %s", sa_path)
        raise ValueError(f"SERVICE_ACCOUNT_JSON_PATH is set but file not found: {sa_path}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
    logger.info("Using service account credentials from %s", sa_path)


class SessionOutputFormatter:
    """Formats and displays agent session events."""
    
    @staticmethod
    def format_text(content: str, max_len: int = 500) -> str:
        """Truncate text content if necessary."""
        if len(content) <= max_len:
            return content
        return content[:max_len-3] + "..."
    
    @staticmethod
    def format_function_call(func_name: str, args: dict, max_args_len: int = 200) -> str:
        """Format function call with optional argument truncation."""
        args_json = json.dumps(args, indent=2)
        if len(args_json) > max_args_len:
            args_json = args_json[:max_args_len-3] + "..."
        return f"  Function: {func_name}\n  Args: {args_json}"
    
    @staticmethod
    def display_event(event: dict):
        """Display an event from the session stream."""
        if "content" not in event:
            print(f"[{event.get('author', 'system')}]: {event}")
            return
        
        author = event.get("author", "unknown")
        parts = event["content"].get("parts", [])
        
        for part in parts:
            if "text" in part:
                text = SessionOutputFormatter.format_text(part["text"])
                print(f"[{author}]: {text}\n")
            
            elif "functionCall" in part:
                call = part["functionCall"]
                formatted = SessionOutputFormatter.format_function_call(
                    call.get("name", "unknown"),
                    call.get("args", {})
                )
                print(f"[{author}] Tool Call:\n{formatted}\n")
            
            elif "functionResponse" in part:
                response = part["functionResponse"]
                content = str(response.get("content", ""))
                content = SessionOutputFormatter.format_text(content, max_len=300)
                print(f"[{author}] Tool Response: {content}\n")


class AgentTester:
    """Manages testing of deployed agent instances."""
    
    def __init__(self, project: str, location: str, agent_id: str):
        """
        Initialize agent tester.
        
        Args:
            project: GCP project ID
            location: Vertex AI location
            agent_id: Deployed agent engine resource ID
        """
        self.project = project
        self.location = location
        self.agent_id = agent_id
        
        vertexai.init(project=project, location=location)
        logger.info("Initialized tester for agent: %s", agent_id)
    
    def query(self, question: str):
        """
        Send a query to the deployed agent and stream the response.
        
        Args:
            question: Query to send to the agent
        """
        try:
            logger.info("Query: %s", question)
            print(f"\n{'='*60}")
            print(f"USER QUERY: {question}")
            print(f"{'='*60}\n")
            
            agent_engine = agent_engines.get(self.agent_id)
            
            # Query the deployed agent
            response = agent_engine.query(input=question)
            SessionOutputFormatter.display_event(response)
            
        except Exception as e:
            logger.error("Query failed: %s", e)
            raise
    
    def run_test_suite(self, test_queries: Optional[list] = None):
        """
        Run a series of test queries against the agent.
        
        Args:
            test_queries: List of test questions. Uses defaults if None.
        """
        if test_queries is None:
            test_queries = [
                "What documents do you have access to?",
                "Can you help me find information about system architecture?",
                "Explain the main features of this project",
                "Thank you, goodbye"
            ]
        
        for i, query in enumerate(test_queries, 1):
            try:
                self.query(query)
                print("\n" + "="*60 + "\n")
            except (ValueError, RuntimeError, IOError) as e:
                logger.error("Test %s failed: %s", i, e)
                continue


def _load_agent_config() -> tuple:
    """
    Load agent configuration from environment.
    
    Returns:
        Tuple of (project, location, agent_id)
        
    Raises:
        ValueError: If required configuration is missing
    """
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    agent_id = os.getenv("AGENT_ENGINE_ID")

    missing = []
    if not project:
        missing.append("GOOGLE_CLOUD_PROJECT")
    if not agent_id:
        missing.append("AGENT_ENGINE_ID")

    if missing:
        raise ValueError(
            "Missing required environment variable(s): %s. "
            "Run 'python deployment/deploy.py deploy' or set them in your .env file." % ", ".join(missing)
        )

    return project, location, agent_id


def main():
    """CLI entry point for agent testing."""
    try:
        project, location, agent_id = _load_agent_config()
        tester = AgentTester(project, location, agent_id)
        
        # Interactive mode
        print("\n" + "="*60)
        print("Document Retrieval Agent - Interactive Test")
        print("="*60)
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("Ask a question: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                if not user_input:
                    continue
                tester.query(user_input)
            except KeyboardInterrupt:
                print("\nInterrupted. Goodbye!")
                break
            except (ValueError, RuntimeError, IOError, OSError) as e:
                logger.error("Error: %s", e)
                continue
                
    except ValueError as e:
        logger.error(e)
        sys.exit(1)
    except (RuntimeError, IOError, OSError) as e:
        logger.error("Fatal error: %s", e)
        sys.exit(1)

    


if __name__ == "__main__":
    main()
