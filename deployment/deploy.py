"""
Deployment utilities for the document retrieval agent.

Handles provisioning and managing agent instances on Vertex AI Agent Engine,
including creation, deletion, and configuration updates.
"""

import logging
import os
import sys
import argparse
from pathlib import Path

import vertexai
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp
from google.api_core.exceptions import NotFound
from dotenv import set_key

from rag.agent import root_agent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# If a service account key path is provided via env, prefer it for authentication
sa_path = os.getenv("SERVICE_ACCOUNT_JSON_PATH")
if sa_path:
    if not os.path.exists(sa_path):
        raise ValueError(
            "SERVICE_ACCOUNT_JSON_PATH is set but file not found: %s" % sa_path
        )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
    logger.info("Using service account credentials from %s", sa_path)


class AgentDeploymentManager:
    """Manages lifecycle of agent deployment on Vertex AI."""

    def __init__(self, project: str, location: str, staging_bucket: str):
        """
        Initialize deployment manager with GCP configuration.

        Args:
            project: GCP project ID
            location: Vertex AI location (e.g., us-central1)
            staging_bucket: GCS bucket for staging deployment artifacts
        """
        self.project = project
        self.location = location
        self.staging_bucket = staging_bucket

        vertexai.init(
            project=project,
            location=location,
            staging_bucket=staging_bucket,
        )
        logger.info("Initialized deployment manager for %s/%s", project, location)

    def deploy(self, agent, display_name: str = "document_retrieval_agent") -> str:
        """
        Deploy agent to Vertex AI Agent Engine.

        Args:
            agent: ADK Agent instance to deploy
            display_name: Display name for the deployed agent

        Returns:
            Resource ID of the deployed agent engine
        """
        try:
            logger.info("Deploying agent: %s", display_name)

            # Explicitly specify requirements to ensure all dependencies are included
            # The automatic detection misses some critical packages like vertexai
            requirements = [
                "google-cloud-aiplatform[adk,agent-engines]>=1.108.0",
                "google-adk>=1.10.0",
                "google-genai>=0.1.0",
                "google-auth>=2.25.0",
                "google-cloud-storage>=2.10.0",
                "python-dotenv>=1.0.0",
                "pydantic>=2.0.0",
                "cloudpickle>=3.0.0",
            ]

            # Deploy agent using agent_engines.create
            remote_app = agent_engines.create(
                agent,
                display_name=display_name,
                requirements=requirements,
            )

            agent_engine_id = remote_app.resource_name
            logger.info("Successfully deployed agent: %s", agent_engine_id)
            return agent_engine_id

        except Exception as e:
            logger.error("Deployment failed: %s", e)
            raise

    def delete(self, agent_engine_id: str) -> bool:
        """
        Delete a deployed agent engine.

        Args:
            agent_engine_id: Resource ID of the agent to delete

        Returns:
            True if deletion was successful
        """
        try:
            logger.info("Deleting agent: %s", agent_engine_id)
            agent_engines.delete(agent_engine_id)
            logger.info("Agent deleted successfully")
            return True
        except NotFound:
            logger.warning("Agent not found: %s", agent_engine_id)
            return False
        except Exception as e:
            logger.error("Deletion failed: %s", e)
            raise


def _load_configuration() -> tuple:
    """Load and validate deployment configuration from environment.

    Returns:
        Tuple of (project, location, staging_bucket)

    Raises:
        ValueError: If required configuration is missing
    """
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    bucket = os.getenv("STAGING_BUCKET")

    missing = []
    if not project:
        missing.append("GOOGLE_CLOUD_PROJECT")
    if not bucket:
        missing.append("STAGING_BUCKET")

    if missing:
        raise ValueError(
            "Missing required environment variable(s): %s. "
            "Set them in your .env or export them in your shell. See README for details."
            % ", ".join(missing)
        )

    return project, location, bucket


def _save_deployment_config(env_file_path: str, agent_engine_id: str):
    """Persist deployment configuration to .env file.

    Args:
        env_file_path: Path to .env configuration file
        agent_engine_id: Resource ID of deployed agent
    """
    set_key(env_file_path, "AGENT_ENGINE_ID", agent_engine_id)
    logger.info("Configuration saved: AGENT_ENGINE_ID=%s", agent_engine_id)


def main():
    """CLI entry point for agent deployment operations."""
    parser = argparse.ArgumentParser(
        prog="deploy", description="Manage document retrieval agent deployment"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Deploy subcommand
    deploy_parser = subparsers.add_parser("deploy", help="Deploy agent to cloud")
    deploy_parser.set_defaults(func=_deploy_agent)

    # Delete subcommand
    delete_parser = subparsers.add_parser("delete", help="Remove deployed agent")
    delete_parser.add_argument("agent_id", help="Resource ID of agent to delete")
    delete_parser.set_defaults(func=_delete_agent)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except (ValueError, FileNotFoundError, OSError) as e:
        logger.error("Operation failed: %s", e)
        sys.exit(1)


def _deploy_agent(_args):
    """Internal function to handle deployment logic."""
    project, location, staging_bucket = _load_configuration()
    env_file = Path(__file__).parent.parent / ".env"

    manager = AgentDeploymentManager(project, location, staging_bucket)
    agent_engine_id = manager.deploy(root_agent)
    _save_deployment_config(str(env_file), agent_engine_id)

    print("\n✓ Deployment successful!")
    print(f"  Agent Engine ID: {agent_engine_id}")
    print("\nNext steps:")
    print("  1. Test locally: python deployment/run.py")
    print("  2. Or use the deployed agent with the resource ID above")


def _delete_agent(args):
    """Internal function to handle deletion logic."""
    project, location, staging_bucket = _load_configuration()

    manager = AgentDeploymentManager(project, location, staging_bucket)
    success = manager.delete(args.agent_id)

    if success:
        print("✓ Agent deleted successfully")
    else:
        print("✗ Agent deletion failed or agent not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
