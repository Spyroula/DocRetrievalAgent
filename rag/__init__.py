"""Document Retrieval Agent with Vertex AI RAG Engine."""

import os
from dotenv import load_dotenv

load_dotenv()

project_env = os.getenv("GOOGLE_CLOUD_PROJECT")
if not project_env:
    try:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError
        _, project_id = google.auth.default()
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
    except (ImportError, DefaultCredentialsError):
        pass

os.environ.setdefault("GOOGLE_CLOUD_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

# Do not import `agent` here to avoid import-time side effects.
