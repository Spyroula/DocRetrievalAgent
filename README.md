# Document Retrieval Agent with Vertex AI RAG Engine

A conversational AI agent that answers questions about documents using Retrieval-Augmented Generation (RAG). Built with Google's Agent Development Kit (ADK) and Vertex AI RAG Engine.

## Overview

This agent leverages Vertex AI's RAG capabilities to retrieve relevant documents and synthesize answers with proper citations. It demonstrates ML engineering best practices including:

- **Document Management**: Upload and manage document collections in RAG corpus
- **Semantic Search**: Retrieve relevant passages using embeddings
- **Citation Tracking**: Provide source references for all retrieved information
- **Cloud-Native Architecture**: Deploy to Vertex AI Agent Engine for scalability
- **Local Testing**: Run without GCP credentials using built-in stubs

**Status Note**: Corpus creation and agent functionality are fully working. Agent Engine deployment is experiencing API compatibility issues with the latest ADK version (documented in Deployment section).

## Architecture

```
User Query
    ↓
ADK Agent
    ↓
VertexAiRagRetrieval Tool
    ↓
Vertex AI RAG Engine (Corpus + Embeddings)
    ↓
Retrieved Passages
    ↓
Gemini LLM (Synthesis + Citations)
    ↓
Final Response with Citations
```

## Quick Start

### Prerequisites

- **Google Cloud Project** with billing enabled
- **Python 3.10+**
- **uv** package manager: https://docs.astral.sh/uv/

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Spyroula/DocRetrievalAgent.git
   cd DocRetrievalAgent
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Set up environment variables:**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env with your GCP project details
   # Required:
   # - GOOGLE_CLOUD_PROJECT=your-project-id
   # - GOOGLE_CLOUD_LOCATION=us-west1 (recommended: us-west1, us-east4, or europe-west1)
   #   Note: us-central1 may be restricted for new projects
   # - STAGING_BUCKET=gs://your-staging-bucket-name
   ```

4. **Authenticate with Google Cloud:**
   ```bash
   gcloud auth application-default login
   ```
   
   Or use a service account key:
   ```bash
   export SERVICE_ACCOUNT_JSON_PATH=/path/to/service-account-key.json
   ```

5. **Prepare RAG corpus and upload documents:**
   
   The repository includes a sample document in `test_docs/` directory. You can use it to test the agent:
   
   ```bash
   # Validate configuration (dry-run)
   uv run python rag/shared_libraries/prepare_corpus_and_data.py --dry-run
   
   # Create corpus and upload the sample document from test_docs/
   uv run python rag/shared_libraries/prepare_corpus_and_data.py \
     --sample-dir ./test_docs \
     --display-name "My Document Corpus"
   ```
   
   To add your own documents:
   ```bash
   # Add your PDF/TXT/MD files to test_docs/
   cp /path/to/your/document.pdf test_docs/
   
   # Then run the corpus creation command again
   ```
   
   This will:
   - Create a new RAG corpus in Vertex AI
   - Upload all PDF/TXT/MD files from `test_docs/`
   - Generate embeddings for semantic search
   - Save the corpus ID to `.env` as `RAG_CORPUS`
   
   **Corpus script options:**
   - `--dry-run`: Validate setup without uploading
   - `--sample-dir <path>`: Upload all PDFs/TXT/MD files from a directory
   - `--urls-file <path>`: Download and upload documents from URLs (newline-separated file)
   - `--display-name <name>`: Custom corpus name (default: DocumentCorpus)
   - `--no-upload`: Create corpus but skip document uploads

### Local Testing (Without GCP)

The agent can be instantiated and tested locally without a Google account:

```bash
# Tests run in local mode automatically
uv run pytest -q

# Run agent in Python with local stubs
python -c "from rag.agent import build_retrieval_agent; agent = build_retrieval_agent(); print('Agent ready:', agent.name)"
```

### Local Testing

Run the agent locally using the ADK CLI:

```bash
# Interactive CLI mode
adk run rag

# Web UI mode
adk web
# Then select the RAG agent from the dropdown
```

## Deployment

### Deploy to Vertex AI Agent Engine

⚠️ **Known Issue**: Agent Engine deployment is currently experiencing API compatibility issues with the latest ADK version. The corpus creation and agent instantiation work correctly, but the deployment to Agent Engine encounters an API mismatch error ("Either app or both app_name and agent must be provided").

**Current Status:**
- ✅ Corpus creation and document indexing: Working
- ✅ Local agent instantiation and testing: Working
- ⚠️ Agent Engine deployment: API compatibility issue

**Deployment command** (may require ADK version update):
```bash
# Deploy the agent
uv run python deployment/deploy.py deploy

# This will:
# - Package the agent with dependencies
# - Attempt to deploy to Vertex AI Agent Engine
# - Update .env with AGENT_ENGINE_ID (if successful)
```

**Alternative Deployment Options:**
1. **Wait for ADK API update** - Google is actively updating the Agent Engine APIs
2. **Pin to older ADK version** - Specify a compatible version in `pyproject.toml` (Note: This may introduce other compatibility issues or missing features)

### Test Deployed Agent

```bash
uv run python deployment/run.py
```

This will:
- Create a session with the deployed agent
- Send test queries
- Display responses with streaming updates

### Delete Deployed Agent

```bash
uv run python deployment/deploy.py delete
```

## Corpus Preparation

### Automatic Setup (Recommended)

The repository includes a sample document in the `test_docs/` directory for quick testing.

**Create corpus with the included sample document:**
```bash
# Validate configuration (dry-run)
uv run python rag/shared_libraries/prepare_corpus_and_data.py --dry-run

# Create corpus and upload the sample document
uv run python rag/shared_libraries/prepare_corpus_and_data.py \
  --sample-dir ./test_docs \
  --display-name "My Document Corpus"
```

**To add your own documents:**
```bash
# Add your PDF, TXT, or MD files to test_docs/
cp /path/to/your/documents/*.pdf test_docs/

# Re-run the corpus creation (will upload all files in test_docs/)
uv run python rag/shared_libraries/prepare_corpus_and_data.py \
  --sample-dir ./test_docs \
  --display-name "My Document Corpus"
```

**Alternative: Upload from URLs**
```bash
uv run python rag/shared_libraries/prepare_corpus_and_data.py --urls-file document_urls.txt
```

**Expected output:**
```
Corpus created: projects/.../ragCorpora/...
Uploading: test_docs/document1.pdf
Uploaded: test_docs/document1.pdf
Corpus setup completed successfully
RAG_CORPUS saved to .env
```

**CLI Options:**
- `--dry-run`: Validate setup without uploading
- `--sample-dir <path>`: Upload all PDFs/TXT/MD files from a directory
- `--urls-file <path>`: Download and upload documents from URLs (newline-separated)
- `--display-name <name>`: Custom corpus name (default: DocumentCorpus)
- `--no-upload`: Create corpus but skip document uploads

### Manual Corpus Management

```python
from rag.shared_libraries.prepare_corpus_and_data import CorpusManager

# Initialize
manager = CorpusManager(
    project_id="your-project",
    location="us-central1"
)

# Create or get corpus
corpus = manager.get_or_create_corpus(display_name="MyCorpus")

# Upload documents
manager.upload_documents(corpus, ["path/to/doc1.pdf", "path/to/doc2.pdf"])

# List files
files = manager.list_corpus_files(corpus)
print(f"Corpus contains {len(files)} documents")
```

### Supported Document Types
- **PDF files** (.pdf)
- **Text files** (.txt)
- **Markdown** (.md)

## Cloud Deployment

### Deploy to Vertex AI Agent Engine

```bash
# 1. Ensure environment is configured
cat .env  # Verify variables are set

# 2. Deploy agent
uv run python deployment/deploy.py

# This will:
# - Package the agent with dependencies
# - Deploy to Vertex AI Agent Engine
# - Update .env with AGENT_ENGINE_ID
```

### Verify Deployment

```bash
# Check deployed agents
gcloud ai agent-engines list --location=us-central1

# Get agent details
gcloud ai agent-engines describe <AGENT_ENGINE_ID> --location=us-central1
```

### Delete Deployed Agent

```bash
# Get agent ID from .env
AGENT_ID=$(grep AGENT_ENGINE_ID .env | cut -d= -f2)

# Delete (via deployment script)
uv run python deployment/deploy.py --delete
```

## Testing Deployed Agent

### Interactive Testing

```bash
# Start interactive testing
uv run python deployment/run.py

# Example queries:
# > What is RAG?
# > Explain the main concepts
# > Tell me about document retrieval
# > quit
```

### Programmatic Testing

```python
from deployment.run import AgentTester

tester = AgentTester(
    project="your-project",
    location="us-central1",
    agent_id="projects/.../agentEngines/..."
)

# Send a query
tester.query("What is machine learning?")
```

## Customization

### Customize Agent Instructions

Edit `rag/prompts.py` to modify:
- Agent behavior and personality
- Citation format requirements
- Tool usage guidelines
- Response structure

### Use Different LLM Models

In `rag/agent.py`, change the model parameter:

```python
agent = Agent(
    model='gemini-2.5-pro',  # or other available models
    # ... rest of configuration
)
```

Available models:
- `gemini-2.0-flash` (default) - Fast, cost-effective
- `gemini-2.0-flash-001` - Specific version
- `gemini-2.5-pro` - Higher capability

### Adjust RAG Retrieval Settings

In `rag/agent.py`:

```python
retrieval_tool = VertexAiRagRetrieval(
    similarity_top_k=10,              # Number of documents to retrieve
    vector_distance_threshold=0.6,    # Relevance threshold (0-1)
)
```

### Change Embedding Model

Default: `text-embedding-004` (set in `prepare_corpus_and_data.py`)

To use different embeddings:
```python
embedding_model_config = rag.EmbeddingModelConfig(
    publisher_model="publishers/google/models/text-embedding-005"
)
```

## Project Structure

```
DocRetrievalAgent/
├── rag/
│   ├── __init__.py                       # Package initialization
│   ├── agent.py                          # Agent factory and root_agent
│   ├── prompts.py                        # Agent instructions
│   └── shared_libraries/
│       └── prepare_corpus_and_data.py    # Corpus management CLI
├── deployment/
│   ├── deploy.py                         # Deployment to Agent Engine
│   └── run.py                            # Interactive agent testing
├── tests/
│   └── test_agent.py                     # Unit tests
├── pyproject.toml                        # Project dependencies & config
├── .env.example                          # Environment variables template
├── requirements.txt                      # Dependencies (reference)
└── README.md                             # This file
```

## Environment Variables Reference

| Variable | Required | Example | Description |
|---|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | Yes | `my-project-123` | GCP Project ID |
| `GOOGLE_CLOUD_LOCATION` | No | `us-central1` | Vertex AI location (default: us-central1) |
| `STAGING_BUCKET` | Yes | `gs://my-bucket` | GCS bucket for deployment artifacts |
| `RAG_CORPUS` | Auto-set | `projects/123/.../ragCorpora/abc` | RAG corpus resource ID |
| `AGENT_ENGINE_ID` | Auto-set | `projects/123/.../agentEngines/xyz` | Deployed agent ID |
| `SERVICE_ACCOUNT_JSON_PATH` | Optional | `/path/to/key.json` | Service account key file path |

## Running Tests

### Unit Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test
uv run pytest tests/test_agent.py::test_agent_creation -v

# With coverage report
uv run pytest tests/ --cov=rag
```

### Local Testing (Without GCP)

The agent includes built-in stubs for local testing without a Google Cloud account:

```bash
# Tests automatically use local mode
uv run pytest -q

# Instantiate agent locally
python -c "from rag.agent import build_retrieval_agent; agent = build_retrieval_agent(); print('Ready:', agent.name)"
```

## Docker Deployment (Optional)

### Build and Push Image

```bash
# Build
docker build -t doc-retrieval-agent:latest .

# Tag for GCP
docker tag doc-retrieval-agent:latest gcr.io/YOUR_PROJECT/doc-retrieval-agent:latest

# Push to Container Registry
docker push gcr.io/YOUR_PROJECT/doc-retrieval-agent:latest
```

### Deploy to Cloud Run (Alternative)

```bash
gcloud run deploy doc-retrieval-agent \
  --image gcr.io/YOUR_PROJECT/doc-retrieval-agent:latest \
  --platform managed \
  --region us-central1 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=YOUR_PROJECT
```

## Troubleshooting

### Error: "Either app or both app_name and agent must be provided" (Agent Engine Deployment)
**Issue**: Agent Engine API compatibility issue with current ADK version.

**Context**: 
- Corpus creation works perfectly
- Agent instantiation succeeds locally
- Deployment to Agent Engine encounters internal API mismatch

**Solutions**:
1. **Short-term**: Use Cloud Run deployment (see Docker section)
2. **Monitor updates**: Check ADK release notes for API fixes
3. **Pin versions**: Specify compatible ADK version in dependencies
4. **Alternative**: Enable Cloud Resource Manager API if not already enabled:
   ```bash
   gcloud services enable cloudresourcemanager.googleapis.com
   ```

### Error: "RAG_CORPUS environment variable not set"
**Solution**: Run `prepare_corpus_and_data.py` first to create and register your corpus.

### Error: "For new projects, RAG Engine in us-central1 is restricted"
**Solution**: Some regions have capacity limits for new projects. Try a different region:
```bash
# In .env, change:
GOOGLE_CLOUD_LOCATION=us-west1
# or us-east4, europe-west1, asia-northeast1
```

### Error: "ResourceExhausted" during document upload
**Solution**: You've hit the embedding model quota. Request a quota increase:
1. Go to Google Cloud Console → Quotas
2. Search for "Vertex AI API"
3. Select embedding model quota
4. Request increase

### Error: "Could not retrieve GCP credentials"
**Solution**: This is expected if you don't have a Google Cloud account. The agent uses local stubs by default. For cloud deployment:
1. Create a Google Cloud project (or use an existing one)
2. Run `gcloud auth application-default login`
3. Or set `SERVICE_ACCOUNT_JSON_PATH` to a service account key

### Agent not responding to queries
**Solution**: Check that:
1. `RAG_CORPUS` is set correctly in `.env`
2. The corpus contains documents (verify via GCP Console)
3. Google Cloud permissions are configured correctly

### Deployment timeout
**Solution**: 
- Ensure `STAGING_BUCKET` exists and is accessible
- Check that your GCP project has sufficient quota
- Verify network connectivity and IAM permissions

## Performance Metrics

### Typical Performance
- **Query latency**: 2-5 seconds
- **Retrieved documents**: Top 10 most relevant chunks
- **Citation accuracy**: Depends on document structure

### Optimization Tips
1. Adjust `similarity_top_k` for speed/quality tradeoff
2. Use smaller documents (chunk before uploading)
3. Increase `vector_distance_threshold` to filter low-relevance results

## References

- [Vertex AI RAG Engine Docs](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-overview)
- [Google ADK Documentation](https://docs.cloud.google.com/agent-builder/agent-development-kit/overview)
- [Gemini API Reference](https://ai.google.dev/api)


