# Document Retrieval Agent

An intelligent document retrieval system powered by **Google Vertex AI RAG Engine**, **Gemini 2.0 Flash**, and **Google ADK (Agent Development Kit)**. This production-ready RAG (Retrieval-Augmented Generation) agent enables users to ask natural language questions about their documents and receive accurate, citation-backed answers.

## Overview

The Document Retrieval Agent combines the power of semantic search with large language models to deliver contextually relevant answers from your document corpus. Built on Google Cloud's Vertex AI platform, it leverages:

- **Vertex AI RAG Engine**: Managed semantic search over document embeddings for fast, relevant retrieval
- **Gemini 2.0 Flash**: Advanced language model for intelligent query understanding and response generation
- **Google ADK**: Agent framework that enables the LLM to decide when to retrieve documents vs. answer directly
- **Multiple Interfaces**: Streamlit UI for end users, ADK Web for debugging, and FastAPI for programmatic access

The agent analyzes incoming queries, determines if document retrieval is needed, searches the RAG corpus for relevant passages, and synthesizes coherent responses with proper citations. It's designed for production use with local development support and flexible deployment options.

## Architecture

![Architecture Diagram](./RAG_architecture.png)

### System Flow

```
┌─────────────┐
│  User Query │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│   Gemini 2.0 Flash (ADK)    │
│   - Analyzes intent         │
│   - Decides on tool usage   │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  search_documents Tool      │
│  (VertexAiRagRetrieval)     │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Vertex AI RAG Engine       │
│  - Semantic search          │
│  - Document embeddings      │
│  - Relevance ranking        │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Retrieved Passages         │
│  (Top 10 relevant chunks)   │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Response Generation        │
│  - Synthesize answer        │
│  - Add citations            │
│  - Return to user           │
└─────────────────────────────┘
```

### Key Components

1. **Agent Layer (ADK)**
   - Orchestrates the entire retrieval and generation process
   - Uses Gemini 2.0 Flash for intelligent decision-making
   - Determines when document retrieval is necessary vs. direct response

2. **Retrieval Layer (Vertex AI RAG Engine)**
   - Manages document corpus with automatic embeddings
   - Performs semantic search over document chunks
   - Returns ranked passages based on query relevance

3. **Interface Layer**
   - **Streamlit**: Interactive chat UI for end users
   - **ADK Web**: Development and debugging interface
   - **FastAPI**: REST API for programmatic access and integrations

4. **Document Corpus**
   - Supports PDF, TXT, and MD files
   - Automatically chunked and embedded by Vertex AI
   - Currently includes 16 test documents (Google 10-Q + ML/AI research papers)

## Quick Start

### Prerequisites

Before you begin, ensure you have the following:

**Required:**
- **Python 3.11+**: The project is built and tested with Python 3.11
- **Google Cloud Project**: A GCP project with billing enabled
- **Vertex AI API**: Enabled in your GCP project
- **gcloud CLI**: Installed and authenticated
  ```bash
  # Install gcloud CLI
  # Visit: https://cloud.google.com/sdk/docs/install
  
  # Authenticate
  gcloud auth application-default login
  ```

**Package Manager:**
- **uv** (recommended): Fast Python package manager
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Alternative**: pip/venv (though uv is strongly recommended for speed)

**GCP Resources:**
- **Cloud Storage Bucket**: For staging deployment artifacts
- **Vertex AI Location**: Recommend `us-west1`, `us-east4`, or `europe-west1`
  - Note: `us-central1` may have restrictions for new projects

**Optional:**
- **Service Account Key**: For non-interactive authentication (set `SERVICE_ACCOUNT_JSON_PATH` in `.env`)

### Installation & Setup

**1. Clone the Repository**
```bash
git clone https://github.com/Spyroula/DocRetrievalAgent.git
cd DocRetrievalAgent
```

**2. Install Dependencies**
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

**3. Configure Environment**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your settings
# Required variables:
#   - GOOGLE_CLOUD_PROJECT: Your GCP project ID
#   - GOOGLE_CLOUD_LOCATION: GCP region (e.g., us-west1)
#   - STAGING_BUCKET: GCS bucket for deployments (e.g., gs://your-project-staging)
```

**4. Authenticate with Google Cloud**
```bash
# Set your active project
gcloud config set project YOUR_PROJECT_ID

# Authenticate for local development
gcloud auth application-default login
```

**5. Enable Required APIs**
```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Enable Cloud Storage API
gcloud services enable storage.googleapis.com

# Enable Cloud Resource Manager API (for deployments)
gcloud services enable cloudresourcemanager.googleapis.com
```

**6. Create Cloud Storage Staging Bucket**
```bash
# Create bucket in your preferred region
gsutil mb -l us-west1 gs://your-project-staging

# Or use gcloud
gcloud storage buckets create gs://your-project-staging --location=us-west1
```

### Prepare RAG Corpus and Upload Documents

The RAG corpus is where your documents are stored and indexed for retrieval. The corpus manager script handles creation, document upload, and embedding generation.

**Quick Start with Test Documents**
```bash
# Validate configuration (dry run)
uv run python rag/_shared_libraries/prepare_corpus_and_data.py --dry-run

# Create corpus and upload test documents
uv run python rag/_shared_libraries/prepare_corpus_and_data.py \
  --sample-dir ./test_docs \
  --display-name "My Document Corpus"
```

This will:
- Create a new RAG corpus in Vertex AI (or use existing if found)
- Upload all PDF, TXT, and MD files from `test_docs/` directory
- Automatically generate embeddings for semantic search
- Save the corpus ID to `.env` as `RAG_CORPUS`

**Upload Custom Documents**
```bash
# Upload from a specific directory
uv run python rag/_shared_libraries/prepare_corpus_and_data.py \
  --sample-dir /path/to/your/documents \
  --display-name "Company Knowledge Base"

# Upload from URLs (newline-separated file)
uv run python rag/_shared_libraries/prepare_corpus_and_data.py \
  --urls-file document_urls.txt \
  --display-name "External Resources"
```

**Programmatic Usage**
```python
from rag._shared_libraries.prepare_corpus_and_data import CorpusManager

# Initialize manager
manager = CorpusManager(
    project_id="your-project-id",
    location="us-west1"
)

# Create or get corpus
corpus = manager.get_or_create_corpus(display_name="My Corpus")

# Upload documents
manager.upload_documents(corpus, [
    "path/to/document1.pdf",
    "path/to/document2.txt"
])

# List uploaded files
files = manager.list_corpus_files(corpus)
for file in files:
    print(f"- {file.display_name}")
```

**Corpus Script Options**

```bash
uv run python rag/_shared_libraries/prepare_corpus_and_data.py [OPTIONS]
```

| Option | Description | Example |
|--------|-------------|---------|
| `--dry-run` | Validate configuration without making changes | `--dry-run` |
| `--sample-dir <path>` | Upload all PDF/TXT/MD files from directory | `--sample-dir ./test_docs` |
| `--urls-file <file>` | Upload documents from newline-separated URLs | `--urls-file urls.txt` |
| `--display-name <name>` | Set display name for the corpus | `--display-name "Company Docs"` |
| `--project-id <id>` | Override GCP project ID (defaults to .env) | `--project-id my-project` |
| `--location <region>` | Override GCP location (defaults to .env) | `--location us-east4` |

**Supported File Types**
- PDF (`.pdf`)
- Plain text (`.txt`)
- Markdown (`.md`)

**Notes:**
- Document chunking and embedding are handled automatically by Vertex AI
- Large documents are split into smaller chunks for better retrieval
- The corpus ID is automatically saved to your `.env` file
- Embeddings may take a few minutes to generate for large corpora

## Deployment

### Deploy to Vertex AI Agent Engine

Vertex AI Agent Engine provides a fully managed environment for hosting your agent with automatic scaling and monitoring.

**Prerequisites:**
- RAG corpus created and `RAG_CORPUS` set in `.env`
- Staging bucket created and `STAGING_BUCKET` set in `.env`
- Compute Engine API enabled

**Deploy the Agent:**
```bash
uv run python deployment/deploy.py deploy
```

**What Happens During Deployment:**
1. **Packaging**: The agent and all dependencies are serialized and packaged
2. **Upload**: Agent artifacts are uploaded to your Cloud Storage staging bucket
3. **Requirements**: All Python dependencies are explicitly specified (including `google-cloud-aiplatform`, `google-adk`, `google-genai`, etc.)
4. **Agent Engine Creation**: A new Reasoning Engine resource is created in Vertex AI
5. **Configuration**: The `AGENT_ENGINE_ID` is automatically saved to your `.env` file

**Deployment Process:**
```
Deploying agent...
├─ Packaging agent with dependencies
├─ Uploading to gs://your-staging-bucket/agent_engine/
│  ├─ agent_engine.pkl
│  ├─ requirements.txt
│  └─ dependencies.tar.gz
├─ Creating Agent Engine in Vertex AI
└─ Agent Engine created: projects/.../reasoningEngines/...
```

**Using the Deployed Agent:**
```python
import vertexai
from vertexai import agent_engines
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Vertex AI
vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION")
)

# Get the deployed agent
agent_engine_id = os.getenv("AGENT_ENGINE_ID")
agent_engine = agent_engines.get(agent_engine_id)

# Query the agent
response = agent_engine.query(input="What are the key findings in the documents?")
print(response)
```

**Management Commands:**
```bash
# Delete deployed agent
uv run python deployment/deploy.py delete

# Get agent status
gcloud ai agent-engines describe AGENT_ENGINE_ID \
  --project=YOUR_PROJECT \
  --location=us-west1
```

**Important Notes:**
- The deployment explicitly specifies all requirements to avoid dependency detection issues
- Initial deployment takes 3-5 minutes while the Reasoning Engine initializes
- The deployed agent uses the same RAG corpus as your local development environment
- Agent Engine provides automatic scaling based on request volume

### Testing the Deployed Agent

Once your agent is deployed to Vertex AI Agent Engine, you can test it using various methods.

**Method 1: Python Script**

Create a test script to query your deployed agent:

```python
# test_deployed_agent.py
import vertexai
from vertexai import agent_engines
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Vertex AI
vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION")
)

# Get the deployed agent
agent_engine_id = os.getenv("AGENT_ENGINE_ID")
print(f"Testing agent: {agent_engine_id}")

agent_engine = agent_engines.get(agent_engine_id)

# Test queries
test_queries = [
    "What documents are in the corpus?",
    "Summarize the key findings about machine learning",
    "What was Google's revenue in Q3 2023?"
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    response = agent_engine.query(input=query)
    print(response)
```

Run the test:
```bash
uv run python test_deployed_agent.py
```

**Method 2: Interactive Testing**

Use the Python REPL for interactive testing:

```bash
uv run python
```

```python
>>> import vertexai
>>> from vertexai import agent_engines
>>> import os
>>> 
>>> # Setup
>>> vertexai.init(project="your-project", location="us-west1")
>>> agent = agent_engines.get("projects/.../reasoningEngines/...")
>>> 
>>> # Query
>>> response = agent.query(input="What is in the corpus?")
>>> print(response)
```

**Method 3: Google Cloud Console**

1. Navigate to [Vertex AI Agent Builder](https://console.cloud.google.com/ai/agent-builder)
2. Go to **Agent Engines** section
3. Find your deployed agent (name: `document_retrieval_agent`)
4. Click on it to open the details page
5. Use the **Test** tab to send queries and view responses

**Method 4: REST API Testing**

Query the deployed agent using curl:

```bash
# Get access token
ACCESS_TOKEN=$(gcloud auth print-access-token)

# Make API request
curl -X POST \
  "https://us-west1-aiplatform.googleapis.com/v1/projects/YOUR_PROJECT/locations/us-west1/reasoningEngines/YOUR_ENGINE_ID:query" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What are the key topics in the documents?"
  }'
```

**Expected Response Format:**

```json
{
  "response": "Based on the documents in the corpus, the key topics include...",
  "citations": [
    {
      "source": "document_name.pdf",
      "page": 5,
      "text": "relevant passage..."
    }
  ]
}
```

**Monitoring and Logs:**

View agent execution logs:
```bash
# View logs in Cloud Console
gcloud logging read "resource.type=aiplatform.googleapis.com/ReasoningEngine" \
  --project=YOUR_PROJECT \
  --limit=50 \
  --format=json
```

Or visit the [Cloud Logging Console](https://console.cloud.google.com/logs).

**Performance Testing:**

```python
import time

# Measure response time
start = time.time()
response = agent_engine.query(input="Test query")
elapsed = time.time() - start

print(f"Response time: {elapsed:.2f} seconds")
```

**Troubleshooting:**
- **404 Not Found**: Verify `AGENT_ENGINE_ID` is correct and agent is deployed
- **Permission Denied**: Ensure you have `aiplatform.reasoningEngines.query` permission
- **Timeout**: First query after deployment may be slow (~30s) due to cold start
- **Empty Response**: Check that `RAG_CORPUS` contains documents and embeddings are ready

## Local Development Interfaces

The project provides three different interfaces for local development and testing, each suited for different use cases.

### Streamlit UI (End User Interface)

A user-friendly chat interface designed for end users to interact with the agent.

**Features:**
- Clean, conversational chat interface
- Message history with conversation context
- Example questions for quick testing
- Real-time streaming responses
- Citation display for retrieved documents

**Launch:**
```bash
streamlit run app.py
```

**Access:** http://localhost:8501

**Usage:**
1. Type your question in the chat input
2. Click example questions for quick testing
3. View responses with citations
4. Conversation history is maintained during the session

**Screenshot:**
```
┌─────────────────────────────────────┐
│  Document Retrieval Agent           │
├─────────────────────────────────────┤
│  Example Questions:                 │
│  • What is machine learning?        │
│  • Summarize Google's Q3 earnings   │
├─────────────────────────────────────┤
│  You: What documents are available? │
│  Agent: Based on the corpus, I found│
│         16 documents including...   │
│         [Citations: doc1.pdf, ...]  │
└─────────────────────────────────────┘
```

### ADK Web UI (Developer/Debugging Interface)

Google's Agent Development Kit web interface for debugging and tracing agent behavior.

**Features:**
- Agent execution tracing
- Tool call inspection
- Step-by-step execution breakdown
- Performance metrics
- Request/response logging

**Launch:**
```bash
# Start both Streamlit and ADK Web
./start_apps.sh

# Or launch ADK Web only
uv run adk web agents
```

**Access:** http://localhost:8001

**Usage:**
1. Select `doc_agent` from the agent dropdown
2. Enter your query in the input field
3. Click "Send" to execute
4. Inspect the execution trace:
   - Agent reasoning steps
   - Tool calls (search_documents)
   - Retrieved passages
   - Response generation

**When to Use:**
- Debugging agent behavior
- Understanding tool call decisions
- Optimizing retrieval performance
- Investigating unexpected responses

### FastAPI Backend (Programmatic Interface)

RESTful API for integrating the agent into other applications and services.

**Features:**
- Synchronous query endpoint
- Streaming response endpoint
- Health check endpoint
- OpenAPI documentation (Swagger UI)
- CORS enabled for frontend integration

**Launch:**
```bash
uvicorn api_app:app --reload
```

**Access:** http://localhost:8000

**API Documentation:** http://localhost:8000/docs

**Endpoints:**

| Method | Endpoint | Description | Request Body |
|--------|----------|-------------|--------------|
| `GET` | `/` | API information | - |
| `GET` | `/health` | Health check | - |
| `POST` | `/query` | Synchronous query | `{"question": "..."}` |
| `POST` | `/query/stream` | Streaming query | `{"question": "..."}` |

**Example Usage:**

```bash
# Health check
curl http://localhost:8000/health

# Synchronous query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'

# Streaming query (receives Server-Sent Events)
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain neural networks"}'
```

**Python Client Example:**

```python
import requests

# Query the API
response = requests.post(
    "http://localhost:8000/query",
    json={"question": "What documents discuss AI?"}
)

result = response.json()
print(result["answer"])
print(result["citations"])
```

**Streaming Client Example:**

```python
import requests

response = requests.post(
    "http://localhost:8000/query/stream",
    json={"question": "Summarize the corpus"},
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

**When to Use:**
- Building custom frontends
- Integrating with existing applications
- Automated testing and CI/CD pipelines
- Creating chatbots or voice assistants

## Customization

### Customize Agent Instructions

The agent's behavior is controlled by system instructions defined in `rag/prompts.py`.

**Edit Instructions:**
```python
# rag/prompts.py

RETRIEVAL_AGENT_INSTRUCTIONS = """
You are an expert document retrieval assistant specializing in [YOUR DOMAIN].

Your responsibilities:
1. Analyze user queries to understand their information needs
2. Use the search_documents tool to find relevant information
3. Synthesize clear, accurate answers with proper citations
4. [ADD YOUR CUSTOM INSTRUCTIONS]

Guidelines:
- Always cite sources when using retrieved information
- State clearly when information is not available in the corpus
- [ADD YOUR CUSTOM GUIDELINES]

Response format:
- Provide concise, relevant answers
- Include document citations with page numbers when available
- [ADD YOUR CUSTOM FORMAT]
"""
```

**Example Customizations:**

**For Legal Documents:**
```python
RETRIEVAL_AGENT_INSTRUCTIONS = """
You are a legal research assistant. When answering questions:
- Cite specific sections, clauses, and page numbers
- Distinguish between mandatory and discretionary language
- Highlight any conflicting provisions
- Note dates of documents for temporal relevance
"""
```

**For Technical Documentation:**
```python
RETRIEVAL_AGENT_INSTRUCTIONS = """
You are a technical documentation assistant. When answering:
- Provide code examples when relevant
- Explain technical concepts clearly
- Include version numbers and compatibility notes
- Link related topics for further reading
"""
```

**Apply Changes:**
After editing `prompts.py`, restart your applications:
```bash
# Restart Streamlit
streamlit run app.py

# Restart FastAPI
uvicorn api_app:app --reload

# For deployed agents, redeploy
uv run python deployment/deploy.py deploy
```

### Use Different LLM Models

The agent uses Gemini 2.0 Flash by default, but you can switch to other models.

**Available Gemini Models:**
- `gemini-2.0-flash`: Fast, cost-effective
- `gemini-2.5-pro`: More capable, higher cost
- `gemini-2.0-flash-001`: Balanced performance

**Change Model in Agent Definition:**

Edit `rag/agent.py`:
```python
# rag/agent.py
from google import genai

# Change the model name
root_agent = Agent(
    model="gemini-1.5-pro",  # or gemini-1.5-flash, gemini-1.0-pro
    tools=[search_documents],
    system_instruction=RETRIEVAL_AGENT_INSTRUCTIONS,
    generation_config={
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
)
```

**Model Comparison:**

| Model | Speed | Quality | Cost | Best For |
|-------|-------|---------|------|----------|
| gemini-2.0-flash-exp | ⚡⚡⚡ | ⭐⭐⭐⭐ | $ | General use, fast responses |
| gemini-1.5-pro | ⚡⚡ | ⭐⭐⭐⭐⭐ | $$$ | Complex queries, high accuracy |
| gemini-1.5-flash | ⚡⚡⚡ | ⭐⭐⭐ | $$ | Balanced performance |
| gemini-1.0-pro | ⚡⚡ | ⭐⭐⭐ | $$ | Legacy compatibility |

**Adjust Generation Parameters:**
```python
generation_config={
    "temperature": 0.0,        # Lower = more deterministic (0.0-1.0)
    "top_p": 0.9,             # Nucleus sampling threshold (0.0-1.0)
    "top_k": 20,              # Top-k sampling (1-100)
    "max_output_tokens": 4096, # Maximum response length
}
```

### Adjust RAG Retrieval Settings

Fine-tune how the agent retrieves and uses documents.

**Modify Retrieval Tool Configuration:**

Edit `rag/agent.py`:
```python
from vertexai.preview.rag import VertexRagStore, VertexRagRetrieval

# Configure retrieval settings
rag_retrieval_config = VertexRagRetrieval(
    rag_resources=[
        VertexRagStore(
            rag_corpus=os.environ["RAG_CORPUS"],
            similarity_top_k=10,        # Number of chunks to retrieve (1-50)
            vector_distance_threshold=0.3,  # Similarity threshold (0.0-1.0)
        )
    ],
)

# Apply to tool
search_documents = Tool(
    function_declarations=[rag_retrieval_config.to_tool_declaration()]
)
```

**Retrieval Parameters:**

| Parameter | Description | Default | Range | Impact |
|-----------|-------------|---------|-------|--------|
| `similarity_top_k` | Number of chunks retrieved | 10 | 1-50 | Higher = more context, slower |
| `vector_distance_threshold` | Minimum similarity score | 0.3 | 0.0-1.0 | Lower = more lenient matching |

**Tuning Guidelines:**

**For Precise Answers (High Precision):**
```python
similarity_top_k=5,
vector_distance_threshold=0.5,  # Strict matching
```

**For Comprehensive Coverage (High Recall):**
```python
similarity_top_k=20,
vector_distance_threshold=0.2,  # Lenient matching
```

**For Balanced Performance:**
```python
similarity_top_k=10,
vector_distance_threshold=0.3,  # Default balanced
```

**Test Different Settings:**
```python
# Create test configurations
configs = [
    {"top_k": 5, "threshold": 0.5},
    {"top_k": 10, "threshold": 0.3},
    {"top_k": 20, "threshold": 0.2},
]

for config in configs:
    # Update and test
    print(f"Testing: top_k={config['top_k']}, threshold={config['threshold']}")
    # Run test queries and evaluate
```

### Change Embedding Model

Vertex AI RAG Engine uses embedding models to convert documents into vectors.

**Default Embedding Model:**
- `text-embedding-004`: Google's latest embedding model (768 dimensions)

**Available Models:**
- `text-embedding-004`: Latest, highest quality (default)
- `text-embedding-preview-0409`: Preview with enhanced multilingual support
- `text-multilingual-embedding-002`: Optimized for multilingual documents

**Change During Corpus Creation:**

The embedding model is set when creating the RAG corpus. To use a different model:

```python
# rag/_shared_libraries/prepare_corpus_and_data.py

from vertexai.preview import rag

# Create corpus with specific embedding model
corpus = rag.create_corpus(
    display_name="My Corpus",
    embedding_model_config=rag.EmbeddingModelConfig(
        publisher_model="publishers/google/models/text-embedding-004"  # Change here
    )
)
```

**Model Comparison:**

| Model | Dimensions | Languages | Best For |
|-------|------------|-----------|----------|
| text-embedding-004 | 768 | English-focused | General English documents |
| text-embedding-preview-0409 | 768 | Multilingual | Experimental features |
| text-multilingual-embedding-002 | 768 | 100+ languages | Non-English documents |

**Important Notes:**
- Embedding model is fixed at corpus creation time
- Changing models requires creating a new corpus
- All documents in a corpus use the same embedding model
- Re-upload documents after creating corpus with new model

**Create New Corpus with Different Model:**
```bash
# 1. Delete old corpus (optional)
./cleanup_gcp.sh

# 2. Modify prepare_corpus_and_data.py with new embedding model
# 3. Create new corpus
uv run python rag/_shared_libraries/prepare_corpus_and_data.py \
  --sample-dir ./test_docs \
  --display-name "Multilingual Corpus"
```

**Verify Embedding Model:**
```python
from vertexai.preview import rag

corpus = rag.get_corpus(name="projects/.../ragCorpora/...")
print(f"Embedding model: {corpus.embedding_model_config.publisher_model}")
```

## Project Structure

```
DocRetrievalAgent/
├── rag/                                    # Core agent implementation
│   ├── __init__.py                         # Package initialization
│   ├── agent.py                            # ADK Agent definition with tools
│   ├── prompts.py                          # System instructions and prompts
│   └── _shared_libraries/                  # Shared utilities
│       ├── __init__.py
│       └── prepare_corpus_and_data.py      # RAG corpus management
│
├── deployment/                             # Deployment scripts
│   ├── deploy.py                           # Vertex AI Agent Engine deployment
│   └── __pycache__/                        # Python bytecode cache
│
├── tests/                                  # Unit tests
│   ├── test_agent.py                       # Agent functionality tests
│   └── __pycache__/                        # Python bytecode cache
│
├── test_docs/                              # Sample document corpus
│   ├── google_10q_q3_2023.pdf             # Google Q3 2023 10-Q filing
│   └── [15 ML/AI research papers]          # Academic papers on ML/AI
│
├── doc_retrieval_agent.egg-info/          # Package metadata
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
│
├── app.py                                  # Streamlit UI application
├── api_app.py                              # FastAPI REST API
├── start_apps.sh                           # Launch script for Streamlit + ADK Web
├── cleanup_gcp.sh                          # GCP resource cleanup script
│
├── requirements.txt                        # Python dependencies
├── pyproject.toml                          # Project configuration (uv/pip)
├── Dockerfile                              # Container definition for Cloud Run
├── .dockerignore                           # Docker build exclusions
│
├── .env.example                            # Environment variables template
├── .env                                    # Environment variables (gitignored)
├── .gitignore                              # Git exclusions
│
├── README.md                               # This file
└── RAG_architecture.png                    # Architecture diagram
```

### Key Files Explained

**Core Agent Files:**

- **`rag/agent.py`**  
  Defines the ADK Agent with Gemini 2.0 Flash, configures the `search_documents` tool, and sets up RAG retrieval. This is the heart of the agent.

- **`rag/prompts.py`**  
  Contains system instructions that define the agent's behavior, response format, and guidelines. Customize this to change how the agent interacts.

- **`rag/_shared_libraries/prepare_corpus_and_data.py`**  
  CLI tool for creating RAG corpus, uploading documents, and managing the document index. Handles all Vertex AI RAG Engine interactions.

**Interface Files:**

- **`app.py`**  
  Streamlit web application providing a user-friendly chat interface. Includes message history, example questions, and citation display.

- **`api_app.py`**  
  FastAPI REST API with synchronous and streaming endpoints. Provides `/query`, `/query/stream`, and `/health` endpoints for programmatic access.

- **`start_apps.sh`**  
  Convenience script to launch both Streamlit and ADK Web UI simultaneously with proper port management and logging.

**Deployment Files:**

- **`deployment/deploy.py`**  
  Handles deployment to Vertex AI Agent Engine. Packages the agent, uploads to Cloud Storage, creates Reasoning Engine resource, and saves configuration.

- **`Dockerfile`**  
  Container definition for Cloud Run deployment. Includes all dependencies and proper environment setup.

- **`cleanup_gcp.sh`**  
  Safe cleanup script for removing GCP resources (Agent Engine, RAG corpus, staging bucket) with confirmations.

**Configuration Files:**

- **`.env.example` / `.env`**  
  Environment variables for GCP project, location, staging bucket, RAG corpus ID, and optional service account path.

- **`requirements.txt`**  
  All Python dependencies including `google-cloud-aiplatform`, `google-adk`, `google-genai`, `streamlit`, `fastapi`, etc.

- **`pyproject.toml`**  
  Project metadata and configuration for `uv` package manager, including project name, version, and dependencies.

**Testing:**

- **`tests/test_agent.py`**  
  Unit tests verifying agent creation, instruction configuration, corpus utilities, and proper wiring of components.

**Sample Data:**

- **`test_docs/`**  
  Contains 16 sample documents (1 Google 10-Q financial report + 15 ML/AI research papers) for testing and demonstration.

### Development Workflow

```
1. Edit Code
   ├── rag/agent.py         → Modify agent behavior
   ├── rag/prompts.py       → Update instructions
   └── app.py / api_app.py  → Change UI/API

2. Test Locally
   ├── uv run pytest        → Run unit tests
   ├── streamlit run app.py → Test Streamlit UI
   └── uvicorn api_app:app  → Test FastAPI

3. Deploy
   ├── uv run python deployment/deploy.py deploy  → Deploy to Agent Engine
   └── gcloud run deploy                          → Deploy to Cloud Run
```

### Adding New Features

**Add a New Tool:**
1. Define tool function in `rag/agent.py`
2. Add to agent's `tools` list
3. Update system instructions in `rag/prompts.py`
4. Test locally, then redeploy

**Add New UI:**
1. Create new Python file (e.g., `gradio_app.py`)
2. Import `root_agent` from `rag.agent`
3. Implement interface-specific logic
4. Add to `start_apps.sh` if needed

**Add New API Endpoint:**
1. Edit `api_app.py`
2. Add new route with `@app.post()` or `@app.get()`
3. Use `root_agent.query()` for agent interaction
4. Update API documentation

## Environment Variables Reference

All configuration is managed through environment variables defined in the `.env` file.

### Required Variables

| Variable | Description | Example | Notes |
|----------|-------------|---------|-------|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | `my-project-123` | Must have billing enabled |
| `GOOGLE_CLOUD_LOCATION` | GCP region for Vertex AI | `us-west1` | Recommended: `us-west1`, `us-east4`, `europe-west1` |
| `STAGING_BUCKET` | Cloud Storage bucket for deployments | `gs://my-project-staging` | Created during setup |

### Auto-Generated Variables

These are automatically set by scripts and should not be manually edited:

| Variable | Description | Set By | Example |
|----------|-------------|--------|---------|
| `RAG_CORPUS` | Vertex AI RAG corpus resource ID | `prepare_corpus_and_data.py` | `projects/123/locations/us-west1/ragCorpora/456` |
| `AGENT_ENGINE_ID` | Deployed Agent Engine resource ID | `deployment/deploy.py` | `projects/123/locations/us-west1/reasoningEngines/789` |

### Optional Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `SERVICE_ACCOUNT_JSON_PATH` | Path to service account key file | None (uses ADC) | `/path/to/service-account.json` |

### Setup Instructions

**1. Create `.env` file:**
```bash
cp .env.example .env
```

**2. Edit required variables:**
```bash
# .env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-west1
STAGING_BUCKET=gs://your-project-staging
```

**3. Auto-generated variables are added by scripts:**
```bash
# Created by prepare_corpus_and_data.py
RAG_CORPUS=projects/131134798920/locations/us-west1/ragCorpora/1234567890

# Created by deployment/deploy.py
AGENT_ENGINE_ID=projects/131134798920/locations/us-west1/reasoningEngines/9876543210
```

### Variable Details

**`GOOGLE_CLOUD_PROJECT`**
- Your Google Cloud project ID (not the project name)
- Find it: `gcloud config get-value project`
- Must have Vertex AI API enabled
- Must have billing enabled

**`GOOGLE_CLOUD_LOCATION`**
- Region where Vertex AI resources are created
- Choose a region close to your users
- Avoid `us-central1` for new projects (may have restrictions)
- Recommended regions:
  - **us-west1** (Oregon) - West Coast US
  - **us-east4** (Virginia) - East Coast US
  - **europe-west1** (Belgium) - Europe
  - **asia-northeast1** (Tokyo) - Asia

**`STAGING_BUCKET`**
- Cloud Storage bucket for deployment artifacts
- Must include `gs://` prefix
- Must be in the same project
- Must be globally unique
- Created with: `gsutil mb -l us-west1 gs://your-bucket-name`

**`RAG_CORPUS`**
- Full resource path to your Vertex AI RAG corpus
- Automatically set when running corpus creation script
- Format: `projects/{project_id}/locations/{location}/ragCorpora/{corpus_id}`
- Do not edit manually

**`AGENT_ENGINE_ID`**
- Full resource path to deployed Agent Engine
- Automatically set when deploying to Vertex AI
- Format: `projects/{project_id}/locations/{location}/reasoningEngines/{engine_id}`
- Do not edit manually

**`SERVICE_ACCOUNT_JSON_PATH`**
- Optional: Path to service account key JSON file
- Used for non-interactive authentication
- Default: Uses Application Default Credentials (ADC)
- Setup ADC: `gcloud auth application-default login`

### Security Best Practices

**✅ DO:**
- Keep `.env` file in `.gitignore` (already configured)
- Use Application Default Credentials when possible
- Rotate service account keys regularly if used
- Use separate projects for dev/staging/prod
- Set minimal IAM permissions on service accounts

**❌ DON'T:**
- Commit `.env` to version control
- Share `.env` files via email or chat
- Hard-code credentials in source code
- Use production credentials in development
- Grant overly broad IAM permissions

### Troubleshooting

**"Environment variable not set" errors:**
```bash
# Check if .env exists
ls -la .env

# View current values (excluding sensitive data)
grep -v "SERVICE_ACCOUNT" .env

# Source .env manually for debugging
source .env
echo $GOOGLE_CLOUD_PROJECT
```

**"Invalid project ID" errors:**
```bash
# Verify project ID
gcloud projects list

# Set active project
gcloud config set project YOUR_PROJECT_ID
```

**"Bucket not found" errors:**
```bash
# List buckets
gsutil ls

# Create staging bucket
gsutil mb -l us-west1 gs://your-staging-bucket
```

**Corpus or Agent Engine ID missing:**
```bash
# Recreate corpus (sets RAG_CORPUS)
uv run python rag/_shared_libraries/prepare_corpus_and_data.py \
  --sample-dir ./test_docs \
  --display-name "My Corpus"

# Redeploy agent (sets AGENT_ENGINE_ID)
uv run python deployment/deploy.py deploy
```

## Running Tests

The project includes unit tests to verify agent configuration, corpus utilities, and system integration.

### Quick Start

Run all tests:
```bash
uv run pytest
```

Run with verbose output:
```bash
uv run pytest -v
```

Run with coverage:
```bash
uv run pytest --cov=rag --cov-report=html
```

### Test Suite Overview

**Location:** `tests/test_agent.py`

**Tests Included:**

| Test | Description | Validates |
|------|-------------|-----------|
| `test_agent_creation` | Agent instantiation | Agent object created successfully |
| `test_instructions_defined` | System instructions | Prompts are properly configured |
| `test_root_agent_exists` | Root agent module | `root_agent` variable accessible |
| `test_corpus_utils_exist` | Corpus utilities | Corpus management functions work |

### Running Specific Tests

**Run a single test:**
```bash
uv run pytest tests/test_agent.py::test_agent_creation
```

**Run tests matching a pattern:**
```bash
uv run pytest -k "agent"
```

**Run with detailed output:**
```bash
uv run pytest -vv -s
```

### Test Output

**Success:**
```
tests/test_agent.py::test_agent_creation PASSED              [ 25%]
tests/test_agent.py::test_instructions_defined PASSED        [ 50%]
tests/test_agent.py::test_root_agent_exists PASSED           [ 75%]
tests/test_agent.py::test_corpus_utils_exist PASSED          [100%]

========================= 4 passed in 2.31s ==========================
```

**Failure Example:**
```
tests/test_agent.py::test_agent_creation FAILED              [ 25%]

FAILED tests/test_agent.py::test_agent_creation - AssertionError: ...
========================= 1 failed, 3 passed in 2.45s ================
```

### Understanding the Tests

**1. Agent Creation Test**
```python
def test_agent_creation():
    """Verify agent can be instantiated"""
    agent = build_retrieval_agent()
    assert agent is not None
    assert isinstance(agent, Agent)
```
- Ensures agent can be built successfully
- Validates agent is proper ADK Agent instance

**2. Instructions Test**
```python
def test_instructions_defined():
    """Verify system instructions are configured"""
    assert RETRIEVAL_AGENT_INSTRUCTIONS is not None
    assert len(RETRIEVAL_AGENT_INSTRUCTIONS) > 0
```
- Checks that system prompts are defined
- Ensures instructions are not empty

**3. Root Agent Test**
```python
def test_root_agent_exists():
    """Verify root_agent module variable exists"""
    assert root_agent is not None
```
- Validates the pre-built agent is accessible
- Falls back to building agent if needed

**4. Corpus Utilities Test**
```python
def test_corpus_utils_exist():
    """Verify corpus management utilities load correctly"""
    # Uses importlib to load module
    assert spec is not None
    assert module is not None
```
- Tests corpus utilities can be imported
- Validates prepare_corpus_and_data.py exists

### Test Configuration

**pytest Configuration:** `pyproject.toml`
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

### Running Tests in CI/CD

**GitHub Actions Example:**
```yaml
- name: Run tests
  run: |
    uv sync
    uv run pytest -v --cov=rag
```

**Local Pre-commit Hook:**
```bash
#!/bin/bash
# .git/hooks/pre-commit
uv run pytest
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
```

### Troubleshooting Tests

**Import Errors:**
```bash
# Ensure dependencies are installed
uv sync

# Verify Python path
uv run python -c "import sys; print('\n'.join(sys.path))"
```

**Environment Variable Issues:**
```bash
# Tests use stub implementations when GCP credentials absent
# To test with real GCP:
export GOOGLE_CLOUD_PROJECT=your-project
export RAG_CORPUS=your-corpus-id
uv run pytest
```

**Corpus Utilities Import:**
- Test uses `importlib` to bypass pytest's import mechanism
- File must exist at `rag/_shared_libraries/prepare_corpus_and_data.py`
- Test passes even without GCP credentials

### Writing New Tests

**Add test to `tests/test_agent.py`:**
```python
def test_agent_has_tools():
    """Verify agent has search_documents tool"""
    agent = build_retrieval_agent()
    assert hasattr(agent, 'tools')
    assert len(agent.tools) > 0

def test_generation_config():
    """Verify generation configuration"""
    agent = build_retrieval_agent()
    assert agent.generation_config is not None
    assert agent.generation_config['temperature'] == 0.2
```

**Run new test:**
```bash
uv run pytest tests/test_agent.py::test_agent_has_tools -v
```

### Coverage Reports

**Generate HTML coverage report:**
```bash
uv run pytest --cov=rag --cov-report=html
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

**View coverage in terminal:**
```bash
uv run pytest --cov=rag --cov-report=term-missing
```

**Example output:**
```
---------- coverage: platform darwin, python 3.11.4 -----------
Name                                        Stmts   Miss  Cover   Missing
-------------------------------------------------------------------------
rag/__init__.py                                 2      0   100%
rag/agent.py                                   45      5    89%   23-27
rag/prompts.py                                  8      0   100%
rag/_shared_libraries/prepare_corpus.py       123     45    63%   45-89
-------------------------------------------------------------------------
TOTAL                                         178     50    72%
```

### Best Practices

**Before Committing:**
```bash
# Run tests
uv run pytest -v

# Check coverage
uv run pytest --cov=rag

# Format code
black rag/ tests/

# Lint code
ruff check rag/ tests/
```

**Continuous Testing:**
```bash
# Watch for changes and auto-run tests
uv run pytest-watch
```

## Docker Deployment (Optional)

Deploy the FastAPI application using Docker for containerized deployment to Cloud Run, Kubernetes, or local Docker environments.

### Prerequisites

- Docker installed and running
- `.env` file configured with required variables
- (For Cloud Run) gcloud CLI authenticated

### Build Docker Image

**Build locally:**
```bash
docker build -t doc-retrieval-agent .
```

**Build with specific tag:**
```bash
docker build -t gcr.io/YOUR_PROJECT/doc-retrieval-agent:v1.0 .
```

**Build for multiple platforms (Apple Silicon):**
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t doc-retrieval-agent .
```

### Understanding the Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY rag/ rag/
COPY api_app.py .
COPY .env.example .env

# Expose port (Cloud Run uses PORT environment variable)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the FastAPI application
CMD uvicorn api_app:app --host 0.0.0.0 --port ${PORT:-8000}
```

**Key Features:**
- Uses Python 3.11 slim base image
- Installs gcc for building Python packages
- Copies only necessary files (see `.dockerignore`)
- Includes health check for container orchestration
- Supports Cloud Run's dynamic PORT variable

### Run Container Locally

**Basic run:**
```bash
docker run -p 8000:8000 \
  --env-file .env \
  doc-retrieval-agent
```

**Run with interactive shell:**
```bash
docker run -it -p 8000:8000 \
  --env-file .env \
  doc-retrieval-agent /bin/bash
```

**Run in background:**
```bash
docker run -d -p 8000:8000 \
  --env-file .env \
  --name doc-agent \
  doc-retrieval-agent
```

**View logs:**
```bash
docker logs doc-agent
docker logs -f doc-agent  # Follow logs
```

**Stop container:**
```bash
docker stop doc-agent
docker rm doc-agent
```

### Test Docker Container

**Health check:**
```bash
curl http://localhost:8000/health
```

**Query API:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What documents are in the corpus?"}'
```

### Deploy to Google Cloud Run

**1. Build and push to Artifact Registry:**
```bash
# Configure Docker for Artifact Registry
gcloud auth configure-docker

# Build and tag
docker build -t gcr.io/YOUR_PROJECT/doc-retrieval-agent:latest .

# Push to registry
docker push gcr.io/YOUR_PROJECT/doc-retrieval-agent:latest
```

**2. Deploy to Cloud Run:**
```bash
gcloud run deploy doc-retrieval-agent \
  --image gcr.io/YOUR_PROJECT/doc-retrieval-agent:latest \
  --platform managed \
  --region us-west1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=YOUR_PROJECT,GOOGLE_CLOUD_LOCATION=us-west1,RAG_CORPUS=YOUR_CORPUS_ID
```

**Or use source-based deployment (easier):**
```bash
gcloud run deploy doc-retrieval-agent \
  --source . \
  --platform managed \
  --region us-west1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=YOUR_PROJECT,RAG_CORPUS=YOUR_CORPUS_ID
```

### Deploy to Kubernetes

**1. Create Kubernetes manifests:**

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: doc-retrieval-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: doc-retrieval-agent
  template:
    metadata:
      labels:
        app: doc-retrieval-agent
    spec:
      containers:
      - name: agent
        image: gcr.io/YOUR_PROJECT/doc-retrieval-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: GOOGLE_CLOUD_PROJECT
          value: "your-project-id"
        - name: GOOGLE_CLOUD_LOCATION
          value: "us-west1"
        - name: RAG_CORPUS
          value: "projects/.../ragCorpora/..."
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

**service.yaml:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: doc-retrieval-agent
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: doc-retrieval-agent
```

**2. Deploy to Kubernetes:**
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check status
kubectl get pods
kubectl get services

# View logs
kubectl logs -l app=doc-retrieval-agent -f
```

### Docker Compose (Local Development)

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}
      - GOOGLE_CLOUD_LOCATION=${GOOGLE_CLOUD_LOCATION}
      - RAG_CORPUS=${RAG_CORPUS}
    env_file:
      - .env
    volumes:
      - ./rag:/app/rag
      - ./api_app.py:/app/api_app.py
    restart: unless-stopped
```

**Run with Docker Compose:**
```bash
# Start services
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Optimization Tips

**Reduce Image Size:**
```dockerfile
# Use multi-stage build
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY rag/ rag/
COPY api_app.py .
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Layer Caching:**
- Copy `requirements.txt` first (changes less frequently)
- Copy source code last (changes more frequently)
- This improves rebuild times

**Use .dockerignore:**
```
# .dockerignore
__pycache__/
*.pyc
.git/
.env
.venv/
tests/
*.md
.pytest_cache/
```

### Troubleshooting

**Container fails to start:**
```bash
# Check logs
docker logs doc-agent

# Run interactively to debug
docker run -it --entrypoint /bin/bash doc-retrieval-agent
```

**Port already in use:**
```bash
# Use different port
docker run -p 8080:8000 --env-file .env doc-retrieval-agent
```

**Environment variables not working:**
```bash
# Verify .env file
cat .env

# Pass variables explicitly
docker run -p 8000:8000 \
  -e GOOGLE_CLOUD_PROJECT=your-project \
  -e RAG_CORPUS=your-corpus \
  doc-retrieval-agent
```

**Health check failing:**
```bash
# Test health endpoint inside container
docker exec doc-agent curl http://localhost:8000/health

# Check container health status
docker inspect --format='{{.State.Health.Status}}' doc-agent
```

### CI/CD Integration

**GitHub Actions Example:**
```yaml
name: Build and Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Cloud SDK
        uses: google-github-actions/setup-gcloud@v1
        
      - name: Build and push
        run: |
          docker build -t gcr.io/${{ secrets.GCP_PROJECT }}/doc-retrieval-agent:${{ github.sha }} .
          docker push gcr.io/${{ secrets.GCP_PROJECT }}/doc-retrieval-agent:${{ github.sha }}
          
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy doc-retrieval-agent \
            --image gcr.io/${{ secrets.GCP_PROJECT }}/doc-retrieval-agent:${{ github.sha }} \
            --region us-west1
```

## Troubleshooting

Common errors and their solutions when working with the Document Retrieval Agent.

### Error: "Either app or both app_name and agent must be provided"

**Problem:**  
Agent Engine deployment fails with this error during the `agent_engines.create()` call.

**Cause:**  
The ADK API changed and requires explicit app/agent parameters, or the agent object is not properly configured.

**Solution:**

**1. Verify deployment script has explicit requirements:**
```python
# deployment/deploy.py
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

remote_app = agent_engines.create(
    agent,
    display_name=display_name,
    requirements=requirements,  # Explicitly specify
)
```

**2. Check agent is properly built:**
```python
from rag.agent import root_agent
print(f"Agent type: {type(root_agent)}")
print(f"Agent has tools: {hasattr(root_agent, 'tools')}")
```

**3. Update dependencies:**
```bash
uv sync --upgrade
```

### Error: "RAG_CORPUS environment variable not set"

**Problem:**  
Application fails to start with missing `RAG_CORPUS` variable.

**Cause:**  
RAG corpus hasn't been created yet, or `.env` file is not properly configured.

**Solution:**

**1. Check if .env exists:**
```bash
ls -la .env
cat .env | grep RAG_CORPUS
```

**2. Create RAG corpus:**
```bash
uv run python rag/_shared_libraries/prepare_corpus_and_data.py \
  --sample-dir ./test_docs \
  --display-name "My Document Corpus"
```

This will automatically:
- Create a new RAG corpus in Vertex AI
- Upload documents from the specified directory
- Save the corpus ID to `.env` as `RAG_CORPUS`

**3. Verify corpus ID format:**
```bash
# Should look like:
RAG_CORPUS=projects/123456789/locations/us-west1/ragCorpora/987654321
```

**4. Manual setup if needed:**
```python
from vertexai.preview import rag
import os

# List existing corpora
corpora = rag.list_corpora()
for corpus in corpora:
    print(f"Corpus: {corpus.name}")
    
# Use existing corpus
# Add to .env: RAG_CORPUS=projects/.../ragCorpora/...
```

### Error: "For new projects, RAG Engine in us-central1 is restricted"

**Problem:**  
Cannot create RAG corpus in `us-central1` region with new GCP projects.

**Cause:**  
Google has regional restrictions for new projects in `us-central1`.

**Solution:**

**1. Change to a supported region:**
```bash
# Edit .env
GOOGLE_CLOUD_LOCATION=us-west1  # or us-east4, europe-west1
```

**Recommended regions:**
- `us-west1` (Oregon) - Best for West Coast US
- `us-east4` (Virginia) - Best for East Coast US  
- `europe-west1` (Belgium) - Best for Europe
- `asia-northeast1` (Tokyo) - Best for Asia

**2. Recreate corpus in new region:**
```bash
# Update .env with new location
# Then recreate corpus
uv run python rag/_shared_libraries/prepare_corpus_and_data.py \
  --sample-dir ./test_docs \
  --display-name "My Corpus"
```

**3. Update all resources to same region:**
```bash
# Ensure consistency
gcloud config set compute/region us-west1
```

### Error: "ResourceExhausted" during document upload

**Problem:**  
Document upload fails with `ResourceExhausted` or quota exceeded errors.

**Cause:**  
Vertex AI embedding generation has rate limits and quotas.

**Solutions:**

**1. Check current quotas:**
```bash
gcloud compute quotas list \
  --project=YOUR_PROJECT \
  --filter="metric:aiplatform.googleapis.com"
```

**2. Request quota increase:**
- Visit [GCP Quotas page](https://console.cloud.google.com/iam-admin/quotas)
- Filter for "Vertex AI API"
- Select "Embeddings requests per minute"
- Click "Edit Quotas" and request increase

**3. Upload documents in smaller batches:**
```python
from rag._shared_libraries.prepare_corpus_and_data import CorpusManager
import time

manager = CorpusManager(project_id="your-project", location="us-west1")
corpus = manager.get_or_create_corpus(display_name="My Corpus")

# Upload in batches with delays
documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf", ...]
batch_size = 5

for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    manager.upload_documents(corpus, batch)
    print(f"Uploaded batch {i//batch_size + 1}")
    time.sleep(60)  # Wait between batches
```

**4. Reduce document size:**
```bash
# Split large PDFs
pdftk large.pdf burst output page_%04d.pdf

# Compress PDFs
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
   -dNOPAUSE -dQUIET -dBATCH \
   -sOutputFile=compressed.pdf input.pdf
```

### Error: "Could not retrieve GCP credentials"

**Problem:**  
Application cannot authenticate with Google Cloud.

**Cause:**  
No valid credentials found via Application Default Credentials (ADC) or service account key.

**Solutions:**

**1. Setup Application Default Credentials:**
```bash
gcloud auth application-default login
```

**2. Verify credentials:**
```bash
gcloud auth application-default print-access-token
```

**3. Use service account key (alternative):**
```bash
# Download service account key from GCP Console
# Save as service-account.json

# Add to .env
SERVICE_ACCOUNT_JSON_PATH=/path/to/service-account.json

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

**4. Check IAM permissions:**

Required roles for the service account or user:
- `roles/aiplatform.user` - Use Vertex AI services
- `roles/storage.objectAdmin` - Access Cloud Storage
- `roles/iam.serviceAccountUser` - Use service accounts

```bash
# Grant roles
gcloud projects add-iam-policy-binding YOUR_PROJECT \
  --member="user:your-email@example.com" \
  --role="roles/aiplatform.user"
```

**5. Docker/Container environments:**
```bash
# Mount credentials into container
docker run -p 8000:8000 \
  -v ~/.config/gcloud:/root/.config/gcloud \
  --env-file .env \
  doc-retrieval-agent
```

### Agent not responding to queries

**Problem:**  
Agent receives queries but doesn't return responses, or returns empty responses.

**Possible Causes & Solutions:**

**1. RAG corpus is empty:**
```python
from vertexai.preview import rag

corpus = rag.get_corpus(name=os.getenv("RAG_CORPUS"))
files = rag.list_files(corpus=corpus.name)
print(f"Number of documents: {len(list(files))}")

# If empty, upload documents
uv run python rag/_shared_libraries/prepare_corpus_and_data.py \
  --sample-dir ./test_docs
```

**2. Embeddings not yet generated:**
```bash
# Wait 5-10 minutes after upload for embeddings to be created
# Check corpus status in console:
# https://console.cloud.google.com/vertex-ai/rag
```

**3. Query too vague:**
```python
# Bad query
response = agent.query(input="hi")

# Good query
response = agent.query(input="What are the main topics in the Google 10-Q filing?")
```

**4. Check agent configuration:**
```python
from rag.agent import root_agent

print(f"Model: {root_agent.model}")
print(f"Tools: {root_agent.tools}")
print(f"Instructions: {root_agent.system_instruction[:100]}...")
```

**5. Enable debug logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run query and check logs
response = root_agent.query(input="test query")
```

### Deployment timeout

**Problem:**  
Agent Engine deployment times out after 10-15 minutes.

**Cause:**  
Large dependencies, cold start, or Agent Engine initialization issues.

**Solutions:**

**1. Check deployment logs:**
```bash
gcloud logging read "resource.type=aiplatform.googleapis.com/ReasoningEngine" \
  --project=YOUR_PROJECT \
  --limit=50 \
  --format=json
```

**2. Verify requirements are correct:**
```python
# In deployment/deploy.py
requirements = [
    "google-cloud-aiplatform[adk,agent-engines]>=1.108.0",
    "google-adk>=1.10.0",
    "google-genai>=0.1.0",
    # ... all required packages
]
```

**3. Redeploy with clean state:**
```bash
# Delete existing deployment
uv run python deployment/deploy.py delete

# Wait 2 minutes
sleep 120

# Deploy fresh
uv run python deployment/deploy.py deploy
```

**4. Use Cloud Run instead (faster):**
```bash
# Cloud Run deploys in 2-3 minutes vs 5-10 for Agent Engine
gcloud run deploy doc-retrieval-agent \
  --source . \
  --region us-west1 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=YOUR_PROJECT,RAG_CORPUS=YOUR_CORPUS
```

**5. Check Agent Engine status:**
```bash
# List all reasoning engines
gcloud ai reasoning-engines list \
  --location=us-west1 \
  --project=YOUR_PROJECT
```

### General Debugging Tips

**Enable verbose logging:**
```bash
export GOOGLE_SDK_PYTHON_LOGGING_DEBUG=1
uv run python your_script.py
```

**Test components individually:**
```bash
# Test corpus access
uv run python -c "from vertexai.preview import rag; print(rag.list_corpora())"

# Test agent creation
uv run python -c "from rag.agent import root_agent; print(root_agent)"

# Test API endpoints
curl http://localhost:8000/health
```

**Check GCP service status:**
- Visit [Google Cloud Status Dashboard](https://status.cloud.google.com/)
- Check for Vertex AI or RAG Engine incidents

**Contact support:**
```bash
# Include these details in support requests:
echo "Project: $(gcloud config get-value project)"
echo "Location: $GOOGLE_CLOUD_LOCATION"
echo "Python: $(python --version)"
echo "SDK: $(gcloud version)"
```

## Performance Metrics

Understanding the performance characteristics of your Document Retrieval Agent helps optimize for speed, cost, and quality.

### Response Time Benchmarks

Typical response times for the Document Retrieval Agent (Gemini 2.0 Flash):

| Component | Cold Start | Warm Start | Notes |
|-----------|-----------|------------|-------|
| **Local Development** | 2-3s | 0.5-1.5s | First query loads model |
| **Agent Engine** | 30-45s | 1-2s | First query after deployment |
| **Cloud Run** | 5-10s | 0.5-1.5s | Container cold start |
| **RAG Retrieval** | - | 200-500ms | Semantic search |
| **LLM Generation** | - | 500-1500ms | Response synthesis |

**Total end-to-end latency:**
- **Warm:** 1-3 seconds (typical usage)
- **Cold:** 5-45 seconds (first query or after idle)

### Measuring Response Time

**Python Client:**
```python
import time
from rag.agent import root_agent

# Measure single query
start = time.time()
response = root_agent.query(input="What documents are available?")
elapsed = time.time() - start

print(f"Response time: {elapsed:.2f}s")
```

**API Endpoint:**
```bash
# Using curl with timing
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test query"}' \
  -w "\nTotal time: %{time_total}s\n"
```

**Comprehensive Benchmark Script:**
```python
import time
import statistics
from rag.agent import root_agent

queries = [
    "What is machine learning?",
    "Summarize Google's Q3 2023 earnings",
    "Explain transformer architecture",
    "What are the main topics in the corpus?",
]

response_times = []

for query in queries:
    start = time.time()
    response = root_agent.query(input=query)
    elapsed = time.time() - start
    response_times.append(elapsed)
    print(f"Query: {query[:40]}... - {elapsed:.2f}s")

print(f"\nAverage: {statistics.mean(response_times):.2f}s")
print(f"Median: {statistics.median(response_times):.2f}s")
print(f"Min: {min(response_times):.2f}s")
print(f"Max: {max(response_times):.2f}s")
```

### Throughput Metrics

**Single Instance:**
- **Requests per minute:** 20-30 (sequential)
- **Concurrent requests:** 5-10 (with async)
- **Max tokens/minute:** ~50,000 (Gemini 2.0 Flash)

**Agent Engine (Auto-scaling):**
- **Initial capacity:** 1-2 concurrent requests
- **Max auto-scale:** 10+ instances
- **Scales based on:** Request volume and latency

**Cloud Run (Configured):**
```bash
gcloud run deploy doc-retrieval-agent \
  --min-instances 1 \          # Always warm
  --max-instances 100 \        # Auto-scale up to 100
  --concurrency 10             # 10 requests per instance
```

### Cost Analysis

**Vertex AI RAG Engine:**
| Component | Unit Cost | Typical Usage | Monthly Est. |
|-----------|-----------|---------------|--------------|
| Embedding generation | $0.00001/1K chars | 1M chars (initial) | $0.01 |
| Storage (embeddings) | $0.10/GB/month | 1GB corpus | $0.10 |
| Retrieval queries | $0.001/1K queries | 100K queries | $100 |

**Gemini 2.0 Flash:**
| Metric | Cost | Notes |
|--------|------|-------|
| Input tokens | $0.075/1M tokens | Context + query |
| Output tokens | $0.30/1M tokens | Generated response |
| Free tier | 1,500 req/day | First 60 days |

**Example monthly cost (10K queries):**
```
Retrieval: 10,000 queries × $0.001/1K = $10.00
LLM Input: 10,000 × 1,000 tokens × $0.075/1M = $0.75
LLM Output: 10,000 × 500 tokens × $0.30/1M = $1.50
Storage: 1GB corpus = $0.10
Total: ~$12.35/month
```

**Cloud Run (if used):**
- **Free tier:** 2M requests/month
- **Beyond free:** $0.40/million requests
- **Memory:** $0.0000025/GB-second
- **CPU:** $0.00002400/vCPU-second

### Optimizing Performance

**1. Reduce Cold Starts**

**Cloud Run:**
```bash
# Keep minimum instances warm
gcloud run deploy doc-retrieval-agent \
  --min-instances 1 \
  --region us-west1
```

**Agent Engine:**
- First query after deployment is always slow (~30s)
- Subsequent queries are fast (1-2s)
- Use a scheduled warmup query

**Warmup Script:**
```bash
# cron job to keep agent warm
*/5 * * * * curl -X POST https://your-agent-url/query \
  -H "Content-Type: application/json" \
  -d '{"question": "ping"}'
```

**2. Optimize Retrieval Settings**

**Reduce number of retrieved chunks:**
```python
# In rag/agent.py
rag_retrieval_config = VertexRagRetrieval(
    rag_resources=[
        VertexRagStore(
            rag_corpus=os.environ["RAG_CORPUS"],
            similarity_top_k=5,  # Reduced from 10 (faster)
            vector_distance_threshold=0.4,  # More strict matching
        )
    ],
)
```

**Impact:**
- `similarity_top_k=5`: ~200ms retrieval time
- `similarity_top_k=20`: ~500ms retrieval time

**3. Use Streaming Responses**

**FastAPI streaming:**
```python
# Reduces perceived latency - user sees first tokens immediately
response = requests.post(
    "http://localhost:8000/query/stream",
    json={"question": "long query"},
    stream=True
)

for line in response.iter_lines():
    print(line.decode('utf-8'), end='', flush=True)
```

**4. Batch Similar Queries**

**Cache frequent questions:**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_response(question: str):
    return root_agent.query(input=question)
```

**5. Adjust Generation Parameters**

**Faster but less thorough:**
```python
generation_config={
    "temperature": 0.0,      # More deterministic
    "max_output_tokens": 512, # Shorter responses
}
```

**Slower but higher quality:**
```python
generation_config={
    "temperature": 0.7,       # More creative
    "max_output_tokens": 2048, # Longer responses
}
```

### Quality Metrics

**Retrieval Quality:**
```python
# Test retrieval accuracy
def evaluate_retrieval(test_cases):
    """
    test_cases = [
        {"query": "...", "expected_docs": ["doc1.pdf", "doc2.pdf"]},
        ...
    ]
    """
    correct = 0
    total = len(test_cases)
    
    for case in test_cases:
        response = root_agent.query(input=case["query"])
        # Check if expected docs are cited
        retrieved_docs = extract_citations(response)
        if any(doc in retrieved_docs for doc in case["expected_docs"]):
            correct += 1
    
    precision = correct / total
    print(f"Retrieval precision: {precision:.2%}")
    return precision
```

**Response Quality:**
- **Relevance:** Does it answer the question?
- **Accuracy:** Is the information correct?
- **Citations:** Are sources properly cited?
- **Completeness:** Does it cover all aspects?

**Manual evaluation template:**
```
Query: "..."
Response: "..."

Relevance: 1-5 ⭐
Accuracy: 1-5 ⭐
Citations: 1-5 ⭐
Completeness: 1-5 ⭐

Notes: ...
```

### Monitoring in Production

**Cloud Monitoring Metrics:**
```bash
# View Agent Engine metrics
gcloud monitoring time-series list \
  --project=YOUR_PROJECT \
  --filter='metric.type="aiplatform.googleapis.com/reasoning_engine/request_count"'
```

**Key metrics to track:**
- **Request count:** Total queries
- **Error rate:** Failed queries / total
- **P50/P95/P99 latency:** Response time percentiles
- **Token usage:** Input + output tokens
- **Cost per query:** Total cost / queries

**Set up alerts:**
```bash
# Alert on high error rate
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="Agent High Error Rate" \
  --condition-threshold-value=0.05 \
  --condition-threshold-duration=300s
```

**Custom logging:**
```python
import logging
import time

logger = logging.getLogger(__name__)

def query_with_logging(question: str):
    start = time.time()
    try:
        response = root_agent.query(input=question)
        duration = time.time() - start
        logger.info(f"Query successful | duration={duration:.2f}s | question={question[:50]}")
        return response
    except Exception as e:
        duration = time.time() - start
        logger.error(f"Query failed | duration={duration:.2f}s | error={str(e)}")
        raise
```

### Performance Comparison

**Model comparison (approximate):**

| Model | Speed | Quality | Cost | Best For |
|-------|-------|---------|------|----------|
| Gemini 2.5 Flash | ⚡⚡⚡ | ⭐⭐⭐⭐ | $ | Production, high volume |
| Gemini 2.5 Pro | ⚡⚡ | ⭐⭐⭐⭐⭐ | $$$ | Complex queries, accuracy critical |
| Gemini 2.0 Flash | ⚡⚡⚡ | ⭐⭐⭐ | $$ | Balanced use cases |

**Deployment comparison:**

| Deployment | Cold Start | Warm Latency | Scaling | Cost |
|------------|-----------|--------------|---------|------|
| Local Dev | 2-3s | 0.5-1.5s | Manual | Free (API only) |
| Agent Engine | 30-45s | 1-2s | Auto | Low (managed) |
| Cloud Run | 5-10s | 0.5-1.5s | Auto | Low (pay per use) |
| Kubernetes | 10-30s | 0.5-1.5s | Manual/HPA | Medium (infra) |


## References

- [Vertex AI RAG Engine Docs](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-overview)
- [Google ADK Documentation](https://docs.cloud.google.com/agent-builder/agent-development-kit/overview)
- [Gemini API Reference](https://ai.google.dev/api)

