# DeepResearch API

An advanced FastAPI-based service for deep research on RFP documents and queries using AI capabilities.

## Features

- **Async Request Support**: Built with FastAPI to handle concurrent requests efficiently
- **Multiple Research Backends**: Supports different backends including open-deepresearch, perplexity, and standalone
- **Structured Query Generation**: Generates detailed research questions from RFP documents
- **Batch Processing**: Process multiple research queries in parallel
- **File Upload Support**: Upload RFP documents directly for processing
- **Model Selection**: Choose AI models for different parts of the research process

## API Endpoints

### Core Endpoints

- **`/deepresearch/generate-queries/`**: Generate structured research queries from an RFP document with additional metadata for display
- **`/deepresearch/single-query/`**: Run deep research on a single query
- **`/deepresearch/batch-queries/`**: Process multiple research queries in batch

### Complete Processing Endpoints

- **`/deepresearch/process-complete-rfp/`**: Takes RFP text and initiates asynchronous deep research, returning a task ID
- **`/deepresearch/status/{task_id}`**: Checks the status of an asynchronous task and returns results when complete

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd deepresearch-try
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root with the following contents:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PERPLEXITY_API_KEY=your_perplexity_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

## Running the API Server

Start the FastAPI server with:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API documentation will be available at: http://localhost:8000/docs

## API Usage Examples

### Generate Research Queries from RFP

```python
import requests
import json

url = "http://localhost:8000/deepresearch/generate-queries/"
data = {
    "rfp_text": "Your RFP text here...",
    "model_name": "o3-mini",
    "temperature": 1.0
}

response = requests.post(url, json=data)
result = response.json()
print(json.dumps(result, indent=2))
```

### Process a Single Research Query

```python
import requests

url = "http://localhost:8000/deepresearch/single-query/"
data = {
    "query": "What are the latest developments in quantum computing?",
    "backend": "open-deepresearch",
    "planner_model": "gpt-4o-mini",
    "writer_model": "gpt-4o-mini",
    "report_structure": "1. Overview\n2. Key Developments\n3. Future Directions"
}

response = requests.post(url, json=data)
result = response.json()
print(result)
```

### Process Multiple Queries in Batch

```python
import requests

url = "http://localhost:8000/deepresearch/batch-queries/"
data = {
    "queries": [
        "What are the latest developments in quantum computing?",
        "How is AI being used in healthcare?"
    ],
    "backend": "open-deepresearch",
    "planner_model": "gpt-4o-mini",
    "writer_model": "gpt-4o-mini",
    "report_structure": "1. Overview\n2. Key Points\n3. Conclusion"
}

response = requests.post(url, json=data)
result = response.json()
print(result)
```

### Process Complete RFP

```python
import requests

url = "http://localhost:8000/deepresearch/process-complete-rfp/"
data = {
    "rfp_text": "Your RFP text here...",
    "backend": "open-deepresearch",
    "model_name": "o3-mini",
    "temperature": 1.0,
    "planner_model": "gpt-4o-mini",
    "writer_model": "gpt-4o-mini",
    "report_structure": "1. Topic Overview\n2. Key Insights\n3. Recommendations"
}

response = requests.post(url, json=data)
result = response.json()
print(result)
```

## Technical Details

- Built with FastAPI for async request handling
- Utilizes LangChain for AI capabilities
- Supports multiple AI model selection
- Uses OpenAI models by default
- Optional integration with Perplexity API for enhanced research
- Includes Tavily search integration
- Comprehensive logging system with rotation policies

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- LangChain
- OpenAI API credentials
- Perplexity API credentials (optional)
- Tavily API credentials (optional)

## Logging System

The DeepResearch API includes a comprehensive logging system that captures important events and errors:

- **Log Location**: Logs are stored in the `logs/deepresearch.log` file within the project directory
- **Log Rotation**: Automatic log rotation is implemented to prevent excessive file sizes
  - Maximum log file size: 10MB
  - Backup count: 5 files
- **Log Format**: `timestamp - module_name - log_level - message`
- **Output Destinations**: All logs are sent to both console and log file
- **Log Levels**: The system uses standard Python logging levels (INFO, WARNING, ERROR, etc.)

The logging system helps with debugging, monitoring API performance, and tracking errors during operation.

## System Architecture

The DeepResearch API follows a modular architecture designed for extensibility, concurrent processing, and robust research capabilities:

```
┌──────────────────────────────────────────────────────────────────┐
│                        DeepResearch API                          │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                       FastAPI Application                        │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │  API Endpoints  │  │ Request Models  │  │  Error Handling │   │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘   │
│           │                    │                    │            │
└───────────┼────────────────────┼────────────────────┼────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Core Processing Logic                      │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │ Query Generator │  │ Research Engine │  │ Results Compiler│   │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘   │
│           │                    │                    │            │
└───────────┼────────────────────┼────────────────────┼────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Research Backends                         │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │Open-DeepResearch│  │    Perplexity   │  │   Standalone    │   │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘   │
│           │                    │                    │            │
└───────────┼────────────────────┼────────────────────┼────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                        External Services                         │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │   OpenAI API    │  │ Perplexity API  │  │   Tavily API    │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Component Descriptions

1. **API Layer (FastAPI Application)**
   - **API Endpoints**: RESTful endpoints exposed for clients to interact with the system
   - **Request Models**: Pydantic models for data validation and serialization
   - **Error Handling**: Centralized error handling and logging system with file rotation

2. **Core Processing Logic**
   - **Query Generator**: Analyzes RFP documents and generates structured research queries
   - **Research Engine**: Processes research queries through selected backends
   - **Results Compiler**: Aggregates and formats research results into comprehensive reports

3. **Research Backends**
   - **Open-DeepResearch**: Primary backend using LangGraph for advanced research capabilities
   - **Perplexity**: Alternative backend leveraging Perplexity API for web-based research
   - **Standalone**: Fallback backend using direct LLM calls for research

4. **External Services**
   - **OpenAI API**: Provides language model capabilities for query generation and research
   - **Perplexity API**: Specialized research API for comprehensive web-based information retrieval
   - **Tavily API**: Search API for sourcing relevant information

### Data Flow

1. **RFP Processing Flow**:
   ```
   RFP Document → Query Generation → Research Queries → Research Engine →
   Multiple Backend Processing → Results Aggregation → Final Report
   ```

2. **Single Query Flow**:
   ```
   Research Query → Research Engine → Backend Selection →
   External API Calls → Results Formatting → Response
   ```

3. **Batch Query Flow**:
   ```
   Multiple Queries → Concurrent Processing → Individual Research →
   Results Collection → Consolidated Response
   ```

### Async Implementation

The system leverages FastAPI's async capabilities to:
1. Handle multiple concurrent user requests
2. Process batch queries in parallel
3. Maintain responsiveness during long-running research operations
4. Execute long-running tasks in the background while immediately returning a task ID

#### Asynchronous Task Processing

All API endpoints that perform research operations now work asynchronously:

1. When a request is received, a unique task ID is generated and returned immediately
2. The actual processing happens in the background using FastAPI's BackgroundTasks
3. Clients can poll the `/deepresearch/status/{task_id}` endpoint to check task status
4. Once processing is complete, the full results are available through the status endpoint

#### Example Usage

```python
# Start a research task
response = requests.post(
    "http://localhost:8001/deepresearch/process-complete-rfp/",
    json={"rfp_text": "Your RFP document here", "backend": "open-deepresearch"}
)
task_id = response.json()["task_id"]

# Check status until complete
while True:
    status_response = requests.get(f"http://localhost:8001/deepresearch/status/{task_id}")
    status_data = status_response.json()

    if status_data["status"] == "COMPLETED":
        # Process the results
        results = status_data["result"]
        break
    elif status_data["status"] == "FAILED":
        # Handle error
        print(f"Task failed: {status_data.get('result', {}).get('error')}")
        break

    # Wait before polling again
    time.sleep(5)
```

### Failover Mechanism

The backend selection follows a priority order with automatic failover:
1. Attempts to use Open-DeepResearch first
2. Falls back to Perplexity if available and Open-DeepResearch fails
3. Uses standalone research as a final fallback
