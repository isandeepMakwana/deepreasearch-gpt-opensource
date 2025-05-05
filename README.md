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

- **`/deepresearch/generate-queries/`**: Generate structured research queries from an RFP document
- **`/deepresearch/display-queries/`**: Generate and display queries with additional metadata
- **`/deepresearch/single-query/`**: Run deep research on a single query
- **`/deepresearch/batch-queries/`**: Process multiple research queries in batch

### Complete Processing Endpoints

- **`/deepresearch/process-complete-rfp/`**: Takes RFP text and returns complete results after deep research
- **`/deepresearch/upload-and-process-rfp/`**: Accepts either a file upload or direct RFP text and returns complete results

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
    "backend": "open-deepresearch",
    "model_name": "o3-mini",
    "temperature": 1.0
}

response = requests.post(url, json=data)
result = response.json()
print(json.dumps(result, indent=2))
```

### Display Generated Queries with Metadata

```python
import requests
import json

url = "http://localhost:8000/deepresearch/display-queries/"
data = {
    "rfp_text": "Your RFP text here...",
    "backend": "open-deepresearch",
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
    "model_name": "gpt-4o-mini",
    "temperature": 0.7
}

response = requests.post(url, json=data)
result = response.json()
print(result["report"])
```

### Process Complete RFP

```python
import requests

url = "http://localhost:8000/deepresearch/process-complete-rfp/"
data = {
    "rfp_text": "Your RFP text here...",
    "backend": "open-deepresearch",
    "model_name": "o3-mini",
    "temperature": 1.0
}

response = requests.post(url, json=data)
result = response.json()
```

### Upload and Process RFP File

```python
import requests

url = "http://localhost:8000/deepresearch/upload-and-process-rfp/"
files = {"file": open("your_rfp.txt", "rb")}
data = {
    "backend": "open-deepresearch",
    "model_name": "o3-mini",
    "temperature": 1.0,
    "planner_model": "gpt-4o-mini",
    "writer_model": "gpt-4o-mini"
}

response = requests.post(url, files=files, data=data)
result = response.json()
```

## Technical Details

- Built with FastAPI for async request handling
- Utilizes LangChain for AI capabilities
- Supports multiple AI model selection
- Uses OpenAI models by default
- Optional integration with Perplexity API for enhanced research
- Includes Tavily search integration

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- LangChain
- OpenAI API credentials
- Perplexity API credentials (optional)
- Tavily API credentials (optional)

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
   - **Error Handling**: Centralized error handling and logging

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

### Failover Mechanism

The backend selection follows a priority order with automatic failover:
1. Attempts to use Open-DeepResearch first
2. Falls back to Perplexity if available and Open-DeepResearch fails
3. Uses standalone research as a final fallback
