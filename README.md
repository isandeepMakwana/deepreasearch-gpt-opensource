# DeepResearch API

A FastAPI-based service for deep research on RFP documents and queries using AI capabilities.

## Features

- **Async Request Support**: Built with FastAPI to handle concurrent requests efficiently
- **Multiple Research Backends**: Supports different backends including open-deepresearch, perplexity, and standalone
- **Structured Query Generation**: Generates detailed research questions from RFP documents
- **Batch Processing**: Process multiple research queries in parallel
- **File Upload Support**: Upload RFP documents directly for processing

## API Endpoints

### Core Endpoints

- **`/generate-queries/`**: Generate structured research queries from an RFP document
- **`/single-query/`**: Run deep research on a single query
- **`/batch-queries/`**: Process multiple research queries in batch

### Complete Processing Endpoints

- **`/process-complete-rfp/`**: Takes RFP text and returns complete results after deep research
- **`/upload-and-process-rfp/`**: Accepts either a file upload or direct RFP text and returns complete results

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

Or use the provided script:

```bash
./run_api.sh
```

The API documentation will be available at: http://localhost:8000/docs

## API Usage Examples

### Generate Research Queries from RFP

```python
import requests
import json

url = "http://localhost:8000/generate-queries/"
data = {
    "rfp_text": "Your RFP text here...",
    "backend": "open-deepresearch"  # or "perplexity" or "standalone"
}

response = requests.post(url, json=data)
result = response.json()
print(json.dumps(result, indent=2))
```

### Process a Single Research Query

```python
import requests

url = "http://localhost:8000/single-query/"
data = {
    "query": "What are the latest developments in quantum computing?"
}

response = requests.post(url, json=data)
result = response.json()
print(result["report"])
```

### Process Complete RFP

```python
import requests

url = "http://localhost:8000/process-complete-rfp/"
data = {
    "rfp_text": "Your RFP text here...",
    "backend": "open-deepresearch"
}

response = requests.post(url, json=data)
result = response.json()
```

### Upload and Process RFP File

```python
import requests

url = "http://localhost:8000/upload-and-process-rfp/"
files = {"file": open("your_rfp.txt", "rb")}
data = {"backend": "open-deepresearch"}

response = requests.post(url, files=files, data=data)
result = response.json()
```

## Technical Details

- Built with FastAPI for async request handling
- Utilizes LangChain for AI capabilities
- Uses OpenAI models by default
- Optional integration with Perplexity API for enhanced research

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- LangChain
- OpenAI API credentials
- Perplexity API credentials (optional)
