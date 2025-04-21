import os
import json
import logging
from typing import Dict, List, Optional, Union, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import existing functions from main.py
from main import (
    generate_rfp_queries,
    send_queries_to_deep_research,
    standalone_research,
    process_rfp_with_deep_research,
    extract_json_from_text
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="DeepResearch API",
    description="API for processing RFPs with deep research capabilities",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request/response models
class RFPInput(BaseModel):
    rfp_text: str
    backend: str = "open-deepresearch"

class QueryInput(BaseModel):
    query: str
    backend: str = "open-deepresearch"

class QueriesInput(BaseModel):
    queries: List[str]
    backend: str = "open-deepresearch"

class ResearchQueryModel(BaseModel):
    heading: str = Field(description="The heading or category of the query")
    subheading: Optional[str] = Field(None, description="The subheading of the query, if applicable")
    questions: List[str] = Field(description="List of specific research questions for this heading/subheading")

class ResearchPlanModel(BaseModel):
    title: str = Field(description="Title of the research plan")
    description: str = Field(description="Brief description of the research objectives")
    queries: List[ResearchQueryModel] = Field(description="Structured research queries")

# Routes
@app.post("/generate-queries/")
async def api_generate_queries(input_data: RFPInput):
    """
    Generate structured research queries from an RFP document.
    """
    try:
        result = generate_rfp_queries(input_data.rfp_text)
        return result
    except Exception as e:
        logger.error(f"Error generating queries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/single-query/")
async def api_single_query(input_data: QueryInput):
    """
    Run deep research on a single query.
    """
    try:
        result = await standalone_research(input_data.query)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-queries/")
async def api_batch_queries(input_data: QueriesInput):
    """
    Run deep research on multiple queries in batch.
    """
    try:
        results = await send_queries_to_deep_research(input_data.queries, input_data.backend)
        return results
    except Exception as e:
        logger.error(f"Error processing batch queries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-complete-rfp/")
async def process_complete_rfp(input_data: RFPInput):
    """
    Process an RFP document with deep research and return the complete final result.
    
    This endpoint takes an RFP text, generates queries, runs deep research, and returns
    the compiled final result with all findings.
    """
    try:
        logger.info("Starting complete RFP processing")
        # Process the RFP with deep research
        results = await process_rfp_with_deep_research(input_data.rfp_text, input_data.backend)
        
        # Return the complete results
        return {
            "status": "success",
            "title": results.get("title", "Research Report"),
            "description": results.get("description", "Deep research analysis"),
            "categories": results.get("categories", []),
            "meta": results.get("meta", {})
        }
    except Exception as e:
        logger.error(f"Error in complete RFP processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-and-process-rfp/")
async def upload_and_process_rfp(
    file: Optional[UploadFile] = File(None),
    rfp_text: Optional[str] = Form(None),
    backend: str = Form("open-deepresearch")
):
    """
    Process an RFP document from either an uploaded file or direct text input.
    Returns the complete results after deep research is done.
    """
    try:
        # Determine the RFP text source
        if file is not None:
            # Read the file content
            contents = await file.read()
            rfp_content = contents.decode("utf-8")
            logger.info(f"Processing RFP from uploaded file: {file.filename}")
        elif rfp_text is not None:
            # Use the provided text directly
            rfp_content = rfp_text
            logger.info("Processing RFP from text input")
        else:
            # No input provided
            raise HTTPException(
                status_code=400, 
                detail="Either a file upload or RFP text must be provided"
            )
        
        # Process the RFP with deep research
        results = await process_rfp_with_deep_research(rfp_content, backend)
        
        # Return the complete results
        return {
            "status": "success",
            "title": results.get("title", "Research Report"),
            "description": results.get("description", "Deep research analysis"),
            "categories": results.get("categories", []),
            "meta": results.get("meta", {})
        }
    except Exception as e:
        logger.error(f"Error processing RFP: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
