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
    title="Advanced Research API",
    description="API for advanced RFP processing and research capabilities",
    version="2.0.0",
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
    model_name: str = "o3-mini"
    planner_model: Optional[str] = "o4-mini"
    writer_model: Optional[str] = "o3-mini"
    temperature: float = 1.0
    report_structure: Optional[str] = None


class QueryInput(BaseModel):
    query: str
    backend: str = "open-deepresearch"
    model_name: str = "gpt-4o-mini"
    temperature: float = 1.0

class QueriesInput(BaseModel):
    queries: List[str]
    backend: str = "open-deepresearch"
    planner_model: str = "gpt-4o-mini"
    writer_model: str = "gpt-4o-mini"

class ResearchQueryModel(BaseModel):
    heading: str = Field(description="The heading or category of the query")
    subheading: Optional[str] = Field(None, description="The subheading of the query, if applicable")
    questions: List[str] = Field(description="List of specific research questions for this heading/subheading")

class ResearchPlanModel(BaseModel):
    title: str = Field(description="Title of the research plan")
    description: str = Field(description="Brief description of the research objectives")
    queries: List[ResearchQueryModel] = Field(description="Structured research queries")

# Routes
@app.post("/deepresearch/generate-queries/")
async def api_generate_queries(input_data: RFPInput):
    """
    Generate structured research queries from an RFP document.
    """
    try:
        result = generate_rfp_queries(
            rfp_text=input_data.rfp_text,
            model_name=input_data.model_name,
            temperature=input_data.temperature
        )
        return result
    except Exception as e:
        logger.error(f"Error generating queries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deepresearch/display-queries/")
async def display_generated_queries(input_data: RFPInput):
    """
    Generate and display structured research queries from an RFP document without running the full research process.
    
    This endpoint is similar to generate-queries but is specifically designed for displaying the generated queries
    in a more user-friendly format with additional metadata.
    """
    try:
        logger.info("Generating queries for display")
        # Use the existing function to generate queries
        result = generate_rfp_queries(
            rfp_text=input_data.rfp_text,
            model_name=input_data.model_name,
            temperature=input_data.temperature
        )
        # Add additional metadata for display purposes
        enhanced_result = { # Simple hash for demo purposes
            "title": result.get("title", "Research Plan"),
            "description": result.get("description", "Generated research queries"),
            "queries": result.get("queries", []),
            "status": "completed",
            "rfp_length": len(input_data.rfp_text),
            "model_used": input_data.model_name
        }
        
        return enhanced_result
    except Exception as e:
        logger.error(f"Error displaying generated queries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deepresearch/single-query/")
async def api_single_query(input_data: QueryInput):
    """
    Run deep research on a single query.
    """
    try:
        # Modify standalone_research to support model parameters
        result = await standalone_research(
            query=input_data.query,
            model_name=input_data.model_name,
            temperature=input_data.temperature
        )
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deepresearch/batch-queries/")
async def api_batch_queries(input_data: QueriesInput):
    """
    Run deep research on multiple queries in batch.
    """
    try:
        # Pass the model parameters to the underlying function
        results = await send_queries_to_deep_research(
            queries=input_data.queries, 
            backend=input_data.backend,
            planner_model=input_data.planner_model,
            writer_model=input_data.writer_model
        )
        return results
    except Exception as e:
        logger.error(f"Error processing batch queries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deepresearch/process-complete-rfp/")
async def process_complete_rfp(input_data: RFPInput):
    """
    Process an RFP document with deep research and return the complete final result.
    
    This endpoint takes an RFP text, generates queries, runs deep research, and returns
    the compiled final result with all findings.
    """
    try:
        logger.info("Starting complete RFP processing")
        REPORT_STRUCTURE = """
            1. Topic Overview
            • What is the topic?
            • Why is it important?

            2. Key Insights (grouped by sub-topic)
            • Sub-topic 1:
                - Key Point A
                - Key Point B
            • Sub-topic 2:
                - Key Point A
                - Key Point B
            • Sub-topic 3 (if needed):
                - Key Point A
                - Key Point B
        """
        # Process the RFP with deep research, now with model parameters
        results = await process_rfp_with_deep_research(
            rfp_text=input_data.rfp_text, 
            backend=input_data.backend,
            model_name=input_data.model_name,
            temperature=input_data.temperature,
            planner_model=input_data.planner_model,
            writer_model=input_data.writer_model,
            report_structure=REPORT_STRUCTURE if not input_data.report_structure else input_data.report_structure

        )
        
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

@app.post("/deepresearch/upload-and-process-rfp/")
async def upload_and_process_rfp(
    file: Optional[UploadFile] = File(None),
    rfp_text: Optional[str] = Form(None),
    backend: str = Form("open-deepresearch"),
    model_name: str = Form("o3-mini"),
    temperature: float = Form(1.0),
    planner_model: str = Form("gpt-4o-mini"),
    writer_model: str = Form("gpt-4o-mini")
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
        
        # Process the RFP with deep research, including model parameters
        results = await process_rfp_with_deep_research(
            rfp_text=rfp_content, 
            backend=backend,
            model_name=model_name,
            temperature=temperature,
            planner_model=planner_model,
            writer_model=writer_model
        )
        
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
