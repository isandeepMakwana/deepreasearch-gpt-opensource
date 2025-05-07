import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from main import (
    generate_rfp_queries,
    process_rfp_with_deep_research,
    send_queries_to_deep_research,
)
from schema import QueriesSchema, QuerySchema, RFPSchema, QuaryGenreatorSchema

# Configure logging
from logger_config import setup_logger

logger = setup_logger(__name__)

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


# Routes
@app.post("/deepresearch/generate-queries/")
async def display_generated_queries(input_data: QuaryGenreatorSchema):
    """
    Generate structured research queries from an RFP document without running the full research process.

    This endpoint is similar to generate-queries but is specifically designed for displaying the generated queries
    in a more user-friendly format with additional metadata.
    """
    try:
        logger.info("Generating queries for display")
        # Use the existing function to generate queries
        result = generate_rfp_queries(
            rfp_text=input_data.rfp_text,
            model_name=input_data.model_name,
            temperature=input_data.temperature,
        )
        # Add additional metadata for display purposes
        enhanced_result = {  # Simple hash for demo purposes
            "title": result.get("title", "Research Plan"),
            "description": result.get("description", "Generated research queries"),
            "queries": result.get("queries", []),
            "status": "completed",
            "rfp_length": len(input_data.rfp_text),
            "model_used": input_data.model_name,
        }

        return enhanced_result
    except Exception as e:
        logger.error(f"Error displaying generated queries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deepresearch/single-query/")
async def api_single_query(input_data: QuerySchema):
    """
    Run deep research on a single query.
    """
    try:
        results = await send_queries_to_deep_research(
            queries=[input_data.query],
            backend=input_data.backend,
            planner_model=input_data.planner_model,
            writer_model=input_data.writer_model,
            report_structure=input_data.report_structure,
        )
        return results
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deepresearch/batch-queries/")
async def api_batch_queries(input_data: QueriesSchema):
    """
    Run deep research on multiple queries in batch.
    """
    try:
        # Pass the model parameters to the underlying function
        results = await send_queries_to_deep_research(
            queries=input_data.queries,
            backend=input_data.backend,
            planner_model=input_data.planner_model,
            writer_model=input_data.writer_model,
            report_structure=input_data.report_structure,
        )
        return results
    except Exception as e:
        logger.error(f"Error processing batch queries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deepresearch/process-complete-rfp/")
async def process_complete_rfp(input_data: RFPSchema):
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
            report_structure=(
                REPORT_STRUCTURE
                if (
                    not input_data.report_structure or input_data.report_structure == ""
                )
                else input_data.report_structure
            ),
        )

        # Return the complete results
        return {
            "status": "success",
            "title": results.get("title", "Research Report"),
            "description": results.get("description", "Deep research analysis"),
            "categories": results.get("categories", []),
            "meta": results.get("meta", {}),
        }
    except Exception as e:
        logger.error(f"Error in complete RFP processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
