import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from main import (
    generate_rfp_queries,
    process_rfp_with_deep_research,
    send_queries_to_deep_research,
)
from schema import QueriesSchema, QuerySchema, RFPSchema, QuaryGenreatorSchema
from task_manager import (
    register_task,
    process_in_background,
    get_task_status,
    update_task_stage,
)

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


def process_json(input_json):
    if (
        not input_json
        or "categories" not in input_json
        or not isinstance(input_json["categories"], list)
    ):
        return []

    result = []

    for category in input_json["categories"]:
        if (
            "subheading" in category
            and "findings" in category
            and isinstance(category["findings"], list)
        ):
            findings_content = []
            for finding in category["findings"]:
                if "question" in finding and "answer" in finding:
                    findings_content.append(
                        f"{finding['question']}\n{finding['answer']}"
                    )

            content_str = "\n\n".join(findings_content)

            output_obj = {"heading": category["subheading"], "content": content_str}

            result.append(output_obj)

    return result


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
async def api_single_query(input_data: QuerySchema, background_tasks: BackgroundTasks):
    """
    Run deep research on a single query asynchronously.
    Returns a task ID that can be used to check the status of the task.
    """
    try:
        # Register a new task with 3 stages (planning, research, writing)
        task_id = register_task(total_stages=1)

        # Add the task to the background tasks
        background_tasks.add_task(
            process_in_background,
            task_id,
            send_queries_to_deep_research,
            queries=[input_data.query],
            backend=input_data.backend,
            planner_model=input_data.planner_model,
            writer_model=input_data.writer_model,
            report_structure=input_data.report_structure,
        )

        # Return enhanced status information
        return {
            "task_id": task_id,
            "status": "PENDING",
            "stages": {
                "total_stages": 1,
                "completed_stages": 0,
                "current_stage": 1,
                "stage_results": [None],
            },
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deepresearch/batch-queries/")
async def api_batch_queries(
    input_data: QueriesSchema, background_tasks: BackgroundTasks
):
    """
    Run deep research on multiple queries in batch asynchronously.
    Returns a task ID that can be used to check the status of the task.
    """
    try:
        # Calculate total stages based on number of queries (planning, research, writing for each query)
        total_stages = len(input_data.queries)

        # Register a new task with appropriate number of stages
        task_id = register_task(total_stages=total_stages)

        # Add the task to the background tasks
        background_tasks.add_task(
            process_in_background,
            task_id,
            send_queries_to_deep_research,
            queries=input_data.queries,
            backend=input_data.backend,
            planner_model=input_data.planner_model,
            writer_model=input_data.writer_model,
            report_structure=input_data.report_structure,
        )

        # Return enhanced status information
        return {
            "task_id": task_id,
            "status": "PENDING",
            "stages": {
                "total_stages": total_stages,
                "completed_stages": 0,
                "current_stage": 1,
                "stage_results": [None] * total_stages,
            },
        }
    except Exception as e:
        logger.error(f"Error processing batch queries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deepresearch/process-complete-rfp/")
async def process_complete_rfp(
    input_data: RFPSchema, background_tasks: BackgroundTasks
):
    """
    Process a complete RFP document asynchronously.
    Returns a task ID that can be used to check the status of the task.
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

        # Register a new task with 12 stages
        task_id = register_task(total_stages=12)

        # Define a wrapper function to process results
        async def process_and_format_results():
            # Update stage: Query Generation
            update_task_stage(task_id, 0, {"status": "Generating queries from RFP"})

            # Process the RFP with deep research
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
                        not input_data.report_structure
                        or input_data.report_structure == ""
                    )
                    else input_data.report_structure
                ),
                # Pass callback for stage updates
                stage_callback=lambda stage, data: update_task_stage(
                    task_id, stage, data
                ),
            )

            # Update final stage: Formatting results
            update_task_stage(task_id, 11, {"status": "Formatting final results"})

            # Format the results
            formatted_results = process_json(
                {
                    "status": "success",
                    "title": results.get("title", "Research Report"),
                    "description": results.get("description", "Deep research analysis"),
                    "categories": results.get("categories", []),
                    "meta": results.get("meta", {}),
                }
            )

            return formatted_results

        # Add the task to the background tasks
        background_tasks.add_task(
            process_in_background, task_id, process_and_format_results
        )

        # Return enhanced status information
        return {
            "task_id": task_id,
            "status": "PENDING",
            "stages": {
                "total_stages": 12,
                "completed_stages": 0,
                "current_stage": 0,
                "stage_results": [None] * 12,
            },
        }
    except Exception as e:
        logger.error(f"Error in complete RFP processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/deepresearch/status/{task_id}")
async def get_task_status_endpoint(task_id: str):
    """
    Get the status of a task by its ID.
    Returns:
    - task_id: The ID of the task
    - status: Current status (PENDING, PROCESSING, COMPLETED, FAILED, NOT_FOUND)
    - completed_stages: Number of stages completed
    - total_stages: Total number of stages
    - current_stage: Index of the current stage being processed
    - stage_results: Results for each stage (if available)
    - result: Final result (if the task is completed)
    """
    try:
        status_info = get_task_status(task_id)

        # Extract stage information for easier access in the API response
        stage_info = status_info.get("stages", {})

        # Restructure the response for clarity
        response = {
            "task_id": status_info["task_id"],
            "status": status_info["status"],
            "completed_stages": stage_info.get("completed_stages", 0),
            "total_stages": stage_info.get("total_stages", 0),
            "current_stage": stage_info.get("current_stage", 0),
            "stage_results": stage_info.get("stage_results", []),
            "result": status_info.get("result"),
        }

        return response
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
