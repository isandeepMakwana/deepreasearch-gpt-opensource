import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Celery
from celery.result import AsyncResult
from .celery_app import celery_app

# Import the schema and tasks
from .schema import RFPInput
from .tasks import process_rfp_with_deep_research_task
from .logger_config import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Deep Research Async API",
    description="Submit an RFP document for deep research asynchronously, then poll status.",
    version="1.0.0",
)

# Add CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/deepresearch/process-complete-rfp/")
def process_complete_rfp(input_data: RFPInput):
    """
    Submits a long-running 'deep research' task to Celery.
    Returns a task_id that can be used to check status.
    """
    logger.info("Received RFP for processing...")

    # Launch Celery task (returns AsyncResult object)
    task_result = process_rfp_with_deep_research_task.delay(
        rfp_text=input_data.rfp_text,
        backend=input_data.backend,
        model_name=input_data.model_name,
        planner_model=input_data.planner_model,
        writer_model=input_data.writer_model,
        temperature=input_data.temperature,
        report_structure=input_data.report_structure
        or "concised Report contaning key finding in bullet points",
    )

    return {"status": "submitted", "task_id": task_result.id}


@app.get("/tasks/status/{task_id}")
def get_task_status(task_id: str):
    """
    Poll the status of a Celery task. Returns progress or final result if done.
    """
    result = celery_app.AsyncResult(task_id)

    if result.state == "PENDING":
        return {"status": "PENDING", "progress": 0}
    elif result.state == "PROGRESS":
        meta = result.info or {}
        return {"status": "PROGRESS", "progress": meta.get("progress", 0)}
    elif result.state == "SUCCESS":
        return {"status": "SUCCESS", "result": result.result}
    elif result.state == "FAILURE":
        return {
            "status": "FAILURE",
            "error": str(result.info),
        }
    else:
        # RETRY, REVOKED, etc.
        return {"status": result.state}
