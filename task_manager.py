import time
import uuid
from typing import Dict, Any, Optional, List

# A simple in-memory store for task results (replace with something persistent in production)
tasks_results: Dict[str, Any] = {}
tasks_status: Dict[str, str] = {}
tasks_stages: Dict[str, Dict[str, Any]] = {}
tasks_current_stage: Dict[str, int] = {}


def generate_task_id() -> str:
    """Generate a unique task ID"""
    return str(uuid.uuid4())


def register_task(task_id: Optional[str] = None, total_stages: int = 1) -> str:
    """Register a new task and return its ID"""
    if task_id is None:
        task_id = generate_task_id()

    tasks_status[task_id] = "PENDING"
    tasks_results[task_id] = None
    tasks_stages[task_id] = {
        "total": total_stages,
        "completed": 0,
        "stage_results": [None] * total_stages,
    }
    tasks_current_stage[task_id] = 1
    return task_id


def update_task_status(task_id: str, status: str) -> None:
    """Update the status of a task"""
    if task_id in tasks_status:
        tasks_status[task_id] = status


def update_task_stage(task_id: str, stage_index: int, stage_result: Any = None) -> None:
    """Update the current stage of a task and optionally store stage result"""
    if task_id in tasks_stages:
        tasks_current_stage[task_id] = stage_index
        tasks_stages[task_id]["completed"] = stage_index

        if (
            stage_result is not None
            and 0 <= stage_index < tasks_stages[task_id]["total"]
        ):
            tasks_stages[task_id]["stage_results"][stage_index] = stage_result

        # If this is the last stage, mark the task as completed
        if stage_index >= tasks_stages[task_id]["total"] - 1:
            update_task_status(task_id, "COMPLETED")


def store_task_result(task_id: str, result: Any) -> None:
    """Store the result of a completed task"""
    if task_id in tasks_results:
        tasks_results[task_id] = result
        tasks_status[task_id] = "COMPLETED"

        # Update completed stages to match total stages
        if task_id in tasks_stages:
            total_stages = tasks_stages[task_id]["total"]
            tasks_stages[task_id]["completed"] = total_stages
            tasks_current_stage[task_id] = total_stages


def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get the current status and result (if available) of a task"""
    if task_id not in tasks_status:
        return {"task_id": task_id, "status": "NOT_FOUND"}

    status = tasks_status[task_id]
    result = None

    if status == "COMPLETED":
        result = tasks_results[task_id]

    # Get stage information
    stages_info = {}
    if task_id in tasks_stages:
        stages_info = {
            "total_stages": tasks_stages[task_id]["total"],
            "completed_stages": tasks_stages[task_id]["completed"],
            "current_stage": tasks_current_stage.get(task_id, 0),
            "stage_results": tasks_stages[task_id]["stage_results"],
        }

    return {
        "task_id": task_id,
        "status": status,
        "stages": stages_info,
        "result": result if status == "COMPLETED" else None,
    }


async def process_in_background(task_id: str, task_func, *args, **kwargs) -> None:
    """Process a task in the background and store its result"""
    try:
        # Update status to processing
        update_task_status(task_id, "PROCESSING")

        # Check if the task function is a generator or async generator
        # This allows for stage-by-stage processing with progress updates
        if hasattr(task_func, "__aiter__") or hasattr(task_func, "__iter__"):
            stage_index = 0
            async_result = None

            # Handle async generators
            if hasattr(task_func, "__aiter__"):
                async for stage_result in task_func(*args, **kwargs):
                    # Update the current stage
                    update_task_stage(task_id, stage_index, stage_result)
                    async_result = stage_result
                    stage_index += 1
            # Handle regular generators
            elif hasattr(task_func, "__iter__"):
                for stage_result in task_func(*args, **kwargs):
                    # Update the current stage
                    update_task_stage(task_id, stage_index, stage_result)
                    async_result = stage_result
                    stage_index += 1

            # Update completed stages if not all stages were processed
            if task_id in tasks_stages:
                total_stages = tasks_stages[task_id]["total"]
                if stage_index < total_stages:
                    # Mark the last processed stage
                    tasks_current_stage[task_id] = (
                        stage_index - 1 if stage_index > 0 else 0
                    )
                    tasks_stages[task_id]["completed"] = (
                        stage_index - 1 if stage_index > 0 else 0
                    )

            # Store the final result
            store_task_result(task_id, async_result)
        else:
            # Execute the task as a regular function
            result = await task_func(*args, **kwargs)

            # For non-generator tasks, we need to handle stage progression differently
            if task_id in tasks_stages:
                total_stages = tasks_stages[task_id]["total"]

                # If there's only one stage, just mark it as completed
                if total_stages == 1:
                    update_task_stage(task_id, 0, result)
                else:
                    # For multi-stage tasks, mark all stages as completed
                    for i in range(total_stages):
                        # Only update result for the last stage
                        stage_result = (
                            result if i == total_stages - 1 else {"status": "Completed"}
                        )
                        update_task_stage(task_id, i, stage_result)

            # Store the result
            store_task_result(task_id, result)
    except Exception as e:
        # Update status to failed
        update_task_status(task_id, "FAILED")
        tasks_results[task_id] = {"error": str(e)}

        # Log the stage where the failure occurred
        if task_id in tasks_current_stage:
            current_stage = tasks_current_stage[task_id]
            if task_id in tasks_stages and "stage_results" in tasks_stages[task_id]:
                tasks_stages[task_id]["stage_results"][current_stage] = {
                    "error": str(e)
                }
