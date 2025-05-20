from pydantic import BaseModel
from typing import List, Optional


class RFPInput(BaseModel):
    rfp_text: str
    backend: str = "open-deepresearch"
    model_name: str = "o3-mini"
    planner_model: Optional[str] = "gpt-4o-mini"
    writer_model: Optional[str] = "gpt-4o-mini"
    temperature: float = 1.0
    report_structure: Optional[str] = None


# If you want to keep your other schemas from your original code, place them here:
class QuerySchema(BaseModel):
    query: str
    backend: str = "open-deepresearch"
    planner_model: str = "gpt-4o-mini"
    writer_model: str = "gpt-4o-mini"
    report_structure: Optional[str] = (
        "Comprehensive analysis with key findings, details, and implications"
    )


class QueriesSchema(BaseModel):
    queries: List[str]
    backend: str = "open-deepresearch"
    planner_model: str = "gpt-4o-mini"
    writer_model: str = "gpt-4o-mini"
    report_structure: Optional[str] = (
        "Comprehensive analysis with key findings, details, and implications"
    )
