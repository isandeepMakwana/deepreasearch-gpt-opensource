from pydantic import BaseModel, Field
from typing import List, Optional


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
    subheading: Optional[str] = Field(
        None, description="The subheading of the query, if applicable"
    )
    questions: List[str] = Field(
        description="List of specific research questions for this heading/subheading"
    )


class ResearchPlanModel(BaseModel):
    title: str = Field(description="Title of the research plan")
    description: str = Field(description="Brief description of the research objectives")
    queries: List[ResearchQueryModel] = Field(description="Structured research queries")
