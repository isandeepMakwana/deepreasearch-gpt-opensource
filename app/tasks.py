# app/tasks.py

import json
import uuid
import logging
import asyncio

from .celery_app import celery_app
from .logger_config import setup_logger
from .utils import extract_json_from_text

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from .open_deep_research import answer_query_with_deep_research

logger = setup_logger(__name__)

def generate_rfp_queries(rfp_text: str, model_name: str = "o3-mini", temperature: float = 1) -> dict:
    logger.info(f"Generating research queries using {model_name}")

    query_generation_prompt = """
    You are an expert AI research assistant specializing in RFP analysis and strategic consulting.
    Below is an excerpt from an RFP document. Your task is to analyze this document thoroughly
    and generate deep, insightful research queries...

    RFP EXCERPT:
    {rfp_content}

    OUTPUT FORMAT:
    {{
        "queries": [
            {{
            "heading": "Category Name",
            "subheading": "Subcategory Name",
            "questions": [
                "Specific question 1?",
                "Specific question 2?"
            ]
            }}
        ]
    }}
    """
    
    try:
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        prompt = PromptTemplate.from_template(query_generation_prompt)
        chain = prompt | llm | StrOutputParser()
        raw_response = chain.invoke({"rfp_content": rfp_text})

        json_str = extract_json_from_text(raw_response)
        result = json.loads(json_str)
        return result
    except Exception as e:
        logger.error(f"Error generating queries: {str(e)}")
        return {
            "error": str(e),
            "raw_text": "",
        }


@celery_app.task(bind=True)
def process_rfp_with_deep_research_task(
    self,
    rfp_text: str,
    backend: str,
    planner_model_provider: str,
    planner_model: str,
    writer_model_provider: str,
    writer_model: str,
    temperature: float,
):

    logger.info("Started background RFP deep research...")

    # Step 1: Generate queries
    query_plan = generate_rfp_queries(
        rfp_text=rfp_text
    )
    if "queries" not in query_plan:
        return {
            "status": "failed",
            "message": "No queries generated.",
            "details": query_plan
        }

    # Build a list of queries (strings)
    all_queries = []
    for category in query_plan.get("queries", []):
        heading = category.get("heading", "Unknown")
        subheading = category.get("subheading", "")
        for question in category.get("questions", []):
            context = f"[{heading}]"
            if subheading:
                context += f" > {subheading}"
            all_queries.append(f"{context} {question}")

    total_count = len(all_queries)
    logger.info(f"Using asyncio.gather() for {total_count} queries in parallel...")

    # Optional: Update state at 0% just to show "progress" is started
    self.update_state(
        state="PROGRESS",
        meta={"progress": 25}
    )  
    

    # Step 2: Define an async function that calls 'answer_query_with_deep_research' for each query in parallel
    async def run_all_in_parallel(queries):
        coros = [answer_query_with_deep_research(
            q,
            planner_model_provider, planner_model,
            writer_model_provider, writer_model
        ) for q in queries]
        results = await asyncio.gather(*coros, return_exceptions=True)
        return results
    # Step 3: Actually run them in concurrency
    # Because this is a Celery task, we'll do an asyncio run in a synchronous context
    results = asyncio.run(run_all_in_parallel(all_queries))

    # Step 4: Build a "research_results" list that has status + query + report
    research_results = []
    for i, q in enumerate(all_queries):
        res = results[i]
        if isinstance(res, Exception):
            logger.error(f"Query {q} failed: {res}")
            research_results.append({
                "query": q,
                "report": f"Error: {str(res)}",
                "status": "failed",
                "backend": backend,
            })
        else:
            research_results.append({
                "query": q,
                "report": res,
                "status": "success",
                "backend": backend,
            })

    # Step 5: Compile final structure
    compiled = compile_final_results(query_plan, research_results, backend)

    # Optional: Mark final progress at 100%
    self.update_state(
        state="PROGRESS",
        meta={"progress": 100.0}
    )

    # Return the final result
    print(compiled)
    return compiled


def compile_final_results(query_plan: dict, research_results: list, backend: str) -> dict:
    compiled = {
        "status": "success",
        "title": query_plan.get("title", "Research Report"),
        "description": query_plan.get("description", "Deep research analysis"),
        "categories": [],
        "raw_results": research_results,
        "meta": {
            "backend": backend,
            "query_count": len(research_results),
            "success_count": sum(1 for r in research_results if r["status"] == "success"),
            "failure_count": sum(1 for r in research_results if r["status"] == "failed"),
        },
    }

    # First, build a category map with findings
    category_map = {}
    result_index = 0

    for category in query_plan.get("queries", []):
        heading_str = category.get("heading", "")
        subheading_str = category.get("subheading", "")

        # If you want to combine heading & subheading into one string:
        if subheading_str:
            heading_str += f" - {subheading_str}"

        if heading_str not in category_map:
            category_map[heading_str] = []

        # For each question, grab its corresponding item from research_results
        for _ in category.get("questions", []):
            if result_index < len(research_results):
                result = research_results[result_index]
                question_text = "Unknown question"
                answer_text = "No results available"

                # Parse question
                splitted = result["query"].split("] ", 1)
                if len(splitted) == 2:
                    question_text = splitted[-1]
                else:
                    question_text = result["query"]

                if result["status"] == "success":
                    answer_text = result.get("report", "No results available")
                else:
                    answer_text = "Error occurred"

                # Append this Q/A to the list for this heading
                category_map[heading_str].append({
                    "question": question_text,
                    "answer": answer_text
                })

                result_index += 1

    # Now transform each heading’s Q/A list into the new format
    final_categories = []
    for heading_str, findings in category_map.items():
        # Combine question + answer pairs into a single content string
        content_parts = []
        for item in findings:
            q = item["question"]
            a = item["answer"]
            # e.g. "Q?\nA"
            content_parts.append(f"{q}\n{a}")

        # Join them with blank lines or some delimiter
        content_str = "\n\n".join(content_parts)

        final_categories.append({
            "heading": heading_str,
            "content": content_str
        })

    # Attach final_categories to compiled
    compiled["categories"] = final_categories

    return compiled
