import os
import json
import asyncio
import logging
import re
import re
from typing import Dict, List, Optional, Union, Any

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from tqdm import tqdm

# For deep research integration
from dotenv import load_dotenv
load_dotenv()

# Import perplexity module
from perplexity import answer_query_with_perplexity

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define output schemas for structured parsing
class ResearchQuery(BaseModel):
    heading: str = Field(description="The heading or category of the query")
    subheading: Optional[str] = Field(None, description="The subheading of the query, if applicable")
    questions: List[str] = Field(description="List of specific research questions for this heading/subheading")

class ResearchPlan(BaseModel):
    title: str = Field(description="Title of the research plan")
    description: str = Field(description="Brief description of the research objectives")
    queries: List[ResearchQuery] = Field(description="Structured research queries")
    
def generate_rfp_queries(rfp_text: str, model_name: str = "o3-mini", temperature: float = 1) -> dict:
    """
    Generate structured research queries from an RFP document.
    
    Args:
        rfp_text (str): The text content of the RFP
        model_name (str): The LLM model to use for query generation
        temperature (float): Temperature setting for the LLM
        
    Returns:
        dict: Structured JSON with research queries
    """
    logger.info(f"Generating research queries using {model_name}")
    
    # Enhanced prompt with better structure and instructions
    query_generation_prompt = """
    You are an expert AI research assistant specializing in RFP analysis and strategic consulting.
    
    Below is an excerpt from an RFP document. Your task is to analyze this document thoroughly 
    and generate deep, insightful research queries that will help the proposal team understand:
    
    1. The explicit requirements
    2. The implicit needs behind those requirements
    3. The client's context, challenges, and strategic objectives
    4. The competitive landscape
    5. Technical insights relevant to this opportunity
    6. Include the agency name [RFP Client Name(full name) from RFP EXCERPT] in each question to facilitate more effective searches.
    
    RFP EXCERPT:
    {rfp_content}
    
    RESEARCH CATEGORIES:
    1. Agency Background & Strategic Alignment
       A. Agency Mission & Vision
       B. Current Initiatives & Priorities
 
    2. Business Drivers & Problem Statement
       A. Overarching Pain Points
       B. Critical Events Leading to the RFP
       C. Expected Outcomes & Success Metrics

    3. Technical Research
       A. Requirements Coverage
    
    
    INSTRUCTIONS:
    - For each category and subcategory, generate 1-2 specific, detailed research questions(but in simple language).
    - Make sure that the generated question can be used later while drafting an award winning proposal.
    - Prioritize questions that require deep research beyond the RFP text
    - Focus on questions that would provide strategic advantage if answered
    - Include questions about hidden requirements, unstated needs, and context
    - Make questions specific and actionable, not general
    
    OUTPUT FORMAT:
    Provide your response as a JSON structure with research categories, subcategories, and specific questions.
    Use the following structure:
    
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
        # Initialize the language model with the specified parameters
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # Create the prompt template
        prompt = PromptTemplate.from_template(query_generation_prompt)
        
        # Create the chain using the modern approach
        chain = prompt | llm | StrOutputParser()
        
        # Run the chain with the input
        raw_response = chain.invoke({"rfp_content": rfp_text})
        
        # Extract JSON from response (handle cases where the model might add explanatory text)
        json_str = extract_json_from_text(raw_response)
        
        # Parse the response into structured data
        try:
            result = json.loads(json_str)
            logger.info(f"Successfully generated {len(result.get('queries', []))} research query categories")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {raw_response}")
            # Fallback: Return the raw text
            return {"title": "Research Plan", "description": "Generated queries (unstructured)", "raw_text": raw_response}
            
    except Exception as e:
        logger.error(f"Error generating queries: {str(e)}")
        raise

def extract_json_from_text(text: str) -> str:
    """
    Extract JSON content from text that might contain additional markdown or explanations.
    """
    # Look for JSON between triple backticks
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if json_match:
        return json_match.group(1).strip()
    
    # Look for JSON between regular backticks
    json_match = re.search(r'```\s*([\s\S]*?)\s*```', text)
    if json_match:
        return json_match.group(1).strip()
    
    # If no backticks, check if the entire text is JSON
    try:
        json.loads(text)
        return text
    except:
        pass
    
    # If all else fails, return the original text
    return text

async def send_queries_to_deep_research(
    queries: List[str], 
    backend: str = "open-deepresearch",
    planner_model: str = "gpt-4o-mini",
    writer_model: str = "gpt-4o-mini",
    max_search_depth: int = 2,
    report_structure: str = "Comprehensive analysis with key findings, details, and implications"
) -> List[Dict[str, Any]]:
    """
    Send the generated queries to a deep research backend and return the results.
    
    Args:
        queries: List of query strings to research
        backend: Which backend to use ("open-deepresearch", "perplexity", or "standalone")
        planner_model: Model to use for planning (when using open-deepresearch)
        writer_model: Model to use for writing (when using open-deepresearch)
        max_search_depth: Maximum search depth for researching
        
    Returns:
        List of research results, one per query
    """
    logger.info(f"Sending {len(queries)} queries to {backend} backend")
    
    results = []
    
    if backend == "open-deepresearch":
        # Import here to avoid circular imports
        try:
            from open_deep_research.graph import builder
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.types import Command
            import uuid
            
            memory = MemorySaver()
            graph = builder.compile(checkpointer=memory)
            
            # Process each query
            for query in tqdm(queries, desc="Processing queries"):
                try:
                    # Create a unique thread for this query
                    thread_id = str(uuid.uuid4())
                    thread_config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "search_api": "tavily",
                            "planner_provider": "openai",
                            "planner_model": planner_model,
                            "writer_provider": "openai",
                            "writer_model": writer_model,
                            "max_search_depth": max_search_depth,
                            "report_structure": report_structure,
                            # "researcher_model" : "openai:o4-mini",
                            # "eval_model": "openai:o3-mini"
                        }
                    }
                    
                    # Start the deep research
                    async for event in graph.astream({"topic": query}, thread_config, stream_mode="updates"):
                        if '__interrupt__' in event:
                            interrupt_value = event['__interrupt__'][0].value
                            logger.debug(f"INTERRUPT (Query: {query}): {interrupt_value}")
                    
                    # Finalize the report
                    async for event in graph.astream(Command(resume=True), thread_config, stream_mode="updates"):
                        pass
                    
                    final_state = graph.get_state(thread_config)
                    final_report = final_state.values.get("final_report", "")
                    
                    results.append({
                        "query": query,
                        "report": final_report,
                        "status": "success",
                        "backend": "open-deepresearch"
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing query '{query}' with open-deepresearch: {str(e)}")
                    logger.info(f"Falling back to standalone research for query: {query}")
                    # Fall back to standalone research
                    # standalone_result = await standalone_research(query)
                    # results.append(standalone_result)
                    
        except ImportError as e:
            logger.error(f"Failed to import open-deepresearch modules: {str(e)}")
            logger.info("Falling back to standalone or perplexity research")
            if os.getenv("PERPLEXITY_API_KEY"):
                logger.info("PERPLEXITY_API_KEY found, using perplexity backend as fallback")
                return await send_queries_to_deep_research(queries, backend="perplexity")
            else:
                logger.info("Using standalone research as fallback")
                return await send_queries_to_deep_research(queries, backend="standalone")
    
    elif backend == "perplexity":
        # Use perplexity for in-depth web research
        logger.info(f"Sending {len(queries)} queries to perplexity backend")
        for query in queries:
            try:
                # Use the new perplexity module
                perplexity_result = await answer_query_with_perplexity(query)
                results.append({
                    "query": query,
                    "report": perplexity_result["result"],
                    "status": "success",
                    "backend": "perplexity",
                    "sources": perplexity_result.get("sources", [])
                })
                
            except Exception as e:
                logger.error(f"Error processing query '{query}' with Perplexity: {str(e)}")
                logger.info(f"Falling back to standalone research for query: {query}")
                standalone_result = await standalone_research(query)
                results.append(standalone_result)
    
    elif backend == "standalone":
        # Use a standalone approach with LangChain for research
        for query in queries:
            standalone_result = await standalone_research(query)
            results.append(standalone_result)
    
    else:
        logger.error(f"Unknown backend: {backend}")
        for query in queries:
            results.append({
                "query": query,
                "error": f"Unknown backend: {backend}",
                "status": "failed",
                "backend": backend
            })
    
    return results

async def standalone_research(
    query: str,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Perform standalone research using LangChain and OpenAI.
    This is a fallback when other backends are not available.
    
    Args:
        query: The research query
        model_name: The model to use for research
        temperature: Temperature setting for the model
        
    Returns:
        Dictionary with research results
    """
    logger.info(f"Performing standalone research for query: {query} using model {model_name}")
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        research_prompt = """
        You are a world-class research assistant with exceptional skills in finding accurate, 
        detailed, and nuanced information. I need you to conduct deep research on the following query:
        
        QUERY: {query}
        
        Please follow these research guidelines:
        1. Thoroughly analyze the query to understand all its dimensions and implications
        2. Consider historical context, current developments, and future implications
        3. Identify key stakeholders, their motivations, and perspectives
        4. Examine potential challenges, limitations, and trade-offs
        5. Present balanced viewpoints, including competing theories or approaches
        6. Provide concrete examples, case studies, or precedents when relevant
        7. Consider industry-specific nuances and domain knowledge
        8. Cite specific sources and data where possible (organizations, reports, studies)
        
        FORMAT YOUR RESPONSE AS FOLLOWS:
        1. KEY FINDINGS: 3-5 bullet points summarizing the most important insights
        2. DETAILED ANALYSIS: A comprehensive exploration of the topic organized by relevant subtopics
        3. IMPLICATIONS: What these findings mean for stakeholders and decision-makers
        4. RECOMMENDATIONS: Actionable next steps or considerations based on the research
        
        Your research should be comprehensive, nuanced, and actionable. Avoid overgeneralizations
        and aim for specific, concrete information that would genuinely help someone understand this topic deeply.
        """
        
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        prompt = PromptTemplate.from_template(research_prompt)
        chain = prompt | llm | StrOutputParser()
        
        research_report = await asyncio.to_thread(chain.invoke, {"query": query})
        
        return {
            "query": query,
            "report": research_report,
            "status": "success",
            "backend": "standalone"
        }
        
    except Exception as e:
        logger.error(f"Error in standalone research: {str(e)}")
        return {
            "query": query,
            "error": str(e),
            "status": "failed",
            "backend": "standalone"
        }

async def process_rfp_with_deep_research(
    rfp_text: str, 
    backend: str = "open-deepresearch",
    model_name: str = "o3-mini",
    temperature: float = 1.0,
    planner_model: str = "gpt-4o-mini",
    writer_model: str = "gpt-4o-mini",
    report_structure: str = "concised Report contaning key finding in bullet points"
) -> Dict[str, Any]:
    """
    Process an RFP document with deep research:
    1. Generate structured queries from the RFP
    2. Send these queries to the deep research backend
    3. Compile the results into a comprehensive report
    
    Args:
        rfp_text: The text content of the RFP
        backend: Which backend to use for deep research
        model_name: The model to use for query generation
        temperature: Temperature setting for query generation
        planner_model: Model to use for planning (when using open-deepresearch)
        writer_model: Model to use for writing (when using open-deepresearch)
        
    Returns:
        Dict with the full research results
    """
    # Step 1: Generate structured queries
    query_plan = generate_rfp_queries(rfp_text, model_name=planner_model, temperature=temperature)
    
    # Step 2: Extract individual queries for research
    all_queries = []
    for category in query_plan.get("queries", []):
        for question in category.get("questions", []):
            context = f"{category.get('heading')}"
            if category.get('subheading'):
                context += f" > {category.get('subheading')}"
            all_queries.append(f"[{context}] {question}")
    
    # Step 3: Send queries to deep research
    research_results = await send_queries_to_deep_research(
        all_queries, 
        backend=backend,
        planner_model=planner_model,
        writer_model=writer_model,
        report_structure = report_structure
    )
    
    # Step 4: Compile the results
    compiled_results = {
        "title": query_plan.get("title", "Research Report"),
        "description": query_plan.get("description", "Deep research analysis"),
        "categories": [],
        "raw_results": research_results,
        "meta": {
            "backend": backend,
            "query_count": len(all_queries),
            "success_count": sum(1 for r in research_results if r.get("status") == "success"),
            "failure_count": sum(1 for r in research_results if r.get("status") == "failed")
        }
    }
    
    # Organize results by category
    category_map = {}
    result_index = 0
    
    for category in query_plan.get("queries", []):
        category_key = f"{category.get('heading')}"
        if category.get('subheading'):
            category_key += f" > {category.get('subheading')}"
            
        if category_key not in category_map:
            category_map[category_key] = {
                "heading": category.get('heading'),
                "subheading": category.get('subheading'),
                "findings": []
            }
            
        for _ in category.get("questions", []):
            if result_index < len(research_results):
                result = research_results[result_index]
                if result.get("status") == "success":
                    category_map[category_key]["findings"].append({
                        "question": result.get("query", "").split("] ", 1)[-1],
                        "answer": result.get("report", "No results available")
                    })
                result_index += 1
    
    # Add organized categories to results
    compiled_results["categories"] = list(category_map.values())
    
    return compiled_results

# ----------------------------- MAIN WORKFLOW -----------------------------
if __name__ == "__main__":
    import re
    import argparse
    
    parser = argparse.ArgumentParser(description='RFP Deep Research Agent')
    parser.add_argument('--rfp', type=str, help='Path to RFP text file')
    parser.add_argument('--backend', type=str, default='open-deepresearch', 
                        choices=['open-deepresearch', 'perplexity', 'standalone'],
                        help='Deep research backend to use')
    parser.add_argument('--output', type=str, help='Path to save output JSON')
    parser.add_argument('--sample', action='store_true', help='Use sample RFP text')
    args = parser.parse_args()
    
    # Check if the input file exists or use default
    if os.path.exists('input/rfp.txt'):
        print("===== AGENT #1: Generating Queries from default RFP file =====")
        with open('input/rfp.txt', 'r') as f:
            rfp_text = f.read()
    
    # Generate queries and send to deep research
    async def main():
        try:
            results = await process_rfp_with_deep_research(rfp_text, backend=args.backend)
            
            # Output results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {args.output}")
            else:
                print("\n===== GENERATED RESEARCH PLAN =====")
                print(f"Title: {results['title']}")
                print(f"Description: {results['description']}")
                print(f"Categories: {len(results['categories'])}")
                print(f"Total queries: {results['meta']['query_count']}")
                print(f"Successful queries: {results['meta']['success_count']}")
                
                print("\n===== SAMPLE FINDINGS =====")
                if results['categories']:
                    first_category = results['categories'][0]
                    print(f"Category: {first_category['heading']}")
                    if first_category.get('subheading'):
                        print(f"Subheading: {first_category['subheading']}")
                        
                    if first_category['findings']:
                        first_finding = first_category['findings'][0]
                        print(f"\nQuestion: {first_finding['question']}")
                        print(f"Answer (preview): {first_finding['answer'][:500]}...")
                
                print("\nTo see full results, use the --output option to save to a file.")
        except Exception as e:
            logger.error(f"Error in main execution: {str(e)}")
            print(f"\nError: {str(e)}")
    
    # Run the async main function
    asyncio.run(main())