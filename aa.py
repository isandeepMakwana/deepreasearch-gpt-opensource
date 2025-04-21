
import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Union, Any

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# For deep research integration
from dotenv import load_dotenv
load_dotenv()

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
    
def generate_rfp_queries(rfp_text: str, model_name: str = "gpt-4o-mini", temperature: float = 0.7) -> dict:
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
    5. Financial and technical insights relevant to this opportunity
    
    RFP EXCERPT:
    {rfp_content}
    
    RESEARCH CATEGORIES:
    1. Agency Background & Strategic Alignment
       A. Agency Mission & Vision
       B. Current Initiatives & Priorities
       C. Organizational Structure & Decision-Making Process
    
    2. Incumbent Analysis
       A. Incumbent Vendor (if any)
       B. Incumbent Performance
       C. Differentiation Strategy
       D. Contract History & Transition Points
    
    3. Business Drivers & Problem Statement
       A. Overarching Pain Points
       B. Critical Events Leading to the RFP
       C. Stated Goals vs. Implied Needs
       D. Expected Outcomes & Success Metrics
    
    4. Financial Research
       A. Budget Allocated for the Project
       B. Agency's Spending History
       C. Propensity to Spend
       D. Comparable Budget Research
       E. Total Cost of Ownership Considerations
    
    5. Technical Research
       A. Requirements Coverage
       B. Current Tools/Technologies in Use
       C. Integration Points & Ecosystem
       D. Technical Constraints & Legacy Systems
    
    6. Competitive Intelligence
       A. Who Else Might Bid
       B. Previous Bids or Awards
       C. Competitor Strengths & Weaknesses
       D. Your Unique Value Proposition
    
    7. Procurement Process & Decision Criteria
       A. Evaluation Methodology
       B. Key Decision Makers
       C. Timing Considerations
       D. Compliance Requirements
    
    INSTRUCTIONS:
    - For each category and subcategory, generate 3-5 specific, detailed research questions
    - Prioritize questions that require deep research beyond the RFP text
    - Focus on questions that would provide strategic advantage if answered
    - Include questions about hidden requirements, unstated needs, and context
    - Make questions specific and actionable, not general
    
    OUTPUT FORMAT:
    Provide your response as a JSON structure with research categories, subcategories, and specific questions.
    Use the following structure:
    
    ```json
    {
      "title": "Research Plan for [RFP Title/Client Name]",
      "description": "Strategic research questions for responding to [Client]'s RFP for [Project/Service]",
      "queries": [
        {
          "heading": "Category Name",
          "subheading": "Subcategory Name",
          "questions": [
            "Specific question 1?",
            "Specific question 2?",
            "Specific question 3?"
          ]
        },
        // Additional categories...
      ]
    }
    """

    try:
        # Initialize the language model with the specified parameters
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # Create a prompt template
        prompt_template = PromptTemplate(
            input_variables=["rfp_content"],
            template=query_generation_prompt
        )
        
        # Create and run the chain
        chain = LLMChain(llm=llm, prompt=prompt_template)
        raw_response = chain.run(rfp_content=rfp_text)
        
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

async def send_queries_to_deep_research(queries: List[str], backend: str = "open-deepresearch") -> List[Dict[str, Any]]:
    """
    Send the generated queries to a deep research backend and return the results.
    
    Args:
        queries: List of query strings to research
        backend: Which backend to use ("open-deepresearch" or "perplexity")
        
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
            for query in queries:
                try:
                    # Create a unique thread for this query
                    thread_id = str(uuid.uuid4())
                    thread_config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "search_api": "tavily",
                            "planner_provider": "openai",
                            "planner_model": "gpt-4o-mini",
                            "writer_provider": "openai",
                            "writer_model": "gpt-4o-mini",
                            "max_search_depth": 2,
                            "report_structure": "Comprehensive analysis with key findings, details, and implications"
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
                    results.append({
                        "query": query,
                        "error": str(e),
                        "status": "failed",
                        "backend": "open-deepresearch"
                    })
                    
        except ImportError as e:
            logger.error(f"Failed to import open-deepresearch modules: {str(e)}")
            for query in queries:
                results.append({
                    "query": query,
                    "error": "open-deepresearch backend not available",
                    "status": "failed",
                    "backend": "open-deepresearch"
                })
    
    elif backend == "perplexity":
        # Use Perplexity API for deep research
        import requests
        
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        if not perplexity_api_key:
            logger.error("PERPLEXITY_API_KEY not found in environment variables")
            for query in queries:
                results.append({
                    "query": query,
                    "error": "PERPLEXITY_API_KEY not configured",
                    "status": "failed",
                    "backend": "perplexity"
                })
            return results
        
        for query in queries:
            try:
                url = "https://api.perplexity.ai/chat/completions"
                payload = {
                    "model": "sonar-deep-research",
                    "messages": [
                        {"role": "user", "content": query}
                    ],
                    "max_tokens": 4000
                }
                headers = {
                    "Authorization": f"Bearer {perplexity_api_key}",
                    "Content-Type": "application/json"
                }
                
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                response_data = response.json()
                
                results.append({
                    "query": query,
                    "report": response_data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                    "status": "success",
                    "backend": "perplexity"
                })
                
            except Exception as e:
                logger.error(f"Error processing query '{query}' with Perplexity: {str(e)}")
                results.append({
                    "query": query, 
                    "error": str(e),
                    "status": "failed", 
                    "backend": "perplexity"
                })
    
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

async def process_rfp_with_deep_research(rfp_text: str, backend: str = "open-deepresearch") -> Dict[str, Any]:
    """
    Process an RFP document with deep research:
    1. Generate structured queries from the RFP
    2. Send these queries to the deep research backend
    3. Compile the results into a comprehensive report
    
    Args:
        rfp_text: The text content of the RFP
        backend: Which backend to use for deep research
        
    Returns:
        Dict with the full research results
    """
    # Step 1: Generate structured queries
    query_plan = generate_rfp_queries(rfp_text)
    
    # Step 2: Extract individual queries for research
    all_queries = []
    for category in query_plan.get("queries", []):
        for question in category.get("questions", []):
            context = f"{category.get('heading')}"
            if category.get('subheading'):
                context += f" > {category.get('subheading')}"
            all_queries.append(f"[{context}] {question}")
    
    # Step 3: Send queries to deep research
    research_results = await send_queries_to_deep_research(all_queries, backend=backend)
    
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
                        choices=['open-deepresearch', 'perplexity'],
                        help='Deep research backend to use')
    parser.add_argument('--output', type=str, help='Path to save output JSON')
    args = parser.parse_args()
    
    # If no arguments provided, use the sample RFP text
    if not args.rfp:
        print("===== AGENT #1: Generating Queries from Sample RFP =====")
        rfp_text = """ The City of Charleston
Procurement Division
75 Calhoun Street, Suite 3500
Charleston, South Carolina 29401
P) 843-724-7312 F) 843-720-3872
www.charleston-sc.gov
Proposal Number: 24-P028R Proposals will be received until: November 5, 2024 @ 1:00pm
Proposal Title: Comprehensive Data Analytics System
Mailing Date: October 3, 2024 Direct Inquiries to: Robin B. Robinson
Vendor Name: FEIN/SS#:
Vendor Address:
City – State – Zip:
Telephone Number: Fax Number:
Minority or Women Owned Business:
Are you a certified Minority or Women-Owned business in the State of South Carolina? If so, please provide a copy of your certificate with your response.
 Yes  No
Authorized Signature: _____________________________ Title: __________________________
Date: _________________________
I certify that this bid is made without prior understanding, agreement, or connection with any corporation, firm, or person submitting a bid
for the same materials, supplies, equipment or services and is in all respects fair and without collusion or fraud. I agree to abide by all
conditions of this bid and certify that I am authorized to sign this bid for the bidder. This signed page must be included with bid
submission.
1. 2. IMPORTANT
This solicitation seeks proposals responding to the Scope of Work for Comprehensive Data
Analytics System. This solicitation does not commit the City of Charleston to award a
contract, to pay any costs incurred in the preparation of applications submitted, or to procure
or contract for the services. The City reserves the right to accept or reject any, all or any part
of any proposal received as a result of this Solicitation, or to cancel in part or in its entirety
this Solicitation if it is in the best interest of the City to do so. The City shall be the sole
judge as to whether proposals submitted meet all requirements contained in this solicitation.
The City of Charleston, South Carolina has received funds from the Bureau of Justice
Assistance, and are bidding these items utilizing the 2023 Smart Policing Initiative."""
    else:
        print(f"===== AGENT #1: Generating Queries from RFP file: {args.rfp} =====")
        with open(args.rfp, 'r') as f:
            rfp_text = f.read()
    
    # Generate queries and send to deep research
    async def main():
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
    
    # Run the async main function
    asyncio.run(main())
