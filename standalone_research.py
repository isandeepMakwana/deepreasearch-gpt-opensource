import asyncio
from typing import Any, Dict


async def standalone_research(
    query: str, model_name: str = "gpt-4o-mini", temperature: float = 0.7
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
    logger.info(
        f"Performing standalone research for query: {query} using model {model_name}"
    )

    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI

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
            "backend": "standalone",
        }

    except Exception as e:
        logger.error(f"Error in standalone research: {str(e)}")
        return {
            "query": query,
            "error": str(e),
            "status": "failed",
            "backend": "standalone",
        }
