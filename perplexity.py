import json
import logging
import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key from environment variable
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not PERPLEXITY_API_KEY:
    logger.warning("PERPLEXITY_API_KEY not found in environment variables")


def perform_perplexity_research(
    query: str, model: str = "sonar-deep-research", max_tokens: int = 500
) -> Dict[str, Any]:
    """
    Perform deep research on a query using the Perplexity API.

    Args:
        query: The research query to investigate
        model: The Perplexity model to use (default: sonar-deep-research)
        max_tokens: Maximum number of tokens in the response

    Returns:
        Dictionary containing the research results
    """
    if not PERPLEXITY_API_KEY:
        raise ValueError("PERPLEXITY_API_KEY not set in environment variables")

    url = "https://api.perplexity.ai/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"Conduct detailed research on the following topic. Provide comprehensive information with sources. Topic: {query}",
            }
        ],
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        logger.info(f"Sending research query to Perplexity: {query}")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise exception for non-200 status codes

        result = response.json()
        logger.info("Successfully received response from Perplexity API")

        # Extract the content from the response
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]

            # Format the response as a structured dictionary
            return {
                "query": query,
                "result": content,
                "model": model,
                "sources": extract_sources_from_content(content),
            }
        else:
            logger.error(f"Unexpected response format from Perplexity API: {result}")
            return {
                "query": query,
                "result": "Error: Unable to extract content from Perplexity response",
                "error": str(result),
            }

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Perplexity API: {str(e)}")
        return {"query": query, "result": f"Error: {str(e)}", "error": str(e)}


def extract_sources_from_content(content: str) -> List[Dict[str, str]]:
    """
    Extract source citations from the Perplexity response content.

    Args:
        content: The text content from Perplexity response

    Returns:
        List of source dictionaries with title and url keys
    """
    sources = []

    # Simple extraction based on common patterns in Perplexity responses
    # This is a basic implementation and might need refinement based on actual response format
    lines = content.split("\n")

    for i, line in enumerate(lines):
        # Look for source patterns like "[1]" or "Source:" or "Reference:"
        if (
            (line.strip().startswith("[") and "]" in line)
            or "Source:" in line
            or "Reference:" in line
            or "http://" in line
            or "https://" in line
        ):

            # Extract the URL
            url_start = line.find("http")
            if url_start != -1:
                url_end = line.find(" ", url_start)
                if url_end == -1:
                    url_end = len(line)
                url = line[url_start:url_end].strip()

                # Try to extract a title
                title = line[:url_start].strip()
                title = (
                    title.replace("[", "")
                    .replace("]", "")
                    .replace("Source:", "")
                    .replace("Reference:", "")
                    .strip()
                )

                if not title and i > 0:
                    title = lines[i - 1].strip()

                sources.append(
                    {"title": title if title else "Unknown Source", "url": url}
                )

    return sources


async def answer_query_with_perplexity(query: str) -> Dict[str, Any]:
    """
    Asynchronous wrapper for the perform_perplexity_research function.
    This maintains compatibility with the aa.py interface that expects async functions.

    Args:
        query: The research query to investigate

    Returns:
        Dictionary containing the research results
    """
    return perform_perplexity_research(query)


# For testing/demonstration purposes
if __name__ == "__main__":
    test_query = "What are the latest developments in quantum computing and their potential impact on cryptography?"

    # Check if API key is available
    if PERPLEXITY_API_KEY:
        result = perform_perplexity_research(test_query)
        print(json.dumps(result, indent=2))
    else:
        print("Please set PERPLEXITY_API_KEY in your .env file to run this script")
