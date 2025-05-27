import re
import json


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON content from text that might contain additional markdown or explanations.
    """
    # Look for JSON between triple backticks
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if json_match:
        return json_match.group(1).strip()

    # Look for JSON between regular backticks
    json_match = re.search(r"```\s*([\s\S]*?)\s*```", text)
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
