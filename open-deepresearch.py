import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from open_deep_research.graph import builder

load_dotenv()

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

REPORT_STRUCTURE = """give me the sections and subsections of the report, and the key points for each questions."""

thread_template = {
    "configurable": {
        "thread_id": None,  # We'll set this dynamically
        "search_api": "tavily",
        "planner_provider": "openai",
        "planner_model": "gpt-4o-mini",
        "writer_provider": "openai",
        "writer_model": "gpt-4o-mini",
        "max_search_depth": 1,
        "report_structure": REPORT_STRUCTURE,
    }
}


async def answer_query_with_deep_research(query: str) -> str:
    """
    Runs the 'graph' with a single query as the topic.
    Returns the final report (the detailed answer) as a string.
    """

    # Each query gets a unique thread_id so the states won't overlap
    local_thread = dict(thread_template)
    local_thread["configurable"] = dict(thread_template["configurable"])
    local_thread["configurable"]["thread_id"] = str(uuid.uuid4())

    # 1. Start the deep research
    async for event in graph.astream(
        {"topic": query}, local_thread, stream_mode="updates"
    ):
        if "__interrupt__" in event:
            interrupt_value = event["__interrupt__"][0].value
            print(f"INTERRUPT (Query: {query}): {interrupt_value}")

    # 2. (Optional) You could refine or add more instructions with Command(resume="...")

    # 3. Finalize the report
    async for event in graph.astream(
        Command(resume=True), local_thread, stream_mode="updates"
    ):
        pass  # We won't print stream updates here, but you could if desired

    final_state = graph.get_state(local_thread)
    final_report = final_state.values.get("final_report", "")
    return final_report
