from dotenv import load_dotenv
from typing import Optional
from google.adk import Agent
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

# Load environment variables
load_dotenv()

def summarize_content(
    content: Optional[str] = None,
    url: Optional[str] = None,
    length: str = "medium",
    tool_context: Optional[ToolContext] = None,
) -> str:
    """
    Summarizes content provided directly or extracted from a URL.

    Args:
        content (str, optional): The content to summarize.
        url (str, optional): The URL from which to get content for summarization.
        length (str): The desired length of the summary (short, medium, or long).
        tool_context (ToolContext): The context for the tool.

    Returns:
        str: The summarized content.
    """
    if url and tool_context:
        try:
            content_key = f"extracted_content_{url}"
            if content_key in tool_context.state:
                content = tool_context.state[content_key]
            else:
                return f"Error: No content found for URL {url}. Please extract it first."
        except Exception as e:
            return f"Error accessing session state: {str(e)}"

    if not content:
        return "Error: No content provided for summarization."

    # This is a simplified summarization logic.
    # In a real application, you might call another LLM for a more abstractive summary.
    word_count = len(content.split())
    if length == "short":
        return " ".join(content.split()[:100]) + "..."
    elif length == "medium":
        return " ".join(content.split()[:300]) + "..."
    else:
        return " ".join(content.split()[:600]) + "..."

def create_summarizer_agent():
    """
    Creates an agent specialized in summarizing text.
    """
    summarizer_tool = FunctionTool(
        func=summarize_content,
    )

    summarizer_agent = Agent(
        name="summarizer",
        model="gemini-1.5-flash",
        description="An agent that summarizes text.",
        instruction="""You are a text summarization specialist.

        You can summarize content provided directly or from a URL that has been previously analyzed.
        
        When a URL is provided, retrieve the content from the session state.
        
        Generate a summary of the requested length and present it clearly.
        """,
        tools=[summarizer_tool],
    )

    return summarizer_agent 