from dotenv import load_dotenv
import os
from google.adk import Agent
from google.adk.tools.langchain_tool import LangchainTool
from langchain_community.tools import TavilySearchResults

# Load environment variables
load_dotenv()

def create_search_agent():
    """
    Creates an agent specialized in web searching using Tavily Search.
    """
    # Check if API key is available
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables")

    # Create Tavily Search tool with LangChain
    tavily_tool_instance = TavilySearchResults(
        max_results=5,  # Return 5 results per search
        search_depth="advanced",  # Use advanced search for more comprehensive results
        include_answer=True,  # Include a direct answer when possible
        include_raw_content=True,  # Include the raw content from search results
        include_images=False  # Don't include images in the results
    )

    # Wrap the LangChain tool for ADK
    adk_tavily_tool = LangchainTool(tool=tavily_tool_instance)

    # Create and return the search agent
    search_agent = Agent(
        name="search_agent",
        model="gemini-1.5-flash",
        description="A specialized agent that searches the web for information using Tavily Search API.",
        instruction="""You are a web research specialist.

        When asked to find information about a topic, craft an effective search query and use the TavilySearchResults tool.

        After receiving search results:
        1. Parse the response which may contain a direct answer and multiple search results.
        2. Format the results in a clear, structured way, with each result showing the title, link, and a brief preview of the content.
        3. Highlight the most relevant results based on the original query.
        4. If Tavily provided a direct answer, present that first as the most likely answer.

        If the search doesn't return useful results, suggest refined search terms for a follow-up search.

        Avoid making up information - only report what is found in the search results.
        """,
        tools=[adk_tavily_tool]
    )

    return search_agent 