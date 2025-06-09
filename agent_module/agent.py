from google.adk import Agent
from google.adk.tools.agent_tool import AgentTool
from .search_agent import create_search_agent
from .content_extractor import create_content_extractor_agent
from .summarizer import create_summarizer_agent

# Create instances of the specialist agents
search_agent_instance = create_search_agent()
content_extractor_instance = create_content_extractor_agent()
summarizer_agent_instance = create_summarizer_agent()

# Wrap specialist agents as tools for the coordinator
search_tool = AgentTool(
    agent=search_agent_instance
)
extractor_tool = AgentTool(
    agent=content_extractor_instance
)
summarizer_tool = AgentTool(
    agent=summarizer_agent_instance
)

# Define the root agent (the coordinator)
root_agent = Agent(
    name="research_coordinator",
    model="gemini-1.5-flash",
    description="The coordinator for a team of research specialists.",
    instruction="""You are the coordinator of a research team.
    Your team has three specialists:
    - A search agent to find information online.
    - A content extractor to get content from URLs.
    - A summarizer to condense information.

    Your job is to delegate tasks to the correct specialist based on the user's request.
    For complex requests, you may need to chain the specialists together. For example:
    1. Search for information.
    2. Extract content from the most relevant URL.
    3. Summarize the content.
    
    Always present the final results to the user in a clear and organized way.
    """,
    tools=[search_tool, extractor_tool, summarizer_tool]
) 