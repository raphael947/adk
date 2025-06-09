from dotenv import load_dotenv
import os
import asyncio
import concurrent.futures
import requests
from bs4 import BeautifulSoup
from typing import Optional
from google.adk import Agent
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

# Load environment variables
load_dotenv()

def extract_with_requests(url: str) -> dict:
    """
    Fallback content extraction using requests and BeautifulSoup.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find main content
        main_content = None
        content_selectors = [
            'article', 'main', '[role="main"]', '.content', '#content',
            '.post-content', '.entry-content', '.article-content',
            '.main-content', '#main_article'
        ]
        
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body')
        
        if main_content:
            # Get title
            title = soup.find('title')
            title = title.get_text().strip() if title else url.split('/')[-1]
            
            # Extract text
            text = main_content.get_text()
            
            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return {
                "title": title,
                "content": text,
                "method": "requests+beautifulsoup"
            }
        else:
            return {"error": "Could not find main content"}
            
    except Exception as e:
        return {"error": f"Requests extraction failed: {str(e)}"}

def extract_content_from_url(url: str, include_headers: bool = True, tool_context: Optional[ToolContext] = None) -> dict:
    """
    Extracts content from a URL using multiple methods.

    Args:
        url (str): The URL to extract content from.
        include_headers (bool): Whether to include headers in the output.
        tool_context (ToolContext): The context for the tool.

    Returns:
        dict: A dictionary containing the extracted content.
    """
    if not url:
        return {"error": "URL cannot be empty."}

    print(f"DEBUG: Attempting to extract content from URL: {url}")

    # First try crawl4ai if available
    try:
        from crawl4ai import AsyncWebCrawler
        
        async def _crawl():
            crawler = AsyncWebCrawler()
            try:
                # Try new API first
                result = await crawler.arun(url=url)
                return result
            except AttributeError:
                # Try older API
                try:
                    await crawler.astart()
                    result = await crawler.arun(url=url)
                    await crawler.aclose()
                    return result
                except Exception as e:
                    try:
                        await crawler.aclose()
                    except:
                        pass
                    raise e
            except Exception as e:
                raise e

        try:
            # Run the async crawler in a new event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If there's already a running event loop, we need to use a different approach
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, _crawl())
                        result = future.result(timeout=30)
                else:
                    result = loop.run_until_complete(_crawl())
            except RuntimeError:
                # No event loop running, create a new one
                result = asyncio.run(_crawl())
                
            if result and hasattr(result, 'markdown') and result.markdown:
                print("DEBUG: Successfully extracted content using Crawl4AI")
                
                # Get title with fallback
                title = getattr(result, "title", "No title found")
                if not title or title == "No title found":
                    title = url.split("/")[-1] or "Extracted Content"
                
                # Store the extracted content in the session state
                if tool_context:
                    try:
                        tool_context.state[f"extracted_content_{url}"] = result.markdown
                        print("DEBUG: Content stored in session state")
                    except Exception as e:
                        print(f"DEBUG: Warning - Could not store content in session state: {e}")
                    
                return {
                    "title": title,
                    "url": url,
                    "content_preview": result.markdown[:500] + "..." if len(result.markdown) > 500 else result.markdown,
                    "status": "Content extracted successfully using Crawl4AI",
                    "method": "crawl4ai"
                }
            else:
                print("DEBUG: Crawl4AI returned no content, trying fallback method")
                
        except Exception as e:
            print(f"DEBUG: Crawl4AI failed with error: {str(e)}")
            
    except ImportError:
        print("DEBUG: Crawl4AI not available, using fallback method")
    except Exception as e:
        print(f"DEBUG: Unexpected error with Crawl4AI: {str(e)}")

    # Fallback to requests + BeautifulSoup
    print("DEBUG: Using requests + BeautifulSoup fallback")
    result = extract_with_requests(url)
    
    if "error" not in result:
        # Store the extracted content in the session state
        if tool_context:
            try:
                tool_context.state[f"extracted_content_{url}"] = result["content"]
                print("DEBUG: Content stored in session state using fallback method")
            except Exception as e:
                print(f"DEBUG: Warning - Could not store content in session state: {e}")
                
        return {
            "title": result["title"],
            "url": url,
            "content_preview": result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"],
            "status": "Content extracted successfully using fallback method",
            "method": result["method"]
        }
    else:
        print(f"DEBUG: All extraction methods failed: {result['error']}")
        return {"error": f"All extraction methods failed. Last error: {result['error']}"}

def create_content_extractor_agent():
    """
    Creates an agent specialized in extracting content from web pages.
    """
    content_extractor_tool = FunctionTool(
        func=extract_content_from_url,
    )

    content_extractor_agent = Agent(
        name="content_extractor",
        model="gemini-1.5-flash",
        description="An agent that extracts content from a URL using multiple extraction methods.",
        instruction="""You are a specialist in extracting content from web pages.
        
        Given a URL, use the `extract_content_from_url` tool to fetch the content.
        
        The tool will try multiple extraction methods:
        1. Crawl4AI (advanced web crawling)
        2. Requests + BeautifulSoup (fallback method)
        
        Always confirm that the content has been successfully extracted and stored in the session state for other agents to use.
        
        If extraction fails, provide detailed error information to help diagnose the issue.
        """,
        tools=[content_extractor_tool],
    )

    return content_extractor_agent 