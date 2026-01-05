import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
import time

# We need the LLM to summarize the page, so we will create a setup function that accepts the LLM
def get_tools(llm):
    """
    Returns a list of tools for the agent to use.
    Requires an LLM instance for the page reading tool.
    """
    
    # 1. The Search Tool (Free via DuckDuckGo)
    search_tool = DuckDuckGoSearchRun()
    
    @tool
    def web_search(query: str):
        """
        Use this tool to search the internet for information. 
        Input should be a search query (string).
        """
        print(f"\n[Tool] Searching for: {query}")
        time.sleep(2) # Wait 2 seconds to be polite and avoid rate limits
        try:
            return search_tool.run(query)
        except Exception as e:
            # If rate limited, tell the agent to wait
            if "Ratelimit" in str(e) or "429" in str(e):
                return "Search Rate Limit hit. Please wait 5 seconds and try a different query."
            return f"Error searching: {str(e)}"

    # 2. The Page Reader Tool (Custom built)
    @tool
    def read_page(url: str):
        """
        Use this tool to read a specific webpage URL. 
        It fetches the page content and returns a summary of the key information.
        Input should be a valid URL string (e.g., https://example.com/article).
        """
        print(f"\n[Tool] Reading page: {url}")
        
        try:
            # Fetch the page content
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text (simple extraction of paragraphs)
            paragraphs = soup.find_all('p')
            text_content = "\n".join([p.get_text() for p in paragraphs])
            
            # Limit text length to avoid token limits (approx 8000 chars is usually safe for Llama 3)
            # We cut it off to ensure we don't crash, but keep enough for a summary
            max_chars = 10000 
            if len(text_content) > max_chars:
                text_content = text_content[:max_chars] + "...(truncated)"
                
            if not text_content.strip():
                return "The page seems empty or couldn't be parsed properly."

            # Summarize using the LLM
            # We ask the LLM specifically to summarize it for the research report
            summary_prompt = f"""
            Please read the following text from a webpage and summarize the key points relevant to an academic report.
            Focus on facts, dates, key ideas, and definitions.
            
            TEXT FROM PAGE:
            {text_content}
            
            SUMMARY:
            """
            
            # Call the LLM directly
            messages = [HumanMessage(content=summary_prompt)]
            result = llm.invoke(messages)
            
            return f"Summary of {url}:\n{result.content}"
            
        except requests.exceptions.RequestException as e:
            return f"Error fetching the page: {str(e)}"
        except Exception as e:
            return f"Error reading page content: {str(e)}"

    return [web_search, read_page]
