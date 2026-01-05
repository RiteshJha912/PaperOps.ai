import os
import sys
from dotenv import load_dotenv
from agent import create_research_agent

# Load environment variables
load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Get the API Key securely from the .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ==============================================================================

def main():
    print("--- Simple Agentic AI Researcher ---")
    
    # Check for API Key
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found.")
        print("Please check your .env file.")
        sys.exit(1)

    # Get Topic from User
    topic = input("\nEnter a topic to research (e.g., 'Transformers in NLP'): ").strip()
    if not topic:
        print("No topic entered. Exiting.")
        return

    display_key = GROQ_API_KEY[:5] + "..." + GROQ_API_KEY[-4:]
    print(f"\nInitialized Agent for topic: {topic}")
    # print(f"Using API Key: {display_key}") # Debug only
    print("Agent is thinking... (Watch the output below)\n")

    # Create the Agent
    try:
        agent_executor = create_research_agent(GROQ_API_KEY)
    except Exception as e:
        print(f"Error creating agent: {e}")
        return

    prompt_template = """
    You are an academic researcher. Your goal is to write a short report on: "{topic}".
    
    PLAN:
    1. SEARCH: Use 'web_search' to find 3 reliable pages.
    2. READ: Use 'read_page' on the best URLs found.
    3. WRITE: "Final Answer" MUST be the report in Markdown.
    
    REPORT STRUCTURE:
    # {topic}
    ## Key Concepts
    ## Details
    ## References
    
    IMPORTANT: 
    - Do not get stuck searching. If one search fails, try another or just read what you have.
    - If you have read 2-3 pages, STOP searching and write the report immediately.
    - Your Final Answer must be the report in Markdown.
    """
    
    research_instruction = prompt_template.format(topic=topic)

    # Run the Agent
    try:
        result = agent_executor.invoke({"input": research_instruction})
        report_content = result["output"]
        
        print(f"\n[DEBUG] Raw Output length: {len(report_content)}")
        # print(f"[DEBUG] Raw Output content: {report_content[:200]}...")

        if not report_content or "Agent stopped" in report_content:
             print("\n[WARN] The agent might have stopped early or failed to generate the full report.")
             print("Raw Output:", report_content)
        else:
            # Save Report
            filename = "report.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_content)
                
            print(f"\n\nSuccess! Report saved to '{filename}'")
            print("-" * 50)
        # print("Report Preview:\n")
        # print(report_content[:500] + "...\n")
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main()
