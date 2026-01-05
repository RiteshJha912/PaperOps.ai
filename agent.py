from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from tools import get_tools

# System prompt for the ReAct agent
# This tells the agent how to think and use tools
AGENT_PROMPT = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT:
1. You MUST use the exact format above.
2. If you are ready to write the report, your 'Action' must be "Final Answer" (not a tool).
3. The content of the 'Final Answer' must be the FULL MARKDOWN REPORT.
4. Do not output "Final Answer" until you have written the full report.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

def create_research_agent(groq_api_key):
    """
    Creates and returns the agent executor.
    """
    
    # 1. Initialize the LLM (Groq Llama 3.1 8B) with Rate Limit Protection
    import time
    from langchain_core.messages import BaseMessage
    
    class MellowGroq(ChatGroq):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            # Attempt with retries
            for attempt in range(5):
                try:
                    time.sleep(3 + attempt * 2) # Progressive delay: 3s, 5s, 7s...
                    return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                except Exception as e:
                    if "429" in str(e) or "rate_limit" in str(e).lower():
                        print(f"\n[System] Rate Limit hit! Cooling down for {5 + attempt*5} seconds...")
                        time.sleep(5 + attempt * 5)
                        continue
                    raise e
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    llm = MellowGroq(
        temperature=0, 
        model_name="llama-3.1-8b-instant", # Switched to 8B model (Faster & higher limits)
        api_key=groq_api_key
    )

    # 2. Get the tools (passing llm for the summarizer)
    tools = get_tools(llm)

    # 3. Create the Prompt Template
    prompt = PromptTemplate.from_template(AGENT_PROMPT)

    # 4. Create the Agent
    # This combines the LLM, logic, tools, and prompt
    agent = create_react_agent(llm, tools, prompt)

    # 5. Create the Executor (the runtime that actually runs the agent loop)
    # verbose=True lets us see the thinking process in the console!
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=15 # Balanced limit
    )
    
    return agent_executor
