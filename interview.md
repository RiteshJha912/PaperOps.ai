# Project Documentation: Autonomous Research Agent

## 1. One-Line Pitch
An autonomous research agent powered by LangChain and Groq LLMs that uses the ReAct pattern to dynamically search the web, read pages, and generate structured markdown reports on any topic.

## 2. Problem Statement
Manual research requires significant time to query search engines, read through unstructured web pages, and consolidate information into a coherent summary. Existing autonomous agents are often expensive, complex to deploy, or fragile when hitting API rate limits on free tiers.

## 3. Approach / Technical Details
The solution is a terminal-based LangChain agent. A ReAct (Reason + Act) loop governs the workflow: the agent evaluates the prompt, decides whether to search or read, and synthesizes findings. A custom `MellowGroq` wrapper intercepts rate limit errors and implements progressive delays to ensure stability. A summarization tool reads raw HTML via BeautifulSoup, truncates the text to fit context limits, and uses the LLM to pre-summarize large pages before they re-enter the main ReAct loop.

## 4. System Design / Architecture
The architecture consists of three main components:
- **Orchestrator (`main.py`)**: Manages initialization, environment variables, prompt instructions, and user input.
- **Agent Engine (`agent.py`)**: Defines the ReAct loop, system prompts, and the rate-limit-resilient LLM class.
- **Toolset (`tools.py`)**: Provides `web_search` (DuckDuckGo integration) and `read_page` (HTTP fetching, HTML parsing, and context-aware pre-summarization).

Data flows from user input -> agent thought -> search tool -> URL -> fetch/parse -> pre-summary -> agent thought -> markdown report generation.

## 5. Key Features
- **ReAct Prompting Workflow**: Dynamically navigates a multi-step loop (search, read, synthesize) rather than relying on a static, linear chain.
- **Custom Web Scraper & Summarizer**: Bypasses token limits by fetching raw HTML, extracting paragraphs, and employing an intermediate LLM call to summarize large pages before passing data to the main agent.
- **API Rate-Limit Resilience**: A custom `MellowGroq` class intercepts 429 exceptions and applies progressive backoff to prevent runtime crashes.
- **Cost-Free Execution**: Strategically pairs DuckDuckGo Search with the free tier of Groq's high-speed Llama 3 inference.

## 6. Technical Challenges
The primary challenge was handling token limits and API rate restrictions. Fetching full HTML pages easily crashes standard LLM context windows or triggers strict free-tier rate limits. This was solved by truncating scraped text to a safe limit (10,000 characters) and running an isolated LLM call to generate a dense summary. Rate limit crashes were mitigated by building a resilient LLM wrapper that gracefully catches limit exceptions and delays execution.

## 7. Improvements / Future Scope
- **Smarter Scraping**: Enhancing the HTML parser to target specific core tags (e.g., `<article>`, `<main>`) for cleaner text extraction.
- **Conversational Memory**: Adding LangChain memory to allow follow-up conversational questions on the generated report.
- **Asynchronous Execution**: Implementing async processing to search and read multiple sites concurrently for faster generation.

## 8. Impact / Results
Provides developers and students with a free, fully functional, and highly stable template for building ReAct agents. It reduces hours of manual research into a few seconds of automated structure generation.
