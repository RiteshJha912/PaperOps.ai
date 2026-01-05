# DeepResearch Agent 

A professional-grade, autonomous research agent powered by LangChain and Groq LLMs. This tool intelligently searches the web, analyzes academic sources, and compiles structured research reports on any given topic.

##  Features

-   **Autonomous Agentic Workflow**: Uses the ReAct (Reason + Act) pattern to dynamically plan and execute research.
-   **Smart Web Search**: Integrates with DuckDuckGo to find high-quality, relevant sources.
-   **Intelligent Summarization**: Reads and distills complex web pages using Llama 3 models.
-   **Rate-Limit Resilient**: detailed "Mellow Mode" architecture ensures stability even on free API tiers.
-   **Zero Cost**: Built entirely on free tools (Groq API + DuckDuckGo).

##  Tech Stack

-   **Python 3.11+**
-   **LangChain**: For agent orchestration and tool management.
-   **Groq**: For ultra-fast Llama 3 inference.
-   **BeautifulSoup4**: For robust web scraping.

##  Installation

1.  **Clone the Repository** (if applicable)
2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

##  Configuration

1.  Obtain a free API Key from [Groq Console](https://console.groq.com/keys).
2.  **Create a `.env` file** in the project root:
    ```bash
    cp .env.example .env
    ```
3.  Add your API key to the `.env` file:
    ```ini
    GROQ_API_KEY=gsk_your_actual_api_key_here
    ```


##  Usage

Run the agent via the command line:

```bash
# Ensure your virtual environment is active
python main.py
```

Follow the prompts to enter your research topic (e.g., *"Quantum Computing applications in Finance"*). The agent will:
1.  **Think**: Formulate a search strategy.
2.  **Search & Read**: Gather information from multiple sources.
3.  **Write**: Generate a comprehensive markdown report (`report.md`) in the current directory.
