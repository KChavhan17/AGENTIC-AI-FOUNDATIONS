# AGENTIC-AI-FOUNDATIONS

Foundations for building LLM-powered agentic systems using LangChain, Google Gemini (gemini-3.8-flash), and function-calling tools.

---

## Features

- Model Integration: Dynamic initialization using gemini-3.8-flash.
- Prompt Engineering: Dynamic templating via ChatPromptTemplate.
- LCEL Chains: Pipelines built with LangChain Expression Language.
- Memory: Context persistence with InMemoryChatMessageHistory.
- ReAct Agent: Tool-calling agent executing custom math operations.

---

```text
agentic-ai-foundations/
├── LangChain/
│   ├── core_building_blocks.py
│   ├── first_agent.py
│   └── .env
├── .gitignore
└── README.md
```

---

## Setup and Execution

### 1. Install Dependencies
pip install langchain langchain-google-genai langgraph python-dotenv

### 2. Configure Environment
Add your key to LangChain/.env:
GOOGLE_API_KEY=your_gemini_api_key_here

### 3. Run Scripts
cd LangChain
python core_building_blocks.py
python first_agent.py
