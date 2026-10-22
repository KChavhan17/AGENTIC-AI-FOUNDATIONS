import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

# 1. Load .env FIRST so keys exist before resolving anything
script_dir = Path(__file__).resolve().parent
env_path = script_dir / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass


def resolve_model_id() -> str | None:
    """Choose a valid model ID based on the available API keys."""
    preferred = os.getenv("LANGCHAIN_MODEL")
    if preferred:
        return preferred

    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return os.getenv("GEMINI_MODEL", "google_genai:gemma-4-26b-a4b-it")
    if os.getenv("OPENAI_API_KEY"):
        return "openai:gpt-4o-mini"
    return None


def main() -> None:
    model_id = resolve_model_id()
    if not model_id:
        print("No API key found. Set OPENAI_API_KEY or GOOGLE_API_KEY/GEMINI_API_KEY in your .env file.")
        return

    print(f"Using model: {model_id}")

    try:
        from langchain.chat_models import init_chat_model
    except ImportError:
        print("Missing LangChain dependency. Install it with: pip install langchain python-dotenv")
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            print("If using Google models, also install: pip install langchain-google-genai")
        return

    try:
        model = init_chat_model(model_id)
    except Exception as exc:
        print(f"Could not initialize model '{model_id}': {exc}")
        print("Check that the correct provider package is installed and the API key is valid.")
        return

    # ─────────────────────────────────────────────
    # 1. MODELS — The Reasoning Engine
    # ─────────────────────────────────────────────
    response = model.invoke("What is LangChain in one sentence?")
    print("=== Model Response ===")
    print(response.content)
    print()

    # ─────────────────────────────────────────────
    # 2. PROMPT TEMPLATES — Steering the Model
    # ─────────────────────────────────────────────
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

    simple_template = PromptTemplate(
        input_variables=["topic"],
        template="Explain {topic} to a complete beginner in 2-3 sentences."
    )

    formatted = simple_template.format(topic="AI agents")
    print("=== Formatted Prompt ===")
    print(formatted)
    print()

    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful coding tutor. Keep answers short and clear."),
        ("human", "Explain {concept} with a simple Python example."),
    ])

    messages = chat_template.format_messages(concept="list comprehension")
    print("=== Chat Messages ===")
    for msg in messages:
        print(f"  [{msg.type}]: {msg.content[:80]}...")
    print()

    # ─────────────────────────────────────────────
    # 3. CHAINS — Connecting the Pieces with LCEL
    # ─────────────────────────────────────────────
    from langchain_core.output_parsers import StrOutputParser

    chain = chat_template | model | StrOutputParser()

    result = chain.invoke({"concept": "for loops"})
    print("=== Chain Output ===")
    print(result)
    print()

    # ─────────────────────────────────────────────
    # 4. MEMORY — Giving the Model Context
    # ─────────────────────────────────────────────
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage

    memory = InMemoryChatMessageHistory()

    memory.add_message(HumanMessage(content="My name is Alex and I'm learning LangChain"))
    memory.add_message(AIMessage(content="Nice to meet you, Alex! LangChain is a great choice."))
    memory.add_message(HumanMessage(content="What tools should I learn first?"))
    memory.add_message(AIMessage(content="Start with PromptTemplates and simple chains, then move to tools and agents."))

    print("=== Memory Contents ===")
    for msg in memory.messages:
        print(f"  [{msg.type}]: {msg.content[:80]}...")
    print()

    chat_with_memory = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful tutor. Use the conversation history to personalize your responses."),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ])

    chain_with_memory = chat_with_memory | model | StrOutputParser()

    result = chain_with_memory.invoke({
        "history": memory.messages,
        "question": "What was my name again?"
    })
    print("=== Memory-Aware Response ===")
    print(result)
    print()

    # ─────────────────────────────────────────────
    # 5. TOOLS — Giving the Model Abilities
    # ─────────────────────────────────────────────
    from langchain_core.tools import tool

    @tool
    def calculate_bmi(weight_kg: float, height_m: float) -> str:
        """Calculate Body Mass Index (BMI) given weight in kg and height in meters."""
        bmi = weight_kg / (height_m ** 2)
        if bmi < 18.5:
            category = "underweight"
        elif bmi < 25:
            category = "normal weight"
        elif bmi < 30:
            category = "overweight"
        else:
            category = "obese"
        return f"BMI: {bmi:.1f} ({category})"

    @tool
    def get_word_count(text: str) -> int:
        """Count the number of words in a given text string."""
        return len(text.split())

    model_with_tools = model.bind_tools([calculate_bmi, get_word_count])

    response = model_with_tools.invoke("What's the BMI for someone who is 70kg and 1.75m tall?")
    print("=== Tool Call Response ===")
    print(f"Tool calls: {response.tool_calls}")
    print()

    print("✅ All core concepts demonstrated! Next: first_agent.py")


if __name__ == "__main__":
    main()