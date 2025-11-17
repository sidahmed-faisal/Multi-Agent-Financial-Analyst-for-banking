# example_usage.py
from langchain_openai import ChatOpenAI
from agents import FABAgentManager
from Document_processor.Chunker import FABDocumentChunker
import os

# Load environment variables from .env (if present)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # If python-dotenv is not installed, we proceed and expect env vars to be set externally
    pass

# Initialize components
chunker = FABDocumentChunker()

# LLMs - using OpenAI for example (you can replace with any LLM)
orchestrator_llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

specialist_llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Create agent manager
agent_manager = FABAgentManager(orchestrator_llm, specialist_llm, chunker)

# Example queries
queries = [
    "What was the year-over-year percentage change in Net Profit between Q3 2023 and Q3 2024?",
    "How has FAB's Return on Equity trended over the last 6 quarters?",
    "Compare FAB's loan-to-deposit ratio between Q4 2022 and Q4 2023.",
]

# Process queries
for query in queries:
    result = agent_manager.analyze_query(query)
    print(f"\n📊 Query: {query}")
    print(f"✅ Answer: {result['final_answer']}")
    print(f"🔍 Sources used: {len(result['sources_used'])}")
    print(f"🧮 Calculations: {len(result['calculations_performed'])}")
    print("---")

# Get system stats
stats = agent_manager.get_system_stats()
print(f"\n📈 System Stats: {stats}")