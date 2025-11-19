# FAB Financial Analyst - Multi-Agent System

A sophisticated multi-agent financial analysis system for First Abu Dhabi Bank (FAB) that processes financial documents (PDFs) and answers complex financial queries using AI-powered orchestration with LangGraph and LangChain.

## 🎯 Overview

This system transforms unstructured financial documents into actionable insights through:

1. **Intelligent Document Processing**: PDFs → Semantic chunks → Vector embeddings → ChromaDB
2. **Multi-Agent Orchestration**: Specialized AI agents collaborate to answer financial queries
3. **Source Citations**: Every answer is backed by citations to source documents
4. **Full Observability**: LangSmith tracing for complete visibility into agent behavior

## 🏗️ Architecture

### Workflow Graph

The system uses a LangGraph-based workflow with conditional routing between specialized agents:

[![Workflow Diagram](https://mermaid.ink/img/pako:eNqNU9GO2yAQ_BVEpZMj2T7HbmKbnPJy-YR7aqkiYoONhMHC-NpclH8vJilyrhcpb7PszO6wCydYqZpCBKMowrJSkvEGYQkAE-p31RJtXARANep3ioDgkhKNpaM3mvQteNttsMRmvx-Mpe_3wc-Xfuujl-d--2uBEGJcD2YiKl21dDCaWBTMg8WU1dRoTt-JCDxy5xUR1SiI4UoGM-xyw1EaW4UPgUeLiyUqa2_IYW9HkIsbbxRE0RbM7Ww-tQVRDJ7kYeg3jEs-tBcMYiu7Fr-vsIM1XI50pvmv1fzgwV6fJDY7636H4cd6J-9HuLlZx4OOvuA_cnff1G3hZpx2UcOOMlBTRkZhAONCoG8sZQlj4fQco5bypjVoGac3AvfgHD1SPam4OaLkhjA9gWu5AzusWYUlDGGjeQ2R0SMNYUd1R6YQnqZfgKE12VEMkYVXPxhiebaynsgfSnUQMSIGK9VqbFofjX1NDN1xYr9M96-6trek-lWN0kBUpK4GRCf4B6KszOMiKdL18nuZFdkqW4XwaElFvFyVRZJn67zI8yQ9h_DDdU3iVZqkZVqWuZWUxbo4_wWlSVdp?type=png)](https://mermaid.live/edit#pako:eNqNU9GO2yAQ_BVEpZMj2T7HbmKbnPJy-YR7aqkiYoONhMHC-NpclH8vJilyrhcpb7PszO6wCydYqZpCBKMowrJSkvEGYQkAE-p31RJtXARANep3ioDgkhKNpaM3mvQteNttsMRmvx-Mpe_3wc-Xfuujl-d--2uBEGJcD2YiKl21dDCaWBTMg8WU1dRoTt-JCDxy5xUR1SiI4UoGM-xyw1EaW4UPgUeLiyUqa2_IYW9HkIsbbxRE0RbM7Ww-tQVRDJ7kYeg3jEs-tBcMYiu7Fr-vsIM1XI50pvmv1fzgwV6fJDY7636H4cd6J-9HuLlZx4OOvuA_cnff1G3hZpx2UcOOMlBTRkZhAONCoG8sZQlj4fQco5bypjVoGac3AvfgHD1SPam4OaLkhjA9gWu5AzusWYUlDGGjeQ2R0SMNYUd1R6YQnqZfgKE12VEMkYVXPxhiebaynsgfSnUQMSIGK9VqbFofjX1NDN1xYr9M96-6trek-lWN0kBUpK4GRCf4B6KszOMiKdL18nuZFdkqW4XwaElFvFyVRZJn67zI8yQ9h_DDdU3iVZqkZVqWuZWUxbo4_wWlSVdp)

**Workflow Components:**
- **Orchestrator Agent** (GPT-4o): Plans query execution strategy
- **Retrieval Agent** (GPT-4o): Searches document store for relevant data
- **Calculation Agent** (GPT-4o): Performs financial calculations and metric extraction
- **Synthesis Agent** (GPT-4o): Combines results and validates against sources

### System Architecture

```
User Request (FastAPI REST API)
    ↓
┌─────────────────────────────────────────────────────────┐
│ Document Upload Path                                    │
├─────────────────────────────────────────────────────────┤
│ PDF → Gemini LLM (multimodal extraction)               │
│   → Section/page markdown                               │
│   → FABDocumentChunker (~1500 char chunks)             │
│   → Ollama embeddings (768-dim, nomic-embed-text)      │
│   → ChromaDB vector store (persistent)                 │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Query Analysis Path                                     │
├─────────────────────────────────────────────────────────┤
│ Financial Query                                         │
│   → FABAgentManager                                     │
│   → FABWorkflow (LangGraph state machine)              │
│   → Orchestrator plans steps                           │
│   → Conditional routing to Retrieval/Calculation       │
│   → Synthesis combines results                         │
│   → Cited answer with validation                       │
└─────────────────────────────────────────────────────────┘
```

### Technology Stack

**AI & LLM:**
- OpenAI GPT-4o (default for both orchestrator and specialist agents; configurable)
- Google Gemini (document extraction)
- LangChain & LangGraph (agent framework)

**Vector Store & Embeddings:**
- ChromaDB (persistent vector database)
- Ollama (`nomic-embed-text` model, 768-dim)

**Backend:**
- FastAPI (REST API server)
- Python 3.11+
- Pydantic (data validation)

**Observability:**
- LangSmith (agent tracing & monitoring)
- Console logging

### Why GPT-4o (latency & cost)

We use GPT-4o by default for both orchestration and specialist agents because it offers strong reasoning quality with materially lower latency and price than prior GPT‑4 tiers. That balance matters in a multi-agent graph where a single user query can trigger several LLM calls.

- Lower end-to-end latency: keeps interactive queries and multi-step agent flows snappy.
- Better price-performance: reduces per-request cost when multiple agent calls are required.
- Reliable tool use and JSON adherence: pairs well with our “JSON-only” prompts and output sanitizers to keep parsing robust.
- Future-proof: multimodal-capable; while we use Gemini for PDF extraction today, GPT‑4o keeps options open.

You can change models any time (for quality/cost experiments) by editing `backend/main.py` where the `ChatOpenAI` models are set.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.11+** with virtual environment
2. **Ollama** running locally at `http://localhost:11434`
3. **API Keys:**
   - OpenAI API key
   - Google Gemini API key
   - LangSmith API key (optional, for tracing)

### Installation

```powershell
# Clone the repository
git clone https://github.com/sidahmed-faisal/Multi-Agent-Financial-Analyst-for-banking.git
cd Multi-Agent-Financial-Analyst-for-banking

# Install dependencies
pip install -r requirements.txt

# Set up Ollama embedding model
ollama pull nomic-embed-text:latest
ollama serve
```

### Environment Configuration

Create a `.env` file in the project root:

```properties
# Required: OpenAI API
OPENAI_API_KEY=sk-...

# Required: Google Gemini API
GEMINI_API_KEY=AIzaSy...

# Optional: LangSmith Tracing (highly recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=fab-ai-engine
LANGCHAIN_API_KEY=<your-langsmith-api-key>
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### Run the Backend Server

```powershell
# Start FastAPI server with auto-reload
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Run the Streamlit Frontend (Optional)

1) Start the backend API (in a separate terminal):

```powershell
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

2) (Optional) Point the frontend to a different API URL. The Streamlit app reads `FAB_API_URL`:

```powershell
# Default is http://localhost:8000; override if your server runs elsewhere
$env:FAB_API_URL = "http://localhost:8000"
```

3) Launch Streamlit:

```powershell
streamlit run frontend/app.py
```

The app opens at http://localhost:8501. Use it to:
- Upload one or more PDFs via the Upload panel
- Ask a financial question in the Query panel and view the cited answer

Notes
- The frontend calls only two endpoints: `/upload` and `/query`.
- If you change ports/hosts, set `FAB_API_URL` accordingly before launching Streamlit.

### Run the Example Script (Alternative)

If you prefer to test the system without the API server, use the example script:

```powershell
# Query existing documents in the database
python example_usage.py

# OR: Process a new document and then query it
python example_usage.py Documents/FAB-FS-Q1-2025-English.pdf
```

The script will:
1. Process the document (if provided) using the pipeline
2. Store chunks in ChromaDB
3. Run example queries and display results
4. Show system statistics

## 📁 Project Structure

```
Multi-Agent-Financial-Analyst-for-banking/
├── agents/                          # Multi-agent system
│   ├── __init__.py                  # Exports FABAgentManager
│   ├── agent_definitions.py         # BaseAgent, AgentState
│   ├── orchestrator_agent.py        # Query planning & routing
│   ├── retrieval_agent.py           # Document search
│   ├── calculation_agent.py         # Financial calculations
│   ├── synthesis_agent.py           # Answer synthesis & validation
│   └── workflow.py                  # LangGraph workflow
│
├── backend/                         # FastAPI REST API
│   ├── main.py                      # API endpoints
│   └── README.md                    # API documentation
│
├── Document_processor/              # PDF processing pipeline
│   ├── documents_pipeline.py        # File type routing
│   ├── Chunker.py                   # Semantic chunking
│   ├── multimodal_processor.py      # Gemini LLM integration
│   └── ollama_client.py             # Embedding generation
│
├── frontend/                        # Streamlit UI
│   └── app.py                       # Web interface
│
├── monitoring/                      # Observability
│   └── langsmith_tracer.py          # LangSmith integration
│
├── workflow_visualizations/         # Generated workflow diagrams
│   └── fab_workflow_graph.mmd       # Mermaid diagram
│
├── Documents/                       # Upload directory for PDFs
├── chroma_db/                       # ChromaDB persistence
├── example_usage.py                 # CLI example
├── visualize_workflow.py            # Generate workflow diagram
├── requirements.txt                 # Python dependencies
└── .env                             # Environment variables (not committed)
```

## 📊 Features

### Document Processing

✅ **Supported Document Types:**
- Financial Statements: `FAB-FS-Q1-2025-English.pdf`
- Earnings Presentations: `FAB-Earnings-Presentation-Q1-2025.pdf`
- Results Call Transcripts: `FAB-Q1-2025-Results-Call.pdf`

✅ **Processing Pipeline:**
1. Gemini LLM extracts structured markdown with `##Section [NAME]` and `##Page [NUM]` markers
2. Content split into ~1500 character chunks
3. Metadata extraction: quarter, year, section, page, document type
4. Table detection (markdown tables stored separately)
5. Ollama generates 768-dim embeddings with `nomic-embed-text`
6. Chunks stored in ChromaDB with metadata

### Query Processing

✅ **Multi-Agent Workflow:**
1. **Orchestrator** breaks query into steps (RETRIEVE, CALCULATE, SYNTHESIZE)
2. **Retrieval Agent** searches ChromaDB with semantic similarity + filters
3. **Calculation Agent** extracts metrics and performs financial math (using LLM + numexpr)
4. **Synthesis Agent** combines results and validates against source context

✅ **Example Queries:**
- "What was the year-over-year percentage change in Net Profit between Q3 2023 and Q3 2024?"
- "Compare FAB's loan-to-deposit ratio between Q4 2022 and Q4 2023"
- "How has FAB's Return on Equity trended over the last 6 quarters?"
- "What are the key drivers of NIM (Net Interest Margin) changes in Q1 2025?"

### Observability with LangSmith

✅ **Full Tracing:**
- All agent operations traced automatically
- LLM prompt/response inspection
- Token usage tracking
- Performance metrics
- Error debugging

✅ **Trace Dashboard:**
Visit https://smith.langchain.com after configuring `LANGCHAIN_API_KEY` to view:
- Query execution flow
- Agent decision-making
- Retrieval results
- Calculation steps
- Validation outcomes

## 🎮 Usage Examples

### Upload a Document (cURL)

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@Documents/FAB-FS-Q1-2025-English.pdf"
```

### Query Financial Data (cURL)

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was the YoY change in Net Profit Q3 2023 vs Q3 2024?"
  }'
```

### Python Example

```python
from agents import FABAgentManager
from langchain_openai import ChatOpenAI
from Document_processor.Chunker import FABDocumentChunker
import os

# Initialize components
chunker = FABDocumentChunker(chroma_persist_directory="./chroma_db")
orchestrator_llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
specialist_llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Create agent manager
agent_manager = FABAgentManager(orchestrator_llm, specialist_llm, chunker)

# Analyze query
result = agent_manager.analyze_query(
  "What was FAB's Net Profit in Q1 2025?"
)

print(result['final_answer'])
print(f"Sources: {result['sources_used']}")
```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/upload` | POST | Upload and process PDF documents |
| `/query` | POST | Submit financial query for analysis |
| `/stats` | GET | Get system statistics |
| `/documents` | GET | List processed documents |
| `/search` | POST | Direct semantic search |

Full API documentation available at http://localhost:8000/docs when the server is running.

## 🧪 Testing

### Option 1: Test with Example Script (Recommended for Quick Testing)

The example script processes a document and runs queries without needing the API server:

```powershell
# Test with a sample document
python example_usage.py Documents/FAB-FS-Q1-2025-English.pdf
```

This will:
1. ✅ Process the PDF through the document pipeline
2. ✅ Extract sections and metadata using Gemini LLM
3. ✅ Generate embeddings and store in ChromaDB
4. ✅ Run 3 example queries and display answers with sources
5. ✅ Show system statistics

**Query existing documents only** (skip document upload):
```powershell
python example_usage.py
```

### Option 2: Test with API Server

Start the server and use cURL or the interactive docs:

```powershell
# Start server
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal: Upload document
curl -X POST "http://localhost:8000/upload" \
  -F "file=@Documents/FAB-FS-Q1-2025-English.pdf"

# Query the system
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the Net Profit in Q1 2025?"}'
```

### Option 3: Test Workflow Visualization

Generate a visual diagram of the agent workflow:

```powershell
python visualize_workflow.py
```

Generates workflow diagrams in `workflow_visualizations/`

## 📝 Document Naming Conventions

Files must contain one of these identifiers:
- `FS-` for Financial Statements
- `Earnings-Presentation` for Earnings Presentations
- `Results-Call` for Results Call Transcripts

Examples:
- ✅ `FAB-FS-Q1-2025-English.pdf`
- ✅ `FAB-Earnings-Presentation-Q4-2024.pdf`
- ✅ `FAB-Q2-2024-Results-Call.pdf`
- ❌ `financial-report.pdf` (missing identifier)

## 🐛 Troubleshooting

### Issue: "OPENAI_API_KEY is not set"
**Solution:** Ensure `.env` file contains `OPENAI_API_KEY=sk-...`

### Issue: "Ollama server not available"
**Solution:**
```powershell
ollama serve
ollama pull nomic-embed-text:latest
```

### Issue: "No chunks were generated from document"
**Solution:**
- Verify PDF filename follows naming convention
- Check Gemini API key is valid
- Ensure PDF contains extractable text (not scanned images)

### Issue: ChromaDB persistence errors
**Solution:** Delete `./chroma_db` directory and restart server to re-initialize

### Issue: LangSmith traces not appearing
**Solution:**
- Verify `LANGCHAIN_TRACING_V2=true` in `.env`
- Check `LANGCHAIN_API_KEY` is valid
- Install: `pip install langsmith`

## 🚀 Development

### Add New Agent

1. Create agent file in `agents/` (inherit from `BaseAgent`)
2. Implement `execute()` method with tracing
3. Add node to workflow in `workflow.py`
4. Update routing logic in orchestrator

### Extend Document Types

1. Add pattern to `extract_quarter_year_from_filename()` in `documents_pipeline.py`
2. Update Gemini prompt in `multimodal_processor.py`
3. Add metadata extraction logic in `Chunker.py`

### Custom Prompts

Agent prompts are defined in:
- `orchestrator_agent.py`: Query planning
- `retrieval_agent.py`: Search strategy
- `calculation_agent.py`: Metric extraction
- `synthesis_agent.py`: Answer generation

## 📈 Performance

- **Document Upload**: 2-5 minutes per PDF (Gemini extraction + chunking)
- **Query Processing**: 10-30 seconds (depends on complexity)
- **Embedding Generation**: ~50-100ms per chunk (Ollama local with nomic-embed-text)
- **Vector Search**: <100ms for similarity search in ChromaDB

## 🔐 Security Notes

⚠️ **Production Deployment:**
- Never commit API keys to version control
- Add authentication to API endpoints
- Implement rate limiting
- Use HTTPS/TLS
- Deploy with Docker and secrets management (AWS Secrets Manager, Azure Key Vault)
- Enable CORS restrictions
- Add audit logging

## 📚 Dependencies

See `requirements.txt` for full list. Key dependencies:

```
langchain-core
langchain-openai
langgraph
langsmith
chromadb
google-generativeai
fastapi
uvicorn
streamlit
python-dotenv
numexpr
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is proprietary to First Abu Dhabi Bank (FAB).

## 🙋 Support

For issues or questions:
1. Check the [API Documentation](backend/README.md)
2. Review server logs for errors
3. Verify all prerequisites are installed
4. Check LangSmith traces for debugging



---

