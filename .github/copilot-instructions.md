## Purpose

This project is a **multi-agent financial analyst system** for First Abu Dhabi Bank (FAB). It processes PDFs (financial statements, earnings presentations, results call transcripts) → semantic chunks → orchestrates specialized AI agents to answer financial queries with source citations. These instructions help AI agents be immediately productive.

## Big Picture Architecture

```
User Request (FastAPI: backend/main.py)
    ↓
Document Upload Path:
  PDF → Gemini LLM extraction (multimodal_processor) 
    → Section/page markdown → FABDocumentChunker 
    → Ollama embeddings → ChromaDB vector store

Query Analysis Path:
  Financial Query → FABAgentManager (agents package)
    → FABWorkflow (LangGraph state machine)
    → OrchestratorAgent (GPT-4, plans retrieval/calc/synthesis)
    → RetrievalAgent (searches ChromaDB)
    → CalculationAgent (GPT-3.5, financial math)
    → SynthesisAgent (combines + validates)
    → Cited answer
```

**Three major subsystems:**
1. **Document Processing** (`Document_processor/`): PDF → structured markdown → vector chunks
2. **Multi-Agent System** (`agents/`): Query breakdown → orchestrated agent workflow (LangGraph)
3. **REST API** (`backend/main.py`): FastAPI endpoints for upload + query + search

## Key Files by Subsystem

**Document Processing:**
- `Document_processor/documents_pipeline.py` — Router: `process_fab_document(pdf_path)` detects file type (FS-, EARNINGS-PRESENTATION, RESULTS-CALL). Uses Gemini via `multimodal_processor` to extract markdown with strict `#Section [NAME]` + `#Page [NUM]` markers.
- `Document_processor/Chunker.py` — Splits sections → ~1500 char chunks, extracts metadata (quarter/year/section), detects tables (markdown pipes), stores in ChromaDB with Ollama embeddings.
- `Document_processor/multimodal_processor.py` — Calls Gemini LLM for PDF extraction. Loads `GEMINI_API_KEY` from `.env` via dotenv.
- `Document_processor/ollama_client.py` — Calls Ollama at `http://localhost:11434` for embeddings (model: `embeddinggemma`). Returns 1024-dim vectors.

**Multi-Agent Query System:**
- `agents/__init__.py` — Exports `FABAgentManager` (main entry for queries). Uses relative imports (`.workflow`, `.orchestrator_agent`, etc.).
- `agents/workflow.py` — `FABWorkflow` builds LangGraph state machine with 4 nodes (orchestrator → retrieval/calculation/synthesis → synthesis).
- `agents/agent_definitions.py` — `AgentState` TypedDict, `BaseAgent` base class with `_format_context()` helper.
- `agents/orchestrator_agent.py` — Breaks query into step plan. Router to retrieval/calculation/synthesis based on step keywords.
- `agents/retrieval_agent.py` — Searches ChromaDB chunks using semantic similarity + optional filters (quarter, year, section).
- `agents/calculation_agent.py` — Financial math: percentage change, ratios. Uses `numexpr` for safe expression evaluation.
- `agents/synthesis_agent.py` — Combines retrieval results + calculations into final answer. Validates answer against source context.

**REST API:**
- `backend/main.py` — FastAPI server with 6 endpoints: `/health`, `/upload`, `/query`, `/stats`, `/documents`, `/search`. Startup event initializes ChromaDB + LLMs.
- `backend/README.md` — Comprehensive API docs with curl examples and troubleshooting.

## Data Flows & Contracts

**Document Upload Flow:**
- Filename must contain: `FS-` (financial statement), `Earnings-Presentation`, or `Results-Call`
- LLM output (from Gemini) uses exact markers: `##Section [NAME]` + `##Page [NUM]` to delimit sections
- Chunker extracts these markers with regex: `r'##Section\s+([^\n]+)\s+##Page\s+(\d+)'`
- Metadata keys in ChromaDB: `document_type`, `filename`, `quarter`, `year`, `fiscal_period`, `section_name`, `page_number`, `chunk_id`, `content_type` (text|table), `source_document`

**Query Flow (Multi-Agent):**
1. OrchestratorAgent (GPT-4) breaks query into steps: ["RETRIEVE: X", "CALCULATE: Y", "SYNTHESIZE: Z"]
2. For each step, router invokes: RetrievalAgent → ChromaDB search + JSON parse of retrieval plan
3. CalculationAgent validates expressions with numexpr for safe evaluation
4. SynthesisAgent combines results, validates answer against source context
5. Response includes: final_answer, sources_used (metadata), calculations_performed, validation results

## Important Conventions & Patterns

**Import Style:**
- **All intra-package imports use relative syntax** (e.g., `from .agent_definitions import BaseAgent` not `from agent_definitions`). This is critical for package imports to work correctly.
- Update any new intra-package imports to use dot-notation when adding files.

**Environment Variables (via dotenv):**
- `OPENAI_API_KEY` — Required for LangChain ChatOpenAI (GPT-4, GPT-3.5-turbo)
- `GEMINI_API_KEY` — Required for Gemini LLM in document processing
- Both loaded automatically from `.env` via `load_dotenv()` in `example_usage.py`, `backend/main.py`, and `multimodal_processor.py`

**Filename Patterns:**
- Strict patterns: `extract_quarter_year_from_filename()` uses regex — keep naming consistent (e.g., `FAB-FS-Q1-2025-...` or `FAB-Earnings-Presentation-Q1-2025.pdf`)
- Upload endpoint preserves original filename when saving temp file (required for routing to work)

**Section Markers (Critical):**
- Markers are exactly: `##Section [NAME]` and `##Page [NUM]`
- Case-sensitive in extraction regex; Chunker regex: `r'##Section\s+([^\n]+)\s+##Page\s+(\d+)'`
- Table detection expects markdown-table style (pipes and `---` line); stored as separate chunks with `content_type: table`

**Chunk Size:**
- Heuristic: ~1500 characters per chunk before splitting
- Only adjust if you understand embedding/recall tradeoffs

## Integration Points & Runtime Prerequisites

**External Services:**
- **Ollama**: Embedding service at `http://localhost:11434` (configurable). Model: `embeddinggemma:latest`. Start locally: `ollama serve`; load model: `ollama pull embeddinggemma:latest`
- **ChromaDB**: Persistent vector store at `./chroma_db` (configurable). Auto-initialized on first `FABDocumentChunker()` call.
- **LLMs**: OpenAI (via LangChain) for orchestrator and specialists; Google Gemini for document extraction.

**Environment:**
- Python 3.11+ (configured in workspace virtualenv)
- Dependencies: `fastapi`, `uvicorn`, `langchain-openai`, `google-generativeai`, `chromadb`, `python-dotenv`, `langgraph`, `numexpr`
- See `requirements.txt` for full list

**API Server Startup:**
```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
Server initializes `FABDocumentChunker` and `FABAgentManager` on startup (expensive; watch for errors in console).

## Debugging Tips

- **No sections found**: `Chunker.process_document_for_storage` prints warning. Check LLM output contains marker format `##Section` / `##Page` (double hashes).
- **Ollama connection errors**: Verify Ollama running (`ollama serve`), reachable at configured URL, model loaded (`ollama list`).
- **Filename parsing failures**: Inspect `extract_quarter_year_from_filename` in `documents_pipeline.py`; add regex patterns for new naming schemes.
- **LLM API errors**: Check `.env` has valid `OPENAI_API_KEY` and `GEMINI_API_KEY`; verify quota not exceeded.
- **ChromaDB issues**: If corrupted, delete `./chroma_db` directory and restart server (will re-initialize).
- **Query returns no results**: Run `/search` endpoint first to verify documents exist; check `/documents` endpoint for indexed sections.

## Quick Actionable Edits AI Agents Can Make

1. **Add unit tests** for `extract_quarter_year_from_filename()` and `extract_metadata_from_content()` — cover happy path + edge cases (missing fields, malformed names).
2. **Enhance agent prompts** in `orchestrator_agent.py`, `retrieval_agent.py`, `calculation_agent.py` for domain-specific improvements (add financial terminology, validation checks).
3. **Extend ChromaDB filters** in `retrieval_agent.py` — add support for filtering by document_type, content_type (text vs. table).
4. **Add rate limiting + auth** to FastAPI endpoints in `backend/main.py` for production readiness.
5. **Implement LLM retries** with exponential backoff in agent nodes to handle transient API failures.
6. **Add request logging** to FastAPI startup for monitoring query patterns and debugging.

## Where to Look Next

- **Start here**: `Document_processor/documents_pipeline.py` → `process_fab_document()` to understand file type routing logic.
- **Then**: `agents/workflow.py` → `FABWorkflow._build_graph()` to see LangGraph state machine structure.
- **For queries**: `agents/orchestrator_agent.py` → `create_plan()` shows how GPT-4 breaks down queries into steps.
- **For API**: `backend/main.py` → `/upload` and `/query` endpoints show integration points.
- **Deep dive**: `Document_processor/Chunker.py` → `chunk_section_content()` for understanding embedding strategy.

---

**Key principle**: Preserve filename identity during temp file handling in upload endpoints; maintain relative imports in packages; always validate section markers match expected case/format.
