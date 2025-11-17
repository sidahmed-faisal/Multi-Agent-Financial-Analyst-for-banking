# FAB Financial Analyst API Server

A FastAPI-based REST API server for the multi-agent financial analysis system. This server provides endpoints to upload and process FAB financial documents, and to submit financial queries for analysis using the multi-agent system.

## Features

- **Document Upload & Processing**: Upload FAB financial documents (Financial Statements, Earnings Presentations, Results Call Transcripts) and automatically process them using Gemini LLM with semantic chunking
- **Financial Query Analysis**: Submit complex financial queries and receive AI-powered analyses using a multi-agent orchestration system
- **Document Search**: Search processed documents by semantic similarity
- **System Statistics**: Monitor the document store, embedding models, and indexed content
- **Real-time Streaming**: Track document processing and query analysis in real-time
- **LangSmith Tracing**: Full observability into multi-agent behavior, LLM calls, and workflow execution

## LangSmith Tracing & Observability

This project includes comprehensive LangSmith integration for tracing and monitoring all agent behavior:

### What Gets Traced

- **Orchestrator Agent**: Query planning and step decomposition
- **Retrieval Agent**: Document searches, filters, and fallback strategies
- **Calculation Agent**: Financial calculations, metric extraction, and validation
- **Synthesis Agent**: Answer synthesis and validation against source context
- **Workflow Nodes**: Overall workflow execution from start to completion

### Setup

1. **Create LangSmith Account** (if you don't have one):
   - Go to https://smith.langchain.com and create an account
   - Get your API key from your account settings

2. **Configure Environment Variables** (in `.env` file):
   ```properties
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=fab-ai-engine
   LANGCHAIN_API_KEY=<your-langsmith-api-key>
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   ```

3. **Enable Tracing** (automatic):
   - LangSmith tracing is automatically enabled when environment variables are set
   - All agent operations are traced and visible in your LangSmith dashboard

### Viewing Traces

1. Navigate to your LangSmith project dashboard: https://smith.langchain.com/projects
2. Select the `fab-ai-engine` project
3. View traces for:
   - **Individual queries**: See the complete agent workflow
   - **LLM calls**: Inspect prompts, responses, and token usage
   - **Agent operations**: Debug retrieval, calculation, and synthesis steps
   - **Errors**: Track failures and edge cases

### Trace Data Captured

Each trace includes:
- **Inputs**: Query parameters, search criteria, calculation requests
- **Outputs**: Retrieved documents, calculation results, final answers
- **Metadata**: Agent names, operation types, timestamps
- **Performance**: Execution times for each step
- **Errors**: Exception details for debugging

### Example Query Trace Flow

```
Query: "What was the YoY change in Net Profit Q3 2023 vs Q3 2024?"
    ↓
execute_query_start (Workflow)
    ↓
orchestrator_node_initial (Plan creation)
    ├─ create_plan (Orchestrator Agent)
    └─ Generated plan: [RETRIEVE Q3 2023, RETRIEVE Q3 2024, CALCULATE change, SYNTHESIZE]
    ↓
retrieval_node (First retrieval)
    ├─ execute_retrieval (Retrieval Agent)
    └─ Retrieved 3 results from financial statements
    ↓
retrieval_node (Second retrieval)
    ├─ execute_retrieval (Retrieval Agent)
    └─ Retrieved 3 results from financial statements
    ↓
calculation_node (Calculate YoY change)
    ├─ perform_calculation (Calculation Agent)
    └─ Result: 15.28% increase
    ↓
synthesis_node (Combine into final answer)
    ├─ synthesize_answer (Synthesis Agent)
    ├─ validate_answer (Synthesis Agent)
    └─ Final answer with citations
    ↓
execute_query_complete (Workflow)
    └─ All traces published to LangSmith
```


### Environment Setup

1. **Python Virtual Environment**
   - The project uses a Python virtual environment at `C:/Users/User/.virtualenvs/packages-VbbqGVa_/`
   - Activate it in PowerShell:
     ```powershell
     C:/Users/User/.virtualenvs/packages-VbbqGVa_/Scripts/Activate.ps1
     ```

2. **Environment Variables** (in `.env` file)
   ```properties
   # OpenAI and Gemini APIs
   OPENAI_API_KEY=sk-...          # OpenAI API key for GPT-4 and GPT-3.5-turbo
   GEMINI_API_KEY=AIzaSy...       # Google Gemini API key for document processing
   
   # LangSmith Tracing (Optional but recommended)
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=fab-ai-engine
   LANGCHAIN_API_KEY=<your-langsmith-api-key>
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   ```

3. **External Services**
   - **Ollama**: Embedding service running at `http://localhost:11434`
     - Load the embedding model: `ollama pull embeddinggemma:latest`
     - Start Ollama: `ollama serve`
   - **ChromaDB**: Persistent vector database (auto-initialized in `./chroma_db`)

### Installation

```powershell
# Install dependencies
pip install -r requirements.txt

# Or with explicit path
C:/Users/User/.virtualenvs/packages-VbbqGVa_/Scripts/pip.exe install -r requirements.txt
```

## Running the Server

### Development Mode (with auto-reload)

```powershell
# Using the configured Python interpreter
C:/Users/User/.virtualenvs/packages-VbbqGVa_/Scripts/python.exe -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode (with workers)

```powershell
# Run with 4 worker processes
C:/Users/User/.virtualenvs/packages-VbbqGVa_/Scripts/python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### From Project Root

```powershell
# Alternative: Run from the backend directory
cd backend
C:/Users/User/.virtualenvs/packages-VbbqGVa_/Scripts/python.exe main.py
```

The server will start at `http://localhost:8000`

## Interactive API Documentation

Once the server is running, access the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Health Check

**GET** `/health`

Check if the server is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "service": "FAB Financial Analyst API"
}
```

---

### 2. Upload and Process Document

**POST** `/upload`

Upload and process a FAB financial document. The system will:
1. Use Gemini LLM to extract structured content from the PDF
2. Split content into semantic chunks (~1500 chars each)
3. Generate embeddings using Ollama
4. Store chunks in ChromaDB for later retrieval

**Supported Document Types:**
- Financial Statements: `FAB-FS-Q1-2025-English.pdf`
- Earnings Presentations: `FAB-Earnings-Presentation-Q1-2025.pdf`
- Results Calls: `FAB-Q1-2025-Results-Call.pdf`

**Request:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -F "file=@FAB-FS-Q1-2025-English.pdf"
```

**Response (Success):**
```json
{
  "filename": "FAB-FS-Q1-2025-English.pdf",
  "document_type": "financial_statement",
  "metadata": {
    "document_type": "financial_statement",
    "filename": "FAB-FS-Q1-2025-English.pdf",
    "quarter": "Q1",
    "year": "2025",
    "fiscal_period": "2025-Q1",
    "sections": [
      {"section": "Balance Sheet", "page": 3},
      {"section": "Income Statement", "page": 5}
    ],
    "total_pages": 12,
    "pages_covered": [3, 5, 7, 8, 9]
  },
  "chunks_stored": 42,
  "success": true,
  "message": "Successfully processed and stored 42 chunks from FAB-FS-Q1-2025-English.pdf"
}
```

**Response (Error):**
```json
{
  "detail": "Only PDF files are supported"
}
```

---

### 3. Analyze Financial Query

**POST** `/query`

Submit a financial query for multi-agent analysis. The system will:
1. Break down the query into analysis steps
2. Retrieve relevant data from document store
3. Perform calculations (if needed)
4. Synthesize findings with source citations

**Request:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was the year-over-year percentage change in Net Profit between Q3 2023 and Q3 2024?"
  }'
```

**Request Body:**
```json
{
  "query": "How has FAB's Return on Equity trended over the last 6 quarters?"
}
```

**Example Queries:**
- "What was the year-over-year percentage change in Net Profit between Q3 2023 and Q3 2024?"
- "Compare FAB's loan-to-deposit ratio between Q4 2022 and Q4 2023"
- "What are the key drivers of NIM (Net Interest Margin) changes in Q1 2025?"
- "How did FAB's cost-to-income ratio trend through 2024?"

**Response:**
```json
{
  "query": "What was the year-over-year percentage change in Net Profit between Q3 2023 and Q3 2024?",
  "final_answer": "Based on FAB's financial statements, Net Profit After Tax increased by 15.3% YoY from Q3 2023 (AED 3,200 Mn) to Q3 2024 (AED 3,689 Mn). This growth was driven by: 1) Higher net interest income (+12%), 2) Increased fee and commission income (+18%), offset by 3) Higher impairment charges (+8%).",
  "sources_used": [
    {
      "document": "FAB-FS-Q3-2024.pdf",
      "section": "Income Statement",
      "page": 5,
      "quarter": "Q3",
      "year": "2024",
      "content_preview": "Net Profit After Tax for Q3 2024..."
    }
  ],
  "calculations_performed": {
    "step_0": {
      "calculation_type": "percentage_change",
      "formula_used": "(3689 - 3200) / 3200 * 100",
      "result": 15.28,
      "units": "percentage"
    }
  },
  "retrieval_steps": [
    {
      "step": 0,
      "query": "Net Profit After Tax Q3 2023 Q3 2024",
      "results_count": 5
    }
  ],
  "validation": {
    "is_valid": true,
    "unsupported_claims": [],
    "validation_notes": "All claims supported by source documents"
  },
  "processing_steps": 3,
  "success": true
}
```

---

### 4. System Statistics

**GET** `/stats`

Get information about the document store and embedding system.

**Request:**
```bash
curl "http://localhost:8000/stats"
```

**Response:**
```json
{
  "embedding_stats": {
    "total_chunks": 487,
    "embedding_model": "embeddinggemma",
    "embedding_dimension": 1024,
    "ollama_url": "http://localhost:11434"
  },
  "sections_indexed": 42,
  "total_chunks": 487,
  "embedding_model": "embeddinggemma",
  "embedding_dimension": 1024
}
```

---

### 5. List Processed Documents

**GET** `/documents`

Get metadata about all processed documents.

**Request:**
```bash
curl "http://localhost:8000/documents"
```

**Response:**
```json
{
  "total_sections": 42,
  "section_names": [
    "Balance Sheet",
    "Income Statement",
    "Key Highlights",
    "Capital & Liquidity"
  ],
  "total_chunks": 487,
  "documents_loaded": 3
}
```

---

### 6. Search Documents

**POST** `/search`

Direct semantic search in the document store.

**Request:**
```bash
curl -X POST "http://localhost:8000/search?query=Net+Profit+trends&n_results=5"
```

**Query Parameters:**
- `query` (required): Search query string
- `n_results` (optional, default: 5): Number of results (1-100)

**Response:**
```json
{
  "query": "Net Profit trends",
  "n_results_returned": 3,
  "results": [
    {
      "content": "Net Profit After Tax increased from AED 3,200 Mn in Q3 2023 to AED 3,689 Mn in Q3 2024...",
      "metadata": {
        "document_type": "financial_statement",
        "filename": "FAB-FS-Q3-2024.pdf",
        "section_name": "Income Statement",
        "page_number": 5,
        "quarter": "Q3",
        "year": "2024",
        "chunk_id": 15
      },
      "similarity_distance": 0.23,
      "chunk_id": "uuid-12345"
    }
  ]
}
```

---

## Common Issues and Troubleshooting

### Issue: "OPENAI_API_KEY is not set"
**Solution:** 
- Ensure `.env` file contains: `OPENAI_API_KEY=sk-...`
- Or set environment variable: `$env:OPENAI_API_KEY = "sk-..."`
- The `.env` file is loaded automatically via `python-dotenv`

### Issue: "GEMINI_API_KEY is not set"
**Solution:**
- Ensure `.env` file contains: `GEMINI_API_KEY=AIzaSy...`
- Document processing requires Gemini API for PDF extraction

### Issue: "Ollama server not available at http://localhost:11434"
**Solution:**
- Start Ollama: `ollama serve`
- Load embedding model: `ollama pull embeddinggemma:latest`
- Verify it's running: `curl http://localhost:11434/api/tags`

### Issue: "No chunks were generated from the document"
**Solution:**
- Verify the PDF filename follows the expected pattern (contains FS-, Earnings-Presentation, or Results-Call)
- Check that the PDF is readable and contains text
- Review Gemini API key and quota

### Issue: "ChromaDB persistence not working"
**Solution:**
- Ensure write permissions for `./chroma_db` directory
- Check disk space availability
- If corrupted, delete `./chroma_db` and restart server

---

## Example Workflow

### Step 1: Start the Server
```powershell
cd C:\Users\User\Desktop\Multi-Agent-Financial-Analyst-for-banking
C:/Users/User/.virtualenvs/packages-VbbqGVa_/Scripts/python.exe -m uvicorn backend.main:app --reload
```

### Step 2: Verify Server Health
```bash
curl http://localhost:8000/health
```

### Step 3: Upload a Document
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/FAB-FS-Q1-2025-English.pdf"
```

### Step 4: Submit a Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FAB'\''s latest Net Profit?"}'
```

### Step 5: Check System Status
```bash
curl http://localhost:8000/stats
```

---

## Architecture

### Request Flow

```
Client Request
    ↓
FastAPI Endpoint
    ↓
[Document Upload]          [Financial Query]
    ↓                            ↓
Gemini LLM                 Multi-Agent System
(PDF extraction)           ├─ Orchestrator Agent
    ↓                      ├─ Retrieval Agent
Chunker                    ├─ Calculation Agent
(split & embed)            ├─ Synthesis Agent
    ↓                            ↓
ChromaDB                   Response JSON
(vector store)
    ↓
Response JSON
```

### Component Interaction

```
FastAPI Server
├─ Document Processor Pipeline
│  └─ Gemini LLM + Ollama Embeddings + ChromaDB
├─ Multi-Agent System (with LangSmith Tracing)
│  ├─ Orchestrator (GPT-4) → Trace planning
│  ├─ Retrieval Agent (GPT-3.5-turbo) → Trace searches
│  ├─ Calculation Agent (GPT-3.5-turbo) → Trace calculations
│  └─ Synthesis Agent (GPT-3.5-turbo) → Trace synthesis
├─ LangSmith Integration
│  └─ Traces all agent operations and LLM calls
└─ Vector Store (ChromaDB with Ollama embeddings)
```

### Observability Stack

- **LangSmith**: Traces agent behavior, LLM calls, and workflow execution
- **Console Logging**: Real-time debugging output
- **ChromaDB**: Vector store with persistent metadata
- **FastAPI Logging**: HTTP request/response logging

---

## Performance Notes

- **Document Upload**: 2-5 minutes per document (depending on PDF size and Gemini API response time)
- **Query Processing**: 10-30 seconds (depends on query complexity and number of retrieval steps)
- **Embedding Generation**: ~100-200 ms per chunk with Ollama locally
- **ChromaDB Storage**: Persistent on disk; loads on server startup

---

## Security Considerations

⚠️ **Important**: This server is designed for internal use. For production deployment:

1. **API Key Management**
   - Never commit API keys to version control
   - Use `.env` for local development only
   - Deploy to production with environment-based secrets (AWS Secrets Manager, Azure Key Vault, etc.)

2. **Authentication**
   - Add API key or OAuth authentication to endpoints
   - Implement rate limiting
   - Use HTTPS/TLS for transport security

3. **Access Control**
   - Restrict document upload to authenticated users
   - Log all queries and document processing
   - Implement audit trails

---

## Next Steps

1. **Setup LangSmith**: Enable full observability by configuring LangSmith API key in `.env`
2. **Extend Agents**: Customize agent prompts and behaviors for specific use cases
3. **Add Authentication**: Implement API key or OAuth2 authentication
4. **Database**: Replace ChromaDB with production-grade vector database (Pinecone, Weaviate, etc.)
5. **Monitoring**: Configure advanced monitoring and alerting (e.g., Prometheus, Grafana)
6. **Deployment**: Deploy to cloud (AWS, Azure, GCP) with containerization (Docker)

---

## Troubleshooting

### LangSmith Tracing Issues

**Traces not appearing in LangSmith dashboard:**
- Verify `LANGCHAIN_TRACING_V2=true` in `.env`
- Verify `LANGCHAIN_API_KEY` is valid (get from https://smith.langchain.com)
- Check that `monitoring/langsmith_tracer.py` is present in the project
- Ensure `langsmith` package is installed: `pip install langsmith`
- Check server logs for tracer initialization messages

**High latency with tracing enabled:**
- LangSmith traces are sent asynchronously, should not impact query latency
- If issues persist, verify network connectivity to `api.smith.langchain.com`


## Support

For issues, errors, or questions:
1. Check the server logs (printed to console)
2. Review the API documentation at `/docs` (Swagger)
3. Verify all prerequisites are installed and running
4. Check `.env` file has all required API keys

