"""
FastAPI server for FAB Financial Analyst multi-agent system.
Provides endpoints for document processing and financial queries.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the document pipeline and agents
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from Document_processor.documents_pipeline import process_fab_document
from Document_processor.Chunker import FABDocumentChunker
from agents import FABAgentManager
from langchain_openai import ChatOpenAI

# Initialize FastAPI app
app = FastAPI(
    title="FAB Financial Analyst API",
    description="Multi-agent system for analyzing FAB financial documents",
    version="1.0.0"
)

# Global instances (initialized on startup)
chunker = None
agent_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize shared resources on server startup"""
    global chunker, agent_manager
    
    try:
        # Initialize document chunker
        chunker = FABDocumentChunker(chroma_persist_directory="./chroma_db")
        
        # Initialize LLMs
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment")
        
        orchestrator_llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=openai_key
        )
        
        specialist_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            api_key=openai_key
        )
        
        # Initialize agent manager
        agent_manager = FABAgentManager(orchestrator_llm, specialist_llm, chunker)
        
        print("✓ Server initialized successfully")
        print(f"✓ ChromaDB path: ./chroma_db")
        print(f"✓ Agent manager ready with orchestrator and specialist LLMs")
        
    except Exception as e:
        print(f"✗ Error during startup: {str(e)}")
        raise


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for financial queries"""
    query: str
    
    class Config:
        example = {
            "query": "What was the year-over-year percentage change in Net Profit between Q3 2023 and Q3 2024?"
        }


class QueryResponse(BaseModel):
    """Response model for financial queries"""
    query: str
    final_answer: str
    sources_used: List[Dict[str, Any]]
    calculations_performed: Dict[str, Any]
    retrieval_steps: List[Dict[str, Any]]
    validation: Dict[str, Any]
    processing_steps: int
    success: bool


class UploadResponse(BaseModel):
    """Response model for file uploads"""
    filename: str
    document_type: str
    metadata: Dict[str, Any]
    chunks_stored: int
    success: bool
    message: str


class SystemStatsResponse(BaseModel):
    """Response model for system statistics"""
    embedding_stats: Dict[str, Any]
    sections_indexed: int
    total_chunks: int
    embedding_model: str
    embedding_dimension: int


# Endpoints

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint to verify server is running.
    """
    return {"status": "healthy", "service": "FAB Financial Analyst API"}


@app.post("/upload", response_model=UploadResponse)
async def upload_and_process_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload and process a FAB financial document (Financial Statement, Earnings Presentation, or Results Call).
    
    The filename must contain one of these keywords:
    - 'FS-' for Financial Statements
    - 'Earnings-Presentation' or 'Earnings Presentation' for Earnings Presentations
    - 'Results-Call' or 'Results Call' for Results Calls
    
    Supported formats:
    - FAB-FS-Q1-2025-English.pdf
    - FAB-Earnings-Presentation-Q1-2025.pdf
    - FAB-Q1-2025-Results-Call.pdf
    
    Returns:
    - Document metadata (type, quarter, year, sections)
    - Number of chunks stored in ChromaDB
    - Processing status
    """
    if not chunker or not agent_manager:
        raise HTTPException(status_code=503, detail="Server not fully initialized")
    
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    temp_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name
        
        # Process the document
        print(f"\n📄 Processing document: {file.filename}")
        processed_doc = process_fab_document(temp_path)
        
        # Store in ChromaDB
        print(f"💾 Storing chunks in ChromaDB...")
        chunks_stored = chunker.process_document_for_storage(processed_doc)
        
        if chunks_stored == 0:
            raise ValueError("No chunks were generated from the document")
        
        return {
            "filename": file.filename,
            "document_type": processed_doc['document_type'],
            "metadata": processed_doc['metadata'],
            "chunks_stored": chunks_stored,
            "success": True,
            "message": f"Successfully processed and stored {chunks_stored} chunks from {file.filename}"
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid document: {str(e)}")
    except Exception as e:
        print(f"✗ Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file {temp_path}: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def analyze_financial_query(request: QueryRequest) -> Dict[str, Any]:
    """
    Submit a financial query and receive an analysis.
    
    The system will:
    1. Break down the query into steps (retrieval, calculation, synthesis)
    2. Retrieve relevant financial data from processed documents
    3. Perform necessary calculations
    4. Synthesize the information into a comprehensive answer
    
    Example queries:
    - "What was the year-over-year percentage change in Net Profit between Q3 2023 and Q3 2024?"
    - "How has FAB's Return on Equity trended over the last 6 quarters?"
    - "Compare FAB's loan-to-deposit ratio between Q4 2022 and Q4 2023."
    
    Returns:
    - Final answer with citations to source documents
    - Number and details of retrieval steps
    - Calculations performed
    - Validation results
    """
    if not agent_manager:
        raise HTTPException(status_code=503, detail="Agent manager not initialized")
    
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        print(f"\n🤖 Processing query: {request.query}")
        result = agent_manager.analyze_query(request.query)
        
        return {
            "query": request.query,
            "final_answer": result.get('final_answer', 'No answer generated'),
            "sources_used": result.get('sources_used', []),
            "calculations_performed": result.get('calculations_performed', {}),
            "retrieval_steps": result.get('retrieval_steps', []),
            "validation": result.get('validation', {}),
            "processing_steps": result.get('processing_steps', 0),
            "success": result.get('success', True)
        }
    
    except Exception as e:
        print(f"✗ Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/stats", response_model=SystemStatsResponse)
async def get_system_statistics() -> Dict[str, Any]:
    """
    Get system statistics including:
    - Total chunks stored in ChromaDB
    - Embedding model information and dimension
    - Indexed sections
    """
    if not agent_manager:
        raise HTTPException(status_code=503, detail="Agent manager not initialized")
    
    try:
        stats = agent_manager.get_system_stats()
        return {
            "embedding_stats": stats.get('embedding_stats', {}),
            "sections_indexed": stats.get('sections_indexed', 0),
            "total_chunks": stats.get('total_chunks', 0),
            "embedding_model": stats.get('embedding_model', 'unknown'),
            "embedding_dimension": stats.get('embedding_dimension', 0)
        }
    except Exception as e:
        print(f"✗ Error retrieving stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")


@app.get("/documents")
async def list_processed_documents() -> Dict[str, Any]:
    """
    List metadata about all processed documents in the system.
    Requires ChromaDB to have been populated with document chunks.
    """
    if not chunker:
        raise HTTPException(status_code=503, detail="Document chunker not initialized")
    
    try:
        section_index = chunker.create_section_index()
        
        # Extract unique documents from section index
        documents_set = set()
        all_data = chunker.collection.get()
        
        for metadata in all_data.get('metadatas', []):
            doc_info = {
                'filename': metadata.get('filename'),
                'document_type': metadata.get('document_type'),
                'quarter': metadata.get('quarter'),
                'year': metadata.get('year'),
            }
            documents_set.add(str(doc_info))
        
        return {
            "total_sections": len(section_index),
            "section_names": list(section_index.keys()),
            "total_chunks": chunker.collection.count(),
            "documents_loaded": len(documents_set) if documents_set else 0
        }
    except Exception as e:
        print(f"✗ Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@app.post("/search")
async def search_documents(query: str = None, n_results: int = 5) -> Dict[str, Any]:
    """
    Direct search in the document store for relevant chunks.
    
    Parameters:
    - query: Search query string
    - n_results: Number of results to return (default: 5)
    
    Returns:
    - List of relevant chunks with metadata and similarity scores
    """
    if not chunker:
        raise HTTPException(status_code=503, detail="Document chunker not initialized")
    
    if not query or len(query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Search query cannot be empty")
    
    if n_results < 1 or n_results > 100:
        raise HTTPException(status_code=400, detail="n_results must be between 1 and 100")
    
    try:
        results = chunker.search_chunks(query=query, n_results=n_results)
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result.get('content', ''),
                "metadata": result.get('metadata', {}),
                "similarity_distance": result.get('distance'),
                "chunk_id": result.get('id')
            })
        
        return {
            "query": query,
            "n_results_returned": len(formatted_results),
            "results": formatted_results
        }
    except Exception as e:
        print(f"✗ Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    # Use --reload for development (auto-reload on code changes)
    # Use --workers N for production (N = number of worker processes)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
