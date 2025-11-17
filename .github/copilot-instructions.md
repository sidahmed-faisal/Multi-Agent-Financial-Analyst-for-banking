## Purpose

This project processes First Abu Dhabi Bank (FAB) PDFs (financial statements, earnings presentations, results call transcripts) into structured markdown and stores semantic chunks in ChromaDB with Ollama embeddings. These instructions capture the minimal, actionable knowledge an AI coding assistant needs to be productive in this repo.

## Big picture (what runs where)
- Document processing pipeline lives under `Document_processor/`.
- `documents_pipeline.py` (main router) detects file type and invokes specialized processing.
- `multimodal_processor` / Gemini (Google GenAI) is used to extract structured markdown from PDFs.
- `Chunker.py` turns parsed sections into chunks and writes them to a ChromaDB persistent collection using an Ollama embedding client (`ollama_client.py`).

## Key files to inspect
- `Document_processor/documents_pipeline.py` — main entry: `process_fab_document(pdf_path)`; enforces filename-based routing (FS-, EARNINGS-PRESENTATION, RESULTS-CALL).
- `Document_processor/Chunker.py` — chunking strategy (table-aware), chunk size heuristic (~1500 chars), metadata shape stored in ChromaDB.
- `Document_processor/ollama_client.py` — HTTP client that calls Ollama embedding endpoint at `http://localhost:11434` and expects `embeddinggemma` model. Returns embeddings (fallback 1024-dim zero vector on error).

## Data flows and contracts
- Input: PDF file path. Filenames must include tokens like `FS-`, `Earnings-Presentation`, or `Results-Call` to route correctly.
- Output from the LLM extractor: markdown using exact markers:
  - `#Section [SECTION_NAME]` followed by `#Page [PAGE_NUMBER]` — code and chunker rely on these exact markers to extract sections.
- Chunk metadata keys (examples): `document_type`, `filename`, `quarter`, `year`, `fiscal_period`, `section_name`, `page_number`, `chunk_id`, `content_type`, `source_document`.

## Important conventions & patterns
- Strict filename patterns: `extract_quarter_year_from_filename` uses several regexes — keep naming consistent (e.g. `FAB-FS-Q1-2025-...` or `FAB-Earnings-Presentation-Q1-2025.pdf`).
- Section markers are canonical and case-sensitive in extraction regex: `#Section` and `#Page`.
- Table detection in chunker expects markdown-table style (pipes and `---` line). Tables are stored as separate chunks with `content_type: table`.
- Chunk size heuristic: ~1500 characters per chunk before splitting — adjust only if you understand downstream embedding/recall tradeoffs.

## Integration points & runtime prerequisites
- Ollama embedding service: code expects Ollama at `http://localhost:11434` and a model named like `embeddinggemma:latest`. Start Ollama locally and load embedding model before ingesting.
- ChromaDB persistent store: default path `./chroma_db` (change when creating `FABDocumentChunker`).
- Google GenAI usage: `google.generativeai` (Gemini) is used and currently configured with an API key inside the code — replace with environment-based configuration before production.

## Debugging tips
- If no sections are found, `Chunker.process_document_for_storage` prints a warning: check that the LLM output contains the `#Section` / `#Page` markers.
- If embeddings are zeros or unexpected dimension, confirm Ollama is reachable at `ollama_client.base_url` and model name matches.
- For filename parsing failures, inspect `extract_quarter_year_from_filename` in `documents_pipeline.py` to add new patterns.

## Quick actionable edits an assistant might make
- Move hard-coded API keys out of `documents_pipeline.py` to environment variables and update code to read them.
- Add small unit tests for `extract_quarter_year_from_filename` and `extract_metadata_from_content` (happy path + 1-2 edge cases).
- Add a README snippet describing how to start Ollama locally and which model to load for embeddings.

## Where to look next
- Start at `Document_processor/documents_pipeline.py` to understand routing and LLM prompts.
- Inspect `Document_processor/Chunker.py` to see how content is split and what metadata is required by downstream services.

If any section is unclear or you want more detail (example commands to run Ollama/ChromaDB or a suggested scrubbed sample of the GenAI prompt), tell me which part and I will expand or add run instructions.
