
import chromadb
from chromadb.config import Settings
import uuid
import re
from typing import List, Dict, Any
import os
from multimodal_processor import process_fab_document
from Chunker import FABDocumentChunker

class FABDocumentPipeline:
    def __init__(self, chroma_persist_directory: str = "./chroma_db"):
        self.chunker = FABDocumentChunker(chroma_persist_directory)
    
    def process_and_store_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Complete pipeline: Process PDF with Gemini and store in ChromaDB
        """
        # Process document with Gemini (using our previous functions)
        processed_doc = process_fab_document(pdf_path)

        print(processed_doc['document_type'])
        
        # Store in ChromaDB
        chunk_count = self.chunker.process_document_for_storage(processed_doc)
        
        return {
            'filename': processed_doc['metadata']['filename'],
            'document_type': processed_doc['document_type'],
            'chunks_stored': chunk_count,
            'sections_processed': len(processed_doc['metadata']['sections']),
            'metadata': processed_doc['metadata']
        }
    
    def batch_process_documents(self, pdf_directory: str) -> List[Dict[str, Any]]:
        """
        Process all PDF documents in a directory
        """
        results = []
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            try:
                print(f"Processing: {pdf_file}")
                result = self.process_and_store_document(pdf_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                results.append({
                    'filename': pdf_file,
                    'error': str(e),
                    'chunks_stored': 0
                })
        
        return results
    
    def search_financial_data(self, query: str, document_type: str = None, 
                            quarter: str = None, year: str = None, 
                            section: str = None, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Enhanced search with filtering capabilities
        """
        filters = {}
        
        if document_type:
            filters['document_type'] = document_type
        if quarter:
            filters['quarter'] = quarter
        if year:
            filters['year'] = year
        if section:
            filters['section_name'] = section
        
        return self.chunker.search_chunks(query, filters, n_results)