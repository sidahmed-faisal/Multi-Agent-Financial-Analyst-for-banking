import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import uuid
import re
from typing import List, Dict, Any
import os
from multimodal_processor import process_fab_document
from ollama_client import OllamaEmbeddingClient


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function wrapper for ChromaDB"""
    def __init__(self, ollama_client: OllamaEmbeddingClient):
        self.ollama_client = ollama_client
    
    def __call__(self, input: Documents) -> Embeddings:
        """ChromaDB calls this method to get embeddings"""
        return self.ollama_client.get_embeddings_batch(input)


class FABDocumentChunker:
    def __init__(self, chroma_persist_directory: str = "./chroma_db", ollama_url: str = ""):
        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_directory)
        # If an explicit URL is provided prefer it; otherwise OllamaEmbeddingClient will read OLLAMA_URL env var
        self.embedding_client = OllamaEmbeddingClient(base_url=ollama_url if ollama_url else None)
        
        # Create the embedding function wrapper
        embedding_function = OllamaEmbeddingFunction(self.embedding_client)
        
        # Create collection with custom embedding function
        self.collection = self.chroma_client.get_or_create_collection(
            name="fab_financial_documents",
            metadata={"description": "FAB Financial Statements, Presentations, and Earnings Calls"},
            embedding_function=embedding_function
        )
    
    def extract_sections_with_pages(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract sections and their content with page numbers from processed content
        """
        sections = []
        
        # Pattern to match sections with page numbers
        section_pattern = r'#Section\s+([^\n]+)\s+#Page\s+(\d+)(.*?)(?=\n#Section|\Z)'
        matches = re.findall(section_pattern, content, re.DOTALL)
        
        for section_name, page_num, section_content in matches:
            section_content = section_content.strip()
            if section_content:  # Only include non-empty sections
                sections.append({
                    'section_name': section_name.strip(),
                    'page_number': int(page_num),
                    'content': section_content
                })
        
        return sections
    
    def chunk_section_content(self, section: Dict[str, Any], doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split section content into manageable chunks while preserving context
        """
        content = section['content']
        chunks = []
        
        # Strategy: Split by logical boundaries within sections
        # For financial documents, preserve table structures and logical groupings
        
        # First, try to split by markdown tables or major headings
        table_split_pattern = r'(\|.*?\|\n\|.*?\|\n(?:\|.*?\|\n)*)'
        parts = re.split(table_split_pattern, content)
        
        current_chunk = ""
        chunk_id = 0
        
        for part in parts:
            if not part.strip():
                continue
                
            # Check if this part is a table
            is_table = part.strip().startswith('|') and '---' in part
            
            if is_table:
                # If we have accumulated text, create a chunk first
                if current_chunk.strip():
                    chunks.append(self._create_chunk_dict(
                        content=current_chunk.strip(),
                        section=section,
                        doc_metadata=doc_metadata,
                        chunk_id=chunk_id
                    ))
                    chunk_id += 1
                    current_chunk = ""
                
                # Create a separate chunk for the table
                chunks.append(self._create_chunk_dict(
                    content=part.strip(),
                    section=section,
                    doc_metadata=doc_metadata,
                    chunk_id=chunk_id,
                    content_type="table"
                ))
                chunk_id += 1
            else:
                # Regular text - split by paragraphs if too long
                paragraphs = [p.strip() for p in part.split('\n\n') if p.strip()]
                
                for paragraph in paragraphs:
                    # If adding this paragraph would make chunk too big, save current chunk
                    if len(current_chunk) + len(paragraph) > 1500 and current_chunk:
                        chunks.append(self._create_chunk_dict(
                            content=current_chunk.strip(),
                            section=section,
                            doc_metadata=doc_metadata,
                            chunk_id=chunk_id
                        ))
                        chunk_id += 1
                        current_chunk = ""
                    
                    current_chunk += paragraph + "\n\n"
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk_dict(
                content=current_chunk.strip(),
                section=section,
                doc_metadata=doc_metadata,
                chunk_id=chunk_id
            ))
        
        return chunks
    
    def _create_chunk_dict(self, content: str, section: Dict[str, Any], 
                          doc_metadata: Dict[str, Any], chunk_id: int, 
                          content_type: str = "text") -> Dict[str, Any]:
        """Create a standardized chunk dictionary with metadata"""
        return {
            'id': str(uuid.uuid4()),
            'content': content,
            'metadata': {
                'document_type': doc_metadata['document_type'],
                'filename': doc_metadata['filename'],
                'quarter': doc_metadata.get('quarter', 'unknown'),
                'year': doc_metadata.get('year', 'unknown'),
                'fiscal_period': doc_metadata.get('fiscal_period', 'unknown'),
                'section_name': section['section_name'],
                'page_number': section['page_number'],
                'chunk_id': chunk_id,
                'content_type': content_type,
                'full_section': section['section_name'],
                'source_document': doc_metadata['filename'],
                'embedding_model': 'embeddinggemma'
            }
        }
    
    def process_document_for_storage(self, processed_doc: Dict[str, Any]) -> int:
        """
        Process a document and store all chunks in ChromaDB with Ollama embeddings
        Returns number of chunks stored
        """
        content = processed_doc['content']
        metadata = processed_doc['metadata']
        doc_type = processed_doc['document_type']
        
        # Extract sections from content
        sections = self.extract_sections_with_pages(content)
        
        if not sections:
            print(f"Warning: No sections found in {metadata['filename']}")
            return 0
        
        all_chunks = []
        
        for section in sections:
            # Chunk the section content
            section_chunks = self.chunk_section_content(section, metadata)
            all_chunks.extend(section_chunks)
        
        if not all_chunks:
            print(f"Warning: No chunks created from {metadata['filename']}")
            return 0
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for chunk in all_chunks:
            documents.append(chunk['content'])
            metadatas.append(chunk['metadata'])
            ids.append(chunk['id'])
        
        # Store in ChromaDB with Ollama embeddings
        try:
            print(f"Generating embeddings and storing {len(documents)} chunks...")
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✓ Successfully stored {len(documents)} chunks from {metadata['filename']} using embeddinggemma")
        except Exception as e:
            print(f"✗ Error storing chunks: {str(e)}")
            raise
        
        return len(documents)
    
    def search_chunks(self, query: str, filters: Dict[str, Any] = None, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks in ChromaDB using Ollama embeddings
        """
        where_clause = None
        
        if filters and len(filters) > 0:
            # Build where clause with $and operator for multiple filters
            if len(filters) == 1:
                # Single filter - no need for $and
                where_clause = filters
            else:
                # Multiple filters - use $and operator
                conditions = []
                for key, value in filters.items():
                    conditions.append({key: value})
                where_clause = {"$and": conditions}
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results.get('distances') else None,
                        'id': results['ids'][0][i]
                    })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching chunks: {str(e)}")
            return []
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding model and collection"""
        count = self.collection.count()
        
        # Test embedding to get dimension
        test_embedding = self.embedding_client.get_embedding("test")
        
        return {
            'total_chunks': count,
            'embedding_model': 'embeddinggemma',
            'embedding_dimension': len(test_embedding),
            'ollama_url': self.embedding_client.base_url
        }
    
    def create_section_index(self) -> Dict[str, List[str]]:
        """
        Create an index of all sections and their chunk IDs for quick lookup
        """
        # Get all items from collection (this might be slow for large collections)
        all_data = self.collection.get()
        
        section_index = {}
        for i, metadata in enumerate(all_data['metadatas']):
            section_name = metadata['section_name']
            chunk_id = all_data['ids'][i]
            
            if section_name not in section_index:
                section_index[section_name] = []
            section_index[section_name].append(chunk_id)
        
        return section_index